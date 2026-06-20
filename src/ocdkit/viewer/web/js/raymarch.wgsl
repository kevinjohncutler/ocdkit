// Canonical volume ray-march shader for the ocdkit 3D viewer.
// Loaded verbatim by the browser host (volume3d-gpu.js) AND by the wgpu-native
// test (tests/test_raymarch_wgsl.py) — so the shipped shader IS the tested one.
//
// Perspective- or ortho-capable: rays are reconstructed from invViewProj, so the
// browser passes an orbit/perspective camera and the test passes an axis-aligned
// ortho camera (making MIP/mean/additive exactly checkable against numpy).
//
// Two 3D textures: intensity (texture_3d<f32>, any float format) and labels
// (texture_3d<u32>, r8/r16/r32 uint). Sampling is nearest via textureLoad (no
// sampler) — matches the discrete voxel grid and the 2.5D canvas2d view.
//
// Modes (u.dims.w): 0 = additive (emission-absorption), 1 = MIP, 2 = mean.
// Label colour matches volume3d-view.js labelColor (golden-ratio HSV, s=.65 v=1).

struct U {
  invViewProj : mat4x4<f32>,
  camPos      : vec4<f32>,
  boxMin      : vec4<f32>,   // world-space AABB
  boxMax      : vec4<f32>,
  dims        : vec4<f32>,   // NX, NY, NZ, mode
  params      : vec4<f32>,   // nsteps, density, labelOpacity, showLabels
  img         : vec4<f32>,   // intensityScale, _, _, _
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var volTex : texture_3d<f32>;
@group(0) @binding(2) var labTex : texture_3d<u32>;

struct VOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut;
  let xy = p[vi];
  o.pos = vec4<f32>(xy, 0.0, 1.0);
  o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);  // (0,0) = top-left
  return o;
}

fn hsv(h : f32, s : f32, v : f32) -> vec3<f32> {
  let i = floor(h * 6.0);
  let f = h * 6.0 - i;
  let p = v * (1.0 - s);
  let q = v * (1.0 - f * s);
  let t = v * (1.0 - (1.0 - f) * s);
  let m = i32(i) % 6;
  if (m == 0) { return vec3<f32>(v, t, p); }
  if (m == 1) { return vec3<f32>(q, v, p); }
  if (m == 2) { return vec3<f32>(p, v, t); }
  if (m == 3) { return vec3<f32>(p, q, v); }
  if (m == 4) { return vec3<f32>(t, p, v); }
  return vec3<f32>(v, p, q);
}
fn labelColor(lab : u32) -> vec3<f32> {
  if (lab == 0u) { return vec3<f32>(0.0); }
  return hsv(fract(f32(lab) * 0.61803398875), 0.65, 1.0);
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0);
  let pn = u.invViewProj * vec4<f32>(ndc, 0.0, 1.0);
  let pf = u.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  let ro = pn.xyz / pn.w;
  let rd = normalize(pf.xyz / pf.w - ro);

  let inv = vec3<f32>(1.0) / rd;
  let t1 = (u.boxMin.xyz - ro) * inv;
  let t2 = (u.boxMax.xyz - ro) * inv;
  let tmn = min(t1, t2);
  let tmx = max(t1, t2);
  var tnear = max(max(tmn.x, tmn.y), tmn.z);
  tnear = max(tnear, 0.0);
  let tfar = min(min(tmx.x, tmx.y), tmx.z);
  if (tnear > tfar) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }

  let dims = vec3<i32>(i32(u.dims.x), i32(u.dims.y), i32(u.dims.z));
  let mode = i32(u.dims.w);
  let nsteps = i32(u.params.x);
  let density = u.params.y;
  let labelOpacity = u.params.z;
  let showLabels = u.params.w;
  let iscale = u.img.x;
  let span = u.boxMax.xyz - u.boxMin.xyz;

  let dt = (tfar - tnear) / f32(nsteps);
  var t = tnear + dt * 0.5;
  var mipVal = 0.0;
  var mipCol = vec3<f32>(0.0);
  var sumCol = vec3<f32>(0.0);
  var sumA = 0.0;
  var cnt = 0.0;
  var acc = vec4<f32>(0.0);

  for (var i = 0; i < nsteps; i = i + 1) {
    let pwld = ro + rd * t;
    let n = (pwld - u.boxMin.xyz) / span;            // [0,1] in box
    var vc = vec3<i32>(floor(n * vec3<f32>(u.dims.xyz)));
    vc = clamp(vc, vec3<i32>(0), dims - vec3<i32>(1));
    let s = textureLoad(volTex, vc, 0).r * iscale;
    let lab = textureLoad(labTex, vc, 0).r;
    var col = vec3<f32>(s);
    if (showLabels > 0.5 && lab > 0u) {
      col = mix(vec3<f32>(s), labelColor(lab), labelOpacity);
    }
    // MIP
    if (s > mipVal) { mipVal = s; mipCol = col; }
    // mean
    sumCol = sumCol + col; sumA = sumA + s; cnt = cnt + 1.0;
    // additive (emission-absorption, premultiplied "over")
    let a = clamp(s * density, 0.0, 1.0);
    let om = 1.0 - acc.w;
    acc = vec4<f32>(acc.rgb + col * a * om, acc.w + a * om);
    t = t + dt;
  }

  if (mode == 1) { return vec4<f32>(mipCol, clamp(mipVal, 0.0, 1.0)); }
  if (mode == 2) {
    let inv_cnt = 1.0 / max(cnt, 1.0);
    return vec4<f32>(sumCol * inv_cnt, sumA * inv_cnt);
  }
  return acc;  // additive, premultiplied
}
