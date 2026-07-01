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
  img         : vec4<f32>,   // intensityScale, showImage, shadeLabels, _
  light       : vec4<f32>,   // ambient, specular, shininess, headlight
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var volTex : texture_3d<f32>;
@group(0) @binding(2) var labTex : texture_3d<u32>;
// Intensity colormap LUT (256x1 RGBA). Maps the scalar volume value -> colour,
// so the 3D volume uses the SAME image colormap the 2D view selected (grayscale
// is the identity ramp, so it round-trips exactly). Sampled with linear interp.
@group(0) @binding(3) var lutTex : texture_2d<f32>;

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
// Colormap the scalar intensity v in [0,1] via the 256-entry LUT (linear
// interp). For the grayscale LUT (entry i = i/255) this returns exactly v.
fn lutColor(v : f32) -> vec3<f32> {
  let f = clamp(v, 0.0, 1.0) * 255.0;
  let i0 = i32(floor(f));
  let i1 = min(i0 + 1, 255);
  let fr = f - f32(i0);
  let c0 = textureLoad(lutTex, vec2<i32>(i0, 0), 0).rgb;
  let c1 = textureLoad(lutTex, vec2<i32>(i1, 0), 0).rgb;
  return mix(c0, c1, fr);
}
fn labelColor(lab : u32) -> vec3<f32> {
  if (lab == 0u) { return vec3<f32>(0.0); }
  // sinebow(fract(lab·φ)) — matches the 2D ncolor palette (volume-mode.js) so the
  // same ncolor group renders identically in 2D slices and the 3D volume.
  let a = 6.28318530718 * fract(f32(lab) * 0.61803398875);
  return vec3<f32>(sin(a) * 0.5 + 0.5,
                   sin(a + 2.09439510239) * 0.5 + 0.5,
                   sin(a + 4.18879020479) * 0.5 + 0.5);
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
  let showImage = u.img.y;
  let shadeLabels = u.img.z;
  let ambient = u.light.x;
  let specular = u.light.y;
  let shininess = max(u.light.z, 1.0);
  let headlight = u.light.w;
  let span = u.boxMax.xyz - u.boxMin.xyz;
  // headlight follows the camera (-rd); otherwise a fixed world light
  let lightDir = select(normalize(vec3<f32>(0.4, 0.7, 0.6)), -rd, headlight > 0.5);

  // ── Image layer: fixed-step volumetric (MIP / mean / additive) ────────────
  // The intensity scalar is colour-mapped through the LUT (grayscale = identity).
  let dt = (tfar - tnear) / f32(nsteps);
  var t = tnear + dt * 0.5;
  var imgMip = 0.0; var imgSum = 0.0; var imgCnt = 0.0; var imgAcc = vec4<f32>(0.0);
  for (var i = 0; i < nsteps; i = i + 1) {
    let pwld = ro + rd * t;
    let n = (pwld - u.boxMin.xyz) / span;            // [0,1] in box
    var vc = vec3<i32>(floor(n * vec3<f32>(u.dims.xyz)));
    vc = clamp(vc, vec3<i32>(0), dims - vec3<i32>(1));
    let s = textureLoad(volTex, vc, 0).r * iscale;
    imgMip = max(imgMip, s);
    imgSum = imgSum + s; imgCnt = imgCnt + 1.0;
    let a = clamp(s * density, 0.0, 1.0);
    let om = 1.0 - imgAcc.w;
    imgAcc = vec4<f32>(imgAcc.rgb + lutColor(s) * a * om, imgAcc.w + a * om);
    t = t + dt;
  }
  // each layer as premultiplied (colour, alpha). lutColor already encodes the
  // brightness, so the premultiplied colour IS lutColor(value) (grayscale ->
  // vec3(value), matching the old white-times-alpha behaviour exactly).
  var imgPC = vec3<f32>(0.0); var imgA = 0.0;
  if (showImage > 0.5) {
    if (mode == 1) { imgA = clamp(imgMip, 0.0, 1.0); imgPC = lutColor(imgMip); }
    else if (mode == 2) { let m = imgSum / max(imgCnt, 1.0); imgA = clamp(m, 0.0, 1.0); imgPC = lutColor(m); }
    else { imgPC = imgAcc.rgb; imgA = imgAcc.w; }
  }

  // ── Label layer: Amanatides-Woo DDA first-hit ─────────────────────────────
  // Visit EXACTLY the voxels the ray crosses (no fixed-step oversampling) and
  // render the nearest opaque label as a crisp voxel cube, flat-shaded on the
  // entered face. This is the hostpkg mask-render optimisation — no doubled /
  // fuzzy surfaces from re-sampling the same voxel at multiple ray steps.
  var labPC = vec3<f32>(0.0); var labA = 0.0;
  if (showLabels > 0.5) {
    let res = vec3<f32>(u.dims.xyz);
    let dv0 = rd / span * res;                        // ray dir in voxel space
    // Guard zero components (axis-aligned rays) so that axis simply never steps.
    let dv = select(dv0, vec3<f32>(1e-8), abs(dv0) < vec3<f32>(1e-8));
    let p0 = (ro + rd * tnear - u.boxMin.xyz) / span * res;   // entry in voxel coords
    var vox = clamp(floor(p0), vec3<f32>(0.0), res - vec3<f32>(1.0));
    let stp = sign(dv);
    let tDelta = abs(1.0 / dv);
    var tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
    var face = -rd;        // axis-aligned normal of the cube face the ray entered by
    var found = 0u;
    let maxIter = dims.x + dims.y + dims.z + 3;
    for (var g = 0; g < maxIter; g = g + 1) {
      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));
      let lab = textureLoad(labTex, ci, 0).r;
      if (lab > 0u) { found = lab; break; }
      if (tMax.x < tMax.y && tMax.x < tMax.z) {
        vox.x = vox.x + stp.x; tMax.x = tMax.x + tDelta.x; face = vec3<f32>(-stp.x, 0.0, 0.0);
        if (vox.x < 0.0 || vox.x >= res.x) { break; }
      } else if (tMax.y < tMax.z) {
        vox.y = vox.y + stp.y; tMax.y = tMax.y + tDelta.y; face = vec3<f32>(0.0, -stp.y, 0.0);
        if (vox.y < 0.0 || vox.y >= res.y) { break; }
      } else {
        vox.z = vox.z + stp.z; tMax.z = tMax.z + tDelta.z; face = vec3<f32>(0.0, 0.0, -stp.z);
        if (vox.z < 0.0 || vox.z >= res.z) { break; }
      }
    }
    if (found > 0u) {
      var lc = labelColor(found);
      if (shadeLabels > 0.5) {
        // Flat-shade the axis-aligned cube FACE the ray entered -> HARD voxel
        // cubes with clearly visible faces (no smoothing/interpolation).
        let diff = max(dot(face, lightDir), 0.0);
        lc = lc * (ambient + (1.0 - ambient) * diff);
        if (specular > 0.0) {
          let h = normalize(lightDir - rd);
          lc = lc + vec3<f32>(specular * pow(max(dot(face, h), 0.0), shininess));
        }
      }
      labA = clamp(labelOpacity, 0.0, 1.0);
      labPC = lc * labA;
    }
  }

  // label OVER image (premultiplied)
  return vec4<f32>(labPC + imgPC * (1.0 - labA), labA + imgA * (1.0 - labA));
}
