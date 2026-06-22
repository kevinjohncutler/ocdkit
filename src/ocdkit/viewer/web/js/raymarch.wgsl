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

// Per-label occupancy at a voxel (1 if it belongs to `lab`), for normals.
fn occ(vc : vec3<i32>, lab : u32, dims : vec3<i32>) -> f32 {
  let c = clamp(vc, vec3<i32>(0), dims - vec3<i32>(1));
  if (textureLoad(labTex, c, 0).r == lab) { return 1.0; }
  return 0.0;
}
// Outward surface normal of label `lab` from the central-difference gradient of
// its occupancy (occupancy rises inward, so the normal is -grad). Zero if flat.
fn labelNormal(vc : vec3<i32>, lab : u32, dims : vec3<i32>) -> vec3<f32> {
  let g = vec3<f32>(
    occ(vc + vec3<i32>(1, 0, 0), lab, dims) - occ(vc - vec3<i32>(1, 0, 0), lab, dims),
    occ(vc + vec3<i32>(0, 1, 0), lab, dims) - occ(vc - vec3<i32>(0, 1, 0), lab, dims),
    occ(vc + vec3<i32>(0, 0, 1), lab, dims) - occ(vc - vec3<i32>(0, 0, 1), lab, dims));
  let l = length(g);
  if (l < 1e-4) { return vec3<f32>(0.0); }
  return -g / l;
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
  let span = u.boxMax.xyz - u.boxMin.xyz;
  let lightDir = normalize(vec3<f32>(0.4, 0.7, 0.6));   // fixed world light

  let dt = (tfar - tnear) / f32(nsteps);
  var t = tnear + dt * 0.5;

  // Two independent layers accumulated along the ray; the label layer is
  // composited OVER the image layer at the end (labels always on top).
  var imgMip = 0.0; var imgSum = 0.0; var imgCnt = 0.0; var imgAcc = vec4<f32>(0.0);
  var labHit = 0.0; var labMipW = -1.0; var labMipCol = vec3<f32>(0.0);
  var labSum = vec3<f32>(0.0); var labCnt = 0.0; var labAcc = vec4<f32>(0.0);

  for (var i = 0; i < nsteps; i = i + 1) {
    let pwld = ro + rd * t;
    let n = (pwld - u.boxMin.xyz) / span;            // [0,1] in box
    var vc = vec3<i32>(floor(n * vec3<f32>(u.dims.xyz)));
    vc = clamp(vc, vec3<i32>(0), dims - vec3<i32>(1));
    let s = textureLoad(volTex, vc, 0).r * iscale;
    let lab = textureLoad(labTex, vc, 0).r;

    imgMip = max(imgMip, s);
    imgSum = imgSum + s; imgCnt = imgCnt + 1.0;
    let a = clamp(s * density, 0.0, 1.0);
    let om = 1.0 - imgAcc.w;
    imgAcc = vec4<f32>(imgAcc.rgb + vec3<f32>(s) * a * om, imgAcc.w + a * om);

    if (lab > 0u) {
      var lc = labelColor(lab);
      if (shadeLabels > 0.5) {
        let nrm = labelNormal(vc, lab, dims);
        if (dot(nrm, nrm) > 0.25) {                      // shade only at surfaces
          lc = lc * (0.4 + 0.6 * max(dot(nrm, lightDir), 0.0));   // ambient + diffuse
        }
      }
      labHit = 1.0;
      if (labMipW < 0.0) { labMipW = 1.0; labMipCol = lc; }   // nearest label surface (front-to-back)
      labSum = labSum + lc; labCnt = labCnt + 1.0;
      let la = clamp(labelOpacity * density, 0.0, 1.0);
      let lom = 1.0 - labAcc.w;
      labAcc = vec4<f32>(labAcc.rgb + lc * la * lom, labAcc.w + la * lom);
    }
    t = t + dt;
  }

  // each layer as premultiplied (colour*alpha, alpha)
  var imgPC = vec3<f32>(0.0); var imgA = 0.0;
  if (showImage > 0.5) {
    if (mode == 1) { imgA = clamp(imgMip, 0.0, 1.0); imgPC = vec3<f32>(imgMip); }
    else if (mode == 2) { let m = imgSum / max(imgCnt, 1.0); imgA = clamp(m, 0.0, 1.0); imgPC = vec3<f32>(m); }
    else { imgPC = imgAcc.rgb; imgA = imgAcc.w; }
  }
  var labPC = vec3<f32>(0.0); var labA = 0.0;
  if (showLabels > 0.5 && labHit > 0.5) {
    if (mode == 1) { labA = clamp(labelOpacity, 0.0, 1.0); labPC = labMipCol * labA; }
    else if (mode == 2) { let lm = labSum / max(labCnt, 1.0); labA = clamp(labelOpacity, 0.0, 1.0); labPC = lm * labA; }
    else { labPC = labAcc.rgb; labA = labAcc.w; }
  }
  // label OVER image (premultiplied)
  return vec4<f32>(labPC + imgPC * (1.0 - labA), labA + imgA * (1.0 - labA));
}
