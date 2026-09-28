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
// Modes (u.dims.w): 0 = additive (emission-absorption), 1 = MIP, 2 = mean, 3 = MIDA.
// MIDA (maximum intensity difference accumulation; Bruckner and Groller, "Instant
// Volume Visualization using Maximum Intensity Difference Accumulation", 2009):
// front-to-back compositing like EA, but when a sample exceeds the running max
// f along the ray by delta, what is accumulated in front is faded by
// beta = 1 - delta, so new maxima show through as in MIP while depth order and
// translucency are kept. Per voxel opacity is 1 - exp(-density * s * length).
//   I = beta I + (1 - beta A) a s ;  A = beta A + (1 - beta A) a ;  f = max(f, s)
// It composites the INTENSITY and applies the colormap once to I / A at the end,
// as MIP and mean do, so colors stay on the colormap (and HDR and transparency
// work as in MIP).
// Early termination needs A ~ 1 AND f near the top of the window (only a new max
// could lift the fade), so a dim, dense ray keeps marching.
// The fade scales with the ray's path through the voxel (capped at one voxel):
// a full crossing is exactly the published beta, but a ray clipping a voxel's
// corner fades only a little and raises f only as much, the rest following in
// later voxels. (Fading in full on any touch while the voxel's opacity scales
// with the path length drew dark and bright lines along voxel edges.)
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
  win         : vec4<f32>,   // display window lo, 1/(hi-lo) (the 2D histogram bounds); EA exposure; colormap peak
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
// Colormap color (rgb) and alpha (a). Alpha is 1 unless the transparent-low-end
// option is on, in which case it follows the colormap's lightness, so dark values
// fade out instead of drawing as black (and, in EA, stop dimming what's behind).
fn lutRGBA(v : f32) -> vec4<f32> {
  let f = clamp(v, 0.0, 1.0) * 255.0;
  let i0 = i32(floor(f));
  let i1 = min(i0 + 1, 255);
  let fr = f - f32(i0);
  let c0 = textureLoad(lutTex, vec2<i32>(i0, 0), 0);
  let c1 = textureLoad(lutTex, vec2<i32>(i1, 0), 0);
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
  let gamma = u.img.w;                 // intensity gamma: displayed = value^gamma (matches 2D)
  let ambient = u.light.x;
  let specular = u.light.y;
  let shininess = max(u.light.z, 1.0);
  let headlight = u.light.w;
  let span = u.boxMax.xyz - u.boxMin.xyz;
  // headlight follows the camera (-rd); otherwise a fixed world light
  let lightDir = select(normalize(vec3<f32>(0.4, 0.7, 0.6)), -rd, headlight > 0.5);

  // ── Image layer: fixed-step volumetric (MIP / mean / additive) ────────────
  // NEAREST-neighbour DDA — the intensity is NEVER interpolated. An Amanatides-Woo
  // walk visits each voxel the ray crosses exactly once (same as the label layer),
  // so MIP/mean/additive show crisp uniform voxel cubes with no fixed-step
  // "sheet"/wood-grain artefact. Additive weights each voxel's opacity by the ray's
  // path length through it. LUT/colour-map is applied once at the end (MIP/mean).
  var imgPC = vec3<f32>(0.0); var imgA = 0.0;
  if (showImage > 0.5) {
    let res = vec3<f32>(u.dims.xyz);
    let dv0 = rd / span * res;                                // ray dir in voxel space
    let dv = select(dv0, vec3<f32>(1e-8), abs(dv0) < vec3<f32>(1e-8));
    let p0 = (ro + rd * tnear - u.boxMin.xyz) / span * res;   // entry in voxel coords
    var vox = clamp(floor(p0), vec3<f32>(0.0), res - vec3<f32>(1.0));
    let stp = sign(dv);
    let tDelta = abs(1.0 / dv);
    var tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;     // t (world units) to each next face
    var tPrev = 0.0;
    var imgMip = 0.0; var imgSum = 0.0; var imgCnt = 0.0; var imgAcc = vec4<f32>(0.0);
    var midaMax = 0.0;
    let maxIter = dims.x + dims.y + dims.z + 3;
    for (var g = 0; g < maxIter; g = g + 1) {
      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));
      let s = textureLoad(volTex, ci, 0).r * iscale;          // exact voxel value (nearest)
      let tExit = min(tMax.x, min(tMax.y, tMax.z));
      if (mode == 0) {                                        // additive (emission-absorption)
        let sg = pow(clamp((s - u.win.x) * u.win.y, 0.0, 1.0), gamma);                     // gamma per voxel
        let segLen = max(tExit - tPrev, 0.0);                 // path length through this voxel
        let c4 = lutRGBA(sg);
        // Emission-absorption with density = ABSORPTION only: every voxel glows in
        // proportion to its (windowed) intensity whatever the density, and density
        // sets how much nearer glow hides farther glow (0 = pure glow, nothing
        // occludes). Exact per segment: emission is self-absorbed by
        // S(tau) = (1 - e^-tau) / tau. The accumulated glow is shaded by the
        // soft exposure curve after the march.
        let ta = c4.a;
        let tau = sg * density * segLen * ta;
        let S = select(1.0 - 0.5 * tau, (1.0 - exp(-tau)) / tau, tau > 1e-4);
        let T = 1.0 - imgAcc.w;
        imgAcc = vec4<f32>(imgAcc.rgb + c4.rgb * (sg * segLen * S * ta * T), imgAcc.w + (1.0 - exp(-tau)) * T);
        if (imgAcc.w >= 0.995) { break; }
      } else if (mode == 3) {                                 // MIDA
        let sg = pow(clamp((s - u.win.x) * u.win.y, 0.0, 1.0), gamma);
        let segLen = max(tExit - tPrev, 0.0);
        // (with the transparent low end, dark voxels also cover less)
        let a = (1.0 - exp(-sg * density * segLen)) * lutRGBA(sg).a;
        let rise = max(sg - midaMax, 0.0) * min(segLen, 1.0);
        let beta = 1.0 - rise;
        let keep = beta * imgAcc.w;
        imgAcc = vec4<f32>(beta * imgAcc.x + (1.0 - keep) * a * sg, 0.0, 0.0, keep + (1.0 - keep) * a);
        midaMax = midaMax + rise;
        if (imgAcc.w >= 0.995 && midaMax >= 0.99) { break; }
      } else {                                                // MIP / mean
        imgMip = max(imgMip, s);
        imgSum = imgSum + s; imgCnt = imgCnt + 1.0;
      }
      tPrev = tExit;
      if (tMax.x < tMax.y && tMax.x < tMax.z) {
        vox.x = vox.x + stp.x; tMax.x = tMax.x + tDelta.x;
        if (vox.x < 0.0 || vox.x >= res.x) { break; }
      } else if (tMax.y < tMax.z) {
        vox.y = vox.y + stp.y; tMax.y = tMax.y + tDelta.y;
        if (vox.y < 0.0 || vox.y >= res.y) { break; }
      } else {
        vox.z = vox.z + stp.z; tMax.z = tMax.z + tDelta.z;
        if (vox.z < 0.0 || vox.z >= res.z) { break; }
      }
    }
    if (mode == 1) { let v = pow(clamp((imgMip - u.win.x) * u.win.y, 0.0, 1.0), gamma); let c4 = lutRGBA(v); imgA = v * c4.a; imgPC = c4.rgb * c4.a; }
    else if (mode == 2) { let m = pow(clamp((imgSum / max(imgCnt, 1.0) - u.win.x) * u.win.y, 0.0, 1.0), gamma); let c4 = lutRGBA(m); imgA = m * c4.a; imgPC = c4.rgb * c4.a; }
    else if (mode == 3) {
      // the colormap is applied ONCE, to the composited intensity (as in MIP and
      // mean), so every pixel is a true colormap color (blending colormapped
      // samples gave hues off the colormap, e.g. teal + yellow = olive) and HDR
      // and transparency come from the LUT exactly as in MIP
      let v = select(0.0, clamp(imgAcc.x / imgAcc.w, 0.0, 1.0), imgAcc.w > 1e-6);
      let c4 = lutRGBA(v); let ta = c4.a;
      imgA = v * ta; imgPC = c4.rgb * ta;
    }
    else {
      // soft exposure: 1 - e^-x rolls the accumulated glow off to the colormap's
      // peak P (1 in SDR, the display headroom x gain in HDR) instead of clipping;
      // win.z = exposure (set from the data by the host), win.w = P
      // Hue-preserving: the curve is applied to the brightest channel and all three
      // are scaled by the same factor. (Per-channel, a summed yellow saturated
      // red and green first while blue kept rising, washing out to white.)
      let P = max(u.win.w, 1e-3);
      let m = max(imgAcc.r, max(imgAcc.g, imgAcc.b));
      let mc = P * (1.0 - exp(-u.win.z * m / P));
      imgPC = select(vec3<f32>(0.0), imgAcc.rgb * (mc / max(m, 1e-12)), m > 1e-9);
      imgA = max(imgAcc.w, clamp(mc / P, 0.0, 1.0));
    }
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
