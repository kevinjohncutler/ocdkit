// Compute-shader ray-march (the default 3D render path; raymarch.wgsl is the
// fragment-shader twin, kept as the unoptimized reference).
//
// Amanatides-Woo nearest-voxel DDA, MIP / mean / emission-absorption, labels,
// dispatched one thread per pixel into an rgba16float storage texture that a
// trivial blit pass copies to the canvas. Three optimizations over the fragment
// path, each benchmarked never-slower (outputs/repro/bench3d/SUMMARY.md):
//
//  1. Pipeline-override constants for the render state (MODE, SHOW_IMG,
//     SHOW_LAB, SHADE_LAB): the host builds one pipeline per state, so the
//     branches for other modes compile out of the ray loop.
//  2. Labels are marched FIRST; when an opaque label is hit, the image march is
//     skipped (its contribution is multiplied by 1 - labA = 0).
//  3. Empty-space skipping over a coarse BRICK^3 grid: label-free bricks are
//     jumped in every mode; image bricks only in MIP, when the brick's max can't
//     raise the running max. (EA/mean never skip image bricks: real backgrounds
//     are not zero and mean counts every voxel.)
//
// A brick jump recomputes tMax from the ray origin, while the plain DDA
// accumulates tMax += tDelta, so a ray grazing a voxel edge can enter the other
// neighbor (~0.001% of pixels, silhouette edges only).
//
// Emission-absorption alpha is 1 - exp(-tau), which composites exactly however
// voxel boundaries chop the ray (the old clamp(tau) drifted with view angle).
//
// MIDA (maximum intensity difference accumulation; Bruckner and Groller, "Instant
// Volume Visualization using Maximum Intensity Difference Accumulation", 2009):
// front-to-back compositing like EA, but when a sample exceeds the running max
// f along the ray by delta, what is accumulated in front is faded by
// beta = 1 - delta, so new maxima show through as in MIP while depth order and
// translucency are kept. Per voxel opacity is 1 - exp(-density * s * length).
//   I = beta I + (1 - beta A) a s ;  A = beta A + (1 - beta A) a ;  f = max(f, s)
// It composites the INTENSITY (at the data's full range) and applies the window,
// gamma and colormap once to I / A at the end, as MIP and mean do, so colors
// stay on the colormap and the window acts like a LUT (HDR and transparency
// work as in MIP).
// Early termination needs A ~ 1 AND f near the top of the window (only a new max
// could lift the fade), so a dim, dense ray keeps marching.
// The fade follows the path through the voxel continuously (see midaStep): a
// ray clipping a voxel's corner fades only a little. (Fading in full on any
// touch while the voxel's opacity scales with the path length drew dark and
// bright lines along voxel edges.)

struct U {
  invViewProj : mat4x4<f32>,
  camPos      : vec4<f32>,
  boxMin      : vec4<f32>,
  boxMax      : vec4<f32>,
  dims        : vec4<f32>,   // NX, NY, NZ, mode (mode unused here: see MODE)
  params      : vec4<f32>,   // nsteps, density, labelOpacity, showLabels
  img         : vec4<f32>,   // intensityScale, showImage, shadeLabels, gamma
  light       : vec4<f32>,   // ambient, specular, shininess, headlight
  win         : vec4<f32>,   // display window lo, 1/(hi-lo) (the 2D histogram bounds); EA exposure; voxel shading t
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var volTex : texture_3d<f32>;
@group(0) @binding(2) var labTex : texture_3d<u32>;
@group(0) @binding(3) var lutTex : texture_2d<f32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var brickImg : texture_3d<f32>;   // max normalized intensity per brick
@group(0) @binding(6) var brickLab : texture_3d<u32>;   // 1 if the brick holds any label voxel

override MODE : i32 = 1;             // 0 emission-absorption, 1 MIP, 2 mean, 3 MIDA
override SHOW_IMG : bool = true;
override SHOW_LAB : bool = true;
override SHADE_LAB : bool = true;
override BRICK : f32 = 16.0;         // brick edge in voxels (must match the host's brick grid)
// Transparent low end (colormap alpha). A pipeline constant, not a uniform: with
// it off the alpha multiply compiles away (as a runtime value it cost EA ~3%).
override TRANSP : bool = false;
// Voxel shading (EA and MIDA), t = u.win.w in 0..1: how much a voxel's weight
// depends on the ray's path length L through it. t = 0: exactly L (a cube's
// shading peaks where the ray crosses the most of it, like a distance field).
// t = 1: voxel faces, the same weight F for every voxel the ray crosses, as if
// it emitted from the face the ray entered (F = the average crossing length for
// the ray's direction, 1 / (|dx| + |dy| + |dz|) in voxel units). In between, the
// weight is L^(1-t) F^t: geometric, so for any t < 1 a ray clipping a voxel's
// corner (L -> 0) still contributes almost nothing and no lines appear at edges.
fn voxelWeight(L : f32, F : f32, t : f32) -> f32 {
  if (t <= 0.0) { return L; }
  if (t >= 1.0) { return F; }
  return pow(max(L, 1e-7), 1.0 - t) * pow(F, t);
}

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
// One emission-absorption step over a path of length segLen through value s
// (data at its full range; density = absorption only; exact self-absorption).
// acc.x = glow, acc.w = opacity.
fn eaStep(acc : ptr<function, vec4<f32>>, s : f32, segLen : f32, density : f32) {
  let sv = clamp(s, 0.0, 1.0);
  let tau = sv * density * segLen;
  let S = select(1.0 - 0.5 * tau, (1.0 - exp(-tau)) / tau, tau > 1e-4);
  let T = 1.0 - (*acc).w;
  *acc = vec4<f32>((*acc).x + sv * sv * segLen * S * T, 0.0, 0.0, (*acc).w + (1.0 - exp(-tau)) * T);
}
// One MIDA step (see the header): acc.x = I, acc.w = A, mx = running max f.
// The running max f approaches a brighter value s exponentially with distance,
// f <- s - (s - f) e^(-L / MIDA_CATCHUP), and the fade is (1 - f_new) / (1 - f_old).
// Fading and adding happen together along the path (the continuous limit of
// MIDA's fade-then-add): with opacity rate mu = density s and fade rate
// r = -d ln(1 - f)/dt, dA = -r A + mu (1 - A), dI = -r I + mu (1 - A) s. So
// I - s A just fades by the step's total fade (exact), and A has one smooth
// integral, done by 3-point Simpson. The result does not depend on how a
// stretch of one value is divided into voxels, so boundaries between equal
// voxels stay invisible even when a different value shares the ray. (A rise of
// (s - f) min(L, 1) per voxel depended on the division and drew a grid; a
// per-voxel fade-then-add left seams across 1-voxel-thick lines.) Starting from empty space a full rise fades exactly as the
// paper's 1 - (s - f); a ray clipping a voxel's corner (L -> 0) fades almost
// nothing, so no lines appear along voxel edges.
const MIDA_CATCHUP : f32 = 0.8;   // voxels: 5% creases and no edge lines (0.1 gave lines, the old rule 21% creases)
fn midaStep(acc : ptr<function, vec4<f32>>, mx : ptr<function, f32>, s : f32, segLen : f32, density : f32) {
  let sv = clamp(s, 0.0, 1.0);
  let a = 1.0 - exp(-sv * density * segLen);
  let A0 = (*acc).w;
  let I0 = (*acc).x;
  if (sv <= *mx) {                                    // no rise: plain over-compositing
    *acc = vec4<f32>(I0 + (1.0 - A0) * a * sv, 0.0, 0.0, A0 + (1.0 - A0) * a);
    return;
  }
  let mu = sv * density;
  let q = sv - *mx;
  let omL = 1.0 - (sv - q * exp(-segLen / MIDA_CATCHUP));          // 1 - f at the end
  let om0 = max(1.0 - *mx, 1e-6);
  let omM = max(1.0 - (sv - q * exp(-0.5 * segLen / MIDA_CATCHUP)), 1e-6);
  let beta = omL / om0;                                             // the step's total fade
  // A = beta e^(-mu L) A0 + mu Integral_0^L e^(-mu (L - t)) (1 - f(L)) / (1 - f(t)) dt
  let J = segLen / 6.0 * (exp(-mu * segLen) * omL / om0 + 4.0 * exp(-0.5 * mu * segLen) * omL / omM + 1.0);
  let A1 = beta * exp(-mu * segLen) * A0 + mu * J;
  *acc = vec4<f32>(sv * A1 + (I0 - sv * A0) * beta, 0.0, 0.0, A1);
  *mx = 1.0 - omL;
}
fn labelColor(lab : u32) -> vec3<f32> {
  if (lab == 0u) { return vec3<f32>(0.0); }
  let a = 6.28318530718 * fract(f32(lab) * 0.61803398875);
  return vec3<f32>(sin(a) * 0.5 + 0.5,
                   sin(a + 2.09439510239) * 0.5 + 0.5,
                   sin(a + 4.18879020479) * 0.5 + 0.5);
}

// Where the ray leaves the current brick: the first voxel past it (xyz) and the
// ray parameter of that exit (w). Same p0/dv as the DDA, so tMax stays on the ray.
fn brickExit(p0 : vec3<f32>, dv : vec3<f32>, stp : vec3<f32>, vox : vec3<f32>) -> vec4<f32> {
  let lo = floor(vox / BRICK) * BRICK;
  let bnd = lo + max(stp, vec3<f32>(0.0)) * BRICK;
  let tb3 = (bnd - p0) / dv;
  let tb = min(tb3.x, min(tb3.y, tb3.z));
  var nv = clamp(floor(p0 + dv * tb), lo, lo + vec3<f32>(BRICK - 1.0));
  if (tb3.x <= tb3.y && tb3.x <= tb3.z) { nv.x = select(lo.x - 1.0, lo.x + BRICK, stp.x > 0.0); }
  else if (tb3.y <= tb3.z) { nv.y = select(lo.y - 1.0, lo.y + BRICK, stp.y > 0.0); }
  else { nv.z = select(lo.z - 1.0, lo.z + BRICK, stp.z > 0.0); }
  return vec4<f32>(nv, tb);
}

// uv: (0,0) = top-left, matching raymarch.wgsl's vs mapping.
fn shade(uv : vec2<f32>) -> vec4<f32> {
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
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
  let density = u.params.y;
  let labelOpacity = u.params.z;
  let iscale = u.img.x;
  let gamma = u.img.w;
  let ambient = u.light.x;
  let specular = u.light.y;
  let shininess = max(u.light.z, 1.0);
  let headlight = u.light.w;
  let span = u.boxMax.xyz - u.boxMin.xyz;
  let lightDir = select(normalize(vec3<f32>(0.4, 0.7, 0.6)), -rd, headlight > 0.5);

  // ── labels first: the first labeled voxel along the ray ──
  var labPC = vec3<f32>(0.0); var labA = 0.0;
  if (SHOW_LAB) {
    let res = vec3<f32>(u.dims.xyz);
    let dv0 = rd / span * res;
    let dv = select(dv0, vec3<f32>(1e-8), abs(dv0) < vec3<f32>(1e-8));
    let p0 = (ro + rd * tnear - u.boxMin.xyz) / span * res;
    let stp = sign(dv);
    let tDelta = abs(1.0 / dv);
    let maxIter = dims.x + dims.y + dims.z + 3;
    var vox = clamp(floor(p0), vec3<f32>(0.0), res - vec3<f32>(1.0));
    var tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
    var face = -rd;
    var found = 0u;
    var curB = vec3<f32>(-1.0);
    for (var g = 0; g < maxIter; g = g + 1) {
      let bv = floor(vox / BRICK);
      if (any(bv != curB)) {
        curB = bv;
        if (textureLoad(brickLab, vec3<i32>(bv), 0).r == 0u) {
          let j = brickExit(p0, dv, stp, vox);
          // the DDA sets `face` only on its own steps, so a hit right after a
          // jump takes the face of the axis the jump exited through
          let lo = bv * BRICK;
          if (j.x < lo.x || j.x > lo.x + BRICK - 1.0) { face = vec3<f32>(-stp.x, 0.0, 0.0); }
          else if (j.y < lo.y || j.y > lo.y + BRICK - 1.0) { face = vec3<f32>(0.0, -stp.y, 0.0); }
          else { face = vec3<f32>(0.0, 0.0, -stp.z); }
          vox = j.xyz;
          tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
          if (any(vox < vec3<f32>(0.0)) || any(vox >= res)) { break; }
          continue;
        }
      }
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
      if (SHADE_LAB) {
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

  // ── image: skipped entirely under an opaque label ──
  var imgPC = vec3<f32>(0.0); var imgA = 0.0;
  if (SHOW_IMG && labA < 1.0) {
    let res = vec3<f32>(u.dims.xyz);
    let dv0 = rd / span * res;
    let dv = select(dv0, vec3<f32>(1e-8), abs(dv0) < vec3<f32>(1e-8));
    let p0 = (ro + rd * tnear - u.boxMin.xyz) / span * res;
    let stp = sign(dv);
    let tDelta = abs(1.0 / dv);
    let maxIter = dims.x + dims.y + dims.z + 3;
    var vox = clamp(floor(p0), vec3<f32>(0.0), res - vec3<f32>(1.0));
    var tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
    var tPrev = 0.0;
    var imgMip = 0.0; var imgSum = 0.0; var imgCnt = 0.0; var imgAcc = vec4<f32>(0.0);
    var midaMax = 0.0;
    var curB = vec3<f32>(-1.0);
    // voxel shading: the average world length of one voxel crossing along this ray, and t
    let faceLen = 1.0 / max(abs(dv0.x) + abs(dv0.y) + abs(dv0.z), 1e-6);
    let faceMix = clamp(u.win.w, 0.0, 1.0);
    for (var g = 0; g < maxIter; g = g + 1) {
      // Keep this check FLAT (bv computed every step, one combined condition).
      // Nesting it under its own `if (MODE == 1)` measured up to 35% slower for
      // image-only MIP on Apple GPUs, with identical output.
      let bv = floor(vox / BRICK);
      if (MODE == 1 && any(bv != curB)) {
        curB = bv;
        if (textureLoad(brickImg, vec3<i32>(bv), 0).r * iscale <= imgMip) {   // can't raise the max
          let j = brickExit(p0, dv, stp, vox);
          vox = j.xyz; tPrev = j.w;
          tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
          if (any(vox < vec3<f32>(0.0)) || any(vox >= res)) { break; }
          continue;
        }
      }
      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));
      let s = textureLoad(volTex, ci, 0).r * iscale;
      let tExit = min(tMax.x, min(tMax.y, tMax.z));
      if (MODE == 0) {
        // Emission-absorption on the data at its FULL range (no window, no gamma:
        // those act on the result, like a LUT). Density = ABSORPTION only: every
        // voxel glows with s^2 per unit length whatever the density, and density
        // sets how much nearer glow hides farther glow (0 = pure glow).
        eaStep(&imgAcc, s, voxelWeight(max(tExit - tPrev, 0.0), faceLen, faceMix), density);
        if (imgAcc.w >= 0.995) { break; }
      } else if (MODE == 3) {
        // on the data at its full range, like EA: the window acts on the result
        midaStep(&imgAcc, &midaMax, s, voxelWeight(max(tExit - tPrev, 0.0), faceLen, faceMix), density);
        if (imgAcc.w >= 0.995 && midaMax >= 0.99) { break; }
      } else if (MODE == 1) {
        imgMip = max(imgMip, s);
      } else {
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
    if (MODE == 1) { let v = pow(clamp((imgMip - u.win.x) * u.win.y, 0.0, 1.0), gamma); let c4 = lutRGBA(v); let ta = select(1.0, c4.a, TRANSP); imgA = v * ta; imgPC = c4.rgb * ta; }
    else if (MODE == 2) { let m = pow(clamp((imgSum / max(imgCnt, 1.0) - u.win.x) * u.win.y, 0.0, 1.0), gamma); let c4 = lutRGBA(m); let ta = select(1.0, c4.a, TRANSP); imgA = m * ta; imgPC = c4.rgb * ta; }
    else {
      // EA and MIDA end like MIP and mean: one value per pixel, in data units,
      // then the display window, gamma and colormap applied once, as a LUT. So
      // every pixel is a true colormap color, a full-range window never clips,
      // and narrowing it pops whatever projects above its top to the colormap's
      // top (and HDR peak), exactly as in MIP.
      //   MIDA: I / A, a weighted average of data values (never above their max)
      //   EA:   the glow rolled off below 1 by 1 - e^(-k glow), k from the data
      //         (win.z) so the brightest ray along any axis reads 0.95
      var raw = select(0.0, imgAcc.x / imgAcc.w, imgAcc.w > 1e-6);
      if (MODE == 0) { raw = 1.0 - exp(-u.win.z * imgAcc.x); }
      let v = pow(clamp((raw - u.win.x) * u.win.y, 0.0, 1.0), gamma);
      let c4 = lutRGBA(v); let ta = select(1.0, c4.a, TRANSP);
      imgA = v * ta; imgPC = c4.rgb * ta;
    }
  }

  return vec4<f32>(labPC + imgPC * (1.0 - labA), labA + imgA * (1.0 - labA));
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dim = textureDimensions(outTex);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(dim.x), f32(dim.y));
  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), shade(uv));
}
