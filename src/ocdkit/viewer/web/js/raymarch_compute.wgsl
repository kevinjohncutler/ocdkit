// Compute-shader port of raymarch.wgsl (A/B via setRenderMode("compute")).
//
// Identical volume logic (Amanatides-Woo nearest DDA, MIP/mean/additive, labels),
// but dispatched as a compute grid (one thread per pixel) writing an rgba16float
// storage texture, which a trivial blit pass then copies to the canvas. This is
// the modern object-code path (fragment rasterisation of a fullscreen triangle is
// replaced by a direct compute dispatch); it also becomes the basis for future
// compute-only optimisations (empty-space skipping, temporal accumulation).
//
// The shade() body is a verbatim port of raymarch.wgsl's fs() so output matches.

struct U {
  invViewProj : mat4x4<f32>,
  camPos      : vec4<f32>,
  boxMin      : vec4<f32>,
  boxMax      : vec4<f32>,
  dims        : vec4<f32>,   // NX, NY, NZ, mode
  params      : vec4<f32>,   // nsteps, density, labelOpacity, showLabels
  img         : vec4<f32>,   // intensityScale, showImage, shadeLabels, gamma
  light       : vec4<f32>,   // ambient, specular, shininess, headlight
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var volTex : texture_3d<f32>;
@group(0) @binding(2) var labTex : texture_3d<u32>;
@group(0) @binding(3) var lutTex : texture_2d<f32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;

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
  let a = 6.28318530718 * fract(f32(lab) * 0.61803398875);
  return vec3<f32>(sin(a) * 0.5 + 0.5,
                   sin(a + 2.09439510239) * 0.5 + 0.5,
                   sin(a + 4.18879020479) * 0.5 + 0.5);
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
  let mode = i32(u.dims.w);
  let density = u.params.y;
  let labelOpacity = u.params.z;
  let showLabels = u.params.w;
  let iscale = u.img.x;
  let showImage = u.img.y;
  let shadeLabels = u.img.z;
  let gamma = u.img.w;
  let ambient = u.light.x;
  let specular = u.light.y;
  let shininess = max(u.light.z, 1.0);
  let headlight = u.light.w;
  let span = u.boxMax.xyz - u.boxMin.xyz;
  let lightDir = select(normalize(vec3<f32>(0.4, 0.7, 0.6)), -rd, headlight > 0.5);

  var imgPC = vec3<f32>(0.0); var imgA = 0.0;
  if (showImage > 0.5) {
    let res = vec3<f32>(u.dims.xyz);
    let dv0 = rd / span * res;
    let dv = select(dv0, vec3<f32>(1e-8), abs(dv0) < vec3<f32>(1e-8));
    let p0 = (ro + rd * tnear - u.boxMin.xyz) / span * res;
    var vox = clamp(floor(p0), vec3<f32>(0.0), res - vec3<f32>(1.0));
    let stp = sign(dv);
    let tDelta = abs(1.0 / dv);
    var tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
    var tPrev = 0.0;
    var imgMip = 0.0; var imgSum = 0.0; var imgCnt = 0.0; var imgAcc = vec4<f32>(0.0);
    let maxIter = dims.x + dims.y + dims.z + 3;
    for (var g = 0; g < maxIter; g = g + 1) {
      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));
      let s = textureLoad(volTex, ci, 0).r * iscale;
      let tExit = min(tMax.x, min(tMax.y, tMax.z));
      if (mode == 0) {
        let sg = pow(max(s, 0.0), gamma);
        let segLen = max(tExit - tPrev, 0.0);
        let a = clamp(sg * density * segLen, 0.0, 1.0);
        let om = 1.0 - imgAcc.w;
        imgAcc = vec4<f32>(imgAcc.rgb + lutColor(sg) * a * om, imgAcc.w + a * om);
        if (imgAcc.w >= 0.995) { break; }
      } else {
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
    if (mode == 1) { let v = pow(clamp(imgMip, 0.0, 1.0), gamma); imgA = v; imgPC = lutColor(v); }
    else if (mode == 2) { let m = pow(clamp(imgSum / max(imgCnt, 1.0), 0.0, 1.0), gamma); imgA = m; imgPC = lutColor(m); }
    else { imgPC = imgAcc.rgb; imgA = imgAcc.w; }
  }

  var labPC = vec3<f32>(0.0); var labA = 0.0;
  if (showLabels > 0.5) {
    let res = vec3<f32>(u.dims.xyz);
    let dv0 = rd / span * res;
    let dv = select(dv0, vec3<f32>(1e-8), abs(dv0) < vec3<f32>(1e-8));
    let p0 = (ro + rd * tnear - u.boxMin.xyz) / span * res;
    var vox = clamp(floor(p0), vec3<f32>(0.0), res - vec3<f32>(1.0));
    let stp = sign(dv);
    let tDelta = abs(1.0 / dv);
    var tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
    var face = -rd;
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

  return vec4<f32>(labPC + imgPC * (1.0 - labA), labA + imgA * (1.0 - labA));
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dim = textureDimensions(outTex);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(dim.x), f32(dim.y));
  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), shade(uv));
}
