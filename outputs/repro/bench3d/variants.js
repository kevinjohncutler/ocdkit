// Shader variants for the A/B benchmark, derived by CHECKED text patches of the
// shipped raymarch_compute.wgsl. Every patch asserts its anchor occurs exactly the
// expected number of times, so a variant can never silently degrade to baseline.
// Each variant: { name, code, extra: [bind group layout entries >= 5], constants? }
(function (root) {
  "use strict";

  function rep(src, a, b, n = 1) {
    const got = src.split(a).length - 1;
    if (got !== n) throw new Error(`patch anchor found ${got}x (want ${n}): ${a.slice(0, 70)}`);
    return src.split(a).join(b);
  }

  // Split the shipped shade() into head / image block / label block / tail.
  const IMG = "  var imgPC = vec3<f32>(0.0); var imgA = 0.0;\n";
  const LAB = "  var labPC = vec3<f32>(0.0); var labA = 0.0;\n";
  const RET = "  return vec4<f32>(labPC + imgPC * (1.0 - labA), labA + imgA * (1.0 - labA));\n";
  function split(src) {
    const i = src.indexOf(IMG), l = src.indexOf(LAB), r = src.indexOf(RET);
    if (i < 0 || l < 0 || r < 0 || !(i < l && l < r)) throw new Error("shade() layout changed");
    return { head: src.slice(0, i), img: src.slice(i + IMG.length, l),
             lab: src.slice(l + LAB.length, r), tail: src.slice(r) };
  }

  // ── item 3: emission-absorption alpha 1-exp(-tau) instead of clamp(tau) ──
  function eaExp(src) {
    return rep(src, "let a = clamp(sg * density * segLen, 0.0, 1.0);",
                    "let a = 1.0 - exp(-sg * density * segLen);");
  }

  // ── item 4a: label march first; skip the image march when an OPAQUE label is
  //    hit (its result is multiplied by 1-labA = 0). Must be pixel-identical. ──
  function labFirst(src) {
    const s = split(src);
    const img = rep(s.img, "  if (showImage > 0.5) {\n", "  if (showImage > 0.5 && labA < 1.0) {\n");
    return s.head + IMG + LAB + s.lab + img + s.tail;
  }

  // ── item 4b: image marched only up to the label hit, composited IN FRONT of the
  //    label (front fluorescence over label surface). A deliberate LOOK change. ──
  function clipAtLabel(src) {
    const s = split(src);
    let lab = rep(s.lab, "    var found = 0u;\n", "    var found = 0u;\n    var tEnt = 0.0;\n");
    lab = rep(lab, "      if (lab > 0u) { found = lab; break; }\n",
              "      if (lab > 0u) { found = lab; tHit = tEnt; break; }\n      tEnt = min(tMax.x, min(tMax.y, tMax.z));\n");
    let img = rep(s.img, "      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));\n",
                  "      if (tPrev >= tHit) { break; }\n      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));\n");
    const tail = rep(s.tail, RET,
      "  return vec4<f32>(imgPC + labPC * (1.0 - imgA), imgA + labA * (1.0 - imgA));\n");
    return s.head + "  var tHit = 1e30;\n" + LAB + lab + IMG + img + tail;
  }

  // ── item 5: empty-space skipping over a coarse brick grid ──
  //   brickImg (r16float): max normalized intensity per BxBxB brick
  //   brickLab (r8uint):   1 if any label voxel in the brick
  //   MIP skips bricks whose max <= running max (exact); EA skips all-zero bricks
  //   (exact); mean never skips (it counts voxels). Labels skip label-free bricks.
  //   Re-entry after a jump reuses the SAME p0/dv, so tMax stays on the same ray.
  // skipx: exact-DDA variant of skip. Instead of landing ON the brick boundary and
  // picking the exit axis ourselves (float ties can then disagree with the DDA's
  // own tie-break), re-enter a hair BEFORE the exit, still inside the empty brick,
  // and let the normal DDA step across the boundary with its own tie rules.
  function skipx(src, B) {
    let c = skip(src, B);
    const oldExit = c.slice(c.indexOf("fn brickExit("), c.indexOf("}\n", c.indexOf("  return vec4<f32>(nv, tb);")) + 2);
    const newExit = `fn brickExit(p0 : vec3<f32>, dv : vec3<f32>, stp : vec3<f32>, vox : vec3<f32>) -> vec4<f32> {
  let lo = floor(vox / BR) * BR;
  let bnd = lo + max(stp, vec3<f32>(0.0)) * BR;
  let tb3 = (bnd - p0) / dv;
  let tb = max(min(tb3.x, min(tb3.y, tb3.z)) - 1e-3, 0.0);
  let nv = clamp(floor(p0 + dv * tb), lo, lo + vec3<f32>(BR - 1.0));
  return vec4<f32>(nv, tb);
}
`;
    c = rep(c, oldExit, newExit);
    // the landing voxel is inside the (empty) brick we just decided to skip: stay in it
    // (no bounds break, no face override); curB already equals its brick, so no re-check
    c = rep(c, `          let lo = bv * BR;
          if (j.x < lo.x || j.x > lo.x + BR - 1.0) { face = vec3<f32>(-stp.x, 0.0, 0.0); }
          else if (j.y < lo.y || j.y > lo.y + BR - 1.0) { face = vec3<f32>(0.0, -stp.y, 0.0); }
          else { face = vec3<f32>(0.0, 0.0, -stp.z); }
`, "");
    c = rep(c, "          if (any(vox < vec3<f32>(0.0)) || any(vox >= res)) { break; }\n          continue;\n", "", 2);
    return c;
  }

  function skip(src, B, mipOnly) {
    let c = rep(src, "@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;\n",
      "@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;\n" +
      "@group(0) @binding(5) var brickImg : texture_3d<f32>;\n" +
      "@group(0) @binding(6) var brickLab : texture_3d<u32>;\n" +
      `const BR : f32 = ${B}.0;\n` +
      `// exit point of the current brick along the ray: new voxel (xyz) + exit t (w)
fn brickExit(p0 : vec3<f32>, dv : vec3<f32>, stp : vec3<f32>, vox : vec3<f32>) -> vec4<f32> {
  let lo = floor(vox / BR) * BR;
  let bnd = lo + max(stp, vec3<f32>(0.0)) * BR;
  let tb3 = (bnd - p0) / dv;
  let tb = min(tb3.x, min(tb3.y, tb3.z));
  var nv = clamp(floor(p0 + dv * tb), lo, lo + vec3<f32>(BR - 1.0));
  if (tb3.x <= tb3.y && tb3.x <= tb3.z) { nv.x = select(lo.x - 1.0, lo.x + BR, stp.x > 0.0); }
  else if (tb3.y <= tb3.z) { nv.y = select(lo.y - 1.0, lo.y + BR, stp.y > 0.0); }
  else { nv.z = select(lo.z - 1.0, lo.z + BR, stp.z > 0.0); }
  return vec4<f32>(nv, tb);
}
`);
    const s = split(c);
    // image loop: check once per brick entry
    let img = rep(s.img, "    let maxIter = dims.x + dims.y + dims.z + 3;\n",
      "    let maxIter = dims.x + dims.y + dims.z + 3;\n    var curB = vec3<f32>(-1.0);\n");
    img = rep(img, "      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));\n",
      `      let bv = floor(vox / BR);
      if (${mipOnly ? "mode == 1" : "mode != 2"} && any(bv != curB)) {
        curB = bv;
        let bmax = textureLoad(brickImg, vec3<i32>(bv), 0).r * iscale;
        if (${mipOnly ? "bmax <= imgMip" : "(mode == 1 && bmax <= imgMip) || (mode == 0 && bmax <= 0.0)"}) {
          let j = brickExit(p0, dv, stp, vox);
          vox = j.xyz; tPrev = j.w;
          tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
          if (any(vox < vec3<f32>(0.0)) || any(vox >= res)) { break; }
          continue;
        }
      }
      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));
`);
    let lab = rep(s.lab, "    let maxIter = dims.x + dims.y + dims.z + 3;\n",
      "    let maxIter = dims.x + dims.y + dims.z + 3;\n    var curB = vec3<f32>(-1.0);\n");
    lab = rep(lab, "      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));\n",
      `      let bv = floor(vox / BR);
      if (any(bv != curB)) {
        curB = bv;
        if (textureLoad(brickLab, vec3<i32>(bv), 0).r == 0u) {
          let j = brickExit(p0, dv, stp, vox);
          // face normal: the DDA sets \`face\` only on its own steps, so a hit right
          // after a jump must take the face of the axis the jump exited through.
          let lo = bv * BR;
          if (j.x < lo.x || j.x > lo.x + BR - 1.0) { face = vec3<f32>(-stp.x, 0.0, 0.0); }
          else if (j.y < lo.y || j.y > lo.y + BR - 1.0) { face = vec3<f32>(0.0, -stp.y, 0.0); }
          else { face = vec3<f32>(0.0, 0.0, -stp.z); }
          vox = j.xyz;
          tMax = (vox + max(stp, vec3<f32>(0.0)) - p0) / dv;
          if (any(vox < vec3<f32>(0.0)) || any(vox >= res)) { break; }
          continue;
        }
      }
      let ci = clamp(vec3<i32>(vox), vec3<i32>(0), dims - vec3<i32>(1));
`);
    return s.head + IMG + img + LAB + lab + s.tail;
  }

  // ── item 1: label texture holds label IDS; per-label color+visibility in a
  //    storage buffer (hide/recolor = tiny buffer write, no volume re-upload) ──
  function lut(src) {
    let c = rep(src, "@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;\n",
      "@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;\n" +
      "@group(0) @binding(5) var<storage, read> labLut : array<vec4<f32>>;\n");
    c = rep(c, "      if (lab > 0u) { found = lab; break; }\n",
               "      if (lab > 0u && labLut[lab].a > 0.0) { found = lab; break; }\n");
    c = rep(c, "      var lc = labelColor(found);\n", "      var lc = labLut[found].rgb;\n");
    return c;
  }

  // ── item 2 (per-frame form): also write the first-hit label to an r32uint
  //    ID target every frame (GPU picking / hover without a server round trip) ──
  function idbuf(src) {
    let c = rep(src, "@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;\n",
      "@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;\n" +
      "@group(0) @binding(5) var idTex : texture_storage_2d<r32uint, write>;\n" +
      "var<private> gFound : u32 = 0u;\n");
    c = rep(c, "    if (found > 0u) {\n      var lc", "    gFound = found;\n    if (found > 0u) {\n      var lc");
    c = rep(c, "  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), shade(uv));\n",
      "  gFound = 0u;\n  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), shade(uv));\n" +
      "  textureStore(idTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<u32>(gFound, 0u, 0u, 0u));\n");
    return c;
  }

  // ── item 2 (on-demand form): ONE thread marches the cursor ray and writes
  //    (label, x, y, z) to a storage buffer; read back with mapAsync ──
  function pick(src) {
    let c = rep(src, "@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;\n",
      "@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;\n" +
      "@group(0) @binding(5) var<storage, read_write> pickIO : array<u32, 8>;\n" +
      "var<private> gFound : u32 = 0u;\nvar<private> gVox : vec3<i32> = vec3<i32>(-1);\n");
    c = rep(c, "      if (lab > 0u) { found = lab; break; }\n",
               "      if (lab > 0u) { found = lab; gVox = ci; break; }\n");
    c = rep(c, "    if (found > 0u) {\n      var lc", "    gFound = found;\n    if (found > 0u) {\n      var lc");
    c += `
@compute @workgroup_size(1, 1, 1)
fn pk() {
  let uv = vec2<f32>(bitcast<f32>(pickIO[0]), bitcast<f32>(pickIO[1]));
  let r = shade(uv);
  pickIO[2] = gFound;
  pickIO[3] = bitcast<u32>(gVox.x); pickIO[4] = bitcast<u32>(gVox.y); pickIO[5] = bitcast<u32>(gVox.z);
}
`;
    return c;
  }

  // ── extra: luxar-style pipeline-override constants for per-state branches ──
  function overrides(src) {
    let c = rep(src, "fn lutColor(", "override MODE : i32 = 1;\noverride SHOW_IMG : f32 = 1.0;\n" +
      "override SHOW_LAB : f32 = 1.0;\noverride SHADE_LAB : f32 = 1.0;\nfn lutColor(");
    c = rep(c, "  let mode = i32(u.dims.w);\n", "  let mode = MODE;\n");
    c = rep(c, "  let showLabels = u.params.w;\n", "  let showLabels = SHOW_LAB;\n");
    c = rep(c, "  let showImage = u.img.y;\n", "  let showImage = SHOW_IMG;\n");
    c = rep(c, "  let shadeLabels = u.img.z;\n", "  let shadeLabels = SHADE_LAB;\n");
    return c;
  }

  // ── combination of the pixel-identical wins: brick skip (B) + label-first +
  //    override constants (so modes that can't skip compile the checks out) ──
  function combo(src, B, exact, mipOnly) {
    return overrides(labFirst(exact ? skipx(src, B) : skip(src, B, mipOnly)));
  }

  root.BenchVariants = { rep, split, eaExp, labFirst, clipAtLabel, skip, skipx, lut, idbuf, pick, overrides, combo };
})(typeof window !== "undefined" ? window : globalThis);
