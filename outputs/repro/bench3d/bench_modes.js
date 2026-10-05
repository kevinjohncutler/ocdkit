// Per-mode GPU cost of the CURRENT raymarch_compute.wgsl (EA, MIP, mean, MIDA,
// AMIP) in real Chrome WebGPU, plus AMIP shader variants (checked text patches)
// with their pixel difference against the current shader. GPU time per frame from
// timestamp queries. Results POST to /result/modes.
(async function () {
  "use strict";
  const Q = new URLSearchParams(location.search);
  const DS = (Q.get("ds") || "sample3d,dnaA_xy1,5I,cells3d").split(",");
  const VIEWS = (Q.get("views") || "home,zoom").split(",");
  const K = +(Q.get("k") || 20), ROUNDS = +(Q.get("rounds") || 3);
  const W = +(Q.get("w") || 2048), H = +(Q.get("h") || 1280);
  const log = (...a) => { console.log(...a); document.getElementById("log").textContent += a.join(" ") + "\n"; };
  const post = (kind, obj) => fetch("/result/" + kind, { method: "POST", body: JSON.stringify(obj) });
  window.__done = false; window.__err = null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
    device.addEventListener("uncapturederror", (e) => { window.__err = String(e.error.message); console.error(e.error.message); });
    const SRC = await (await fetch("/js/raymarch_compute.wgsl")).text();
    const U = GPUTextureUsage;
    const F16 = new Float32Array(65536);
    { const u = new Uint16Array(65536); for (let i = 0; i < 65536; i++) u[i] = i; F16.set(Float32Array.from(new Float16Array(u.buffer))); }
    const patch = (code, a, b) => { if (code.split(a).length !== 2) throw new Error("patch anchor not unique/found: " + a.slice(0, 60)); return code.replace(a, b); };

    // AMIP variant: carry the transmittance T instead of the optical depth (one
    // exp per voxel, and the early-exit test needs none), and skip all math on
    // empty voxels with a real branch (select() evaluates both sides).
    const SCAT_OLD = `        let sw = clamp((s - u.win.x) * u.win.y, 0.0, 1.0) * wq;
        let dTau = select(0.0, scatK * pow(sw, scatQ) * max(tExit - tPrev, 0.0) * dvLenB, sw > 0.0);   // (pow(0, 0) = 1: empty never scatters)
        imgAcc.x = max(imgAcc.x, sw * exp(-(imgAcc.y + selfDim * dTau)));
        imgAcc.y = imgAcc.y + dTau;
        if (exp(-imgAcc.y) <= imgAcc.x) { break; }     // nothing further back can beat it`;
    const SCAT_NEW = `        let sw = clamp((s - u.win.x) * u.win.y, 0.0, 1.0) * wq;
        if (sw > 0.0) {
          let a = exp(-scatK * pow(sw, scatQ) * max(tExit - tPrev, 0.0) * dvLenB);   // this voxel's transmittance
          imgAcc.x = max(imgAcc.x, sw * imgAcc.y * select(1.0, sqrt(a), selfDim > 0.0));
          imgAcc.y = imgAcc.y * a;
          if (imgAcc.y <= imgAcc.x) { break; }
        }`;
    // AMIP variant: the exit test compares the optical depth with -log(best),
    // updated only when best rises, instead of an exp per voxel.
    const LIM_NEW = `        let sw = clamp((s - u.win.x) * u.win.y, 0.0, 1.0) * wq;
        let dTau = select(0.0, scatK * pow(sw, scatQ) * max(tExit - tPrev, 0.0) * dvLenB, sw > 0.0);   // (pow(0, 0) = 1: empty never scatters)
        let cand = sw * exp(-(imgAcc.y + selfDim * dTau));
        if (cand > imgAcc.x) { imgAcc.x = cand; imgAcc.z = -log(cand); }
        imgAcc.y = imgAcc.y + dTau;
        if (imgAcc.y >= imgAcc.z) { break; }     // nothing further back can beat it`;
    const lim = (c) => patch(patch(c, SCAT_OLD, LIM_NEW),
      "var imgAcc = vec4<f32>(0.0);", "var imgAcc = select(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 1e30, 0.0), MODE == 4);");
    const fastT = (c) => patch(patch(c, SCAT_OLD, SCAT_NEW),
      "var imgAcc = vec4<f32>(0.0);", "var imgAcc = select(vec4<f32>(0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), MODE == 4);");

    function camera(NX, NY, NZ, view) {
      const M = window.Mat4, diag = Math.hypot(NX, NY, NZ);
      const home = M.quatNormalize(M.quatMul(M.quatFromAxisAngle([0, 0, 1], 0.6), M.quatFromAxisAngle([1, 0, 0], Math.PI / 2 - 0.5)));
      const [orient, radius] = { home: [home, diag * 1.5], zoom: [home, diag * 0.55], top: [M.quatFromAxisAngle([0, 0, 1], 0), diag * 1.2] }[view];
      const eye = M.quatRotate(orient, [0, 0, radius]), up = M.quatRotate(orient, [0, 1, 0]);
      const vw = M.lookAt(eye, [0, 0, 0], up), d = Math.hypot(...eye);
      const vp = M.multiply(M.perspective(45 * Math.PI / 180, W / H, Math.max(d * 0.05, d - diag * 0.6), d + diag * 0.6), vw);
      return { eye, invViewProj: M.invert(vp), box: { min: [-NX / 2, -NY / 2, -NZ / 2], max: [NX / 2, NY / 2, NZ / 2] } };
    }
    // _writeUniform() layout from volume3d-gpu.js (56 floats)
    function uniformData(cam, NX, NY, NZ, m) {
      const u = new Float32Array(60);
      u.set(cam.invViewProj, 0);
      u.set([cam.eye[0], cam.eye[1], cam.eye[2], 1], 16);
      u.set([...cam.box.min, 0], 20); u.set([...cam.box.max, 0], 24);
      u.set([NX, NY, NZ, m.mode], 28);
      u.set([Math.min(512, Math.max(NX, NY, NZ) * 2), m.density, 1.0, 0], 32);
      u.set([1.0, 1.0, 1.0, 1.0], 36);
      u.set([0.4, 0.0, 24.0, 1.0], 40);
      u.set([m.win[0], 1 / (m.win[1] - m.win[0]), m.z, m.w], 44);
      u.set([...cam.box.min, 0], 48); u.set([...cam.box.max, m.selfOff ? 1 : 0], 52);
      return u;
    }
    function tex3d(fmt, NX, NY, NZ, data, bpe) {
      const t = device.createTexture({ size: [NX, NY, NZ], dimension: "3d", format: fmt, usage: U.TEXTURE_BINDING | U.COPY_DST });
      device.queue.writeTexture({ texture: t }, data, { bytesPerRow: NX * bpe, rowsPerImage: NY }, [NX, NY, NZ]);
      return t;
    }
    function bricks(vol, grp, NX, NY, NZ, B) {
      const bx = Math.ceil(NX / B), by = Math.ceil(NY / B), bz = Math.ceil(NZ / B);
      const mx = new Float32Array(bx * by * bz), any = new Uint8Array(bx * by * bz);
      for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
        const i = (((z / B) | 0) * by + ((y / B) | 0)) * bx + ((x / B) | 0), j = (z * NY + y) * NX + x, f = F16[vol[j]];
        if (f > mx[i]) mx[i] = f; if (grp[j]) any[i] = 1; }
      return { img: tex3d("r16float", bx, by, bz, new Uint16Array(new Float16Array(mx).buffer), 2), lab: tex3d("r8uint", bx, by, bz, any, 1) };
    }
    const bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } }] });
    const pipeCache = {};
    async function pipe(code, tag, mode) {
      const key = tag + "|" + mode;
      if (pipeCache[key]) return pipeCache[key];
      const mod = device.createShaderModule({ code });
      const errs = (await mod.getCompilationInfo()).messages.filter((m) => m.type === "error");
      if (errs.length) throw new Error("WGSL " + tag + ": " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
      return pipeCache[key] = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: mod, entryPoint: "cs", constants: { MODE: mode, SHOW_IMG: 1, SHOW_LAB: 0, SHADE_LAB: 1, BRICK: 16, TRANSP: 0, CLASSIFY: 0, CUE: 0 } } });
    }
    async function timeFrames(p, bg, ub, ud, k) {
      const qs = device.createQuerySet({ type: "timestamp", count: 2 * k });
      const rs = device.createBuffer({ size: 16 * k, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const rb = device.createBuffer({ size: 16 * k, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      await device.queue.onSubmittedWorkDone();
      for (let i = 0; i < k; i++) {
        device.queue.writeBuffer(ub, 0, ud);
        const enc = device.createCommandEncoder();
        const cp = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        cp.setPipeline(p); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), 1); cp.end();
        device.queue.submit([enc.finish()]);
      }
      const enc = device.createCommandEncoder();
      enc.resolveQuerySet(qs, 0, 2 * k, rs, 0); enc.copyBufferToBuffer(rs, 0, rb, 0, 16 * k);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(rb.getMappedRange().slice(0)); rb.unmap();
      const g = []; for (let i = 0; i < k; i++) g.push(Number(t[2 * i + 1] - t[2 * i]) / 1e6);
      qs.destroy(); rs.destroy(); rb.destroy();
      return g;
    }
    async function readTex(tex) {
      const buf = device.createBuffer({ size: W * H * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: W * 8 }, [W, H, 1]);
      device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const out = new Uint16Array(buf.getMappedRange().slice(0)); buf.unmap(); buf.destroy();
      return out;
    }
    const maxDiff = (a, b) => { let m = 0; for (let i = 3; i < a.length; i += 4) m = Math.max(m, Math.abs(F16[a[i]] - F16[b[i]])); return m; };
    const med = (a) => { const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; };

    const outTex = device.createTexture({ size: [W, H], format: "rgba16float", usage: U.STORAGE_BINDING | U.COPY_SRC });
    const lutTex = device.createTexture({ size: [256, 1], format: "rgba16float", usage: U.TEXTURE_BINDING | U.COPY_DST });
    { const r = new Float32Array(1024); for (let i = 0; i < 256; i++) r.set([i / 255, i / 255, i / 255, 1], 4 * i);
      device.queue.writeTexture({ texture: lutTex }, new Uint16Array(new Float16Array(r).buffer), { bytesPerRow: 2048 }, [256, 1]); }
    const ub = device.createBuffer({ size: 60 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const q25 = Math.log(0.5) / Math.log(0.75);

    for (const ds of DS) {
      const base = "/data/" + ds + "/";
      const meta = await (await fetch(base + "meta.json")).json(), { NX, NY, NZ } = meta;
      const vol = new Uint16Array(await (await fetch(base + "vol_f16.bin")).arrayBuffer());
      const grp = new Uint8Array(await (await fetch(base + "grp_u8.bin")).arrayBuffer());
      const vt = tex3d("r16float", NX, NY, NZ, vol, 2), gt = tex3d("r8uint", NX, NY, NZ, grp, 1), br = bricks(vol, grp, NX, NY, NZ, 16);
      const bg = device.createBindGroup({ layout: bgl, entries: [
        { binding: 0, resource: { buffer: ub } }, { binding: 1, resource: vt.createView() }, { binding: 2, resource: gt.createView() },
        { binding: 3, resource: lutTex.createView() }, { binding: 4, resource: outTex.createView() },
        { binding: 5, resource: br.img.createView() }, { binding: 6, resource: br.lab.createView() }] });
      // windows: the full range, and the low end at the volume's median (background clipped)
      const WINS = { full: [0, 1], clipped: [meta.intensity_median, Math.max(meta.intensity_p99, meta.intensity_median + 0.05)] };
      const CASES = [
        ["EA d0", 0, { density: 0, z: 1, w: 0 }], ["EA d1", 0, { density: 1, z: 1, w: 0 }], ["MIP", 1, { density: 1, z: 0, w: 0 }],
        ["mean", 2, { density: 1, z: 0, w: 0 }], ["MIDA", 3, { density: 0.5, z: 0, w: 0 }],
        ["AMIP 0.25/25", 4, { density: 1, z: 25, w: q25 }], ["AMIP 0.25/25 flat", 4, { density: 1, z: 25, w: q25, selfOff: 1 }],
        ["AMIP 0.25/1e6", 4, { density: 1, z: 1e6, w: q25 }], ["AMIP 0.25/2", 4, { density: 1, z: 2, w: q25 }],
      ];
      for (const view of VIEWS) for (const wn of Object.keys(WINS)) {
        const cam = camera(NX, NY, NZ, view);
        const row = { ds, view, win: wn, W, H, res: {} };
        for (const [name, mode, m] of CASES) {
          const ud = uniformData(cam, NX, NY, NZ, Object.assign({ mode, win: WINS[wn] }, m));
          const VARS = (Q.get("vars") || "fastT").split(",").filter(Boolean), VF = { fastT, lim };
          const vars = [["cur", SRC]].concat(mode === 4 ? VARS.map((v) => [v, VF[v](SRC)]) : []);
          const ps = {}; for (const [tag, code] of vars) ps[tag] = await pipe(code, tag, mode);
          const fid = {};
          if (mode === 4) {
            await timeFrames(ps.cur, bg, ub, ud, 1); const a = await readTex(outTex);
            for (const [tag] of vars.slice(1)) { await timeFrames(ps[tag], bg, ub, ud, 1); fid[tag] = maxDiff(a, await readTex(outTex)); }
          }
          const g = {}; for (const [tag] of vars) g[tag] = [];
          for (let r = 0; r < ROUNDS; r++) for (const [tag] of vars.slice().sort(() => Math.random() - 0.5)) {
            await timeFrames(ps[tag], bg, ub, ud, 3);
            g[tag].push(...await timeFrames(ps[tag], bg, ub, ud, K));
          }
          row.res[name] = { cur: med(g.cur) };
          for (const [tag] of vars.slice(1)) row.res[name][tag] = { gpu: med(g[tag]), maxdiff: fid[tag] };
        }
        await post("modes", row);
        log(ds, view, wn, Object.entries(row.res).map(([n, r]) => `${n}=${r.cur.toFixed(2)}` +
          Object.entries(r).filter(([k]) => k !== "cur").map(([k, v]) => ` ${k} ${v.gpu.toFixed(2)} (d${v.maxdiff.toExponential(1)})`).join("")).join("  "));
      }
      vt.destroy(); gt.destroy(); br.img.destroy(); br.lab.destroy();
    }
    log("DONE"); window.__done = true;
  } catch (e) { window.__err = String(e && e.stack || e); log("ERROR", window.__err); window.__done = true; }
})();
