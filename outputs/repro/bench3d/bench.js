// A/B benchmark of the shipped 3D compute ray-march vs luxar-inspired variants,
// in real Chrome WebGPU. GPU time per frame from timestamp queries; fidelity as
// pixel diffs of each variant's frame against the shipped shader's frame.
// Suites (URL ?suite=): render | pick | update | tx.  Results POST to /result/<suite>.
(async function () {
  "use strict";
  const Q = new URLSearchParams(location.search);
  const SUITE = Q.get("suite") || "render";
  const DS = (Q.get("ds") || "dnaA_xy1,5I,ftsN_xy1,cells3d").split(",");
  const VIEWS = (Q.get("views") || "home,zoom,top").split(",");
  const K = +(Q.get("k") || 15), ROUNDS = +(Q.get("rounds") || 3);
  const W = +(Q.get("w") || 2048), H = +(Q.get("h") || 1280);
  const ONLY = Q.get("variants") ? Q.get("variants").split(",") : null;
  const log = (...a) => { console.log(...a); document.getElementById("log").textContent += a.join(" ") + "\n"; };
  const post = (kind, obj) => fetch("/result/" + kind, { method: "POST", body: JSON.stringify(obj) });
  window.__done = false; window.__err = null;

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const feats = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
    const device = await adapter.requestDevice({ requiredFeatures: feats });
    device.addEventListener("uncapturederror", (e) => { window.__err = String(e.error.message); console.error(e.error.message); });
    const HAS_TS = feats.length > 0;
    const info = adapter.info || {};
    log("adapter", info.vendor, info.architecture, info.description, "timestamps:", HAS_TS);
    const SHIPPED = await (await fetch("/js/raymarch_compute.wgsl")).text();
    const V = window.BenchVariants;
    const U = GPUTextureUsage;

    // f16 bits -> f32 table (for fast pixel diffs)
    const F16 = new Float32Array(65536);
    { const u = new Uint16Array(65536); for (let i = 0; i < 65536; i++) u[i] = i;
      F16.set(Float32Array.from(new Float16Array(u.buffer))); }

    // ── camera: same math as volume3d-gpu.js _camera()/_initCamera() ──
    function camera(NX, NY, NZ, view, w, h) {
      const M = window.Mat4, diag = Math.hypot(NX, NY, NZ);
      const home = M.quatNormalize(M.quatMul(M.quatFromAxisAngle([0, 0, 1], 0.6),
                                             M.quatFromAxisAngle([1, 0, 0], Math.PI / 2 - 0.5)));
      const views = {
        home: [home, diag * 1.5],                                    // the viewer's default view
        zoom: [home, diag * 0.55],                                   // zoomed in: volume fills the frame
        top:  [M.quatFromAxisAngle([0, 0, 1], 0), diag * 1.2],       // straight down the z axis
      };
      const [orient, radius] = views[view];
      const eye = M.quatRotate(orient, [0, 0, radius]);
      const up = M.quatRotate(orient, [0, 1, 0]);
      const vw = M.lookAt(eye, [0, 0, 0], up);
      const d = Math.hypot(eye[0], eye[1], eye[2]);
      const near = Math.max(d * 0.05, d - diag * 0.6), far = d + diag * 0.6;
      const vp = M.multiply(M.perspective(45 * Math.PI / 180, w / h, near, far), vw);
      return { eye, viewProj: vp, invViewProj: M.invert(vp),
               box: { min: [-NX / 2, -NY / 2, -NZ / 2], max: [NX / 2, NY / 2, NZ / 2] } };
    }
    // pickRayWorld() from volume3d-gpu.js, for a pixel center of a w x h target
    function pickRay(cam, px, py, w, h) {
      const M = window.Mat4, ndcX = 2 * px / w - 1, ndcY = 1 - 2 * py / h;
      const un = (z) => { const v = M.transformVec4(cam.invViewProj, [ndcX, ndcY, z, 1]); return [v[0] / v[3], v[1] / v[3], v[2] / v[3]]; };
      const ro = un(0), pf = un(1);
      return { ro, rd: [pf[0] - ro[0], pf[1] - ro[1], pf[2] - ro[2]], boxMin: cam.box.min, boxMax: cam.box.max };
    }
    // _writeUniform() layout from volume3d-gpu.js (44 floats)
    function uniformData(cam, NX, NY, NZ, mode, cfg) {
      const u = new Float32Array(44);
      u.set(cam.invViewProj, 0);
      u.set([cam.eye[0], cam.eye[1], cam.eye[2], 1], 16);
      u.set([...cam.box.min, 0], 20); u.set([...cam.box.max, 0], 24);
      u.set([NX, NY, NZ, mode], 28);
      u.set([Math.min(512, Math.max(NX, NY, NZ) * 2), 1.0, cfg.opacity, cfg.lab], 32);
      u.set([1.0, cfg.img, 1.0, 1.0], 36);
      u.set([0.4, 0.0, 24.0, 1.0], 40);
      return u;
    }
    const labelColorJS = (g) => { if (!g) return [0, 0, 0]; const a = 6.28318530718 * ((g * 0.61803398875) % 1);
      return [Math.sin(a) * 0.5 + 0.5, Math.sin(a + 2.09439510239) * 0.5 + 0.5, Math.sin(a + 4.18879020479) * 0.5 + 0.5]; };

    async function loadDs(ds) {
      const base = "/data/" + ds + "/";
      const meta = await (await fetch(base + "meta.json")).json();
      const [vol, grp, lab, glut] = await Promise.all(["vol_f16.bin", "grp_u8.bin", "lab.bin", "grp_lut.bin"]
        .map((f) => fetch(base + f).then((r) => r.arrayBuffer())));
      return { ds, meta, vol: new Uint16Array(vol), grp: new Uint8Array(grp),
               lab: meta.lab_dtype === "uint8" ? new Uint8Array(lab) : meta.lab_dtype === "uint16" ? new Uint16Array(lab) : new Uint32Array(lab),
               glut: new Uint32Array(glut) };
    }
    function tex3d(fmt, NX, NY, NZ, data, bpe) {
      const t = device.createTexture({ size: [NX, NY, NZ], dimension: "3d", format: fmt, usage: U.TEXTURE_BINDING | U.COPY_DST });
      device.queue.writeTexture({ texture: t }, data, { bytesPerRow: NX * bpe, rowsPerImage: NY }, [NX, NY, NZ]);
      return t;
    }
    // brick grids for the empty-space-skipping variant (timed: a real load-time cost)
    function buildBricks(d, B) {
      const { NX, NY, NZ } = d.meta, t0 = performance.now();
      const bx = Math.ceil(NX / B), by = Math.ceil(NY / B), bz = Math.ceil(NZ / B);
      const mx = new Float32Array(bx * by * bz), any = new Uint8Array(bx * by * bz);
      const v = d.vol, g = d.grp;
      for (let z = 0; z < NZ; z++) { const zb = ((z / B) | 0) * by;
        for (let y = 0; y < NY; y++) { const rb = (zb + ((y / B) | 0)) * bx, row = (z * NY + y) * NX;
          for (let x = 0; x < NX; x++) { const i = rb + ((x / B) | 0), f = F16[v[row + x]];
            if (f > mx[i]) mx[i] = f; if (g[row + x]) any[i] = 1; } } }
      const ms = performance.now() - t0;
      const mx16 = new Uint16Array(new Float16Array(mx).buffer);
      let occ = 0; for (let i = 0; i < any.length; i++) occ += any[i];
      return { img: tex3d("r16float", bx, by, bz, mx16, 2), lab: tex3d("r8uint", bx, by, bz, any, 1), ms,
               labOcc: occ / any.length, dims: [bx, by, bz] };
    }

    const baseEntries = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } },
    ];
    const EXTRA = {
      skip: [{ binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
             { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } }],
      lut: [{ binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }],
      idbuf: [{ binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32uint", viewDimension: "2d" } }],
      pick: [{ binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
    };
    async function mkPipeline(code, kind, entry = "cs", constants) {
      const mod = device.createShaderModule({ code });
      const ci = await mod.getCompilationInfo();
      const errs = ci.messages.filter((m) => m.type === "error");
      if (errs.length) throw new Error("WGSL: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
      const bgl = device.createBindGroupLayout({ entries: baseEntries.concat(EXTRA[kind] || []) });
      const pipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: mod, entryPoint: entry, constants } });
      return { pipeline, bgl };
    }

    async function timeFrames(pipe, bg, ub, udata, w, h, k) {
      const qs = HAS_TS ? device.createQuerySet({ type: "timestamp", count: 2 * k }) : null;
      const rs = HAS_TS ? device.createBuffer({ size: 16 * k, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
      const rb = HAS_TS ? device.createBuffer({ size: 16 * k, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
      await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      for (let i = 0; i < k; i++) {
        device.queue.writeBuffer(ub, 0, udata);               // per-frame uniform write, as shipped
        const enc = device.createCommandEncoder();
        const cp = enc.beginComputePass(HAS_TS ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } } : {});
        cp.setPipeline(pipe); cp.setBindGroup(0, bg);
        cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8), 1); cp.end();
        device.queue.submit([enc.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      const wall = (performance.now() - t0) / k;
      let gpu = [];
      if (HAS_TS) {
        const enc = device.createCommandEncoder();
        enc.resolveQuerySet(qs, 0, 2 * k, rs, 0); enc.copyBufferToBuffer(rs, 0, rb, 0, 16 * k);
        device.queue.submit([enc.finish()]);
        await rb.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(rb.getMappedRange().slice(0));
        rb.unmap();
        for (let i = 0; i < k; i++) gpu.push(Number(t[2 * i + 1] - t[2 * i]) / 1e6);
        qs.destroy(); rs.destroy(); rb.destroy();
      }
      return { gpu, wall };
    }
    async function readTex(tex, w, h, bpp) {
      const buf = device.createBuffer({ size: w * h * bpp, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: w * bpp }, [w, h, 1]);
      device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const out = buf.getMappedRange().slice(0); buf.unmap(); buf.destroy();
      return out;
    }
    function diff(a, b) {   // Uint16 f16 RGBA frames
      let maxAbs = 0, n1 = 0, n8 = 0, sum = 0;
      const px = a.length / 4;
      for (let p = 0; p < px; p++) {
        let m = 0;
        for (let c = 0; c < 4; c++) { const d = Math.abs(F16[a[4 * p + c]] - F16[b[4 * p + c]]); if (d > m) m = d; }
        if (m > maxAbs) maxAbs = m; sum += m; if (m > 1 / 255) n1++; if (m > 8 / 255) n8++;
      }
      return { maxAbs, meanAbs: sum / px, fracGt1of255: n1 / px, fracGt8of255: n8 / px };
    }
    async function savePng(name, u16, w, h) {
      const c = new OffscreenCanvas(w, h), x = c.getContext("2d"), im = x.createImageData(w, h);
      for (let p = 0; p < w * h; p++) {
        for (let ch = 0; ch < 3; ch++) im.data[4 * p + ch] = Math.max(0, Math.min(255, Math.round(F16[u16[4 * p + ch]] * 255)));
        im.data[4 * p + 3] = 255;
      }
      x.putImageData(im, 0, 0);
      await fetch("/png/" + name, { method: "POST", body: await c.convertToBlob({ type: "image/png" }) });
    }
    const pct = (a, q) => { const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1) + 0.5))]; };

    const CFGS = {
      img:      { img: 1, lab: 0, opacity: 1.0 },
      imglab:   { img: 1, lab: 1, opacity: 1.0 },   // viewer default: opaque labels over image
      lab:      { img: 0, lab: 1, opacity: 1.0 },
      imglab50: { img: 1, lab: 1, opacity: 0.5 },
    };
    const MODES = { 0: "EA", 1: "MIP", 2: "mean" };

    // ───────────────────────────── render suite ─────────────────────────────
    async function renderSuite() {
      const defs = [
        ["base", SHIPPED, "base"], ["ea_exp", V.eaExp(SHIPPED), "base"],
        ["lab_first", V.labFirst(SHIPPED), "base"], ["clip", V.clipAtLabel(SHIPPED), "base"],
        ["skip4", V.skip(SHIPPED, 4), "skip", 4], ["skip8", V.skip(SHIPPED, 8), "skip", 8],
        ["skip16", V.skip(SHIPPED, 16), "skip", 16],
        ["lut", V.lut(SHIPPED), "lut"], ["idbuf", V.idbuf(SHIPPED), "idbuf"],
        ["override", V.overrides(SHIPPED), "base"],
        ["combo8", V.combo(SHIPPED, 8), "skip"], ["combo16", V.combo(SHIPPED, 16), "skip"],
        ["skipx8", V.skipx(SHIPPED, 8), "skip"], ["skipx16", V.skipx(SHIPPED, 16), "skip"],
        ["combo8x", V.combo(SHIPPED, 8, true), "skip"], ["combo16x", V.combo(SHIPPED, 16, true), "skip"],
        ["combo16m", V.combo(SHIPPED, 16, false, true), "skip"],
      ].filter((d) => !ONLY || d[0] === "base" || ONLY.includes(d[0]));
      const PER_STATE = { override: ["base", null], combo8: ["skip", 8], combo16: ["skip", 16], combo8x: ["skip", 8], combo16x: ["skip", 16], combo16m: ["skip", 16] };  // override-constant pipelines
      const pipes = {};
      for (const [name, code, kind] of defs) if (!PER_STATE[name]) pipes[name] = await mkPipeline(code, kind);
      const outTex = device.createTexture({ size: [W, H], format: "rgba16float", usage: U.STORAGE_BINDING | U.COPY_SRC });
      const idTex = device.createTexture({ size: [W, H], format: "r32uint", usage: U.STORAGE_BINDING | U.COPY_SRC });
      const lutTex = device.createTexture({ size: [256, 1], format: "rgba16float", usage: U.TEXTURE_BINDING | U.COPY_DST });
      { const r = new Float32Array(256 * 4); for (let i = 0; i < 256; i++) r.set([i / 255, i / 255, i / 255, 1], 4 * i);
        device.queue.writeTexture({ texture: lutTex }, new Uint16Array(new Float16Array(r).buffer), { bytesPerRow: 256 * 8 }, [256, 1]); }
      const ub = device.createBuffer({ size: 44 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

      for (const ds of DS) {
        const d = await loadDs(ds), { NX, NY, NZ } = d.meta;
        const volTex = tex3d("r16float", NX, NY, NZ, d.vol, 2);
        const grpTex = tex3d("r8uint", NX, NY, NZ, d.grp, 1);
        const lb = d.lab.BYTES_PER_ELEMENT;
        const labTex = tex3d(lb === 1 ? "r8uint" : lb === 2 ? "r16uint" : "r32uint", NX, NY, NZ, d.lab, lb);
        const lutArr = new Float32Array((d.meta.max_label + 1) * 4);
        for (let l = 1; l <= d.meta.max_label; l++) lutArr.set([...labelColorJS(d.glut[l]), 1], 4 * l);
        const lutBuf = device.createBuffer({ size: lutArr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(lutBuf, 0, lutArr);
        const bricks = {};
        for (const B of [4, 8, 16]) {
          bricks[B] = buildBricks(d, B);
          post("bricks", { ds, B, build_ms: bricks[B].ms, lab_brick_occupancy: bricks[B].labOcc, dims: bricks[B].dims });
        }
        const common = [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: volTex.createView() },
          null, { binding: 3, resource: lutTex.createView() }, { binding: 4, resource: outTex.createView() }];
        const bgFor = (name, bgl) => {
          const e = common.slice();
          e[2] = { binding: 2, resource: (name === "lut" ? labTex : grpTex).createView() };
          if (name.startsWith("skip")) { const B = +name.replace(/\D/g, "");
            e.push({ binding: 5, resource: bricks[B].img.createView() }, { binding: 6, resource: bricks[B].lab.createView() }); }
          if (name === "lut") e.push({ binding: 5, resource: { buffer: lutBuf } });
          if (name === "idbuf") e.push({ binding: 5, resource: idTex.createView() });
          return device.createBindGroup({ layout: bgl, entries: e });
        };
        const bgs = {}; for (const n in pipes) bgs[n] = bgFor(n, pipes[n].bgl);
        const ovCache = {};

        const MODESEL = (Q.get("modes") || "0,1,2").split(",").map(Number);
        for (const view of VIEWS) for (const cfgName of Object.keys(CFGS)) for (const mode of MODESEL) {
          if (cfgName === "lab" && !MODESEL.includes(1)) continue;
          if (cfgName === "lab" && mode !== 1) continue;          // mode is irrelevant without the image
          const cfg = CFGS[cfgName];
          const cam = camera(NX, NY, NZ, view, W, H);
          const udata = uniformData(cam, NX, NY, NZ, mode, cfg);
          const run = {};
          for (const [name, code] of defs) {
            if (PER_STATE[name]) {
              const [kind, B] = PER_STATE[name], key = `${name}|${mode}|${cfg.img}|${cfg.lab}`;
              if (!ovCache[key]) { const p = await mkPipeline(code, kind, "cs",
                { MODE: mode, SHOW_IMG: cfg.img, SHOW_LAB: cfg.lab, SHADE_LAB: 1 });
                ovCache[key] = { p, bg: bgFor(B ? "skip" + B : "override", p.bgl) }; }
              run[name] = { pipe: ovCache[key].p.pipeline, bg: ovCache[key].bg };
            } else run[name] = { pipe: pipes[name].pipeline, bg: bgs[name] };
          }
          // fidelity: one frame per variant vs the shipped shader
          device.queue.writeBuffer(ub, 0, udata);
          const frames = {};
          for (const name in run) {
            await timeFrames(run[name].pipe, run[name].bg, ub, udata, W, H, 1);
            frames[name] = new Uint16Array(await readTex(outTex, W, H, 8));
          }
          const fid = {};
          for (const name in run) if (name !== "base") fid[name] = diff(frames.base, frames[name]);
          if (ds === "dnaA_xy1" && view === "home") for (const name in run)
            if (["base", "ea_exp", "clip", "skip8"].includes(name)) await savePng(`${ds}_${view}_${cfgName}_${MODES[mode]}_${name}.png`, frames[name], W, H);
          // timing: interleaved rounds, randomized order each round
          const samples = {}; for (const name in run) samples[name] = { gpu: [], wall: [] };
          for (let r = 0; r < ROUNDS; r++) {
            const order = Object.keys(run).sort(() => Math.random() - 0.5);
            for (const name of order) {
              await timeFrames(run[name].pipe, run[name].bg, ub, udata, W, H, 3);   // warm
              const t = await timeFrames(run[name].pipe, run[name].bg, ub, udata, W, H, K);
              samples[name].gpu.push(...t.gpu); samples[name].wall.push(t.wall);
            }
          }
          const row = { ds, view, cfg: cfgName, mode: MODES[mode], W, H, K, ROUNDS, variants: {} };
          for (const name in run) {
            const g = samples[name].gpu;
            row.variants[name] = { gpu_med: pct(g, 0.5), gpu_p10: pct(g, 0.1), gpu_p90: pct(g, 0.9),
                                   wall_med: pct(samples[name].wall, 0.5), fid: fid[name] || null };
          }
          await post(Q.get("tag") || "render", row);
          log(ds, view, cfgName, MODES[mode], Object.entries(row.variants).map(([n, v]) => `${n}=${v.gpu_med.toFixed(2)}`).join(" "));
        }
        [volTex, grpTex, labTex].forEach((t) => t.destroy());
      }
    }

    // ───────────────────────────── pick suite ──────────────────────────────
    // (a) agreement: GPU first-hit label (idbuf variant, label-ID texture = what the
    //     DDA draws) vs the shipped server _march_ray on the same pixel rays;
    // (b) latency: on-demand single-ray GPU pick (dispatch + mapAsync) vs a real
    //     HTTP POST to the server running the shipped march.
    async function pickSuite() {
      const idp = await mkPipeline(V.idbuf(SHIPPED), "idbuf");
      const pkp = await mkPipeline(V.pick(SHIPPED), "pick", "pk");
      const outTex = device.createTexture({ size: [W, H], format: "rgba16float", usage: U.STORAGE_BINDING | U.COPY_SRC });
      const idTex = device.createTexture({ size: [W, H], format: "r32uint", usage: U.STORAGE_BINDING | U.COPY_SRC });
      const lutTex = device.createTexture({ size: [256, 1], format: "rgba16float", usage: U.TEXTURE_BINDING | U.COPY_DST });
      const ub = device.createBuffer({ size: 44 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const io = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      const iorb = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      for (const ds of DS) {
        const d = await loadDs(ds), { NX, NY, NZ } = d.meta;
        const volTex = tex3d("r16float", NX, NY, NZ, d.vol, 2);
        const lb = d.lab.BYTES_PER_ELEMENT;
        const labTex = tex3d(lb === 1 ? "r8uint" : lb === 2 ? "r16uint" : "r32uint", NX, NY, NZ, d.lab, lb);
        const ents = (bgl, extra) => device.createBindGroup({ layout: bgl, entries: [
          { binding: 0, resource: { buffer: ub } }, { binding: 1, resource: volTex.createView() },
          { binding: 2, resource: labTex.createView() }, { binding: 3, resource: lutTex.createView() },
          { binding: 4, resource: outTex.createView() }, extra] });
        const idbg = ents(idp.bgl, { binding: 5, resource: idTex.createView() });
        const pkbg = ents(pkp.bgl, { binding: 5, resource: { buffer: io } });
        for (const view of ["home", "zoom"]) {
          const cam = camera(NX, NY, NZ, view, W, H);
          const udata = uniformData(cam, NX, NY, NZ, 1, CFGS.lab);
          device.queue.writeBuffer(ub, 0, udata);
          await timeFrames(idp.pipeline, idbg, ub, udata, W, H, 1);
          const ids = new Uint32Array(await readTex(idTex, W, H, 4));
          // seeded pixel sample, biased half onto labeled pixels (where picks matter)
          let s = 12345; const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
          const hitPx = []; for (let p = 0; p < ids.length; p += 7) if (ids[p]) hitPx.push(p);
          const px = [];
          for (let i = 0; i < 1000; i++) px.push((rnd() * W * H) | 0);
          for (let i = 0; i < 1000 && hitPx.length; i++) px.push(hitPx[(rnd() * hitPx.length) | 0]);
          const rays = px.map((p) => pickRay(cam, (p % W) + 0.5, ((p / W) | 0) + 0.5, W, H));
          const srv = await (await fetch("/pick_server/" + ds, { method: "POST", body: JSON.stringify(rays) })).json();
          const agree = { both_miss: 0, same: 0, diff_label: 0, gpu_only: 0, server_only: 0 };
          px.forEach((p, i) => { const g = ids[p], sv = srv[i].label;
            if (!g && !sv) agree.both_miss++; else if (g === sv) agree.same++;
            else if (g && sv) agree.diff_label++; else if (g) agree.gpu_only++; else agree.server_only++; });
          const srvMs = srv.map((r) => r.ms);
          // latency: 60 on-demand GPU picks vs 60 real HTTP round trips (sequential)
          const gpuLat = [], httpLat = [];
          for (let i = 0; i < 60; i++) {
            const p = px[1000 + (i % 1000)] ?? px[i];
            const uvb = new Float32Array([((p % W) + 0.5) / W, (((p / W) | 0) + 0.5) / H, 0, 0, 0, 0, 0, 0]);
            let t0 = performance.now();
            device.queue.writeBuffer(io, 0, uvb);
            const enc = device.createCommandEncoder();
            const cp = enc.beginComputePass(); cp.setPipeline(pkp.pipeline); cp.setBindGroup(0, pkbg); cp.dispatchWorkgroups(1); cp.end();
            enc.copyBufferToBuffer(io, 0, iorb, 0, 32); device.queue.submit([enc.finish()]);
            await iorb.mapAsync(GPUMapMode.READ); const got = new Uint32Array(iorb.getMappedRange().slice(0)); iorb.unmap();
            gpuLat.push(performance.now() - t0);
            if (got[2] !== ids[p]) agree.pick_vs_frame_mismatch = (agree.pick_vs_frame_mismatch || 0) + 1;
            t0 = performance.now();
            await (await fetch("/pick_server/" + ds, { method: "POST", body: JSON.stringify([rays[1000 + (i % 1000)] ?? rays[i]]) })).json();
            httpLat.push(performance.now() - t0);
          }
          const row = { ds, view, n: px.length, agree,
            server_march_ms: { med: pct(srvMs, 0.5), p90: pct(srvMs, 0.9), max: Math.max(...srvMs) },
            gpu_pick_ms: { med: pct(gpuLat, 0.5), p90: pct(gpuLat, 0.9) },
            http_pick_ms: { med: pct(httpLat, 0.5), p90: pct(httpLat, 0.9) } };
          await post("pick", row); log(JSON.stringify(row));
        }
        volTex.destroy(); labTex.destroy();
      }
    }

    // ────────────────────────── update suite (item 1) ─────────────────────────
    // Cost to change how cells look: today = re-upload the whole group volume;
    // LUT variant = rewrite one 16-byte entry (or the whole LUT).
    async function updateSuite() {
      for (const ds of DS) {
        const d = await loadDs(ds), { NX, NY, NZ } = d.meta;
        const grpTex = tex3d("r8uint", NX, NY, NZ, d.grp, 1);
        const lutBuf = device.createBuffer({ size: (d.meta.max_label + 1) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        await device.queue.onSubmittedWorkDone();
        const vol = [], one = [], all = [];
        const full = new Float32Array((d.meta.max_label + 1) * 4);
        for (let i = 0; i < 20; i++) {
          let t0 = performance.now();
          device.queue.writeTexture({ texture: grpTex }, d.grp, { bytesPerRow: NX, rowsPerImage: NY }, [NX, NY, NZ]);
          await device.queue.onSubmittedWorkDone(); vol.push(performance.now() - t0);
          t0 = performance.now();
          device.queue.writeBuffer(lutBuf, 16 * (1 + (i % d.meta.max_label)), new Float32Array([1, 0, 0, 0]));
          await device.queue.onSubmittedWorkDone(); one.push(performance.now() - t0);
          t0 = performance.now();
          device.queue.writeBuffer(lutBuf, 0, full);
          await device.queue.onSubmittedWorkDone(); all.push(performance.now() - t0);
        }
        const row = { ds, voxels: NX * NY * NZ, max_label: d.meta.max_label,
          group_volume_reupload_ms: pct(vol, 0.5), lut_one_entry_ms: pct(one, 0.5), lut_full_ms: pct(all, 0.5),
          label_tex_bytes_groups: NX * NY * NZ, label_tex_bytes_ids: NX * NY * NZ * d.lab.BYTES_PER_ELEMENT };
        await post("update", row); log(JSON.stringify(row));
        grpTex.destroy(); lutBuf.destroy();
      }
    }

    // ───────────────────────────── transport suite ─────────────────────────────
    // Today: GET JSON bundle (float64 gzip+b64) -> shipped decodeBundle -> shipped
    // VolumeGPU._uploadTextures (normalize, f16, cube-instance build, writeTexture).
    // Candidate: GET raw float16 (+groups) binary, optionally HTTP gzip -> writeTexture.
    async function txSuite() {
      const VG = window.VolumeGPU;
      for (const ds of DS) {
        const meta = await (await fetch("/data/" + ds + "/meta.json")).json(), { NX, NY, NZ } = meta, N = NX * NY * NZ;
        const res = { current: [], f16: [], f16gz: [] };
        let refF16 = null, mism = null;
        for (let rep = 0; rep < 3; rep++) {
          { // current
            const t0 = performance.now();
            const r = await fetch("/tx/current/" + ds);
            const txt = await r.text(); const t1 = performance.now();
            const j = JSON.parse(txt); const t2 = performance.now();
            const decoded = await window.decodeBundle(j); const t3 = performance.now();
            const fake = { device, NX, NY, NZ, cubePipeline: {}, _buildCubeInstances: VG.prototype._buildCubeInstances };
            let tCube = 0; const bc = fake._buildCubeInstances;
            fake._buildCubeInstances = function (f) { const a = performance.now(); bc.call(this, f); tCube = performance.now() - a; };
            VG.prototype._uploadTextures.call(fake, decoded);
            await device.queue.onSubmittedWorkDone(); const t4 = performance.now();
            res.current.push({ total: t4 - t0, fetch: t1 - t0, json: t2 - t1, decode: t3 - t2, upload: t4 - t3,
                               cube_build: tCube, bytes: txt.length, enc_ms: +r.headers.get("X-Enc-Ms") });
            if (rep === 0) { // what the shipped path actually put on the GPU (for bitwise compare)
              const a = decoded.image.data; let lo = Infinity, hi = -Infinity;
              for (let i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
              const f = new Float32Array(N), sc = 1 / (hi - lo); for (let i = 0; i < N; i++) f[i] = (a[i] - lo) * sc;
              refF16 = new Uint16Array(new Float16Array(f).buffer);
            }
            fake.volTex.destroy(); fake.labTex.destroy(); if (fake.cubeInstBuf) fake.cubeInstBuf.destroy();
          }
          for (const gz of [0, 1]) {
            const t0 = performance.now();
            const r = await fetch(`/tx/f16/${ds}?gz=${gz}`);
            const buf = await r.arrayBuffer(); const t1 = performance.now();
            const vol = new Uint16Array(buf, 0, N), grp = new Uint8Array(buf, 2 * N, N);
            const vt = tex3d("r16float", NX, NY, NZ, vol, 2), gt = tex3d("r8uint", NX, NY, NZ, grp, 1);
            await device.queue.onSubmittedWorkDone(); const t2 = performance.now();
            res[gz ? "f16gz" : "f16"].push({ total: t2 - t0, fetch: t1 - t0, upload: t2 - t1,
              bytes_decoded: buf.byteLength, enc_ms: +r.headers.get("X-Enc-Ms") });
            if (rep === 0 && !gz) { let n = 0, mx = 0; for (let i = 0; i < N; i++) if (vol[i] !== refF16[i]) { n++; mx = Math.max(mx, Math.abs(F16[vol[i]] - F16[refF16[i]])); }
              mism = { n_mismatch: n, frac: n / N, max_abs: mx }; }
            vt.destroy(); gt.destroy();
          }
        }
        const row = { ds, voxels: N, bitwise_vs_shipped: mism };
        for (const k in res) { row[k] = {}; for (const f of Object.keys(res[k][0])) row[k][f] = pct(res[k].map((x) => x[f]), 0.5); }
        await post("tx", row); log(JSON.stringify(row));
      }
    }

    if (SUITE === "render") await renderSuite();
    else if (SUITE === "pick") await pickSuite();
    else if (SUITE === "update") await updateSuite();
    else if (SUITE === "tx") await txSuite();
    log("DONE"); window.__done = true;
  } catch (e) { window.__err = String(e && e.stack || e); log("ERROR", window.__err); window.__done = true; }
})();
