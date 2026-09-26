// Interactive A/B viewer: the shipped raymarch_compute.wgsl vs the benchmarked
// variants (variants.js), on the prepared datasets, with live GPU time per frame.
(async function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const stats = $("stats");
  if (!navigator.gpu) { stats.textContent = "WebGPU unavailable in this browser (use Chrome)."; return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const HAS_TS = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({ requiredFeatures: HAS_TS ? ["timestamp-query"] : [] });
  device.addEventListener("uncapturederror", (e) => { stats.textContent = "GPU error: " + e.error.message; });
  const canvas = $("c"), ctx = canvas.getContext("webgpu");
  const FORMAT = "rgba16float";
  try { ctx.configure({ device, format: FORMAT, colorSpace: "display-p3", alphaMode: "premultiplied", toneMapping: { mode: "extended" } }); }
  catch (e) { ctx.configure({ device, format: FORMAT, alphaMode: "premultiplied" }); }
  const M = window.Mat4, V = window.BenchVariants, U = GPUTextureUsage, B = 16;

  const [SHIPPED, BLIT] = await Promise.all(["/baseline/raymarch_compute.wgsl", "/js/blit.wgsl"].map((u) => fetch(u).then((r) => r.text())));
  const CODE = {
    base: [SHIPPED, false], ea: [V.eaExp(SHIPPED), false], clip: [V.clipAtLabel(SHIPPED), false],
    combo: [V.combo(SHIPPED, B, false, true), true], combo_ea: [V.combo(V.eaExp(SHIPPED), B, false, true), true],
  };
  const NAMES = { base: "Shipped", combo: "Combined", combo_ea: "Combined + EA fix", ea: "Shipped + EA fix", clip: "Clip at label" };

  const C = GPUShaderStage.COMPUTE;
  const baseEntries = [
    { binding: 0, visibility: C, buffer: { type: "uniform" } },
    { binding: 1, visibility: C, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 2, visibility: C, texture: { sampleType: "uint", viewDimension: "3d" } },
    { binding: 3, visibility: C, texture: { sampleType: "float", viewDimension: "2d" } },
    { binding: 4, visibility: C, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } },
  ];
  const brickEntries = [
    { binding: 5, visibility: C, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 6, visibility: C, texture: { sampleType: "uint", viewDimension: "3d" } },
  ];
  const bglBase = device.createBindGroupLayout({ entries: baseEntries });
  const bglBrick = device.createBindGroupLayout({ entries: baseEntries.concat(brickEntries) });
  const modules = {};
  for (const k in CODE) modules[k] = device.createShaderModule({ code: CODE[k][0] });
  const pipeCache = {};
  function pipelineFor(v, st) {
    const brick = CODE[v][1];
    const key = brick ? `${v}|${st.mode}|${st.img}|${st.lab}` : v;
    if (!pipeCache[key]) pipeCache[key] = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [brick ? bglBrick : bglBase] }),
      compute: { module: modules[v], entryPoint: "cs",
                 constants: brick ? { MODE: st.mode, SHOW_IMG: st.img, SHOW_LAB: st.lab, SHADE_LAB: 1 } : undefined } });
    return pipeCache[key];
  }
  const blitBgl = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } }] });
  const bmod = device.createShaderModule({ code: BLIT });
  const blit = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [blitBgl] }),
    vertex: { module: bmod, entryPoint: "vs" }, fragment: { module: bmod, entryPoint: "fs", targets: [{ format: FORMAT }] },
    primitive: { topology: "triangle-list" } });

  const lutTex = device.createTexture({ size: [256, 1], format: "rgba16float", usage: U.TEXTURE_BINDING | U.COPY_DST });
  { const r = new Float32Array(1024); for (let i = 0; i < 256; i++) r.set([i / 255, i / 255, i / 255, 1], 4 * i);
    device.queue.writeTexture({ texture: lutTex }, new Uint16Array(new Float16Array(r).buffer), { bytesPerRow: 2048 }, [256, 1]); }
  const ub = device.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const F16 = new Float32Array(65536);
  { const u = new Uint16Array(65536); for (let i = 0; i < 65536; i++) u[i] = i; F16.set(Float32Array.from(new Float16Array(u.buffer))); }

  // ── state ──
  const st = { variant: "combo", lastAlt: "combo", mode: 1, img: 1, lab: 1, opacity: 1, density: 1, res: 1, spin: false };
  let vol = null;            // { NX, NY, NZ, tex..., bg: {base, brick} }
  let orient, radius, home;

  function tex3d(fmt, n, data, bpe) {
    const t = device.createTexture({ size: n, dimension: "3d", format: fmt, usage: U.TEXTURE_BINDING | U.COPY_DST });
    device.queue.writeTexture({ texture: t }, data, { bytesPerRow: n[0] * bpe, rowsPerImage: n[1] }, n);
    return t;
  }
  async function loadDs(ds) {
    stats.textContent = "loading " + ds + "...";
    const base = "/data/" + ds + "/";
    const meta = await (await fetch(base + "meta.json")).json();
    const [v16, grp] = await Promise.all(["vol_f16.bin", "grp_u8.bin"].map((f) => fetch(base + f).then((r) => r.arrayBuffer())));
    const { NX, NY, NZ } = meta, v = new Uint16Array(v16), g = new Uint8Array(grp);
    const bx = Math.ceil(NX / B), by = Math.ceil(NY / B), bz = Math.ceil(NZ / B);
    const mx = new Float32Array(bx * by * bz), any = new Uint8Array(bx * by * bz);
    for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) {
      const rb = (((z / B) | 0) * by + ((y / B) | 0)) * bx, row = (z * NY + y) * NX;
      for (let x = 0; x < NX; x++) { const i = rb + ((x / B) | 0), f = F16[v[row + x]]; if (f > mx[i]) mx[i] = f; if (g[row + x]) any[i] = 1; }
    }
    if (vol) [vol.vt, vol.gt, vol.bi, vol.bl].forEach((t) => t.destroy());
    const vt = tex3d("r16float", [NX, NY, NZ], v, 2), gt = tex3d("r8uint", [NX, NY, NZ], g, 1);
    const bi = tex3d("r16float", [bx, by, bz], new Uint16Array(new Float16Array(mx).buffer), 2);
    const bl = tex3d("r8uint", [bx, by, bz], any, 1);
    vol = { NX, NY, NZ, vt, gt, bi, bl, out: null };
    const diag = Math.hypot(NX, NY, NZ);
    orient = M.quatNormalize(M.quatMul(M.quatFromAxisAngle([0, 0, 1], 0.6), M.quatFromAxisAngle([1, 0, 0], Math.PI / 2 - 0.5)));
    radius = diag * 1.5; home = { orient: orient.slice(), radius };
    Object.keys(timing).forEach((k) => delete timing[k]);
    request();
  }

  // ── GPU timing: per-variant EMA from timestamp queries ──
  const timing = {};
  const qs = HAS_TS ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const qres = HAS_TS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const qrb = HAS_TS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  let qBusy = false;
  const stateKey = () => `${st.mode}|${st.img}|${st.lab}|${st.opacity}|${st.res}`;

  function ensureOut(w, h) {
    if (vol.out && vol.w === w && vol.h === h) return;
    if (vol.out) vol.out.destroy();
    vol.out = device.createTexture({ size: [w, h], format: "rgba16float", usage: U.STORAGE_BINDING | U.TEXTURE_BINDING });
    vol.w = w; vol.h = h;
    const e = [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: vol.vt.createView() },
      { binding: 2, resource: vol.gt.createView() }, { binding: 3, resource: lutTex.createView() },
      { binding: 4, resource: vol.out.createView() }];
    vol.bgBase = device.createBindGroup({ layout: bglBase, entries: e });
    vol.bgBrick = device.createBindGroup({ layout: bglBrick, entries: e.concat([
      { binding: 5, resource: vol.bi.createView() }, { binding: 6, resource: vol.bl.createView() }]) });
    vol.bgBlit = device.createBindGroup({ layout: blitBgl, entries: [{ binding: 0, resource: vol.out.createView() }] });
  }

  function render() {
    if (!vol) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr * st.res)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr * st.res));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ensureOut(w, h);
    const { NX, NY, NZ } = vol, diag = Math.hypot(NX, NY, NZ);
    const eye = M.quatRotate(orient, [0, 0, radius]), up = M.quatRotate(orient, [0, 1, 0]);
    const d = Math.hypot(...eye), near = Math.max(d * 0.05, d - diag * 0.6), far = d + diag * 0.6;
    const vp = M.multiply(M.perspective(Math.PI / 4, w / h, near, far), M.lookAt(eye, [0, 0, 0], up));
    const u = new Float32Array(48);
    u.set(M.invert(vp), 0); u.set([...eye, 1], 16);
    u.set([-NX / 2, -NY / 2, -NZ / 2, 0], 20); u.set([NX / 2, NY / 2, NZ / 2, 0], 24);
    u.set([NX, NY, NZ, st.mode], 28); u.set([512, st.density, st.opacity, st.lab], 32);
    u.set([1, st.img, 1, 1], 36); u.set([0.4, 0, 24, 1], 40); u.set([0, 1, 0, 0], 44);
    device.queue.writeBuffer(ub, 0, u);
    const enc = device.createCommandEncoder();
    const doTs = HAS_TS && !qBusy;
    const cp = enc.beginComputePass(doTs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    cp.setPipeline(pipelineFor(st.variant, st)); cp.setBindGroup(0, CODE[st.variant][1] ? vol.bgBrick : vol.bgBase);
    cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); cp.end();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(blit); rp.setBindGroup(0, vol.bgBlit); rp.draw(3); rp.end();
    if (doTs) { enc.resolveQuerySet(qs, 0, 2, qres, 0); enc.copyBufferToBuffer(qres, 0, qrb, 0, 16); }
    device.queue.submit([enc.finish()]);
    if (doTs) {
      qBusy = true;
      const v = st.variant, key = stateKey();
      qrb.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigUint64Array(qrb.getMappedRange()); const ms = Number(t[1] - t[0]) / 1e6; qrb.unmap(); qBusy = false;
        if (ms > 0 && ms < 1000) { const k = v + "|" + key, o = timing[k]; timing[k] = o ? o * 0.85 + ms * 0.15 : ms; }
        showStats(w, h);
      });
    }
  }
  function showStats(w, h) {
    const key = stateKey(), cur = timing[st.variant + "|" + key], base = timing["base|" + key];
    let s = `<b>${NAMES[st.variant]}</b>  ${w}x${h}<br>GPU march: ${cur ? cur.toFixed(2) + " ms" : "-"}`;
    if (st.variant !== "base") {
      s += `<br>Shipped (same view state): ${base ? base.toFixed(2) + " ms" : "press V to measure"}`;
      if (cur && base) s += `<br>Ratio: <b>${(cur / base).toFixed(2)}x</b>`;
    }
    if (!HAS_TS) s += "<br>(no timestamp-query on this adapter)";
    stats.innerHTML = s;
  }

  let pending = false;
  function request() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); if (st.spin) { orient = M.quatNormalize(M.quatMul(M.quatFromAxisAngle(M.quatRotate(orient, [0, 1, 0]), 0.01), orient)); request(); } }); } }

  // ── input ──
  let drag = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => { drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", () => { drag = false; });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
    const S = (Math.PI * 1.4) / (canvas.clientHeight || 1);
    const up = M.quatRotate(orient, [0, 1, 0]), right = M.quatRotate(orient, [1, 0, 0]);
    orient = M.quatNormalize(M.quatMul(M.quatMul(M.quatFromAxisAngle(up, -dx * S), M.quatFromAxisAngle(right, -dy * S)), orient));
    request();
  });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); if (!vol) return;
    const diag = Math.hypot(vol.NX, vol.NY, vol.NZ);
    radius = Math.max(diag * 0.2, Math.min(diag * 10, radius * Math.exp(e.deltaY * 0.0015))); request(); }, { passive: false });
  window.addEventListener("resize", request);
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === "v" || e.key === "V") { st.variant = st.variant === "base" ? st.lastAlt : "base"; $("variant").value = st.variant; request(); }
    if ((e.key === "h" || e.key === "H") && home) { orient = home.orient.slice(); radius = home.radius; request(); }
    if ((e.key === "z" || e.key === "Z") && vol) { radius = Math.hypot(vol.NX, vol.NY, vol.NZ) * 0.55; request(); }
  });
  const seg = (id, fn) => $(id).querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); fn(b.dataset.v); request(); }));
  seg("mode", (v) => { st.mode = +v; });
  seg("layers", (v) => { st.img = v === "lab" ? 0 : 1; st.lab = v === "img" ? 0 : 1; });
  seg("spin", (v) => { st.spin = v === "1"; });
  $("variant").addEventListener("change", (e) => { st.variant = e.target.value; if (st.variant !== "base") st.lastAlt = st.variant; request(); });
  $("ds").addEventListener("change", (e) => loadDs(e.target.value));
  $("opacity").addEventListener("input", (e) => { st.opacity = +e.target.value; request(); });
  $("density").addEventListener("input", (e) => { st.density = +e.target.value; request(); });
  $("res").addEventListener("change", (e) => { st.res = +e.target.value; request(); });
  await loadDs("dnaA_xy1");
})();
