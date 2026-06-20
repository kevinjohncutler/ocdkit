/* volume3d-gpu.js — raw-WebGPU 3D volume renderer (no three.js).
 *
 * Ray-marches the volume bundle via the canonical raymarch.wgsl (validated
 * headless by tests/test_raymarch_wgsl.py) using the camera math in mat4.js
 * (validated by tests/js/mat4.test.mjs). Browser-only (WebGPU device + canvas);
 * the verifiable pieces it depends on (shader, camera) are tested elsewhere.
 *
 * Must use a DEDICATED canvas (never the canvas2d one) — getContext locks a
 * canvas to one context type. Returns null when WebGPU is unavailable so the
 * page can fall back to the 2.5D view.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VolumeGPU = api.VolumeGPU;
})(this, function () {
  "use strict";
  const Mat4 = (typeof require !== "undefined") ? require("./mat4.js")
                                                : (typeof window !== "undefined" ? window.Mat4 : globalThis.Mat4);

  function labelUintFormat(maxLabel) {
    if (maxLabel <= 0xff) return ["r8uint", Uint8Array, 1];
    if (maxLabel <= 0xffff) return ["r16uint", Uint16Array, 2];
    return ["r32uint", Uint32Array, 4];
  }

  class VolumeGPU {
    static async create(canvas, decoded, opts = {}) {
      if (typeof navigator === "undefined" || !navigator.gpu) return null;
      let adapter, device;
      try {
        adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) return null;
        device = await adapter.requestDevice();
      } catch (e) { return null; }
      const ctx = canvas.getContext("webgpu");
      if (!ctx) return null;

      const self = new VolumeGPU();
      self.canvas = canvas; self.device = device; self.ctx = ctx;
      self.format = "rgba16float";
      try {
        ctx.configure({ device, format: self.format, colorSpace: "display-p3",
                        alphaMode: "premultiplied", toneMapping: { mode: "extended" } });
      } catch (e) {
        ctx.configure({ device, format: self.format, alphaMode: "premultiplied" });
      }

      const wgsl = await (await fetch(opts.shaderUrl || "js/raymarch.wgsl")).text();
      const mod = device.createShaderModule({ code: wgsl });
      self.bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint", viewDimension: "3d" } },
        ],
      });
      self.pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [self.bgl] }),
        vertex: { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: self.format }] },
        primitive: { topology: "triangle-list" },
      });

      self._initState(decoded, opts);
      self._uploadTextures(decoded);
      self._makeBindGroup();
      self._attachInput();
      self.render();
      return self;
    }

    _initState(decoded, opts) {
      const m = decoded.meta;
      this.NX = m.width; this.NY = m.height; this.NZ = m.depth;
      this.decoded = decoded;
      this.mode = opts.mode != null ? opts.mode : 1;   // MIP
      this.density = opts.density != null ? opts.density : 1.0;
      this.labelOpacity = 0.6;
      this.showLabels = decoded.mask ? 1.0 : 0.0;
      this.zScale = opts.zScale != null ? opts.zScale : 1.0;
      this.nsteps = Math.min(512, Math.max(this.NX, this.NY, this.NZ) * 2);
      this.yaw = 0.7; this.pitch = 0.4;
      this.fovy = Math.PI / 4; this.near = 0.05;
      this.uniform = device_buf(this.device, 40 * 4);
    }

    _uploadTextures(decoded) {
      const { device, NX, NY, NZ } = this;
      // intensity -> r32float normalized [0,1]
      const N = NX * NY * NZ;
      const f = new Float32Array(N);
      if (decoded.image) {
        const a = decoded.image.data;
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
        const sc = hi > lo ? 1 / (hi - lo) : 0;
        for (let i = 0; i < N; i++) f[i] = (a[i] - lo) * sc;
      } else if (decoded.mask) {
        const a = decoded.mask.data;            // no intensity: show label occupancy
        for (let i = 0; i < N; i++) f[i] = a[i] > 0 ? 1 : 0;
      }
      this.volTex = device.createTexture({
        size: [NX, NY, NZ], dimension: "3d", format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture({ texture: this.volTex }, f.buffer,
        { bytesPerRow: NX * 4, rowsPerImage: NY }, [NX, NY, NZ]);

      // labels -> uint, format by max label
      let maxLabel = 0;
      if (decoded.mask) { const a = decoded.mask.data; for (let i = 0; i < a.length; i++) if (a[i] > maxLabel) maxLabel = a[i]; }
      const [fmt, Ctor, bpe] = labelUintFormat(maxLabel);
      const lab = new Ctor(N);
      if (decoded.mask) lab.set(decoded.mask.data.subarray ? decoded.mask.data.subarray(0, N) : decoded.mask.data);
      this.labTex = device.createTexture({
        size: [NX, NY, NZ], dimension: "3d", format: fmt,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture({ texture: this.labTex }, lab.buffer,
        { bytesPerRow: NX * bpe, rowsPerImage: NY }, [NX, NY, NZ]);
    }

    _makeBindGroup() {
      this.bindGroup = this.device.createBindGroup({
        layout: this.bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: this.volTex.createView() },
          { binding: 2, resource: this.labTex.createView() },
        ],
      });
    }

    _box() {
      const sx = this.NX, sy = this.NY, sz = this.NZ * this.zScale;
      return { min: [-sx / 2, -sy / 2, -sz / 2], max: [sx / 2, sy / 2, sz / 2],
               radius: Math.max(sx, sy, sz) * 1.6 };
    }

    _writeUniform() {
      const box = this._box();
      const cam = Mat4.orbitCamera({
        target: [0, 0, 0], up: [0, 1, 0], radius: box.radius,
        yaw: this.yaw, pitch: this.pitch, fovy: this.fovy,
        aspect: this.canvas.width / Math.max(1, this.canvas.height),
        near: this.near, far: box.radius * 4,
      });
      const u = new Float32Array(40);
      u.set(cam.invViewProj, 0);
      u.set([cam.eye[0], cam.eye[1], cam.eye[2], 1], 16);
      u.set([box.min[0], box.min[1], box.min[2], 0], 20);
      u.set([box.max[0], box.max[1], box.max[2], 0], 24);
      u.set([this.NX, this.NY, this.NZ, this.mode], 28);
      u.set([this.nsteps, this.density, this.labelOpacity, this.showLabels], 32);
      u.set([1.0, 0, 0, 0], 36);
      this.device.queue.writeBuffer(this.uniform, 0, u);
    }

    render() {
      const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
      const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr) || this.canvas.width);
      const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr) || this.canvas.height);
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      this._writeUniform();
      const enc = this.device.createCommandEncoder();
      const rp = enc.beginRenderPass({
        colorAttachments: [{
          view: this.ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store",
        }],
      });
      rp.setPipeline(this.pipeline); rp.setBindGroup(0, this.bindGroup); rp.draw(3); rp.end();
      this.device.queue.submit([enc.finish()]);
    }

    setMode(m) { this.mode = m | 0; this.render(); }
    setDensity(d) { this.density = +d; this.render(); }
    setLabelOpacity(o) { this.labelOpacity = +o; this.render(); }
    setShowLabels(on) { this.showLabels = on ? 1 : 0; this.render(); }
    setZScale(z) { this.zScale = +z; this.render(); }

    _attachInput() {
      const c = this.canvas; const self = this; let drag = false, lx = 0, ly = 0;
      c.addEventListener("pointerdown", (e) => { drag = true; lx = e.clientX; ly = e.clientY; c.setPointerCapture(e.pointerId); });
      c.addEventListener("pointerup", (e) => { drag = false; try { c.releasePointerCapture(e.pointerId); } catch (_) {} });
      c.addEventListener("pointermove", (e) => {
        if (!drag) return;
        self.yaw += (e.clientX - lx) * 0.01;
        self.pitch = Math.max(-1.5, Math.min(1.5, self.pitch + (e.clientY - ly) * 0.01));
        lx = e.clientX; ly = e.clientY; self.render();
      });
      c.addEventListener("wheel", (e) => { e.preventDefault(); self.fovy = Math.max(0.1, Math.min(1.5, self.fovy * (e.deltaY > 0 ? 1.05 : 0.95))); self.render(); }, { passive: false });
    }

    destroy() { try { this.ctx.unconfigure(); } catch (_) {} try { this.device.destroy(); } catch (_) {} }
  }

  function device_buf(device, size) {
    return device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  return { VolumeGPU };
});
