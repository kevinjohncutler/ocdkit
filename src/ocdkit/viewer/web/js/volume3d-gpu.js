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

      const _v = (typeof window !== "undefined" && window.__AV__) ? ("?v=" + window.__AV__) : "";
      const wgsl = await (await fetch((opts.shaderUrl || "js/raymarch.wgsl") + _v)).text();
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
      if (typeof window !== "undefined" && window.OverlayLayer) {
        try { self.overlays = await window.OverlayLayer.create(device, self.format, decoded, opts); }
        catch (e) { self.overlays = null; }
      }
      self._initCamera(opts);
      self.render();
      return self;
    }

    _initState(decoded, opts) {
      const m = decoded.meta;
      this.NX = m.width; this.NY = m.height; this.NZ = m.depth;
      this.decoded = decoded;
      this.mode = opts.mode != null ? opts.mode : 1;   // MIP
      this.density = opts.density != null ? opts.density : 1.0;
      this.labelOpacity = 1.0;                             // opaque labels by default
      this.showImage = decoded.image ? 1.0 : 0.0;          // grayscale intensity layer
      this.showLabels = decoded.mask ? 1.0 : 0.0;          // coloured labels, composited on top
      this.shadeLabels = 1.0;                              // diffuse-light the label surfaces
      this.ambient = 0.4; this.specular = 0.0; this.shininess = 24.0; this.headlight = 1.0;
      this.zScale = opts.zScale != null ? opts.zScale : 1.0;
      this.nsteps = Math.min(512, Math.max(this.NX, this.NY, this.NZ) * 2);
      // Camera = quaternion arcball (free rotation, no three.js); see _initCamera.
      this.uniform = device_buf(this.device, 44 * 4);
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
      // Centred, right-handed (no axis reflection -> orbit rotation stays correct).
      const sx = this.NX, sy = this.NY, sz = this.NZ * this.zScale;
      return { min: [-sx / 2, -sy / 2, -sz / 2], max: [sx / 2, sy / 2, sz / 2],
               diag: Math.hypot(sx, sy, sz) };
    }

    // Quaternion arcball: free rotation in any orientation — no gimbal lock /
    // pole limit, no three.js. orient = camera orientation; eye = target +
    // orient*(0,0,radius); up = orient*(0,1,0). Left-drag rotate (around the
    // current up/right), right-/shift-drag pan, wheel dolly.
    _initCamera(opts) {
      this.target = [0, 0, 0];
      this.orient = Mat4.quatNormalize(Mat4.quatMul(
        Mat4.quatFromAxisAngle([1, 0, 0], -0.5),
        Mat4.quatFromAxisAngle([0, 1, 0], 0.6)));            // initial 3/4 view
      this.fovy = ((opts.fovy != null ? opts.fovy : 45)) * Math.PI / 180;
      this.radius = Math.hypot(this.NX, this.NY, this.NZ * this.zScale) * 1.5;
      this._home = { orient: this.orient.slice(), radius: this.radius, target: this.target.slice() };
      this._attachInput();
    }

    /** Reset rotation / pan / zoom to the initial home view. */
    resetView() {
      this.orient = this._home.orient.slice();
      this.radius = this._home.radius;
      this.target = this._home.target.slice();
      this.render();
    }

    _eye() {
      const o = Mat4.quatRotate(this.orient, [0, 0, this.radius]);
      return [this.target[0] + o[0], this.target[1] + o[1], this.target[2] + o[2]];
    }

    _camera() {
      const diag = Math.hypot(this.NX, this.NY, this.NZ * this.zScale);
      const eye = this._eye();
      const up = Mat4.quatRotate(this.orient, [0, 1, 0]);
      const view = Mat4.lookAt(eye, this.target, up);
      const proj = Mat4.perspective(this.fovy, this.canvas.width / Math.max(1, this.canvas.height),
        Math.max(0.01, this.radius - diag * 1.2), this.radius + diag * 1.2);
      const viewProj = Mat4.multiply(proj, view);
      return { eye, viewProj, invViewProj: Mat4.invert(viewProj) };
    }

    /** World-space pick ray for a canvas pixel — same math as the render shader
     *  (WebGPU depth near=0/far=1, column-major invViewProj), so a pick matches
     *  exactly what's drawn. Returns {ro, rd, boxMin, boxMax} for the server. */
    pickRayWorld(px, py) {
      const cam = this._camera(), box = this._box();
      const W = this.canvas.clientWidth || this.canvas.width;
      const H = this.canvas.clientHeight || this.canvas.height;
      const ndcX = 2 * px / W - 1, ndcY = 1 - 2 * py / H;
      const un = (z) => {
        const v = Mat4.transformVec4(cam.invViewProj, [ndcX, ndcY, z, 1]);
        return [v[0] / v[3], v[1] / v[3], v[2] / v[3]];
      };
      const ro = un(0.0), pf = un(1.0);
      return { ro, rd: [pf[0] - ro[0], pf[1] - ro[1], pf[2] - ro[2]], boxMin: box.min, boxMax: box.max };
    }

    _attachInput() {
      const c = this.canvas, self = this;
      let drag = 0, lx = 0, ly = 0;   // 0 none, 1 rotate, 2 pan
      c.addEventListener("contextmenu", (e) => e.preventDefault());
      c.addEventListener("pointerdown", (e) => {
        // picker / fill tool: click the cell under the cursor (ray-pick) instead
        // of rotating. Left button only; other buttons still rotate/pan.
        if (e.button === 0 && !e.shiftKey && typeof window.__viewerActiveTool === "function") {
          const t = window.__viewerActiveTool();      // 'picker' | 'fill' | 'erase' act on the cell
          if ((t === "picker" || t === "fill" || t === "erase") && typeof window.__viewerVolume3DPick === "function") {
            const r = c.getBoundingClientRect();
            window.__viewerVolume3DPick(self.pickRayWorld(e.clientX - r.left, e.clientY - r.top), t);
            return;                                  // consume — no drag
          }
        }
        drag = (e.button === 2 || e.button === 1 || e.shiftKey) ? 2 : 1;
        lx = e.clientX; ly = e.clientY; c.setPointerCapture(e.pointerId);
      });
      c.addEventListener("pointerup", (e) => { drag = 0; try { c.releasePointerCapture(e.pointerId); } catch (_) {} });
      // Accumulate movement and apply ONCE per animation frame. Rendering on every
      // raw pointermove makes trackpads (which fire a burst of high-frequency micro
      // events per frame) jitter/oscillate; coalescing to the frame rate is smooth.
      let pdx = 0, pdy = 0, raf = 0;
      const applyDrag = () => {
        raf = 0;
        if (!drag || (pdx === 0 && pdy === 0)) return;
        const dx = pdx, dy = pdy; pdx = 0; pdy = 0;
        const H = c.clientHeight || c.height || 1;
        const up = Mat4.quatRotate(self.orient, [0, 1, 0]);
        const right = Mat4.quatRotate(self.orient, [1, 0, 0]);
        if (drag === 1) {                       // arcball rotate (free, no poles)
          const S = (Math.PI * 1.4) / H;
          const q = Mat4.quatMul(Mat4.quatFromAxisAngle(up, -dx * S), Mat4.quatFromAxisAngle(right, -dy * S));
          self.orient = Mat4.quatNormalize(Mat4.quatMul(q, self.orient));
        } else {                                // pan target in screen plane
          const td = self.radius * Math.tan(self.fovy / 2);
          const px = (2 * dx * td) / H, py = (2 * dy * td) / H;
          self.target = [self.target[0] - right[0] * px + up[0] * py,
                         self.target[1] - right[1] * px + up[1] * py,
                         self.target[2] - right[2] * px + up[2] * py];
        }
        self.render();
      };
      c.addEventListener("pointermove", (e) => {
        if (!drag) return;
        pdx += e.clientX - lx; pdy += e.clientY - ly;
        lx = e.clientX; ly = e.clientY;
        if (!raf) raf = requestAnimationFrame(applyDrag);
      });
      c.addEventListener("wheel", (e) => {
        e.preventDefault();
        const diag = Math.hypot(self.NX, self.NY, self.NZ * self.zScale);
        self.radius = Math.max(diag * 0.2, Math.min(diag * 10, self.radius * (e.deltaY > 0 ? 1 / 0.9 : 0.9)));
        self.render();
      }, { passive: false });
    }

    _writeUniform(cam) {
      const box = this._box();
      const u = new Float32Array(44);
      u.set(cam.invViewProj, 0);
      u.set([cam.eye[0], cam.eye[1], cam.eye[2], 1], 16);
      u.set([box.min[0], box.min[1], box.min[2], 0], 20);
      u.set([box.max[0], box.max[1], box.max[2], 0], 24);
      u.set([this.NX, this.NY, this.NZ, this.mode], 28);
      u.set([this.nsteps, this.density, this.labelOpacity, this.showLabels], 32);
      u.set([1.0, this.showImage, this.shadeLabels, 0], 36);   // iscale, showImage, shadeLabels
      u.set([this.ambient, this.specular, this.shininess, this.headlight], 40);  // light
      this.device.queue.writeBuffer(this.uniform, 0, u);
    }

    render() {
      const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
      const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr) || this.canvas.width);
      const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr) || this.canvas.height);
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      const cam = this._camera();
      this._writeUniform(cam);
      const enc = this.device.createCommandEncoder();
      const rp = enc.beginRenderPass({
        colorAttachments: [{
          view: this.ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store",
        }],
      });
      rp.setPipeline(this.pipeline); rp.setBindGroup(0, this.bindGroup); rp.draw(3);
      if (this.overlays) {
        const box = this._box();
        this.overlays.draw(rp, cam.viewProj, box.min, box.max, [this.NX, this.NY, this.NZ]);
      }
      rp.end();
      this.device.queue.submit([enc.finish()]);
    }

    setMode(m) { this.mode = m | 0; this.render(); }
    setShowImage(on) { this.showImage = on ? 1 : 0; this.render(); }
    setShadeLabels(on) { this.shadeLabels = on ? 1 : 0; this.render(); }
    setAmbient(a) { this.ambient = +a; this.render(); }
    setSpecular(s) { this.specular = +s; this.render(); }
    setShininess(s) { this.shininess = +s; this.render(); }
    setHeadlight(on) { this.headlight = on ? 1 : 0; this.render(); }
    setOverlay(name, on) { if (this.overlays) { this.overlays.setEnabled(name, on); this.render(); } }
    setFlowRaw(flowRaw) { if (this.overlays) { this.overlays.setFlow(flowRaw); this.render(); } }
    setDensity(d) { this.density = +d; this.render(); }
    setLabelOpacity(o) { this.labelOpacity = +o; this.render(); }
    setShowLabels(on) { this.showLabels = on ? 1 : 0; this.render(); }
    setZScale(z) { this.zScale = +z; this.render(); }

    destroy() {
      try { this.ctx.unconfigure(); } catch (_) {}
      try { this.device.destroy(); } catch (_) {}
    }
  }

  function device_buf(device, size) {
    return device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  return { VolumeGPU };
});
