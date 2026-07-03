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
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
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
      this._onCam = typeof opts.onCameraChange === "function" ? opts.onCameraChange : null;
      this.nsteps = Math.min(512, Math.max(this.NX, this.NY, this.NZ) * 2);
      // Camera = quaternion arcball (free rotation, no three.js); see _initCamera.
      this.uniform = device_buf(this.device, 44 * 4);
      // Intensity colormap LUT (256x1 RGBA) — same image colormap as the 2D view.
      this.lutTex = this.device.createTexture({
        size: [256, 1, 1], format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.colormap = opts.colormap || "gray";
      this._uploadLut(this.colormap);
    }

    /** Upload the 256-entry image colormap LUT (grayscale = identity ramp). */
    _uploadLut(name) {
      let data = null;
      try {
        if (typeof window !== "undefined" && window.ViewerColormap &&
            window.ViewerColormap.generateImageCmapLut) {
          data = window.ViewerColormap.generateImageCmapLut(name);
        }
      } catch (e) { data = null; }
      if (!data || data.length < 256 * 4) {           // fallback: grayscale ramp
        data = new Uint8Array(256 * 4);
        for (let i = 0; i < 256; i = i + 1) { data[i * 4] = i; data[i * 4 + 1] = i; data[i * 4 + 2] = i; data[i * 4 + 3] = 255; }
      }
      this.device.queue.writeTexture({ texture: this.lutTex }, data,
        { bytesPerRow: 256 * 4, rowsPerImage: 1 }, [256, 1, 1]);
    }

    /** Switch the intensity colormap (e.g. when the 2D view's selector changes). */
    setColormap(name) { this.colormap = name; this._uploadLut(name); this.render(); }

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
      this._labBpe = bpe; this._labCtor = Ctor;       // for in-place updateLabels
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
          { binding: 3, resource: this.lutTex.createView() },
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
      // Default 3/4 view with the volume's +Z (for a spacetime stack, the LAST
      // frame) UP: yaw for the 3/4 azimuth, then pitch so Z is vertical and we
      // look slightly down. (Verified: max-z projects above min-z, axis vertical.)
      this.orient = Mat4.quatNormalize(Mat4.quatMul(
        Mat4.quatFromAxisAngle([0, 0, 1], 0.6),
        Mat4.quatFromAxisAngle([1, 0, 0], Math.PI / 2 - 0.5)));
      this.fovy = ((opts.fovy != null ? opts.fovy : 45)) * Math.PI / 180;
      this.radius = Math.hypot(this.NX, this.NY, this.NZ * this.zScale) * 1.5;
      this._home = { orient: this.orient.slice(), radius: this.radius, target: this.target.slice() };
      this._attachInput();
    }

    /** Reset rotation / pan / zoom to the initial home view (H key). */
    resetView() {
      this.orient = this._home.orient.slice();
      this.radius = this._home.radius;
      this.target = this._home.target.slice();
      this.render();
      if (this._onCam) this._onCam();   // persist the reset so a refresh shows the default too
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
      // Near/far that stay WELL-CONDITIONED when zoomed in. The box is centred at
      // the origin, so size the frustum to the eye->origin distance plus a margin.
      // The old `near = max(0.01, radius - 1.2·diag)` collapsed near to 0.01 once
      // the camera came within ~1 diagonal (zoomed in), giving a near:far ratio in
      // the tens of thousands -> f32 unprojection error -> the ray entry point
      // jittered as the camera moved -> the voxel surfaces shimmered/crawled.
      const d = Math.hypot(eye[0], eye[1], eye[2]);          // eye -> box centre
      const near = Math.max(d * 0.05, d - diag * 0.6);
      const far = d + diag * 0.6;
      const proj = Mat4.perspective(this.fovy, this.canvas.width / Math.max(1, this.canvas.height), near, far);
      const viewProj = Mat4.multiply(proj, view);
      return { eye, viewProj, invViewProj: Mat4.invert(viewProj) };
    }

    /** Re-upload the label texture in place from raw ncolor-group bytes (uint8,
     *  Z·Y·X order) after an edit — no destroy/recreate, so no flash. */
    updateLabels(data) {
      if (!this.labTex || !data) return;
      const NX = this.NX, NY = this.NY, NZ = this.NZ, bpe = this._labBpe || 1;
      let buf;
      if (bpe === 1) {
        buf = (data instanceof Uint8Array) ? data : new Uint8Array(data.buffer || data);
      } else {
        const C = this._labCtor || Uint16Array;
        buf = new C(NX * NY * NZ);
        buf.set(data.subarray ? data.subarray(0, buf.length) : data);
      }
      this.device.queue.writeTexture({ texture: this.labTex }, buf.buffer,
        { bytesPerRow: NX * bpe, rowsPerImage: NY }, [NX, NY, NZ]);
      this.render();
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
        // of rotating. Left button only; other buttons still rotate/pan. Holding
        // space is the orbit/pan override, so it always rotates regardless of tool.
        const spaceHeld = !!(window.__viewerSpacePan && window.__viewerSpacePan());
        if (e.button === 0 && !e.shiftKey && !spaceHeld && typeof window.__viewerActiveTool === "function") {
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

      // ── Smoothed input loop (orbit / pan / zoom) ──────────────────────────
      // A single rAF loop drives all camera motion. Raw pointer + wheel deltas
      // accumulate into this-frame inputs; each frame we EMA-smooth a velocity
      // toward that input — a low-pass that rejects the high-frequency trackpad
      // micro-jitter — and once the gesture ends we let the velocity decay
      // (light momentum) so motion eases to rest instead of snapping on a noisy
      // frame. Coalescing alone (the previous approach) removed the discrete
      // chunking but still mapped every vibration straight to the camera.
      let pdx = 0, pdy = 0, ppx = 0, ppy = 0, pz = 0;   // raw input this frame
      let vrx = 0, vry = 0, vpx = 0, vpy = 0, vz = 0;   // smoothed velocities
      let anim = 0;
      const SMOOTH = 0.35;    // EMA weight toward live input (lower = smoother, more lag)
      const DECAY = 0.8;      // momentum decay per frame after the gesture stops
      const EPS = 1e-3;
      const ema = (v, x, active) => active ? v * (1 - SMOOTH) + x * SMOOTH : v * DECAY;
      const step = () => {
        anim = 0;
        const H = c.clientHeight || c.height || 1;
        const ix = pdx, iy = pdy, qx = ppx, qy = ppy, iz = pz;
        pdx = 0; pdy = 0; ppx = 0; ppy = 0; pz = 0;
        vrx = ema(vrx, ix, drag === 1); vry = ema(vry, iy, drag === 1);
        vpx = ema(vpx, qx, drag === 2); vpy = ema(vpy, qy, drag === 2);
        vz = ema(vz, iz, iz !== 0);

        const up = Mat4.quatRotate(self.orient, [0, 1, 0]);
        const right = Mat4.quatRotate(self.orient, [1, 0, 0]);
        let changed = false;
        if (Math.abs(vrx) > EPS || Math.abs(vry) > EPS) {     // arcball rotate (no poles)
          const S = (Math.PI * 1.4) / H;
          const q = Mat4.quatMul(Mat4.quatFromAxisAngle(up, -vrx * S), Mat4.quatFromAxisAngle(right, -vry * S));
          self.orient = Mat4.quatNormalize(Mat4.quatMul(q, self.orient));
          changed = true;
        }
        if (Math.abs(vpx) > EPS || Math.abs(vpy) > EPS) {     // pan target in screen plane
          const td = self.radius * Math.tan(self.fovy / 2);
          const px = (2 * vpx * td) / H, py = (2 * vpy * td) / H;
          self.target = [self.target[0] - right[0] * px + up[0] * py,
                         self.target[1] - right[1] * px + up[1] * py,
                         self.target[2] - right[2] * px + up[2] * py];
          changed = true;
        }
        if (Math.abs(vz) > EPS) {                             // dolly (radius *= exp(k·v))
          const diag = Math.hypot(self.NX, self.NY, self.NZ * self.zScale);
          self.radius = Math.max(diag * 0.2, Math.min(diag * 10, self.radius * Math.exp(vz * 0.0015)));
          changed = true;
        }
        const moving = !!drag || Math.abs(vrx) > EPS || Math.abs(vry) > EPS ||
            Math.abs(vpx) > EPS || Math.abs(vpy) > EPS || Math.abs(vz) > EPS;
        if (changed) { self.render(); if (self._onCam) self._onCam(); }
        if (moving) anim = requestAnimationFrame(step);
      };
      const ensureAnim = () => { if (!anim) anim = requestAnimationFrame(step); };

      c.addEventListener("pointermove", (e) => {
        if (!drag) return;
        const dx = e.clientX - lx, dy = e.clientY - ly;
        lx = e.clientX; ly = e.clientY;
        if (drag === 2) { ppx += dx; ppy += dy; } else { pdx += dx; pdy += dy; }
        ensureAnim();
      });
      c.addEventListener("wheel", (e) => {
        e.preventDefault();
        pz += e.deltaY;
        ensureAnim();
      }, { passive: false });
    }

    /** Serializable camera state (for persistence across refresh). */
    getCamera() {
      return { orient: Array.from(this.orient), radius: this.radius, target: Array.from(this.target) };
    }
    setCamera(c) {
      if (!c) return;
      if (Array.isArray(c.orient) && c.orient.length === 4) this.orient = c.orient.slice();
      if (typeof c.radius === "number") this.radius = c.radius;
      if (Array.isArray(c.target) && c.target.length === 3) this.target = c.target.slice();
      this.render();
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
      // Render at native device pixels. CRUCIAL: cap the backing to the device's
      // max 2D texture size — on a big Retina display clientWidth*dpr can exceed
      // it, and getCurrentTexture() then yields nothing, so the volume renders
      // BLANK until something shrinks the canvas (the old supersample path made
      // this far worse). The cap keeps every frame renderable.
      const maxDim = (this.device.limits && this.device.limits.maxTextureDimension2D) || 8192;
      let w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr) || this.canvas.width);
      let h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr) || this.canvas.height);
      if (w > maxDim || h > maxDim) { const k = maxDim / Math.max(w, h); w = Math.max(1, Math.floor(w * k)); h = Math.max(1, Math.floor(h * k)); }
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
