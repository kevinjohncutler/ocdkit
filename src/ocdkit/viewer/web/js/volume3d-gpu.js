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

  // Float32Array -> half-float bits (Uint16) for an r16float texture. Halves the
  // volume's GPU memory bandwidth (the ray-march's dominant cost) with NO change
  // to how it's sampled — still NEAREST (textureLoad), never interpolated. The
  // source data is untouched; this is only the normalized [0,1] display copy.
  const _toF16 = (src) => {
    if (typeof Float16Array !== "undefined") {
      return new Uint16Array(new Float16Array(src).buffer);
    }
    const out = new Uint16Array(src.length);
    const fb = new Float32Array(1), ib = new Int32Array(fb.buffer);
    for (let i = 0; i < src.length; i++) {
      fb[0] = src[i]; const x = ib[0];
      let bits = (x >> 16) & 0x8000; const e = (x >> 23) & 0xff; const m = x & 0x7fffff;
      if (e < 113) { out[i] = bits; }
      else if (e > 142) { out[i] = bits | 0x7c00; }
      else { out[i] = (bits | (((e - 112) << 10) | (m >> 13))) & 0xffff; }
    }
    return out;
  };

  // Extended sRGB OETF (linear-light → gamma-encoded), continued past 1.0 so HDR
  // headroom (values >1) survives. Matches colormap.js `_srgbEncodeExt` and the
  // 2D HDR renderer's `l2g` — the display-p3/extended canvas expects this encoding.
  const _l2gExt = (c) => {
    c = c > 0 ? c : 0;
    return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  };

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
      // Whine experiment: opts.sdrCanvas configures a plain 8-bit sRGB surface
      // (exactly what a WebGL mesh viewer uses) instead of our 16-bit-float
      // display-p3 extended one — to test if the per-frame present of the wide
      // HDR-capable surface is what rings, independent of the render workload.
      // (Loses HDR; fine for the SDR A/B.)
      if (opts.sdrCanvas) {
        self.format = (navigator.gpu.getPreferredCanvasFormat && navigator.gpu.getPreferredCanvasFormat()) || "bgra8unorm";
        ctx.configure({ device, format: self.format, alphaMode: "premultiplied" });
      } else {
        self.format = "rgba16float";
        try {
          ctx.configure({ device, format: self.format, colorSpace: "display-p3",
                          alphaMode: "premultiplied", toneMapping: { mode: "extended" } });
        } catch (e) {
          ctx.configure({ device, format: self.format, alphaMode: "premultiplied" });
        }
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

      // ── Object-order cube renderer (MIP prototype; toggle via setRenderMode) ──
      // Rasterises each occupied voxel as a unit cube with MAX blend = MIP, no ray
      // loop. A/B against the raymarch to test whether the coil whine tracks the
      // pipeline (raster vs compute) rather than the workload.
      try {
        const cwgsl = await (await fetch((opts.cubesUrl || "js/cubes.wgsl") + _v)).text();
        const cmod = device.createShaderModule({ code: cwgsl });
        self.cubeBgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
          ],
        });
        self.cubePipeline = device.createRenderPipeline({
          layout: device.createPipelineLayout({ bindGroupLayouts: [self.cubeBgl] }),
          vertex: {
            module: cmod, entryPoint: "vs",
            buffers: [
              { arrayStride: 12, attributes: [{ shaderLocation: 0, format: "float32x3", offset: 0 }] },
              { arrayStride: 16, stepMode: "instance", attributes: [{ shaderLocation: 1, format: "float32x4", offset: 0 }] },
            ],
          },
          fragment: {
            module: cmod, entryPoint: "fs",
            targets: [{
              format: self.format,
              blend: {   // MAX blend -> order-independent maximum intensity projection
                color: { operation: "max", srcFactor: "one", dstFactor: "one" },
                alpha: { operation: "max", srcFactor: "one", dstFactor: "one" },
              },
            }],
          },
          primitive: { topology: "triangle-list", cullMode: "none" },   // MIP: no cull, no depth
        });
        const corners = new Float32Array([
          -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5,0.5,-0.5,  -0.5,0.5,-0.5,
          -0.5,-0.5, 0.5,  0.5,-0.5, 0.5,  0.5,0.5, 0.5,  -0.5,0.5, 0.5,
        ]);
        const idx = new Uint16Array([
          0,1,2, 0,2,3,  4,6,5, 4,7,6,  0,3,7, 0,7,4,
          1,5,6, 1,6,2,  0,4,5, 0,5,1,  3,2,6, 3,6,7,
        ]);
        self.cubeVertBuf = device.createBuffer({ size: corners.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(self.cubeVertBuf, 0, corners);
        self.cubeIdxBuf = device.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(self.cubeIdxBuf, 0, idx);
        self.cubeUniform = device.createBuffer({ size: 28 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      } catch (e) { self.cubePipeline = null; console.warn("[cubes] pipeline init failed", e); }

      // ── Compute-shader ray-march (A/B via setRenderMode("compute")) ──────────
      // Same march as the fragment path, dispatched as a compute grid writing an
      // rgba16float storage texture, then a trivial blit to the canvas.
      try {
        const [ccode, bcode] = await Promise.all([
          fetch((opts.computeUrl || "js/raymarch_compute.wgsl") + _v).then((r) => r.text()),
          fetch((opts.blitUrl || "js/blit.wgsl") + _v).then((r) => r.text()),
        ]);
        const cmod = device.createShaderModule({ code: ccode });
        self.computeBgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } },
          ],
        });
        self.computePipeline = device.createComputePipeline({
          layout: device.createPipelineLayout({ bindGroupLayouts: [self.computeBgl] }),
          compute: { module: cmod, entryPoint: "cs" },
        });
        const bmod = device.createShaderModule({ code: bcode });
        self.blitBgl = device.createBindGroupLayout({
          entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } }],
        });
        self.blitPipeline = device.createRenderPipeline({
          layout: device.createPipelineLayout({ bindGroupLayouts: [self.blitBgl] }),
          vertex: { module: bmod, entryPoint: "vs" },
          fragment: { module: bmod, entryPoint: "fs", targets: [{ format: self.format }] },
          primitive: { topology: "triangle-list" },
        });
      } catch (e) { self.computePipeline = null; console.warn("[compute] pipeline init failed", e); }

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
      // Render-path experiment: "raymarch" (image-order) | "cubes" (object-order
      // MIP, all occupied voxels) | "minimal" (a few hundred cubes — a trivially
      // light raster load, to test whether the whine is the workload or the
      // per-frame WebGPU present into the extended HDR canvas).
      this._renderMode = opts.renderMode || "raymarch";
      this.density = opts.density != null ? opts.density : 1.0;
      this.labelOpacity = 1.0;                             // opaque labels by default
      this.showImage = decoded.image ? 1.0 : 0.0;          // grayscale intensity layer
      this.showLabels = decoded.mask ? 1.0 : 0.0;          // coloured labels, composited on top
      this.shadeLabels = 1.0;                              // diffuse-light the label surfaces
      this.gamma = opts.gamma != null ? opts.gamma : 1.0;  // intensity gamma (matches the 2D slider)
      // HDR: when on, the intensity LUT is the JzAzBz-lifted Display-P3 colormap
      // (values >1 = HDR headroom), exactly like the 2D HDR image layer. The
      // rgba16float / display-p3 / extended canvas emits those >1 values as true
      // HDR. Off = the plain SDR colormap. Driven by the central OcdHdrUI toggle.
      this._hdr = !!opts.hdr;
      this._gain = opts.gain > 0 ? opts.gain : 1.0;
      // Live display EDR headroom (× SDR white) — the SAME source the 2D HDR
      // layer uses. Critical: without a real headroom the lift targets ~203 nits
      // (headroom 1), and the auto-Jz search can land BELOW SDR white, so "HDR
      // on" renders DIMMER than SDR (the inverted look). Default 4× until the
      // probe resolves; re-lift on change.
      this._headroomVal = opts.headroom > 0 ? opts.headroom : 4.0;
      if (typeof window !== "undefined" && window.HdrHeadroom) {
        try {
          this._hh = new window.HdrHeadroom();
          if (this._hh.value > 0) this._headroomVal = this._hh.value;
          this._hh.onChange((v) => {
            if (v > 0) { this._headroomVal = v; if (this._hdr) { this._uploadLut(this.colormap); this._requestRender(); } }
          });
        } catch (e) { /* no probe; keep the 4× fallback */ }
      }
      this.ambient = 0.4; this.specular = 0.0; this.shininess = 24.0; this.headlight = 1.0;
      this.zScale = opts.zScale != null ? opts.zScale : 1.0;
      // Adaptive resolution while moving: a ray-march costs O(pixels·steps), so
      // zoomed-in orbit (most pixels hit the volume) is the slow case. Render at a
      // dynamic fraction of native pixels chosen to hold the interactive frame
      // time near the display refresh (up to ~120 fps): full-res when there's
      // headroom (zoomed out), scaled down only as needed (zoomed in). A full-res
      // frame is drawn once motion settles, so the still is always sharp.
      this._interacting = false;
      this._dynScale = 1.0;              // current adaptive scale (drives render resolution)
      this.minScale = opts.minScale != null ? opts.minScale : 0.4;   // floor
      this.targetFps = opts.targetFps != null ? opts.targetFps : 120;
      this._frameEMA = 0; this._period = 0; this._lastFrameMs = 0; this._probe = 0;
      this._onCam = typeof opts.onCameraChange === "function" ? opts.onCameraChange : null;
      this._onFps = typeof opts.onFps === "function" ? opts.onFps : null;
      this.nsteps = Math.min(512, Math.max(this.NX, this.NY, this.NZ) * 2);
      // Fewer ray samples while moving (motion masks the slight MIP thin-feature
      // dimming; mean is unaffected) — the raymarch is pixels*steps bound, so this
      // stacks with the dynamic-resolution downscale to reach a high interactive
      // frame rate. The settled frame uses the full step count for a clean still.
      this.nstepsInteract = Math.max(96, Math.round(this.nsteps * 0.5));
      // Camera = quaternion arcball (free rotation, no three.js); see _initCamera.
      this.uniform = device_buf(this.device, 44 * 4);
      // Intensity colormap LUT (256x1 RGBA), stored FLOAT so HDR entries can
      // exceed 1.0. Values are gamma-encoded (extended-sRGB/P3 transfer), matching
      // what the display-p3 + extended-tone-mapping canvas expects — same as the
      // shipped SDR path (rgba8unorm stored the encoded colormap), just float so
      // the HDR lift's >1 headroom survives. The shader reads it verbatim.
      this.lutTex = this.device.createTexture({
        size: [256, 1, 1], format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.colormap = opts.colormap || "gray";
      this._uploadLut(this.colormap);
    }

    /** Upload the 256-entry image colormap LUT as float32 RGBA.
     *
     * SDR (default): the plain image colormap, byte-for-byte identical to the 2D
     * view (generateImageCmapLut / grayscale ramp), just normalised to [0,1].
     * HDR: the JzAzBz-lifted linear Display-P3 colormap (generateImageCmapLutHdr,
     * values >1) re-encoded through the extended-sRGB transfer so the >1 headroom
     * lands in the canvas the same way the SDR encoded values do. */
    _uploadLut(name) {
      const N = 256;
      const out = new Float32Array(N * 4);
      const CM = (typeof window !== "undefined") ? window.ViewerColormap : null;
      let filled = false;
      if (this._hdr && CM && CM.generateImageCmapLutHdr) {
        try {
          const peak = this._headroomVal * (this._gain || 1) * 203.0;  // ×BT.2408 white
          const lin = CM.generateImageCmapLutHdr(name, { auto: true, peakNits: peak });
          if (lin && lin.length >= N * 4) {
            for (let i = 0; i < N * 4; i += 4) {
              out[i]     = _l2gExt(lin[i]);
              out[i + 1] = _l2gExt(lin[i + 1]);
              out[i + 2] = _l2gExt(lin[i + 2]);
              out[i + 3] = 1.0;
            }
            filled = true;
          }
        } catch (e) { filled = false; }
      }
      if (!filled) {                                   // SDR: identical colours to now
        let u8 = null;
        try { if (CM && CM.generateImageCmapLut) u8 = CM.generateImageCmapLut(name); } catch (e) { u8 = null; }
        if (u8 && u8.length >= N * 4) {
          for (let i = 0; i < N * 4; i += 1) out[i] = u8[i] / 255;
        } else {                                       // fallback: grayscale ramp
          for (let i = 0; i < N; i += 1) { const v = i / 255; out[i * 4] = v; out[i * 4 + 1] = v; out[i * 4 + 2] = v; out[i * 4 + 3] = 1.0; }
        }
      }
      this.device.queue.writeTexture({ texture: this.lutTex }, _toF16(out).buffer,
        { bytesPerRow: N * 8, rowsPerImage: 1 }, [N, 1, 1]);
    }

    /** Switch the intensity colormap (e.g. when the 2D view's selector changes). */
    setColormap(name) { this.colormap = name; this._uploadLut(name); this._requestRender(); }

    /** HDR on/off — swaps the LUT between the plain SDR colormap and the lifted
     *  Display-P3 one. Driven by the shared OcdHdrUI toggle. */
    setHdr(on) { this._hdr = !!on; this._uploadLut(this.colormap); this._requestRender(); }
    /** HDR gain (0.25–4): scales the lift's peak-nits target, like the 2D slider. */
    setGain(g) { this._gain = g > 0 ? g : 1.0; if (this._hdr) { this._uploadLut(this.colormap); this._requestRender(); } }

    _uploadTextures(decoded) {
      const { device, NX, NY, NZ } = this;
      // intensity -> normalized [0,1], stored r16float (half the memory bandwidth
      // of r32float). Sampled NEAREST (textureLoad) via a per-voxel DDA — never
      // interpolated.
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
        size: [NX, NY, NZ], dimension: "3d", format: "r16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture({ texture: this.volTex }, _toF16(f).buffer,
        { bytesPerRow: NX * 2, rowsPerImage: NY }, [NX, NY, NZ]);
      this._buildCubeInstances(f);   // object-order prototype instance buffer

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
      if (this.cubePipeline) {
        this.cubeBindGroup = this.device.createBindGroup({
          layout: this.cubeBgl,
          entries: [
            { binding: 0, resource: { buffer: this.cubeUniform } },
            { binding: 1, resource: this.lutTex.createView() },
          ],
        });
      }
    }

    /** Build the per-voxel instance buffer for the cube renderer: one (i,j,k,value)
     *  entry per OCCUPIED voxel. Capped so a dense volume can't allocate unboundedly
     *  (the prototype targets sparse volumes; dense needs culling/slicing later). */
    _buildCubeInstances(f) {
      if (!this.cubePipeline) return;
      const { NX, NY, NZ, device } = this;
      const eps = 0.02;              // occupancy threshold (contributes to MIP)
      const CAP = 4000000;
      let n = 0;
      for (let i = 0; i < f.length; i++) if (f[i] > eps) n++;
      const count = Math.min(n, CAP);
      const data = new Float32Array(count * 4);
      let w = 0;
      for (let z = 0; z < NZ && w < count; z++) {
        for (let y = 0; y < NY && w < count; y++) {
          const row = y * NX + z * NX * NY;
          for (let x = 0; x < NX; x++) {
            const v = f[row + x];
            if (v > eps) { const o = w * 4; data[o] = x; data[o + 1] = y; data[o + 2] = z; data[o + 3] = v; if (++w >= count) break; }
          }
        }
      }
      this.cubeInstanceCount = w;
      if (n > CAP) console.warn(`[cubes] ${n} occupied voxels exceed cap ${CAP}; rendering first ${w} (dense volume — needs culling/slicing)`);
      if (this.cubeInstBuf) this.cubeInstBuf.destroy();
      this.cubeInstBuf = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(this.cubeInstBuf, 0, data);
    }

    _writeCubeUniform(cam) {
      const box = this._box();
      const u = new Float32Array(28);
      u.set(cam.viewProj, 0);
      u.set([box.min[0], box.min[1], box.min[2], 0], 16);
      u.set([box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2], this.gamma], 20);
      u.set([this.NX, this.NY, this.NZ, 0], 24);
      this.device.queue.writeBuffer(this.cubeUniform, 0, u);
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
      const _now = (typeof performance !== "undefined" && performance.now)
        ? () => performance.now() : () => Date.now();
      // Adaptive resolution controller (AIMD): each interactive frame, track the
      // frame interval; shrink the render scale when we're slower than the target
      // and probe it back up when there's headroom. Quantised steps + a probe
      // cooldown keep a steady load from churning the canvas size.
      // Measure the display refresh period once (idle rAF interval = one vsync).
      // This is the budget basis; we must never infer it from render times, since
      // a persistently zoomed-in (slow) session never observes a fast frame.
      (function measureRefresh() {
        let n = 0, last = 0, best = 1e9;
        const tick = (t) => {
          if (last) best = Math.min(best, t - last);
          last = t;
          if (++n < 8) requestAnimationFrame(tick);
          else self._displayPeriod = best;
        };
        if (typeof requestAnimationFrame === "function") requestAnimationFrame((t) => { last = t; requestAnimationFrame(tick); });
      })();
      const tuneScale = () => {
        const nowMs = _now();
        const fdt = nowMs - self._lastFrameMs;
        self._lastFrameMs = nowMs;
        if (fdt <= 0 || fdt > 200) return;            // new gesture / stall — skip
        self._frameEMA = self._frameEMA ? self._frameEMA * 0.8 + fdt * 0.2 : fdt;
        if (self._onFps) self._onFps(1000 / Math.max(self._frameEMA, 0.001), self._dynScale);
        // Target one display refresh (capped so a request for >refresh fps just
        // targets the refresh — you can't beat vsync). Small slack for noise.
        const period = Math.max(self._displayPeriod || (1000 / self.targetFps), 1000 / self.targetFps);
        const budget = period * 1.15;
        if (self._frameEMA > budget) {                // slower than the refresh -> shrink now
          self._dynScale = Math.max(self.minScale, self._dynScale - 0.12);
          self._probe = 40;
        } else if (self._probe > 0) {
          self._probe -= 1;
        } else if (self._dynScale < 1.0) {            // sustaining the refresh with headroom -> probe up
          self._dynScale = Math.min(1.0, self._dynScale + 0.06);
          self._probe = 20;
        }
      };
      const step = () => {
        anim = 0;
        if (drag || Math.abs(vrx) > EPS || Math.abs(vry) > EPS || Math.abs(vpx) > EPS ||
            Math.abs(vpy) > EPS || Math.abs(vz) > EPS) tuneScale();
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
        self._interacting = moving;                 // low-res while moving, full-res on the settle frame
        if (changed || !moving) { self.render(); if (self._onCam) self._onCam(); }
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
      const steps = this._interacting ? (this.nstepsInteract || this.nsteps) : this.nsteps;
      u.set([steps, this.density, this.labelOpacity, this.showLabels], 32);
      u.set([1.0, this.showImage, this.shadeLabels, this.gamma], 36);   // iscale, showImage, shadeLabels, gamma
      u.set([this.ambient, this.specular, this.shininess, this.headlight], 40);  // light
      this.device.queue.writeBuffer(this.uniform, 0, u);
    }

    // Coalesce state-change renders into ONE render per animation frame. Rapid
    // setter calls (an HDR toggle fires setGain+setHdr; a slider drag fires many
    // input events) otherwise each dispatch a separate full raymarch, so the GPU
    // ramps idle->busy repeatedly and irregularly — the bursty power draw the VRM
    // inductors whine at. Deduped to the display refresh, the load is smooth and
    // regular. (The camera loop already renders once per rAF, so it stays direct.)
    _requestRender() {
      if (this._rafPending) return;
      this._rafPending = true;
      const raf = (typeof requestAnimationFrame === "function")
        ? requestAnimationFrame : (cb) => setTimeout(cb, 16);
      raf(() => { this._rafPending = false; this.render(); });
    }

    render() {
      const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
      // Render at native device pixels. CRUCIAL: cap the backing to the device's
      // max 2D texture size — on a big Retina display clientWidth*dpr can exceed
      // it, and getCurrentTexture() then yields nothing, so the volume renders
      // BLANK until something shrinks the canvas (the old supersample path made
      // this far worse). The cap keeps every frame renderable.
      const maxDim = (this.device.limits && this.device.limits.maxTextureDimension2D) || 8192;
      const dyn = this._interacting ? (this._dynScale || 1) : 1;   // adaptive downscale while moving
      let w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr * dyn) || this.canvas.width);
      let h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr * dyn) || this.canvas.height);
      if (w > maxDim || h > maxDim) { const k = maxDim / Math.max(w, h); w = Math.max(1, Math.floor(w * k)); h = Math.max(1, Math.floor(h * k)); }
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      const cam = this._camera();
      const rm = this._renderMode;
      const useCompute = rm === "compute" && this._ensureComputeTargets(w, h);
      const useCubes = (rm === "cubes" || rm === "minimal") && this.cubePipeline && this.cubeInstanceCount > 0;
      if (useCubes) this._writeCubeUniform(cam); else this._writeUniform(cam);
      const enc = this.device.createCommandEncoder();
      if (useCompute) {
        // Image-order march as a compute dispatch -> rgba16float storage texture.
        const cp = enc.beginComputePass();
        cp.setPipeline(this.computePipeline);
        cp.setBindGroup(0, this.computeBindGroup);
        cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8), 1);
        cp.end();
      }
      const rp = enc.beginRenderPass({
        colorAttachments: [{
          view: this.ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store",
        }],
      });
      if (useCompute) {
        rp.setPipeline(this.blitPipeline); rp.setBindGroup(0, this.blitBindGroup); rp.draw(3);
      } else if (useCubes) {
        // Object-order MIP: rasterise the occupied voxels, MAX-blended. No ray loop.
        rp.setPipeline(this.cubePipeline);
        rp.setBindGroup(0, this.cubeBindGroup);
        rp.setVertexBuffer(0, this.cubeVertBuf);
        rp.setVertexBuffer(1, this.cubeInstBuf);
        rp.setIndexBuffer(this.cubeIdxBuf, "uint16");
        const n = (rm === "minimal") ? Math.min(this.cubeInstanceCount, 300) : this.cubeInstanceCount;
        rp.drawIndexed(36, n);
      } else {
        rp.setPipeline(this.pipeline); rp.setBindGroup(0, this.bindGroup); rp.draw(3);
      }
      if (this.overlays) {
        const box = this._box();
        this.overlays.draw(rp, cam.viewProj, box.min, box.max, [this.NX, this.NY, this.NZ]);
      }
      rp.end();
      this.device.queue.submit([enc.finish()]);
    }

    /** (Re)create the compute storage texture + bind groups when the size changes. */
    _ensureComputeTargets(w, h) {
      if (!this.computePipeline) return false;
      if (this._computeTex && this._computeW === w && this._computeH === h) return true;
      if (this._computeTex) this._computeTex.destroy();
      this._computeTex = this.device.createTexture({
        size: [w, h, 1], format: "rgba16float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      this._computeW = w; this._computeH = h;
      const view = this._computeTex.createView();
      this.computeBindGroup = this.device.createBindGroup({
        layout: this.computeBgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: this.volTex.createView() },
          { binding: 2, resource: this.labTex.createView() },
          { binding: 3, resource: this.lutTex.createView() },
          { binding: 4, resource: view },
        ],
      });
      this.blitBindGroup = this.device.createBindGroup({
        layout: this.blitBgl, entries: [{ binding: 0, resource: view }],
      });
      return true;
    }

    /** Render-path experiment. "raymarch" (image-order) | "cubes" (object-order
     *  MIP) | "minimal" (~300 cubes, trivially light raster). Returns the new mode. */
    setRenderMode(mode) {
      const ok = { raymarch: 1, compute: 1, cubes: 1, minimal: 1 };
      this._renderMode = ok[mode] ? mode : "raymarch";
      this._requestRender(); return this._renderMode;
    }
    toggleRenderMode() {   // cycle raymarch -> compute -> cubes -> minimal -> raymarch
      const next = { raymarch: "compute", compute: "cubes", cubes: "minimal", minimal: "raymarch" };
      return this.setRenderMode(next[this._renderMode] || "compute");
    }
    getRenderMode() { return this._renderMode; }

    setMode(m) { this.mode = m | 0; this._requestRender(); }
    setShowImage(on) { this.showImage = on ? 1 : 0; this._requestRender(); }
    setShadeLabels(on) { this.shadeLabels = on ? 1 : 0; this._requestRender(); }
    setGamma(g) { this.gamma = +g > 0 ? +g : 1.0; this._requestRender(); }
    setAmbient(a) { this.ambient = +a; this._requestRender(); }
    setSpecular(s) { this.specular = +s; this._requestRender(); }
    setShininess(s) { this.shininess = +s; this._requestRender(); }
    setHeadlight(on) { this.headlight = on ? 1 : 0; this._requestRender(); }
    setOverlay(name, on) { if (this.overlays) { this.overlays.setEnabled(name, on); this._requestRender(); } }
    setFlowRaw(flowRaw) { if (this.overlays) { this.overlays.setFlow(flowRaw); this._requestRender(); } }
    setDensity(d) { this.density = +d; this._requestRender(); }
    setLabelOpacity(o) { this.labelOpacity = +o; this._requestRender(); }
    setShowLabels(on) { this.showLabels = on ? 1 : 0; this._requestRender(); }
    setZScale(z) { this.zScale = +z; this._requestRender(); }

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
