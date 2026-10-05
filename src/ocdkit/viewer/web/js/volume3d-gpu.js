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

  // 1 - value on half-float bits, through a 64K table (built on first use)
  let _INV16 = null;
  function _invertF16(src) {
    if (!_INV16) {
      const H = halfTable(), f = new Float32Array(65536);
      for (let h = 0; h < 65536; h += 1) f[h] = Number.isFinite(H[h]) ? 1 - H[h] : 0;
      _INV16 = _toF16(f);
    }
    const out = new Uint16Array(src.length);
    for (let i = 0; i < src.length; i += 1) out[i] = _INV16[src[i]];
    return out;
  }

  /** Level the frames (z slices) of a half-float volume: divide each slice by its
   *  median, in the data's own units (valueRange [lo, hi]; the texture holds
   *  (v - lo) / (hi - lo)), and scale to the median of all slice medians. For a
   *  time-lapse or depth stack whose illumination drifts (phase contrast is
   *  illumination x sample, so the correction is a division, not a shift): a
   *  global window can then hide the background of every frame, instead of the
   *  brightest frames lighting up as a textured face of the volume. The median is
   *  read from a 4096-bin histogram per slice. Values clamp to [0, 1]. */
  function _levelFramesF16(src, NX, NY, NZ, valueRange) {
    const H = halfTable(), per = NX * NY, BINS = 4096;
    const lo = valueRange ? +valueRange[0] : 0, span = valueRange ? +valueRange[1] - lo : 1;
    const med = new Float64Array(NZ), hist = new Uint32Array(BINS);
    for (let z = 0; z < NZ; z++) {
      hist.fill(0);
      for (let i = z * per, e = i + per; i < e; i++) {
        const t = H[src[i]];
        hist[Math.min(BINS - 1, Math.max(0, (t * BINS) | 0))]++;
      }
      let acc = 0, b = 0;
      for (; b < BINS; b++) { acc += hist[b]; if (acc * 2 >= per) break; }
      med[z] = lo + ((b + 0.5) / BINS) * span;              // data units
    }
    const sorted = Array.from(med).sort((a, c) => a - c), g = sorted[NZ >> 1];
    const f = new Float32Array(src.length);
    for (let z = 0; z < NZ; z++) {
      const k = med[z] > 0 ? g / med[z] : 1;
      for (let i = z * per, e = i + per; i < e; i++) {
        const v = (lo + H[src[i]] * span) * k;               // data units, leveled
        const t = span > 0 ? (v - lo) / span : 0;
        f[i] = t < 0 ? 0 : t > 1 ? 1 : t;
      }
    }
    return _toF16(f);
  }

  /** Fade the z ends of a half-float volume: the outermost 15% of slices at each
   *  end (at least 4) blend toward the volume's background (its median) on a
   *  raised-cosine ramp. A widefield stack cuts the microscope's out-of-focus
   *  light cones off at its first and last slice; without a fade, rays skimming
   *  those faces see that cut-off halo light as a flat bright sheet, inconsistent
   *  with the same light inside the volume. (Tapering a truncated signal to its
   *  baseline is the standard way to hide the cut.) */
  function _fadeZEndsF16(src, NX, NY, NZ) {
    const H = halfTable(), per = NX * NY, BINS = 4096, hist = new Uint32Array(BINS);
    for (let i = 0; i < src.length; i++) hist[Math.min(BINS - 1, Math.max(0, (H[src[i]] * BINS) | 0))]++;
    let acc = 0, b = 0;
    for (; b < BINS; b++) { acc += hist[b]; if (acc * 2 >= src.length) break; }
    const bg = (b + 0.5) / BINS;
    const w = Math.min(Math.floor(NZ / 2), Math.max(4, Math.round(0.15 * NZ)));
    const f = new Float32Array(src.length);
    for (let z = 0; z < NZ; z++) {
      const d = Math.min(z, NZ - 1 - z);                    // slices from the nearer end
      const k = d >= w ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * (d + 0.5) / w);
      for (let i = z * per, e = i + per; i < e; i++) f[i] = bg + (H[src[i]] - bg) * k;
    }
    return _toF16(f);
  }

  // half-float bits -> float, as a 64K lookup table (built on first use)
  let _HALF = null;
  function halfTable() {
    if (_HALF) return _HALF;
    _HALF = new Float32Array(65536);
    if (typeof Float16Array !== "undefined") {
      const u = new Uint16Array(65536); for (let i = 0; i < 65536; i++) u[i] = i;
      _HALF.set(Float32Array.from(new Float16Array(u.buffer)));
    } else {
      for (let i = 0; i < 65536; i++) {
        const s = i & 0x8000 ? -1 : 1, e = (i >> 10) & 0x1f, m = i & 0x3ff;
        _HALF[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
      }
    }
    return _HALF;
  }

  // ── empty-space-skipping brick grids (raymarch_compute.wgsl bindings 5/6) ──
  // Brick edge B voxels; grid dims ceil(N/B). Values are Z,Y,X-ordered volumes.
  const BRICK = 16;
  function brickDims(NX, NY, NZ, B) { return [Math.ceil(NX / B), Math.ceil(NY / B), Math.ceil(NZ / B)]; }
  /** Max value per brick. `vals` is a Float32Array, or Uint16Array of
   *  half-float bits when `half` is true. Returns Float32Array (bx*by*bz). */
  function brickMax(vals, NX, NY, NZ, B, half) {
    const [bx, by, bz] = brickDims(NX, NY, NZ, B);
    const out = new Float32Array(bx * by * bz), H = half ? halfTable() : null;
    for (let z = 0; z < NZ; z++) {
      const zb = ((z / B) | 0) * by;
      for (let y = 0; y < NY; y++) {
        const rb = (zb + ((y / B) | 0)) * bx, row = (z * NY + y) * NX;
        for (let x = 0; x < NX; x++) {
          const v = H ? H[vals[row + x]] : vals[row + x], i = rb + ((x / B) | 0);
          if (v > out[i]) out[i] = v;
        }
      }
    }
    return out;
  }
  /** 1 per brick holding any nonzero label, else 0. Returns Uint8Array. */
  function brickAny(lab, NX, NY, NZ, B) {
    const [bx, by, bz] = brickDims(NX, NY, NZ, B);
    const out = new Uint8Array(bx * by * bz);
    for (let z = 0; z < NZ; z++) {
      const zb = ((z / B) | 0) * by;
      for (let y = 0; y < NY; y++) {
        const rb = (zb + ((y / B) | 0)) * bx, row = (z * NY + y) * NX;
        for (let x = 0; x < NX; x++) if (lab[row + x]) out[rb + ((x / B) | 0)] = 1;
      }
    }
    return out;
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
      // Fetch every shader source in parallel. Only the compute march (the default
      // path) and its blit are compiled now; the fragment march and the cube
      // renderer are A/B alternatives, compiled on first use (setRenderMode).
      const _get = (u) => fetch(u + _v).then((r) => r.text());
      const [fragCode, cubesCode, ccode, bcode] = await Promise.all([
        _get(opts.shaderUrl || "js/raymarch.wgsl"), _get(opts.cubesUrl || "js/cubes.wgsl").catch(() => null),
        _get(opts.computeUrl || "js/raymarch_compute.wgsl"), _get(opts.blitUrl || "js/blit.wgsl")]);
      self._fragCode = fragCode; self._cubesCode = cubesCode;
      self.bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint", viewDimension: "3d" } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
        ],
      });


      // ── Compute-shader ray-march (A/B via setRenderMode("compute")) ──────────
      // Same march as the fragment path, dispatched as a compute grid writing an
      // rgba16float storage texture, then a trivial blit to the canvas.
      try {
        self.computeModule = device.createShaderModule({ code: ccode });
        self.computeBgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
          ],
        });
        self.computeLayout = device.createPipelineLayout({ bindGroupLayouts: [self.computeBgl] });
        // The render state (mode, which layers, shading) is baked into the shader as
        // pipeline-override constants, so there is one pipeline per state. Build the
        // default state now (fails loudly here if the shader doesn't compile).
        self._computePipes = {};
        self.computePipeline = self._computePipelineFor(opts.mode != null ? opts.mode : 1, 1, 1, 1);
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
      if (self._winData) self._applyWindow(self._winData[0], self._winData[1]);
      self._makeBindGroup();
      self._prewarmComputePipelines();
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
      this._invert = !!opts.invert;                    // inverted display (setInvert)
      this._facesMix = Math.min(1, Math.max(0, +opts.facesMix || 0));   // voxel shading in EA / MIDA (setFacesMix)
      // Render-path experiment: "raymarch" (image-order) | "cubes" (object-order
      // MIP, all occupied voxels) | "minimal" (a few hundred cubes — a trivially
      // light raster load, to test whether the whine is the workload or the
      // per-frame WebGPU present into the extended HDR canvas).
      this._renderMode = opts.renderMode || "raymarch";
      // Compute-shader march is pixel-identical to the fragment path and measurably
      // faster at high resolution (holds frame rate on fast orbits where the
      // fragment path floors its adaptive resolution). Fall back if it failed init.
      if (this._renderMode === "compute" && !this.computePipeline) this._renderMode = "raymarch";
      // EA density = absorption only (0 = pure glow, nothing occludes)
      this.density = opts.density != null ? opts.density : 0.0;
      this._eaExposure = 1.0;                      // set from the data (_updateExposure)
      this._lutPeakAll = 1.0;                      // brightest colormap channel, as stored
      this.labelOpacity = 1.0;                             // opaque labels by default
      this.showImage = (decoded.image || decoded.imageF16) ? 1.0 : 0.0;   // grayscale intensity layer
      this.showLabels = decoded.mask ? 1.0 : 0.0;          // coloured labels, composited on top
      this.shadeLabels = 1.0;                              // diffuse-light the label surfaces
      this.gamma = opts.gamma != null ? opts.gamma : 1.0;  // intensity gamma (matches the 2D slider)
      // HDR: when on, the intensity LUT is the JzAzBz-lifted Display-P3 colormap
      // (values >1 = HDR headroom), exactly like the 2D HDR image layer. The
      // rgba16float / display-p3 / extended canvas emits those >1 values as true
      // HDR. Off = the plain SDR colormap. Driven by the central OcdHdrUI toggle.
      this._hdr = !!opts.hdr;
      this._gain = opts.gain > 0 ? opts.gain : 1.0;
      this._transparent = !!opts.transparent;           // colormap alpha follows lightness
      this._classify = !!opts.classify;                 // window per voxel in EA / MIDA (setClassify)
      this._level = !!opts.levelFrames;                 // frames leveled before display (setLevelFrames)
      this._fadeZ = !!opts.fadeZEnds;                   // z ends faded to background (setFadeZEnds)
      this._depthCue = Math.min(0.99, Math.max(0, +opts.depthCue || 0));   // depth cue strength (setDepthCue)
      this._amipQ = opts.amipPower != null && !Number.isNaN(+opts.amipPower) ? Math.max(0, +opts.amipPower) : Math.log(0.5) / Math.log(0.75);   // AMIP: power q (Infinity: nothing attenuates)
      this._amipSelf = opts.amipSelfDim !== false;   // AMIP: a voxel dims its own light (half its own path)
      this._amipDepth = opts.amipDepth > 0 ? Math.min(1e6, Math.max(0.5, +opts.amipDepth)) : 25;   // AMIP: attenuation depth (voxels)
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
      // Off by default: measured, it misfired even on small volumes (each resize
      // reallocates the compute target, the stall reads as a slow frame, and the
      // controller shrinks again), so a drag went blurry and hitched. The compute
      // march is fast enough at native resolution. opts.adaptiveResolution = true
      // re-enables it.
      this.adaptive = opts.adaptiveResolution === true;
      this._dynScale = 1.0;              // current adaptive scale (drives render resolution)
      this.minScale = opts.minScale != null ? opts.minScale : 0.4;   // floor
      this.targetFps = opts.targetFps != null ? opts.targetFps : 120;
      this._fpsCap = opts.fpsCap > 0 ? opts.fpsCap : 0;   // 0 = uncapped; e.g. 60 to quiet the coil whine
      this._lastRenderT = 0;
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
      this.uniform = device_buf(this.device, 56 * 4);
      // Display window (the 2D histogram bounds), in the volume's data units.
      // valueRange maps those units to the normalized texture; see setWindow.
      this.valueRange = decoded.valueRange || null;
      this._win = [0, 1];                                    // lo, 1/(hi-lo), texture units
      this._winData = Array.isArray(opts.window) ? opts.window.slice() : null;
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
      const HCM = (typeof window !== "undefined") ? window.HdrColormap : null;
      if (this._hdr && HCM && HCM.hdrLutForHeadroom) {
        try {
          // brightest channel = headroom x gain exactly (shared with the 2D image)
          const lin = HCM.hdrLutForHeadroom(name, this._headroomVal, this._gain || 1);
          this._lutPeak = HCM.lutPeak(lin);
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
      // transparent low end: alpha from the colormap's lightness (else 1)
      const HCMa = (typeof window !== "undefined") ? window.HdrColormap : null;
      if (this._transparent && HCMa && HCMa.transparentAlpha) {
        const a = HCMa.transparentAlpha(name);
        for (let i = 0; i < N && i < a.length; i += 1) out[i * 4 + 3] = a[i];
      }
      // brightest stored channel (the EA exposure curve rolls off to it) and a
      // per-entry emission table for the exposure estimate
      let peak = 0;
      this._lutMaxc = new Float32Array(N); this._lutAlpha = new Float32Array(N);
      for (let i = 0; i < N; i += 1) {
        const m = Math.max(out[4 * i], out[4 * i + 1], out[4 * i + 2]);
        this._lutMaxc[i] = m; this._lutAlpha[i] = out[4 * i + 3];
        if (m > peak) peak = m;
      }
      this._lutPeakAll = peak > 0 ? peak : 1;
      this._scheduleExposure();
      this.device.queue.writeTexture({ texture: this.lutTex }, _toF16(out).buffer,
        { bytesPerRow: N * 8, rowsPerImage: 1 }, [N, 1, 1]);
    }

    /** Switch the intensity colormap (e.g. when the 2D view's selector changes). */
    setColormap(name) { this.colormap = name; this._uploadLut(name); this._requestRender(); }

    /** HDR on/off — swaps the LUT between the plain SDR colormap and the lifted
     *  Display-P3 one. Driven by the shared OcdHdrUI toggle. */
    setHdr(on) { this._hdr = !!on; this._uploadLut(this.colormap); this._requestRender(); }
    /** Transparent low end: dark colormap values fade out instead of drawing as
     *  black (in EA they also stop dimming what is behind them). */
    setTransparent(on) {
      const was = this._transparent;
      this._transparent = !!on;
      this._uploadLut(this.colormap);
      if (was !== this._transparent) this._prewarmComputePipelines();   // the other states, in the background
      this._requestRender();
    }
    /** HDR gain (0.25–4): scales the lift's peak-nits target, like the 2D slider. */
    setGain(g) { this._gain = g > 0 ? g : 1.0; if (this._hdr) { this._uploadLut(this.colormap); this._requestRender(); } }

    _uploadTextures(decoded) {
      const { device, NX, NY, NZ } = this;
      // intensity -> normalized [0,1], stored r16float (half the memory bandwidth
      // of r32float). Sampled NEAREST (textureLoad) via a per-voxel DDA — never
      // interpolated.
      const N = NX * NY * NZ;
      let f16;
      if (decoded.imageF16) {
        // already normalized to [0,1] and half-float on the server (GET
        // /api/volume_raw): upload the bytes as-is, no per-voxel work here
        f16 = decoded.imageF16.data;
      } else {
        const f = new Float32Array(N);
        if (decoded.image) {
          const a = decoded.image.data;
          let lo = Infinity, hi = -Infinity;
          for (let i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
          const sc = hi > lo ? 1 / (hi - lo) : 0;
          for (let i = 0; i < N; i++) f[i] = (a[i] - lo) * sc;
          this.valueRange = [lo, hi];
        } else if (decoded.mask) {
          const a = decoded.mask.data;            // no intensity: show label occupancy
          for (let i = 0; i < N; i++) f[i] = a[i] > 0 ? 1 : 0;
        }
        f16 = _toF16(f);
      }
      this._volF16Orig = f16;        // as loaded; the texture holds 1 - value while inverted
      this._volF16Leveled = null;    // frames leveled (setLevelFrames), built on first use
      this._volF16Faded = {};        // z ends faded (setFadeZEnds), per leveling state
      f16 = this._displayF16();
      this._volF16 = f16;            // kept for the cube renderer and the EA exposure estimate
      this._cubesBuilt = false;
      this.volTex = device.createTexture({
        size: [NX, NY, NZ], dimension: "3d", format: "r16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture({ texture: this.volTex }, f16.buffer,
        { bytesPerRow: NX * 2, rowsPerImage: NY }, [NX, NY, NZ]);
      if (this._renderMode === "cubes" || this._renderMode === "minimal") this._ensureCubes();
      this._scheduleExposure();

      // brick grids for the compute march's empty-space skipping
      const bd = brickDims(NX, NY, NZ, BRICK);
      this.brickImgTex = device.createTexture({
        size: bd, dimension: "3d", format: "r16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this._brickMaxHost = brickMax(f16, NX, NY, NZ, BRICK, true);     // (also bounds the depth cue's data box)
      device.queue.writeTexture({ texture: this.brickImgTex }, _toF16(this._brickMaxHost).buffer,
        { bytesPerRow: bd[0] * 2, rowsPerImage: bd[1] }, bd);
      this.brickLabTex = device.createTexture({
        size: bd, dimension: "3d", format: "r8uint",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });

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
      this._writeLabelBricks(lab);
    }

    _writeLabelBricks(lab) {
      const { NX, NY, NZ } = this, bd = brickDims(NX, NY, NZ, BRICK);
      this.device.queue.writeTexture({ texture: this.brickLabTex }, brickAny(lab, NX, NY, NZ, BRICK),
        { bytesPerRow: bd[0], rowsPerImage: bd[1] }, bd);
    }

    /** Pipeline for one render state (override constants), built on first use. */
    _computePipelineFor(mode, showImg, showLab, shade) {
      const tr = this._transparent ? 1 : 0;
      const cl = this._classify ? 1 : 0;
      const cu = this._depthCue > 0 ? 1 : 0;
      const key = `${mode}|${showImg ? 1 : 0}|${showLab ? 1 : 0}|${shade ? 1 : 0}|${tr}|${cl}|${cu}`;
      if (!this._computePipes[key]) {
        this._computePipes[key] = this.device.createComputePipeline({
          layout: this.computeLayout,
          compute: { module: this.computeModule, entryPoint: "cs",
                     constants: { MODE: mode, SHOW_IMG: showImg ? 1 : 0, SHOW_LAB: showLab ? 1 : 0,
                                  SHADE_LAB: shade ? 1 : 0, BRICK, TRANSP: tr, CLASSIFY: cl, CUE: cu } },
        });
      }
      return this._computePipes[key];
    }

    /** Compile every other render state in the background so switching mode or
     *  layers never stalls on a shader compile. */
    _prewarmComputePipelines() {
      if (!this.computeModule || !this.device.createComputePipelineAsync) return;
      const tr = this._transparent ? 1 : 0;              // the current transparency state
      const cl = this._classify ? 1 : 0;                 // and window-per-voxel state
      const cu = this._depthCue > 0 ? 1 : 0;             // and depth cue state
      for (const mode of [0, 1, 2, 3, 4]) for (const img of [0, 1]) for (const lab of [0, 1]) for (const sh of [0, 1]) {
        const key = `${mode}|${img}|${lab}|${sh}|${tr}|${cl}|${cu}`;
        if (this._computePipes[key]) continue;
        this.device.createComputePipelineAsync({
          layout: this.computeLayout,
          compute: { module: this.computeModule, entryPoint: "cs",
                     constants: { MODE: mode, SHOW_IMG: img, SHOW_LAB: lab, SHADE_LAB: sh, BRICK, TRANSP: tr, CLASSIFY: cl, CUE: cu } },
        }).then((p) => { if (!this._computePipes[key]) this._computePipes[key] = p; }).catch(() => {});
      }
    }

    /** Build the object-order cube renderer's instance buffer on first use (it is
     *  an A/B experiment, so a normal load doesn't pay for it). */
    _ensureCubes() {
      this._initCubePipeline();
      if (this._cubesBuilt || !this.cubePipeline || !this._volF16) return;
      const H = halfTable(), h = this._volF16, f = new Float32Array(h.length);
      for (let i = 0; i < h.length; i++) f[i] = H[h[i]];
      this._buildCubeInstances(f);
      this._cubesBuilt = true;
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

    /** Fragment-shader march (A/B alternative to the compute path), built on first use. */
    _ensureFragmentPipeline() {
      if (this.pipeline) return this.pipeline;
      const mod = this.device.createShaderModule({ code: this._fragCode });
      this.pipeline = this.device.createRenderPipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
        vertex: { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list" },
      });
      return this.pipeline;
    }

    /** Object-order cube renderer (A/B experiment), built on first use. */
    _initCubePipeline() {
      if (this.cubePipeline !== undefined || !this._cubesCode) return;
      const device = this.device;
      // ── Object-order cube renderer (MIP prototype; toggle via setRenderMode) ──
      // Rasterises each occupied voxel as a unit cube with MAX blend = MIP, no ray
      // loop. A/B against the raymarch to test whether the coil whine tracks the
      // pipeline (raster vs compute) rather than the workload.
      try {
        const cmod = device.createShaderModule({ code: this._cubesCode });
        this.cubeBgl = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
          ],
        });
        this.cubePipeline = device.createRenderPipeline({
          layout: device.createPipelineLayout({ bindGroupLayouts: [this.cubeBgl] }),
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
              format: this.format,
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
        this.cubeVertBuf = device.createBuffer({ size: corners.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.cubeVertBuf, 0, corners);
        this.cubeIdxBuf = device.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.cubeIdxBuf, 0, idx);
        this.cubeUniform = device.createBuffer({ size: 28 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      } catch (e) { this.cubePipeline = null; console.warn("[cubes] pipeline init failed", e); }
      if (this.cubePipeline) {
        this.cubeBindGroup = device.createBindGroup({
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
      this._writeLabelBricks(buf);
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
      c.addEventListener("pointerup", (e) => {
        const was = drag; drag = 0; try { c.releasePointerCapture(e.pointerId); } catch (_) {}
        if (was) ensureAnim();                          // full-quality frame on release
      });

      // ── Input loop (orbit / pan / zoom) ─────────────────────────────────
      // Pointer and wheel deltas accumulate between frames; each animation frame
      // applies exactly what arrived since the last one and renders once. No
      // smoothing and no momentum: the camera tracks the hand with no lag and
      // stops when the hand stops. (The previous EMA smoothing trailed the input
      // by ~4 degrees and took ~10 frames to settle after the hand stopped.)
      let pdx = 0, pdy = 0, ppx = 0, ppy = 0, pz = 0;   // input accumulated since the last frame
      let anim = 0;
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
        if (!self.adaptive) return;
        // Target one display refresh (capped so a request for >refresh fps just
        // targets the refresh — you can't beat vsync). Small slack for noise.
        const period = Math.max(self._displayPeriod || (1000 / self.targetFps), 1000 / self.targetFps);
        const budget = period * 1.15;
        // WIDE deadband: only change scale when clearly off, then hold. Constantly
        // hunting the scale varies the per-frame GPU load every frame, which is
        // what makes the coil whine 'chirp' (pitch tracks load). Letting the scale
        // settle keeps the load steady -> a steadier, less-obtrusive tone, at no
        // quality cost (it still adapts to hold the frame rate, just stops churning).
        if (self._frameEMA > budget * 1.10) {         // clearly too slow -> shrink
          self._dynScale = Math.max(self.minScale, self._dynScale - 0.12);
          self._probe = 60;
        } else if (self._probe > 0) {
          self._probe -= 1;
        } else if (self._frameEMA < budget * 0.80 && self._dynScale < 1.0) {  // clear headroom -> grow
          self._dynScale = Math.min(1.0, self._dynScale + 0.06);
          self._probe = 40;
        }
        // else: inside the deadband -> HOLD (steady load).
      };
      const step = () => {
        anim = 0;
        const hasInput = pdx !== 0 || pdy !== 0 || ppx !== 0 || ppy !== 0 || pz !== 0;
        if (drag || hasInput) tuneScale();              // fps readout (+ adaptive scale when enabled)
        const H = c.clientHeight || c.height || 1;
        const up = Mat4.quatRotate(self.orient, [0, 1, 0]);
        const right = Mat4.quatRotate(self.orient, [1, 0, 0]);
        if (pdx !== 0 || pdy !== 0) {                   // arcball rotate (no poles)
          const S = (Math.PI * 1.4) / H;
          const q = Mat4.quatMul(Mat4.quatFromAxisAngle(up, -pdx * S), Mat4.quatFromAxisAngle(right, -pdy * S));
          self.orient = Mat4.quatNormalize(Mat4.quatMul(q, self.orient));
        }
        if (ppx !== 0 || ppy !== 0) {                   // pan target in screen plane
          const td = self.radius * Math.tan(self.fovy / 2);
          const px = (2 * ppx * td) / H, py = (2 * ppy * td) / H;
          self.target = [self.target[0] - right[0] * px + up[0] * py,
                         self.target[1] - right[1] * px + up[1] * py,
                         self.target[2] - right[2] * px + up[2] * py];
        }
        if (pz !== 0) {                                 // dolly (radius *= exp(k * wheel delta))
          const diag = Math.hypot(self.NX, self.NY, self.NZ * self.zScale);
          self.radius = Math.max(diag * 0.2, Math.min(diag * 10, self.radius * Math.exp(pz * 0.0015)));
        }
        pdx = 0; pdy = 0; ppx = 0; ppy = 0; pz = 0;
        self._interacting = !!drag;                     // adaptive scale (if enabled) only mid-drag
        // Frame-rate cap (opt-in, for coil whine): while dragging, skip renders
        // inside the cap interval; input keeps accumulating into the camera, and
        // the release frame always renders.
        const capMs = self._fpsCap > 0 ? 1000 / self._fpsCap : 0;
        const nowT = _now();
        const throttled = capMs > 0 && drag && (nowT - self._lastRenderT) < capMs * 0.98;
        if (!throttled) { self.render(); self._lastRenderT = nowT; if (self._onCam) self._onCam(); }
        else anim = requestAnimationFrame(step);        // render the held-back frame next vsync
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

    /** Spin continuously about one of the volume's axes (0 = x, 1 = y, 2 = z,
     *  set with setSpinAxis; z by default, a turntable), in degrees per second.
     *  Time-based, so the speed is the same on any display. Dragging still works
     *  while spinning, and the axis can change mid-spin. */
    setSpin(on, degPerSec) {
      this._spin = !!on;
      this._spinRate = ((degPerSec > 0 ? degPerSec : 30) * Math.PI) / 180;
      if (!this._spin || this._spinRaf || typeof requestAnimationFrame !== "function") {
        if (!this._spin && this._onCam) this._onCam();          // remember where it stopped
        return;
      }
      let last = 0;
      const tick = (t) => {
        if (!this._spin) { this._spinRaf = 0; return; }
        const dt = last ? Math.min(0.1, (t - last) / 1000) : 0;
        last = t;
        if (dt > 0) {
          const a = this._spinAxis == null ? 2 : this._spinAxis;
          const q = Mat4.quatFromAxisAngle([a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0], this._spinRate * dt);
          this.orient = Mat4.quatNormalize(Mat4.quatMul(q, this.orient));
          this.render();
        }
        this._spinRaf = requestAnimationFrame(tick);
      };
      this._spinRaf = requestAnimationFrame(tick);
    }
    isSpinning() { return !!this._spin; }
    setSpinAxis(a) { this._spinAxis = (a === 0 || a === 1) ? a : 2; }

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
      const u = new Float32Array(56);
      u.set(cam.invViewProj, 0);
      u.set([cam.eye[0], cam.eye[1], cam.eye[2], 1], 16);
      u.set([box.min[0], box.min[1], box.min[2], 0], 20);
      u.set([box.max[0], box.max[1], box.max[2], 0], 24);
      u.set([this.NX, this.NY, this.NZ, this.mode], 28);
      const steps = this._interacting ? (this.nstepsInteract || this.nsteps) : this.nsteps;
      u.set([steps, this.density, this.labelOpacity, this.showLabels], 32);
      u.set([1.0, this.showImage, this.shadeLabels, this.gamma], 36);   // iscale, showImage, shadeLabels, gamma
      u.set([this.ambient, this.specular, this.shininess, this.headlight], 40);  // light
      // window, EA exposure, voxel shading (AMIP: its depth and power q instead)
      const blk = this.mode === 4;
      const q = this._amipQ, none = !Number.isFinite(q);              // (q = Infinity: nothing attenuates)
      u.set([this._win[0], this._win[1], blk ? (none ? 1e30 : this._amipDepth) : this._eaExposure, blk ? (none ? 1 : q) : (this._facesMix || 0)], 44);
      const cb = this._cueBox || box;                                   // depth cue: visible data's box, strength
      u.set([cb.min[0], cb.min[1], cb.min[2], this._depthCue || 0], 48);
      u.set([cb.max[0], cb.max[1], cb.max[2], blk && !this._amipSelf ? 1 : 0], 52);   // w: AMIP without self-dimming
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
      if (!this.orient) return;   // a deferred render (exposure update) can land before the camera exists
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
        cp.setPipeline(this._computePipelineFor(this.mode, this.showImage > 0.5, this.showLabels > 0.5, this.shadeLabels > 0.5));
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
        rp.setPipeline(this._ensureFragmentPipeline()); rp.setBindGroup(0, this.bindGroup); rp.draw(3);
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
          { binding: 5, resource: this.brickImgTex.createView() },
          { binding: 6, resource: this.brickLabTex.createView() },
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
      if (this._renderMode === "cubes" || this._renderMode === "minimal") this._ensureCubes();
      this._requestRender(); return this._renderMode;
    }
    toggleRenderMode() {   // cycle raymarch -> compute -> cubes -> minimal -> raymarch
      const next = { raymarch: "compute", compute: "cubes", cubes: "minimal", minimal: "raymarch" };
      return this.setRenderMode(next[this._renderMode] || "compute");
    }
    getRenderMode() { return this._renderMode; }

    /** Cap the interactive frame rate (fps; 0 = uncapped). A lower cap lets the
     *  GPU hold a lower clock -> less coil whine, at full image quality. */
    setFpsCap(fps) { this._fpsCap = fps > 0 ? fps : 0; this._lastRenderT = 0; this._requestRender(); return this._fpsCap; }
    getFpsCap() { return this._fpsCap; }

    setMode(m) { this.mode = m | 0; this._requestRender(); }
    setShowImage(on) { this.showImage = on ? 1 : 0; this._requestRender(); }
    setShadeLabels(on) { this.shadeLabels = on ? 1 : 0; this._requestRender(); }
    setGamma(g) { this.gamma = +g > 0 ? +g : 1.0; this._requestRender(); }

    /** EA exposure from the data: the glow (s^2 per unit length, as the shader)
     *  summed along each axis for every ray (the other two axes subsampled 2x2)
     *  at the data's FULL range, and the k that brings the brightest of those
     *  rays to 0.95 in the shader's 1 - e^(-k glow). EA's value is then in the
     *  data's 0..1 and the display window, gamma and colormap act on it like a
     *  LUT (as in MIP): a full-range window never clips, narrowing it pops. It
     *  depends only on the data (and inversion), never on the window, gamma or
     *  colormap, so moving those never makes the image rescale itself. */
    _scheduleExposure() {
      if (this._expPending) return;
      this._expPending = true;
      const run = () => { this._expPending = false; this._updateExposure(); this._requestRender(); };
      if (typeof setTimeout === "function") setTimeout(run, 0); else run();
    }
    _updateExposure() {
      const v = this._volF16;
      if (!v) return;
      const { NX, NY, NZ } = this, H = halfTable();
      const emit = new Float32Array(65536);               // half-float bits -> glow per unit length (s^2, as the shader)
      for (let h = 0; h < 65536; h += 1) {
        let q = H[h];
        if (!(q > 0)) continue;
        if (q > 1) q = 1;
        emit[h] = q * q;
      }
      const dims = [NX, NY, NZ], strides = [1, NX, NX * NY];
      let top = 0;
      for (let m = 0; m < 3; m += 1) {                    // sums do not depend on direction
        const p = (m + 1) % 3, q = (m + 2) % 3;
        const n = Math.ceil(dims[p] / 2) * Math.ceil(dims[q] / 2), sums = new Float32Array(n);
        for (let step = 0; step < dims[m]; step += 1) {
          const base = step * strides[m];
          let k = 0;
          for (let ip = 0; ip < dims[p]; ip += 2) {
            const rowBase = base + ip * strides[p];
            for (let iq = 0; iq < dims[q]; iq += 2, k += 1) sums[k] += emit[v[rowBase + iq * strides[q]]];
          }
        }
        for (let k = 0; k < n; k += 1) if (sums[k] > top) top = sums[k];
      }
      this._eaExposure = top > 0 ? -Math.log(0.05) / top : 1.0;          // the brightest ray reads 0.95
    }
    /** Display window from the 2D histogram, in the volume's data units (the
     *  0..255 of the viewer's 8-bit volume). Applied like gamma: per sample in
     *  emission-absorption (values below lo turn transparent), and to the
     *  projected value in MIP and mean. */
    setWindow(lo, hi) { this._applyWindow(lo, hi); this._requestRender(); }
    // (no render: create() applies the initial window before the camera exists)
    _applyWindow(lo, hi) {
      this._winData = [lo, hi];
      // the window is a position on the histogram's 0..255 axis, which spans the
      // data's min..max, the same range the texture is normalized over; so it
      // maps directly, at full precision (valueRange is the data's own units)
      let tl = lo / 255, th = hi / 255;
      // inverted: the texture holds 1 - t, so the window [tl, th] becomes
      // [1 - th, 1 - tl], giving (th - t) / (th - tl) = 1 - the normal display
      if (this._invert) { const a = 1 - th; th = 1 - tl; tl = a; }
      this._win = [tl, 1 / Math.max(th - tl, 1e-6)];
      this._updateCueBox();
    }
    /** The depth cue measures depth from the front of the VISIBLE data: the
     *  bounding box (world units) of the bricks whose max is above the window's
     *  low end (brick-coarse, 16 voxels; the whole volume if none). */
    _updateCueBox() {
      const bm = this._brickMaxHost;
      if (!bm || !this._win) { this._cueBox = null; return; }
      const box = this._box();
      const { NX, NY, NZ } = this, [bx, by, bz] = brickDims(NX, NY, NZ, BRICK), lo = this._win[0];
      let x0 = bx, y0 = by, z0 = bz, x1 = -1, y1 = -1, z1 = -1;
      for (let z = 0, i = 0; z < bz; z++) for (let y = 0; y < by; y++) for (let x = 0; x < bx; x++, i++) {
        if (bm[i] > lo) {
          if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
          if (z < z0) z0 = z; if (z > z1) z1 = z;
        }
      }
      if (x1 < 0) { this._cueBox = null; return; }
      const zs = this.zScale || 1;
      this._cueBox = {
        min: [box.min[0] + x0 * BRICK, box.min[1] + y0 * BRICK, box.min[2] + z0 * BRICK * zs],
        max: [box.min[0] + Math.min(NX, (x1 + 1) * BRICK), box.min[1] + Math.min(NY, (y1 + 1) * BRICK),
              box.min[2] + Math.min(NZ, (z1 + 1) * BRICK) * zs],
      };
    }
    /** Depth cue strength, 0 (off) .. 0.99: fades each voxel by 1 / (1 + d / L)^2
     *  with depth d behind the visible data's front (see CUE in raymarch_compute.wgsl).
     *  On/off switches a pipeline constant (the other state compiles in the background). */
    setDepthCue(s) {
      const was = this._depthCue > 0;
      this._depthCue = Math.min(0.99, Math.max(0, Number.isFinite(+s) ? +s : 0));
      this._requestRender();
      if (was !== this._depthCue > 0) this._prewarmComputePipelines();
    }
    /** Inverted display (dark objects bright, e.g. phase contrast): upload 1 - value
     *  and flip the window, so every projection (MIP, mean, EA, MIDA) and the
     *  empty-space bricks work on the inverted intensities unchanged. */
    setInvert(on) {
      on = !!on;
      if (on === !!this._invert) return;
      this._invert = on;
      this._refreshVolume();
    }
    /** Level the frames (z slices): divide each by its median background before
     *  display, so illumination drift along z (a time-lapse, or depth attenuation)
     *  no longer lights up the brightest frames. Applied before Invert. */
    setLevelFrames(on) {
      on = !!on;
      if (on === !!this._level) return;
      this._level = on;
      this._refreshVolume();
    }
    isLevelFrames() { return !!this._level; }
    /** Fade the z ends (the first and last 15% of slices) toward the background, so
     *  the out-of-focus light a widefield stack cuts off there does not show as a flat
     *  sheet on the top and bottom faces. Applied after leveling, before Invert. */
    setFadeZEnds(on) {
      on = !!on;
      if (on === !!this._fadeZ) return;
      this._fadeZ = on;
      this._refreshVolume();
    }
    isFadeZEnds() { return !!this._fadeZ; }
    /** The volume as displayed: as loaded, frames leveled if on, z ends faded if on,
     *  then inverted if on. */
    _displayF16() {
      let f16 = this._volF16Orig;
      if (this._level && f16) {
        if (!this._volF16Leveled) this._volF16Leveled = _levelFramesF16(f16, this.NX, this.NY, this.NZ, this.valueRange);
        f16 = this._volF16Leveled;
      }
      if (this._fadeZ && f16) {
        const key = this._level ? "leveled" : "orig";
        if (!this._volF16Faded) this._volF16Faded = {};
        if (!this._volF16Faded[key]) this._volF16Faded[key] = _fadeZEndsF16(f16, this.NX, this.NY, this.NZ);
        f16 = this._volF16Faded[key];
      }
      return this._invert ? _invertF16(f16) : f16;
    }
    /** Re-upload the displayed volume (after invert / leveling changed) with its bricks. */
    _refreshVolume() {
      const orig = this._volF16Orig;
      if (orig && this.volTex) {
        const { device, NX, NY, NZ } = this;
        const f16 = this._displayF16();
        this._volF16 = f16;
        this._cubesBuilt = false;
        device.queue.writeTexture({ texture: this.volTex }, f16.buffer, { bytesPerRow: NX * 2, rowsPerImage: NY }, [NX, NY, NZ]);
        const bd = brickDims(NX, NY, NZ, BRICK);
        this._brickMaxHost = brickMax(f16, NX, NY, NZ, BRICK, true);
        device.queue.writeTexture({ texture: this.brickImgTex }, _toF16(this._brickMaxHost).buffer,
          { bytesPerRow: bd[0] * 2, rowsPerImage: bd[1] }, bd);
        if (this._renderMode === "cubes" || this._renderMode === "minimal") this._ensureCubes();
      }
      if (this._winData) this._applyWindow(this._winData[0], this._winData[1]);
      this._scheduleExposure();
      this._requestRender();
    }
    isInverted() { return !!this._invert; }
    /** Voxel shading in EA and MIDA, 0..1 (see voxelWeight in raymarch_compute.wgsl):
     *  0 weights each voxel by the ray's path length through it (a cube shaded
     *  like a distance field), 1 weights every voxel the ray crosses the same
     *  (voxel faces), values between blend the two. A uniform, so it costs
     *  nothing and needs no new pipeline. */
    setFacesMix(t) {
      this._facesMix = Math.min(1, Math.max(0, Number.isFinite(+t) ? +t : 0));
      this._requestRender();
    }
    /** Window per voxel in EA and MIDA (CLASSIFY in raymarch_compute.wgsl): values
     *  at or below the window's low end become empty space and opacity follows
     *  the windowed value, instead of compositing the full range and windowing
     *  the result. A pipeline constant, so the other state compiles in the
     *  background, like the transparency toggle. */
    setClassify(on) {
      const was = this._classify;
      this._classify = !!on;
      this._requestRender();
      if (was !== this._classify) this._prewarmComputePipelines();
    }
    isClassify() { return !!this._classify; }
    /** Attenuated MIP (mode 4) depth, voxels: how many voxels of full-brightness
     *  material (the window's top) dim the light passing through them by 95%. */
    setAmipDepth(v) {
      this._amipDepth = Math.min(1e6, Math.max(0.5, Number.isFinite(+v) ? +v : 25));
      this._requestRender();
    }
    /** Attenuated MIP (mode 4) power q (0..Infinity): a voxel attenuates in proportion
     *  to its windowed brightness^q, so higher q = faint material (halo, noise) dims less;
     *  0 = everything above the window's low end alike, Infinity = nothing attenuates. */
    setAmipPower(p) {
      this._amipQ = Number.isNaN(+p) ? 1 : Math.max(0, +p);
      this._requestRender();
    }
    /** Attenuated MIP: whether a voxel dims its own light by half its own path (true, the
     *  gradient across each voxel cube) or shows it at its front face (false). */
    setAmipSelfDim(on) { this._amipSelf = !!on; this._requestRender(); }
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
      this._spin = false;
      try { this.ctx.unconfigure(); } catch (_) {}
      try { this.device.destroy(); } catch (_) {}
    }
  }

  function device_buf(device, size) {
    return device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  return { VolumeGPU, brickMax, brickAny, brickDims, BRICK };
});
