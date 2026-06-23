/*
 * HdrColormap — HDR-lifted image colormaps + a real-time WebGPU renderer.
 *
 * Two parts, sharing one validated lift (a JS port of
 * ocdkit.plot.hdr_cmap.make_hdr_cmap_lit / colormaps' `_liftToHdr`):
 *
 *   • generateImageCmapLutHdr(name, opts) → Float32Array(256*4) of LINEAR
 *     Display-P3 (1.0 = SDR white; >1.0 = HDR). A uniform Jz lift to the
 *     max-chroma "optimal" Jz for the target headroom + a single gamut-capped
 *     chroma scale. So the map gets brighter AND more vivid, never washed.
 *
 *   • HdrColormapRenderer — paints a scalar image through that LUT on a
 *     rgba16float / display-p3 / toneMapping:'extended' WebGPU canvas, so values
 *     >1.0 emit true HDR. Pair with HdrHeadroom and it drives the colormap peak
 *     to the live display headroom (never clipping). setHdr(false) → clean SDR.
 *
 *   const r = await HdrColormap.HdrColormapRenderer.create(canvas, { headroom });
 *   r.setImage(scalarFloat32, w, h);
 *   r.setColormap('viridis', { vmin: 0, vmax: 1 });
 *   r.setHdr(true);                         // false → unlifted SDR, no clip
 *
 * The renderer is engine-agnostic about WHERE the headroom comes from — a plain
 * browser falls back to a fixed multiple; a native webview can inject the real
 * value (see HdrHeadroom).
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.HdrColormap = mod;
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const IMAGE_CMAP_LUT_SIZE = 256;
  const HDR_SDR_WHITE_NITS = 203.0;          // ITU-R BT.2408 reference white
  const HDR_PEAK_NITS_DEFAULT = 1600.0;      // Pro Display XDR peak

  // ── colormap stops (matplotlib/cmap, 13–20 evenly spaced) ────────────────
  const COLORMAP_STOPS = {
    viridis: ['#440154', '#481a6c', '#472f7d', '#414487', '#39568c', '#31688e', '#2a788e', '#23888e', '#1f988b', '#22a884', '#35b779', '#54c568', '#7ad151', '#a5db36', '#d2e21b', '#fde725'],
    magma: ['#000004', '#0c0926', '#1b0c41', '#2f0f60', '#4a0c6b', '#65156e', '#7e2482', '#982d80', '#b73779', '#d5446d', '#ed6059', '#f88a5f', '#feb078', '#fed799', '#fcfdbf'],
    plasma: ['#0d0887', '#3a049a', '#5c01a6', '#7e03a8', '#9c179e', '#b52f8c', '#cc4778', '#de5f65', '#ed7953', '#f89540', '#fdb42f', '#fbd524', '#f0f921'],
    inferno: ['#000004', '#0d0829', '#1b0c41', '#320a5e', '#4a0c6b', '#61136e', '#78206c', '#932667', '#ad305e', '#c73e53', '#df5543', '#f17336', '#f9932e', '#fbb535', '#fad948', '#fcffa4'],
    cividis: ['#00204c', '#00336c', '#2a4858', '#43598e', '#5a6c8a', '#6e7f8e', '#808f8a', '#93a08a', '#a8b08c', '#bdc18d', '#d3d291', '#e8e395', '#fdea45'],
    turbo: ['#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4', '#24eca6', '#61fc6c', '#a4fc3c', '#d1e834', '#f3c63a', '#fe9b2d', '#f56516', '#d93806', '#b11901', '#7a0402'],
    gist_ncar: ['#000080', '#0000d4', '#0044ff', '#0099ff', '#00eeff', '#00ff99', '#00ff00', '#66ff00', '#ccff00', '#ffcc00', '#ff6600', '#ff0000', '#cc0000', '#800000'],
    hot: ['#000000', '#230000', '#460000', '#690000', '#8c0000', '#af0000', '#d20000', '#f50000', '#ff1800', '#ff3b00', '#ff5e00', '#ff8100', '#ffa400', '#ffc700', '#ffea00', '#ffff0d', '#ffff4d', '#ffff8d', '#ffffcd', '#ffffff'],
  };

  function hexToRgb(hex) {
    const v = String(hex).replace('#', '');
    if (v.length !== 6) return [0, 0, 0];
    return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
  }
  function interpolateStops(stops, t) {
    if (!stops || !stops.length) return [0, 0, 0];
    if (stops.length === 1) return hexToRgb(stops[0]);
    const c = Math.min(Math.max(t, 0), 0.999999) * (stops.length - 1);
    const i = Math.floor(c), f = c - i;
    const a = hexToRgb(stops[i]), b = hexToRgb(stops[Math.min(i + 1, stops.length - 1)]);
    return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
  }

  // ── colour math (JzAzBz / PQ / P3) — verbatim from hdr_cmap.py ────────────
  const _PQ_M1 = 0.1593017578125, _PQ_M2 = 134.034375;
  const _PQ_C1 = 0.8359375, _PQ_C2 = 18.8515625, _PQ_C3 = 18.6875;
  const _JZ_B = 1.15, _JZ_G = 0.66, _JZ_D = -0.56, _JZ_D0 = 1.6295499532821566e-11;
  const _XYZ_TO_LMS = [0.41478972, 0.579999, 0.0146480, -0.20151000, 1.120649, 0.0531008, -0.01660080, 0.264800, 0.6684799];
  const _LMS_TO_XYZ = [1.9242264358, -1.0047923126, 0.0376514040, 0.3503167621, 0.7264811939, -0.0653844229, -0.0909828110, -0.3127282905, 1.5227665613];
  const _LMS_TO_IAB = [0.5, 0.5, 0.0, 3.524000, -4.066708, 0.542708, 0.199076, 1.096799, -1.295875];
  const _IAB_TO_LMS = [1.0, 0.1386050432715393, 0.05804731615611882, 1.0, -0.13860504327153927, -0.05804731615611891, 1.0, -0.09601924202631895, -0.811891896056039];
  const _P3_FROM_XYZ = [2.4934969119, -0.9313836179, -0.4027107845, -0.8294889696, 1.7626640603, 0.0236246858, 0.0358458302, -0.0761723893, 0.9568845240];
  const _XYZ_FROM_SRGB = [0.4123907993, 0.3575843394, 0.1804807884, 0.2126390059, 0.7151686788, 0.0721923154, 0.0193308187, 0.1191947798, 0.9505321522];

  function _mv(M, v) {
    return [M[0] * v[0] + M[1] * v[1] + M[2] * v[2], M[3] * v[0] + M[4] * v[1] + M[5] * v[2], M[6] * v[0] + M[7] * v[1] + M[8] * v[2]];
  }
  function _pqForward(x) { x = Math.max(x, 0); const xp = Math.pow(x / 10000, _PQ_M1); return Math.pow((_PQ_C1 + _PQ_C2 * xp) / (1 + _PQ_C3 * xp), _PQ_M2); }
  function _pqInverse(x) { x = Math.max(x, 0); const xp = Math.pow(x, 1 / _PQ_M2); const num = Math.max(xp - _PQ_C1, 0); const den = _PQ_C2 - _PQ_C3 * xp; if (den <= 0) return 0; return 10000 * Math.pow(num / den, 1 / _PQ_M1); }
  function _srgbToLinear(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function _xyzToJzazbz(XYZ) {
    const X = XYZ[0], Y = XYZ[1], Z = XYZ[2];
    const Xp = _JZ_B * X - (_JZ_B - 1) * Z, Yp = _JZ_G * Y - (_JZ_G - 1) * X;
    const LMSp = _mv(_XYZ_TO_LMS, [Xp, Yp, Z]).map(_pqForward);
    const IAB = _mv(_LMS_TO_IAB, LMSp), Iz = IAB[0];
    return [(1 + _JZ_D) * Iz / (1 + _JZ_D * Iz) - _JZ_D0, IAB[1], IAB[2]];
  }
  function _jzazbzToXyz(Jab) {
    const Jz = Jab[0], Iz = (Jz + _JZ_D0) / (1 + _JZ_D - _JZ_D * (Jz + _JZ_D0));
    const LMS = _mv(_IAB_TO_LMS, [Iz, Jab[1], Jab[2]]).map(_pqInverse);
    const XYZm = _mv(_LMS_TO_XYZ, LMS), Xp = XYZm[0], Yp = XYZm[1], Z = XYZm[2];
    const X = (Xp + (_JZ_B - 1) * Z) / _JZ_B, Y = (Yp + (_JZ_G - 1) * X) / _JZ_G;
    return [X, Y, Z];
  }
  function _maxChromaP3(Jz, hzRad, peakMult) {
    const cosH = Math.cos(hzRad), sinH = Math.sin(hzRad);
    let lo = 0, hi = 0.5; const eps = 1e-4, maxVal = peakMult + eps;
    for (let k = 0; k < 24; k += 1) {
      const mid = 0.5 * (lo + hi);
      const rgb = _mv(_P3_FROM_XYZ, _jzazbzToXyz([Jz, mid * cosH, mid * sinH]));
      const inG = rgb[0] >= -eps * HDR_SDR_WHITE_NITS && rgb[0] <= maxVal * HDR_SDR_WHITE_NITS
        && rgb[1] >= -eps * HDR_SDR_WHITE_NITS && rgb[1] <= maxVal * HDR_SDR_WHITE_NITS
        && rgb[2] >= -eps * HDR_SDR_WHITE_NITS && rgb[2] <= maxVal * HDR_SDR_WHITE_NITS;
      if (inG) lo = mid; else hi = mid;
    }
    return lo;
  }

  // ── the lift ─────────────────────────────────────────────────────────────
  const _stopsCache = {};
  function _cmapStopsJab(cmapName) {
    if (_stopsCache[cmapName]) return _stopsCache[cmapName];
    const N = IMAGE_CMAP_LUT_SIZE, stops = COLORMAP_STOPS[cmapName];
    const Jz = new Float64Array(N), Cz = new Float64Array(N), hz = new Float64Array(N);
    for (let i = 0; i < N; i += 1) {
      const t = i / (N - 1);
      let rgb;
      if (stops) rgb = interpolateStops(stops, t);
      else { const v = Math.round(t * 255); rgb = [v, v, v]; }   // grayscale fallback
      const lin = [_srgbToLinear(rgb[0] / 255), _srgbToLinear(rgb[1] / 255), _srgbToLinear(rgb[2] / 255)];
      const XYZ = _mv(_XYZ_FROM_SRGB, lin).map((v) => v * HDR_SDR_WHITE_NITS);
      const jab = _xyzToJzazbz(XYZ);
      Jz[i] = jab[0]; Cz[i] = Math.hypot(jab[1], jab[2]); hz[i] = Math.atan2(jab[2], jab[1]);
    }
    let sdrJzMax = 0; for (let i = 0; i < N; i += 1) if (Jz[i] > sdrJzMax) sdrJzMax = Jz[i];
    const out = { N, Jz, Cz, hz, sdrJzMax };
    _stopsCache[cmapName] = out;
    return out;
  }

  // Uniform Jz scale + single gamut-capped chroma scale (the original _liftToHdr).
  function _liftToHdr(s, hdrJz, peakMult) {
    const jzScale = (s.sdrJzMax > 1e-3) ? (hdrJz / s.sdrJzMax) : 1.0;
    let safeScale = Infinity;
    for (let i = 0; i < s.N; i += 1) {
      if (s.Cz[i] < 1e-3) continue;
      const sc = (_maxChromaP3(s.Jz[i] * jzScale, s.hz[i], peakMult) * 0.95) / s.Cz[i];
      if (sc < safeScale) safeScale = sc;
    }
    if (!isFinite(safeScale) || safeScale < 0.1) safeScale = 1.0;
    return { jzScale, safeScale };
  }

  // Max-chroma "optimal" Jz for a headroom: the lightness where the P3-at-headroom
  // gamut admits the most chroma (most saturated). Headroom-only; cached.
  const _optimalJzCache = {};
  function computeOptimalHdrJz(opts) {
    opts = opts || {};
    const peakMult = (opts.peakNits || HDR_PEAK_NITS_DEFAULT) / HDR_SDR_WHITE_NITS;
    const key = Math.round(peakMult * 1000);
    if (_optimalJzCache[key] != null) return _optimalJzCache[key];
    let bestJz = 0.155, bestCz = 0;
    for (let jz = 0.05; jz < 0.55; jz += 0.005) {
      let minCz = Infinity;
      for (let hd = 0; hd < 360; hd += 15) {
        const cz = _maxChromaP3(jz, hd * Math.PI / 180, peakMult);
        if (cz < minCz) minCz = cz;
      }
      if (minCz > bestCz) { bestCz = minCz; bestJz = jz; }
    }
    _optimalJzCache[key] = bestJz;
    return bestJz;
  }

  function hdrCmapStats(cmapName, opts) {
    opts = opts || {};
    const s = _cmapStopsJab(cmapName);
    const peakMult = (opts.peakNits || HDR_PEAK_NITS_DEFAULT) / HDR_SDR_WHITE_NITS;
    const hdrJz = opts.lift === false ? null
      : (opts.auto ? computeOptimalHdrJz(opts) : (opts.hdrJz || 0.30));
    const jzScale = (hdrJz == null) ? 1.0 : ((s.sdrJzMax > 1e-3) ? hdrJz / s.sdrJzMax : 1.0);
    const safeScale = (hdrJz == null) ? 1.0 : _liftToHdr(s, hdrJz, peakMult).safeScale;
    let peakY = 0;
    for (let i = 0; i < s.N; i += 1) {
      const cz = s.Cz[i] * safeScale;
      const Y = _jzazbzToXyz([s.Jz[i] * jzScale, cz * Math.cos(s.hz[i]), cz * Math.sin(s.hz[i])])[1];
      if (Y > peakY) peakY = Y;
    }
    return { hdrJz, safeScale, headroom: peakMult, peakNits: peakY };
  }

  // → Float32Array(256*4) LINEAR Display-P3 (1.0 = SDR white; >1 = HDR).
  function generateImageCmapLutHdr(cmapName, opts) {
    opts = opts || {};
    const N = IMAGE_CMAP_LUT_SIZE, s = _cmapStopsJab(cmapName);
    const peakMult = (opts.peakNits || HDR_PEAK_NITS_DEFAULT) / HDR_SDR_WHITE_NITS;
    let jzScale = 1.0, safeScale = 1.0;
    if (opts.lift !== false) {
      const hdrJz = (opts.auto === false && opts.hdrJz != null) ? opts.hdrJz : computeOptimalHdrJz(opts);
      const L = _liftToHdr(s, hdrJz, peakMult);
      jzScale = L.jzScale; safeScale = L.safeScale;
    }
    const out = new Float32Array(N * 4);
    for (let i = 0; i < N; i += 1) {
      const cz = s.Cz[i] * safeScale;
      const p3 = _mv(_P3_FROM_XYZ, _jzazbzToXyz([s.Jz[i] * jzScale, cz * Math.cos(s.hz[i]), cz * Math.sin(s.hz[i])]));
      const o = i * 4;
      out[o] = Math.max(p3[0], 0) / HDR_SDR_WHITE_NITS;
      out[o + 1] = Math.max(p3[1], 0) / HDR_SDR_WHITE_NITS;
      out[o + 2] = Math.max(p3[2], 0) / HDR_SDR_WHITE_NITS;
      out[o + 3] = 1.0;
    }
    return out;
  }

  // Gamma-encoded uint8 SDR colormap LUT (256*4 RGBA, Display-P3 transfer) — the
  // unlifted colormap for the WebGL2 SDR backend, matching the WebGPU colours at
  // SDR brightness (same lift source). Written directly to a display-p3 canvas.
  function sdrLutU8(cmapName) {
    const lin = generateImageCmapLutHdr(cmapName, { lift: false });   // linear P3 <=1
    const out = new Uint8Array(IMAGE_CMAP_LUT_SIZE * 4);
    const enc = function (c) { c = Math.max(0, Math.min(1, c)); const e = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; return Math.round(e * 255); };
    for (let i = 0; i < lin.length; i += 4) { out[i] = enc(lin[i]); out[i + 1] = enc(lin[i + 1]); out[i + 2] = enc(lin[i + 2]); out[i + 3] = 255; }
    return out;
  }

  // ── WebGPU renderer ──────────────────────────────────────────────────────
  let _devicePromise = null;
  function getDevice() {
    if (!_devicePromise) {
      _devicePromise = (async () => {
        if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU unavailable');
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('no WebGPU adapter');
        const dev = await adapter.requestDevice();
        if (dev.addEventListener) dev.addEventListener('uncapturederror', (e) => {
          try { console.error('[HdrColormap] WebGPU:', e.error && e.error.message); } catch (_) { /* ignore */ }
        });
        return dev;
      })();
    }
    return _devicePromise;
  }

  // Pre-warm the ONE shared device at module load so the first GPU tile/image
  // doesn't pay cold adapter+device init on its paint path (which shows as a
  // blank canvas for a beat). Now that every ocdkit WebGPU path awaits THIS
  // singleton (colormap tiles, HDR images, linked layers), warming it here is
  // contention-free — there is no second concurrent requestDevice to serialise
  // against. Fire-and-forget; overlaps with the rest of page/script parse.
  try {
    if (typeof navigator !== 'undefined' && navigator.gpu) getDevice().catch(function () {});
  } catch (e) { /* no WebGPU → WebGL2 fallback, nothing to warm */ }

  // Draws the image as a quad in image-pixel space, transformed by a 3x3 matrix
  // (image-px → clip) — the SAME convention the viewer's vertex shader uses, so
  // a host can feed its pan/zoom matrix straight in. m0/m1/m2 are the columns
  // (w ignored), avoiding mat3x3f uniform-stride pitfalls. The colormap LUT is
  // linear (1.0 = SDR white); the shader gamma-encodes for the display-p3 canvas,
  // and toneMapping:'extended' lets >1 emit true HDR. The scalar image is sampled
  // with textureLoad (no float-filtering feature needed).
  const SHADER = `
  struct U { m0: vec4f, m1: vec4f, m2: vec4f, imgSize: vec2f, vmin: f32, vmax: f32, count: f32, gamma: f32, _b: f32, _c: f32 };
  @group(0) @binding(0) var<uniform> u: U;
  @group(0) @binding(1) var<storage, read> lut: array<vec4f>;
  @group(0) @binding(2) var img: texture_2d<f32>;
  struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
  fn l2g(c: f32) -> f32 { let x = max(c, 0.0); if (x <= 0.0031308) { return 12.92 * x; } return 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
  @vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var corners = array<vec2f, 4>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 1));
    let c = corners[i];
    let M = mat3x3f(u.m0.xyz, u.m1.xyz, u.m2.xyz);
    let clip = M * vec3f(c * u.imgSize, 1.0);
    var o: VOut;
    o.pos = vec4f(clip.xy, 0.0, 1.0);
    o.uv = c;
    return o;
  }
  @fragment fn fs(in: VOut) -> @location(0) vec4f {
    let px = clamp(vec2i(in.uv * u.imgSize), vec2i(0), vec2i(u.imgSize) - vec2i(1));
    let val = textureLoad(img, px, 0).r;
    let t0 = clamp((val - u.vmin) / max(u.vmax - u.vmin, 1e-9), 0.0, 1.0);
    let t = pow(t0, u.gamma);
    let idx = min(u32(t * (u.count - 1.0) + 0.5), u32(u.count) - 1u);
    let cc = lut[idx];
    return vec4f(l2g(cc.r), l2g(cc.g), l2g(cc.b), 1.0);
  }`;

  class HdrColormapRenderer {
    // Async because it needs a GPUDevice. opts: { device?, headroom?, hdr=true }.
    static async create(canvas, opts) {
      const r = new HdrColormapRenderer();
      await r._init(canvas, opts || {});
      return r;
    }

    async _init(canvas, opts) {
      this.canvas = canvas;
      this.device = opts.device || await getDevice();
      this.ctx = canvas.getContext('webgpu');
      this._hdr = opts.hdr !== false;
      this._cmap = 'viridis';
      this._vmin = 0; this._vmax = 1;
      this._w = 0; this._h = 0;
      this._cw = 0; this._ch = 0;
      this._raf = 0;
      this.headroom = opts.headroom || null;             // HdrHeadroom instance or null
      this._headroomVal = this.headroom ? this.headroom.value : (opts.headroomValue || 4.0);
      this._configure();

      const mod = this.device.createShaderModule({ code: SHADER });
      // Explicit layout: r32float is 'unfilterable-float' — the 'auto' layout
      // would infer a filterable 'float' sample type, which is invalid for it
      // (async validation error → black canvas, no thrown exception).
      this._bgl = this.device.createBindGroupLayout({
        entries: [
          // uniform holds the transform (vertex) + range/imgSize (both stages)
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        ],
      });
      this.pipeline = this.device.createRenderPipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this._bgl] }),
        vertex: { module: mod, entryPoint: 'vs' },
        fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-strip' },
      });
      this._matrix = null;     // image-px → clip mat3 (col-major 9); null = fit/fill
      this._gamma = 1.0;
      this.backend = 'webgpu'; this.hdr = true;
      this.uBuf = this.device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.lutBuf = this.device.createBuffer({ size: IMAGE_CMAP_LUT_SIZE * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this._mkImageTex(1, 1);
      this._uploadLut();

      if (this.headroom) {
        this._unsub = this.headroom.onChange((v) => { this._headroomVal = v; this._uploadLut(); this.requestRedraw(); });
      }
      // Redraw when the canvas gets/changes its layout size (its backing store
      // tracks CSS px); also covers the load-time race before layout settles.
      if (typeof ResizeObserver !== 'undefined') {
        this._ro = new ResizeObserver(() => this.requestRedraw());
        this._ro.observe(this.canvas);
      }
    }

    _configure() {
      const base = { device: this.device, format: 'rgba16float', colorSpace: 'display-p3', alphaMode: 'premultiplied' };
      try { this.ctx.configure(Object.assign({}, base, { toneMapping: { mode: this._hdr ? 'extended' : 'standard' } })); }
      catch (e) { this.ctx.configure(base); }
    }

    _mkImageTex(w, h) {
      if (this.imgTex) this.imgTex.destroy();
      this.imgTex = this.device.createTexture({
        size: [w, h], format: 'r32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this._bind = this.device.createBindGroup({
        layout: this._bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uBuf } },
          { binding: 1, resource: { buffer: this.lutBuf } },
          { binding: 2, resource: this.imgTex.createView() },
        ],
      });
    }

    // data: Float32Array scalar field, length w*h (row-major, row 0 = top).
    setImage(data, w, h) {
      const f = (data instanceof Float32Array) ? data : Float32Array.from(data);
      if (w !== this._w || h !== this._h) { this._w = w; this._h = h; this._mkImageTex(w, h); }
      // r32float requires bytesPerRow a multiple of 256; w*4 may not be, so pad.
      const bpr = Math.ceil((w * 4) / 256) * 256;
      if (bpr === w * 4) {
        this.device.queue.writeTexture({ texture: this.imgTex }, f, { bytesPerRow: bpr, rowsPerImage: h }, [w, h]);
      } else {
        const padded = new Float32Array((bpr / 4) * h);
        for (let y = 0; y < h; y += 1) padded.set(f.subarray(y * w, y * w + w), y * (bpr / 4));
        this.device.queue.writeTexture({ texture: this.imgTex }, padded, { bytesPerRow: bpr, rowsPerImage: h }, [w, h]);
      }
      this.requestRedraw();
    }

    setRange(vmin, vmax) { this._vmin = vmin; this._vmax = vmax; this.requestRedraw(); }

    // Column-major 3x3 (image-px → clip), e.g. the viewer's computeWebglMatrix
    // output, so the HDR layer tracks pan/zoom exactly. null → fit-to-canvas.
    setTransform(mat3col9) { this._matrix = mat3col9 || null; this.requestRedraw(); }
    setGamma(g) { this._gamma = g || 1.0; this.requestRedraw(); }

    // Fill the canvas, image row 0 at top (used when no transform is set).
    _fillMatrix() { const W = this._w || 1, H = this._h || 1; return [2 / W, 0, 0, 0, -2 / H, 0, -1, 1, 1]; }

    setColormap(name, opts) {
      this._cmap = name;
      if (opts && opts.vmin != null) this._vmin = opts.vmin;
      if (opts && opts.vmax != null) this._vmax = opts.vmax;
      this._uploadLut();
      this.requestRedraw();
    }

    // HDR off → unlifted SDR LUT (values <=1), so standard tone-mapping renders
    // clean SDR instead of hard-clipping the lifted >1 stops to white.
    setHdr(on) { this._hdr = !!on; this._configure(); this._uploadLut(); this.requestRedraw(); }

    setHeadroom(mult) { this._headroomVal = mult; this._uploadLut(); this.requestRedraw(); }

    // Manual HDR gain — scales the lift's peak target (× the live headroom). 1 =
    // the auto/adaptive behavior; <1 dimmer, >1 brighter (clamped by the display).
    setGain(g) { this._gain = g > 0 ? g : 1; this._uploadLut(); this.requestRedraw(); }

    _uploadLut() {
      const peak = this._headroomVal * (this._gain || 1) * HDR_SDR_WHITE_NITS;
      const lut = this._hdr
        ? generateImageCmapLutHdr(this._cmap, { auto: true, peakNits: peak })
        : generateImageCmapLutHdr(this._cmap, { lift: false });
      this.device.queue.writeBuffer(this.lutBuf, 0, lut);
    }

    requestRedraw() { if (!this._raf && typeof requestAnimationFrame !== 'undefined') this._raf = requestAnimationFrame(() => this._render()); }

    _render() {
      this._raf = 0;
      this._renders = (this._renders || 0) + 1;
      if (!this._w) return;
      const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
      const cw = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
      const ch = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
      if (cw !== this._cw || ch !== this._ch) { this._cw = cw; this._ch = ch; this.canvas.width = cw; this.canvas.height = ch; this._configure(); }
      const M = this._matrix || this._fillMatrix();   // columns (w padding ignored)
      const u = new Float32Array(20);
      u[0] = M[0]; u[1] = M[1]; u[2] = M[2];
      u[4] = M[3]; u[5] = M[4]; u[6] = M[5];
      u[8] = M[6]; u[9] = M[7]; u[10] = M[8];
      u[12] = this._w; u[13] = this._h; u[14] = this._vmin; u[15] = this._vmax; u[16] = IMAGE_CMAP_LUT_SIZE; u[17] = this._gamma;
      this.device.queue.writeBuffer(this.uBuf, 0, u);
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
      pass.setPipeline(this.pipeline); pass.setBindGroup(0, this._bind); pass.draw(4); pass.end();
      this.device.queue.submit([enc.finish()]);
    }

    destroy() { if (this._unsub) this._unsub(); if (this.imgTex) this.imgTex.destroy(); }
  }

  return {
    IMAGE_CMAP_LUT_SIZE, HDR_SDR_WHITE_NITS, HDR_PEAK_NITS_DEFAULT, COLORMAP_STOPS,
    generateImageCmapLutHdr, computeOptimalHdrJz, hdrCmapStats, sdrLutU8,
    getDevice, HdrColormapRenderer,
  };
}));
