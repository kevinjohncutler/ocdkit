(function initViewerColormap(global) {
  'use strict';

  // ── Constants ──────────────────────────────────────────────────────────────

  const PALETTE_TEXTURE_SIZE = 1024;
  const DEFAULT_NCOLOR_COUNT = 4;
  const IMAGE_CMAP_LUT_SIZE = 256;

  const IMAGE_COLORMAPS = [
    { value: 'gray', label: 'grayscale' },
    { value: 'gray-clip', label: 'grayclip' },
    { value: 'magma', label: 'magma' },
    { value: 'viridis', label: 'viridis' },
    { value: 'inferno', label: 'inferno' },
    { value: 'plasma', label: 'plasma' },
    { value: 'hot', label: 'hot' },
    { value: 'turbo', label: 'turbo' },
  ];

  const LABEL_COLORMAPS = [
    { value: 'sinebow', label: 'sinebow', hasOffset: true },
    { value: 'viridis', label: 'viridis', hasOffset: false },
    { value: 'magma', label: 'magma', hasOffset: false },
    { value: 'plasma', label: 'plasma', hasOffset: false },
    { value: 'inferno', label: 'inferno', hasOffset: false },
    { value: 'cividis', label: 'cividis', hasOffset: false },
    { value: 'turbo', label: 'turbo', hasOffset: false },
    { value: 'gist_ncar', label: 'gist ncar', hasOffset: false },
    { value: 'vivid', label: 'vivid', hasOffset: true },
    { value: 'pastel', label: 'pastel', hasOffset: true },
    { value: 'gray', label: 'grayscale', hasOffset: false },
  ];

  // Colormap stops from matplotlib/cmap package (pypi cmap)
  // Using 16 evenly-spaced stops for accurate interpolation
  const COLORMAP_STOPS = {
    viridis: [
      '#440154', '#481a6c', '#472f7d', '#414487', '#39568c',
      '#31688e', '#2a788e', '#23888e', '#1f988b', '#22a884',
      '#35b779', '#54c568', '#7ad151', '#a5db36', '#d2e21b', '#fde725'
    ],
    magma: [
      '#000004', '#0c0926', '#1b0c41', '#2f0f60', '#4a0c6b',
      '#65156e', '#7e2482', '#982d80', '#b73779', '#d5446d',
      '#ed6059', '#f88a5f', '#feb078', '#fed799', '#fcfdbf'
    ],
    plasma: [
      '#0d0887', '#3a049a', '#5c01a6', '#7e03a8', '#9c179e',
      '#b52f8c', '#cc4778', '#de5f65', '#ed7953', '#f89540',
      '#fdb42f', '#fbd524', '#f0f921'
    ],
    inferno: [
      '#000004', '#0d0829', '#1b0c41', '#320a5e', '#4a0c6b',
      '#61136e', '#78206c', '#932667', '#ad305e', '#c73e53',
      '#df5543', '#f17336', '#f9932e', '#fbb535', '#fad948', '#fcffa4'
    ],
    cividis: [
      '#00204c', '#00336c', '#2a4858', '#43598e', '#5a6c8a',
      '#6e7f8e', '#808f8a', '#93a08a', '#a8b08c', '#bdc18d',
      '#d3d291', '#e8e395', '#fdea45'
    ],
    turbo: [
      '#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4',
      '#24eca6', '#61fc6c', '#a4fc3c', '#d1e834', '#f3c63a',
      '#fe9b2d', '#f56516', '#d93806', '#b11901', '#7a0402'
    ],
    gist_ncar: [
      '#000080', '#0000d4', '#0044ff', '#0099ff', '#00eeff',
      '#00ff99', '#00ff00', '#66ff00', '#ccff00', '#ffcc00',
      '#ff6600', '#ff0000', '#cc0000', '#800000'
    ],
    hot: [
      '#000000', '#230000', '#460000', '#690000', '#8c0000',
      '#af0000', '#d20000', '#f50000', '#ff1800', '#ff3b00',
      '#ff5e00', '#ff8100', '#ffa400', '#ffc700', '#ffea00',
      '#ffff0d', '#ffff4d', '#ffff8d', '#ffffcd', '#ffffff'
    ],
  };

  // ── Color Math ─────────────────────────────────────────────────────────────

  function sinebowColor(t) {
    const angle = 2 * Math.PI * (t - Math.floor(t));
    const r = Math.sin(angle) * 0.5 + 0.5;
    const g = Math.sin(angle + (2 * Math.PI) / 3) * 0.5 + 0.5;
    const b = Math.sin(angle + (4 * Math.PI) / 3) * 0.5 + 0.5;
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 200];
  }

  function rgbToHex(rgb) {
    const [r, g, b] = rgb;
    return '#' + [r, g, b]
      .map((v) => Math.max(0, Math.min(255, v)).toString(16).padStart(2, '0'))
      .join('');
  }

  function hexToRgb(hex) {
    if (!hex) return [0, 0, 0];
    const value = hex.replace('#', '');
    if (value.length !== 6) return [0, 0, 0];
    const r = parseInt(value.slice(0, 2), 16);
    const g = parseInt(value.slice(2, 4), 16);
    const b = parseInt(value.slice(4, 6), 16);
    return [r, g, b];
  }

  function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = h * 6;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r = 0;
    let g = 0;
    let b = 0;
    if (hp >= 0 && hp < 1) {
      r = c; g = x; b = 0;
    } else if (hp < 2) {
      r = x; g = c; b = 0;
    } else if (hp < 3) {
      r = 0; g = c; b = x;
    } else if (hp < 4) {
      r = 0; g = x; b = c;
    } else if (hp < 5) {
      r = x; g = 0; b = c;
    } else {
      r = c; g = 0; b = x;
    }
    const m = l - c / 2;
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
  }

  function interpolateStops(stops, t) {
    if (!stops || !stops.length) {
      return [0, 0, 0];
    }
    if (stops.length === 1) {
      return hexToRgb(stops[0]);
    }
    const clamped = Math.min(Math.max(t, 0), 0.999999);
    const scaled = clamped * (stops.length - 1);
    const idx = Math.floor(scaled);
    const frac = scaled - idx;
    const a = hexToRgb(stops[idx]);
    const b = hexToRgb(stops[Math.min(idx + 1, stops.length - 1)]);
    return [
      Math.round(a[0] + (b[0] - a[0]) * frac),
      Math.round(a[1] + (b[1] - a[1]) * frac),
      Math.round(a[2] + (b[2] - a[2]) * frac),
    ];
  }

  function seededRandom(seed) {
    let t = seed + 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function hashColorForLabel(label, offset) {
    const golden = 0.61803398875;
    const t = ((label * golden + (offset || 0)) % 1 + 1) % 1;
    const base = sinebowColor(t);
    return [base[0], base[1], base[2]];
  }

  // ── Colormap Lookup ────────────────────────────────────────────────────────

  function colormapHasOffset(cmapValue) {
    const entry = LABEL_COLORMAPS.find(function (c) { return c.value === cmapValue; });
    return entry ? entry.hasOffset : false;
  }

  /**
   * Get color at position t (0-1) for a given colormap name.
   */
  function getColormapColorAtT(t, cmapName) {
    if (cmapName === 'gray') {
      const v = Math.round(t * 255);
      return [v, v, v];
    }
    if (cmapName === 'pastel') {
      return hslToRgb(t, 0.55, 0.72);
    }
    if (cmapName === 'vivid') {
      return hslToRgb(t, 0.9, 0.5);
    }
    if (cmapName === 'sinebow') {
      return sinebowColor(t);
    }
    const stops = COLORMAP_STOPS[cmapName];
    if (stops) {
      return interpolateStops(stops, t);
    }
    return sinebowColor(t);
  }

  /**
   * Get shuffle key for a label. Pure version — accepts shuffle state as params.
   */
  function getLabelShuffleKey(label, shuffle, seed) {
    if (!shuffle) {
      return label;
    }
    const s = (seed | 0) + 1;
    const mix = label ^ (s * 0x9e3779b9);
    return Math.floor(seededRandom(mix) * 1e9);
  }

  // Palette tile size for shuffle=off. Labels 1..N tile the colormap with
  // stride 1/N; label N+1 wraps back to the start (cyclic) or clamps to the
  // endpoint (non-cyclic). Chosen small enough to give visual distinction for
  // typical segmentation counts without needing to track a maxLabel.
  const RAW_PALETTE_TILE = 50;

  /**
   * Get color fraction (0-1) for a label. Pure version.
   *
   * Shuffle off: label N maps to position (N-1)/TILE along the colormap —
   * a fixed stride, no dependence on how many other labels exist. Adjacent
   * labels sit adjacent on the colormap; labels above TILE wrap.
   *
   * Shuffle on: seeded hash, stable per label for a given seed.
   */
  function getLabelColorFraction(label, shuffle, seed, isCyclic, hueOffset) {
    if (shuffle) {
      return seededRandom(getLabelShuffleKey(label, shuffle, seed));
    }
    if (isCyclic) {
      const raw = (label - 1) / RAW_PALETTE_TILE + (hueOffset || 0);
      return ((raw % 1) + 1) % 1;
    }
    const t = ((label - 1) % RAW_PALETTE_TILE) / (RAW_PALETTE_TILE - 1);
    return Math.min(Math.max(t, 0), 1);
  }

  /**
   * Get palette index for a label. Pure version.
   */
  function getLabelOrderValue(label, paletteSize, shuffle, seed) {
    const effectiveSize = paletteSize - 1;
    if (!shuffle) {
      return ((label - 1) % effectiveSize) + 1;
    }
    const seedOffset = (seed | 0) * 97;
    const idx = ((label - 1 + seedOffset) % effectiveSize);
    return idx + 1;
  }

  /**
   * Get colormap color for a label. Pure version — accepts colormap state as params.
   */
  function getColormapColor(label, colormap, shuffle, seed, hueOffset) {
    if (label <= 0) return null;
    const isCyclic = colormapHasOffset(colormap);
    const t = getLabelColorFraction(label, shuffle, seed, isCyclic, hueOffset);
    return getColormapColorAtT(t, colormap);
  }

  // ── Palette Generation ─────────────────────────────────────────────────────

  function generateNColorSwatches(count, hueOffset, colormap) {
    const swatches = [];
    const total = Math.max(2, count);
    const offset = hueOffset || 0;
    const hasCyclicOffset = colormapHasOffset(colormap);

    for (let i = 0; i < total; i += 1) {
      const t = hasCyclicOffset
        ? (offset + i / total) % 1
        : i / (total - 1 || 1);
      const rgb = getColormapColorAtT(t, colormap);
      swatches.push([rgb[0], rgb[1], rgb[2]]);
    }
    return swatches;
  }

  function ensureNColorPaletteLength(targetCount, currentColors, defaultCount, colormap) {
    const target = Math.max(2, targetCount | 0);
    const base = (currentColors && currentColors.length)
      ? currentColors.slice()
      : generateNColorSwatches(defaultCount || DEFAULT_NCOLOR_COUNT, 0.35, colormap);
    if (base.length >= target) {
      return base;
    }
    const next = base.slice();
    for (let i = next.length; i < target; i += 1) {
      const t = (0.35 + i / Math.max(target, 2)) % 1;
      const rgb = sinebowColor(t);
      next.push([rgb[0], rgb[1], rgb[2]]);
    }
    return next;
  }

  function generateSinebowPalette(size, offset, sequential) {
    const count = Math.max(size, 2);
    const palette = new Array(count);
    palette[0] = [0, 0, 0, 0];
    const golden = 0.61803398875;
    for (let i = 1; i < count; i += 1) {
      const t = sequential
        ? ((offset || 0) + (i - 1) / (count - 1)) % 1
        : ((offset || 0) + i * golden) % 1;
      palette[i] = sinebowColor(t);
    }
    return palette;
  }

  /**
   * Build shuffle permutation for a given max label count.
   * Creates a bijection [1..N] -> [1..N] using golden ratio.
   */
  function buildShufflePermutation(maxLabel, seed) {
    const N = maxLabel;
    const golden = 0.61803398875;
    const items = [];
    for (let i = 1; i <= N; i++) {
      const seedOffset = (seed | 0) * 0.1;
      const sortKey = ((i + seedOffset) * golden) % 1;
      items.push({ label: i, sortKey: sortKey });
    }
    items.sort(function (a, b) { return a.sortKey - b.sortKey; });
    const perm = new Array(N + 1);
    perm[0] = 0;
    for (let rank = 0; rank < items.length; rank++) {
      perm[items[rank].label] = rank + 1;
    }
    return perm;
  }

  // ── LUT & Texture Data ────────────────────────────────────────────────────

  function generateImageCmapLut(cmapName) {
    const data = new Uint8Array(IMAGE_CMAP_LUT_SIZE * 4);
    const stops = COLORMAP_STOPS[cmapName];

    for (let i = 0; i < IMAGE_CMAP_LUT_SIZE; i++) {
      const t = i / (IMAGE_CMAP_LUT_SIZE - 1);
      var rgb;
      if (stops) {
        rgb = interpolateStops(stops, t);
      } else {
        const v = Math.round(t * 255);
        rgb = [v, v, v];
      }
      const offset = i * 4;
      data[offset] = rgb[0];
      data[offset + 1] = rgb[1];
      data[offset + 2] = rgb[2];
      data[offset + 3] = 255;
    }
    return data;
  }

  // ── HDR-lifted image colormap LUT ──────────────────────────────────────────
  //
  // Port of ocdkit.plot.hdr_cmap.make_hdr_cmap_lut: lift any image colormap into
  // HDR Display-P3 via a single uniform JzAzBz brightness (Jz) scale plus a
  // single gamut-capped chroma (Cz) scale — one multiplier each, no per-hue
  // distortion, so the colormap's perceptual shape is preserved.
  //
  // Output is gamma-encoded EXTENDED Display-P3 (the sRGB/P3 transfer function
  // continued past 1.0) where 1.0 == SDR reference white and values >1.0 are
  // HDR headroom — exactly what a float16 `drawingBufferColorSpace='display-p3'`
  // canvas expects (mirrors io/figure.py's label popup). Feed the returned
  // Float32Array straight into an RGBA16F LUT texture (FLOAT upload, LINEAR
  // filter). Same reference levels as the Python module, so a viewer-rendered
  // cmap matches a baked Ultra-HDR tile.
  const HDR_SDR_WHITE_NITS = 203.0;       // ITU-R BT.2408 reference white
  const HDR_PEAK_NITS_DEFAULT = 1600.0;   // Apple Pro Display XDR peak
  const HDR_JZ_DEFAULT = 0.30;            // brightest stop ~600 nits; 0.40+ "wow"

  // PQ (Safdar's modified m2), JzAzBz + color matrices — verbatim from hdr_cmap.
  const _PQ_M1 = 0.1593017578125, _PQ_M2 = 134.034375;
  const _PQ_C1 = 0.8359375, _PQ_C2 = 18.8515625, _PQ_C3 = 18.6875;
  const _JZ_B = 1.15, _JZ_G = 0.66, _JZ_D = -0.56, _JZ_D0 = 1.6295499532821566e-11;
  // row-major 3x3
  const _XYZ_TO_LMS = [0.41478972, 0.579999, 0.0146480, -0.20151000, 1.120649, 0.0531008, -0.01660080, 0.264800, 0.6684799];
  const _LMS_TO_XYZ = [1.9242264358, -1.0047923126, 0.0376514040, 0.3503167621, 0.7264811939, -0.0653844229, -0.0909828110, -0.3127282905, 1.5227665613];
  const _LMS_TO_IAB = [0.5, 0.5, 0.0, 3.524000, -4.066708, 0.542708, 0.199076, 1.096799, -1.295875];
  const _IAB_TO_LMS = [1.0, 0.1386050432715393, 0.05804731615611882, 1.0, -0.13860504327153927, -0.05804731615611891, 1.0, -0.09601924202631895, -0.811891896056039];
  const _P3_FROM_XYZ = [2.4934969119, -0.9313836179, -0.4027107845, -0.8294889696, 1.7626640603, 0.0236246858, 0.0358458302, -0.0761723893, 0.9568845240];
  const _XYZ_FROM_SRGB = [0.4123907993, 0.3575843394, 0.1804807884, 0.2126390059, 0.7151686788, 0.0721923154, 0.0193308187, 0.1191947798, 0.9505321522];

  function _mv(M, v) {  // row-major mat3 · vec3
    return [
      M[0] * v[0] + M[1] * v[1] + M[2] * v[2],
      M[3] * v[0] + M[4] * v[1] + M[5] * v[2],
      M[6] * v[0] + M[7] * v[1] + M[8] * v[2],
    ];
  }
  function _pqForward(x) {
    x = Math.max(x, 0.0);
    const xp = Math.pow(x / 10000.0, _PQ_M1);
    return Math.pow((_PQ_C1 + _PQ_C2 * xp) / (1.0 + _PQ_C3 * xp), _PQ_M2);
  }
  function _pqInverse(x) {
    x = Math.max(x, 0.0);
    const xp = Math.pow(x, 1.0 / _PQ_M2);
    const num = Math.max(xp - _PQ_C1, 0.0);
    const den = _PQ_C2 - _PQ_C3 * xp;
    if (den <= 0.0) return 0.0;             // above the PQ peak
    return 10000.0 * Math.pow(num / den, 1.0 / _PQ_M1);
  }
  function _srgbToLinear(c) {
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  // sRGB OETF, continued past 1.0 (extended encoding for HDR headroom).
  function _srgbEncodeExt(c) {
    c = Math.max(c, 0.0);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
  }
  function _xyzToJzazbz(XYZ) {
    const X = XYZ[0], Y = XYZ[1], Z = XYZ[2];
    const Xp = _JZ_B * X - (_JZ_B - 1) * Z;
    const Yp = _JZ_G * Y - (_JZ_G - 1) * X;
    const LMSp = _mv(_XYZ_TO_LMS, [Xp, Yp, Z]).map(_pqForward);
    const IAB = _mv(_LMS_TO_IAB, LMSp);
    const Iz = IAB[0];
    const Jz = (1.0 + _JZ_D) * Iz / (1.0 + _JZ_D * Iz) - _JZ_D0;
    return [Jz, IAB[1], IAB[2]];
  }
  function _jzazbzToXyz(Jab) {
    const Jz = Jab[0];
    const Iz = (Jz + _JZ_D0) / (1.0 + _JZ_D - _JZ_D * (Jz + _JZ_D0));
    const LMS = _mv(_IAB_TO_LMS, [Iz, Jab[1], Jab[2]]).map(_pqInverse);
    const XYZm = _mv(_LMS_TO_XYZ, LMS);
    const Xp = XYZm[0], Yp = XYZm[1], Z = XYZm[2];
    const X = (Xp + (_JZ_B - 1.0) * Z) / _JZ_B;
    const Y = (Yp + (_JZ_G - 1.0) * X) / _JZ_G;
    return [X, Y, Z];
  }
  // Largest Cz keeping (Jz, hue) inside Display-P3 at peakMult x SDR white.
  function _maxChromaP3(Jz, hzRad, peakMult) {
    const cosH = Math.cos(hzRad), sinH = Math.sin(hzRad);
    let lo = 0.0, hi = 0.5;
    const eps = 1e-4, maxVal = peakMult + eps;
    for (let k = 0; k < 24; k += 1) {
      const mid = 0.5 * (lo + hi);
      const rgb = _mv(_P3_FROM_XYZ, _jzazbzToXyz([Jz, mid * cosH, mid * sinH]));
      const inGamut = (rgb[0] >= -eps * HDR_SDR_WHITE_NITS) && (rgb[0] <= maxVal * HDR_SDR_WHITE_NITS)
        && (rgb[1] >= -eps * HDR_SDR_WHITE_NITS) && (rgb[1] <= maxVal * HDR_SDR_WHITE_NITS)
        && (rgb[2] >= -eps * HDR_SDR_WHITE_NITS) && (rgb[2] <= maxVal * HDR_SDR_WHITE_NITS);
      if (inGamut) lo = mid; else hi = mid;
    }
    return lo;
  }

  // Sample `cmapName` at IMAGE_CMAP_LUT_SIZE stops → per-stop JzAzBz (brightness
  // Jz, chroma Cz, hue hz) + the SDR brightness peak. Cached per colormap since
  // the auto-tuning search evaluates a lift many times.
  const _stopsCache = {};
  function _cmapStopsJab(cmapName) {
    if (_stopsCache[cmapName]) return _stopsCache[cmapName];
    const N = IMAGE_CMAP_LUT_SIZE;
    const stops = COLORMAP_STOPS[cmapName];
    const Jz = new Float64Array(N), Cz = new Float64Array(N), hz = new Float64Array(N);
    for (let i = 0; i < N; i += 1) {
      const t = i / (N - 1);
      let rgb;
      if (stops) {
        rgb = interpolateStops(stops, t);
      } else {
        const v = Math.round(t * 255);
        rgb = [v, v, v];                     // grayscale fallback
      }
      const lin = [_srgbToLinear(rgb[0] / 255), _srgbToLinear(rgb[1] / 255), _srgbToLinear(rgb[2] / 255)];
      const XYZ = _mv(_XYZ_FROM_SRGB, lin).map(function (v) { return v * HDR_SDR_WHITE_NITS; });
      const jab = _xyzToJzazbz(XYZ);
      Jz[i] = jab[0];
      Cz[i] = Math.hypot(jab[1], jab[2]);
      hz[i] = Math.atan2(jab[2], jab[1]);
    }
    let sdrJzMax = 0.0;
    for (let i = 0; i < N; i += 1) if (Jz[i] > sdrJzMax) sdrJzMax = Jz[i];
    const out = { N, Jz, Cz, hz, sdrJzMax };
    _stopsCache[cmapName] = out;
    return out;
  }

  // Original `_liftToHdr` (colormaps/js/ui.js) — verbatim algorithm, also the one
  // in ocdkit.plot.hdr_cmap.make_hdr_cmap_lut: a single uniform Jz scale (brings
  // the brightest stop to `hdrJz`) plus a single uniform chroma scale = the
  // largest that keeps EVERY stop inside P3 at `peakMult`× SDR white (min over
  // stops of gamut-max·0.95 / Cz). One multiplier each → no per-hue distortion,
  // the colormap's exact shape is preserved. The wash-out was never here — it was
  // the WebGL2 hard-clip renderer; WebGPU extended tone-mapping fixes that.
  function _liftToHdr(s, hdrJz, peakMult) {
    const jzScale = (s.sdrJzMax > 1e-3) ? (hdrJz / s.sdrJzMax) : 1.0;
    let safeScale = Infinity;
    for (let i = 0; i < s.N; i += 1) {
      if (s.Cz[i] < 1e-3) continue;
      const maxCz = _maxChromaP3(s.Jz[i] * jzScale, s.hz[i], peakMult);
      const sc = (maxCz * 0.95) / s.Cz[i];
      if (sc < safeScale) safeScale = sc;
    }
    if (!isFinite(safeScale) || safeScale < 0.1) safeScale = 1.0;
    return { jzScale, safeScale };
  }

  // Optimal HDR Jz (colormaps/js/ui.js `_setOptimalHdrJz`): the lightness at which
  // the P3-at-headroom gamut admits the MOST chroma — the max inscribed Cz (over
  // all hues) is largest — i.e. where colours can be most saturated. This is the
  // key to "HDR without losing saturation": don't maximise brightness, sit at the
  // vivid sweet spot. Depends only on the headroom (colormap-independent), cached.
  const _optimalJzCache = {};
  function computeOptimalHdrJz(opts) {
    opts = opts || {};
    const peakMult = (opts.peakNits || HDR_PEAK_NITS_DEFAULT) / HDR_SDR_WHITE_NITS;
    const key = Math.round(peakMult * 1000);
    if (_optimalJzCache[key] != null) return _optimalJzCache[key];
    let bestJz = 0.155, bestCz = 0.0;
    for (let jz = 0.05; jz < 0.55; jz += 0.005) {   // 0.55 ≈ 2000 nits — let big headroom keep climbing
      let minCz = Infinity;                          // max inscribed chroma at this Jz
      for (let hd = 0; hd < 360; hd += 15) {
        const cz = _maxChromaP3(jz, hd * Math.PI / 180.0, peakMult);
        if (cz < minCz) minCz = cz;
      }
      if (minCz > bestCz) { bestCz = minCz; bestJz = jz; }
    }
    _optimalJzCache[key] = bestJz;
    return bestJz;
  }

  // Lift diagnostics for UI readouts. { hdrJz, safeScale (gamut chroma cap),
  // headroom, peakNits (brightest stop luminance, cd/m²) }.
  function hdrCmapStats(cmapName, opts) {
    opts = opts || {};
    const s = _cmapStopsJab(cmapName);
    const peakMult = (opts.peakNits || HDR_PEAK_NITS_DEFAULT) / HDR_SDR_WHITE_NITS;
    const hdrJz = opts.lift === false ? null
      : (opts.auto ? computeOptimalHdrJz(opts) : (opts.hdrJz || HDR_JZ_DEFAULT));
    const jzScale = (hdrJz == null) ? 1.0 : ((s.sdrJzMax > 1e-3) ? hdrJz / s.sdrJzMax : 1.0);
    const safeScale = (hdrJz == null) ? 1.0 : _liftToHdr(s, hdrJz, peakMult).safeScale;
    let peakY = 0.0;
    for (let i = 0; i < s.N; i += 1) {
      const cz = s.Cz[i] * safeScale;
      const Y = _jzazbzToXyz([s.Jz[i] * jzScale, cz * Math.cos(s.hz[i]), cz * Math.sin(s.hz[i])])[1];
      if (Y > peakY) peakY = Y;
    }
    return { hdrJz: hdrJz, safeScale: safeScale, headroom: peakMult, peakNits: peakY };
  }

  // Sample colormap `cmapName` and lift to LINEAR Display-P3 (1.0 == SDR white;
  // values >1.0 are HDR — matches colormaps/js `_jzabToP3Display`). The WebGPU
  // strip shader gamma-encodes; an rgba16float + display-p3 + extended-tone-
  // mapping canvas then rolls values >1 gracefully into the display headroom (no
  // per-channel hard clip → no wash-out).
  // opts: { lift=true, auto=true, hdrJz, peakNits=1600 }.
  //   auto=true (default when lifting) → optimal max-chroma Jz for the headroom.
  //   lift=false → SDR baseline (no Jz/Cz scaling, all values <=1.0).
  // Returns Float32Array(IMAGE_CMAP_LUT_SIZE * 4) of linear-light Display-P3.
  function generateImageCmapLutHdr(cmapName, opts) {
    opts = opts || {};
    const N = IMAGE_CMAP_LUT_SIZE;
    const s = _cmapStopsJab(cmapName);
    const peakMult = (opts.peakNits || HDR_PEAK_NITS_DEFAULT) / HDR_SDR_WHITE_NITS;

    let jzScale = 1.0, safeScale = 1.0;
    if (opts.lift !== false) {
      const hdrJz = (opts.auto === false && opts.hdrJz != null)
        ? opts.hdrJz : computeOptimalHdrJz(opts);
      const L = _liftToHdr(s, hdrJz, peakMult);
      jzScale = L.jzScale; safeScale = L.safeScale;
    }

    const out = new Float32Array(N * 4);
    for (let i = 0; i < N; i += 1) {
      const cz = s.Cz[i] * safeScale;
      const p3 = _mv(_P3_FROM_XYZ, _jzazbzToXyz([s.Jz[i] * jzScale, cz * Math.cos(s.hz[i]), cz * Math.sin(s.hz[i])]));
      const o = i * 4;
      out[o]     = Math.max(p3[0], 0.0) / HDR_SDR_WHITE_NITS;   // LINEAR P3, >1.0 = HDR
      out[o + 1] = Math.max(p3[1], 0.0) / HDR_SDR_WHITE_NITS;
      out[o + 2] = Math.max(p3[2], 0.0) / HDR_SDR_WHITE_NITS;
      out[o + 3] = 1.0;
    }
    return out;
  }

  /**
   * Build palette texture data. Pure version — accepts state as params.
   */
  function buildPaletteTextureData(opts) {
    var size = PALETTE_TEXTURE_SIZE;
    var data = new Uint8Array(size * 4);
    if (opts.nColorActive) {
      var palette = (opts.paletteColors && opts.paletteColors.length)
        ? opts.paletteColors
        : generateNColorSwatches(opts.defaultCount || DEFAULT_NCOLOR_COUNT, 0.35, opts.colormap);
      var count = palette.length || 1;
      for (var i = 0; i < size; i += 1) {
        var rgb = i === 0 ? [0, 0, 0] : (palette[(i - 1) % count] || [0, 0, 0]);
        var base = i * 4;
        data[base] = rgb[0] || 0;
        data[base + 1] = rgb[1] || 0;
        data[base + 2] = rgb[2] || 0;
        data[base + 3] = 255;
      }
      return data;
    }
    for (var j = 0; j < size; j += 1) {
      var rgb2 = j === 0 ? [0, 0, 0]
        : (getColormapColor(j, opts.colormap, opts.shuffle, opts.seed, opts.hueOffset) || [0, 0, 0]);
      var base2 = j * 4;
      data[base2] = rgb2[0] || 0;
      data[base2 + 1] = rgb2[1] || 0;
      data[base2 + 2] = rgb2[2] || 0;
      data[base2 + 3] = 255;
    }
    return data;
  }

  // ── CSS Gradient ───────────────────────────────────────────────────────────

  function generateColormapGradient(cmapValue, numStops) {
    var n = numStops || 32;
    var stops = [];
    for (var i = 0; i < n; i++) {
      var t = i / (n - 1);
      var rgb = getColormapColorAtT(t, cmapValue);
      var pct = (t * 100).toFixed(1);
      stops.push('rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ') ' + pct + '%');
    }
    return 'linear-gradient(to right, ' + stops.join(', ') + ')';
  }

  // ── Utility ────────────────────────────────────────────────────────────────

  function collectLabelsFromMask(sourceMask) {
    var seen = new Set();
    for (var i = 0; i < sourceMask.length; i += 1) {
      var value = sourceMask[i];
      if (value > 0) {
        seen.add(value);
      }
    }
    return Array.from(seen).sort(function (a, b) { return a - b; });
  }

  function getImageCmapTypeValue(imageColormap) {
    if (imageColormap === 'gray') return 0;
    if (imageColormap === 'gray-clip') return 1;
    return 2;
  }

  // ── Export ─────────────────────────────────────────────────────────────────

  var api = global.ViewerColormap || {};
  Object.assign(api, {
    // Constants
    PALETTE_TEXTURE_SIZE: PALETTE_TEXTURE_SIZE,
    DEFAULT_NCOLOR_COUNT: DEFAULT_NCOLOR_COUNT,
    IMAGE_CMAP_LUT_SIZE: IMAGE_CMAP_LUT_SIZE,
    IMAGE_COLORMAPS: IMAGE_COLORMAPS,
    LABEL_COLORMAPS: LABEL_COLORMAPS,
    COLORMAP_STOPS: COLORMAP_STOPS,
    // Color math
    sinebowColor: sinebowColor,
    rgbToHex: rgbToHex,
    hexToRgb: hexToRgb,
    hslToRgb: hslToRgb,
    interpolateStops: interpolateStops,
    seededRandom: seededRandom,
    hashColorForLabel: hashColorForLabel,
    // Colormap lookup
    colormapHasOffset: colormapHasOffset,
    getColormapColorAtT: getColormapColorAtT,
    getLabelShuffleKey: getLabelShuffleKey,
    getLabelColorFraction: getLabelColorFraction,
    getLabelOrderValue: getLabelOrderValue,
    getColormapColor: getColormapColor,
    getImageCmapTypeValue: getImageCmapTypeValue,
    // Palette generation
    generateNColorSwatches: generateNColorSwatches,
    ensureNColorPaletteLength: ensureNColorPaletteLength,
    generateSinebowPalette: generateSinebowPalette,
    buildShufflePermutation: buildShufflePermutation,
    // LUT & texture data
    generateImageCmapLut: generateImageCmapLut,
    generateImageCmapLutHdr: generateImageCmapLutHdr,
    computeOptimalHdrJz: computeOptimalHdrJz,
    hdrCmapStats: hdrCmapStats,
    HDR_SDR_WHITE_NITS: HDR_SDR_WHITE_NITS,
    HDR_PEAK_NITS_DEFAULT: HDR_PEAK_NITS_DEFAULT,
    HDR_JZ_DEFAULT: HDR_JZ_DEFAULT,
    buildPaletteTextureData: buildPaletteTextureData,
    // CSS
    generateColormapGradient: generateColormapGradient,
    // Utility
    collectLabelsFromMask: collectLabelsFromMask,
  });
  global.ViewerColormap = api;

})(typeof window !== 'undefined' ? window : globalThis);
