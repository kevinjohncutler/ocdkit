(function initOmniColormap(global) {
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

  /**
   * Get color fraction (0-1) for a label. Pure version.
   */
  function getLabelColorFraction(label, shuffle, seed, maxLabel) {
    const max = Math.max(maxLabel, 2);
    if (!shuffle) {
      return ((label - 1) % max) / (max - 1);
    }
    return seededRandom(getLabelShuffleKey(label, shuffle, seed));
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
  function getColormapColor(label, colormap, shuffle, seed, maxLabel) {
    if (label <= 0) return null;
    const t = getLabelColorFraction(label, shuffle, seed, maxLabel);
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
        : (getColormapColor(j, opts.colormap, opts.shuffle, opts.seed, opts.maxLabel) || [0, 0, 0]);
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

  var api = global.OmniColormap || {};
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
    buildPaletteTextureData: buildPaletteTextureData,
    // CSS
    generateColormapGradient: generateColormapGradient,
    // Utility
    collectLabelsFromMask: collectLabelsFromMask,
  });
  global.OmniColormap = api;

})(typeof window !== 'undefined' ? window : globalThis);
