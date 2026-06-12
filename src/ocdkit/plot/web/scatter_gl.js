/* scatter_gl.js — discrete object-scatter panel for a linked tile viewer.
 *
 * Sibling of spectra_density_gl.js (same module shape + decode/render/highlight
 * interface), but for a SCATTER of discrete objects: each point is one object
 * (cell), so it can be hit-tested back to an id — that's what makes the panel
 * "linked" (hover a point -> snap the grid to that cell; hover a cell -> ring
 * its point). A density backend (SpectraGL's atomic count buffer) deliberately
 * is NOT used here: accumulation collapses points and loses per-object identity.
 *
 * Rendered at the canvas's true device resolution (getBoundingClientRect ×
 * devicePixelRatio) on a 2D canvas. Each point is coloured by an optional scalar
 * value through a 256-entry LUT, so the panel follows the viewer's colormap
 * picker exactly like the tiles and the density panel.
 *
 * Data contract — `<canvas data-scatter="1">` carries (base64 little-endian):
 *   data-x      Float32 (N)        object x values (data coords)
 *   data-y      Float32 (N)        object y values (data coords)
 *   data-cellids Int32 (N)         per-object id (for linking)
 *   data-c      Float32 (N) | -    optional per-object scalar -> LUT colour
 *   data-lut    Uint8 (256*4 RGBA) colormap LUT (swap to follow the picker)
 *   data-xlo/xhi/ylo/yhi  floats   axis ranges (default: data min/max + 5% pad)
 *   data-clo/chi          floats   c-value range for the LUT (default 0..1)
 *   data-point-size       float    point radius in CSS px (default 2.5)
 *   data-color            "r,g,b"  uniform fallback colour when no data-c
 *   data-celllabels       JSON     optional per-object labels (tooltip)
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ScatterGL = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function b64bytes(s) {
    var bin = atob(s), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  function f32(attr) { return attr ? new Float32Array(b64bytes(attr).buffer) : null; }
  function num(cv, name, dflt) { var v = parseFloat(cv.getAttribute(name)); return isNaN(v) ? dflt : v; }

  function decodeAttrs(cv) {
    var x = f32(cv.getAttribute('data-x')) || new Float32Array(0);
    var y = f32(cv.getAttribute('data-y')) || new Float32Array(0);
    var cattr = cv.getAttribute('data-c');
    var col = cv.getAttribute('data-color');
    // default ranges = data extent + 5% pad (per axis)
    function ext(a) {
      var lo = Infinity, hi = -Infinity;
      for (var i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
      if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
      var pad = (hi - lo) * 0.05; return [lo - pad, hi + pad];
    }
    var xe = ext(x), ye = ext(y);
    return {
      x: x, y: y,
      cellIds: cv.getAttribute('data-cellids')
        ? new Int32Array(b64bytes(cv.getAttribute('data-cellids')).buffer) : null,
      c: cattr ? f32(cattr) : null,
      lut: cv.getAttribute('data-lut') ? b64bytes(cv.getAttribute('data-lut')) : null,
      xLo: num(cv, 'data-xlo', xe[0]), xHi: num(cv, 'data-xhi', xe[1]),
      yLo: num(cv, 'data-ylo', ye[0]), yHi: num(cv, 'data-yhi', ye[1]),
      cLo: num(cv, 'data-clo', 0.0), cHi: num(cv, 'data-chi', 1.0),
      pointSize: num(cv, 'data-point-size', 2.5),
      color: col ? col.split(',').map(Number) : [180, 180, 180],
      cellLabels: JSON.parse(cv.getAttribute('data-celllabels') || '[]'),
    };
  }

  // map a (data x, data y) to device pixels in the W×H canvas (y up → row 0 top)
  function projector(cfg, W, H) {
    var xs = W / ((cfg.xHi - cfg.xLo) || 1), ys = H / ((cfg.yHi - cfg.yLo) || 1);
    return {
      px: function (vx) { return (vx - cfg.xLo) * xs; },
      py: function (vy) { return (cfg.yHi - vy) * ys; },
    };
  }

  function colorAt(cfg, i) {
    if (cfg.c && cfg.lut) {
      var t = (cfg.c[i] - cfg.cLo) / ((cfg.cHi - cfg.cLo) || 1);
      var li = Math.min(255, Math.max(0, (t * 255) | 0)) * 4;
      // include the LUT alpha so density→opacity ramps work (e.g. transparent_cmap-
      // style cyan/red where low density is transparent, high density opaque).
      return 'rgba(' + cfg.lut[li] + ',' + cfg.lut[li + 1] + ',' + cfg.lut[li + 2] + ',' + (cfg.lut[li + 3] / 255).toFixed(3) + ')';
    }
    var c = cfg.color; return 'rgb(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ')';
  }

  // Render cfg into the canvas at device resolution. Sync (2D canvas). Stashes a
  // CPU point list (device px + id) on cv.__sgState for hit-testing.
  function render(cv, cfg, cssW, cssH) {
    var dpr = self.devicePixelRatio || 1;
    cssW = cssW || cv.clientWidth || 256;
    cssH = cssH || cv.clientHeight || 256;
    var W = Math.max(1, Math.round(cssW * dpr)), H = Math.max(1, Math.round(cssH * dpr));
    if (cv.width !== W) cv.width = W;
    if (cv.height !== H) cv.height = H;
    var ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    var pr = projector(cfg, W, H), r = Math.max(1, cfg.pointSize * dpr);
    var ids = cfg.cellIds, N = cfg.x.length, pts = new Array(N);
    for (var i = 0; i < N; i++) {
      var px = pr.px(cfg.x[i]), py = pr.py(cfg.y[i]);
      pts[i] = { px: px, py: py, id: ids ? ids[i] : i };
      ctx.fillStyle = colorAt(cfg, i);
      ctx.beginPath(); ctx.arc(px, py, r, 0, 6.283185307, false); ctx.fill();
    }
    cv.__sgState = { cfg: cfg, W: W, H: H, pts: pts, r: r };
    return true;
  }

  // cursor (device px) → point index, or -1 if none within ~r+4px
  function nearestPoint(st, mx, my) {
    var pts = st.pts, thr = st.r + 4 * (self.devicePixelRatio || 1), thr2 = thr * thr;
    var best = -1, bestD = Infinity;
    for (var i = 0; i < pts.length; i++) {
      var dx = pts[i].px - mx, dy = pts[i].py - my, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return bestD <= thr2 ? best : -1;
  }

  function _mark(overlayCv, st, idx) {
    var W = st.W, H = st.H;
    if (overlayCv.width !== W) overlayCv.width = W;
    if (overlayCv.height !== H) overlayCv.height = H;
    var ctx = overlayCv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    if (idx < 0) return -1;
    var p = st.pts[idx];
    // highlight = the point ITSELF turns white (filled at its own footprint plus
    // a hairline so it fully covers the coloured dot) — not a ring around it.
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(p.px, p.py, st.r + (self.devicePixelRatio || 1) * 0.5, 0, 6.283185307, false); ctx.fill();
    return idx;
  }

  // hover: turn the nearest point white on the overlay; returns the point INDEX
  // (caller maps to a cell id via cfg.cellIds[index]). The ``color`` arg is part
  // of the shared LinkedPanel signature but unused here (the mark is always white).
  function highlight(scatterCv, overlayCv, mx, my, color) {
    var st = scatterCv.__sgState; if (!st) return -1;
    return _mark(overlayCv, st, nearestPoint(st, mx, my));
  }

  // reverse link: whiten the point belonging to a known cell id (hover-a-cell ->
  // light its scatter point). Returns the point index, or -1.
  function highlightById(scatterCv, overlayCv, id, color) {
    var st = scatterCv.__sgState; if (!st) return -1;
    var pts = st.pts, idx = -1;
    for (var i = 0; i < pts.length; i++) { if (pts[i].id === id) { idx = i; break; } }
    return _mark(overlayCv, st, idx);
  }

  function clearHighlight(overlayCv) {
    if (overlayCv && overlayCv.width) overlayCv.getContext('2d').clearRect(0, 0, overlayCv.width, overlayCv.height);
  }

  return { decodeAttrs: decodeAttrs, render: render, nearestPoint: nearestPoint,
           highlight: highlight, highlightById: highlightById, clearHighlight: clearHighlight };
}));
