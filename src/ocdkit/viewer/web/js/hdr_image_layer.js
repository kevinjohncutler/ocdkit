/* hdr_image_layer.js — the viewer's image colormap layer.
 *
 * Backed by the dispatched `ColormapImage.createColormapRenderer` — WebGPU (HDR,
 * lifted, adaptive to the display headroom) when available, else WebGL2 (SDR,
 * same colormap clamped). ONE path: the layer renders the colormapped image on a
 * canvas UNDER the main #canvas; the WebGL #canvas keeps drawing labels/masks/
 * overlays transparently on top (its image is suppressed while the layer is on).
 *
 * app.js drives it through window.OcdHdr (all no-ops if unsupported):
 *   OcdHdr.setImage(imageData, w, h)   // RGBA ImageData, on load
 *   OcdHdr.setColormap(name)           // viewer colormap name
 *   OcdHdr.setRange(vmin, vmax)        // contrast window, normalized 0..1
 *   OcdHdr.setGamma(g)                 // display gamma
 *   OcdHdr.setActive(bool)            // real colormap selected
 *   OcdHdr.draw(mat3col9)             // each frame: image→clip matrix
 */
(function () {
  'use strict';
  const CI = window.ColormapImage, HC = window.HdrColormap, HH = window.HdrHeadroom;
  const supported = !!(CI && HC);   // WebGL2 SDR is ~universal; WebGPU adds HDR

  const api = { active: false, backend: null, hdr: false,
    supported: function () { return supported; }, isActive: function () { return this.active; } };
  window.OcdHdr = api;
  if (!supported) {
    api.setImage = api.setColormap = api.setRange = api.setGamma = api.draw = function () {};
    api.setActive = function () {};
    return;
  }

  let canvas = null, renderer = null, headroom = null, ready = false;
  const pending = { img: null, cmap: 'viridis', range: null, gamma: null, matrix: null };
  // originalImageData is already top-row-first (same as the displayed image), and
  // the viewer's matrix maps image-px→clip upright — so NO extra flip (an earlier
  // FLIP_Y guess mirrored the image vs the native render).
  const FLIP_Y = false;

  function ensureCanvas() {
    if (canvas) return canvas;
    const viewer = document.getElementById('viewer');
    canvas = document.createElement('canvas');
    canvas.id = 'hdrCanvas';
    canvas.style.cssText = 'position:absolute; inset:0; width:100%; height:100%; z-index:0; display:none; pointer-events:none;';
    viewer.insertBefore(canvas, viewer.firstChild);   // below #canvas (labels on top)
    return canvas;
  }

  (async function init() {
    ensureCanvas();
    headroom = HH ? new HH() : null;
    const force = (typeof window !== 'undefined' && window.__ocdForceWebgl) || false;
    renderer = await CI.createColormapRenderer(canvas, { hdr: true, headroom: headroom, forceWebgl: force });
    api.backend = renderer.backend; api.hdr = !!renderer.hdr;
    ready = true;
    if (pending.img) api.setImage(pending.img.data, pending.img.w, pending.img.h);
    api.setColormap(pending.cmap);
    if (pending.range) api.setRange(pending.range[0], pending.range[1]);
    if (pending.gamma != null) api.setGamma(pending.gamma);
    if (pending.matrix) api.draw(pending.matrix);
  })().catch(function (e) { console.warn('[OcdHdr] init failed:', e); });

  function intensityFrom(imageData, w, h) {
    const d = imageData.data || imageData;
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y += 1) {
      const srcRow = FLIP_Y ? (h - 1 - y) : y;
      for (let x = 0; x < w; x += 1) {
        const i = (srcRow * w + x) * 4;
        out[y * w + x] = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
      }
    }
    return out;
  }

  api.setImage = function (imageData, w, h) {
    if (!ready) { pending.img = { data: imageData, w: w, h: h }; return; }
    renderer.setImage(intensityFrom(imageData, w, h), w, h);
  };
  api.setColormap = function (name) {
    if (!ready) { pending.cmap = name; return; }
    renderer.setColormap(name);
  };
  api.setRange = function (vmin, vmax) {
    if (!ready) { pending.range = [vmin, vmax]; return; }
    renderer.setRange(vmin, vmax);
  };
  api.setGamma = function (g) {
    if (!ready) { pending.gamma = g; return; }
    renderer.setGamma(g);
  };
  // Force the image SDR (unlifted) vs HDR — lets the central HDR toggle drive
  // the image too. No-op on the WebGL2 backend (already SDR).
  api.setHdr = function (on) { this.hdr = !!on; if (ready && renderer.setHdr) renderer.setHdr(!!on); };
  api.setGain = function (g) { this._gain = g; if (ready && renderer.setGain) renderer.setGain(g); };
  api.setActive = function (on) {
    this.active = !!on;
    if (canvas) canvas.style.display = on ? 'block' : 'none';
    // #canvas has its own opaque background (#111) that would hide the overlay
    // beneath it; clear it while active (the #viewer #111 backdrop remains). The
    // WebGL content itself goes transparent via u_baseAlpha (drawWebglFrame).
    const mc = document.getElementById('canvas');
    if (mc) mc.style.background = on ? 'transparent' : '';
    if (on && ready) renderer.requestRedraw();
  };
  api.draw = function (mat3col9) {
    const m = Array.prototype.slice.call(mat3col9, 0, 9);   // copy: caller reuses the buffer
    if (!ready) { pending.matrix = m; return; }
    renderer.setTransform(m);
  };
  api._dbg = function () {
    if (!renderer) return { ready: false };
    return { backend: renderer.backend, w: renderer._w, h: renderer._h,
      cw: renderer._cw, ch: renderer._ch,
      clientW: canvas && canvas.clientWidth, clientH: canvas && canvas.clientHeight,
      display: canvas && canvas.style.display, vmin: renderer._vmin, vmax: renderer._vmax,
      gamma: renderer._gamma, matrix: renderer._matrix };
  };
})();
