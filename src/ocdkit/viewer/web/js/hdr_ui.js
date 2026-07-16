/* hdr_ui.js — centralized HDR for the viewer chrome.
 *
 * Reality check: WKWebView accepts CSS `color(srgb-linear …)` with >1 components
 * but CLAMPS them to SDR white — so CSS cannot render HDR. The only real HDR
 * surface is a canvas (same as the image layer). So:
 *   • the colormap PREVIEW is a small WebGPU canvas (glows), driven by the gain;
 *   • the IMAGE layer is driven by the same gain (OcdHdr.setGain);
 *   • accent colors keep a centralized `--hdr-gain` / srgb-linear override — it
 *     is SDR-identical at gain 1 and will light up automatically if/when an
 *     engine renders extended CSS color (today it stays SDR).
 *
 * One knob: the gain slider + the HDR toggle. window.OcdHdrUI exposes
 * { available, enabled, gain, setEnabled, setGain, refresh }.
 */
(function () {
  'use strict';
  const root = document.documentElement;
  const HC = window.HdrColormap, CI = window.ColormapImage, HH = window.HdrHeadroom;
  const api = { available: false, enabled: true, gain: 1.0 };
  window.OcdHdrUI = api;

  function available() {
    const hd = !!(window.matchMedia && matchMedia('(dynamic-range: high)').matches);
    return hd && !!(typeof navigator !== 'undefined' && navigator.gpu) && !!(CI && HC);
  }

  // sRGB component → linear-light (for the srgb-linear accent override).
  function toLin(c) { c = Math.max(0, Math.min(1, c)); return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function refreshAccentLinear() {
    const v = getComputedStyle(root).getPropertyValue('--accent-color').trim();
    let r, g, b;
    const hex = /^#?([0-9a-fA-F]{6})$/.exec(v);
    if (hex) { const n = parseInt(hex[1], 16); r = (n >> 16 & 255) / 255; g = (n >> 8 & 255) / 255; b = (n & 255) / 255; }
    else { const m = /rgba?\(([^)]+)\)/.exec(v); if (!m) return; const p = m[1].split(',').map(function (s) { return parseFloat(s); }); r = p[0] / 255; g = p[1] / 255; b = p[2] / 255; }
    root.style.setProperty('--accent-lr', toLin(r).toFixed(4));
    root.style.setProperty('--accent-lg', toLin(g).toFixed(4));
    root.style.setProperty('--accent-lb', toLin(b).toFixed(4));
  }

  // ── HDR colormap preview — a canvas placed INSIDE the cmap dropdown's toggle,
  // over its SDR ::before gradient (z-index -1) and under the chevron (z-index
  // 1), so the dropdown background itself becomes the HDR colormap. A canvas
  // keeps HDR even inside a positioned element (unlike a gain-map image, which
  // Safari forces to SDR under position:relative/absolute). CSS color() clamps,
  // so a canvas is the only real HDR surface. ──
  let pCanvas = null, pR = null, pHeadroom = null;
  function dropdownToggle() {
    const sel = document.getElementById('imageCmapSelect');
    const wrap = sel && sel.closest('.dropdown--gradient-preview');
    return wrap ? wrap.querySelector('.dropdown-toggle') : null;
  }
  function ensurePreview() {
    if (!CI) return;
    const toggle = dropdownToggle();
    if (!toggle) return;
    if (!pCanvas) {
      pCanvas = document.createElement('canvas');
      pCanvas.id = 'hdrCmapPreview';
      pCanvas.style.cssText = 'position:absolute; inset:0; z-index:0; pointer-events:none; border-radius:inherit; clip-path: inset(var(--control-inset) round var(--control-inset-radius)); display:none;';
      pHeadroom = HH ? new HH() : null;
      CI.createColormapRenderer(pCanvas, { hdr: true, headroom: pHeadroom }).then(function (r) {
        pR = r;
        const W = 256, H = 4, ramp = new Float32Array(W * H);
        for (let y = 0; y < H; y += 1) for (let x = 0; x < W; x += 1) ramp[y * W + x] = x / (W - 1);
        r.setImage(ramp, W, H); r.setRange(0, 1);
        updatePreview();
      });
    }
    if (pCanvas.parentElement !== toggle) toggle.appendChild(pCanvas);   // (re)attach after a dropdown re-render
  }
  function cmapName() { const s = document.getElementById('imageCmapSelect'); const n = (s && s.value) || 'viridis'; return (n === 'gray' || n === 'gray-clip') ? null : n; }
  function updatePreview() {
    ensurePreview();
    const n = cmapName();
    // Show the HDR canvas only while HDR is on; otherwise the toggle's SDR
    // ::before gradient shows through.
    if (pCanvas) pCanvas.style.display = (n && api.available && api.enabled) ? 'block' : 'none';
    if (!pR || !n) return;
    pR.setColormap(n);
    if (pR.setHdr) pR.setHdr(api.enabled);
    if (pR.setGain) pR.setGain(api.enabled ? api.gain : 1);
    pR.requestRedraw();
  }

  function apply() {
    root.style.setProperty('--hdr-gain', api.enabled ? String(api.gain) : '1');
    root.classList.toggle('hdr-ui', api.enabled);
    refreshAccentLinear();
    updatePreview();
    if (window.OcdHdr) {
      if (OcdHdr.setHdr) OcdHdr.setHdr(api.enabled);
      if (OcdHdr.setGain) OcdHdr.setGain(api.enabled ? api.gain : 1);
    }
    // Drive the 3D volume too (same lift as the 2D image layer) when it's live.
    try {
      const vg = window.__volumeMode && window.__volumeMode.gpu && window.__volumeMode.gpu();
      if (vg && vg.setHdr) { vg.setGain(api.enabled ? api.gain : 1); vg.setHdr(api.enabled); }
    } catch (e) { /* volume not in 3D mode */ }
    const btn = document.getElementById('hdrToggleBtn');
    if (btn) { btn.setAttribute('aria-pressed', api.enabled ? 'true' : 'false'); btn.classList.toggle('is-on', api.enabled); }
    const sl = document.getElementById('hdrGainSlider');
    if (sl) { sl.disabled = !api.enabled; sl.value = String(api.gain); }
    const out = document.getElementById('hdrGainVal');
    if (out) out.textContent = api.gain.toFixed(2) + '×';
  }

  api.setEnabled = function (on) { api.enabled = !!on && api.available; apply(); };
  api.setGain = function (g) { api.gain = Math.max(0.25, Math.min(4, g)); apply(); };
  api.refresh = function () { refreshAccentLinear(); updatePreview(); };

  function injectStyle() {
    const css =
      ':root { --hdr-gain: 1; }\n' +
      // The preview canvas sits in the dropdown toggle at z-index 0; lift the
      // chevron and the hidden label above it. The chevron keeps its base
      // position:absolute (do NOT force position:relative — that broke its
      // centring); z-index applies to it since it is already positioned.
      '.dropdown--gradient-preview .dropdown-label { position: relative; z-index: 2; }\n' +
      '.dropdown--gradient-preview .dropdown-toggle-chevron { z-index: 2; }\n' +
      ':root.hdr-ui { --accent-color: color(srgb-linear ' +
      'calc(var(--accent-lr, 1) * var(--hdr-gain)) ' +
      'calc(var(--accent-lg, 1) * var(--hdr-gain)) ' +
      'calc(var(--accent-lb, 1) * var(--hdr-gain))); }\n' +
      '#hdrToggleRow { display: none; flex-direction: column; gap: 6px; margin-top: 8px; }\n' +
      ':root.hdr-available #hdrToggleRow { display: flex; }\n' +
      '#hdrToggleBtn { width: 100%; padding: 4px 10px; border-radius: 999px; cursor: pointer;\n' +
      '  border: 1px solid var(--control-border, #444); background: var(--control-surface, #1a1a1a);\n' +
      '  color: var(--panel-text-color, #ccc); font: inherit; font-size: 11px; letter-spacing: .04em; }\n' +
      '#hdrToggleBtn.is-on { background: var(--accent-color); color: var(--accent-ink, #161616); border-color: transparent; }\n' +
      '#hdrGainRow { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--panel-text-color, #aaa); }\n' +
      '#hdrGainRow label { letter-spacing: .04em; }\n' +
      '#hdrGainSlider { flex: 1; accent-color: var(--accent-color); }\n' +
      '#hdrGainSlider:disabled { opacity: .4; }\n' +
      '#hdrGainVal { min-width: 38px; text-align: right; font-variant-numeric: tabular-nums; }\n';
    const s = document.createElement('style'); s.id = 'hdrUiStyle'; s.textContent = css;
    document.head.appendChild(s);
  }

  function injectControls() {
    const panel = document.getElementById('imageCmapPanel');
    if (!panel) return;
    ensurePreview();
    const row = document.createElement('div'); row.id = 'hdrToggleRow'; row.className = 'control';
    const btn = document.createElement('button');
    btn.id = 'hdrToggleBtn'; btn.type = 'button'; btn.textContent = 'HDR'; btn.setAttribute('aria-pressed', 'false');
    btn.addEventListener('click', function () { api.setEnabled(!api.enabled); });
    const gainRow = document.createElement('div'); gainRow.id = 'hdrGainRow';
    const lab = document.createElement('label'); lab.textContent = 'gain'; lab.setAttribute('for', 'hdrGainSlider');
    const slider = document.createElement('input');
    slider.id = 'hdrGainSlider'; slider.type = 'range'; slider.min = '0.25'; slider.max = '4'; slider.step = '0.05'; slider.value = String(api.gain);
    slider.addEventListener('input', function () { api.setGain(parseFloat(slider.value)); });
    const val = document.createElement('span'); val.id = 'hdrGainVal'; val.textContent = api.gain.toFixed(2) + '×';
    gainRow.appendChild(lab); gainRow.appendChild(slider); gainRow.appendChild(val);
    row.appendChild(btn); row.appendChild(gainRow); panel.appendChild(row);
  }

  function start() {
    injectStyle(); injectControls();
    let tries = 0;
    (function poll() {
      api.available = available();
      root.classList.toggle('hdr-available', api.available);
      if (api.available) { api.setEnabled(true); return; }   // default ON when available
      if (tries++ < 20) setTimeout(poll, 250);
    })();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
  // The preview re-syncs on colormap change because app.js's
  // updateImageCmapPanelUI calls OcdHdrUI.refresh() at its end.
})();
