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
  let gainSliderEl = null, gainNumEl = null;     // HDR gain slider + number field
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
    // grayscale uses the 2D HDR layer only while HDR is on: re-check the gate
    if (typeof window.__viewerSyncHdrImageLayer === 'function') {
      try { window.__viewerSyncHdrImageLayer(); } catch (e) { /* app not ready */ }
    }
    // Drive the 3D volume too (same lift as the 2D image layer) when it's live.
    try {
      const vg = window.__volumeMode && window.__volumeMode.gpu && window.__volumeMode.gpu();
      if (vg && vg.setHdr) { vg.setGain(api.enabled ? api.gain : 1); vg.setHdr(api.enabled); }
    } catch (e) { /* volume not in 3D mode */ }
    const btn = document.getElementById('hdrToggleBtn');
    if (btn) { btn.setAttribute('aria-pressed', api.enabled ? 'true' : 'false'); btn.classList.toggle('is-on', api.enabled); }
    // (the range input is detached once the slider component registers it, so
    // keep references instead of looking it up by id)
    if (gainSliderEl) {
      gainSliderEl.value = String(api.gain);
      if (window.ViewerUI && ViewerUI.refreshSlider) ViewerUI.refreshSlider('hdrGainSlider');
    }
    if (gainNumEl) { gainNumEl.value = api.gain.toFixed(2); gainNumEl.disabled = !api.enabled; }
    const gr = document.getElementById('hdrGainRow');
    if (gr) { gr.style.opacity = api.enabled ? '' : '.45'; gr.style.pointerEvents = api.enabled ? '' : 'none'; }
  }

  // Persist the user's HDR choice (enabled + gain) so it survives a refresh.
  // v2: gain is now a multiple of the display's headroom (1 = exactly the
  // headroom), so a gain saved under the old meaning is not reused.
  const HDR_STORE = 'ocdkit-hdr-v2';
  function _persist() {
    try { localStorage.setItem(HDR_STORE, JSON.stringify({ enabled: api.enabled, gain: api.gain })); } catch (e) {}
  }
  function _restore() {
    try { const s = JSON.parse(localStorage.getItem(HDR_STORE) || 'null'); return (s && typeof s.enabled === 'boolean') ? s : null; }
    catch (e) { return null; }
  }

  api.setEnabled = function (on) { api.enabled = !!on && api.available; _persist(); apply(); };
  api.setGain = function (g) { api.gain = Math.max(0.25, Math.min(4, g)); _persist(); apply(); };
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
      '#hdrHeadroomVal { font-size: 10px; opacity: .75; letter-spacing: .02em; }\n';
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
    // Same markup as the other panel sliders (filled track + number field).
    const gainRow = document.createElement('div');
    gainRow.id = 'hdrGainRow'; gainRow.className = 'control slider-inline';
    gainRow.title = 'Multiple of the display headroom: 1 = brightest color exactly at the display limit, above 1 clips';
    gainRow.innerHTML =
      '<span class="control-heading control-heading--lower">gain</span>' +
      '<div class="slider-row">' +
      '<div class="slider" id="hdrGainSliderRoot" data-slider-type="single">' +
      '<input type="range" id="hdrGainSlider" min="0.25" max="4" step="0.05" /></div>' +
      '<div class="number-field" data-number-id="hdrGainInput">' +
      '<input type="number" id="hdrGainInput" min="0.25" max="4" step="0.05" /></div></div>';
    gainSliderEl = gainRow.querySelector('#hdrGainSlider');
    gainNumEl = gainRow.querySelector('#hdrGainInput');
    gainSliderEl.value = String(api.gain); gainNumEl.value = api.gain.toFixed(2);
    gainSliderEl.addEventListener('input', function () { api.setGain(parseFloat(gainSliderEl.value)); });
    gainNumEl.addEventListener('change', function () { api.setGain(parseFloat(gainNumEl.value) || 1); });
    if (window.ViewerUI && ViewerUI.attachNumberInputStepper) {
      ViewerUI.attachNumberInputStepper(gainNumEl, function (d) { api.setGain(api.gain + d); });
    }
    const hr = document.createElement('div'); hr.id = 'hdrHeadroomVal';
    row.appendChild(btn); row.appendChild(gainRow); row.appendChild(hr); panel.appendChild(row);
    const sroot = gainRow.querySelector('#hdrGainSliderRoot');
    if (window.ViewerUI && ViewerUI.registerSlider) {
      sroot.dataset.sliderId = 'hdrGainSlider';
      ViewerUI.registerSlider(sroot);
    }
  }

  // ── Live display headroom ──────────────────────────────────────────────
  // Browsers hide it, but when the viewer server runs on this same Mac it can
  // read it from macOS (GET /api/display_headroom). Publishing it as
  // window.__edrHeadroom lets every HdrHeadroom consumer (2D image, colormap
  // preview, 3D volume) follow it live; the desktop app injects the same value.
  const LOCAL = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
  function showHeadroom() {
    const el = document.getElementById('hdrHeadroomVal');
    if (!el) return;
    const v = window.__edrHeadroom;
    el.textContent = (typeof v === 'number' && v > 0)
      ? 'display headroom ' + v.toFixed(1) + '× (measured)'
      : 'display headroom unknown (assuming 4×)';
  }
  function pollHeadroom() {
    if (!LOCAL) { showHeadroom(); return; }
    fetch('/api/display_headroom').then(function (r) { return r.json(); }).then(function (d) {
      if (d && d.available) {
        window.__edrHeadroom = d.headroom;     // the potential: never the 'current' value, which
                                               // reads 1.0 until HDR is already on screen
        showHeadroom();
        setTimeout(pollHeadroom, 1000);      // follows brightness changes
      } else {
        showHeadroom();                      // no EDR display here: keep the fallback
      }
    }).catch(function () { showHeadroom(); });
  }

  function start() {
    injectStyle(); injectControls(); pollHeadroom();
    let tries = 0;
    (function poll() {
      api.available = available();
      root.classList.toggle('hdr-available', api.available);
      if (api.available) {
        const saved = _restore();
        if (saved) {                       // honour the user's persisted choice
          api.gain = Math.max(0.25, Math.min(4, saved.gain || 1));
          api.enabled = saved.enabled;
          apply();
        } else {
          api.setEnabled(true);            // first run → default ON when available
        }
        return;
      }
      if (tries++ < 20) setTimeout(poll, 250);
    })();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
  // The preview re-syncs on colormap change because app.js's
  // updateImageCmapPanelUI calls OcdHdrUI.refresh() at its end.
})();
