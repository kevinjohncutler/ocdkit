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
  const api = { available: false, enabled: true, gain: 1.0, transparent: false };
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
    const pill = document.getElementById('imageCmapIconPill');   // the colormap icon, when the panel has one
    if (pill) return pill;
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
      const disc = toggle.id === 'imageCmapIconPill';
      CI.createColormapRenderer(pCanvas, { hdr: true, headroom: pHeadroom }).then(function (r) {
        pR = r;
        // the dropdown: a left-to-right ramp; the colormap icon: a disc swept
        // clockwise from the top, like its CSS conic gradient
        const W = disc ? 96 : 256, H = disc ? 96 : 4, ramp = new Float32Array(W * H);
        for (let y = 0; y < H; y += 1) for (let x = 0; x < W; x += 1) {
          ramp[y * W + x] = disc
            ? ((Math.atan2(x + 0.5 - W / 2, H / 2 - (y + 0.5)) / (2 * Math.PI)) + 1) % 1
            : x / (W - 1);
        }
        r.setImage(ramp, W, H); r.setRange(0, 1);
        updatePreview();
      });
    }
    if (pCanvas.parentElement !== toggle) {
      toggle.appendChild(pCanvas);   // (re)attach after a dropdown re-render
      if (toggle.id === 'imageCmapIconPill') pCanvas.style.clipPath = 'none';   // (the icon pill clips it)
    }
  }
  function cmapName() {
    const s = document.getElementById('imageCmapSelect'); const n = (s && s.value) || 'viridis';
    if (document.getElementById('imageCmapIconPill')) return n;          // the icon shows every colormap, grayscale too
    const wrap = s && s.closest('.dropdown--gradient-preview');
    if (!wrap || !wrap.classList.contains('has-gradient')) return null;   // (the dropdown is plain now: the icon shows the colormap)
    return (n === 'gray' || n === 'gray-clip') ? null : n;
  }
  function updatePreview() {
    ensurePreview();
    const n = cmapName();
    // Show the HDR canvas only while HDR is on; otherwise the toggle's SDR
    // ::before gradient shows through.
    if (pCanvas) pCanvas.style.display = (n && api.available && api.enabled) ? 'block' : 'none';
    if (!pR || !n) return;
    pR.setColormap(n);
    if (pR.setTransparent) pR.setTransparent(api.transparent);
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
    if (window.OcdHdr && OcdHdr.setTransparent) OcdHdr.setTransparent(api.transparent);
    try {
      const vt = window.__volumeMode && window.__volumeMode.gpu && window.__volumeMode.gpu();
      if (vt && vt.setTransparent) vt.setTransparent(api.transparent);
    } catch (e) { /* volume not in 3D mode */ }
    const tt = document.getElementById('cmapTransparentToggle');
    if (tt) tt.checked = api.transparent;
    // grayscale uses the 2D HDR layer only while HDR is on (or transparency): re-check the gate
    if (typeof window.__viewerSyncHdrImageLayer === 'function') {
      try { window.__viewerSyncHdrImageLayer(); } catch (e) { /* app not ready */ }
    }
    // Drive the 3D volume too (same lift as the 2D image layer) when it's live.
    try {
      const vg = window.__volumeMode && window.__volumeMode.gpu && window.__volumeMode.gpu();
      if (vg && vg.setHdr) { vg.setGain(api.enabled ? api.gain : 1); vg.setHdr(api.enabled); }
    } catch (e) { /* volume not in 3D mode */ }
    const sw = document.getElementById('hdrToggle');
    if (sw) sw.checked = api.enabled;
    // gain + measured headroom only matter while HDR is on
    // gain + headroom: shown while HDR is on and the gain slider is switched on
    // in the image panel's right-click menu
    for (const id of ['hdrGainRow', 'hdrHeadroomVal']) {
      const el = document.getElementById(id);
      if (el) el.style.display = (api.enabled && api.gainVisible) ? '' : 'none';
    }
    if (window.ViewerUI && ViewerUI.normalizeLabelWidths) ViewerUI.normalizeLabelWidths();
    // re-measure the gain slider once visible (measured while hidden it is 0 wide)
    if (api.enabled && window.ViewerUI && ViewerUI.refreshSlider) {
      requestAnimationFrame(function () { ViewerUI.refreshSlider('hdrGainSlider'); });
    }
    // (the range input is detached once the slider component registers it, so
    // keep references instead of looking it up by id)
    if (gainSliderEl) {
      gainSliderEl.value = String(api.gain);
      if (window.ViewerUI && ViewerUI.refreshSlider) ViewerUI.refreshSlider('hdrGainSlider');
    }
    if (gainNumEl) { gainNumEl.value = api.gain.toFixed(2); gainNumEl.disabled = !api.enabled; }

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
  // Transparent low end, for 2D and 3D; independent of HDR, so saved on its own.
  const TRANSP_STORE = 'ocdkit-cmap-transparent';
  try { api.transparent = localStorage.getItem(TRANSP_STORE) === '1'; } catch (e) {}
  api.setTransparent = function (on) {
    api.transparent = !!on;
    try { localStorage.setItem(TRANSP_STORE, api.transparent ? '1' : '0'); } catch (e) {}
    apply();
    if (window.__refreshCmapIcon) window.__refreshCmapIcon();   // the colormap icon shows the transparency too
  };
  api.refresh = function () { refreshAccentLinear(); updatePreview(); };
  // The gain slider is hidden unless switched on (image panel, right-click).
  const GAIN_VIS_STORE = 'ocdkit-hdr-gain-visible';
  try { api.gainVisible = localStorage.getItem(GAIN_VIS_STORE) === '1'; } catch (e) { api.gainVisible = false; }
  api.setGainVisible = function (on) {
    api.gainVisible = !!on;
    try { localStorage.setItem(GAIN_VIS_STORE, api.gainVisible ? '1' : '0'); } catch (e) {}
    apply();
  };
  (function attachGainMenu() {
    const panel = document.getElementById('intensityPanel');
    if (!panel) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attachGainMenu); return; }
    panel.addEventListener('contextmenu', function (e) {
      if (!api.available || !(window.ViewerUI && ViewerUI.openPanelMenu)) return;
      ViewerUI.openPanelMenu(e, [{
        label: 'HDR gain slider',
        tip: 'Show the HDR gain slider (a multiple of the display headroom) while HDR is on',
        checked: api.gainVisible,
        onChange: function (on) { api.setGainVisible(on); },
      }]);
    });
  })();

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
      '#hdrToggleRow { display: none; flex-direction: column; gap: 6px; margin-top: 0; }\n' +
      ':root.hdr-available #hdrToggleRow { display: flex; }\n' +
      '#hdrToggleChip { display: none; }\n' +
      ':root.hdr-available #hdrToggleChip { display: flex; }\n' +

      '#hdrHeadroomVal { font-size: 10px; opacity: .75; letter-spacing: .02em; }\n';
    const s = document.createElement('style'); s.id = 'hdrUiStyle'; s.textContent = css;
    document.head.appendChild(s);
  }

  function injectControls() {
    const panel = document.getElementById('imageCmapPanel');
    if (!panel) return;
    ensurePreview();
    const row = document.createElement('div'); row.id = 'hdrToggleRow'; row.className = 'control';
    // HDR on/off: a switch, like the "transparent" row above it
    const btn = document.createElement('div');
    btn.className = 'control label-style-row';
    btn.innerHTML = '<span class="control-heading control-heading--lower">HDR</span>' +
      '<label class="toggle toggle-left" title="Render the image in HDR, up to the display\'s measured headroom">' +
      '<input type="checkbox" id="hdrToggle" /><span class="toggle-switch"></span></label>';
    btn.querySelector('#hdrToggle').addEventListener('change', function (e) { api.setEnabled(e.target.checked); });
    // Same markup as the other panel sliders (filled track + number field).
    const gainRow = document.createElement('div');
    gainRow.id = 'hdrGainRow'; gainRow.className = 'control slider-inline';
    gainRow.title = 'Multiple of the display headroom: 1 = brightest color exactly at the display limit, above 1 clips';
    gainRow.innerHTML =
      '<span class="control-heading control-heading--lower control-heading--icon" title="HDR gain"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8"/></svg></span>' +
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
    // HDR is an icon toggle beside alpha and invert when the panel has a slot for it
    const chip = document.getElementById('hdrToggleChip');
    if (chip) {
      const input = btn.querySelector('#hdrToggle');
      chip.title = 'HDR: render the image in HDR, up to the display\'s measured headroom';
      chip.append(input);
      const t = document.createElement('span'); t.className = 'icon-toggle-text'; t.textContent = 'HDR';
      chip.append(t);
    } else {
      row.appendChild(btn);
    }
    row.appendChild(gainRow); row.appendChild(hr); panel.appendChild(row);
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
        setTimeout(pollHeadroom, 2000);      // follows brightness changes
      } else {
        showHeadroom();                      // no EDR display here: keep the fallback
      }
    }).catch(function () { showHeadroom(); });
  }

  function start() {
    injectStyle(); injectControls(); pollHeadroom();
    const tt = document.getElementById('cmapTransparentToggle');
    if (tt) {
      tt.checked = api.transparent;
      tt.addEventListener('change', function () { api.setTransparent(tt.checked); });
    }
    if (api.transparent) apply();          // restore the saved choice even without HDR
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
