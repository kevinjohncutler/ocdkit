/* spectra_density_element.js — <hipr-spectra-density> custom element.
 *
 * A Shadow-DOM wrapper around SpectraGL (spectra_density_gl.js). The point of
 * the element is host-CSS immunity: the density + hover-overlay canvases live
 * inside an open shadow root as plain absolutely-positioned HTML <canvas>es —
 * NOT inside an SVG <foreignObject> and NOT subject to a host rule like
 * JupyterLab's `.jp-RenderedHTMLCommon svg { height:auto }`. This is the same
 * rendering path the working RGB overlay tile already uses, so it renders in
 * every browser that the RGB tile does.
 *
 * Transport-agnostic config:
 *   - element.data = {…SpectraGL cfg…}   (set before connect, preferred), or
 *   - data-* attributes on the element itself (decoded via SpectraGL.decodeAttrs),
 *     including the per-norm variants data-ylines-{self,bitdepth,global} and
 *     data-norm-mode used by the y-axis normalization toggle.
 *
 * Public API:
 *   .draw()              re-render at current box size (debounced via rAF)
 *   .setNorm(mode)       'self' | 'bitdepth' | 'global' → swap ylines + redraw
 *   .activeLineLabel     last hovered cell label (or null)
 * Events:
 *   'hipr-spectra-hover' {detail:{line, label}}  on hover-line change
 *
 * Requires SpectraGL on the global (window.SpectraGL) or as element.spectraGL.
 */
(function () {
  'use strict';
  if (typeof customElements === 'undefined') return;
  if (customElements.get('hipr-spectra-density')) return;

  function b64bytes(s) {
    var bin = atob(s), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }

  class HiprSpectraDensity extends HTMLElement {
    connectedCallback() {
      if (this._mounted) { this.draw(); return; }
      this._mounted = true;
      var root = this.attachShadow({ mode: 'open' });
      root.innerHTML =
        '<style>' +
        ':host{display:block;position:relative;overflow:hidden;line-height:0}' +
        'canvas{position:absolute;left:0;top:0;width:100%;height:100%;display:block}' +
        '.ov{cursor:crosshair}' +
        '</style>' +
        '<canvas class="den"></canvas><canvas class="ov"></canvas>';
      this._den = root.querySelector('.den');
      this._ov = root.querySelector('.ov');
      this._SG = this.spectraGL || window.SpectraGL;
      this._cfg = this._resolveCfg();
      this._wireHover();
      // size may be 0 if the element hasn't laid out yet; observe it.
      if (typeof ResizeObserver !== 'undefined') {
        this._ro = new ResizeObserver(() => this.draw());
        this._ro.observe(this);
      }
      this.draw();
    }

    disconnectedCallback() { if (this._ro) this._ro.disconnect(); }

    _resolveCfg() {
      if (this.data) return this.data;
      if (!this._SG) return null;
      // SpectraGL.decodeAttrs reads data-* from any element via getAttribute.
      var cfg = this._SG.decodeAttrs(this);
      this._norm = this.getAttribute('data-norm-mode') || 'self';
      return cfg;
    }

    setNorm(mode) {
      var b64 = this.getAttribute('data-ylines-' + mode);
      if (!b64 || !this._cfg) return false;
      this._cfg.yLines = new Float32Array(b64bytes(b64).buffer);
      this._norm = mode;
      this.setAttribute('data-norm-mode', mode);
      this.draw();
      return true;
    }

    draw() {
      if (!this._SG || !this._cfg) return;
      if (this._raf) return;
      this._raf = requestAnimationFrame(() => {
        this._raf = 0;
        var r = this.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) return;
        this._SG.render(this._den, this._cfg, r.width, r.height);
      });
    }

    _wireHover() {
      var self = this;
      this._ov.addEventListener('pointermove', function (e) {
        if (!self._den.__sgState) return;
        var r = self._ov.getBoundingClientRect();
        var dpr = window.devicePixelRatio || 1;
        var mx = (e.clientX - r.left) * dpr, my = (e.clientY - r.top) * dpr;
        // High-contrast, out-of-palette highlight so the line reads clearly over
        // the magma ensemble (default red camouflages where the density is bright).
        var hcolor = self.getAttribute('data-highlight-color') || 'rgba(90,230,255,0.98)';
        var line = self._SG.highlight(self._den, self._ov, mx, my, hcolor);
        var label = null;
        var cfg = self._den.__sgState.cfg;
        if (line >= 0 && cfg.cellLabels) label = cfg.cellLabels[line];
        if (line !== self._lastLine) {
          self._lastLine = line; self.activeLineLabel = label;
          self.dispatchEvent(new CustomEvent('hipr-spectra-hover',
            { detail: { line: line, label: label }, bubbles: true, composed: true }));
        }
      });
      this._ov.addEventListener('pointerleave', function () {
        self._SG.clearHighlight(self._ov);
        self._lastLine = -1; self.activeLineLabel = null;
      });
    }
  }

  customElements.define('hipr-spectra-density', HiprSpectraDensity);
  if (typeof window !== 'undefined') window.HiprSpectraDensity = HiprSpectraDensity;
})();
