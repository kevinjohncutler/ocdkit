/*
 * HdrHeadroom — live display HDR (EDR) headroom provider.
 *
 * The browser deliberately withholds the numeric headroom (fingerprinting), so
 * this resolves it from, in priority order:
 *   1. window.__edrHeadroom — a value injected by a NATIVE host (e.g. pywebview's
 *      WKWebView reading NSScreen.maximumExtendedDynamicRangeColorComponentValue
 *      and pushing it in via evaluate_js). This is the real, live value.
 *   2. screen.colorInfo {maximumLuminance, referenceWhiteLuminance} — the W3C
 *      ColorWeb-CG proposal (not yet shipped); headroom = max / reference white.
 *      Feature-detected so it self-activates when it lands.
 *   3. a fixed fallback multiple (no source available — e.g. a plain browser).
 *
 * The value is a multiple of SDR reference white (1.0 = SDR, e.g. 4.0 = 4×). It
 * changes with screen brightness (EDR headroom shrinks as brightness rises), so
 * this polls and notifies subscribers on change. Drive a colormap/render peak to
 * `value` and the content tracks the display limit without clipping.
 *
 *   const hr = new HdrHeadroom();              // auto-detects + polls
 *   hr.onChange(v => rerenderWith(v));
 *   hr.value;  // current multiple of SDR white
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.HdrHeadroom = mod;
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Returns { value, source } or null if no real source is available.
  function detect() {
    const w = (typeof window !== 'undefined') ? window : {};
    const s = w.screen || {};
    const bridged = w.__edrHeadroom;
    if (typeof bridged === 'number' && bridged > 0) {
      return { value: bridged, source: '__edrHeadroom (native bridge)' };
    }
    const ci = s.colorInfo;
    if (ci && ci.maximumLuminance > 0 && ci.referenceWhiteLuminance > 0) {
      return { value: ci.maximumLuminance / ci.referenceWhiteLuminance, source: 'screen.colorInfo' };
    }
    // NOTE: currentEDRHeadroom / highDynamicRangeHeadroom etc. are NATIVE Apple
    // (UIScreen/NSScreen) properties — NOT exposed to web JS. Probed only in case
    // a browser ever aliases them; normally these never resolve in a browser.
    const names = ['highDynamicRangeHeadroom', 'dynamicRangeHeadroom', 'currentEDRHeadroom'];
    for (let i = 0; i < names.length; i += 1) {
      const v = s[names[i]];
      if (typeof v === 'number' && v > 0) return { value: v, source: 'screen.' + names[i] };
    }
    return null;
  }

  class HdrHeadroom {
    constructor(opts) {
      opts = opts || {};
      this.fallback = opts.fallback || 4.0;     // used when no real source
      this.pollMs = opts.pollMs || 500;
      this.epsilon = opts.epsilon || 0.02;      // ignore sub-threshold jitter
      this.value = this.fallback;               // current multiple of SDR white
      this.source = null;                       // label of the resolving source
      this.hasReal = false;                     // true when a real source resolved
      this._subs = [];
      this._timer = null;
      this.refresh();
      if (opts.autostart !== false) this.start();
    }

    // Re-read the source; returns true if value/availability changed.
    refresh() {
      const d = detect();
      const prevVal = this.value, prevReal = this.hasReal;
      this.hasReal = !!d;
      if (d) { this.value = d.value; this.source = d.source; }
      else { this.source = null; this.value = this.fallback; }
      const changed = Math.abs(this.value - prevVal) > this.epsilon || this.hasReal !== prevReal;
      if (changed) this._emit();
      return changed;
    }

    start() { if (!this._timer && typeof setInterval !== 'undefined') this._timer = setInterval(() => this.refresh(), this.pollMs); }
    stop() { if (this._timer) { clearInterval(this._timer); this._timer = null; } }

    onChange(cb) { this._subs.push(cb); return () => { this._subs = this._subs.filter((f) => f !== cb); }; }
    _emit() { for (let i = 0; i < this._subs.length; i += 1) { try { this._subs[i](this.value, this); } catch (e) { /* ignore */ } } }
  }

  HdrHeadroom.detect = detect;
  return HdrHeadroom;
}));
