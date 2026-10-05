/**
 * ViewerUI — Pure DOM helpers, widget registration, and UI utilities.
 *
 * Exports to window.ViewerUI following the existing IIFE + classic script pattern.
 * No ES modules — PyWebView breaks with type="module".
 */
(function (global) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------
  var FILENAME_TRUNCATE = 10;

  var isIOSDevice = typeof navigator !== 'undefined' && (
    /iPad|iPhone|iPod/.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
  );
  var isSafariWebKit = typeof navigator !== 'undefined'
    && /AppleWebKit/.test(navigator.userAgent)
    && !/Chrome|CriOS|Edg|Firefox|FxiOS/.test(navigator.userAgent);

  // ---------------------------------------------------------------------------
  // Math / value utilities
  // ---------------------------------------------------------------------------

  function clamp(value, min, max) {
    if (!Number.isFinite(value)) {
      return min;
    }
    return Math.min(max, Math.max(min, value));
  }

  function normalizeAngle(angle) {
    if (!Number.isFinite(angle)) {
      return 0;
    }
    return Math.atan2(Math.sin(angle), Math.cos(angle));
  }

  // ---------------------------------------------------------------------------
  // Color utilities
  // ---------------------------------------------------------------------------

  function rgbToCss(rgb) {
    return 'rgb(' + (rgb[0] | 0) + ', ' + (rgb[1] | 0) + ', ' + (rgb[2] | 0) + ')';
  }

  function parseCssColor(value) {
    if (!value || typeof value !== 'string') {
      return null;
    }
    var trimmed = value.trim();
    if (trimmed.startsWith('#')) {
      var hex = trimmed.slice(1);
      if (hex.length === 3) {
        return [
          parseInt(hex[0] + hex[0], 16),
          parseInt(hex[1] + hex[1], 16),
          parseInt(hex[2] + hex[2], 16),
        ];
      }
      if (hex.length === 6) {
        return [
          parseInt(hex.slice(0, 2), 16),
          parseInt(hex.slice(2, 4), 16),
          parseInt(hex.slice(4, 6), 16),
        ];
      }
    }
    var match = trimmed.match(/rgba?\(([^)]+)\)/i);
    if (match) {
      var parts = match[1].split(',').map(function (part) { return Number(part.trim()); });
      if (parts.length >= 3 && parts.every(function (val) { return Number.isFinite(val); })) {
        return parts.slice(0, 3).map(function (val) { return Math.max(0, Math.min(255, val)); });
      }
    }
    return null;
  }

  function readableTextColor(rgb) {
    var r = rgb[0], g = rgb[1], b = rgb[2];
    var luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    return luminance > 0.62 ? 'rgba(20, 20, 20, 0.86)' : '#f6f6f6';
  }

  function lightenRgb(rgb, amount) {
    if (amount === undefined) { amount = 0.25; }
    var clampChannel = function (value) { return Math.max(0, Math.min(255, value)); };
    var mix = function (a, b) { return Math.round(a + (b - a) * amount); };
    return [
      clampChannel(mix(rgb[0], 255)),
      clampChannel(mix(rgb[1], 255)),
      clampChannel(mix(rgb[2], 255)),
    ];
  }

  // ---------------------------------------------------------------------------
  // Cursor builders
  // ---------------------------------------------------------------------------

  function svgCursorUrl(svg) {
    return 'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '") 16 16, auto';
  }

  function buildDotCursorCss(size, rgba) {
    if (size === undefined) { size = 16; }
    if (rgba === undefined) { rgba = [128, 128, 128, 0.5]; }
    var r = rgba[0], g = rgba[1], b = rgba[2], a = rgba[3];
    var radius = Math.max(2, Math.floor(size / 4));
    var cx = Math.floor(size / 2);
    var cy = Math.floor(size / 2);
    var svg = '<?xml version="1.0" encoding="UTF-8"?>'
      + '<svg xmlns="http://www.w3.org/2000/svg" width="' + size + '" height="' + size
      + '" viewBox="0 0 ' + size + ' ' + size + '">'
      + '<circle cx="' + cx + '" cy="' + cy + '" r="' + radius
      + '" fill="rgb(' + r + ',' + g + ',' + b + ')" fill-opacity="' + a + '" />'
      + '</svg>';
    return 'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '") ' + cx + ' ' + cy + ', auto';
  }

  // ---------------------------------------------------------------------------
  // String / format utilities
  // ---------------------------------------------------------------------------

  function truncateFilename(name, keep) {
    if (keep === undefined) { keep = FILENAME_TRUNCATE; }
    if (!name || typeof name !== 'string') {
      return '';
    }
    var lastDot = name.lastIndexOf('.');
    var hasExt = lastDot > 0 && lastDot < name.length - 1;
    var base = hasExt ? name.slice(0, lastDot) : name;
    var ext = hasExt ? name.slice(lastDot) : '';
    if (base.length <= keep * 2) {
      return base + ext;
    }
    return base.slice(0, keep) + '...' + base.slice(-keep) + ext;
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) {
      return '--';
    }
    var units = ['B', 'KB', 'MB', 'GB', 'TB'];
    var value = Math.max(0, bytes);
    var idx = 0;
    while (value >= 1024 && idx < units.length - 1) {
      value /= 1024;
      idx += 1;
    }
    return value.toFixed(idx >= 2 ? 1 : 0) + ' ' + units[idx];
  }

  // ---------------------------------------------------------------------------
  // DOM interaction utilities
  // ---------------------------------------------------------------------------

  function flashButton(btn) {
    if (!btn) return;
    btn.classList.remove('shortcut-flash');
    void btn.offsetWidth; // force reflow to restart animation
    btn.classList.add('shortcut-flash');
    btn.addEventListener('animationend', function () { btn.classList.remove('shortcut-flash'); }, { once: true });
  }

  function setPanelCollapsed(panel, collapsed) {
    if (!panel) {
      return;
    }
    if (collapsed) {
      panel.classList.add('panel-collapsed');
    } else {
      panel.classList.remove('panel-collapsed');
    }
  }

  function showConfirmDialog(message, opts) {
    var confirmText = (opts && opts.confirmText) || 'OK';
    var cancelText = (opts && opts.cancelText) || 'Cancel';
    return new Promise(function (resolve) {
      var backdrop = document.createElement('div');
      backdrop.className = 'omni-confirm-backdrop';
      var dialog = document.createElement('div');
      dialog.className = 'omni-confirm';
      var msg = document.createElement('div');
      msg.className = 'omni-confirm-message';
      msg.textContent = message;
      var actions = document.createElement('div');
      actions.className = 'omni-confirm-actions';
      var cancelBtn = document.createElement('button');
      cancelBtn.type = 'button';
      cancelBtn.className = 'omni-confirm-button';
      cancelBtn.textContent = cancelText;
      var okBtn = document.createElement('button');
      okBtn.type = 'button';
      okBtn.className = 'omni-confirm-button omni-confirm-primary';
      okBtn.textContent = confirmText;
      actions.appendChild(cancelBtn);
      actions.appendChild(okBtn);
      dialog.appendChild(msg);
      dialog.appendChild(actions);
      backdrop.appendChild(dialog);
      document.body.appendChild(backdrop);
      var cleanup = function (result) {
        backdrop.remove();
        document.removeEventListener('keydown', onKey);
        resolve(result);
      };
      var onKey = function (evt) {
        if (evt.key === 'Enter') {
          evt.preventDefault();
          cleanup(true);
        } else if (evt.key === 'Escape') {
          evt.preventDefault();
          cleanup(false);
        }
      };
      document.addEventListener('keydown', onKey);
      cancelBtn.addEventListener('click', function () { cleanup(false); });
      okBtn.addEventListener('click', function () { cleanup(true); });
      backdrop.addEventListener('click', function (evt) {
        if (evt.target === backdrop) cleanup(false);
      });
      setTimeout(function () { okBtn.focus(); }, 0);
    });
  }

  function showPromptDialog(message, opts) {
    var confirmText = (opts && opts.confirmText) || 'OK';
    var cancelText = (opts && opts.cancelText) || 'Cancel';
    var defaultValue = (opts && opts.defaultValue) || '';
    var placeholder = (opts && opts.placeholder) || '';
    return new Promise(function (resolve) {
      var backdrop = document.createElement('div');
      backdrop.className = 'omni-confirm-backdrop';
      var dialog = document.createElement('div');
      dialog.className = 'omni-confirm';
      var msg = document.createElement('div');
      msg.className = 'omni-confirm-message';
      msg.textContent = message;
      var input = document.createElement('input');
      input.type = 'text';
      input.className = 'omni-prompt-input';
      input.value = defaultValue;
      input.placeholder = placeholder;
      var actions = document.createElement('div');
      actions.className = 'omni-confirm-actions';
      var cancelBtn = document.createElement('button');
      cancelBtn.type = 'button';
      cancelBtn.className = 'omni-confirm-button';
      cancelBtn.textContent = cancelText;
      var okBtn = document.createElement('button');
      okBtn.type = 'button';
      okBtn.className = 'omni-confirm-button omni-confirm-primary';
      okBtn.textContent = confirmText;
      actions.appendChild(cancelBtn);
      actions.appendChild(okBtn);
      dialog.appendChild(msg);
      dialog.appendChild(input);
      dialog.appendChild(actions);
      backdrop.appendChild(dialog);
      document.body.appendChild(backdrop);
      var cleanup = function (result) {
        backdrop.remove();
        document.removeEventListener('keydown', onKey);
        resolve(result);
      };
      var onKey = function (evt) {
        if (evt.key === 'Enter') {
          evt.preventDefault();
          cleanup(input.value.trim() || null);
        } else if (evt.key === 'Escape') {
          evt.preventDefault();
          cleanup(null);
        }
      };
      document.addEventListener('keydown', onKey);
      cancelBtn.addEventListener('click', function () { cleanup(null); });
      okBtn.addEventListener('click', function () { cleanup(input.value.trim() || null); });
      backdrop.addEventListener('click', function (evt) {
        if (evt.target === backdrop) cleanup(null);
      });
      setTimeout(function () { input.focus(); input.select(); }, 0);
    });
  }

  function suppressDoubleTapZoom(element) {
    if (!element || typeof element.addEventListener !== 'function') {
      return;
    }
    var lastTouchTime = 0;
    var lastTouchX = 0;
    var lastTouchY = 0;
    var reset = function () {
      lastTouchTime = 0;
      lastTouchX = 0;
      lastTouchY = 0;
    };
    element.addEventListener('touchend', function (evt) {
      if (!evt || (evt.touches && evt.touches.length > 0)) {
        return;
      }
      var changed = evt.changedTouches && evt.changedTouches[0];
      if (!changed) {
        return;
      }
      if (evt.target && typeof evt.target.closest === 'function') {
        var interactive = evt.target.closest('input, textarea, select, [contenteditable="true"]');
        if (interactive) {
          reset();
          return;
        }
      }
      var now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      var dt = now - lastTouchTime;
      var dx = changed.clientX - lastTouchX;
      var dy = changed.clientY - lastTouchY;
      var distanceSq = (dx * dx) + (dy * dy);
      if (Number.isFinite(dt) && dt > 0 && dt < 350 && distanceSq < 400) {
        evt.preventDefault();
        evt.stopPropagation();
        reset();
        return;
      }
      lastTouchTime = now;
      lastTouchX = changed.clientX;
      lastTouchY = changed.clientY;
    }, { passive: false });
  }

  function attachNumberInputStepper(input, onAdjust) {
    if (!input || typeof onAdjust !== 'function') {
      return;
    }
    input.addEventListener('keydown', function (evt) {
      if (evt.key !== 'ArrowUp' && evt.key !== 'ArrowDown') {
        return;
      }
      evt.preventDefault();
      var base = Number(input.step || '1');
      var step = Number.isFinite(base) && base > 0 ? base : 1;
      var factor = evt.shiftKey ? 5 : 1;
      var direction = evt.key === 'ArrowUp' ? 1 : -1;
      onAdjust(step * factor * direction);
    });
  }

  // ---------------------------------------------------------------------------
  // Slider helpers
  // ---------------------------------------------------------------------------

  function valueToPercent(input) {
    var min = Number(input.min || 0);
    var max = Number(input.max || 1);
    var span = max - min;
    if (!Number.isFinite(span) || span === 0) {
      return 0;
    }
    var value = Number(input.value || min);
    return clamp((value - min) / span, 0, 1);
  }

  function percentToValue(percent, input) {
    var min = Number(input.min || 0);
    var max = Number(input.max || 1);
    var span = max - min;
    var raw = min + clamp(percent, 0, 1) * span;
    var step = Number(input.step || '1');
    if (!Number.isFinite(step) || step <= 0) {
      return clamp(raw, min, max);
    }
    var snapped = Math.round((raw - min) / step) * step + min;
    var precision = (step.toString().split('.')[1] || '').length;
    var factor = Math.pow(10, precision);
    return clamp(Math.round(snapped * factor) / factor, min, max);
  }

  function pointerPercent(evt, container, inset) {
    var rect = container.getBoundingClientRect();
    var pad = inset > 0 ? inset : 0;
    var usable = rect.width - 2 * pad;
    if (usable <= 0) {
      return 0;
    }
    var ratio = (evt.clientX - rect.left - pad) / usable;
    return clamp(ratio, 0, 1);
  }

  function updateNativeRangeFill(input) {
    if (!input || !(isIOSDevice && isSafariWebKit)) {
      return;
    }
    var percent = valueToPercent(input);
    var root = input.closest('.slider');
    if (!root) {
      return;
    }
    var trackRadius = parseFloat(getComputedStyle(root).getPropertyValue('--slider-track-radius'))
      || Math.round(root.clientHeight / 2);
    var usable = Math.max(0, root.clientWidth - trackRadius * 2);
    var fillPx = Math.round(usable * percent);
    root.style.setProperty('--slider-fill-px', fillPx + 'px');
    var knob = root.querySelector('.slider-native-knob');
    if (knob) {
      var knobSize = parseFloat(getComputedStyle(root).getPropertyValue('--slider-knob-size')) || 16;
      knob.style.left = (trackRadius + fillPx - knobSize / 2) + 'px';
    }
  }

  // ---------------------------------------------------------------------------
  // Value-pill slider art: concentric pills
  // ---------------------------------------------------------------------------

  // SVG path of a pill (a stadium: flat top and bottom, semicircular ends, the
  // same shape as every other pill in the panel) whose flat edges run from xa to
  // xb, centered on cy, with end radius r. A handle (r = b), the fill around it
  // (r = b + 3) and the track (r = b + 4) share their end centers, so the gaps
  // between them are exactly constant, straight and round alike.
  function pillPath(xa, xb, cy, r) {
    var f = function (v) { return v.toFixed(2); };
    return 'M' + f(xa) + ' ' + f(cy - r) + 'L' + f(xb) + ' ' + f(cy - r) +
      'A' + f(r) + ' ' + f(r) + ' 0 0 1 ' + f(xb) + ' ' + f(cy + r) +
      'L' + f(xa) + ' ' + f(cy + r) +
      'A' + f(r) + ' ' + f(r) + ' 0 0 1 ' + f(xa) + ' ' + f(cy - r) + 'Z';
  }

  // ---------------------------------------------------------------------------
  // Slider registry
  // ---------------------------------------------------------------------------

  var sliderRegistry = new Map();

  function registerSlider(root) {
    var id = root.dataset.sliderId || root.dataset.slider || root.id;
    if (!id) {
      return;
    }
    var type = (root.dataset.sliderType || 'single').toLowerCase();
    var inputs = Array.from(root.querySelectorAll('input[type="range"]'));
    if (!inputs.length) {
      return;
    }
    if (type === 'dual' && inputs.length < 2) {
      console.warn('slider ' + id + ' configured as dual but only one range input found');
      return;
    }
    // Value pills: the slider's number fields become its thumbs, each a pill
    // showing its value; drag it to slide, click it to type. A dual slider holds
    // its two (class slider-thumb-input) inside its root; a single slider uses the
    // number field beside it in its .slider-row, which is then hidden.
    var pillInputs = Array.from(root.querySelectorAll('input.slider-thumb-input'));
    var sideField = null;
    if (type !== 'dual' && !pillInputs.length && !root.classList.contains('slider--gradient-track')) {
      var row = root.parentElement;
      var nf = row && row.classList.contains('slider-row') ? row.querySelector(':scope > .number-field') : null;
      var ni = nf ? nf.querySelector('input[type="number"]') : null;
      if (ni) { sideField = nf; pillInputs = [ni]; }
    }
    if (isIOSDevice && isSafariWebKit) {
      root.classList.add('slider-native');
      if (!root.querySelector('.slider-native-track')) {
        var track = document.createElement('div');
        track.className = 'slider-native-track';
        var fill = document.createElement('div');
        fill.className = 'slider-native-fill';
        track.appendChild(fill);
        var knob = document.createElement('div');
        knob.className = 'slider-native-knob';
        root.appendChild(track);
        root.appendChild(knob);
      }
      inputs.forEach(function (input) {
        updateNativeRangeFill(input);
        input.addEventListener('input', function () { updateNativeRangeFill(input); });
        input.addEventListener('change', function () { updateNativeRangeFill(input); });
      });
      if (pillInputs.length) {          // native sliders: the value fields go below instead
        var below = document.createElement('div');
        below.className = 'clip-values';
        pillInputs.forEach(function (p) { below.appendChild(p); });
        root.insertAdjacentElement('afterend', below);
      }
      return;
    }

    root.innerHTML = '';
    var track = document.createElement('div');
    track.className = 'slider-track';
    root.appendChild(track);
    var thumbs = inputs.map(function () {
      var thumb = document.createElement('div');
      thumb.className = 'slider-thumb';
      root.appendChild(thumb);
      return thumb;
    });

    var trackRadiusOf = function (e) {
      return parseFloat(getComputedStyle(e.track).getPropertyValue('--slider-track-radius'))
        || Math.round(e.track.clientHeight / 2);
    };

    var pills = pillInputs.length === thumbs.length && (type === 'dual' ? thumbs.length === 2 : thumbs.length === 1);
    var art = null;
    if (pills) {
      root.classList.add('slider--pills');
      if (sideField) { sideField.style.display = 'none'; }        // its input moves into the thumb
      thumbs.forEach(function (t, i) {
        pillInputs[i].classList.add('slider-thumb-input');
        t.classList.add('slider-thumb--pill');
        t.appendChild(pillInputs[i]);
      });
      // track, fill and handles drawn as one SVG, each an exact offset of the next
      var NS = 'http://www.w3.org/2000/svg';
      art = { svg: document.createElementNS(NS, 'svg') };
      art.svg.setAttribute('class', 'slider-pill-art');
      art.svg.setAttribute('aria-hidden', 'true');
      ['track', 'outline', 'fill'].concat(thumbs.map(function (_, i) { return 'h' + i; })).forEach(function (k) {
        var p = document.createElementNS(NS, 'path');
        p.setAttribute('class', 'pill-art-' + (k[0] === 'h' ? 'handle' : k));
        art.svg.appendChild(p);
        art[k] = p;
      });
      root.insertBefore(art.svg, thumbs[0]);
      pillInputs.forEach(function (inp, i) {
        inp.addEventListener('focus', function () { art['h' + i].classList.add('is-editing'); });
        inp.addEventListener('blur', function () { art['h' + i].classList.remove('is-editing'); });
      });
    }

    var entry = {
      id: id,
      type: type === 'dual' ? 'dual' : 'single',
      pills: pills,
      root: root,
      inputs: inputs,
      track: track,
      thumbs: thumbs,
      activePointer: null,
      activeThumb: null,
    };

    if (!entry.root.hasAttribute('tabindex')) {
      entry.root.tabIndex = 0;
    }

    // Pill geometry: pill width w, gap g between a pill and the track edge. The low
    // pill's left edge and the high pill's left edge minus w both travel over the
    // same usable length, so the pills meet (never overlap) when lo == hi, and the
    // fill spans both pills like one longer pill.
    // One width per slider, wide enough for its widest value (its min and max at
    // its step's decimals, or any wider value it has shown), so a pill never
    // changes width, and jumps, while it is dragged. At least --slider-pill-width.
    var measureCtx = null, pillW = 0;
    var textWidth = function (inp, text) {
      var cs = getComputedStyle(inp);
      if (!measureCtx) { measureCtx = document.createElement('canvas').getContext('2d'); }
      measureCtx.font = cs.fontWeight + ' ' + cs.fontSize + ' ' + cs.fontFamily;
      return measureCtx.measureText(text).width;
    };
    var pillWidth = function () {
      if (!pillW) {
        pillInputs.forEach(function (inp) {
          var st = String(inp.step || '1'), dec = st.indexOf('.') >= 0 ? st.split('.')[1].length : 0;
          [inp.min, inp.max].forEach(function (v) {
            if (v !== '' && Number.isFinite(Number(v))) { pillW = Math.max(pillW, textWidth(inp, Number(v).toFixed(dec))); }
          });
        });
      }
      if (entry.activePointer === null) {          // (frozen while dragging)
        pillInputs.forEach(function (inp) { pillW = Math.max(pillW, textWidth(inp, String(inp.value))); });
      }
      var minW = parseFloat(getComputedStyle(entry.root).getPropertyValue('--slider-pill-width')) || 32;
      return Math.max(minW, Math.ceil(pillW + 12));
    };
    var pillGeom = function () {
      var w0 = pillWidth();
      var ws = entry.thumbs.map(function (t) { t.style.width = w0 + 'px'; return w0; });
      var g = Math.max(0, (entry.track.clientHeight - entry.thumbs[0].offsetHeight) / 2);
      var sum = ws.reduce(function (a, b) { return a + b; }, 0);
      return { ws: ws, w: ws[0], g: g, usable: Math.max(0, entry.track.clientWidth - sum - 2 * g) };
    };

    var apply = function () {
      if (entry.pills) {
        var dual = entry.type === 'dual';
        if (dual && Number(entry.inputs[0].value) > Number(entry.inputs[1].value)) {
          var t0 = entry.inputs[0].value; entry.inputs[0].value = entry.inputs[1].value; entry.inputs[1].value = t0;
        }
        var pg = pillGeom();
        // dual: the low pill sits left of its value and the high pill right of it;
        // single: one pill, the fill runs from the track's start to it
        var lefts = entry.inputs.map(function (inp, i) {
          return pg.g + (i === 1 ? pg.ws[0] : 0) + Math.round(pg.usable * valueToPercent(inp));
        });
        entry.thumbs.forEach(function (t, i) { t.style.left = lefts[i] + 'px'; });
        var W = entry.track.clientWidth, Hh = entry.track.clientHeight;
        var hb = entry.thumbs[0].offsetHeight / 2;
        if (art && W > 0 && hb > 0) {               // (hidden sliders measure 0: drawn once shown)
          var cy = Hh / 2, gap = pg.g, last = lefts.length - 1;
          var inset = parseFloat(getComputedStyle(entry.root).getPropertyValue('--control-inset')) || 1;   // fill inside the track, as on plain sliders
          art.svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Hh);
          art.svg.setAttribute('width', W); art.svg.setAttribute('height', Hh);
          art.track.setAttribute('d', pillPath(gap + hb, W - gap - hb, cy, hb + gap));
          art.outline.setAttribute('d', pillPath(gap + hb, W - gap - hb, cy, hb + gap - 0.5));
          art.fill.setAttribute('d', pillPath((dual ? lefts[0] : gap) + hb, lefts[last] + pg.ws[last] - hb, cy, hb + gap - inset));
          lefts.forEach(function (L, i) { art['h' + i].setAttribute('d', pillPath(L + hb, L + pg.ws[i] - hb, cy, hb)); });
        }
        return;
      }
      if (entry.type === 'dual') {
        var minInput = entry.inputs[0];
        var maxInput = entry.inputs[1];
        var minValue = Number(minInput.value);
        var maxValue = Number(maxInput.value);
        if (minValue > maxValue) {
          var temp = minValue;
          minValue = maxValue;
          maxValue = temp;
          minInput.value = String(minValue);
          maxInput.value = String(maxValue);
        }
        var minPercent = valueToPercent(minInput);
        var maxPercent = valueToPercent(maxInput);
        var dualRadius = trackRadiusOf(entry);
        var dualUsable = Math.max(0, entry.track.clientWidth - dualRadius * 2);
        var x0 = Math.round(dualUsable * minPercent);
        var x1 = Math.round(dualUsable * maxPercent);
        entry.track.style.setProperty('--slider-track-radius', dualRadius + 'px');
        entry.track.style.setProperty('--slider-fill-left', x0 + 'px');
        entry.track.style.setProperty('--slider-fill-px', (x1 - x0) + 'px');
        entry.thumbs[0].style.left = (dualRadius + x0) + 'px';
        entry.thumbs[1].style.left = (dualRadius + x1) + 'px';
      } else {
        var input = entry.inputs[0];
        var percent = valueToPercent(input);
        var trackRadius = trackRadiusOf(entry);
        var usable = Math.max(0, entry.track.clientWidth - trackRadius * 2);
        var fillPx = Math.round(usable * percent);
        entry.track.style.setProperty('--slider-fill-px', fillPx + 'px');
        entry.track.style.setProperty('--slider-track-radius', trackRadius + 'px');
        entry.thumbs[0].style.left = (trackRadius + fillPx) + 'px';
      }
    };

    var stepInputValue = function (input, direction) {
      if (!input) {
        return;
      }
      var min = Number(input.min || 0);
      var max = Number(input.max || 1);
      var step = Number(input.step || '1');
      if (!Number.isFinite(step) || step <= 0) {
        step = 1;
      }
      var precision = (step.toString().split('.')[1] || '').length;
      var factor = Math.pow(10, precision);
      var current = Number(input.value || min);
      var next = current + direction * step;
      next = clamp(next, min, max);
      next = Math.round(next * factor) / factor;
      input.value = String(next);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      apply();
    };

    var setValueFromPercent = function (index, percent) {
      var input = entry.inputs[index];
      if (!input) {
        return;
      }
      var value = percentToValue(percent, input);
      if (entry.type === 'dual') {
        if (index === 0) {
          var other = Number(entry.inputs[1].value);
          if (value > other) {
            value = other;
          }
        } else {
          var other2 = Number(entry.inputs[0].value);
          if (value < other2) {
            value = other2;
          }
        }
      }
      input.value = String(value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      apply();
    };

    var pickThumb = function (percent) {
      if (entry.type !== 'dual') {
        return 0;
      }
      var distances = entry.inputs.map(function (input) { return Math.abs(percent - valueToPercent(input)); });
      var bestIndex = 0;
      var bestDistance = distances[0];
      for (var i = 1; i < distances.length; i += 1) {
        if (distances[i] < bestDistance) {
          bestDistance = distances[i];
          bestIndex = i;
        }
      }
      return bestIndex;
    };

    // pills: the pointer's x -> the dragged pill's value, keeping the grab point
    var pillPercent = function (evt, index, grab) {
      var pg = pillGeom(), rect = entry.root.getBoundingClientRect();
      var left = evt.clientX - rect.left - grab - pg.g - (index === 1 ? pg.ws[0] : 0);
      return pg.usable > 0 ? clamp(left / pg.usable, 0, 1) : 0;
    };
    var onPillDown = function (evt) {
      var hit = entry.thumbs.findIndex(function (t) { return t.contains(evt.target); });
      var editing = hit >= 0 && document.activeElement === pillInputs[hit];
      if (editing) { return; }                     // typing in a pill: leave the caret alone
      evt.preventDefault();
      var index = hit, grab;
      if (index < 0) {                             // on the track: the nearer pill, centered on the pointer
        var cx = entry.thumbs.map(function (t) { var b = t.getBoundingClientRect(); return Math.abs(evt.clientX - (b.left + b.width / 2)); });
        index = cx.length < 2 || cx[0] <= cx[1] ? 0 : 1;
        grab = entry.thumbs[index].offsetWidth / 2;
      } else {
        grab = evt.clientX - entry.thumbs[index].getBoundingClientRect().left;
      }
      entry.activePointer = evt.pointerId;
      entry.activeThumb = index;
      entry.pillGrab = grab;
      entry.pillDownX = evt.clientX;
      entry.pillMoved = hit < 0;                   // a press on a pill is a click until it moves
      entry.root.setPointerCapture(entry.activePointer);
      entry.root.dataset.active = 'true';
      if (hit < 0) { setValueFromPercent(index, pillPercent(evt, index, grab)); }
    };
    var onPillMove = function (evt) {
      if (entry.activePointer === null || evt.pointerId !== entry.activePointer) { return; }
      if (!entry.pillMoved && Math.abs(evt.clientX - entry.pillDownX) < 3) { return; }
      entry.pillMoved = true;
      setValueFromPercent(entry.activeThumb, pillPercent(evt, entry.activeThumb, entry.pillGrab));
    };
    var onPillUp = function (evt) {
      if (entry.activePointer === null || evt.pointerId !== entry.activePointer) { return; }
      try { entry.root.releasePointerCapture(entry.activePointer); } catch (_) { /* ignore */ }
      var index = entry.activeThumb, clicked = !entry.pillMoved;
      entry.activePointer = null;
      entry.activeThumb = null;
      entry.root.dataset.active = 'false';
      if (clicked && pillInputs[index]) {          // a click on a pill: type a value
        pillInputs[index].focus({ preventScroll: true });
        pillInputs[index].select();
      }
    };

    var onPointerDown = function (evt) {
      evt.preventDefault();
      var percent = pointerPercent(evt, entry.root, trackRadiusOf(entry));
      var thumbIndex = entry.type === 'dual' ? pickThumb(percent) : 0;
      entry.activePointer = evt.pointerId;
      entry.activeThumb = thumbIndex;
      entry.root.setPointerCapture(entry.activePointer);
      entry.root.dataset.active = 'true';
      if (entry.root.focus) {
        entry.root.focus({ preventScroll: true });
      }
      var targetInput = entry.inputs[thumbIndex];
      if (targetInput) {
        targetInput.focus();
      }
      setValueFromPercent(thumbIndex, percent);
    };

    var onPointerMove = function (evt) {
      if (entry.activePointer === null || evt.pointerId !== entry.activePointer) {
        return;
      }
      var percent = pointerPercent(evt, entry.root, trackRadiusOf(entry));
      setValueFromPercent(entry.activeThumb != null ? entry.activeThumb : 0, percent);
    };

    entry.root.addEventListener('keydown', function (evt) {
      if (evt.target && evt.target.classList && evt.target.classList.contains('slider-thumb-input')) {
        if (evt.key === 'Enter' || evt.key === 'Escape') { evt.target.blur(); }
        return;                                    // a pill being typed in handles its own keys
      }
      if (evt.key === 'ArrowLeft' || evt.key === 'ArrowDown') {
        evt.preventDefault();
        var index = entry.type === 'dual' ? (entry.activeThumb != null ? entry.activeThumb : 0) : 0;
        stepInputValue(entry.inputs[index], -1);
        return;
      }
      if (evt.key === 'ArrowRight' || evt.key === 'ArrowUp') {
        evt.preventDefault();
        var index2 = entry.type === 'dual' ? (entry.activeThumb != null ? entry.activeThumb : 0) : 0;
        stepInputValue(entry.inputs[index2], 1);
        return;
      }
      if (evt.key === 'Home') {
        evt.preventDefault();
        var index3 = entry.type === 'dual' ? (entry.activeThumb != null ? entry.activeThumb : 0) : 0;
        var input = entry.inputs[index3];
        if (input) {
          input.value = String(input.min || 0);
          input.dispatchEvent(new Event('input', { bubbles: true }));
          apply();
        }
        return;
      }
      if (evt.key === 'End') {
        evt.preventDefault();
        var index4 = entry.type === 'dual' ? (entry.activeThumb != null ? entry.activeThumb : 0) : 0;
        var input2 = entry.inputs[index4];
        if (input2) {
          input2.value = String(input2.max || 1);
          input2.dispatchEvent(new Event('input', { bubbles: true }));
          apply();
        }
      }
    });

    var onPointerRelease = function (evt) {
      if (entry.activePointer === null || evt.pointerId !== entry.activePointer) {
        return;
      }
      try {
        entry.root.releasePointerCapture(entry.activePointer);
      } catch (_) {
        /* ignore */
      }
      var percent = pointerPercent(evt, entry.root, trackRadiusOf(entry));
      setValueFromPercent(entry.activeThumb != null ? entry.activeThumb : 0, percent);
      entry.activePointer = null;
      entry.activeThumb = null;
      entry.root.dataset.active = 'false';
    };

    entry.root.addEventListener('pointerdown', pills ? onPillDown : onPointerDown);
    entry.root.addEventListener('pointermove', pills ? onPillMove : onPointerMove);
    entry.root.addEventListener('pointerup', pills ? onPillUp : onPointerRelease);
    entry.root.addEventListener('pointercancel', pills ? onPillUp : onPointerRelease);

    inputs.forEach(function (input) {
      input.addEventListener('focus', function () {
        entry.root.dataset.focused = 'true';
      });
      input.addEventListener('blur', function () {
        entry.root.dataset.focused = 'false';
      });
      input.addEventListener('input', apply);
      input.addEventListener('change', apply);
    });

    entry.apply = apply;
    apply();
    sliderRegistry.set(id, entry);
    // value pills are drawn to the track's size: redraw when it changes (a label
    // width change, a panel shown, a window resize)
    if (pills && typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(function () { apply(); }).observe(entry.track);
    }
  }

  // Within each panel section, every row label gets the width of the section's
  // widest visible label, so the sliders and dropdowns after them start (and end)
  // at the same x. Re-run when rows are shown or hidden.
  function normalizeLabelWidths() {
    document.querySelectorAll('.panel-section').forEach(function (sec) {
      var heads = Array.from(sec.querySelectorAll(
        '.slider-inline > .control-heading--lower, .label-style-row > .control-heading--lower'));
      heads.forEach(function (h) { h.style.minWidth = ''; });
      var vis = heads.filter(function (h) { return h.offsetParent !== null && h.getBoundingClientRect().width > 0; });
      var w = 0;
      vis.forEach(function (h) { w = Math.max(w, Math.ceil(h.getBoundingClientRect().width)); });
      if (w > 0) { heads.forEach(function (h) { h.style.minWidth = w + 'px'; }); }
    });
  }
  var normalizePending = false;
  function scheduleNormalizeLabelWidths() {
    if (normalizePending) { return; }
    normalizePending = true;
    requestAnimationFrame(function () { normalizePending = false; normalizeLabelWidths(); });
  }
  if (typeof MutationObserver !== 'undefined' && typeof document !== 'undefined') {
    var startNormalize = function () {
      scheduleNormalizeLabelWidths();
      new MutationObserver(scheduleNormalizeLabelWidths).observe(document.body,
        { subtree: true, childList: true, attributes: true, attributeFilter: ['hidden'] });
    };
    if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', startNormalize); }
    else { startNormalize(); }
  }

  function refreshSlider(id) {
    var entry = sliderRegistry.get(id);
    if (entry && typeof entry.apply === 'function') {
      entry.apply();
    }
  }

  // ---------------------------------------------------------------------------
  // Dropdown registry
  // ---------------------------------------------------------------------------

  var dropdownRegistry = new Map();
  var dropdownOpenId = null;

  function closeDropdown(entry) {
    if (!entry) {
      return;
    }
    entry.root.dataset.open = 'false';
    entry.button.setAttribute('aria-expanded', 'false');
    if (entry.menu) {
      entry.menu.setAttribute('aria-hidden', 'true');
      entry.menu.scrollTop = 0;
    }
    dropdownOpenId = null;
  }

  function openDropdown(entry) {
    if (!entry) {
      return;
    }
    if (dropdownOpenId && dropdownOpenId !== entry.id) {
      closeDropdown(dropdownRegistry.get(dropdownOpenId));
    }
    entry.root.dataset.open = 'true';
    positionDropdown(entry);
    entry.button.setAttribute('aria-expanded', 'true');
    if (entry.menuWrapper) {
      entry.menuWrapper.focus({ preventScroll: true });
    }
    if (entry.menu) {
      entry.menu.setAttribute('aria-hidden', 'false');
    }
    dropdownOpenId = entry.id;
  }

  function toggleDropdown(entry) {
    if (!entry) {
      return;
    }
    var isOpen = entry.root.dataset.open === 'true';
    if (isOpen) {
      closeDropdown(entry);
    } else {
      openDropdown(entry);
    }
  }

  function positionDropdown(entry) {
    if (!entry || !entry.menu) {
      return;
    }
    entry.menu.style.minWidth = '100%';
  }

  // Rounded equilateral triangle, apex pointing right (rotate via CSS to
  // re-aim). Shared by collapsible headings and dropdown toggles so the
  // glyph stays consistent across the UI. `currentColor` honors the host's
  // text color, so accent overrides keep working.
  function makeChevron(opts) {
    var ns = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '-1.15 -1.15 2.3 2.3');
    svg.setAttribute('width', '0.85em');
    svg.setAttribute('height', '0.85em');
    svg.setAttribute('aria-hidden', 'true');
    if (opts && opts.className) svg.classList.add(opts.className);
    svg.style.transition = 'transform 0.18s ease, color 0.12s ease';
    svg.style.transformOrigin = '50% 50%';
    svg.style.opacity = '0.85';
    svg.style.flex = '0 0 auto';
    svg.style.verticalAlign = 'middle';
    svg.style.overflow = 'visible';
    var tri = document.createElementNS(ns, 'path');
    tri.setAttribute('d', 'M 1 0 L -0.5 -0.866 L -0.5 0.866 Z');
    tri.setAttribute('fill', 'currentColor');
    tri.setAttribute('stroke', 'currentColor');
    tri.setAttribute('stroke-width', '0.28');
    tri.setAttribute('stroke-linejoin', 'round');
    tri.setAttribute('stroke-linecap', 'round');
    svg.appendChild(tri);
    return svg;
  }

  function registerDropdown(root) {
    var select = root.querySelector('select');
    if (!select) {
      return;
    }
    var id = root.dataset.dropdownId || select.id || ('dropdown-' + dropdownRegistry.size);
    root.dataset.dropdownId = id;
    root.dataset.open = root.dataset.open || 'false';

    var originalOptions = Array.from(select.options).map(function (opt) {
      return {
        value: opt.value,
        label: opt.textContent || opt.value,
        disabled: opt.disabled,
      };
    });

    select.classList.add('dropdown-input');
    root.innerHTML = '';
    root.appendChild(select);

    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'dropdown-toggle';
    button.setAttribute('aria-haspopup', 'listbox');
    button.setAttribute('aria-expanded', 'false');
    var labelSpan = document.createElement('span');
    labelSpan.className = 'dropdown-label';
    button.appendChild(labelSpan);
    button.appendChild(makeChevron({ className: 'dropdown-toggle-chevron' }));
    var menu = document.createElement('div');
    menu.className = 'dropdown-menu';
    menu.setAttribute('role', 'listbox');
    menu.setAttribute('aria-hidden', 'true');
    menu.id = id + '-menu';
    button.setAttribute('aria-controls', menu.id);
    root.appendChild(button);
    var menuWrapper = document.createElement('div');
    menuWrapper.className = 'dropdown-menu-wrap';
    menuWrapper.appendChild(menu);
    menuWrapper.tabIndex = -1;
    root.appendChild(menuWrapper);

    var entry = {
      id: id,
      root: root,
      select: select,
      button: button,
      menu: menu,
      menuWrapper: menuWrapper,
      options: originalOptions,
      loop: root.dataset.loop === 'true' ? { size: 5, mode: 'loop' } : null,
      countLabel: root.dataset.countLabel || 'items',
      confirm: root.dataset.apply === 'confirm',
      tooltipDisabled: root.dataset.tooltipDisabled === 'true',
    };

    var applySelection = function () {
      var selectedOption = select.options[select.selectedIndex];
      var displayLabel = selectedOption ? selectedOption.textContent : 'Select';
      labelSpan.textContent = displayLabel;
      if (selectedOption) {
        var fullLabel = selectedOption.dataset.fullPath || selectedOption.dataset.fullLabel || selectedOption.title || selectedOption.textContent;
        if (fullLabel) {
          if (entry.tooltipDisabled) {
            button.removeAttribute('title');
            button.removeAttribute('data-tooltip');
          } else if (entry.id === 'imageNavigator') {
            button.dataset.tooltip = fullLabel;
            button.removeAttribute('title');
          } else {
            button.removeAttribute('title');
            button.removeAttribute('data-tooltip');
          }
        }
      }
      menu.querySelectorAll('.dropdown-option').forEach(function (child) {
        var isSelected = child.dataset.value === select.value;
        child.dataset.selected = isSelected ? 'true' : 'false';
        var color = isSelected ? 'var(--accent-ink, #161616)' : 'var(--panel-text-color)';
        child.style.setProperty('color', color, 'important');
      });
    };

    var buildOption = function (opt) {
      var item = document.createElement('div');
      item.className = 'dropdown-option';
      item.dataset.value = opt.value;
      item.setAttribute('role', 'option');
      if (opt.title && !entry.tooltipDisabled && entry.id === 'imageNavigator') {
        item.dataset.tooltip = opt.title;
      }
      if (opt.disabled) {
        item.setAttribute('aria-disabled', 'true');
        item.style.opacity = '0.45';
        item.style.pointerEvents = 'none';
      }
      if (opt.deletable && typeof opt.onDelete === 'function') {
        var label = document.createElement('span');
        label.className = 'dropdown-option-label';
        label.textContent = opt.label;
        item.appendChild(label);
        var delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.className = 'dropdown-option-delete';
        delBtn.textContent = '\u00d7';
        delBtn.title = 'Remove model';
        delBtn.addEventListener('pointerdown', function (evt) {
          evt.preventDefault();
          evt.stopPropagation();
          opt.onDelete(opt.value, opt.label);
        });
        item.appendChild(delBtn);
      } else {
        item.textContent = opt.label;
      }
      item.addEventListener('pointerdown', function (evt) {
        evt.preventDefault();
        if (opt.disabled) {
          return;
        }
        select.value = opt.value;
        select.dispatchEvent(new Event('change', { bubbles: true }));
        applySelection();
        closeDropdown(entry);
      });
      return item;
    };

    // Sticky-bottom action items: any option whose value starts with `__`
    // (sentinel) is rendered as an `.dropdown-add` row at the bottom of the
    // menu, regardless of loop windowing. Used for "+ Add model…",
    // "Open image file…", "Open image folder…", etc.
    var isAction = function (opt) { return typeof opt.value === 'string' && opt.value.indexOf('__') === 0; };

    var buildMenu = function () {
      menu.innerHTML = '';
      var opts = entry.options || [];
      var loopEnabled = Boolean(entry.loop && entry.loop.mode === 'loop');
      if (loopEnabled && opts.length) {
        var loopOptions = opts.filter(function (opt) { return !isAction(opt); });
        var actionOptions = opts.filter(isAction);
        var size = entry.loop.size || 5;
        var total = loopOptions.length;
        // Don't render the same item more than once when total < window size.
        var visibleCount = Math.min(size, total);
        var half = Math.floor(visibleCount / 2);
        var currentIndex = Math.max(0, loopOptions.findIndex(function (opt) { return opt.value === select.value; }));
        for (var i = -half; i <= half; i += 1) {
          if ((i + half) >= visibleCount) break;
          var idx = total > 0 ? (currentIndex + i + total) % total : 0;
          var opt = loopOptions[idx];
          if (!opt) continue;
          menu.appendChild(buildOption(opt));
        }
        actionOptions.forEach(function (action) {
          var actionRow = buildOption(action);
          actionRow.classList.add('dropdown-add');
          menu.appendChild(actionRow);
        });
        var footer = document.createElement('div');
        footer.className = 'dropdown-footer';
        var count = document.createElement('span');
        count.className = 'dropdown-count';
        count.textContent = total ? (total + ' ' + entry.countLabel) : ('0 ' + entry.countLabel);
        var toggleBtn = document.createElement('button');
        toggleBtn.type = 'button';
        toggleBtn.className = 'dropdown-expand';
        toggleBtn.textContent = 'Show all';
        toggleBtn.addEventListener('click', function (evt) {
          evt.preventDefault();
          entry.loop.mode = 'full';
          buildMenu();
        });
        footer.appendChild(count);
        footer.appendChild(toggleBtn);
        menu.appendChild(footer);
      } else {
        var fullLoopOptions = opts.filter(function (opt) { return !isAction(opt); });
        var fullActionOptions = opts.filter(isAction);
        fullLoopOptions.forEach(function (opt) {
          menu.appendChild(buildOption(opt));
        });
        fullActionOptions.forEach(function (action) {
          var actionRow = buildOption(action);
          actionRow.classList.add('dropdown-add');
          menu.appendChild(actionRow);
        });
        if (entry.loop) {
          var footer2 = document.createElement('div');
          footer2.className = 'dropdown-footer';
          var count2 = document.createElement('span');
          count2.className = 'dropdown-count';
          count2.textContent = fullLoopOptions.length + ' ' + entry.countLabel;
          var toggleBtn2 = document.createElement('button');
          toggleBtn2.type = 'button';
          toggleBtn2.className = 'dropdown-expand';
          toggleBtn2.textContent = 'Show less';
          toggleBtn2.addEventListener('click', function (evt) {
            evt.preventDefault();
            entry.loop.mode = 'loop';
            buildMenu();
          });
          footer2.appendChild(count2);
          footer2.appendChild(toggleBtn2);
          menu.appendChild(footer2);
        }
      }
      applySelection();
    };

    var shiftSelection = function (delta) {
      if (!entry.loop || entry.loop.mode !== 'loop') return;
      var loopOptions = entry.options.filter(function (opt) { return !isAction(opt); });
      if (!loopOptions.length) return;
      var currentIndex = Math.max(0, loopOptions.findIndex(function (opt) { return opt.value === select.value; }));
      var nextIndex = (currentIndex + delta + loopOptions.length) % loopOptions.length;
      var nextValue = loopOptions[nextIndex].value;
      select.value = nextValue;
      if (!entry.confirm) {
        select.dispatchEvent(new Event('change', { bubbles: true }));
      }
      buildMenu();
    };

    button.addEventListener('click', function () {
      toggleDropdown(entry);
    });

    select.addEventListener('change', function () {
      if (entry.loop) {
        buildMenu();
      } else {
        applySelection();
      }
    });

    if (entry.loop) {
      var wheelVelocity = 0;
      var wheelAccumulator = 0;
      var wheelAnimating = false;
      var wheelLastTime = 0;

      var getStepPx = function () {
        var styles = getComputedStyle(menuWrapper);
        var raw = styles.getPropertyValue('--slider-track-height')
          || getComputedStyle(document.documentElement).getPropertyValue('--slider-track-height');
        var parsed = parseFloat(raw);
        return Number.isFinite(parsed) && parsed > 0 ? parsed : 32;
      };

      var animateWheel = function (ts) {
        if (!entry.loop || entry.loop.mode !== 'loop') {
          wheelAnimating = false;
          wheelVelocity = 0;
          wheelAccumulator = 0;
          return;
        }
        if (!wheelLastTime) {
          wheelLastTime = ts;
        }
        var dt = Math.min(48, ts - wheelLastTime);
        wheelLastTime = ts;
        var friction = Math.pow(0.9, dt / 16);
        wheelVelocity *= friction;
        wheelAccumulator += wheelVelocity * (dt / 16);

        var stepPx = getStepPx();
        while (Math.abs(wheelAccumulator) >= stepPx) {
          var direction = wheelAccumulator > 0 ? 1 : -1;
          shiftSelection(direction);
          wheelAccumulator -= direction * stepPx;
        }

        if (Math.abs(wheelVelocity) < 0.05) {
          wheelAnimating = false;
          wheelVelocity = 0;
          wheelAccumulator = 0;
          wheelLastTime = 0;
          return;
        }
        requestAnimationFrame(animateWheel);
      };

      menuWrapper.addEventListener('wheel', function (evt) {
        if (!entry.loop || entry.loop.mode !== 'loop') {
          return;
        }
        evt.preventDefault();
        wheelVelocity += evt.deltaY * 0.6;
        if (!wheelAnimating) {
          wheelAnimating = true;
          wheelLastTime = 0;
          requestAnimationFrame(animateWheel);
        }
      }, { passive: false });
    }

    menuWrapper.addEventListener('keydown', function (evt) {
      if (evt.key === 'ArrowDown' || evt.key === 'ArrowUp') {
        evt.preventDefault();
        var delta = evt.key === 'ArrowDown' ? 1 : -1;
        if (entry.loop && entry.loop.mode === 'loop') {
          shiftSelection(delta);
          return;
        }
        var opts = entry.options.filter(function (opt) { return !opt.disabled; });
        if (!opts.length) return;
        var currentIndex = Math.max(0, opts.findIndex(function (opt) { return opt.value === select.value; }));
        var nextIndex = Math.min(opts.length - 1, Math.max(0, currentIndex + delta));
        var nextValue = opts[nextIndex].value;
        select.value = nextValue;
        if (!entry.confirm) {
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
        applySelection();
        return;
      }
      if (evt.key === 'Enter') {
        evt.preventDefault();
        if (entry.confirm) {
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
        closeDropdown(entry);
        return;
      }
    });

    buildMenu();
    entry.applySelection = applySelection;
    entry.buildMenu = buildMenu;
    positionDropdown(entry);
    dropdownRegistry.set(id, entry);
  }

  function refreshDropdown(id) {
    var entry = dropdownRegistry.get(id);
    if (!entry) return;
    if (entry.loop && typeof entry.buildMenu === 'function') {
      entry.buildMenu();
      return;
    }
    if (typeof entry.applySelection === 'function') {
      entry.applySelection();
    }
  }

  function getDropdown(id) {
    return dropdownRegistry.get(id);
  }

  function getOpenDropdownId() {
    return dropdownOpenId;
  }

  function closeOpenDropdown() {
    if (!dropdownOpenId) {
      return false;
    }
    var entry = dropdownRegistry.get(dropdownOpenId);
    if (entry) {
      closeDropdown(entry);
    } else {
      dropdownOpenId = null;
    }
    return true;
  }

  function repositionOpenDropdown() {
    if (!dropdownOpenId) {
      return;
    }
    var entry = dropdownRegistry.get(dropdownOpenId);
    if (entry) {
      positionDropdown(entry);
    } else {
      dropdownOpenId = null;
    }
  }

  // Document-level click-outside handler for dropdowns
  document.addEventListener('pointerdown', function (evt) {
    if (!dropdownOpenId) {
      return;
    }
    var entry = dropdownRegistry.get(dropdownOpenId);
    if (!entry) {
      dropdownOpenId = null;
      return;
    }
    if (!entry.root.contains(evt.target)) {
      closeDropdown(entry);
    }
  });

  // ---------------------------------------------------------------------------
  // Tooltip system
  // ---------------------------------------------------------------------------

  var tooltipState = null;
  // Optional resolver consulted before the dataset.tooltip lookup. The tooltip
  // editor registers one to surface user overrides at hover-time without
  // mutating any element's data-tooltip (keeping the canonical string intact
  // even when other code paths set/refresh it).
  var tooltipResolver = null;
  function setTooltipResolver(fn) {
    tooltipResolver = (typeof fn === 'function') ? fn : null;
  }

  function initTooltips() {
    if (tooltipState) {
      return;
    }
    var tooltip = document.createElement('div');
    tooltip.className = 'omni-tooltip';
    tooltip.setAttribute('role', 'tooltip');
    var tooltipInner = document.createElement('span');
    tooltipInner.className = 'omni-tooltip-inner';
    tooltip.appendChild(tooltipInner);
    document.body.appendChild(tooltip);
    tooltipState = {
      tooltip: tooltip,
      tooltipInner: tooltipInner,
      timer: null,
      target: null,
      lastPoint: null,
    };

    var adoptTitle = function (el) {
      var title = el.getAttribute('title');
      if (title) {
        el.dataset.tooltip = title;
        el.removeAttribute('title');
      }
    };
    document.querySelectorAll('[title]').forEach(adoptTitle);
    // Code that sets a title later (a label that changes with state, a node added
    // after load) would otherwise get the browser's native tooltip on top of
    // ours: adopt those titles as they appear.
    if (typeof MutationObserver === 'function') {
      new MutationObserver(function (records) {
        records.forEach(function (r) {
          if (r.type === 'attributes') {
            if (r.target.nodeType === 1 && r.target.hasAttribute('title')) adoptTitle(r.target);
            return;
          }
          r.addedNodes.forEach(function (node) {
            if (node.nodeType !== 1) return;
            if (node.hasAttribute('title')) adoptTitle(node);
            node.querySelectorAll('[title]').forEach(adoptTitle);
          });
        });
      }).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['title'] });
    }

    var getTooltipText = function (el) {
      if (!el) return '';
      // Tooltip editor (when active) returns an override string; treat
      // anything else (null/undefined/empty) as no override.
      var override = tooltipResolver ? tooltipResolver(el) : null;
      if (typeof override === 'string' && override.length) return override;
      var existing = el.dataset.tooltip;
      if (existing) return existing;
      var title = el.getAttribute('title');
      if (title) {
        el.dataset.tooltip = title;
        el.removeAttribute('title');
        return title;
      }
      return '';
    };

    var positionTooltip = function (x, y, target) {
      var tip = tooltipState.tooltip;
      if (!tip) return;
      var padding = 12;
      var offset = 14;
      var left = x + offset;
      var top = y + offset;
      var rect = tip.getBoundingClientRect();
      var vw = window.innerWidth;
      var vh = window.innerHeight;
      if (left + rect.width + padding > vw) {
        left = Math.max(padding, x - rect.width - offset);
      }
      if (top + rect.height + padding > vh) {
        top = Math.max(padding, y - rect.height - offset);
      }
      if (Number.isFinite(left) && Number.isFinite(top)) {
        tip.style.left = left + 'px';
        tip.style.top = top + 'px';
      } else if (target) {
        var trect = target.getBoundingClientRect();
        tip.style.left = Math.min(vw - rect.width - padding, Math.max(padding, trect.left)) + 'px';
        tip.style.top = Math.min(vh - rect.height - padding, Math.max(padding, trect.bottom + offset)) + 'px';
      }
    };

    var showTooltip = function (el, point) {
      var text = getTooltipText(el);
      if (!text) return;
      var tip = tooltipState.tooltip;
      tooltipState.tooltipInner.textContent = text;
      tip.classList.add('visible');
      var coords = point || { x: 0, y: 0 };
      positionTooltip(coords.x, coords.y, el);
    };

    var hideTooltip = function () {
      if (!tooltipState) return;
      var tip = tooltipState.tooltip;
      if (tooltipState.timer) {
        clearTimeout(tooltipState.timer);
        tooltipState.timer = null;
      }
      tooltipState.target = null;
      tip.classList.remove('visible');
    };

    var scheduleTooltip = function (target, point) {
      if (!target) return;
      if (tooltipState.timer) {
        clearTimeout(tooltipState.timer);
        tooltipState.timer = null;
      }
      tooltipState.target = target;
      tooltipState.lastPoint = point;
      tooltipState.timer = setTimeout(function () {
        if (tooltipState && tooltipState.target === target) {
          showTooltip(target, tooltipState.lastPoint);
        }
      }, 200);
    };

    var refreshDynamicTooltip = function (target, point) {
      if (!target || target.dataset.tooltipDynamic !== 'true') {
        return false;
      }
      if (target.closest('[data-tooltip-disabled="true"]')) {
        return false;
      }
      var text = getTooltipText(target);
      if (!text) {
        if (tooltipState.target === target) {
          hideTooltip();
        }
        return true;
      }
      if (tooltipState.tooltip.classList.contains('visible') && tooltipState.target === target) {
        tooltipState.tooltipInner.textContent = text;
        positionTooltip(point.x, point.y, target);
        return true;
      }
      scheduleTooltip(target, point);
      return true;
    };

    document.addEventListener('pointerover', function (evt) {
      var target = evt.target.closest('[data-tooltip], [title], [data-tooltip-dynamic="true"]');
      if (!target) return;
      if (target.closest('[data-tooltip-disabled="true"]')) return;
      scheduleTooltip(target, { x: evt.clientX, y: evt.clientY });
    });

    document.addEventListener('pointermove', function (evt) {
      if (!tooltipState) return;
      var point = { x: evt.clientX, y: evt.clientY };
      tooltipState.lastPoint = point;
      var dynamicTarget = evt.target.closest('[data-tooltip-dynamic="true"]');
      if (dynamicTarget) {
        refreshDynamicTooltip(dynamicTarget, point);
      }
      if (tooltipState.tooltip.classList.contains('visible')
          && !document.documentElement.classList.contains('apple-material')) {
        positionTooltip(point.x, point.y, tooltipState.target);
      }
    });

    document.addEventListener('pointerout', function (evt) {
      if (!tooltipState) return;
      var target = evt.target.closest('[data-tooltip], [title], [data-tooltip-dynamic="true"]');
      if (!target) return;
      if (target.closest('[data-tooltip-disabled="true"]')) return;
      if (tooltipState.target === target) {
        hideTooltip();
      }
    });

    document.addEventListener('focusin', function (evt) {
      var target = evt.target.closest('[data-tooltip], [title], [data-tooltip-dynamic="true"]');
      if (!target) return;
      if (target.closest('[data-tooltip-disabled="true"]')) return;
      if (tooltipState.timer) clearTimeout(tooltipState.timer);
      tooltipState.target = target;
      var rect = target.getBoundingClientRect();
      tooltipState.lastPoint = { x: rect.left + rect.width / 2, y: rect.top };
      tooltipState.timer = setTimeout(function () {
        if (tooltipState && tooltipState.target === target) {
          showTooltip(target, tooltipState.lastPoint);
        }
      }, 200);
    });

    document.addEventListener('focusout', function () {
      hideTooltip();
    });
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  var api = {
    // Constants
    FILENAME_TRUNCATE: FILENAME_TRUNCATE,
    isIOSDevice: isIOSDevice,
    isSafariWebKit: isSafariWebKit,

    // Math / value utilities
    clamp: clamp,
    normalizeAngle: normalizeAngle,
    valueToPercent: valueToPercent,
    percentToValue: percentToValue,
    pointerPercent: pointerPercent,

    // Color utilities
    rgbToCss: rgbToCss,
    parseCssColor: parseCssColor,
    readableTextColor: readableTextColor,
    lightenRgb: lightenRgb,

    // Cursor builders
    svgCursorUrl: svgCursorUrl,
    buildDotCursorCss: buildDotCursorCss,

    // String / format utilities
    truncateFilename: truncateFilename,
    formatBytes: formatBytes,

    // DOM interaction utilities
    flashButton: flashButton,
    setPanelCollapsed: setPanelCollapsed,
    showConfirmDialog: showConfirmDialog,
    showPromptDialog: showPromptDialog,
    suppressDoubleTapZoom: suppressDoubleTapZoom,
    attachNumberInputStepper: attachNumberInputStepper,
    updateNativeRangeFill: updateNativeRangeFill,

    // Slider system
    registerSlider: registerSlider,
    refreshSlider: refreshSlider,

    // Dropdown system
    registerDropdown: registerDropdown,
    refreshDropdown: refreshDropdown,
    normalizeLabelWidths: scheduleNormalizeLabelWidths,
    openDropdown: openDropdown,
    closeDropdown: closeDropdown,
    toggleDropdown: toggleDropdown,
    positionDropdown: positionDropdown,
    getDropdown: getDropdown,
    getOpenDropdownId: getOpenDropdownId,
    closeOpenDropdown: closeOpenDropdown,
    repositionOpenDropdown: repositionOpenDropdown,

    // Tooltip system
    initTooltips: initTooltips,
    setTooltipResolver: setTooltipResolver,

    // Shared rounded-chevron SVG (used by collapsible headings and dropdowns).
    makeChevron: makeChevron,
  };

  global.ViewerUI = api;
})(window);
