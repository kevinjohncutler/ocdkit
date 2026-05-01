/*
 * ViewerPluginPanel — dynamic segmentation panel.
 *
 * Reads /api/plugins on startup, renders the active plugin's WidgetSpec
 * declarations into #sidebar as a fully host-styled panel. If no plugin
 * is registered, no panel is inserted (the container does not exist in DOM).
 *
 * Plugin authors customize layout / icons / accent / labels via the spec —
 * the prebuilt widget kinds (slider, slider_log, number, toggle, dropdown,
 * text, file, color, colormap) are host-defined.
 *
 * Public API on window.ViewerPluginPanel:
 *   init()              — async; fetch plugins, build panel, register widgets.
 *   getPluginName()     — string | null
 *   getModel()          — string | null (active model id from manifest.models)
 *   getParams()         — { [widgetName]: value }, current values
 *   hasCapability(cap)  — boolean (e.g. "resegment", "clear_cache")
 *   onParamsChanged(fn) — register a callback fired on any widget change
 */
(function (global) {
  'use strict';

  // -------------------------------------------------------------------------
  // Internal state
  // -------------------------------------------------------------------------

  var manifests = [];       // [{name, version, widgets, models, capabilities, ...}]
  var activeName = null;    // currently active plugin name
  var activeManifest = null;
  var values = {};          // { widgetName: currentValue }
  var widgetSpecs = [];     // current plugin's WidgetSpec list (sorted)
  var rootEl = null;        // #segmentationPanel element (when present)
  var paramsListeners = [];

  // -------------------------------------------------------------------------
  // Small helpers
  // -------------------------------------------------------------------------

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'dataset') {
          Object.keys(attrs[k]).forEach(function (dk) { node.dataset[dk] = attrs[k][dk]; });
        } else if (k === 'style') {
          Object.keys(attrs[k]).forEach(function (sk) { node.style[sk] = attrs[k][sk]; });
        } else if (k.indexOf('on') === 0 && typeof attrs[k] === 'function') {
          node.addEventListener(k.slice(2).toLowerCase(), attrs[k]);
        } else if (attrs[k] === true) {
          node.setAttribute(k, '');
        } else if (attrs[k] !== false && attrs[k] !== null && attrs[k] !== undefined) {
          node.setAttribute(k, String(attrs[k]));
        }
      });
    }
    if (children) {
      (Array.isArray(children) ? children : [children]).forEach(function (c) {
        if (c == null) return;
        node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
      });
    }
    return node;
  }

  function safeId(name) {
    return 'plug-' + String(name).replace(/[^A-Za-z0-9_-]/g, '_');
  }

  function notifyChange() {
    paramsListeners.forEach(function (fn) {
      try { fn(getParams()); } catch (e) { console.warn('[plugin-panel] listener error', e); }
    });
  }

  // -------------------------------------------------------------------------
  // visible_when evaluation
  // -------------------------------------------------------------------------

  function evalVisibleWhen() {
    widgetSpecs.forEach(function (spec) {
      if (!spec.visibleWhen) return;
      var node = document.getElementById(safeId('row-' + spec.name));
      if (!node) return;
      var visible = Object.keys(spec.visibleWhen).every(function (k) {
        return values[k] === spec.visibleWhen[k];
      });
      node.style.display = visible ? '' : 'none';
    });
  }

  // -------------------------------------------------------------------------
  // Renderers per WidgetKind
  // -------------------------------------------------------------------------

  function renderSlider(spec, isLog) {
    var min = Number(spec.min);
    var max = Number(spec.max);
    var step = spec.step != null ? Number(spec.step) : (isLog ? 0.001 : (max - min) / 100);
    var initial = values[spec.name];
    var sliderId = safeId(spec.name + '-slider');
    var inputId = safeId(spec.name + '-input');

    // Log sliders use a 0..1000 internal range that maps to log space.
    var rangeMin = isLog ? 0 : min;
    var rangeMax = isLog ? 1000 : max;
    var rangeStep = isLog ? 1 : step;
    function valueToRange(v) {
      if (!isLog) return Number(v);
      var lo = Math.log(min); var hi = Math.log(max);
      return Math.round(((Math.log(Math.max(min, Number(v))) - lo) / (hi - lo)) * 1000);
    }
    function rangeToValue(r) {
      if (!isLog) return Number(r);
      var lo = Math.log(min); var hi = Math.log(max);
      return Math.exp(lo + (Number(r) / 1000) * (hi - lo));
    }

    var slider = el('div', {
      class: 'slider',
      dataset: { sliderId: sliderId, sliderType: 'single' },
    }, [
      el('input', {
        type: 'range',
        id: sliderId,
        min: rangeMin, max: rangeMax, step: rangeStep,
        value: valueToRange(initial),
      }),
    ]);

    var input = el('input', {
      type: 'number',
      id: inputId,
      min: min, max: max,
      step: step,
      value: Number(initial),
    });
    var inputBox = el('div', {
      class: 'number-field',
      dataset: { numberId: inputId },
    }, [input]);

    var sliderInput = slider.querySelector('input[type="range"]');
    sliderInput.addEventListener('input', function () {
      var v = rangeToValue(sliderInput.value);
      values[spec.name] = v;
      input.value = isLog ? Number(v.toPrecision(4)) : v;
      evalVisibleWhen();
      notifyChange();
    });
    input.addEventListener('change', function () {
      var v = Number(input.value);
      if (!Number.isFinite(v)) return;
      v = Math.max(min, Math.min(max, v));
      values[spec.name] = v;
      input.value = v;
      sliderInput.value = valueToRange(v);
      if (typeof global.ViewerUI !== 'undefined' && global.ViewerUI.refreshSlider) {
        global.ViewerUI.refreshSlider(sliderId);
      } else {
        sliderInput.dispatchEvent(new Event('input'));
      }
      evalVisibleWhen();
      notifyChange();
    });

    return el('div', { class: 'control slider-inline' }, [
      el('span', { class: 'control-heading control-heading--lower' }, spec.label),
      el('div', { class: 'slider-row' }, [slider, inputBox]),
    ]);
  }

  function renderNumber(spec) {
    var inputId = safeId(spec.name + '-input');
    var input = el('input', {
      type: 'number',
      id: inputId,
      min: spec.min != null ? spec.min : undefined,
      max: spec.max != null ? spec.max : undefined,
      step: spec.step != null ? spec.step : undefined,
      value: Number(values[spec.name]),
    });
    input.addEventListener('change', function () {
      var v = Number(input.value);
      if (!Number.isFinite(v)) return;
      values[spec.name] = v;
      evalVisibleWhen();
      notifyChange();
    });
    return el('div', { class: 'control' }, [
      el('div', { class: 'control-heading' }, spec.label),
      el('div', { class: 'number-field', dataset: { numberId: inputId } }, [input]),
    ]);
  }

  function renderToggle(spec) {
    var inputId = safeId(spec.name + '-toggle');
    var input = el('input', { type: 'checkbox', id: inputId });
    input.checked = Boolean(values[spec.name]);
    input.addEventListener('change', function () {
      values[spec.name] = input.checked;
      evalVisibleWhen();
      notifyChange();
    });
    return el('div', { class: 'control toggle-group' }, [
      el('label', { class: 'toggle' }, [
        input,
        el('span', { class: 'toggle-switch' }),
        el('span', { class: 'toggle-label' }, spec.label),
      ]),
    ]);
  }

  function renderDropdown(spec) {
    var dropdownId = safeId(spec.name + '-dropdown');
    var selectId = safeId(spec.name + '-select');
    var select = el('select', { id: selectId });
    (spec.choices || []).forEach(function (c) {
      var opt = el('option', { value: c }, String(c));
      if (c === values[spec.name]) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener('change', function () {
      values[spec.name] = select.value;
      evalVisibleWhen();
      notifyChange();
    });
    return el('div', { class: 'control' }, [
      el('div', { class: 'control-heading' }, spec.label),
      el('div', {
        class: 'dropdown',
        dataset: { dropdownId: dropdownId, tooltipDisabled: 'true' },
      }, [select]),
    ]);
  }

  function renderText(spec) {
    var inputId = safeId(spec.name + '-text');
    var input = el('input', {
      type: 'text', id: inputId, value: values[spec.name] != null ? String(values[spec.name]) : '',
    });
    input.addEventListener('change', function () {
      values[spec.name] = input.value;
      evalVisibleWhen();
      notifyChange();
    });
    return el('div', { class: 'control' }, [
      el('div', { class: 'control-heading' }, spec.label),
      input,
    ]);
  }

  function renderFile(spec) {
    var inputId = safeId(spec.name + '-file');
    var input = el('input', { type: 'file', id: inputId });
    input.addEventListener('change', function () {
      var f = input.files && input.files[0];
      values[spec.name] = f ? f.name : null;
      notifyChange();
    });
    return el('div', { class: 'control' }, [
      el('div', { class: 'control-heading' }, spec.label),
      input,
    ]);
  }

  function renderColor(spec) {
    var inputId = safeId(spec.name + '-color');
    var input = el('input', {
      type: 'color', id: inputId, value: values[spec.name] || '#ffffff',
    });
    input.addEventListener('change', function () {
      values[spec.name] = input.value;
      notifyChange();
    });
    return el('div', { class: 'control' }, [
      el('div', { class: 'control-heading' }, spec.label),
      input,
    ]);
  }

  function renderSegmented(spec) {
    var groupId = safeId(spec.name + '-segmented');
    var icons = spec.choiceIcons || {};
    var group = el('div', {
      class: 'kernel-toggle segmented-toggle seg-mode-toggle',
      id: groupId, role: 'group', 'aria-label': spec.label,
    });
    function refresh() {
      Array.from(group.children).forEach(function (btn) {
        var on = btn.dataset.value === values[spec.name];
        btn.dataset.active = on ? 'true' : 'false';
        btn.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
    }
    (spec.choices || []).forEach(function (c) {
      var glyph = icons[c];
      var btn = el('button', {
        type: 'button', class: 'kernel-option',
        dataset: { value: String(c) },
        'aria-label': String(c), title: String(c),
      }, glyph ? null : String(c));
      if (glyph) {
        // CSS-class glyph (e.g. "seg-mode-icon-dbscan") vs literal text glyph
        if (/^[\w-]+$/.test(glyph) && glyph.indexOf('-') > 0) {
          btn.appendChild(el('span', { class: 'seg-mode-icon ' + glyph, 'aria-hidden': 'true' }));
        } else if (/^https?:|^data:|^\//.test(glyph)) {
          var img = el('span', { class: 'seg-mode-icon', 'aria-hidden': 'true' });
          img.style.backgroundImage = 'url("' + glyph + '")';
          btn.appendChild(img);
        } else {
          btn.appendChild(document.createTextNode(glyph));
        }
      }
      btn.addEventListener('click', function () {
        values[spec.name] = c;
        refresh();
        evalVisibleWhen();
        notifyChange();
      });
      group.appendChild(btn);
    });
    setTimeout(refresh, 0);
    return el('div', { class: 'control' }, [
      el('div', { class: 'control-heading' }, spec.label),
      group,
    ]);
  }

  function renderColormap(spec) {
    // Colormap names come from the host's colormap module if present.
    var choices = (global.ViewerColormap && typeof global.ViewerColormap.listNames === 'function')
      ? global.ViewerColormap.listNames()
      : (spec.choices || []);
    return renderDropdown(Object.assign({}, spec, { choices: choices }));
  }

  function renderCustom(spec) {
    // Escape hatch: plugin's own JS attaches via this anchor.
    return el('div', {
      class: 'control plugin-custom',
      dataset: { pluginCustom: spec.name },
    });
  }

  var WIDGET_RENDERERS = {
    slider:     function (s) { return renderSlider(s, false); },
    slider_log: function (s) { return renderSlider(s, true); },
    number:     renderNumber,
    toggle:     renderToggle,
    dropdown:   renderDropdown,
    segmented:  renderSegmented,
    text:       renderText,
    file:       renderFile,
    color:      renderColor,
    colormap:   renderColormap,
    custom:     renderCustom,
  };

  // -------------------------------------------------------------------------
  // Panel construction
  // -------------------------------------------------------------------------

  function applyAccent(node, accent) {
    if (!accent) return;
    node.style.setProperty('--accent-color', accent);
    node.style.setProperty('--accent-hover', accent);
  }

  // Inject the host-managed styles for collapsible headings exactly once.
  // Hover on the heading shifts the chevron color to the active accent
  // (and bumps opacity to 1). The label stays in its normal heading color.
  function ensureCollapsibleStyles() {
    if (document.getElementById('plugin-collapsible-styles')) return;
    var style = document.createElement('style');
    style.id = 'plugin-collapsible-styles';
    style.textContent =
      '.plugin-collapsible-heading:hover .plugin-collapsible-chevron {' +
      '  color: var(--accent-color);' +
      '  opacity: 1;' +
      '}';
    document.head.appendChild(style);
  }

  // Shared rounded-triangle chevron (defined in ui-utils.js so both
  // plugin headings and dropdown toggles share the same glyph). Adds the
  // `plugin-collapsible-chevron` class for hover/accent styling.
  function makeChevron() {
    return global.ViewerUI.makeChevron({ className: 'plugin-collapsible-chevron' });
  }

  // Render a group heading that doubles as a clickable expand/collapse
  // control. Backed by a hidden `WidgetSpec` (a toggle with `asHeader=true`)
  // whose `values[name]` drives `visible_when` for the rest of the group.
  function buildCollapsibleHeading(groupName, headerSpec) {
    ensureCollapsibleStyles();
    var heading = el('div', {
      class: 'control-heading plugin-collapsible-heading',
      role: 'button', tabindex: '0',
      'aria-controls': safeId('group-' + groupName),
    });
    heading.style.display = 'flex';
    heading.style.alignItems = 'center';
    heading.style.justifyContent = 'space-between';
    heading.style.cursor = 'pointer';
    heading.style.userSelect = 'none';

    var label = headerSpec.label || groupName;
    heading.appendChild(el('span', { class: 'plugin-collapsible-label' }, label));

    var chevron = makeChevron();
    heading.appendChild(chevron);

    function refresh() {
      var open = Boolean(values[headerSpec.name]);
      heading.setAttribute('aria-expanded', open ? 'true' : 'false');
      // Same equilateral triangle in both states — just rotated 90°.
      chevron.style.transform = open ? 'rotate(90deg)' : 'rotate(0deg)';
    }
    function toggle() {
      values[headerSpec.name] = !values[headerSpec.name];
      refresh();
      // `as_header` widgets are UI-state (collapse/expand), not plugin
      // params. Re-evaluate visibility for gated widgets in the same group,
      // but DON'T fire `notifyChange` — otherwise every chevron click
      // wastes a /api/resegment round-trip producing an identical mask.
      evalVisibleWhen();
    }
    heading.addEventListener('click', toggle);
    heading.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggle();
      }
    });
    if (headerSpec.help) heading.dataset.tooltip = headerSpec.help;
    setTimeout(refresh, 0);
    return heading;
  }

  // Same chevron pattern as `buildCollapsibleHeading` but driven by a local
  // boolean (not a `WidgetSpec`). Used by host-managed collapsible groups
  // like the Network Predictions overlays section.
  function buildHostCollapsibleHeading(labelText, contentEl, defaultOpen) {
    ensureCollapsibleStyles();
    var open = Boolean(defaultOpen);
    var heading = el('div', {
      class: 'control-heading plugin-collapsible-heading',
      role: 'button', tabindex: '0',
    });
    heading.style.display = 'flex';
    heading.style.alignItems = 'center';
    heading.style.justifyContent = 'space-between';
    heading.style.cursor = 'pointer';
    heading.style.userSelect = 'none';
    heading.appendChild(el('span', { class: 'plugin-collapsible-label' }, labelText));
    var chevron = makeChevron();
    heading.appendChild(chevron);

    function refresh() {
      heading.setAttribute('aria-expanded', open ? 'true' : 'false');
      chevron.style.transform = open ? 'rotate(90deg)' : 'rotate(0deg)';
      contentEl.style.display = open ? '' : 'none';
    }
    function toggle() {
      open = !open;
      refresh();
    }
    heading.addEventListener('click', toggle);
    heading.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggle();
      }
    });
    refresh();
    return heading;
  }

  function applyIcon(node, icon) {
    if (!icon) return;
    var wrap = el('span', { class: 'plugin-widget-icon', 'aria-hidden': 'true' });
    if (/^https?:|^data:|^\//.test(icon)) {
      wrap.style.backgroundImage = 'url("' + icon + '")';
    } else {
      wrap.textContent = icon;
    }
    node.insertBefore(wrap, node.firstChild);
  }

  function buildPanel(manifest) {
    activeManifest = manifest;
    widgetSpecs = (manifest.widgets || []).slice().sort(function (a, b) {
      var pa = a.placement != null ? a.placement : 1e9;
      var pb = b.placement != null ? b.placement : 1e9;
      return pa - pb;
    });

    // Defaults for widget values
    values = {};
    widgetSpecs.forEach(function (s) { values[s.name] = s.default; });

    var panel = el('div', { class: 'panel-section', id: 'segmentationPanel' });

    // Plugin selector (only when more than one is registered)
    if (manifests.length > 1) {
      var psSel = el('select', { id: 'pluginSelector' });
      manifests.forEach(function (m) {
        var opt = el('option', { value: m.name }, m.name);
        if (m.name === activeName) opt.selected = true;
        psSel.appendChild(opt);
      });
      psSel.addEventListener('change', function () {
        selectPlugin(psSel.value);
      });
      panel.appendChild(el('div', { class: 'control' }, [
        el('div', { class: 'control-heading' }, 'Plugin'),
        el('div', {
          class: 'dropdown',
          dataset: { dropdownId: 'pluginSelector', tooltipDisabled: 'true' },
        }, [psSel]),
      ]));
    }

    // Model selector — keep id="segmentationModel" for compatibility with
    // existing app.js handlers (file picker, persistence, payload).
    // The host dropdown's `__add__` sentinel renders this as a sticky
    // "+ Add model…" row at the bottom, regardless of where the loop
    // window is currently scrolled.
    if (Array.isArray(manifest.models) && manifest.models.length) {
      var modelSelect = el('select', { id: 'segmentationModel' });
      manifest.models.forEach(function (m) {
        modelSelect.appendChild(el('option', { value: m }, m));
      });
      var addOpt = el('option', { value: '__add__' }, '+ Add model…');
      modelSelect.appendChild(addOpt);

      var fileInput = el('input', {
        type: 'file', id: 'segmentationModelFile',
        accept: '.pth,.pt,.onnx,.ckpt',
        style: { display: 'none' },
      });

      function syncModelDropdown() {
        if (typeof global.ViewerUI === 'undefined') return;
        var dd = global.ViewerUI.getDropdown('segmentationModel');
        if (!dd) return;
        dd.options = Array.from(modelSelect.options).map(function (o) {
          return {
            value: o.value,
            label: o.textContent || o.value,
            disabled: o.disabled,
            title: o.title || o.textContent || o.value,
          };
        });
        if (typeof dd.buildMenu === 'function') dd.buildMenu();
      }

      fileInput.addEventListener('change', function () {
        var f = fileInput.files && fileInput.files[0];
        if (!f) {
          modelSelect.value = manifest.models[0] || '';
          return;
        }
        var value = 'file:' + f.name;
        // Insert just before __add__ so the new entry stays in the loop window
        // while the action sticks at the bottom.
        var opt = el('option', { value: value, title: f.name }, f.name);
        modelSelect.insertBefore(opt, addOpt);
        modelSelect.value = value;
        if (typeof window !== 'undefined') {
          window.customSegmentationModelPath = f.path || f.name;
        }
        syncModelDropdown();
        notifyChange();
      });

      modelSelect.addEventListener('change', function () {
        if (modelSelect.value === '__add__') {
          fileInput.click();
        }
      });

      panel.appendChild(el('div', { class: 'control' }, [
        el('div', { class: 'control-heading' }, 'Model'),
        el('div', {
          class: 'dropdown',
          dataset: { dropdownId: 'segmentationModel', loop: 'true', countLabel: 'models', tooltipDisabled: 'true' },
        }, [modelSelect]),
        fileInput,
      ]));
    }

    // Pre-scan: a widget with `asHeader=true` and a `group` becomes that
    // group's clickable heading (chevron). It is NOT rendered as a row.
    // Map: groupName -> spec acting as header (first one wins).
    var headerSpecForGroup = {};
    widgetSpecs.forEach(function (spec) {
      if (spec.asHeader && spec.group && !headerSpecForGroup[spec.group]) {
        headerSpecForGroup[spec.group] = spec;
      }
    });

    // Widgets, grouped by `group`
    var currentGroup = null;
    widgetSpecs.forEach(function (spec) {
      var renderer = WIDGET_RENDERERS[spec.kind];
      if (!renderer) {
        console.warn('[plugin-panel] unknown widget kind:', spec.kind);
        return;
      }
      if (spec.group && spec.group !== currentGroup) {
        currentGroup = spec.group;
        var headerSpec = headerSpecForGroup[currentGroup];
        if (headerSpec) {
          panel.appendChild(buildCollapsibleHeading(currentGroup, headerSpec));
        } else {
          panel.appendChild(el('div', { class: 'control-heading' }, currentGroup));
        }
      } else if (!spec.group) {
        currentGroup = null;
      }
      // The header-bearing widget itself does not get its own row — its
      // value is driven entirely by the heading chevron.
      if (spec.asHeader && spec.group && headerSpecForGroup[spec.group] === spec) {
        return;
      }
      var row = renderer(spec);
      row.id = safeId('row-' + spec.name);
      applyAccent(row, spec.accent);
      applyIcon(row.querySelector('.control-heading, .toggle-label') || row, spec.icon);
      if (spec.help) row.dataset.tooltip = spec.help;
      panel.appendChild(row);
    });

    // Action row — Segment + Clear only. Clear-cache (when the plugin
    // advertises it) lives on the right-click context menu, not the panel.
    var actionRow = el('div', { class: 'control seg-action-row' });
    actionRow.appendChild(el('button', {
      id: 'segmentButton', class: 'accent-button', type: 'button',
    }, 'Segment'));
    actionRow.appendChild(el('button', {
      id: 'clearMasksButton', class: 'accent-button', type: 'button',
      title: 'Clear all masks (Cmd/Ctrl+X)',
    }, 'Clear'));
    panel.appendChild(actionRow);
    panel.appendChild(el('div', { id: 'segmentStatus', class: 'status' }));

    // Host-managed display toggles. These control visibility of overlays
    // that the host renders from `extras` returned by plugin.run(). They
    // are NOT plugin params (not sent to /api/segment). Toggles appear for
    // every key declared in `manifest.displayOverlays`, but stay disabled
    // until the matching extras payload actually arrives — wired by
    // setOverlayAvailability().
    panel.appendChild(buildDisplaySection(manifest.displayOverlays || []));

    return panel;
  }

  // The IDs here must match the lookups in app.js (overlay handlers).
  // `extrasKey` is the field on /api/segment's response that gates each
  // toggle. `defaultOn: true` matches the matching JS global's initial
  // value so the toggle's checked state agrees with the renderer at first
  // paint — without this, the checkbox shows OFF while a graph still
  // renders because some host globals start `true` (e.g. showAffinityGraph).
  var DISPLAY_TOGGLES = [
    { id: 'affinityGraphToggle', label: 'Affinity Graph', extrasKey: 'affinityGraph', group: 'reconstruction', defaultOn: true },
    { id: 'pointsOverlayToggle', label: 'Final Points',   extrasKey: 'points',         group: 'reconstruction' },
    { id: 'vectorOverlayToggle', label: 'Displacement',   extrasKey: 'vector',         group: 'reconstruction', rowId: 'vectorOverlayRow', title: 'overall pixel displacement' },
    { id: 'flowOverlayToggle',     label: 'Flow',     extrasKey: 'flowOverlay',     group: 'predictions' },
    { id: 'distanceOverlayToggle', label: 'Distance', extrasKey: 'distanceOverlay', group: 'predictions' },
  ];

  function buildDisplaySection(declaredKeys) {
    var declared = new Set(declaredKeys || []);
    // `display: contents` makes the wrapper invisible to layout, so the
    // headings + toggle rows participate directly in the parent
    // .panel-section's flex column gap (12px) — same spacing as the
    // Plugin / Model / Parameters sections above.
    var wrapper = el('div', {
      id: 'displayOverlaysSection',
      style: { display: 'contents' },
    });

    // Each display group can be plain (always-visible heading) or
    // collapsible (clickable chevron heading). Collapsible groups start in
    // the requested `defaultOpen` state. Network Predictions is debug-y, so
    // it starts collapsed by default.
    var groups = [
      { key: 'reconstruction', heading: 'Mask Reconstruction' },
      { key: 'predictions',    heading: 'Network Predictions', collapsible: true, defaultOpen: false },
    ];

    var anyDeclared = false;
    groups.forEach(function (g) {
      var groupToggles = DISPLAY_TOGGLES.filter(function (t) {
        return t.group === g.key && declared.has(t.extrasKey);
      });
      if (!groupToggles.length) return;
      anyDeclared = true;

      var row = el('div', { class: 'control toggle-group' });
      groupToggles.forEach(function (t) {
        var attrs = { type: 'checkbox', id: t.id, disabled: true };
        if (t.defaultOn) attrs.checked = true;
        var input = el('input', attrs);
        var label = el('label', { class: 'toggle' }, [
          input,
          el('span', { class: 'toggle-switch' }),
          el('span', { class: 'toggle-label' }, t.label),
        ]);
        if (t.rowId) label.id = t.rowId;
        if (t.title) label.dataset.tooltip = t.title;
        row.appendChild(label);
      });

      if (g.collapsible) {
        // Reuse the same chevron-heading machinery as plugin asHeader
        // widgets. The collapse state lives on a closure boolean rather
        // than `values` since this is a host-managed (not plugin-driven)
        // section. The row goes inside an inner div so we can show/hide it.
        wrapper.appendChild(buildHostCollapsibleHeading(g.heading, row, g.defaultOpen));
        wrapper.appendChild(row);
      } else {
        wrapper.appendChild(el('div', { class: 'control-heading' }, g.heading));
        wrapper.appendChild(row);
      }
    });

    // Drop the wrapper if no overlays were declared — `display: contents`
    // would still preserve the gap. Use `display: none` instead.
    if (!anyDeclared) wrapper.style.display = 'none';
    return wrapper;
  }

  // Called by the host after each /api/segment response. Enables/disables
  // each declared display toggle based on whether the matching extras key
  // arrived in this response. Toggles stay visible at all times — only the
  // disabled state changes.
  function setOverlayAvailability(extras) {
    extras = extras || {};
    DISPLAY_TOGGLES.forEach(function (t) {
      var input = document.getElementById(t.id);
      if (!input) return;
      var has = Boolean(extras[t.extrasKey]);
      input.disabled = !has;
      if (!has && input.checked) input.checked = false;
    });
  }

  function mountPanel(panel) {
    if (rootEl && rootEl.parentNode) rootEl.parentNode.removeChild(rootEl);
    var sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    // Insert at the very top of the sidebar (above #systemInfoPanel).
    sidebar.insertBefore(panel, sidebar.firstChild);
    rootEl = panel;

    // Register host-managed widgets (sliders, dropdowns) so styling/interaction
    // hooks attach. Defer to next tick so layout has settled.
    setTimeout(function () {
      if (typeof global.ViewerUI === 'undefined') return;
      panel.querySelectorAll('[data-slider-id]').forEach(function (n) {
        try { global.ViewerUI.registerSlider(n); } catch (e) { console.warn(e); }
      });
      panel.querySelectorAll('[data-dropdown-id]').forEach(function (n) {
        try { global.ViewerUI.registerDropdown(n); } catch (e) { console.warn(e); }
      });
    }, 0);

    // Show/hide GPU toggle row based on plugin capability.
    var gpuRow = document.getElementById('useGpuRow');
    if (gpuRow) {
      var showGpu = activeManifest && activeManifest.capabilities && activeManifest.capabilities.set_use_gpu;
      gpuRow.style.display = showGpu ? '' : 'none';
    }

    evalVisibleWhen();
    // Re-apply any tooltip overrides the user has saved locally — plugin
    // widget tooltips are set fresh each rebuild so they need a re-pass.
    if (global.TooltipEditor && global.TooltipEditor.applyOverrides) {
      global.TooltipEditor.applyOverrides();
    }
    // Don't notify on mount — listeners pull current values when they need
    // them. Firing here would auto-segment on first page load.
  }

  function unmountPanel() {
    if (rootEl && rootEl.parentNode) rootEl.parentNode.removeChild(rootEl);
    rootEl = null;
    activeManifest = null;
    widgetSpecs = [];
    values = {};
    var gpuRow = document.getElementById('useGpuRow');
    if (gpuRow) gpuRow.style.display = 'none';
    notifyChange();
  }

  function selectPlugin(name) {
    if (!name) return Promise.resolve(null);
    return fetch('/api/plugin/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name }),
    }).then(function (r) { return r.json(); }).then(function (resp) {
      if (resp && resp.manifest) {
        activeName = resp.active || name;
        // Update local manifest cache too.
        var idx = manifests.findIndex(function (m) { return m.name === resp.manifest.name; });
        if (idx >= 0) manifests[idx] = resp.manifest;
        else manifests.push(resp.manifest);
        mountPanel(buildPanel(resp.manifest));
      }
      return resp;
    }).catch(function (err) {
      console.warn('[plugin-panel] select failed', err);
    });
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  function init() {
    return fetch('/api/plugins').then(function (r) { return r.json(); }).then(function (resp) {
      manifests = (resp && resp.plugins) || [];
      activeName = (resp && resp.active) || (manifests[0] && manifests[0].name) || null;
      if (!activeName || !manifests.length) {
        // No plugin → no panel. Container is absent from DOM by design.
        return null;
      }
      var manifest = manifests.find(function (m) { return m.name === activeName; }) || manifests[0];
      mountPanel(buildPanel(manifest));
      return manifest;
    }).catch(function (err) {
      console.warn('[plugin-panel] init failed', err);
      return null;
    });
  }

  function getPluginName() { return activeName; }

  function getModel() {
    var sel = document.getElementById('segmentationModel');
    return sel ? sel.value : null;
  }

  function getParams() {
    // Exclude `as_header` widgets — they're UI-state (collapse/expand) for
    // the host, never meant to reach the plugin.
    var out = {};
    var skip = new Set();
    widgetSpecs.forEach(function (s) { if (s.asHeader) skip.add(s.name); });
    Object.keys(values).forEach(function (k) {
      if (!skip.has(k)) out[k] = values[k];
    });
    return out;
  }

  function hasCapability(cap) {
    return Boolean(activeManifest && activeManifest.capabilities && activeManifest.capabilities[cap]);
  }

  // Element to highlight with the chasing-outline animation while inference
  // runs. Plugins can override via SegmentationPlugin.progress_target; default
  // is the segmentation panel root.
  function getProgressTarget() {
    var sel = activeManifest && activeManifest.progressTarget;
    if (sel) {
      var el = document.querySelector(sel);
      if (el) return el;
    }
    return document.getElementById('segmentationPanel');
  }

  function onParamsChanged(fn) {
    if (typeof fn === 'function') paramsListeners.push(fn);
  }

  global.ViewerPluginPanel = {
    init: init,
    getPluginName: getPluginName,
    getModel: getModel,
    getParams: getParams,
    hasCapability: hasCapability,
    onParamsChanged: onParamsChanged,
    selectPlugin: selectPlugin,
    setOverlayAvailability: setOverlayAvailability,
    getProgressTarget: getProgressTarget,
  };
})(window);
