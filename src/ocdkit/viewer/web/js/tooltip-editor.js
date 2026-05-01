/*
 * TooltipEditor — in-place editing of tooltip text.
 *
 *   Ctrl/Cmd + Shift + T → toggle edit mode
 *   Ctrl/Cmd + Shift + E → copy override JSON to clipboard (export)
 *   Esc                  → close popover or exit edit mode
 *
 * Overrides are stored in localStorage under `ocdkit.tooltipOverrides.v1`
 * and applied on page load + after plugin-panel rebuilds. Source files are
 * never modified by this module — the export step puts a JSON blob on the
 * clipboard so the user (or assistant) can patch the originating Python
 * WidgetSpec.help / dataset.tooltip site explicitly.
 *
 * Element keying:
 *   - Plugin widget rows  → `plugin:<pluginName>:widget:<widgetName>`
 *     (resolved from id="row-<widgetName>")
 *   - Anything else with an id → `id:<elementId>`
 *   - Otherwise: skipped (no stable handle to reapply on later loads).
 */
(function (global) {
  'use strict';

  var STORAGE_KEY = 'ocdkit.tooltipOverrides.v1';
  var overrides = {};
  var editMode = false;
  var indicatorEl = null;
  var openPopover = null;

  // ---------- storage ----------

  function loadFromStorage() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      overrides = raw ? JSON.parse(raw) : {};
    } catch (e) {
      console.warn('[tooltip-editor] localStorage load failed:', e);
      overrides = {};
    }
  }

  function saveToStorage() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    } catch (e) {
      console.warn('[tooltip-editor] localStorage save failed:', e);
    }
  }

  // ---------- key resolution ----------

  function keyForElement(el) {
    if (!el || !el.closest) return null;
    var rowMatch = el.closest('[id^="row-"]');
    if (rowMatch && rowMatch.id) {
      var widgetName = rowMatch.id.replace(/^row-/, '');
      var pluginName = global.ViewerPluginPanel && global.ViewerPluginPanel.getPluginName
        ? global.ViewerPluginPanel.getPluginName()
        : null;
      if (pluginName) return 'plugin:' + pluginName + ':widget:' + widgetName;
    }
    var withId = el.closest('[id]');
    if (withId && withId.id) return 'id:' + withId.id;
    return null;
  }

  function findElementsForKey(key) {
    if (!key) return [];
    if (key.indexOf('plugin:') === 0) {
      var parts = key.split(':');
      var pluginName = parts[1];
      var widgetName = parts[3];
      var current = global.ViewerPluginPanel && global.ViewerPluginPanel.getPluginName
        ? global.ViewerPluginPanel.getPluginName()
        : null;
      if (current !== pluginName) return [];
      var row = document.getElementById('row-' + widgetName);
      return row ? [row] : [];
    }
    if (key.indexOf('id:') === 0) {
      var el = document.getElementById(key.slice(3));
      return el ? [el] : [];
    }
    return [];
  }

  // ---------- override resolver ----------

  // Hover-time resolver. ui-utils.js's tooltip system calls this before its
  // own dataset.tooltip lookup; returning a string here surfaces an override
  // without ever mutating data-tooltip (so async code paths that re-set the
  // canonical text — updateSystemInfo, plugin-panel rebuilds — don't clobber
  // edits, and reverts trivially restore the original).
  function resolveOverride(el) {
    if (!el) return null;
    var key = keyForElement(el);
    if (!key) return null;
    var entry = overrides[key];
    return entry && typeof entry.override === 'string' ? entry.override : null;
  }

  // No-op kept for back-compat with plugin-panel.js's mountPanel hook.
  function applyOverrides() {}

  function revertOverride(key) {
    delete overrides[key];
    saveToStorage();
    updateIndicator();
  }

  // ---------- edit-mode UI ----------

  function setEditMode(active) {
    editMode = !!active;
    document.body.classList.toggle('tooltip-edit-mode', editMode);
    if (editMode) showIndicator();
    else { closePopover(); hideIndicator(); }
  }

  function showIndicator() {
    if (indicatorEl) return;
    indicatorEl = document.createElement('div');
    indicatorEl.className = 'tooltip-edit-indicator';
    document.body.appendChild(indicatorEl);
    updateIndicator();
  }

  function hideIndicator() {
    if (indicatorEl) { indicatorEl.remove(); indicatorEl = null; }
  }

  function updateIndicator() {
    if (!indicatorEl) return;
    var n = Object.keys(overrides).length;
    var plural = n === 1 ? '' : 's';
    indicatorEl.innerHTML =
      '<strong>Tooltip edit mode</strong> — ' + n + ' override' + plural +
      ' &middot; <kbd>Ctrl+Shift+E</kbd> export &middot; <kbd>Ctrl+Shift+T</kbd> exit';
  }

  function flashIndicator(msg) {
    if (!indicatorEl) return;
    indicatorEl.textContent = msg;
    setTimeout(updateIndicator, 1500);
  }

  // ---------- editor popover ----------

  function closePopover() {
    if (openPopover) { openPopover.remove(); openPopover = null; }
  }

  function openEditor(el, x, y) {
    closePopover();
    var key = keyForElement(el);
    if (!key) {
      console.warn('[tooltip-editor] no stable key for element; cannot edit', el);
      return;
    }
    // dataset.tooltip is always the canonical (host-set) string — overrides
    // live only in the in-memory map and localStorage, never mutate the DOM.
    var original = el.dataset.tooltip || '';
    var existingOverride = overrides[key] && overrides[key].override;
    var current = (typeof existingOverride === 'string') ? existingOverride : original;

    var pop = document.createElement('div');
    pop.className = 'tooltip-edit-popover';

    var head = document.createElement('div');
    head.className = 'tooltip-edit-popover-head';
    head.textContent = key;
    pop.appendChild(head);

    var ta = document.createElement('textarea');
    ta.className = 'tooltip-edit-popover-textarea';
    ta.value = current;
    ta.rows = Math.max(3, Math.min(12, current.split('\n').length + 1));
    pop.appendChild(ta);

    var origBlock = document.createElement('details');
    origBlock.className = 'tooltip-edit-popover-original';
    var origSummary = document.createElement('summary');
    origSummary.textContent = 'original';
    origBlock.appendChild(origSummary);
    var origPre = document.createElement('pre');
    origPre.textContent = original;
    origBlock.appendChild(origPre);
    pop.appendChild(origBlock);

    var btnRow = document.createElement('div');
    btnRow.className = 'tooltip-edit-popover-buttons';

    var saveBtn = document.createElement('button');
    saveBtn.type = 'button';
    saveBtn.textContent = 'Save';
    saveBtn.className = 'tooltip-edit-save';
    saveBtn.addEventListener('click', function () {
      var newText = ta.value;
      if (newText === original) {
        revertOverride(key);
      } else {
        overrides[key] = {
          original: original,
          override: newText,
          ts: new Date().toISOString(),
        };
        saveToStorage();
        updateIndicator();
      }
      closePopover();
    });

    var revertBtn = document.createElement('button');
    revertBtn.type = 'button';
    revertBtn.textContent = 'Revert';
    revertBtn.className = 'tooltip-edit-revert';
    revertBtn.addEventListener('click', function () {
      revertOverride(key);
      closePopover();
    });

    var cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.addEventListener('click', closePopover);

    btnRow.appendChild(saveBtn);
    btnRow.appendChild(revertBtn);
    btnRow.appendChild(cancelBtn);
    pop.appendChild(btnRow);

    document.body.appendChild(pop);
    openPopover = pop;

    // Position: prefer (x+8, y+8), but keep on-screen.
    var rect = pop.getBoundingClientRect();
    var pad = 12;
    var left = Math.min(x + 8, window.innerWidth - rect.width - pad);
    var top = Math.min(y + 8, window.innerHeight - rect.height - pad);
    pop.style.left = Math.max(pad, left) + 'px';
    pop.style.top = Math.max(pad, top) + 'px';

    setTimeout(function () { ta.focus(); ta.select(); }, 0);
  }

  // ---------- export ----------

  function exportToClipboard() {
    var keys = Object.keys(overrides);
    if (!keys.length) {
      flashIndicator('No overrides to export');
      return;
    }
    var lines = ['# tooltip overrides — ' + new Date().toISOString()];
    lines.push('# ' + keys.length + ' entries; copy to assistant or paste into source.');
    keys.forEach(function (k) {
      lines.push('#   ' + k);
    });
    lines.push('');
    lines.push(JSON.stringify({ version: 1, overrides: overrides }, null, 2));
    var blob = lines.join('\n') + '\n';

    var done = function () { flashIndicator('Copied ' + keys.length + ' override(s) to clipboard'); };
    var fail = function (err) {
      console.error('[tooltip-editor] clipboard write failed', err);
      window.prompt('Copy these overrides:', blob);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(blob).then(done, fail);
    } else {
      fail(new Error('clipboard API unavailable'));
    }
  }

  // ---------- event wiring ----------

  function isEditableInput(el) {
    if (!el) return false;
    if (el === openPopover) return false;
    if (openPopover && openPopover.contains(el)) return true;
    var tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
  }

  document.addEventListener('click', function (evt) {
    if (!editMode) return;
    if (openPopover && openPopover.contains(evt.target)) return;
    var t = evt.target.closest('[data-tooltip]');
    if (!t) {
      closePopover();
      return;
    }
    evt.preventDefault();
    evt.stopPropagation();
    openEditor(t, evt.clientX, evt.clientY);
  }, true);

  document.addEventListener('keydown', function (e) {
    var meta = e.ctrlKey || e.metaKey;
    if (meta && e.shiftKey && (e.key === 'T' || e.key === 't')) {
      e.preventDefault();
      setEditMode(!editMode);
      return;
    }
    if (meta && e.shiftKey && (e.key === 'E' || e.key === 'e')) {
      e.preventDefault();
      exportToClipboard();
      return;
    }
    if (e.key === 'Escape' && editMode) {
      if (openPopover) closePopover();
      else setEditMode(false);
    }
  });

  // Initial load + register resolver with the tooltip system.
  loadFromStorage();
  function registerResolver() {
    if (global.ViewerUI && global.ViewerUI.setTooltipResolver) {
      global.ViewerUI.setTooltipResolver(resolveOverride);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', registerResolver);
  } else {
    registerResolver();
  }

  global.TooltipEditor = {
    setEditMode: setEditMode,
    isEditMode: function () { return editMode; },
    getOverrides: function () { return JSON.parse(JSON.stringify(overrides)); },
    clearAll: function () {
      overrides = {};
      saveToStorage();
      updateIndicator();
    },
    // Kept for back-compat with the plugin-panel.js hook; resolver-based
    // approach makes per-element re-application unnecessary.
    applyOverrides: applyOverrides,
    exportToClipboard: exportToClipboard,
  };
})(window);
