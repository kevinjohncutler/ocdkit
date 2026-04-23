/**
 * TEMPORARY debug panel for testing -apple-visual-effect material values.
 * Two independent selectors: one for main panels, one for overlay elements.
 * Remove this file + the <script> tag in index.html when done.
 *
 * Uses a dynamic <style> sheet instead of inline styles because WebKit
 * ignores inline -apple-visual-effect set via style.setProperty().
 */
(function () {
  const GLASS_VALUES = [
    ['none', 'none (fallback CSS)'],
    ['-apple-system-glass-material', 'Glass'],
    ['-apple-system-glass-material-clear', 'Glass Clear'],
    ['-apple-system-glass-material-subdued', 'Glass Subdued'],
    ['-apple-system-glass-material-media-controls', 'Glass Media'],
    ['-apple-system-glass-material-media-controls-subdued', 'Glass Media Subdued'],
    ['-apple-system-blur-material-ultra-thin', 'Blur Ultra-Thin'],
    ['-apple-system-blur-material-thin', 'Blur Thin'],
    ['-apple-system-blur-material', 'Blur Regular'],
    ['-apple-system-blur-material-thick', 'Blur Thick'],
    ['-apple-system-blur-material-chrome', 'Blur Chrome'],
  ];
  // Glass inflates small overlay elements — only offer blur for overlays
  const BLUR_ONLY = [
    ['none', 'none (fallback CSS)'],
    ['-apple-system-blur-material-ultra-thin', 'Blur Ultra-Thin'],
    ['-apple-system-blur-material-thin', 'Blur Thin'],
    ['-apple-system-blur-material', 'Blur Regular'],
    ['-apple-system-blur-material-thick', 'Blur Thick'],
    ['-apple-system-blur-material-chrome', 'Blur Chrome'],
  ];

  const PANEL_SEL = '.panel-section';
  // Target .dropdown-menu (inner), not .dropdown-menu-wrap (outer) —
  // the wrap has max-height/overflow/::before that inflate the glass area
  // Tooltips excluded — they use opacity:0 when hidden and glass renders through that
  const OVERLAY_SEL = '.dropdown[data-open="true"] .dropdown-menu, .omni-confirm, .omni-context-menu, .histogram-surface';

  // Dynamic stylesheet we rewrite on every change
  const sheet = document.createElement('style');
  sheet.id = 'debugMaterialSheet';
  document.head.appendChild(sheet);

  let panelVal = '-apple-system-glass-material-subdued';
  let overlayVal = '-apple-system-glass-material-clear';

  function updateSheet() {
    const rules = [];
    if (panelVal !== 'none') {
      rules.push(`${PANEL_SEL} { -apple-visual-effect: ${panelVal} !important; }`);
    } else {
      rules.push(`${PANEL_SEL} { -apple-visual-effect: none !important; }`);
    }
    if (overlayVal !== 'none') {
      rules.push(`${OVERLAY_SEL} { -apple-visual-effect: ${overlayVal} !important; }`);
    } else {
      rules.push(`${OVERLAY_SEL} { -apple-visual-effect: none !important; }`);
    }
    sheet.textContent = rules.join('\n');
  }

  function buildSelect(id, label, options, defaultVal) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;gap:6px;';
    const lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.cssText = 'font-size:11px;min-width:60px;';
    const sel = document.createElement('select');
    sel.id = id;
    sel.style.cssText = 'flex:1;font-size:11px;background:#333;color:#eee;border:1px solid #555;border-radius:4px;padding:2px 4px;';
    for (const [val, name] of options) {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = name;
      if (val === defaultVal) opt.selected = true;
      sel.appendChild(opt);
    }
    wrap.appendChild(lbl);
    wrap.appendChild(sel);
    return wrap;
  }

  const panel = document.createElement('div');
  panel.id = 'debugMaterialPanel';
  panel.style.cssText = [
    'position:fixed', 'top:8px', 'left:50%', 'transform:translateX(-50%)',
    'z-index:99999', 'background:rgba(30,30,30,0.92)', 'border:1px solid #555',
    'border-radius:10px', 'padding:10px 14px', 'display:flex', 'flex-direction:column',
    'gap:6px', 'font-family:-apple-system,sans-serif', 'color:#eee',
    'backdrop-filter:blur(12px)', '-webkit-backdrop-filter:blur(12px)',
    'box-shadow:0 4px 20px rgba(0,0,0,0.4)', 'cursor:move', 'user-select:none',
  ].join(';');

  const title = document.createElement('div');
  title.textContent = 'Material Debug';
  title.style.cssText = 'font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;opacity:0.6;text-align:center;';
  panel.appendChild(title);

  panel.appendChild(buildSelect('dbgPanelMat', 'Panels', GLASS_VALUES, panelVal));
  panel.appendChild(buildSelect('dbgOverlayMat', 'Overlays', GLASS_VALUES, overlayVal));

  document.body.appendChild(panel);

  document.getElementById('dbgPanelMat').addEventListener('change', (e) => {
    panelVal = e.target.value;
    updateSheet();
  });

  document.getElementById('dbgOverlayMat').addEventListener('change', (e) => {
    overlayVal = e.target.value;
    updateSheet();
  });

  // Initial
  updateSheet();

  // Draggable
  let dragging = false, dx = 0, dy = 0;
  panel.addEventListener('mousedown', (e) => {
    if (e.target.tagName === 'SELECT') return;
    dragging = true;
    const r = panel.getBoundingClientRect();
    dx = e.clientX - r.left;
    dy = e.clientY - r.top;
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    panel.style.left = (e.clientX - dx) + 'px';
    panel.style.top = (e.clientY - dy) + 'px';
    panel.style.transform = 'none';
  });
  document.addEventListener('mouseup', () => { dragging = false; });
})();
