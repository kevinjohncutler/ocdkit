"""Static asset loading and HTML template rendering.

The viewer ships a fixed JS/CSS/HTML layout. Plugins do not customize this —
they declare widget specs and the frontend renders them into the settings
pane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

VIEWER_DIR = Path(__file__).resolve().parent
WEB_DIR = (VIEWER_DIR / "web").resolve()
LOG_DIR = Path.home() / ".ocdkit" / "logs"
LOG_FILE = LOG_DIR / "viewer.log"

INDEX_HTML = WEB_DIR / "index.html"
APP_JS = WEB_DIR / "app.js"
HTML_DIR = WEB_DIR / "html"
CSS_DIR = WEB_DIR / "css"
POINTER_JS = WEB_DIR / "js" / "pointer-state.js"
LOGGING_JS = WEB_DIR / "js" / "logging.js"
HISTORY_JS = WEB_DIR / "js" / "history.js"
BRUSH_JS = WEB_DIR / "js" / "brush.js"
PAINTING_JS = WEB_DIR / "js" / "painting.js"
INTERACTIONS_JS = WEB_DIR / "js" / "interactions.js"
COLORMAP_JS = WEB_DIR / "js" / "colormap.js"
UI_UTILS_JS = WEB_DIR / "js" / "ui-utils.js"
STATE_PERSISTENCE_JS = WEB_DIR / "js" / "state-persistence.js"
FILE_NAVIGATION_JS = WEB_DIR / "js" / "file-navigation.js"
PLUGIN_PANEL_JS = WEB_DIR / "js" / "plugin-panel.js"
TOOLTIP_EDITOR_JS = WEB_DIR / "js" / "tooltip-editor.js"

HTML_FRAGMENTS = [
    HTML_DIR / "left-panel.html",
    HTML_DIR / "viewer.html",
    HTML_DIR / "sidebar.html",
]

CSS_FILES = [
    CSS_DIR / "layout.css",
    CSS_DIR / "tools.css",
    CSS_DIR / "controls.css",
    CSS_DIR / "viewer.css",
]

CSS_LINKS = (
    '    <link rel="stylesheet" href="/static/css/layout.css" />',
    '    <link rel="stylesheet" href="/static/css/tools.css" />',
    '    <link rel="stylesheet" href="/static/css/controls.css" />',
    '    <link rel="stylesheet" href="/static/css/viewer.css" />',
)

JS_FILES = [
    POINTER_JS,
    LOGGING_JS,
    HISTORY_JS,
    COLORMAP_JS,
    UI_UTILS_JS,
    STATE_PERSISTENCE_JS,
    FILE_NAVIGATION_JS,
    PLUGIN_PANEL_JS,
    TOOLTIP_EDITOR_JS,
    BRUSH_JS,
    PAINTING_JS,
    INTERACTIONS_JS,
    APP_JS,
]

JS_STATIC_PATHS = (
    "/static/js/pointer-state.js",
    "/static/js/logging.js",
    "/static/js/history.js",
    "/static/js/colormap.js",
    "/static/js/ui-utils.js",
    "/static/js/state-persistence.js",
    "/static/js/file-navigation.js",
    "/static/js/plugin-panel.js",
    "/static/js/tooltip-editor.js",
    "/static/js/brush.js",
    "/static/js/painting.js",
    "/static/js/interactions.js",
    "/static/app.js",
)

_INDEX_HTML_CACHE: dict[str, object] = {"content": "", "mtime": None}
_LAYOUT_MARKUP_CACHE: dict[str, object] = {"markup": "", "mtimes": {}}
_INLINE_CSS_CACHE: dict[str, object] = {"text": "", "mtimes": {}}
_INLINE_JS_CACHE: dict[str, object] = {"text": "", "mtimes": {}}


def _file_cache_buster(path: Path) -> str:
    """Return a stable suffix that changes only when ``path`` does.

    Uses the file's mtime in nanoseconds so editing any tracked asset forces a
    fresh fetch but unchanged assets reuse the browser cache. Falls back to
    ``"0"`` if the file disappeared (broken deploy).
    """
    try:
        return str(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return "0"


def _bundle_cache_buster() -> str:
    """Hash-equivalent buster for the entire JS+CSS bundle.

    Cheap aggregate: sum mtimes of every tracked file. Changes when any
    bundled file changes, but doesn't require reading file contents.
    """
    total = 0
    for path in JS_FILES + CSS_FILES + HTML_FRAGMENTS + [INDEX_HTML]:
        try:
            total += path.stat().st_mtime_ns
        except FileNotFoundError:
            pass
    return str(total)

CAPTURE_LOG_SCRIPT = """<script>
(function(){
  if (window.__viewerLogPush) { return; }
  var queue = [];
  var endpoint = '/api/log';
  var maxBatch = 25;
  var flushTimer = null;
  function flush(){
    if (!queue.length) { return; }
    var payload = queue.slice();
    queue.length = 0;
    try {
      fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entries: payload })
      }).catch(function(){ });
    } catch (err) {
      console.warn('[log] flush failed', err);
    }
  }
  function schedule(){
    if (flushTimer) { return; }
    flushTimer = setTimeout(function(){ flushTimer = null; flush(); }, 300);
  }
  window.__viewerLogPush = function(kind, data){
    try {
      queue.push({ kind: kind, data: data, ts: Date.now() });
      if (queue.length >= maxBatch) {
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        flush();
      } else {
        schedule();
      }
    } catch (err) {
      console.warn('[log] push failed', err);
    }
  };
  window.addEventListener('error', function(evt){
    window.__viewerLogPush('JS_ERROR', {
      message: evt.message || '',
      filename: evt.filename || '',
      lineno: evt.lineno || 0,
      colno: evt.colno || 0,
      stack: evt.error && evt.error.stack ? String(evt.error.stack) : ''
    });
  });
})();
</script>"""

# Restore accent color before main JS runs to eliminate flash on reload.
RESTORE_ACCENT_SCRIPT = """<script>
(function(){
  try {
    var raw = localStorage.getItem('__viewer_accent');
    if (raw) {
      var a = JSON.parse(raw);
      if (a && a.c) {
        var s = document.documentElement.style;
        s.setProperty('--accent-color', a.c);
        if (a.h) s.setProperty('--accent-hover', a.h);
        if (a.k) s.setProperty('--accent-ink', a.k);
      }
    }
  } catch(_){}
  document.documentElement.style.opacity = '0';
  document.documentElement.style.transition = 'opacity 80ms ease-in';
})();
</script>"""

# Inline at the very top of <head> so the browser's first paint matches the
# OS appearance — eliminates the FOUC flash before the external CSS bundles
# load. Uses the CSS system colors ``Canvas`` and ``CanvasText`` (resolved by
# the browser at runtime to the OS's native window background and text
# colors). ``color-scheme: light dark`` tells the WebView the page supports
# both modes so those system colors resolve to the correct variant.
#
# Reference: https://www.w3.org/TR/css-color-4/#css-system-colors
EARLY_BACKGROUND_STYLE = """<style id="early-bg">
  :root { color-scheme: light dark; }
  html, body { margin: 0; height: 100%; background: Canvas; color: CanvasText; }
</style>"""


_PROBE_ORIGINS: list[str] = []


def set_trust_probe_origins(origins: list[str]) -> None:
    """Set alternate origins (e.g. IP, FQDN) the banner probes to detect trust.

    Called by :func:`ocdkit.viewer.app.run_server` once it knows what SANs
    the cert covers. The banner then attempts a cross-origin image load to
    one of these — if the load succeeds, the browser trusts the CA, and the
    banner stays hidden.
    """
    _PROBE_ORIGINS[:] = origins


_TRUST_BANNER_HTML = """<style>
  #ocdkit-trust-banner {
    display: none;
    position: fixed; top: 0; left: 0; right: 0; z-index: 99999;
    padding: 0.45rem 0.9rem;
    font: 13px/1.4 -apple-system, system-ui, sans-serif;
    background: var(--panel-surface);
    color: var(--panel-text-color);
    border-bottom: 1px solid var(--panel-border);
    backdrop-filter: blur(var(--control-blur));
    -webkit-backdrop-filter: blur(var(--control-blur));
    align-items: center; gap: 0.7rem;
  }
  #ocdkit-trust-banner .ocdkit-trust-prompt { opacity: 0.85; }
  #ocdkit-trust-banner #ocdkit-trust-install-btn {
    color: inherit; text-decoration: none; font-weight: 500;
    border: 1px solid var(--panel-border);
    padding: 0.2rem 0.55rem; border-radius: 4px;
  }
  #ocdkit-trust-banner #ocdkit-trust-install-btn:hover {
    background: var(--accent-color);
    color: var(--accent-ink, #161616);
    border-color: var(--accent-color);
  }
  #ocdkit-trust-banner #ocdkit-trust-status {
    margin-left: 0.5rem; font-size: 12px; opacity: 0;
    transition: opacity 0.2s;
  }
  #ocdkit-trust-banner .ocdkit-trust-other {
    color: inherit; opacity: 0.55; text-decoration: none;
    font-size: 12px; margin-left: 0.5rem;
  }
  #ocdkit-trust-banner #ocdkit-trust-dismiss {
    background: transparent; color: inherit; opacity: 0.55;
    border: 0; font-size: 18px; cursor: pointer; padding: 0 0.3rem;
  }
  #ocdkit-trust-banner #ocdkit-trust-dismiss:hover { opacity: 1; }
  .ocdkit-trust-toast {
    position: fixed; top: 0.6rem; right: 0.6rem; z-index: 99999;
    background: var(--panel-surface);
    color: var(--panel-text-color);
    border: 1px solid var(--panel-border);
    backdrop-filter: blur(var(--control-blur));
    -webkit-backdrop-filter: blur(var(--control-blur));
    padding: 0.6rem 0.9rem; border-radius: 6px;
    font: 13px/1.4 -apple-system, system-ui, sans-serif;
    max-width: 340px;
  }
</style>
<div id="ocdkit-trust-banner">
  <span class="ocdkit-trust-prompt">🔒 Browser doesn't trust this site?</span>
  <a id="ocdkit-trust-install-btn" href="/trust/install" download>
    Install trust certificate</a>
  <span id="ocdkit-trust-status"></span>
  <a class="ocdkit-trust-other" href="/trust/install">other formats…</a>
  <span style="flex: 1;"></span>
  <button id="ocdkit-trust-dismiss" aria-label="Dismiss">×</button>
</div>
<script>
  (function() {
    if (location.protocol !== "https:") return;
    const banner = document.getElementById("ocdkit-trust-banner");
    const btn = document.getElementById("ocdkit-trust-install-btn");
    const status = document.getElementById("ocdkit-trust-status");
    const xBtn = document.getElementById("ocdkit-trust-dismiss");
    if (!banner || !btn || !xBtn || !status) return;

    // Use style.display, not the `hidden` attribute. The banner uses inline
    // `display: flex` which would override the [hidden]{display:none} UA rule.
    function hide() { banner.style.display = "none"; }
    function show() { banner.style.display = "flex"; }
    function dismissForever() {
      try { localStorage.setItem("ocdkit-trust-dismissed", "1"); } catch (e) {}
      hide();
    }
    xBtn.addEventListener("click", dismissForever);

    // Per-OS install plan. Two styles:
    //   "download" = browser saves an installer file; OS handles the rest
    //                (Windows .reg, iOS .mobileconfig)
    //   "copy-cmd" = clipboard gets a self-contained shell one-liner with the
    //                cert embedded; user pastes into Terminal. No file
    //                download — avoids ~/Downloads filename guessing,
    //                Gatekeeper, and unsigned-mobileconfig SSL trust gap.
    const ua = navigator.userAgent;
    let plan;
    if (/iPhone|iPad/.test(ua)) {
      plan = {style: "download", url: "/trust/ca.mobileconfig",
              label: "Install trust profile",
              done: "Open Settings → Profile Downloaded to install."};
    } else if (/Mac OS X/.test(ua)) {
      plan = {style: "copy-cmd", cmdUrl: "/trust/install-cmd-macos.txt",
              label: "Install trust certificate",
              done: "📋 Command copied. Open Terminal (⌘+Space → 'Terminal') and paste."};
    } else if (/Windows/.test(ua)) {
      plan = {style: "download", url: "/trust/ca.reg",
              label: "Install trust certificate",
              done: "Double-click the downloaded file → click Yes."};
    } else if (/Linux|CrOS/.test(ua)) {
      plan = {style: "copy-cmd", cmdUrl: "/trust/install-cmd-linux.txt",
              label: "Install trust certificate",
              done: "📋 Command copied (Debian/Ubuntu). Open a terminal and paste."};
    } else {
      plan = {style: "download", url: "/trust/install",
              label: "Install trust certificate",
              done: ""};
    }

    if (plan.style === "download") {
      btn.href = plan.url;
      btn.setAttribute("download", "");
    } else {
      // copy-cmd: prevent default (don't navigate); just copy & paste
      btn.href = "#";
      btn.removeAttribute("download");
    }
    btn.textContent = plan.label;

    function showStatus(text) {
      status.textContent = text;
      status.style.opacity = "1";
    }

    function showToast(text, ms) {
      ms = ms || 5000;
      const t = document.createElement("div");
      t.className = "ocdkit-trust-toast";
      t.textContent = text;
      document.body.appendChild(t);
      setTimeout(function() { t.remove(); }, ms);
    }

    btn.addEventListener("click", function(ev) {
      // Set localStorage IMMEDIATELY and hide banner IMMEDIATELY. Status
      // shows as a toast so user sees acknowledgement without the banner
      // hanging around.
      try { localStorage.setItem("ocdkit-trust-dismissed", "1"); } catch (e) {}
      hide();

      if (plan.style === "copy-cmd") {
        ev.preventDefault();
        if (!navigator.clipboard) {
          showToast("⚠ Clipboard unavailable — open " + plan.cmdUrl + " manually");
          return;
        }
        fetch(plan.cmdUrl, {cache: "no-store"})
          .then(function(r) { return r.text(); })
          .then(function(cmd) { return navigator.clipboard.writeText(cmd); })
          .then(function() { showToast(plan.done, 8000); })
          .catch(function() {
            showToast("⚠ Couldn't copy. Open " + plan.cmdUrl + " manually");
          });
      } else if (plan.done) {
        showToast(plan.done, 8000);
      }
    });

    // Default: stay HIDDEN. Only show after we definitively determine trust
    // is missing (all cross-origin probes fail). If probes succeed (or are
    // unavailable), the banner never appears — no flash, no manual dismiss.
    //
    //   1. localStorage says dismissed → done, hidden
    //   2. No probe origins → can't confirm trust state → stay hidden
    //   3. Run probes:
    //        any onload → trust confirmed → hidden + localStorage set
    //        all onerror → trust not installed → show banner
    let dismissed = false;
    try { dismissed = !!localStorage.getItem("ocdkit-trust-dismissed"); } catch (e) {}
    if (dismissed) return;

    function trustConfirmed() {
      try { localStorage.setItem("ocdkit-trust-dismissed", "1"); } catch (e) {}
      hide();
    }

    const probes = (window.__OCDKIT_TRUST_PROBES__ || [])
      .filter(function(p) {
        try { return new URL(p).host !== location.host; } catch (e) { return false; }
      });
    if (probes.length === 0) return;  // can't auto-detect; stay hidden

    let pending = probes.length;
    let confirmed = false;
    probes.forEach(function(origin) {
      const img = new Image();
      img.onload = function() {
        if (confirmed) return;
        confirmed = true;
        trustConfirmed();
      };
      img.onerror = function() {
        if (confirmed) return;
        pending--;
        if (pending === 0) {
          // ALL probes failed → trust definitely not installed → show banner
          show();
        }
      };
      img.src = origin + "/trust/_pixel?_=" + Date.now();
    });
  })();
</script>"""


def _load_fragment(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("<!--"):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines)


def _get_index_template() -> str:
    global _INDEX_HTML_CACHE
    try:
        mtime = INDEX_HTML.stat().st_mtime_ns
    except FileNotFoundError:
        mtime = -1
    cached_content = _INDEX_HTML_CACHE.get("content")
    if cached_content and _INDEX_HTML_CACHE.get("mtime") == mtime:
        return cached_content  # type: ignore[return-value]
    content = INDEX_HTML.read_text(encoding="utf-8")
    _INDEX_HTML_CACHE = {"content": content, "mtime": mtime}
    return content


def _snapshot_mtimes(paths: Sequence[Path]) -> dict[str, int]:
    mtimes: dict[str, int] = {}
    for path in paths:
        try:
            mtimes[str(path)] = path.stat().st_mtime_ns
        except FileNotFoundError:
            mtimes[str(path)] = -1
    return mtimes


def _get_layout_markup() -> str:
    global _LAYOUT_MARKUP_CACHE
    mtimes = _snapshot_mtimes(HTML_FRAGMENTS)
    cached_markup = _LAYOUT_MARKUP_CACHE.get("markup")
    cached_mtimes = _LAYOUT_MARKUP_CACHE.get("mtimes")
    if cached_markup and cached_mtimes == mtimes:
        return cached_markup  # type: ignore[return-value]
    markup = "\n".join(_load_fragment(path) for path in HTML_FRAGMENTS)
    _LAYOUT_MARKUP_CACHE = {"markup": markup, "mtimes": mtimes}
    return markup


def _concat_cached_text(paths: Sequence[Path], cache: dict[str, object]) -> str:
    mtimes = _snapshot_mtimes(paths)
    cached_text = cache.get("text")
    if cached_text and cache.get("mtimes") == mtimes:
        return cached_text  # type: ignore[return-value]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    cache["text"] = text
    cache["mtimes"] = mtimes
    return text


def _prime_static_caches() -> None:
    if not WEB_DIR.exists():
        return
    try:
        _INDEX_HTML_CACHE["content"] = INDEX_HTML.read_text(encoding="utf-8")
        _INDEX_HTML_CACHE["mtime"] = INDEX_HTML.stat().st_mtime_ns
    except FileNotFoundError:
        _INDEX_HTML_CACHE["content"] = ""
        _INDEX_HTML_CACHE["mtime"] = None
    if all(p.exists() for p in HTML_FRAGMENTS):
        _LAYOUT_MARKUP_CACHE["markup"] = "\n".join(
            _load_fragment(path) for path in HTML_FRAGMENTS
        )
        _LAYOUT_MARKUP_CACHE["mtimes"] = _snapshot_mtimes(HTML_FRAGMENTS)
    if all(p.exists() for p in CSS_FILES):
        _INLINE_CSS_CACHE["text"] = "\n".join(
            path.read_text(encoding="utf-8") for path in CSS_FILES
        )
        _INLINE_CSS_CACHE["mtimes"] = _snapshot_mtimes(CSS_FILES)
    if all(p.exists() for p in JS_FILES):
        _INLINE_JS_CACHE["text"] = "\n\n".join(
            path.read_text(encoding="utf-8") for path in JS_FILES
        )
        _INLINE_JS_CACHE["mtimes"] = _snapshot_mtimes(JS_FILES)


_prime_static_caches()


def render_index(
    config: dict[str, object],
    *,
    inline_assets: bool,
    ui_mode: str = "browser",
) -> str:
    """Render index.html with config + asset hot-loading.

    ``ui_mode`` is "browser" (default) or "desktop". The desktop mode adds a
    ``data-ui="desktop"`` attribute on ``<html>`` so CSS can switch to a
    translucent background that lets pywebview's native vibrancy show through.
    """
    html = _get_index_template()
    layout_markup = _get_layout_markup()
    placeholder = '    <div id="app"></div>'
    if placeholder in html:
        html = html.replace(
            placeholder,
            f'    <div id="app">\n{layout_markup}\n    </div>',
        )
    # Tag the root element so CSS can react to the launch context.
    if ui_mode and ui_mode != "browser":
        html = html.replace(
            '<html lang="en">',
            f'<html lang="en" data-ui="{ui_mode}">',
            1,
        )
    # Inject the early-paint background style as the FIRST thing in <head>
    # so the browser's first paint is dark, not the default white. Eliminates
    # the FOUC flash before the external CSS bundles arrive.
    html = html.replace(
        "<head>",
        f"<head>\n    {EARLY_BACKGROUND_STYLE}",
        1,
    )
    # Trust install banner. The probe-origins <script> MUST come BEFORE the
    # banner JS so window.__OCDKIT_TRUST_PROBES__ is populated when the
    # banner IIFE reads it. We compose both in one replace so order is
    # explicit and not dependent on str.replace insertion semantics.
    banner_block = ""
    if _PROBE_ORIGINS:
        banner_block += (
            f'<script>window.__OCDKIT_TRUST_PROBES__ = '
            f'{json.dumps(_PROBE_ORIGINS)};</script>\n    '
        )
    banner_block += _TRUST_BANNER_HTML
    html = html.replace("<body>", f"<body>\n    {banner_block}", 1)
    # Honor OCDKIT_VIEWER_TITLE if set (without a circular import we just read
    # the env var here; app.py / launchers set it before render time).
    import os as _os
    configured_title = _os.environ.get("OCDKIT_VIEWER_TITLE")
    if configured_title:
        from html import escape as _html_escape
        safe = _html_escape(configured_title, quote=False)
        # Replace the literal default that ships in the index template.
        html = html.replace(
            "<title>ocdkit.viewer</title>",
            f"<title>{safe}</title>",
            1,
        )
    config_json = json.dumps(config).replace("</", "<\\/")
    debug_webgl = bool(config.get("debugWebgl"))
    config_script = (
        f"<script>window.__VIEWER_CONFIG__ = {config_json}; "
        f"window.__VIEWER_WEBGL_LOGGING__ = {json.dumps(debug_webgl)};</script>"
    )
    capture_script = "    " + CAPTURE_LOG_SCRIPT.strip().replace("\n", "\n    ")
    accent_script = "    " + RESTORE_ACCENT_SCRIPT.strip().replace("\n", "\n    ")
    css_links = list(CSS_LINKS)
    script_tag = '    <script src="/static/app.js"></script>'
    keep_order_comment = (
        "<!-- IMPORTANT: Viewer scripts must remain classic scripts in this order. "
        'Switching to type="module" breaks PyWebView image loading. -->'
    )

    if inline_assets:
        css_text = _concat_cached_text(CSS_FILES, _INLINE_CSS_CACHE)
        html = html.replace(css_links[0], f"    <style>{css_text}</style>")
        for link in css_links[1:]:
            html = html.replace(f"{link}\n", "")
            html = html.replace(link, "")
        js_bundle = _concat_cached_text(JS_FILES, _INLINE_JS_CACHE)
        bundled_script = (
            f"<script>\n/* {keep_order_comment[5:-4]} */\n{js_bundle}\n</script>"
        )
        bundled_script = "    " + bundled_script.replace("\n", "\n    ")
        html = html.replace(
            script_tag,
            "\n".join([config_script, accent_script, capture_script, bundled_script]),
        )
    else:
        # Per-file mtime-based suffix so editing one JS file invalidates only
        # that one in the browser cache. CSS_FILES is parallel to CSS_LINKS by
        # construction; JS_FILES is parallel to JS_STATIC_PATHS.
        for link, css_path in zip(css_links, CSS_FILES):
            html = html.replace(
                link, link.replace('.css"', f'.css?v={_file_cache_buster(css_path)}"')
            )
        script_parts = [
            config_script,
            accent_script,
            capture_script,
            f"    {keep_order_comment}",
        ]
        for url_path, fs_path in zip(JS_STATIC_PATHS, JS_FILES):
            script_parts.append(
                f'    <script src="{url_path}?v={_file_cache_buster(fs_path)}"></script>'
            )
        html = html.replace(script_tag, "\n".join(script_parts))
    return html


def build_html(
    config: Mapping[str, Any],
    *,
    inline_assets: bool = True,
    ui_mode: str = "browser",
) -> str:
    return render_index(dict(config), inline_assets=inline_assets, ui_mode=ui_mode)


def append_gui_log(message: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a", encoding="utf-8", errors="ignore") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass
