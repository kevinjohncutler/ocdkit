"""Feed the live macOS EDR (HDR) headroom into a native-webview page.

Browsers deliberately hide the display's HDR headroom from JavaScript (it leaks
viewing conditions — a fingerprinting vector). But a *native* host can read it:
on macOS, ``NSScreen.maximumExtendedDynamicRangeColorComponentValue`` is the
current headroom (a multiple of SDR white, 1.0 = SDR, brightness-dependent). A
pywebview / WKWebView app can read it via PyObjC and inject it into the page.

The shared ``plot/web/hdr_headroom.js`` (``HdrHeadroom``) consumes the injected
``window.__edrHeadroom``, so an HdrColormapRenderer then drives the colormap peak
to the real, live display headroom — never clipping, re-adapting as the user
changes screen brightness.

Usage with any pywebview window::

    import webview
    from ocdkit.viewer.edr_bridge import start_edr_pump
    win = webview.create_window("…", url)
    webview.start(lambda: start_edr_pump(win))

Generic and macOS-specific only in ``read_edr_headroom`` — on other platforms it
returns ``None`` and the page falls back to a fixed headroom.
"""
from __future__ import annotations

import threading
import time


def read_edr_headroom():
    """Current EDR headroom of the main screen as a multiple of SDR white, or
    ``None`` if unavailable (non-macOS, no PyObjC, or no EDR display)."""
    try:
        from AppKit import NSScreen
    except Exception:
        return None
    try:
        scr = NSScreen.mainScreen()
        if scr is None:
            return None
        return float(scr.maximumExtendedDynamicRangeColorComponentValue())
    except Exception:
        return None


def start_edr_pump(window, interval: float = 0.5, on_value=None):
    """Poll the EDR headroom and inject it into ``window`` as
    ``window.__edrHeadroom`` every ``interval`` seconds (only when it changes).

    ``window`` is a pywebview Window (anything with ``evaluate_js(str)``).
    ``on_value(hr)`` is an optional callback (e.g. to log). Returns the daemon
    thread. If the platform can't report a headroom the thread idles harmlessly
    and the page keeps its fallback.
    """
    def _run():
        last = None
        while True:
            hr = read_edr_headroom()
            if hr is not None and hr != last:
                try:
                    window.evaluate_js(f"window.__edrHeadroom = {hr};")
                except Exception:
                    pass
                if on_value is not None:
                    try:
                        on_value(hr)
                    except Exception:
                        pass
                last = hr
            time.sleep(interval)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
