#!/usr/bin/env python
"""Native-webview harness for the HDR colormap prototype.

Opens ``hdr_cmap_prototype.html`` in pywebview's macOS WKWebView and pumps the
LIVE EDR headroom — ``NSScreen.maximumExtendedDynamicRangeColorComponentValue``,
the value the browser deliberately hides from JS — into the page as
``window.__edrHeadroom`` every 0.5 s. The page's ``detectHeadroom()`` already
consumes that global, so the colormap re-adapts to your real display headroom and
follows brightness changes, never clipping.

This also answers the open question: does WKWebView actually EDR-COMPOSITE the
WebGPU canvas inside a third-party app? If the lifted panel glows brighter than
SDR white here, it does. If it looks flat/SDR, the host window needs an EDR
opt-in (deeper PyObjC work) even though the headroom number reads fine.

Run:  python scripts/hdr_harness.py
(needs:  pip install pywebview pyobjc-framework-Cocoa)
"""
import functools
import http.server
import os
import socketserver
import sys
import threading
import time

# Make the in-repo package importable so we dogfood the shared bridge helper.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ocdkit.viewer.edr_bridge import start_edr_pump  # noqa: E402

# Serve the package root so BOTH /viewer/web/ and the shared /plot/web/ modules
# resolve over HTTP.
WEB_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "ocdkit"))
PORT = 8013
URL = f"http://127.0.0.1:{PORT}/viewer/web/hdr_cmap_prototype.html"


def _serve():
    """Serve the package root so /viewer/web/ and /plot/web/ both resolve
    (self-contained — no dependency on a separately-running server)."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB_DIR)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        httpd.serve_forever()


def main():
    try:
        import webview
    except ImportError:
        raise SystemExit("pywebview not installed — run: pip install pywebview")

    threading.Thread(target=_serve, daemon=True).start()
    time.sleep(0.3)
    print(f"[harness] serving {WEB_DIR}\n[harness] opening {URL}")
    print("[harness] move the window onto your HDR display; change screen "
          "brightness and watch the in-page 'reported headroom' track it.")
    win = webview.create_window(
        "ocdkit HDR colormap — EDR-pumped", URL, width=1120, height=860)
    # debug=True enables the WKWebView inspector + context menu (right-click →
    # Reload, or ⌘R) so edits to the served files reload in place.
    webview.start(
        lambda: start_edr_pump(win, on_value=lambda hr: print(f"[edr] headroom = {hr:.3f}x  -> window.__edrHeadroom")),
        debug=True)


if __name__ == "__main__":
    main()
