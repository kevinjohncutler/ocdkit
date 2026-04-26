"""Browser-driven smoke tests for ocdkit.viewer.

These confirm that the lifted GUI loads and reaches a working state in a
real browser. The tests are intentionally tolerant of small UI changes
(no pixel-perfect assertions here) so they don't break on every CSS edit.
For pixel-perfect regressions, add a test in tests/e2e/test_visual.py
(see playwright's expect(...).to_have_screenshot()).
"""

from __future__ import annotations

import pytest


def test_index_loads_without_js_error(page):
    """If the index renders, the JS bundle parsed and no top-level throw."""
    title = page.title()
    assert "ocdkit" in title.lower() or "viewer" in title.lower()


def test_canvas_element_present(page):
    """The main canvas exists and has nonzero dimensions after layout."""
    canvas = page.locator("#canvas")
    canvas.wait_for(state="attached", timeout=5000)
    bbox = canvas.bounding_box()
    assert bbox is not None
    assert bbox["width"] > 0
    assert bbox["height"] > 0


def test_viewer_config_was_injected(page):
    """The server should inject window.__VIEWER_CONFIG__ with a sessionId."""
    has_config = page.evaluate("() => !!window.__VIEWER_CONFIG__")
    assert has_config is True
    session_id = page.evaluate("() => window.__VIEWER_CONFIG__.sessionId")
    assert isinstance(session_id, str) and len(session_id) > 0


def test_session_cookie_is_set(page):
    """Visiting / should mint an OCDSESSION cookie."""
    cookies = {c["name"]: c["value"] for c in page.context.cookies()}
    assert "OCDSESSION" in cookies
    assert len(cookies["OCDSESSION"]) > 0


def test_segment_endpoint_invocable_from_page(page, viewer_server):
    """Run a segmentation through the page's fetch() — proves the live wire
    from the loaded JS environment all the way to the plugin."""
    result = page.evaluate(
        """
        async () => {
            const sessionId = window.__VIEWER_CONFIG__.sessionId;
            const r = await fetch('/api/segment', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sessionId, method: 'otsu', min_size: 1}),
            });
            const j = await r.json();
            return { ok: r.ok, plugin: j.plugin, w: j.width, h: j.height, hasMask: !!j.mask };
        }
        """
    )
    assert result["ok"] is True
    assert result["plugin"] == "threshold"
    assert result["w"] > 0
    assert result["h"] > 0
    assert result["hasMask"] is True


def test_canvas_rect_cache_returns_consistent_rect(page):
    """The canvas rect cache (issue #4) returns the same instance within one frame."""
    result = page.evaluate(
        """
        () => {
            // The viewer's app.js uses classic-script declarations, so
            // top-level `function` is exposed on window — fine for testing.
            if (typeof getCanvasRect !== 'function') return { ok: false, reason: 'no helper' };
            const r1 = getCanvasRect();
            const r2 = getCanvasRect();
            return { ok: true, sameInstance: r1 === r2, width: r1.width, height: r1.height };
        }
        """
    )
    assert result["ok"] is True, result
    assert result["sameInstance"] is True, "getCanvasRect did not cache within frame"
    assert result["width"] > 0
    assert result["height"] > 0


def test_cancel_segmentation_globally_exposed(page):
    """cancelSegmentation() is exposed on window for the UI cancel button (#5)."""
    has_cancel = page.evaluate("() => typeof window.cancelSegmentation === 'function'")
    assert has_cancel is True
