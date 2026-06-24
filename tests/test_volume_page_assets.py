"""Phase B: the main viewer page ships the 3D volume canvas + modules."""
from ocdkit.viewer.assets import build_html


def _cfg(**kw):
    base = {
        "sessionId": "s1", "width": 40, "height": 40,
        "isVolume": True, "volumeDepth": 10, "currentSlice": 5,
    }
    base.update(kw)
    return base


def test_page_has_volume_canvas_and_toggle():
    html = build_html(_cfg(), inline_assets=False)
    assert 'id="volumeViewer"' in html
    assert 'id="viewModeBar"' in html
    assert 'data-view="3d"' in html


def test_page_loads_volume_modules_after_app():
    html = build_html(_cfg(), inline_assets=False)
    for src in ("mat4.js", "volume3d.js", "volume3d-view.js",
                "volume3d-overlays.js", "volume3d-overlays-gpu.js",
                "volume3d-gpu.js", "volume-mode.js"):
        assert f"/static/js/{src}" in html, src
    # volume-mode.js must load after app.js (needs __VIEWER_CONFIG__ + DOM)
    assert html.index("/static/app.js") < html.index("/static/js/volume-mode.js")


def test_inline_bundle_includes_volume_mode():
    html = build_html(_cfg(), inline_assets=True)
    assert 'id="volumeViewer"' in html
    assert "volume-mode.js" in html  # provenance comment is inlined with the bundle


def test_canvas_present_even_for_2d_image():
    # The canvas/toggle are static markup; volume-mode.js hides them for 2D.
    html = build_html(_cfg(isVolume=False), inline_assets=False)
    assert 'id="volumeViewer"' in html
