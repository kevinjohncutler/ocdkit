"""Headless Playwright smoke test for the 2.5D volume-viewer page.

canvas2d works in Playwright's bundled Chromium (unlike WebGPU), so the 2.5D
slice path is genuinely browser-verifiable. Injects a synthetic bundle (built by
omnipose _volume3d) via window.__TEST_BUNDLE__, loads volume.html over file://,
and asserts the canvas renders, slices navigate, layers switch, and overlays draw.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

from omnipose.gui import _volume3d as v3

HTML = "/Volumes/DataDrive/ocdkit/src/ocdkit/viewer/web/volume.html"

_CANVAS_SUM = """() => { const c=document.getElementById('stage'); const x=c.getContext('2d');
  const d=x.getImageData(0,0,c.width,c.height).data; let s=0;
  for (let i=0;i<d.length;i+=4) s += d[i]+d[i+1]+d[i+2]; return s; }"""


def _synth_bundle():
    D, H, W = 6, 40, 40
    m = np.zeros((D, H, W), np.int32)
    for d in range(D):
        cy = 8 + d * 3
        m[d, cy:cy + 6, 8:14] = 1
        m[d, 24:30, 10 + d:16 + d] = 2
    raw = (np.random.default_rng(1).random((D, H, W)) * 1000).astype(np.uint16)
    return v3.build_bundle(raw, m, edges=[[1, 2]], use_gpu=False)


@pytest.mark.skipif(not os.path.exists(HTML), reason="volume.html missing")
def test_volume_page_renders_and_navigates():
    bundle_js = json.dumps(_synth_bundle())
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:  # browser not installed
            pytest.skip(f"chromium unavailable: {e}")
        page = browser.new_page()
        errs = []
        page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.add_init_script(f"window.__TEST_BUNDLE__ = {bundle_js};")
        page.goto("file://" + HTML)

        try:
            page.wait_for_function("window.__vv !== undefined", timeout=15000)
        except Exception:
            browser.close()
            raise AssertionError("view never initialized; status=%r errors=%r"
                                 % (page.text_content("#status") if not page.is_closed() else "?", errs))

        assert "loaded" in (page.text_content("#status") or ""), page.text_content("#status")

        # base render is non-blank
        assert page.evaluate(_CANVAS_SUM) > 0

        # navigating slices changes the image (volume varies across z)
        page.evaluate("window.__vv.setSlice(0)")
        s_first = page.evaluate(_CANVAS_SUM)
        page.evaluate("window.__vv.setSlice(window.__vv.D - 1)")
        s_last = page.evaluate(_CANVAS_SUM)
        assert s_first != s_last

        # every layer renders without error
        for layer in ("flow", "distance", "mask", "image"):
            page.evaluate(f"window.__vv.setLayer('{layer}')")
            assert page.evaluate(_CANVAS_SUM) > 0

        # overlays draw pixels on top of the mask layer
        page.evaluate("window.__vv.setLayer('mask'); window.__vv.setOverlay('trajectories', false); window.__vv.setOverlay('affinity', false)")
        base = page.evaluate(_CANVAS_SUM)
        page.evaluate("window.__vv.setOverlay('affinity', true)")
        with_aff = page.evaluate(_CANVAS_SUM)
        page.evaluate("window.__vv.setOverlay('trajectories', true)")
        with_traj = page.evaluate(_CANVAS_SUM)
        assert with_aff != base, "affinity overlay drew nothing"
        assert with_traj != with_aff, "trajectory overlay drew nothing"

        assert not errs, "JS errors: " + "; ".join(errs)
        browser.close()
