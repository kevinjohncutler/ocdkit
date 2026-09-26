"""The clip row: a double-ended percentile slider that sets the display window."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

GEOMETRY = """() => {
  const root = document.querySelector('[data-slider-id=clipPercentile]').getBoundingClientRect();
  const thumbs = [...document.querySelectorAll('[data-slider-id=clipPercentile] .slider-thumb')]
    .map((t) => { const b = t.getBoundingClientRect(); return [b.left, b.right]; });
  return { root: [root.left, root.right], thumbs };
}"""


def _set(page, lo, hi):
    page.fill("#clipLoInput", str(lo))
    page.press("#clipLoInput", "Enter")
    page.fill("#clipHiInput", str(hi))
    page.press("#clipHiInput", "Enter")
    page.wait_for_timeout(300)
    return page.evaluate("() => window.__viewerGetWindow()")


def test_clip_percentiles_set_the_window(page):
    page.wait_for_function("() => typeof window.__viewerGetWindow === 'function'")
    full = _set(page, 0, 100)
    narrow = _set(page, 5, 95)
    assert full[0] <= narrow[0] < narrow[1] <= full[1]
    assert narrow != full
    # the slider mirrors the typed values and they persist with the histogram options
    assert page.input_value("#clipLoInput") == "5"
    assert page.input_value("#clipHiInput") == "95"
    prefs = page.evaluate("() => JSON.parse(localStorage.getItem('ocdkit-histogram-prefs'))")
    assert prefs["clipLo"] == 5 and prefs["clipHi"] == 95


def test_clip_thumbs_stay_inside_the_track(page):
    _set(page, 0, 100)
    g = page.evaluate(GEOMETRY)
    left, right = g["root"]
    for a, b in g["thumbs"]:
        assert left - 0.5 <= a and b <= right + 0.5, (g, "thumb overhangs the track")


def test_clip_thumb_follows_the_pointer(page):
    _set(page, 1, 99)
    box = page.locator("[data-slider-id=clipPercentile]").bounding_box()
    y = box["y"] + box["height"] / 2
    a, b = page.evaluate(GEOMETRY)["thumbs"][1]
    start = (a + b) / 2
    page.mouse.move(start, y)
    page.mouse.down()
    page.mouse.move(start - 40, y, steps=4)
    a, b = page.evaluate(GEOMETRY)["thumbs"][1]
    page.mouse.up()
    assert abs((a + b) / 2 - (start - 40)) <= 1.5
    assert float(page.input_value("#clipHiInput")) < 99
