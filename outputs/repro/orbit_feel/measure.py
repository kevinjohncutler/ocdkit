"""Measure how the 3D view responds to a mouse drag in the real viewer.

Drives a scripted drag (real mouse events) on the 3D canvas and records every
render: camera orientation, backing-store size, and time. Reports
  * lag       - how far the camera trails the input when the drag stops, and how
                many renders it takes to settle after the last input
  * momentum  - rotation that happens after the button is released
  * resizes   - backing-store size changes during the gesture (each one
                reallocates the compute target and can hitch)
  * frames    - render intervals during the drag
Usage: measure.py URL    (a running viewer that opens a volume)
"""
import json
import math
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790/"

HOOK = """(() => {
  const g = window.__volumeMode.gpu(); window.__log = [];
  const orig = g.render.bind(g);
  g.render = function () { const r = orig(); window.__log.push({t: performance.now(),
    q: Array.from(g.orient), w: g.canvas.width, h: g.canvas.height}); return r; };
})()"""


def angle(q1, q2):
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True,
                              args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"])
        pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
        pg.goto(URL, wait_until="load")
        pg.wait_for_function("window.__volumeMode !== undefined", timeout=30000)
        pg.eval_on_selector('[data-view="3d"]', "el => el.click()")
        pg.wait_for_function("window.__volumeMode.gpu() !== null", timeout=30000)
        time.sleep(1.5)
        pg.evaluate("document.querySelector('[data-view=\"3d\"]').blur()")
        pg.evaluate(HOOK)
        box = pg.query_selector("#volumeViewer").bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        pg.mouse.move(cx, cy)
        pg.mouse.down()
        t_start = pg.evaluate("performance.now()")
        steps, dx = 40, 8
        for i in range(steps):
            pg.mouse.move(cx + dx * (i + 1), cy)
            time.sleep(0.008)
        t_stop = pg.evaluate("performance.now()")
        time.sleep(0.25)                               # hold still, button down
        pg.mouse.up()
        t_up = pg.evaluate("performance.now()")
        time.sleep(1.2)
        log = pg.evaluate("window.__log")
        b.close()

    q_final = log[-1]["q"]
    during = [e for e in log if e["t"] <= t_stop]
    after_stop = [e for e in log if t_stop < e["t"] <= t_up]
    after_up = [e for e in log if e["t"] > t_up]
    q_at_stop = during[-1]["q"] if during else log[0]["q"]
    settle = next((i for i, e in enumerate(after_stop + after_up) if angle(e["q"], q_final) < 0.05), None)
    sizes = [(e["w"], e["h"]) for e in log]
    resizes = sum(1 for a, c in zip(sizes, sizes[1:]) if a != c)
    iv = [c["t"] - a["t"] for a, c in zip(during, during[1:])]
    q_up = (after_stop[-1] if after_stop else during[-1])["q"]
    out = {
        "renders_during_drag": len(during),
        "trail_deg_when_input_stops": round(angle(q_at_stop, q_final), 2),
        "renders_to_settle_after_input_stops": settle,
        "rotation_after_release_deg": round(angle(q_up, q_final), 2),
        "resolution_changes": resizes,
        "min_backing_px": min(w * h for w, h in sizes), "max_backing_px": max(w * h for w, h in sizes),
        "median_render_interval_ms": round(sorted(iv)[len(iv) // 2], 1) if iv else None,
        "total_rotation_deg": round(angle(log[0]["q"], q_final), 1),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
