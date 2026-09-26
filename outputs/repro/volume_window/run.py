"""The 2D histogram window must drive the 3D view.

Opens the real viewer, switches to 3D, then drags the histogram's UPPER handle
with the real mouse (not a JS call) and checks that the 3D render changes in
every projection mode. Screenshots land next to this script.
Usage: run.py URL
"""
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright
from skimage import io as skio

HERE = Path(__file__).resolve().parent
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790/"


def shot(pg, name):
    p = str(HERE / name)
    pg.query_selector("#volumeViewer").screenshot(path=p)
    return skio.imread(p)[..., :3].astype(np.float32)


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True,
                          args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"])
    pg = b.new_page(viewport={"width": 1400, "height": 900})
    errs = []
    pg.on("pageerror", lambda e: errs.append(f"{e}\n{getattr(e, 'stack', '')}"))
    pg.goto(URL, wait_until="load")
    pg.wait_for_function("window.__volumeMode !== undefined", timeout=30000)
    pg.eval_on_selector('[data-view="3d"]', "el => el.click()")
    pg.wait_for_function("window.__volumeMode.gpu() !== null", timeout=30000)
    time.sleep(1.0)
    ok = True
    for proj, name in ((1, "MIP"), (2, "mean"), (0, "EA")):
        # reload so every mode starts from the image's default window; the viewer
        # reopens 3D by itself (this also exercises applying the window at open)
        pg.goto(URL, wait_until="load")
        pg.wait_for_function("window.__volumeMode && window.__volumeMode.gpu() !== null", timeout=30000)
        pg.evaluate(f"window.__volumeMode.setProj({proj})")
        time.sleep(0.8)
        w0 = pg.evaluate("window.__viewerGetWindow()")
        before = shot(pg, f"window_{name}_before.png")
        box = pg.query_selector("#histogram").bounding_box()
        x_hi = box["x"] + box["width"] * w0[1] / 255.0
        y = box["y"] + box["height"] * 0.5
        pg.mouse.move(x_hi, y)
        pg.mouse.down()
        pg.mouse.move(x_hi - box["width"] * 0.35, y, steps=8)   # pull the upper bound down
        pg.mouse.up()
        time.sleep(0.5)
        w1 = pg.evaluate("window.__viewerGetWindow()")
        after = shot(pg, f"window_{name}_after.png")
        changed = int((np.abs(after - before).max(-1) > 2).sum())
        print(f"{name}: window {w0} -> {w1}, 3D pixels changed: {changed}")
        ok &= w1[1] < w0[1] and changed > 1000
    print("errors:", errs)
    print("RESULT:", "PASS" if ok and not errs else "FAIL")
    b.close()
