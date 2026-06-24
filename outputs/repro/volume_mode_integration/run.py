"""Integration check: the embedded 3D view-mode in the MAIN ocdkit viewer.

Launches a fresh viewer server, loads a synthetic volume into a session, opens
the main app page, switches to the 3D mode, and verifies VolumeGPU mounts and
renders a non-blank frame — all in real headless Chrome (the only browser tier
that exposes WebGPU).
"""
import json
import subprocess
import sys
import time
import urllib.request

import numpy as np
import tifffile
from playwright.sync_api import sync_playwright

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
VOL = "/tmp/volmode_integration_vol.tif"


def wait_up(timeout=40):
    for _ in range(timeout * 2):
        try:
            urllib.request.urlopen(BASE + "/", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    tifffile.imwrite(VOL, (np.random.default_rng(0).random((20, 80, 80)) * 1000).astype(np.uint16))
    srv = subprocess.Popen(
        [sys.executable, "-c",
         f"import uvicorn; uvicorn.run('ocdkit.viewer.app:create_app', factory=True, "
         f"host='127.0.0.1', port={PORT}, log_level='warning')"],
    )
    try:
        if not wait_up():
            print("SERVER FAILED TO START"); return 1
        with sync_playwright() as p:
            ctx = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"])
            pg = ctx.new_page()
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__VIEWER_CONFIG__ && window.__VIEWER_CONFIG__.sessionId", timeout=15000)
            sid = pg.evaluate("window.__VIEWER_CONFIG__.sessionId")
            # load the volume into this session, then reload so CONFIG.isVolume flips
            r = pg.request.post(BASE + "/api/open_image", data={"sessionId": sid, "path": VOL})
            assert r.ok, f"open_image {r.status}"
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__volumeMode !== undefined", timeout=15000)

            is_vol = pg.evaluate("window.__VIEWER_CONFIG__.isVolume")
            bar_visible = pg.eval_on_selector("#viewModeBar", "el => !el.hidden")
            print("isVolume:", is_vol, "| viewModeBar visible:", bar_visible)

            # switch to 3D and let it mount + render
            pg.eval_on_selector('[data-view="3d"]', "el => el.click()")
            pg.wait_for_function("window.__volumeMode.gpu() !== null", timeout=20000)
            pg.wait_for_timeout(600)

            gpu_ok = pg.evaluate("window.__volumeMode.gpu() !== null")
            canvas_shown = pg.eval_on_selector("#volumeViewer", "el => !el.hidden")
            # non-blank render: sample the canvas
            shot = pg.query_selector("#volumeViewer").screenshot(path="/tmp/volmode_3d.png")
            from skimage import io as skio
            img = skio.imread("/tmp/volmode_3d.png")[..., :3].astype(np.float32)
            nonblank = float(img.std())
            print("gpu mounted:", gpu_ok, "| 3D canvas shown:", canvas_shown,
                  "| render std:", round(nonblank, 2))
            print("JS errors:", errs or "none")

            ok = (is_vol and bar_visible and gpu_ok and canvas_shown
                  and nonblank > 1.0 and not errs)
            print("RESULT:", "PASS" if ok else "FAIL")
            ctx.close()
            return 0 if ok else 1
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
