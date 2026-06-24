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
MASKS = "/tmp/volmode_integration_vol_masks.tif"  # *_masks sidecar → auto-loads


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
    # label volume (a few blobs) as a *_masks sidecar so it auto-loads
    lab = np.zeros((20, 80, 80), np.uint8)
    zz, yy, xx = np.mgrid[0:20, 0:80, 0:80]
    lab[((xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 10) ** 2) < 9 ** 2] = 1
    lab[((xx - 55) ** 2 + (yy - 55) ** 2 + (zz - 10) ** 2) < 8 ** 2] = 2
    tifffile.imwrite(MASKS, lab)
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
            bar_visible = pg.eval_on_selector("#viewModePanel", "el => !el.hidden")
            print("isVolume:", is_vol, "| View pane visible:", bar_visible)

            # volumetric mask overlays the 2D view (auto-loaded sidecar)
            pg.wait_for_function("window.__viewerMaskApplied === true", timeout=10000)
            mask_2d = pg.evaluate("window.__viewerMaskApplied")
            print("2D mask applied:", mask_2d)

            # 2.5D slice scrub: arrow-key / slider changes the displayed slice
            z0 = pg.evaluate("window.__volumeMode.getSlice()")
            pg.evaluate("window.__volumeMode.showSlice(window.__volumeMode.getSlice() + 5)")
            pg.wait_for_timeout(300)
            z1 = pg.evaluate("window.__volumeMode.getSlice()")
            slice_changed = (z1 == z0 + 5)
            print("slice scrub:", z0, "->", z1, "ok:", slice_changed)

            # edit persistence: a 2D stroke writes the LABEL to the server volume
            zc = pg.evaluate("window.__volumeMode.getSlice()")
            pg.evaluate("""(function(){const W=window.__VIEWER_CONFIG__.width,H=window.__VIEWER_CONFIG__.height,
              cx=W>>1,cy=H>>1,idx=[]; for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++)
              if(dx*dx+dy*dy<=16)idx.push((cy+dy)*W+(cx+dx));
              window.__volumeMode.setBrushDim(2); window.__onViewerStroke(idx,[222]);})()""")
            pg.wait_for_timeout(900)
            _r = pg.request.get(f"{BASE}/api/mask_slice/{sid}?z={zc}&axis=0&kind=instance")
            _a = np.frombuffer(_r.body(), dtype=np.dtype(_r.headers.get("x-mask-dtype", "uint8")))
            edit_persisted = int(_a.max()) >= 222
            print("edit persisted (2D stroke -> server volume):", edit_persisted)

            # orthogonal slicing: switch to Y axis → dims change, slider re-ranges
            pg.evaluate("window.__volumeMode.setAxis(1)")
            pg.wait_for_timeout(700)   # reinit + new image decode
            ax = pg.evaluate("window.__volumeMode.getAxis()")
            smax = pg.eval_on_selector("#sliceSlider", "el => parseInt(el.max, 10)")
            vshape = pg.evaluate("window.__VIEWER_CONFIG__.volumeShape")
            axis_ok = (ax == 1 and smax == vshape[1] - 1)
            print("axis switch -> Y:", ax, "| slider max:", smax, "(expect", vshape[1] - 1, ") ok:", axis_ok)
            pg.evaluate("window.__volumeMode.setAxis(0)")   # back to Z for the 3D test
            pg.wait_for_timeout(500)

            # switch to 3D and let it mount + render
            pg.eval_on_selector('[data-view="3d"]', "el => el.click()")
            pg.wait_for_function("window.__volumeMode.gpu() !== null", timeout=20000)
            pg.wait_for_timeout(600)

            # H key homes the 3D camera: move target, press H, expect reset to ~origin
            pg.evaluate("var g=window.__volumeMode.gpu(); g.target=[12,12,12]; g.render();")
            pg.evaluate("window.dispatchEvent(new KeyboardEvent('keydown', {key:'h'}))")
            pg.wait_for_timeout(150)
            tgt = pg.evaluate("window.__volumeMode.gpu().target")
            home_ok = all(abs(v) < 1.0 for v in tgt)
            print("H home reset:", [round(v, 2) for v in tgt], "ok:", home_ok)

            gpu_ok = pg.evaluate("window.__volumeMode.gpu() !== null")
            canvas_shown = pg.eval_on_selector("#volumeViewer", "el => !el.hidden")
            # masks auto-loaded from the sidecar → VolumeGPU shows labels
            show_labels = pg.evaluate("window.__volumeMode.gpu().showLabels")
            # projection switch (MIP→Mean)
            pg.evaluate("window.__volumeMode.setProj(2)")
            pg.wait_for_timeout(200)
            proj = pg.evaluate("window.__volumeMode.getProj()")
            print("masks shown (showLabels):", show_labels, "| projection after setProj(2):", proj)
            # non-blank render: sample the canvas
            shot = pg.query_selector("#volumeViewer").screenshot(path="/tmp/volmode_3d.png")
            from skimage import io as skio
            img = skio.imread("/tmp/volmode_3d.png")[..., :3].astype(np.float32)
            nonblank = float(img.std())
            print("gpu mounted:", gpu_ok, "| 3D canvas shown:", canvas_shown,
                  "| render std:", round(nonblank, 2))
            print("JS errors:", errs or "none")

            ok = (is_vol and bar_visible and mask_2d and slice_changed and edit_persisted
                  and axis_ok and home_ok and gpu_ok and canvas_shown and show_labels == 1
                  and proj == 2 and nonblank > 1.0 and not errs)
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
