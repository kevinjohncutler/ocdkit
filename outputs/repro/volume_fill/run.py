"""End-to-end check of the 3D colour picker + 3D fill (merge / delete) through the
real app.js → server path in headless Chrome.

Scenario: pick cell 1, fill cell 3 → cell 3 merges into cell 1; then erase-fill
cell 2 → cell 2 is deleted. Verified against the SERVER instance volume.
"""
import subprocess
import sys
import time
import urllib.request

import numpy as np
import tifffile
from playwright.sync_api import sync_playwright

PORT = 8797
BASE = f"http://127.0.0.1:{PORT}"
VOL = "/tmp/volfill_vol.tif"
MASKS = "/tmp/volfill_vol_masks.tif"


def wait_up(timeout=40):
    for _ in range(timeout * 2):
        try:
            urllib.request.urlopen(BASE + "/", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    shape = (10, 30, 40)
    tifffile.imwrite(VOL, (np.random.default_rng(0).random(shape) * 1000).astype(np.uint16))
    m = np.zeros(shape, np.uint8)
    m[:, 4:10, 4:36] = 1
    m[:, 14:20, 4:36] = 2
    m[:, 24:28, 4:36] = 3
    tifffile.imwrite(MASKS, m)
    import os as _os
    _ed = MASKS.replace(".tif", "_edited.tif")
    if _os.path.exists(_ed):
        _os.remove(_ed)

    srv = subprocess.Popen(
        [sys.executable, "-c",
         f"import uvicorn; uvicorn.run('ocdkit.viewer.app:create_app', factory=True, "
         f"host='127.0.0.1', port={PORT}, log_level='warning')"],
    )
    ok_all = True
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
            assert pg.request.post(BASE + "/api/open_image", data={"sessionId": sid, "path": VOL}).ok
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__viewerMaskApplied === true", timeout=20000)
            pg.wait_for_timeout(800)

            def count(label):
                r = pg.request.get(f"{BASE}/api/mask_slice/{sid}?z=5&axis=0&kind=instance")
                a = np.frombuffer(r.body(), dtype=np.dtype(r.headers.get("x-mask-dtype", "uint8")))
                return int((a == label).sum())

            assert count(1) > 0 and count(2) > 0 and count(3) > 0, "setup masks missing"

            # --- pick cell 1 (y=6,x=20), then fill cell 3 (y=26,x=20) → merge 3 into 1 ---
            pg.evaluate("([x,y])=>window.__viewerVolumePick(x,y)", [20, 6])
            pg.wait_for_timeout(400)
            pg.evaluate("([x,y])=>window.__viewerVolumeFill(x,y)", [20, 26])
            pg.wait_for_timeout(900)
            merged = count(3) == 0 and count(1) > 0
            print(f"merge (pick cell1, fill cell3): cell3 gone={count(3) == 0} cell1 kept={count(1) > 0} -> {'PASS' if merged else 'FAIL'}")
            ok_all = ok_all and merged

            # --- erase-fill cell 2 (y=16,x=20) → delete the whole cell ---
            pg.keyboard.down("e")           # hold erase (startEraseOverride)
            pg.wait_for_timeout(150)
            pg.evaluate("([x,y])=>window.__viewerVolumeFill(x,y)", [20, 16])
            pg.wait_for_timeout(900)
            pg.keyboard.up("e")
            deleted = count(2) == 0
            print(f"delete (erase-fill cell2): cell2 gone={deleted} -> {'PASS' if deleted else 'FAIL'}")
            ok_all = ok_all and deleted

            # --- undo brings cell 2 back (server-owned history) ---
            pg.keyboard.press("Control+z")
            pg.wait_for_timeout(700)
            restored = count(2) > 0
            print(f"undo restores cell2: {restored} -> {'PASS' if restored else 'FAIL'}")
            ok_all = ok_all and restored

            print("JS errors:", errs or "none")
            ok_all = ok_all and not errs
            ctx.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
