"""End-to-end repro of the undo/redo bug: the reported scenario + permutations,
driven through the REAL app.js → server path in headless Chrome.

Reported bug: draw the wrong colour, undo it, draw the right colour elsewhere —
when the new draw ends, the undone (bad) draw resurfaces. Root cause was a split
client/server history; undo/redo is now server-owned. This harness presses the
actual Ctrl+Z / Ctrl+Shift+Z and checks the SERVER volume + displayed slice.
"""
import subprocess
import sys
import time
import urllib.request

import numpy as np
import tifffile
from playwright.sync_api import sync_playwright

PORT = 8798
BASE = f"http://127.0.0.1:{PORT}"
VOL = "/tmp/volundo_vol.tif"
MASKS = "/tmp/volundo_vol_masks.tif"


def wait_up(timeout=40):
    for _ in range(timeout * 2):
        try:
            urllib.request.urlopen(BASE + "/", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def instance_max_at(pg, sid, z, y, x, w):
    r = pg.request.get(f"{BASE}/api/mask_slice/{sid}?z={z}&axis=0&kind=instance")
    a = np.frombuffer(r.body(), dtype=np.dtype(r.headers.get("x-mask-dtype", "uint8")))
    return int(a[y * w + x])


def stroke(pg, cx, cy, colour, r=4):
    """Paint `colour` (group) as a disk at (cx,cy) via the real stroke hook."""
    pg.evaluate(
        """([cx,cy,col,rr])=>{const W=window.__VIEWER_CONFIG__.width,idx=[];
        for(let dy=-rr;dy<=rr;dy++)for(let dx=-rr;dx<=rr;dx++)
          if(dx*dx+dy*dy<=rr*rr)idx.push((cy+dy)*W+(cx+dx));
        window.__volumeMode.setBrushDim(2); window.__onViewerStroke(idx,[col]);}""",
        [cx, cy, colour, r],
    )
    pg.wait_for_timeout(700)


def main():
    tifffile.imwrite(VOL, (np.random.default_rng(0).random((12, 60, 60)) * 1000).astype(np.uint16))
    lab = np.zeros((12, 60, 60), np.uint8)
    lab[:, 8:14, 8:52] = 1
    lab[:, 20:26, 8:52] = 2
    tifffile.imwrite(MASKS, lab)
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
            r = pg.request.post(BASE + "/api/open_image", data={"sessionId": sid, "path": VOL})
            assert r.ok, f"open_image {r.status}"
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__viewerMaskApplied === true", timeout=20000)
            pg.wait_for_timeout(800)
            W = pg.evaluate("window.__VIEWER_CONFIG__.width")
            z = pg.evaluate("window.__volumeMode.getSlice()")
            AX, AY = 14, 40     # region A (the 'bad' draw, empty space)
            BX, BY = 45, 40     # region B (the 'good' draw, empty space)

            def press_undo():
                pg.keyboard.press("Control+z"); pg.wait_for_timeout(600)

            def press_redo():
                pg.keyboard.press("Control+Shift+z"); pg.wait_for_timeout(600)

            # ---- the reported scenario ----
            stroke(pg, AX, AY, 2)                                  # bad colour at A
            bad_after_draw = instance_max_at(pg, sid, z, AY, AX, W) != 0
            press_undo()                                           # undo the bad draw
            bad_after_undo = instance_max_at(pg, sid, z, AY, AX, W) == 0
            stroke(pg, BX, BY, 3)                                  # good colour at B
            bad_after_good = instance_max_at(pg, sid, z, AY, AX, W) == 0   # MUST stay gone
            good_present = instance_max_at(pg, sid, z, BY, BX, W) != 0
            t1 = bad_after_draw and bad_after_undo and bad_after_good and good_present
            print(f"reported-bug scenario: bad drawn={bad_after_draw} undone={bad_after_undo} "
                  f"stays-gone-after-good-draw={bad_after_good} good-present={good_present} -> {'PASS' if t1 else 'FAIL'}")
            ok_all = ok_all and t1

            # ---- clean undo/redo cycle (region B drawn above) ----
            press_undo()                                          # undo the good draw
            good_gone = instance_max_at(pg, sid, z, BY, BX, W) == 0
            press_redo()                                          # redo it
            good_back = instance_max_at(pg, sid, z, BY, BX, W) != 0
            t2 = good_gone and good_back
            print(f"undo/redo cycle: undone={good_gone} redone={good_back} -> {'PASS' if t2 else 'FAIL'}")
            ok_all = ok_all and t2

            # ---- redo is truncated by a new edit ----
            press_undo()                                          # undo good draw again
            stroke(pg, 30, 50, 4)                                 # new edit forks history
            can_redo = pg.evaluate("window.__viewerVolumeCanRedo && window.__viewerVolumeCanRedo()")
            press_redo()                                          # should be a no-op
            good_still_gone = instance_max_at(pg, sid, z, BY, BX, W) == 0
            t3 = (not can_redo) and good_still_gone
            print(f"redo truncated by new edit: canRedo={can_redo} good-stays-gone={good_still_gone} -> {'PASS' if t3 else 'FAIL'}")
            ok_all = ok_all and t3

            # ---- undo button disabled at the bottom of history ----
            for _ in range(8):
                press_undo()
            can_undo = pg.evaluate("window.__viewerVolumeCanUndo && window.__viewerVolumeCanUndo()")
            btn_disabled = pg.evaluate("(document.getElementById('undoButton')||{}).disabled === true")
            t4 = (not can_undo) and btn_disabled
            print(f"undo exhausted: canUndo={can_undo} button-disabled={btn_disabled} -> {'PASS' if t4 else 'FAIL'}")
            ok_all = ok_all and t4

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
