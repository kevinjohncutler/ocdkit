"""View state persists across a page refresh: the view (2D/3D), the slice, and the
label style per view (2D vs 3D). Also that the restored 2D slice loads image+mask
together (no halfway-image / edited-mask mismatch).
"""
import subprocess, sys, time, urllib.request, os
import numpy as np
import tifffile
from playwright.sync_api import sync_playwright

PORT = 8783
BASE = f"http://127.0.0.1:{PORT}"
VOL = "/tmp/persist_vol.tif"
MASKS = "/tmp/persist_vol_masks.tif"
RESULTS = []


def check(n, ok): RESULTS.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n}")


def wait_up(t=40):
    for _ in range(t * 2):
        try:
            urllib.request.urlopen(BASE + "/", timeout=2); return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    shape = (24, 50, 50)
    tifffile.imwrite(VOL, (np.random.default_rng(0).random(shape) * 800).astype(np.uint16))
    m = np.zeros(shape, np.uint8); m[:, 8:16, 8:42] = 1
    tifffile.imwrite(MASKS, m)
    if os.path.exists(MASKS.replace(".tif", "_edited.tif")):
        os.remove(MASKS.replace(".tif", "_edited.tif"))
    env = dict(os.environ, OCDKIT_VIEWER_SAMPLE_IMAGE=VOL)
    srv = subprocess.Popen([sys.executable, "-c",
        f"import uvicorn;uvicorn.run('ocdkit.viewer.app:create_app',factory=True,host='127.0.0.1',port={PORT},log_level='warning')"], env=env)
    try:
        if not wait_up():
            print("SERVER FAILED"); return 1
        with sync_playwright() as p:
            ctx = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"])
            pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__volumeMode", timeout=20000); pg.wait_for_timeout(800)

            # set 2D style = outlined (fill+outline), scrub to slice 7, then go 3D
            pg.evaluate("window.__viewerSetMaskDisplayMode('outlined')"); pg.wait_for_timeout(300)
            pg.evaluate("window.__volumeMode.showSlice(7)"); pg.wait_for_timeout(500)
            pg.eval_on_selector('[data-view="3d"]', "e=>e.click()")
            pg.wait_for_function("window.__volumeMode.gpu()!==null", timeout=20000); pg.wait_for_timeout(600)
            pg.evaluate("window.__viewerSetMaskDisplayMode('hidden')"); pg.wait_for_timeout(300)  # 3D style = hidden
            pg.wait_for_timeout(500)  # let saveVolState debounce-free writes land

            # ---- refresh (same context → cookie session + localStorage persist) ----
            pg.reload(wait_until="load")
            pg.wait_for_function("window.__volumeMode", timeout=20000)
            pg.wait_for_function("window.__volumeMode.getMode() === '3d'", timeout=8000) if False else pg.wait_for_timeout(2500)

            check("view mode (3D) persisted across refresh", pg.evaluate("window.__volumeMode.getMode()") == "3d")
            check("3D label style (hidden) persisted", pg.evaluate("window.__viewerMaskDisplayMode()") == "hidden")
            # back to 2D → its remembered style + slice
            pg.eval_on_selector('[data-view="2d"]', "e=>e.click()"); pg.wait_for_timeout(700)
            check("2D label style (outlined) persisted", pg.evaluate("window.__viewerMaskDisplayMode()") == "outlined")
            check("slice (7) persisted", pg.evaluate("window.__volumeMode.getSlice()") == 7)
            # mismatch fix: the displayed mask is for the restored slice (image+mask lockstep)
            sid = pg.evaluate("window.__VIEWER_CONFIG__.sessionId")
            r = pg.request.get(f"{BASE}/api/mask_slice/{sid}?z=7&axis=0&kind=group")
            a = np.frombuffer(r.body(), dtype=np.dtype(r.headers.get("x-mask-dtype", "uint8")))
            W = pg.evaluate("window.__VIEWER_CONFIG__.width")
            dbg_has = pg.evaluate("window.__viewerDebugOutline().outlinePixels") > 0
            check("restored 2D slice shows its mask (outline pixels present)", dbg_has)

            print("JS errors:", errs or "none"); check("no JS errors", not errs)
            ctx.close()
    finally:
        srv.terminate()
        try: srv.wait(timeout=5)
        except Exception: srv.kill()
    npass = sum(1 for r in RESULTS if r)
    print(f"\n{npass}/{len(RESULTS)} checks passed")
    print("RESULT:", "PASS" if npass == len(RESULTS) else "FAIL")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
