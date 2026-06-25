"""End-to-end 3D-view picker / fill: click the cell under the cursor on the
rendered volume (ray-pick) to pick its colour, delete it (erase), or merge cells.
Drives real clicks on the 3D canvas in headless Chrome; verifies the server volume.
"""
import subprocess, sys, time, urllib.request, os
import numpy as np
import tifffile
from playwright.sync_api import sync_playwright

PORT = 8789
BASE = f"http://127.0.0.1:{PORT}"
VOL = "/tmp/v3dpick_vol.tif"
MASKS = "/tmp/v3dpick_vol_masks.tif"


def wait_up(t=40):
    for _ in range(t * 2):
        try:
            urllib.request.urlopen(BASE + "/", timeout=2); return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    shape = (40, 80, 80)
    zz, yy, xx = np.mgrid[0:40, 0:80, 0:80]
    big = ((xx - 40) ** 2 + (yy - 40) ** 2 + (zz - 20) ** 2) < 16 ** 2     # central cell
    small = ((xx - 12) ** 2 + (yy - 12) ** 2 + (zz - 20) ** 2) < 7 ** 2     # off-centre cell
    img = np.zeros(shape, np.float32); img[big] = 800; img[small] = 800
    tifffile.imwrite(VOL, (img + np.random.default_rng(0).random(shape) * 50).astype(np.uint16))
    m = np.zeros(shape, np.uint8); m[big] = 1; m[small] = 2
    tifffile.imwrite(MASKS, m)
    if os.path.exists(MASKS.replace(".tif", "_edited.tif")):
        os.remove(MASKS.replace(".tif", "_edited.tif"))

    srv = subprocess.Popen([sys.executable, "-c",
        f"import uvicorn;uvicorn.run('ocdkit.viewer.app:create_app',factory=True,host='127.0.0.1',port={PORT},log_level='warning')"])
    ok_all = True
    try:
        if not wait_up():
            print("SERVER FAILED"); return 1
        with sync_playwright() as p:
            ctx = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"])
            pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__VIEWER_CONFIG__ && window.__VIEWER_CONFIG__.sessionId", timeout=15000)
            sid = pg.evaluate("window.__VIEWER_CONFIG__.sessionId")
            assert pg.request.post(BASE + "/api/open_image", data={"sessionId": sid, "path": VOL}).ok
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__viewerMaskApplied === true", timeout=20000); pg.wait_for_timeout(800)

            def cnt(l):
                r = pg.request.get(f"{BASE}/api/mask_slice/{sid}?z=20&axis=0&kind=instance")
                a = np.frombuffer(r.body(), dtype=np.dtype(r.headers.get("x-mask-dtype", "uint8")))
                return int((a == l).sum())

            g1 = pg.request.get(f"{BASE}/api/ncolor_map/{sid}").json()["groups"][1]
            pg.eval_on_selector('[data-view="3d"]', "e=>e.click()")
            pg.wait_for_function("window.__volumeMode.gpu()!==null", timeout=20000); pg.wait_for_timeout(1000)
            box = pg.query_selector("#volumeViewer").bounding_box()
            center = (box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)

            # 1) picker: click the central cell → active colour becomes cell 1's colour
            pg.query_selector('.tool-stop[data-mode="picker"]').click(); pg.wait_for_timeout(150)
            pg.mouse.click(*center); pg.wait_for_timeout(700)
            picked = pg.evaluate("window.__viewerCurrentLabel()")
            t1 = picked == g1
            print(f"3D picker: picked colour {picked} == cell1 colour {g1} -> {'PASS' if t1 else 'FAIL'}")
            ok_all = ok_all and t1

            # 1b) picker on empty space (ray that misses the volume) → zero marker
            pg.evaluate("""async()=>{await window.__viewerVolume3DPick(
                {ro:[9999,9999,9999],rd:[1,0,0],boxMin:[-40,-40,-40],boxMax:[40,40,40]},'picker');}""")
            pg.wait_for_timeout(400)
            t1b = pg.evaluate("window.__viewerCurrentLabel()") == 0
            print(f"3D picker on empty → zero marker (currentLabel 0): {t1b} -> {'PASS' if t1b else 'FAIL'}")
            ok_all = ok_all and t1b

            # 2) erase-fill: delete the central cell under the cursor (in-place, no flash)
            pg.query_selector('.tool-stop[data-mode="fill"]').click(); pg.wait_for_timeout(150)
            pg.keyboard.down("e"); pg.wait_for_timeout(150)
            pg.mouse.click(*center); pg.wait_for_timeout(1500)
            pg.keyboard.up("e")
            mode_after = pg.evaluate("window.__volumeMode.getMode()")
            t2 = cnt(1) == 0 and cnt(2) > 0 and mode_after == "3d"
            print(f"3D erase-fill: cell1 deleted={cnt(1) == 0} cell2 kept={cnt(2) > 0} stayed-3d={mode_after=='3d'} -> {'PASS' if t2 else 'FAIL'}")
            ok_all = ok_all and t2

            # 3) undo restores the deleted cell (server-owned history)
            pg.keyboard.press("Control+z"); pg.wait_for_timeout(800)
            t3 = cnt(1) > 0
            print(f"3D undo restores deleted cell: {t3} -> {'PASS' if t3 else 'FAIL'}")
            ok_all = ok_all and t3

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
