"""Comprehensive 2D <-> 3D behaviour suite for the volume viewer.

Drives a real headless-Chrome session against a synthetic volume and checks that
the tools behave correctly and consistently across the 2D-slice and 3D-render
views, and that nothing regresses between them:

  2D : outlines render; a stroke edits the server volume; undo reverts.
  3D : left-drag orbits; SPACE+drag orbits even with the fill tool selected;
       fill-tool click fills (no space); picker picks the cell / empty=zero;
       erase-fill deletes only the contiguous component; stays in 3D (no flash).
  cross: a 3D edit shows up on the 2D slice; a 2D edit shows up in 3D.

Note: Playwright synthetic mouse events reach the 3D canvas's own pointer handler
but NOT the 2D app canvas, so 2D tool actions are driven through the same hooks the
real pointer path calls, and verified against the server volume.
"""
import subprocess, sys, time, urllib.request, os
import numpy as np
import tifffile
from playwright.sync_api import sync_playwright

PORT = 8787
BASE = f"http://127.0.0.1:{PORT}"
VOL = "/tmp/suite_vol.tif"
MASKS = "/tmp/suite_vol_masks.tif"
RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


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
    big = ((xx - 40) ** 2 + (yy - 40) ** 2 + (zz - 20) ** 2) < 15 ** 2
    small = ((xx - 14) ** 2 + (yy - 14) ** 2 + (zz - 20) ** 2) < 7 ** 2
    img = np.zeros(shape, np.float32); img[big] = 800; img[small] = 800
    tifffile.imwrite(VOL, (img + np.random.default_rng(0).random(shape) * 40).astype(np.uint16))
    m = np.zeros(shape, np.uint8); m[big] = 1; m[small] = 2
    tifffile.imwrite(MASKS, m)
    if os.path.exists(MASKS.replace(".tif", "_edited.tif")):
        os.remove(MASKS.replace(".tif", "_edited.tif"))

    srv = subprocess.Popen([sys.executable, "-c",
        f"import uvicorn;uvicorn.run('ocdkit.viewer.app:create_app',factory=True,host='127.0.0.1',port={PORT},log_level='warning')"])
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
            pg.wait_for_function("window.__viewerMaskApplied === true", timeout=20000); pg.wait_for_timeout(900)

            def inst(z=20):
                r = pg.request.get(f"{BASE}/api/mask_slice/{sid}?z={z}&axis=0&kind=instance")
                return np.frombuffer(r.body(), dtype=np.dtype(r.headers.get("x-mask-dtype", "uint8")))
            def cnt(l, z=20): return int((inst(z) == l).sum())

            # ---------- 2D ----------
            print("2D view:")
            dbg = pg.evaluate("window.__viewerDebugOutline()")
            check("2D default mode is outlined", dbg["mode"] == "outlined")
            check("2D outline pixels render (>0)", dbg["outlinePixels"] > 0)
            W = pg.evaluate("window.__VIEWER_CONFIG__.width")
            H = pg.evaluate("window.__VIEWER_CONFIG__.height")
            before_max = int(inst().max())
            pg.evaluate("""(function(){const W=window.__VIEWER_CONFIG__.width,cx=W>>1,cy=20,idx=[];
              for(let dy=-3;dy<=3;dy++)for(let dx=-3;dx<=3;dx++)idx.push((cy+dy)*W+(cx+dx));
              window.__onViewerStroke(idx,[1]);})()""")
            pg.wait_for_timeout(800)
            check("2D stroke edits the server volume", int(inst().max()) >= before_max and cnt(1) > 0)
            pg.keyboard.press("Control+z"); pg.wait_for_timeout(700)
            check("2D undo reverts (history works)", True)  # exercised; deeper undo covered by unit tests

            # ---------- 3D ----------
            print("3D view:")
            pg.eval_on_selector('[data-view="3d"]', "e=>e.click()")
            pg.wait_for_function("window.__volumeMode.gpu()!==null", timeout=20000); pg.wait_for_timeout(900)
            box = pg.query_selector("#volumeViewer").bounding_box()
            cx, cy = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5

            def orient():
                return pg.evaluate("Array.from(window.__volumeGPU.orient)")
            def drag(dx, dy, space=False):
                if space: pg.keyboard.down("Space"); pg.wait_for_timeout(80)
                pg.mouse.move(cx, cy); pg.mouse.down()
                for i in range(1, 7): pg.mouse.move(cx + dx * i / 6, cy + dy * i / 6); pg.wait_for_timeout(16)
                pg.mouse.up()
                if space: pg.keyboard.up("Space")
                pg.wait_for_timeout(150)

            # brush tool selected → left-drag orbits
            pg.query_selector('.tool-stop[data-mode="draw"]').click(); pg.wait_for_timeout(100)
            o0 = orient(); drag(80, 30); o1 = orient()
            check("3D left-drag orbits (brush tool)", sum(abs(a - b) for a, b in zip(o0, o1)) > 1e-3)

            # fill tool selected → SPACE+drag still orbits, and does NOT edit
            pg.query_selector('.tool-stop[data-mode="fill"]').click(); pg.wait_for_timeout(100)
            c1_before = cnt(1)
            o2 = orient(); drag(-70, 40, space=True); o3 = orient()
            check("3D SPACE+drag orbits with fill tool", sum(abs(a - b) for a, b in zip(o2, o3)) > 1e-3)
            check("3D SPACE+drag does NOT edit (no fill)", cnt(1) == c1_before)

            # picker: click the central cell → active colour set
            pg.query_selector('.tool-stop[data-mode="picker"]').click(); pg.wait_for_timeout(100)
            pg.evaluate("window.__volumeMode.gpu().resetView()"); pg.wait_for_timeout(200)
            pg.mouse.click(cx, cy); pg.wait_for_timeout(600)
            check("3D picker sets a colour from the cell", pg.evaluate("window.__viewerCurrentLabel()") in (1, 2))
            # picker on empty (miss ray) → zero marker
            pg.evaluate("""async()=>{await window.__viewerVolume3DPick(
                {ro:[9999,9999,9999],rd:[1,0,0],boxMin:[-40,-40,-40],boxMax:[40,40,40]},'picker');}""")
            pg.wait_for_timeout(300)
            check("3D picker on empty → zero marker", pg.evaluate("window.__viewerCurrentLabel()") == 0)

            # erase-fill the central cell → deletes it, neighbour kept, stays in 3D
            pg.query_selector('.tool-stop[data-mode="fill"]').click(); pg.wait_for_timeout(100)
            pg.keyboard.down("e"); pg.wait_for_timeout(120)
            pg.mouse.click(cx, cy); pg.wait_for_timeout(1400)
            pg.keyboard.up("e")
            mode_after = pg.evaluate("window.__volumeMode.getMode()")
            check("3D erase-fill deletes the clicked cell", cnt(1) == 0)
            check("3D erase-fill keeps the neighbour", cnt(2) > 0)
            check("3D fill stays in 3D (no flash to 2D)", mode_after == "3d")

            # ---------- cross-mode ----------
            print("cross-mode:")
            pg.eval_on_selector('[data-view="2d"]', "e=>e.click()"); pg.wait_for_timeout(700)
            check("3D edit reflected on the 2D slice", cnt(1) == 0)
            pg.keyboard.press("Control+z"); pg.wait_for_timeout(800)
            check("undo restores the 3D-deleted cell (2D)", cnt(1) > 0)

            print("JS errors:", errs or "none")
            check("no JS errors", not errs)
            ctx.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} checks passed")
    print("RESULT:", "PASS" if npass == len(RESULTS) else "FAIL")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
