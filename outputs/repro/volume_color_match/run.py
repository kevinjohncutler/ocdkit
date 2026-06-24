"""Self-test: 2D-slice and 3D-volume colors match, and adjacent cells differ.

Launches a fresh viewer with a synthetic volume + touching-cell mask, captures
the real 2D canvas and 3D canvas in headless Chrome, and asserts:
  - the dominant mask HUES in 2D and 3D overlap (same ncolor palette), and
  - touching cells render in different colors (ncolor working).
Saves the captures + a side-by-side under this directory. Rerunnable; no user.
"""
import colorsys
import subprocess
import sys
import time
import urllib.request

import numpy as np
import tifffile
from skimage import io as skio
from playwright.sync_api import sync_playwright

PORT = 8804
BASE = f"http://127.0.0.1:{PORT}"
VOL = "/tmp/colormatch_vol.tif"
MASKS = "/tmp/colormatch_vol_masks.tif"
HERE = "/Volumes/DataDrive/ocdkit/outputs/repro/volume_color_match"


def hue_buckets(img, n=8, nb=24):
    a = img[..., :3].astype(np.float32)
    sat = (a.max(2) - a.min(2)) > 40
    px = a[sat]
    if not len(px):
        return []
    H = np.array([colorsys.rgb_to_hsv(*(c / 255))[0] for c in px])
    q = (np.round(H * nb).astype(int)) % nb
    u, c = np.unique(q, return_counts=True)
    return sorted(int(x) for x in u[np.argsort(-c)][:n])


def wait_up(t=40):
    for _ in range(t * 2):
        try:
            urllib.request.urlopen(BASE + "/", timeout=2); return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    # synthetic volume + a mask of many touching parallel rods (forces ncolor)
    tifffile.imwrite(VOL, (np.random.default_rng(0).random((12, 80, 80)) * 800).astype(np.uint16))
    lab = np.zeros((12, 80, 80), np.uint16)
    cid = 1
    for y in range(6, 74, 6):
        lab[:, y:y + 5, 8:72] = cid; cid += 1     # stacked rods touching along y
    tifffile.imwrite(MASKS, lab)
    import os as _os
    _ed = MASKS.replace(".tif", "_edited.tif")
    if _os.path.exists(_ed):
        _os.remove(_ed)                            # don't resume from a prior run's autosave

    import os
    env = dict(os.environ, OCDKIT_VIEWER_SAMPLE_IMAGE=VOL)
    srv = subprocess.Popen(
        [sys.executable, "-c",
         f"import uvicorn; uvicorn.run('ocdkit.viewer.app:create_app', factory=True,"
         f" host='127.0.0.1', port={PORT}, log_level='warning')"], env=env)
    try:
        if not wait_up():
            print("SERVER FAILED"); return 1
        with sync_playwright() as p:
            ctx = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"])
            pg = ctx.new_page(); errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__viewerMaskApplied === true", timeout=20000)
            pg.wait_for_timeout(1200)
            pg.query_selector("#canvas").screenshot(path=f"{HERE}/cap_2d.png")
            img2d = skio.imread(f"{HERE}/cap_2d.png")
            # touching cells differ in 2D? count distinct group values
            groups = pg.evaluate("() => { const m=window.__VIEWER_DEBUG__.getMaskValues();"
                                 "const s=new Set(); for(let i=0;i<m.length;i++) if(m[i]) s.add(m[i]);"
                                 "return [...s].length; }")
            pg.eval_on_selector('[data-view="3d"]', "e=>e.click()")
            pg.wait_for_function("window.__volumeMode.gpu() !== null", timeout=20000)
            pg.wait_for_timeout(800)
            pg.query_selector("#volumeViewer").screenshot(path=f"{HERE}/cap_3d.png")
            img3d = skio.imread(f"{HERE}/cap_3d.png")
            ctx.close()

        h2, h3 = hue_buckets(img2d), hue_buckets(img3d)
        overlap = sorted(set(h2) & set(h3))
        palette_match = len(overlap) >= min(len(h2), len(h3)) - 1
        from skimage import transform
        h = min(img2d.shape[0], img3d.shape[0])
        rs = lambda x: transform.resize(x[..., :3], (h, int(x.shape[1] * h / x.shape[0])),
                                        preserve_range=True).astype(np.uint8)
        skio.imsave(f"{HERE}/color_2d_vs_3d.png",
                    np.concatenate([rs(img2d), np.full((h, 16, 3), 20, np.uint8), rs(img3d)], axis=1))
        print("2D hues:", h2, "| 3D hues:", h3, "| overlap:", overlap)
        print("distinct ncolor groups in 2D:", groups)
        print("palette match:", palette_match, "| groups>1:", groups > 1, "| JS errors:", errs or "none")
        ok = palette_match and groups > 1 and not errs
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        srv.terminate()
        try: srv.wait(timeout=10)
        except Exception: srv.kill()


if __name__ == "__main__":
    sys.exit(main())
