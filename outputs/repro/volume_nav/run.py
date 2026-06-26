"""Switching the open image via the dropdown — incl. between 3D stacks — actually
loads (reload-based, session persists via cookie). Also checks load-masks auto-
detects a sibling *_masks file.
"""
import subprocess, sys, time, urllib.request, os, shutil
import numpy as np
import tifffile
from playwright.sync_api import sync_playwright

PORT = 8784
BASE = f"http://127.0.0.1:{PORT}"
DIR = "/tmp/navdir"


def wait_up(t=40):
    for _ in range(t * 2):
        try:
            urllib.request.urlopen(BASE + "/", timeout=2); return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    shutil.rmtree(DIR, ignore_errors=True); os.makedirs(DIR)
    tifffile.imwrite(f"{DIR}/volA.tif", (np.random.default_rng(0).random((12, 40, 40)) * 255).astype(np.uint8))
    tifffile.imwrite(f"{DIR}/volB.tif", (np.random.default_rng(1).random((20, 50, 50)) * 255).astype(np.uint8))
    mB = np.zeros((20, 50, 50), np.uint8); mB[:, 8:16, 8:42] = 1
    tifffile.imwrite(f"{DIR}/volB_masks.tif", mB)         # sibling mask for auto-detect
    env = dict(os.environ, OCDKIT_VIEWER_SAMPLE_IMAGE=f"{DIR}/volA.tif")
    srv = subprocess.Popen([sys.executable, "-c",
        f"import uvicorn;uvicorn.run('ocdkit.viewer.app:create_app',factory=True,host='127.0.0.1',port={PORT},log_level='warning')"], env=env)
    results = []
    def check(n, ok): results.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    try:
        if not wait_up():
            print("SERVER FAILED"); return 1
        with sync_playwright() as p:
            ctx = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"])
            pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__VIEWER_CONFIG__", timeout=20000); pg.wait_for_timeout(700)
            d0 = pg.evaluate("window.__VIEWER_CONFIG__.volumeDepth")
            check("loaded volA as a volume (depth 12)", pg.evaluate("!!window.__VIEWER_CONFIG__.isVolume") and d0 == 12)

            # switch to volB via the dropdown (option value = path) → change event
            pg.evaluate("""() => {
                const sel = document.getElementById('imageNavigator');
                const opt = Array.from(sel.options).find(o => (o.value||'').endsWith('volB.tif'));
                sel.value = opt.value; sel.dispatchEvent(new Event('change', {bubbles:true}));
            }""")
            # volume switch triggers a reload; wait for the new config
            ok = False
            for _ in range(40):
                try:
                    if pg.evaluate("window.__VIEWER_CONFIG__ && window.__VIEWER_CONFIG__.volumeDepth") == 20:
                        ok = True; break
                except Exception:
                    pass
                pg.wait_for_timeout(250)
            check("dropdown switch to volB loaded (depth 20)", ok)
            check("volB path is current", "volB.tif" in (pg.evaluate("window.__VIEWER_CONFIG__.imagePath") or ""))

            # load-masks auto-detects volB_masks.tif (no dialog)
            pg.wait_for_timeout(500)
            r = pg.request.post(f"{BASE}/api/auto_mask", data={"sessionId": pg.evaluate("window.__VIEWER_CONFIG__.sessionId")})
            j = r.json() if r.ok else {}
            check("load-masks auto-detects volB_masks.tif", bool(j.get("loaded")) and "volB_masks" in (j.get("path") or ""))

            print("JS errors:", errs or "none"); check("no JS errors", not errs)
            ctx.close()
    finally:
        srv.terminate()
        try: srv.wait(timeout=5)
        except Exception: srv.kill()
    npass = sum(1 for r in results if r)
    print(f"\n{npass}/{len(results)} checks passed")
    print("RESULT:", "PASS" if npass == len(results) else "FAIL")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
