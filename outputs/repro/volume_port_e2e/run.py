"""End-to-end check of the ported 3D renderer in the REAL viewer, on real data.

Launches the viewer, opens a (Z, Y, X) stack with its *_masks sidecar, switches
to the 3D view (binary float16 transport -> brick grids -> per-state compute
pipelines), then cycles EA / MIP / mean, hides and shows labels, and does a 3D
fill (label texture + label bricks rebuilt in place). Fails on any page error or
WebGPU console error, or a blank render. Screenshots land next to this script.

Usage: VOLUME_E2E_STACK=<dir>/<name>.tif run.py   (the <name>_masks.tif sidecar must exist)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright
from skimage import io as skio

HERE = Path(__file__).resolve().parent
PORT = 8798
BASE = f"http://127.0.0.1:{PORT}"


def wait_up(timeout=60):
    for _ in range(timeout * 2):
        try:
            urllib.request.urlopen(BASE + "/", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def shot(pg, name):
    p = str(HERE / name)
    pg.query_selector("#volumeViewer").screenshot(path=p)
    return float(skio.imread(p)[..., :3].astype(np.float32).std())


def main():
    src = Path(os.environ["VOLUME_E2E_STACK"])
    work = Path(tempfile.mkdtemp(prefix="volume_e2e_"))     # edits autosave here, not next to the source
    vol = work / src.name
    shutil.copy(src, vol)
    shutil.copy(src.with_name(src.stem + "_masks.tif"), work / (src.stem + "_masks.tif"))
    srv = subprocess.Popen([sys.executable, "-c",
                            f"import uvicorn; uvicorn.run('ocdkit.viewer.app:create_app', factory=True, "
                            f"host='127.0.0.1', port={PORT}, log_level='warning')"])
    checks = {}
    try:
        assert wait_up(), "server failed to start"
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome", headless=True,
                                  args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"])
            pg = b.new_page(viewport={"width": 1400, "height": 900})
            errs = []
            pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
            pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}")
                  if (m.type == "error" and "Failed to load resource" not in m.text)
                  or "WebGPU" in m.text or "Invalid" in m.text else None)
            # failed requests are reported by URL (the console message omits it); a
            # missing favicon is harmless, anything else counts as an error
            pg.on("response", lambda r: errs.append(f"HTTP {r.status}: {r.url}")
                  if r.status >= 400 and not r.url.endswith("favicon.ico") else None)
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__VIEWER_CONFIG__ && window.__VIEWER_CONFIG__.sessionId", timeout=20000)
            sid = pg.evaluate("window.__VIEWER_CONFIG__.sessionId")
            assert pg.request.post(BASE + "/api/open_image", data={"sessionId": sid, "path": str(vol)}).ok
            pg.goto(BASE + "/", wait_until="load")
            pg.wait_for_function("window.__volumeMode !== undefined", timeout=30000)
            pg.wait_for_function("window.__viewerMaskApplied === true", timeout=30000)

            t0 = time.time()
            pg.eval_on_selector('[data-view="3d"]', "el => el.click()")
            pg.wait_for_function("window.__volumeMode.gpu() !== null", timeout=60000)
            checks["3d_mount_s"] = round(time.time() - t0, 2)
            pg.wait_for_timeout(800)
            g = "window.__volumeMode.gpu()"
            checks["used_binary_f16"] = pg.evaluate(f"!!{g}.decoded.imageF16")
            checks["bricks"] = pg.evaluate(f"!!({g}.brickImgTex && {g}.brickLabTex)")

            for proj, name in ((0, "EA"), (1, "MIP"), (2, "mean")):
                pg.evaluate(f"window.__volumeMode.setProj({proj})")
                pg.wait_for_timeout(500)
                checks[f"std_{name}"] = round(shot(pg, f"e2e_{name}.png"), 2)
            pg.evaluate(f"{g}.setShowLabels(0)"); pg.wait_for_timeout(400)
            checks["std_labels_hidden"] = round(shot(pg, "e2e_labels_hidden.png"), 2)
            pg.evaluate(f"{g}.setShowLabels(1)"); pg.wait_for_timeout(400)

            # old vs new transport, measured inside the real app on the same session
            checks["load_old_json_ms"] = pg.evaluate(f"""(async () => {{ const t = performance.now();
                await window.decodeBundle(await (await fetch('/api/volume_bundle/{sid}')).json());
                return Math.round(performance.now() - t); }})()""")
            checks["load_new_raw_ms"] = pg.evaluate(f"""(async () => {{ const t = performance.now();
                const r = await fetch('/api/volume_raw/{sid}'); await r.arrayBuffer();
                return Math.round(performance.now() - t); }})()""")

            # 3D erase at the cell under the canvas centre, then the SAME refresh the
            # UI runs after a 3D fill (ncolor volume -> updateLabels: label texture
            # and label bricks rewritten in place)
            before = pg.request.get(f"{BASE}/api/ncolor_volume/{sid}").body()
            hit = pg.evaluate(f"""(async () => {{
                const g = {g}, c = g.canvas, r = c.getBoundingClientRect();
                const post = (u, ray) => fetch(u, {{method:'POST', headers:{{'content-type':'application/json'}},
                                                   body: JSON.stringify(ray)}}).then((x) => x.json());
                for (const [fx, fy] of [[.5,.5],[.45,.5],[.55,.5],[.5,.45],[.5,.55],[.4,.4],[.6,.6]]) {{
                  const ray = g.pickRayWorld(r.width * fx, r.height * fy);
                  const h = await post('/api/pick_ray/{sid}', ray);
                  if (!h.label) continue;
                  await post('/api/fill_ray/{sid}?erase=1', ray);
                  const vol = await (await fetch('/api/ncolor_volume/{sid}')).arrayBuffer();
                  g.updateLabels(new Uint8Array(vol));
                  return h.label;
                }}
                return 0; }})()""")
            pg.wait_for_timeout(1500)
            after = pg.request.get(f"{BASE}/api/ncolor_volume/{sid}").body()
            checks["fill_label"] = hit
            checks["fill_changed_volume"] = before != after
            checks["std_after_fill"] = round(shot(pg, "e2e_after_fill.png"), 2)
            a = skio.imread(str(HERE / "e2e_mean.png"))[..., :3].astype(int)
            c = skio.imread(str(HERE / "e2e_after_fill.png"))[..., :3].astype(int)
            checks["fill_pixels_changed"] = int((np.abs(a - c).max(-1) > 2).sum())   # the erased cell
            checks["errors"] = errs
            b.close()
    finally:
        srv.terminate()
        shutil.rmtree(work, ignore_errors=True)
    for k, v in checks.items():
        print(f"{k}: {v}")
    ok = (checks.get("used_binary_f16") and checks.get("bricks") and not checks.get("errors")
          and all(checks.get(f"std_{n}", 0) > 1.0 for n in ("EA", "MIP", "mean", "labels_hidden", "after_fill"))
          and checks.get("fill_label") and checks.get("fill_changed_volume")
          and checks.get("fill_pixels_changed", 0) > 0)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
