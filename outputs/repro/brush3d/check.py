"""Real Chrome against an ISOLATED test server (port 8791, its own sample copies):
3D brush hover preview, paint by drag, space + drag rotates, undo."""
import pathlib, numpy as np
from playwright.sync_api import sync_playwright
out = pathlib.Path(__file__).parent
URL = "http://127.0.0.1:8791/"
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", args=["--headless=new", "--enable-unsafe-webgpu"])
    pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1, color_scheme="dark")
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(4000)
    pg.click('[data-view="3d"]'); pg.wait_for_timeout(3500)
    pg.evaluate("() => { const s = document.getElementById('projModeSelect'); s.value = '1'; s.dispatchEvent(new Event('change', { bubbles: true })); }"); pg.wait_for_timeout(800)
    sid = pg.evaluate("window.__VIEWER_CONFIG__.sessionId")
    count = lambda: pg.evaluate("""async (sid) => { const r = await fetch('/api/ncolor_volume/' + encodeURIComponent(sid)); const a = new Uint8Array(await r.arrayBuffer()); let n = 0; for (const v of a) if (v) n++; return n; }""", sid)
    print("tool:", pg.evaluate("window.__viewerActiveTool()"), "| session:", sid, "| labeled voxels before:", count())
    cv = pg.locator("canvas#volumeViewer").bounding_box()
    cx, cy = cv["x"] + cv["width"] / 2, cv["y"] + cv["height"] / 2
    # hover: preview
    pg.mouse.move(cx - 60, cy); pg.wait_for_timeout(500)
    print("hover brush:", pg.evaluate("window.__volumeMode.gpu()._brush"))
    pg.screenshot(path=str(out / "hover.png"), clip={"x": cx - 250, "y": cy - 180, "width": 500, "height": 360})
    # paint: drag
    pg.mouse.down(); 
    for i in range(1, 11): pg.mouse.move(cx - 60 + 12 * i, cy + 2 * i); pg.wait_for_timeout(30)
    pg.screenshot(path=str(out / "painting.png"), clip={"x": cx - 250, "y": cy - 180, "width": 500, "height": 360})
    pg.mouse.up(); pg.wait_for_timeout(2500)
    after = count(); print("labeled voxels after the stroke:", after, "| can undo:", pg.evaluate("window.__viewerVolumeCanUndo()"))
    pg.mouse.move(cx + 300, cy + 250); pg.wait_for_timeout(500)
    pg.screenshot(path=str(out / "painted.png"), clip={"x": cx - 250, "y": cy - 180, "width": 500, "height": 360})
    # space + drag rotates, no paint
    o0 = pg.evaluate("Array.from(window.__volumeMode.gpu().orient)")
    pg.mouse.move(cx, cy); pg.keyboard.down("Space"); pg.wait_for_timeout(100)
    pg.mouse.down(); pg.mouse.move(cx + 80, cy + 30, steps=8); pg.mouse.up(); pg.keyboard.up("Space"); pg.wait_for_timeout(1500)
    o1 = pg.evaluate("Array.from(window.__volumeMode.gpu().orient)")
    print("space + drag rotated:", not np.allclose(o0, o1), "| labeled voxels unchanged:", count() == after)
    # undo
    pg.evaluate("window.__viewerVolumeUndo()"); pg.wait_for_timeout(2000)
    print("after undo:", count())
    print("errors:", errs)
    b.close()
