"""Real Chrome, isolated test server (port 8791): stroke continuity (independent
vs continuous picks along a line), the EA opacity pick, and that a painted
stroke stays after the server commits it."""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", args=["--headless=new", "--enable-unsafe-webgpu"])
    pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1, color_scheme="dark")
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://127.0.0.1:8791/"); pg.wait_for_timeout(4000)
    pg.click('[data-view="3d"]'); pg.wait_for_timeout(3500)
    pg.evaluate("() => { const s = document.getElementById('projModeSelect'); s.value = '1'; s.dispatchEvent(new Event('change', { bubbles: true })); }"); pg.wait_for_timeout(800)
    res = pg.evaluate("""() => { const g = window.__volumeMode.gpu(); g.setShowLabels(false);
      const c = g.canvas, W = c.clientWidth, H = c.clientHeight, out = {};
      for (const row of [0.42, 0.5, 0.58]) {
        const y = row * H, ind = [], cont = []; let prev = null;
        for (let x = 0.2 * W; x < 0.8 * W; x += 3) {
          const a = g.pickBrush(x, y); ind.push(a ? a.point : null);
          const bb = g.pickBrush(x, y, prev ? { prev, window: 7 } : null); if (bb) prev = bb.point; cont.push(bb ? bb.point : null);
        }
        const jumps = (arr) => { let mx = 0, big = 0; for (let i = 1; i < arr.length; i++) { if (!arr[i] || !arr[i - 1]) continue;
          const d = Math.hypot(...arr[i].map((v, k) => v - arr[i - 1][k])); mx = Math.max(mx, d); if (d > 8) big++; } return [mx.toFixed(1), big]; };
        out['row ' + row] = { independent: jumps(ind), continuous: jumps(cont) };
      }
      return out; }""")
    for k, v in res.items(): print(k, "| independent picks: largest jump", v["independent"][0], "voxels,", v["independent"][1], "jumps > 8 | continuous:", v["continuous"][0], "voxels,", v["continuous"][1], "jumps > 8")
    # EA with density: the opacity-0.5 surface (in front of, or at, the MIP pick)
    pg.evaluate("() => { const s = document.getElementById('projModeSelect'); s.value = '0'; s.dispatchEvent(new Event('change', { bubbles: true })); window.__volumeMode.gpu().setDensity(0.5); }"); pg.wait_for_timeout(500)
    print("EA vs MIP depth at center:", pg.evaluate("""() => { const g = window.__volumeMode.gpu(), c = g.canvas, x = c.clientWidth / 2, y = c.clientHeight / 2;
      const ea = g.pickBrush(x, y); g.mode = 1; const mip = g.pickBrush(x, y); g.mode = 0;
      return { ea: ea && ea.point.map(v => v.toFixed(1)), mip: mip && mip.point.map(v => v.toFixed(1)) }; }"""))
    # a real stroke stays after the commit
    pg.evaluate("() => { const s = document.getElementById('projModeSelect'); s.value = '1'; s.dispatchEvent(new Event('change', { bubbles: true })); window.__volumeMode.gpu().setShowLabels(true); }"); pg.wait_for_timeout(500)
    sid = pg.evaluate("window.__VIEWER_CONFIG__.sessionId")
    count = lambda: pg.evaluate("""async (sid) => { const r = await fetch('/api/ncolor_volume/' + encodeURIComponent(sid)); const a = new Uint8Array(await r.arrayBuffer()); let n = 0; for (const v of a) if (v) n++; return n; }""", sid)
    host = lambda: pg.evaluate("() => { const l = window.__volumeMode.gpu()._labHost; let n = 0; for (const v of l) if (v) n++; return n; }")
    n0 = count()
    cv = pg.locator("canvas#volumeViewer").bounding_box(); cx, cy = cv["x"] + cv["width"] / 2, cv["y"] + cv["height"] / 2
    pg.mouse.move(cx - 80, cy); pg.mouse.down()
    for i in range(1, 15): pg.mouse.move(cx - 80 + 12 * i, cy + 3 * i); pg.wait_for_timeout(25)
    pg.mouse.up(); pg.wait_for_timeout(3000)
    print("server labeled voxels:", n0, "->", count(), "| 3D view's labels (host copy) after commit:", host(), "| status:", pg.evaluate("document.getElementById('volFps').textContent"))
    print("errors:", errs)
    b.close()
