"""Drive the bench page in real Google Chrome (Playwright channel="chrome"; the
bundled Chromium has no WebGPU adapter).

Usage: run.py DATA_DIR "suite=render&k=15&rounds=3" [timeout_s]
Starts server.py, opens the page, waits for window.__done, prints the log.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
data, query = sys.argv[1], sys.argv[2]
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 3600
page = sys.argv[4] if len(sys.argv) > 4 else "index.html"   # the bench page (modes.html: per-mode suite)
port = 8765
srv = subprocess.Popen([sys.executable, str(HERE / "server.py"), data, str(port)])
try:
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html")
            break
        except Exception:
            time.sleep(0.2)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=[
            "--enable-unsafe-webgpu", "--enable-dawn-features=allow_unsafe_apis",
            "--disable-dawn-features=timestamp_quantization"])
        pg = b.new_page()
        pg.on("console", lambda m: print("[console]", m.text, flush=True) if m.type == "error" else None)
        pg.goto(f"http://127.0.0.1:{port}/{page}?{query}")
        t0 = time.time()
        last = 0
        while not pg.evaluate("window.__done === true"):
            if time.time() - t0 > timeout:
                print("TIMEOUT")
                break
            txt = pg.evaluate("document.getElementById('log').textContent")
            if len(txt) > last:
                print(txt[last:], end="", flush=True)
                last = len(txt)
            time.sleep(1)
        txt = pg.evaluate("document.getElementById('log').textContent")
        print(txt[last:], flush=True)
        err = pg.evaluate("window.__err")
        if err:
            print("ERR:", err)
        b.close()
finally:
    srv.terminate()
