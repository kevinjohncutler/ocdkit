"""Where does the time go between clicking '3D volume' and the first 3D frame?

Instruments the real viewer page (fetch, WebGPU shader/pipeline creation,
texture uploads, VolumeGPU setup steps), clicks 3D, and waits until the GPU has
finished the first frame. Runs a cold load (fresh profile: shader caches empty)
then a warm reload in the same profile.
Usage: load_timing.py URL
"""
import json
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1]

INSTRUMENT = """(() => {
  const T0 = performance.now(); const L = window.__tl = [];
  const mark = (what, t0, extra) => L.push(Object.assign({what, start: +(t0 - T0).toFixed(1),
                                                          ms: +(performance.now() - t0).toFixed(1)}, extra || {}));
  window.__tlMark = mark; window.__tlT0 = T0;
  const of = window.fetch;
  window.fetch = async function (u, o) { const t = performance.now(); const r = await of(u, o);
    const url = String(u).split('?')[0];
    if (/volume|wgsl|ncolor/.test(url)) { const c = r.clone(); c.arrayBuffer().then((b) =>
      mark('fetch ' + url.replace(/.*\\//, ''), t, {bytes: b.byteLength})); }
    return r; };
  const D = GPUDevice.prototype;
  for (const fn of ['createShaderModule', 'createComputePipeline', 'createRenderPipeline']) {
    const o = D[fn]; D[fn] = function (d) { const t = performance.now(); const r = o.call(this, d);
      mark(fn, t); return r; }; }
  const Q = GPUQueue.prototype, ow = Q.writeTexture;
  Q.writeTexture = function (...a) { const t = performance.now(); const r = ow.apply(this, a);
    mark('writeTexture', t, {bytes: a[1].byteLength}); return r; };
  const V = window.VolumeGPU.prototype;
  for (const fn of ['_uploadTextures', '_prewarmComputePipelines', '_uploadLut', '_makeBindGroup']) {
    if (!V[fn]) continue; const o = V[fn];
    V[fn] = function (...a) { const t = performance.now(); const r = o.apply(this, a); mark(fn, t); return r; }; }
  const oc = window.VolumeGPU.create;
  window.VolumeGPU.create = async function (...a) { const t = performance.now(); const g = await oc.apply(this, a);
    mark('VolumeGPU.create (total)', t);
    const t2 = performance.now(); await g.device.queue.onSubmittedWorkDone(); mark('first frame GPU done (after create)', t2);
    window.__firstFrame = performance.now() - T0; return g; };
})()"""


def run(browser_ctx, label):
    pg = browser_ctx.new_page()
    pg.goto(URL, wait_until="load")
    pg.wait_for_function("window.__volumeMode !== undefined && !!window.VolumeGPU", timeout=30000)
    time.sleep(1.0)
    pg.evaluate(INSTRUMENT)
    t_click = pg.evaluate("(() => { const t = performance.now() - window.__tlT0;"
                          " document.querySelector('[data-view=\"3d\"]').click(); return t; })()")
    pg.wait_for_function("window.__firstFrame !== undefined", timeout=60000)
    ff = pg.evaluate("window.__firstFrame")
    tl = pg.evaluate("window.__tl")
    pg.close()
    print(f"\n== {label}: click -> first frame GPU done: {ff - t_click:.0f} ms")
    for e in sorted(tl, key=lambda e: e["start"]):
        if e["ms"] >= 2 or "frame" in e["what"] or "fetch" in e["what"]:
            print(f"  +{e['start'] - t_click:7.1f} ms  {e['ms']:7.1f} ms  {e['what']}"
                  + (f"  ({e['bytes'] / 1e6:.1f} MB)" if e.get("bytes") else ""))


def run_restore(browser_ctx, label):
    """Reload: the viewer reopens 3D by itself. Instrument before any page script
    runs, then time navigation start -> first 3D frame."""
    pg = browser_ctx.new_page()
    pg.add_init_script("""(() => { const iv = setInterval(() => {
        if (window.VolumeGPU && window.GPUDevice && !window.__tl) { clearInterval(iv); (%s)(); } }, 1); })()"""
                       % INSTRUMENT.strip()[1:-3].strip().removeprefix("() =>").strip().join(["() => ", ""]))
    pg.goto(URL, wait_until="commit")
    pg.wait_for_function("window.__firstFrame !== undefined", timeout=60000)
    nav = pg.evaluate("performance.timeOrigin")
    ff = pg.evaluate("window.__firstFrame + window.__tlT0")
    tl = pg.evaluate("window.__tl")
    t0 = pg.evaluate("window.__tlT0")
    pg.close()
    print(f"\n== {label}: navigation start -> first 3D frame GPU done: {ff:.0f} ms")
    for e in sorted(tl, key=lambda e: e["start"]):
        if e["ms"] >= 2 or "frame" in e["what"] or "fetch" in e["what"]:
            print(f"  +{e['start'] + t0:7.1f} ms  {e['ms']:7.1f} ms  {e['what']}"
                  + (f"  ({e['bytes'] / 1e6:.1f} MB)" if e.get("bytes") else ""))


with sync_playwright() as p:
    prof = tempfile.mkdtemp(prefix="load_timing_")
    ctx = p.chromium.launch_persistent_context(prof, channel="chrome", headless=True,
                                               args=["--headless=new", "--enable-unsafe-webgpu", "--use-angle=metal"],
                                               viewport={"width": 1400, "height": 900}, device_scale_factor=2)
    run(ctx, "cold (fresh profile)")
    run_restore(ctx, "reload with 3D remembered (same profile)")
    ctx.close()
