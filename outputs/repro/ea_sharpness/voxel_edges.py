"""Why EA looks interpolated while MIP looks blocky, with no interpolation anywhere.

Renders a volume of independent random voxels (no two neighbors alike) with the
viewer's compute shader, at 4 screen pixels per voxel, orthographic, straight
down an axis and tilted a few degrees. "Blockiness" is the fraction of
horizontally adjacent pixel pairs with (nearly) identical values: a perfectly
blocky image at 4 px per voxel scores 0.75, a smooth one near 0.
Writes voxel_edges.html next to this script.
"""
import base64
import io
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tests"))
import test_raymarch_compute_wgsl as T   # noqa: E402
import wgpu.utils                        # noqa: E402


def blockiness(img):
    d = np.abs(np.diff(img, axis=1))
    return float((d < 1e-3 * max(float(img.max()), 1e-6)).mean())


def png(img):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = np.percentile(img, [1, 99.5])
    buf = io.BytesIO()
    plt.imsave(buf, np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1), cmap="gray", format="png")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    dev = wgpu.utils.get_default_device()
    n, W = 32, 128
    vol = np.random.default_rng(0).uniform(0.1, 1.0, (n, n, n)).astype(np.float16)
    sc = T.Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    dims = (n, n, n)
    rows, cells = [], []
    for tilt_deg in (0, 1, 3, 10):
        inv, *_ = T._ortho(math.radians(tilt_deg), 0.0, n / 2)
        shift = n * math.tan(math.radians(tilt_deg))
        for name, mode, dens in (("MIP", 1, 0.0), ("EA density 0", 0, 0.0), ("EA density 0.3", 0, 0.3), ("EA density 2", 0, 2.0)):
            u = T._uniform(inv, dims, mode, density=dens, show_lab=0, exposure=0.02)
            out = sc.compute(u, mode, 1, 0, W, W)
            img = out[..., 3] if mode == 1 else out[..., 0]
            b = blockiness(img[8:-8, 8:-8])
            rows.append((tilt_deg, round(shift, 1), name, b))
            cells.append((f"{name}, tilt {tilt_deg} deg", img[32:96, 32:96]))
    print(f"{'tilt':>5} {'offset back-to-front (voxels)':>30}  {'mode':16s} blockiness")
    for t, s, name, b in rows:
        print(f"{t:>5} {s:>30}  {name:16s} {b:.2f}")
    table = "".join(f"<tr><td>{t}</td><td>{s}</td><td>{name}</td><td>{b:.2f}</td></tr>" for t, s, name, b in rows)
    figs = "".join(f"<figure><img src='data:image/png;base64,{png(im)}'><figcaption>{c}</figcaption></figure>" for c, im in cells)
    (HERE / "voxel_edges.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Voxel Edges</title><style>
:root{{--bg:#fafafa;--fg:#171717;--muted:#525252;--line:#d4d4d4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171717;--fg:#e5e5e5;--muted:#a3a3a3;--line:#404040}}}}
body{{background:var(--bg);color:var(--fg);font:13px -apple-system,system-ui,sans-serif;margin:20px;max-width:1000px}}
table{{border-collapse:collapse}}td,th{{border:1px solid var(--line);padding:3px 8px;text-align:left}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}}figure{{margin:0}}
figure img{{width:100%;image-rendering:pixelated;border:1px solid var(--line)}}figcaption{{color:var(--muted);font-size:11px}}</style></head><body>
<h2>Why EA looks interpolated and MIP looks blocky</h2>
<p>Random voxels rendered by the viewer's shader, 4 screen pixels per voxel, orthographic, no interpolation anywhere. Blockiness = share of neighboring pixel pairs with identical values (0.75 is perfectly blocky here, 0 is smooth).</p>
<table><tr><th>tilt (deg)</th><th>sideways offset, front to back (voxels)</th><th>mode</th><th>blockiness</th></tr>{table}</table>
<div class="row">{figs}</div></body></html>""")
    print(HERE / "voxel_edges.html")


if __name__ == "__main__":
    main()
