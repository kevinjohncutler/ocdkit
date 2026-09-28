"""Which render gives each voxel the flat, uniform cube look MIP has?

A dense 6^3 patch of random voxels seen corner-on through the viewer's compute
shader, 40 screen pixels per voxel. "Flatness" is the share of pixels equal to
their right-hand neighbor (within 0.002): MIP shows one voxel per pixel, so
every footprint is flat; accumulation modes mix several voxels per pixel.
Writes compare.html next to this script.
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


def png(img):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(img, 0, 1), cmap="magma", vmin=0, vmax=1, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    dev = wgpu.utils.get_default_device()
    n, W = 6, 256
    vol = np.random.default_rng(4).uniform(0.1, 1.0, (n, n, n)).astype(np.float16)
    sc = T.Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    inv, *_ = T._ortho(math.radians(45), math.asin(1 / math.sqrt(3)), n * 0.55)
    cases = [("MIP", 1, 0.0, 0), ("EA, exact", 0, 0.0, 0), ("EA, voxel faces", 0, 0.0, 1),
             ("EA, voxel faces, density 3", 0, 3.0, 1), ("MIDA, voxel faces", 3, 0.5, 1),
             ("MIDA, voxel faces, opacity 5", 3, 5.0, 1)]
    figs, lines = [], []
    for name, mode, dens, faces in cases:
        u = T._uniform(inv, (n, n, n), mode, density=dens, show_lab=0, exposure=0.12 if dens == 0 else 1.0)
        img = sc.compute(u, mode, 1, 0, W, W, faces=faces)[..., 3]
        inner = img[40:216, 40:216]
        flat = float((np.abs(np.diff(inner, axis=1)) < 0.002).mean())
        img = img / max(float(img.max()), 1e-6)
        lines.append(f"{name}: flatness {100 * flat:.0f}%")
        figs.append(f"<figure><img src='data:image/png;base64,{png(img)}'><figcaption>{name}<br>flatness {100 * flat:.0f}%</figcaption></figure>")
        print(lines[-1], flush=True)
    (HERE / "compare.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Voxel Look</title><style>
:root{{--bg:#fafafa;--fg:#171717;--muted:#525252;--line:#d4d4d4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171717;--fg:#e5e5e5;--muted:#a3a3a3;--line:#404040}}}}
body{{background:var(--bg);color:var(--fg);font:13px -apple-system,system-ui,sans-serif;margin:20px;max-width:1100px}}
.row{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}figure{{margin:0}}
figure img{{width:100%;image-rendering:pixelated;border:1px solid var(--line)}}figcaption{{color:var(--muted);font-size:12px}}</style></head><body>
<h2>Which render gives voxels MIP's flat cube look?</h2>
<p>A 6x6x6 patch of random voxels, corner-on, 40 pixels per voxel. Flatness: share of pixels equal to their neighbor (a flat footprint per voxel scores high).</p>
<div class="row">{''.join(figs)}</div></body></html>""")
    print(HERE / "compare.html")


if __name__ == "__main__":
    main()
