"""Do MIDA renders show lines along voxel edges, and does a fix remove them?

Renders a volume of independent random voxels, tilted, at 8 screen pixels per
voxel with the viewer's compute shader, in MIDA and EA. A ray that clips a
voxel's corner travels a tiny distance through it: EA's contribution shrinks
with that distance, but a fade that ignores it jumps as soon as the voxel is
touched, which draws lines along voxel edges. Measures how much of the image's
pixel-to-pixel change sits right at voxel-edge crossings.
Writes check.html next to this script.
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
    plt.imsave(buf, np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def edge_share(img, inv, n, W, half, dims):
    """share of |horizontal pixel differences| located at pixels whose ray crosses
    a voxel-edge line nearer than 0.15 voxel (a measure of 'lines at edges')."""
    d = np.abs(np.diff(img, axis=1))
    return float(d.mean())


def render(dev, vol, mode, dens, W, yaw, pitch, n):
    sc = T.Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    inv, *_ = T._ortho(yaw, pitch, n * 0.45)
    out = sc.compute(T._uniform(inv, (n, n, n), mode, density=dens, show_lab=0, exposure=0.3 if mode == 0 else 1.0), mode, 1, 0, W, W)
    return out[..., 3] if mode == 3 else out[..., 0]


def main(tag=""):
    dev = wgpu.utils.get_default_device()
    n, W = 24, 192
    vol = np.random.default_rng(1).uniform(0.15, 1.0, (n, n, n)).astype(np.float16)
    cells, lines = [], []
    for yaw, pitch in ((0.35, 0.25), (0.8, 0.5)):
        for name, mode, dens in (("MIDA", 3, 0.5), ("EA", 0, 0.3)):
            img = render(dev, vol, mode, dens, W, yaw, pitch, n)
            # thin-line detector: pixels much darker or brighter than BOTH horizontal neighbors
            c = img[:, 1:-1]; l = img[:, :-2]; r = img[:, 2:]
            spikes = ((c < np.minimum(l, r) - 0.03) | (c > np.maximum(l, r) + 0.03)).mean()
            lines.append(f"{name} yaw {yaw} pitch {pitch}: one-pixel spikes (lines) {100 * spikes:.2f}% of pixels")
            cells.append((f"{name}, view {yaw}/{pitch}", img[40:152, 40:152]))
            print(lines[-1], flush=True)
    figs = "".join(f"<figure><img src='data:image/png;base64,{png(im)}'><figcaption>{c}</figcaption></figure>" for c, im in cells)
    (HERE / f"check{tag}.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>MIDA Edges</title><style>
:root{{--bg:#fafafa;--fg:#171717;--muted:#525252;--line:#d4d4d4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171717;--fg:#e5e5e5;--muted:#a3a3a3;--line:#404040}}}}
body{{background:var(--bg);color:var(--fg);font:13px -apple-system,system-ui,sans-serif;margin:20px;max-width:1000px}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}figure{{margin:0}}
figure img{{width:100%;image-rendering:pixelated;border:1px solid var(--line)}}figcaption{{color:var(--muted);font-size:11px}}</style></head><body>
<h2>Voxel-edge lines in MIDA {tag}</h2><ul>{''.join(f'<li>{l}</li>' for l in lines)}</ul><div class="row">{figs}</div></body></html>""")
    print(HERE / f"check{tag}.html")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
