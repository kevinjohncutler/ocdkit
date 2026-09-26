"""Compare 3D projection models on fluorescence volumes (CPU reference renders).

Orthographic projection along one axis, one voxel per ray step (length 1),
grayscale, values windowed to the 1st..99.5th percentile like the viewer's
automatic histogram bounds. Models:

  MIP                    max along the ray
  EA (current)           emission tied to opacity:
                           color += s * (1 - exp(-d*s)) * T ;  T *= exp(-d*s)
  EA (absorption-only)   the same per-voxel color and emission, but density sets
                         absorption only (the current model divided by d), then a
                         soft exposure curve so sums above 1 roll off to white:
                           acc   += c(s) * s * S(d*s) * T ;  T *= exp(-d*s)
                           color  = 1 - exp(-exposure * acc)
                         with S(t) = (1 - exp(-t)) / t (exact per-voxel
                         self-absorption; S(0) = 1), so d = 0 is pure emission.
                         c(s) = s for the grayscale colormap.

Writes compare.html next to this script (images embedded).
"""
import base64
import io
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def window(v):
    lo, hi = np.percentile(v, [1, 99.5])
    return np.clip((v - lo) / (hi - lo), 0, 1).astype(np.float64)


def mip(s, axis):
    return s.max(axis=axis)


def march(s, axis, step):
    s = np.moveaxis(s, axis, 0)
    out = np.zeros(s.shape[1:])
    T = np.ones(s.shape[1:])
    for k in range(s.shape[0]):
        out, T = step(s[k], out, T)
    return out


def ea_current(d):
    def step(sk, out, T):
        a = 1 - np.exp(-d * sk)
        return out + sk * a * T, T * np.exp(-d * sk)
    return step


def ea_absorption_only(d):
    def step(sk, acc, T):
        t = d * sk
        S = np.where(t > 1e-6, (1 - np.exp(-t)) / np.maximum(t, 1e-12), 1 - t / 2)
        return acc + sk * sk * S * T, T * np.exp(-t)          # color c(s)=s times emission s
    return step


def exposure(acc, e):
    return 1 - np.exp(-e * acc)


def png(img):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def volumes():
    from ocdkit.viewer.sample_image import generate_sample_volume
    from skimage import data
    img, _ = generate_sample_volume()
    yield "built-in 3D sample (widefield-blurred rods + beads)", window(img.astype(np.float64))
    nuc = data.cells3d()[:, 1].astype(np.float64)                 # nuclei, 60 x 256 x 256
    yield "skimage cells3d nuclei (confocal)", window(nuc)


def main():
    rows = []
    for name, v in volumes():
        for axis, view in ((0, "top (along z)"), (1, "side (along y)")):
            n = v.shape[axis]
            cells = [("MIP", mip(v, axis))]
            for d in (0.05, 1.0, 5.0):
                cells.append((f"EA current, density {d:g}", march(v, axis, ea_current(d))))
            # exposure set ONCE from the data at density 0 (brightest 0.5% of pixels
            # reach ~95%), then held for every density, as the viewer would do
            acc0 = march(v, axis, ea_absorption_only(0.0))
            e = -np.log(0.05) / max(np.percentile(acc0, 99.5), 1e-6)
            for d in (0.0, 0.5, 2.0):
                acc = acc0 if d == 0 else march(v, axis, ea_absorption_only(d))
                cells.append((f"EA absorption-only, density {d:g}", exposure(acc, e)))
            rows.append((f"{name}: {view}, {n} voxels deep", cells))
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>EA Models</title><style>',
            ':root{--bg:#fafafa;--fg:#171717;--muted:#525252;--line:#d4d4d4}',
            '@media (prefers-color-scheme:dark){:root{--bg:#171717;--fg:#e5e5e5;--muted:#a3a3a3;--line:#404040}}',
            'body{background:var(--bg);color:var(--fg);font:13px -apple-system,system-ui,sans-serif;margin:20px}',
            '.row{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-bottom:22px}',
            'figure{margin:0}figure img{width:100%;image-rendering:pixelated;border:1px solid var(--line)}',
            'figcaption{color:var(--muted);font-size:11px;margin-top:3px}h3{margin:6px 0}',
            '@media (max-width:900px){.row{grid-template-columns:repeat(2,1fr)}}</style></head><body>',
            '<h2>3D projection models on fluorescence data</h2>',
            '<p>Columns: MIP; the current EA at three densities; the proposed absorption-only EA (density = absorption only, fixed emission, soft exposure) at three densities. Same data window for all.</p>']
    for title, cells in rows:
        html.append(f"<h3>{title}</h3><div class='row'>")
        for cap, img in cells:
            html.append(f"<figure><img src='data:image/png;base64,{png(img)}'><figcaption>{cap}</figcaption></figure>")
        html.append("</div>")
    html.append("</body></html>")
    (HERE / "compare.html").write_text("\n".join(html))
    print(HERE / "compare.html")


if __name__ == "__main__":
    main()
