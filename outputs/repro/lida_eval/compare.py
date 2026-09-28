"""Would LIDA (Liang et al. 2012) show anything MIDA misses on our data?

NumPy reference renders along a volume axis (one ray per pixel, one sample per
voxel), all on intensity with the colormap applied at the end, as the viewer's
MIDA does:
  MIP    max along the ray
  MIDA   the viewer's recurrence: beta = 1 - max(s - running max, 0)
  LIDA   the paper's recipe: smooth each ray profile (Gaussian, standing in for
         their Gaussian prefilter + moving least squares), split it at local
         minima into feature regions, keep regions with
         (1 - lam) * mean intensity + lam * mean gradient magnitude >= T, and
         inside them fade on every change: beta = 1 - |p_i - p_{i-1}| (Eq. 4)
Opacity per voxel is 1 - exp(-kappa * s) for both MIDA and LIDA. Also a
synthetic case built for LIDA: a dim blob directly behind a bright one.
Usage: VOL=<(Z, Y, X) tif> compare.py   (writes compare.html next to this script)
"""
import base64
import io
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

HERE = Path(__file__).resolve().parent
KAPPA, SIGMA, LAM = 0.5, 2.0, 0.5


def window(v):
    lo, hi = np.percentile(v, [1, 99])
    return np.clip((v - lo) / (hi - lo), 0, 1)


def mida(s, axis):
    s = np.moveaxis(s, axis, 0)
    I = np.zeros(s.shape[1:]); A = np.zeros(s.shape[1:]); f = np.zeros(s.shape[1:])
    for si in s:
        a = 1 - np.exp(-KAPPA * si)
        beta = 1 - np.maximum(si - f, 0)
        keep = beta * A
        I = beta * I + (1 - keep) * a * si
        A = keep + (1 - keep) * a
        f = np.maximum(f, si)
    return np.where(A > 1e-6, I / np.maximum(A, 1e-6), 0)


def lida(s, axis, T):
    grad = np.sqrt(sum(g ** 2 for g in np.gradient(gaussian_filter(s, 1.0))))
    grad = np.clip(grad / max(np.percentile(grad, 99.9), 1e-9), 0, 1)
    s, grad = np.moveaxis(s, axis, 0), np.moveaxis(grad, axis, 0)
    p = gaussian_filter1d(s, SIGMA, axis=0)                     # smoothed ray profiles
    n = s.shape[0]
    # transition points: local minima of the smoothed profile -> region ids
    dmin = np.zeros_like(p, bool)
    dmin[1:-1] = (p[1:-1] < p[:-2]) & (p[1:-1] <= p[2:])
    region = np.cumsum(dmin, axis=0)
    # per-region selection score, (1 - lam) * mean s + lam * mean |grad|
    flat_r = region.reshape(n, -1)
    sel = np.zeros_like(p, bool)
    for j in range(flat_r.shape[1]):
        r = flat_r[:, j]
        ss, gg = s.reshape(n, -1)[:, j], grad.reshape(n, -1)[:, j]
        cnt = np.bincount(r)
        score = (1 - LAM) * np.bincount(r, ss) / np.maximum(cnt, 1) + LAM * np.bincount(r, gg) / np.maximum(cnt, 1)
        sel.reshape(n, -1)[:, j] = score[r] >= T
    I = np.zeros(s.shape[1:]); A = np.zeros(s.shape[1:]); prev = p[0]
    for i in range(n):
        a = 1 - np.exp(-KAPPA * s[i])
        beta = np.where(sel[i] & (i > 0), 1 - np.abs(p[i] - prev), 1.0)
        keep = beta * A
        I = beta * I + (1 - keep) * a * s[i]
        A = keep + (1 - keep) * a
        prev = p[i]
    return np.where(A > 1e-6, I / np.maximum(A, 1e-6), 0)


def png(img):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def synthetic():
    z, y, x = np.mgrid[0:48, 0:64, 0:64]
    bg = np.random.default_rng(0).normal(0.05, 0.02, z.shape)
    bright = np.exp(-((z - 14) ** 2 + (y - 32) ** 2 + (x - 26) ** 2) / (2 * 4.0 ** 2))
    dim = 0.45 * np.exp(-((z - 34) ** 2 + (y - 32) ** 2 + (x - 38) ** 2) / (2 * 4.0 ** 2))
    return np.clip(bg + bright + dim, 0, 1)


def main():
    import tifffile
    cases = [("synthetic: dim blob partly behind a bright one", synthetic())]
    from ocdkit.viewer.sample_image import generate_sample_volume
    cases.append(("built-in sample (widefield)", window(generate_sample_volume()[0].astype(np.float64))))
    if os.environ.get("VOL"):
        cases.append(("your stack", window(tifffile.imread(os.environ["VOL"]).astype(np.float64))))
    rows, lines = [], []
    for name, v in cases:
        for axis, view in ((0, "along z"), (1, "along y")):
            m_ip, m_da = v.max(axis), mida(v, axis)
            l0, l25 = lida(v, axis, 0.0), lida(v, axis, 0.25)
            d = np.abs(l0 - m_da)
            lines.append(f"{name}, {view}: LIDA (T=0) vs MIDA, mean |diff| {d.mean():.3f}, "
                         f"pixels differing by more than 0.05: {100 * (d > 0.05).mean():.1f}%, by more than 0.15: {100 * (d > 0.15).mean():.1f}%")
            rows.append((f"{name}, {view}", [("MIP", m_ip), ("MIDA", m_da), ("LIDA T=0", l0), ("LIDA T=0.25", l25),
                                             ("|LIDA T=0 - MIDA| x4", np.clip(4 * d, 0, 1))]))
            print(lines[-1], flush=True)
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>LIDA vs MIDA</title><style>',
            ':root{--bg:#fafafa;--fg:#171717;--muted:#525252;--line:#d4d4d4}',
            '@media (prefers-color-scheme:dark){:root{--bg:#171717;--fg:#e5e5e5;--muted:#a3a3a3;--line:#404040}}',
            'body{background:var(--bg);color:var(--fg);font:13px -apple-system,system-ui,sans-serif;margin:20px;max-width:1200px}',
            '.row{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:18px}figure{margin:0}',
            'figure img{width:100%;image-rendering:pixelated;border:1px solid var(--line)}figcaption{color:var(--muted);font-size:11px}',
            'li{margin:3px 0}@media (max-width:800px){.row{grid-template-columns:repeat(2,1fr)}}</style></head><body>',
            '<h2>LIDA vs MIDA on our data</h2><ul>' + ''.join(f'<li>{l}</li>' for l in lines) + '</ul>']
    for title, cells in rows:
        html.append(f"<h3>{title}</h3><div class='row'>")
        html += [f"<figure><img src='data:image/png;base64,{png(im)}'><figcaption>{c}</figcaption></figure>" for c, im in cells]
        html.append("</div>")
    html.append("</body></html>")
    (HERE / "compare.html").write_text("\n".join(html))
    print(HERE / "compare.html")


if __name__ == "__main__":
    main()
