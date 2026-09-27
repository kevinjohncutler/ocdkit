"""Is EA's softness a renderer error or the nature of a ray sum?

1. Renderer: the real compute shader (headless wgpu) on a volume of perfectly
   sharp structures (a one-voxel line and a one-voxel point, no blur). If the
   march blurred, EA (density 0) would come out wider than MIP; it must equal
   the exact column sum instead.
2. Data: the built-in sample volume (widefield PSF blur) and its unblurred
   ground truth (the label masks). Sum vs max projections of each, computed in
   NumPy, plus the shader's EA on the blurred data, which must match the NumPy
   emission sum. (With the gray colormap each voxel adds color s times
   emission s, so EA at density 0 is the ray sum of s squared.)
3. A single bead through the same widefield PSF: its width under MIP and under
   a ray sum, which is the blur EA inherits from the data.
Writes check.html (images embedded) next to this script.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tests"))
import test_raymarch_compute_wgsl as T   # noqa: E402  (shader harness: Scene, _ortho, _uniform)
import wgpu.utils                        # noqa: E402


def ea_sum(scene, dims, n, k, yaw=0.0, pitch=0.0):
    """EA at density 0 with a tiny exposure, inverted to the raw emission sum."""
    inv, *_ = T._ortho(yaw, pitch, n / 2)
    out = scene.compute(T._uniform(inv, dims, 0, density=0.0, show_lab=0, exposure=k), 0, 1, 0, n, n)
    return -np.log(np.clip(1 - out[..., 0], 1e-6, 1)) / k


def mip(scene, dims, n, yaw=0.0, pitch=0.0):
    inv, *_ = T._ortho(yaw, pitch, n / 2)
    return scene.compute(T._uniform(inv, dims, 1, show_lab=0), 1, 1, 0, n, n)[..., 3]


def fwhm(profile):
    p = profile / profile.max()
    return int((p >= 0.5).sum())


def width_at(profile, frac):
    p = profile / profile.max()
    return int((p >= frac).sum())


def sharpness(img):
    """Mean gradient magnitude over the 1..99.5 percentile range (higher = crisper)."""
    lo, hi = np.percentile(img, [1, 99.5])
    g = np.hypot(*np.gradient((img - lo) / max(hi - lo, 1e-9)))
    return float(g.mean())


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
    lines = []

    # 1. sharp synthetic structures
    n, nz = 64, 48
    vol = np.zeros((nz, n, n), np.float16)
    vol[24, :, 20] = 1.0            # a one-voxel line along y
    vol[30, 40, 44] = 1.0           # a single voxel
    sc = T.Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    dims = (n, n, nz)
    k = 0.02
    ea = ea_sum(sc, dims, n, k)
    mx = mip(sc, dims, n)
    ref = vol.astype(np.float32)[:, ::-1, :].sum(0)
    err = float(np.abs(ea - ref).max())
    lines.append(f"top-down, sharp line: FWHM EA {fwhm(ea[10])} px, MIP {fwhm(mx[10])} px; "
                 f"EA vs exact column sum, max error {err:.4f}")
    for yaw, pitch in ((0.5, 0.3), (1.1, -0.4)):
        ea_o, mx_o = ea_sum(sc, dims, n, k, yaw, pitch), mip(sc, dims, n, yaw, pitch)
        r = int(np.argmax(mx_o.max(1)))
        lines.append(f"oblique view (yaw {yaw}, pitch {pitch}): line FWHM EA {fwhm(ea_o[r])} px, MIP {fwhm(mx_o[r])} px")

    # 2. the sample volume: blurred data vs its unblurred ground truth
    from ocdkit.viewer.sample_image import generate_sample_volume
    img, lab = generate_sample_volume()
    img = img.astype(np.float32)
    truth = (lab > 0).astype(np.float32)
    c = img.shape[1] // 2
    crop = np.s_[:, c - 32:c + 32, c - 32:c + 32]
    v = (img[crop] / 255.0).astype(np.float16)
    sc2 = T.Scene(dev, np.ascontiguousarray(v), np.zeros(v.shape, np.uint8))
    ea2 = ea_sum(sc2, (64, 64, v.shape[0]), 64, 0.002)
    ref2 = (v.astype(np.float32) ** 2)[:, ::-1, :].sum(0)
    rel = float(np.abs(ea2 - ref2).max() / ref2.max())
    lines.append(f"sample volume crop: shader EA vs NumPy sum of s squared, max relative error {rel:.4f}")

    # 3. one bead through the widefield PSF
    from ocdkit.viewer.sample_image import widefield_psf
    psf = widefield_psf((64, 64, 64))
    for name, pr in (("MIP", psf.max(0)), ("sum", psf.sum(0)), ("sum of s squared (EA, gray)", (psf ** 2).sum(0))):
        lines.append(f"single bead after widefield blur, {name} projection: FWHM {fwhm(pr[32])} px, "
                     f"width at 10% of peak {width_at(pr[32], 0.1)} px, at 2% {width_at(pr[32], 0.02)} px")
    proj = {
        "blurred data: MIP": img.max(0), "blurred data: sum (EA at density 0)": img.sum(0),
        "ground truth: MIP": truth.max(0), "ground truth: sum": truth.sum(0),
    }
    for name, p in proj.items():
        lines.append(f"sharpness (mean normalized gradient), {name}: {sharpness(p):.4f}")

    # 4. the same oblique view through the real shader, blurred data vs ground truth
    oblique = {}
    W, yaw, pitch = 192, 0.7, 0.45
    half = 0.5 * float(np.linalg.norm(img.shape)) * 1.05
    for tag, volf in (("blurred data", img / 255.0), ("ground truth", truth * 0.8)):
        v3 = np.ascontiguousarray(volf.astype(np.float16))
        s3 = T.Scene(dev, v3, np.zeros(v3.shape, np.uint8))
        d3 = (v3.shape[2], v3.shape[1], v3.shape[0])
        inv, *_ = T._ortho(yaw, pitch, half)
        oblique[f"{tag}: MIP (shader, oblique)"] = s3.compute(T._uniform(inv, d3, 1, show_lab=0), 1, 1, 0, W, W)[..., 3]
        acc = s3.compute(T._uniform(inv, d3, 0, density=0.0, show_lab=0, exposure=0.002), 0, 1, 0, W, W)[..., 0]
        oblique[f"{tag}: EA density 0 (shader, oblique)"] = -np.log(np.clip(1 - acc, 1e-6, 1)) / 0.002
    for name, pr in oblique.items():
        lines.append(f"sharpness, {name}: {sharpness(pr):.4f}")

    for l in lines:
        print(l)
    cells = "".join(f"<figure><img src='data:image/png;base64,{png(p)}'><figcaption>{k}</figcaption></figure>"
                    for k, p in proj.items())
    cells += f"<figure><img src='data:image/png;base64,{png(ea2)}'><figcaption>shader EA, center crop (top-down)</figcaption></figure>"
    cells2 = "".join(f"<figure><img src='data:image/png;base64,{png(p)}'><figcaption>{k}</figcaption></figure>"
                     for k, p in oblique.items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>EA Sharpness</title><style>
:root{{--bg:#fafafa;--fg:#171717;--muted:#525252;--line:#d4d4d4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171717;--fg:#e5e5e5;--muted:#a3a3a3;--line:#404040}}}}
body{{background:var(--bg);color:var(--fg);font:13px -apple-system,system-ui,sans-serif;margin:20px;max-width:1100px}}
.row{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}figure{{margin:0}}
figure img{{width:100%;image-rendering:pixelated;border:1px solid var(--line)}}figcaption{{color:var(--muted);font-size:11px}}
li{{margin:3px 0}}@media (max-width:800px){{.row{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body>
<h2>Is EA's softness an error?</h2><ul>{''.join(f'<li>{l}</li>' for l in lines)}</ul>
<p>Top-down projections of the built-in sample volume (each normalized to its own 1..99.5 percentile).</p>
<div class="row">{cells}</div>
<p>The same oblique view rendered by the viewer's shader: blurred sample data (left pair) and its unblurred ground truth (right pair).</p>
<div class="row" style="grid-template-columns:repeat(4,1fr)">{cells2}</div></body></html>"""
    (HERE / "check.html").write_text(html)
    print(HERE / "check.html")


if __name__ == "__main__":
    main()
