"""Demo: HDR-lifted colormaps through the css-img / SVG image_grid pipe.

Renders a grayscale gradient with viridis / magma / inferno / rdbu in:
  - SDR (uint8 P3, current default behavior)
  - HDR (float linear-P3 with values > 1.0, peak ≈ 1600 nits)

Saves figures/hdr_cmap_demo.svg — open in Safari to see HDR on an EDR
display (Chrome on macOS works too). Also prints the lifted LUTs'
out-of-SDR-range fractions so you can confirm the lift actually used
the headroom.
"""
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor':  'none',
    'savefig.facecolor': 'none',
    'axes.edgecolor':  'gray',
    'axes.labelcolor': 'gray',
    'xtick.color':     'gray',
    'ytick.color':     'gray',
    'text.color':      'gray',
    'axes.titlecolor': 'gray',
})

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from ocdkit.plot.hdr_cmap import (  # noqa: E402
    make_hdr_cmap_lut, apply_hdr_cmap,
    SDR_WHITE_NITS, HDR_PEAK_NITS_DEFAULT,
)
from ocdkit.plot import imshow  # noqa: E402

# linear Display-P3 → Y (relative luminance)
P3_Y_WEIGHTS = np.array([0.2289745641, 0.6917385218, 0.0792869141])


def lut_stats(name):
    sdr = make_hdr_cmap_lut(name, hdr_jz=0.155,
                             hdr_peak_nits=SDR_WHITE_NITS)  # SDR baseline
    hdr = make_hdr_cmap_lut(name, hdr_peak_nits=HDR_PEAK_NITS_DEFAULT)
    # both LUTs are linear-P3 normalized so 1.0 ≡ their respective peak.
    # Convert to absolute nits via Y-weighted sum × peak-nits.
    peak_sdr_nits = float((sdr @ P3_Y_WEIGHTS).max() * SDR_WHITE_NITS)
    peak_hdr_nits = float((hdr @ P3_Y_WEIGHTS).max() * HDR_PEAK_NITS_DEFAULT)
    print(
        f"  {name:12s}  "
        f"SDR peak Y={peak_sdr_nits:6.1f} nits   "
        f"HDR peak Y={peak_hdr_nits:6.1f} nits   "
        f"lift {peak_hdr_nits / max(peak_sdr_nits, 1e-3):4.2f}x"
    )


def main():
    print("Lift stats (peak_mult=7.88 ≈ 1600 nits):")
    for name in ('viridis', 'magma', 'inferno', 'plasma',
                 'cividis', 'RdBu', 'twilight'):
        lut_stats(name)

    H, W = 64, 1024
    grad = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))

    figures_dir = REPO / 'figures'
    figures_dir.mkdir(exist_ok=True)

    print("\nFull pipeline (HDR + SDR side by side):")
    cmaps = ['viridis', 'magma', 'inferno', 'plasma', 'cividis']
    for name in cmaps:
        t0 = time.perf_counter()
        rgb_hdr = apply_hdr_cmap(grad, name)
        dt = (time.perf_counter() - t0) * 1000
        peak_nits = float((rgb_hdr @ P3_Y_WEIGHTS).max() * HDR_PEAK_NITS_DEFAULT)
        print(f"  apply_hdr_cmap({name})   {dt:6.1f} ms   "
              f"linear-P3 max={float(rgb_hdr.max()):.3f}  "
              f"peak Y={peak_nits:6.1f} nits  dtype={rgb_hdr.dtype}")

    # Two routes through the new `cmap=` kwarg on imshow:
    out_hdr = figures_dir / 'hdr_cmap_demo_hdr.svg'
    out_sdr = figures_dir / 'hdr_cmap_demo_sdr.svg'

    # HDR route: pre-applied float linear-P3 tiles. (imshow's per-call
    # cmap kwarg applies one cmap to all 2D items, so to show per-cmap
    # tiles in a single grid we apply manually here.)
    hdr_tiles = [apply_hdr_cmap(grad, c) for c in cmaps]
    fig_hdr = imshow(hdr_tiles, figsize=2, titles=cmaps)
    out_hdr.write_text(fig_hdr._inner.to_string())
    print(f"\nSaved HDR demo:  {out_hdr}")

    # SDR baseline through the same `cmap=` kwarg, using hdr=False.
    from cmap import Colormap
    sdr_tiles = []
    for c in cmaps:
        cm = Colormap(c)
        rgba = np.asarray(cm(grad))
        sdr_tiles.append((rgba[..., :3] * 255).astype(np.uint8))
    fig_sdr = imshow(sdr_tiles, figsize=2, titles=cmaps)
    out_sdr.write_text(fig_sdr._inner.to_string())
    print(f"Saved SDR demo:  {out_sdr}")

    # And one figure exercising the new imshow(cmap=, hdr=True) wiring
    # directly on a single 2D array.
    out_kw = figures_dir / 'hdr_cmap_demo_kwarg.svg'
    fig_kw = imshow(grad, cmap='viridis', hdr=True, figsize=4,
                    titles='imshow(grad, cmap="viridis", hdr=True)')
    out_kw.write_text(fig_kw._inner.to_string())
    print(f"Saved kwarg demo: {out_kw}")

    # ─────────────────────────────────────────────────────────────────
    # SDR-fallback A/B test
    # ─────────────────────────────────────────────────────────────────
    # LEFT tile  : jxl-p3, uint8 sRGB-curve viridis (native SDR cmap).
    # RIGHT tile : Ultra-HDR JPEG (apply_hdr_cmap → linear-P3 float →
    #              libuhdr base + gain map). The SDR base layer of the
    #              UHDR JPEG is the *native non-lifted cmap* (via
    #              HdrCmapArray._sdr_base_p3_u8), not libuhdr's
    #              auto-tone-map — which used to desaturate bright
    #              stops (the original reason for switching off PQ-JXL).
    #
    # Expected on HDR display: RIGHT tile glows (~600 nits peak).
    # Expected on SDR display (or HDR display toggled off): LEFT and
    # RIGHT should look pixel-identical — both are the same SDR cmap.
    bumps = np.tile(np.linspace(0, 1, 1024, dtype=np.float32), (192, 1))
    out_sdr_only = figures_dir / 'hdr_cmap_compare_sdr.svg'
    out_hdr_only = figures_dir / 'hdr_cmap_compare_hdr.svg'

    fig_sdr_only = imshow(bumps, cmap='viridis', hdr=False, figsize=6,
                          titles='viridis · jxl-p3 (native SDR cmap)')
    out_sdr_only.write_text(fig_sdr_only._inner.to_string())

    fig_hdr_only = imshow(bumps, cmap='viridis', hdr=True, figsize=6,
                          titles='viridis · UHDR (HDR-lifted, SDR base = native cmap)')
    out_hdr_only.write_text(fig_hdr_only._inner.to_string())

    out_html = figures_dir / 'hdr_cmap_compare.html'
    out_html.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>HDR vs SDR cmap A/B</title>
<style>
  body {{
    background: #111; color: #ccc; font-family: -apple-system, sans-serif;
    margin: 20px; line-height: 1.45;
  }}
  .row {{ display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
  .col {{ flex: 1 1 480px; min-width: 380px; }}
  h2 {{ font-size: 14px; color: #888; margin: 6px 0; font-weight: normal; }}
  p  {{ font-size: 13px; color: #888; max-width: 80ch; }}
  code {{ color: #ddc; }}
</style></head>
<body>
<h1 style="font-size:16px;color:#aaa">HDR vs SDR cmap A/B (Ultra-HDR + native SDR base)</h1>
<p>Toggle display HDR (macOS: System Settings → Displays → "High Dynamic
Range"). On HDR the right tile glows. On SDR the two should be
pixel-identical — the UHDR's SDR base layer is the same native viridis
the left tile uses.</p>
<div class="row">
  <div class="col"><h2>SDR baseline (jxl-p3, uint8 viridis)</h2>
    <object data="hdr_cmap_compare_sdr.svg" type="image/svg+xml"
            style="width:100%"></object>
  </div>
  <div class="col"><h2>HDR UHDR (apply_hdr_cmap viridis)</h2>
    <object data="hdr_cmap_compare_hdr.svg" type="image/svg+xml"
            style="width:100%"></object>
  </div>
</div>
</body></html>""")
    print(f"Saved A/B page:  {out_html}")
    print(f"  SDR JXL:       {out_sdr_only}")
    print(f"  HDR UHDR:      {out_hdr_only}")
    print("  Open the HTML in Safari, then toggle Displays → HDR off.")
    print("  Tiles should now look identical in SDR mode; HDR mode the")
    print("  right tile gains brightness from the gain map.")


if __name__ == '__main__':
    main()
