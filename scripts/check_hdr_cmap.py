"""Quick numpy-only sanity check for ocdkit.plot.hdr_cmap.

No matplotlib — imports just the colormap-lift math + cmap package.
Prints peak Y luminance (in nits) for SDR baseline vs HDR-lifted LUT
to confirm the lift actually pushes brightness up.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from ocdkit.plot.hdr_cmap import (  # noqa: E402
    make_hdr_cmap_lut,
    SDR_WHITE_NITS, HDR_PEAK_NITS_DEFAULT,
    _XYZ_FROM_SRGB, _P3_FROM_XYZ, _srgb_to_linear,
)
from cmap import Colormap  # noqa: E402

# Linear Display-P3 → relative Y (BT.2020-ish weights for P3)
P3_Y_WEIGHTS = np.array([0.2289745641, 0.6917385218, 0.0792869141])


def naive_sdr_lut(name, hdr_peak_nits=HDR_PEAK_NITS_DEFAULT, n=256):
    """Original SDR cmap converted to linear-P3, normalized to encoder peak.
    No JzAzBz lift — yardstick for what the lifted LUT improves on."""
    cm = Colormap(name)
    rgba = np.asarray(cm(np.linspace(0, 1, n)))
    lin_srgb = _srgb_to_linear(rgba[:, :3])
    XYZ_abs = lin_srgb @ _XYZ_FROM_SRGB.T * SDR_WHITE_NITS
    return np.maximum(XYZ_abs @ _P3_FROM_XYZ.T / hdr_peak_nits, 0)


def peak_nits(lut_normed, peak):
    return float((lut_normed @ P3_Y_WEIGHTS).max() * peak)


names = ['viridis', 'magma', 'inferno', 'plasma', 'cividis',
         'RdBu', 'twilight']
print(f"{'cmap':12s}  {'SDR peak':>10s}  {'HDR peak':>10s}   lift   "
      f"max linear-P3")
for nm in names:
    sdr = naive_sdr_lut(nm)
    hdr = make_hdr_cmap_lut(nm)
    p_sdr = peak_nits(sdr, HDR_PEAK_NITS_DEFAULT)
    p_hdr = peak_nits(hdr, HDR_PEAK_NITS_DEFAULT)
    print(
        f"  {nm:10s}  "
        f"{p_sdr:7.1f} nits  {p_hdr:7.1f} nits   "
        f"{p_hdr / max(p_sdr, 1e-3):4.2f}x   "
        f"SDR={float(sdr.max()):.3f}  HDR={float(hdr.max()):.3f}"
    )

# Quick visual sanity: how the brightest viridis stop differs.
sdr_v = naive_sdr_lut('viridis')
hdr_v = make_hdr_cmap_lut('viridis')
print("\nviridis brightest stop (last index):")
print(f"  SDR  lin-P3 RGB = {sdr_v[-1]}  ({sdr_v[-1] @ P3_Y_WEIGHTS * HDR_PEAK_NITS_DEFAULT:.1f} nits)")
print(f"  HDR  lin-P3 RGB = {hdr_v[-1]}  ({hdr_v[-1] @ P3_Y_WEIGHTS * HDR_PEAK_NITS_DEFAULT:.1f} nits)")

print("\nDarkest stop (purple):")
print(f"  SDR  {sdr_v[0]}  ({sdr_v[0] @ P3_Y_WEIGHTS * HDR_PEAK_NITS_DEFAULT:.2f} nits)")
print(f"  HDR  {hdr_v[0]}  ({hdr_v[0] @ P3_Y_WEIGHTS * HDR_PEAK_NITS_DEFAULT:.2f} nits)")
