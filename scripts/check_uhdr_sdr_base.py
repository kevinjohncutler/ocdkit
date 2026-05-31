"""Decode the UHDR JPEG embedded in the A/B comparison and check whether
the SDR base layer actually matches the native viridis we passed in.

If they match: the encoder is honoring our `sdr=` arg, and any drift the
user sees on SDR display is a viewer/EDR-headroom issue, not our base.

If they differ: opencodecs isn't routing `sdr=` to libuhdr correctly, and
we need a different code path (encode_native or encode_assembled).
"""
import base64
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import opencodecs.uhdr as uhdr
from cmap import Colormap
from ocdkit.plot.svg import _srgb_to_display_p3_uint8
from ocdkit.plot.hdr_cmap import apply_hdr_cmap

# 1) Decode the SDR base of the HDR comparison SVG.
svg_path = REPO / 'figures' / 'hdr_cmap_compare_hdr.svg'
svg = svg_path.read_text()
m = re.search(r'data:image/jpeg;base64,([A-Za-z0-9+/=]+)', svg)
if not m:
    raise SystemExit(f"No jpeg data URL found in {svg_path}")
jpeg_bytes = base64.b64decode(m.group(1))
parts = uhdr.decode(jpeg_bytes, want_hdr=False, want_gainmap=True, want_base=True)
print("UHDR decode keys:", sorted(parts.keys()))
print(f"  width={parts['width']} height={parts['height']}")
print(f"  gainmap shape: {parts['gainmap_u8'].shape}, dtype: {parts['gainmap_u8'].dtype}")
print(f"  gainmap metadata:")
for k, v in parts['gainmap_metadata'].items():
    print(f"    {k}: {v}")

base_jpeg = parts['base_compressed']
print(f"  raw SDR base JPEG: {len(base_jpeg)} bytes (starts with {base_jpeg[:4].hex()})")

# Decode the raw SDR JPEG via imagecodecs/Pillow-free path.
import opencodecs
sdr_decoded = opencodecs.read(base_jpeg)  # returns (H, W, 3|4) uint8
if sdr_decoded.ndim == 3 and sdr_decoded.shape[-1] == 4:
    sdr_decoded = sdr_decoded[..., :3]
print(f"\nDecoded SDR base shape: {sdr_decoded.shape}, dtype: {sdr_decoded.dtype}")

# 2) Build what the native SDR base SHOULD look like at the thumb's size.
H, W = sdr_decoded.shape[:2]
bumps_thumb = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
cm = Colormap('viridis')
native_srgb_u8 = (np.asarray(cm(bumps_thumb))[..., :3] * 255 + 0.5).clip(0, 255).astype(np.uint8)
native_p3_u8 = _srgb_to_display_p3_uint8(native_srgb_u8)
print(f"Native expected shape:   {native_p3_u8.shape}")

# 3) Compare.
sdr_rgb = sdr_decoded
diff = sdr_rgb.astype(np.int16) - native_p3_u8.astype(np.int16)
print(f"\nSDR-base vs native cmap:")
print(f"  max abs diff: {int(np.abs(diff).max())}")
print(f"  mean abs diff: {float(np.abs(diff).mean()):.2f}")

# 4) Also pull the SDR base our HdrCmapArray carries (before any resize),
#    to confirm that input → encoder is what we expect.
ref_full = apply_hdr_cmap(np.tile(np.linspace(0, 1, 1024, dtype=np.float32), (192, 1)),
                          'viridis')
print(f"\nHdrCmapArray._sdr_base_p3_u8 shape (pre-resize): {ref_full._sdr_base_p3_u8.shape}")
print(f"  sample row middle of HDR-tile SDR-base:    "
      f"R,G,B = {tuple(int(v) for v in ref_full._sdr_base_p3_u8[96, 512])}")
print(f"  same x in 1024-wide native cmap (sRGB→P3): "
      f"R,G,B = {tuple(int(v) for v in _srgb_to_display_p3_uint8((np.asarray(cm(np.array([0.5], dtype=np.float32))) * 255 + 0.5).clip(0,255).astype(np.uint8)[None])[0,0])}")
