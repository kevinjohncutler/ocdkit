"""Direct pixel diff between the two A/B SVGs:
   - hdr_cmap_compare_sdr.svg → decode the embedded JXL.
   - hdr_cmap_compare_hdr.svg → decode the UHDR JPEG's SDR base.

If the byte diff is below JPEG-q95 noise (~3-5 max, <1 mean), the two
tiles contain the same SDR pixels, and any visible difference on screen
is browser/decoder behavior, not our encoder."""
import base64
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import opencodecs
import opencodecs.uhdr as uhdr

figs = REPO / 'figures'
sdr_svg = (figs / 'hdr_cmap_compare_sdr.svg').read_text()
hdr_svg = (figs / 'hdr_cmap_compare_hdr.svg').read_text()


def extract_b64(svg, mime):
    m = re.search(fr'data:{re.escape(mime)};base64,([A-Za-z0-9+/=]+)', svg)
    if not m:
        raise SystemExit(f"no {mime} in svg")
    return base64.b64decode(m.group(1))


# SDR tile: plain JXL with display-p3 color tag.
sdr_jxl_bytes = extract_b64(sdr_svg, 'image/jxl')
sdr_tile_u8 = opencodecs.read(sdr_jxl_bytes)
if sdr_tile_u8.shape[-1] == 4:
    sdr_tile_u8 = sdr_tile_u8[..., :3]
print(f"SDR tile JXL → {sdr_tile_u8.shape} {sdr_tile_u8.dtype}")

# HDR tile: UHDR JPEG. Pull the raw SDR base out via libuhdr's api-4
# decode and decode the inner JPEG with imagecodecs.
hdr_jpeg = extract_b64(hdr_svg, 'image/jpeg')
parts = uhdr.decode(hdr_jpeg, want_hdr=False, want_gainmap=False, want_base=True)
base_jpeg = parts['base_compressed']
sdr_base_u8 = opencodecs.read(base_jpeg)
if sdr_base_u8.shape[-1] == 4:
    sdr_base_u8 = sdr_base_u8[..., :3]
print(f"UHDR SDR base → {sdr_base_u8.shape} {sdr_base_u8.dtype}")

if sdr_tile_u8.shape != sdr_base_u8.shape:
    print(f"\nSHAPE MISMATCH — can't diff directly")
    sys.exit(1)

diff = sdr_tile_u8.astype(np.int16) - sdr_base_u8.astype(np.int16)
print(f"\nPixel-level diff (SDR-JXL vs UHDR-SDR-base):")
print(f"  max abs:  {int(np.abs(diff).max())}")
print(f"  mean abs: {float(np.abs(diff).mean()):.2f}")
print(f"  p99 abs:  {int(np.percentile(np.abs(diff), 99))}")

# Per-row diff to spot banding (which would alternate across rows in
# a column-gradient image — i.e. the gradient is column-only, all rows
# identical).
mid = sdr_tile_u8.shape[0] // 2
print(f"\nSample row {mid}, first 10 columns:")
print(f"  SDR JXL:   {sdr_tile_u8[mid, :10].tolist()}")
print(f"  UHDR base: {sdr_base_u8[mid, :10].tolist()}")

# Banding signature: count unique colors per row. Banding from a
# 256-stop LUT would compress 1024 columns into ≤256 unique values.
unique_jxl = len(np.unique(sdr_tile_u8[mid].view(np.dtype((np.void,
    sdr_tile_u8[mid].dtype.itemsize * 3)))))
unique_uhdr = len(np.unique(sdr_base_u8[mid].view(np.dtype((np.void,
    sdr_base_u8[mid].dtype.itemsize * 3)))))
print(f"\nUnique colors in mid row "
      f"({sdr_tile_u8.shape[1]} cols wide):")
print(f"  SDR JXL:   {unique_jxl}")
print(f"  UHDR base: {unique_uhdr}")
