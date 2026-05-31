"""Decode the jxl-p3 SDR tile, the UHDR SDR base, and what cmap('viridis')
would produce raw, then print bright-yellow-end RGB values side by side."""
import base64, re, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import opencodecs
import opencodecs.uhdr as uhdr
from cmap import Colormap
from ocdkit.plot.svg import _srgb_to_display_p3_uint8

# 1) what cmap('viridis') would emit (sRGB bytes — the matplotlib equiv)
cm = Colormap('viridis')
t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
mpl_srgb = (np.asarray(cm(t))[:, :3] * 255 + 0.5).clip(0, 255).astype(np.uint8)

# 2) what _srgb_to_display_p3_uint8 turns the same sRGB into (jxl-p3 path)
ocdkit_p3 = _srgb_to_display_p3_uint8(mpl_srgb[None, :, :])[0]

# 3) actual bytes embedded in the jxl-p3 SDR tile
sdr_svg = (REPO / 'figures' / 'hdr_cmap_compare_sdr.svg').read_text()
jxl_b64 = re.search(r'data:image/jxl;base64,([A-Za-z0-9+/=]+)', sdr_svg).group(1)
jxl_arr = opencodecs.read(base64.b64decode(jxl_b64))
if jxl_arr.shape[-1] == 4:
    jxl_arr = jxl_arr[..., :3]
# sample at the 5 positions equivalent to the t values above (full-width
# of the embedded thumb)
H, W = jxl_arr.shape[:2]
sample_cols = np.round(t * (W - 1)).astype(int)
mid = H // 2
embedded_jxl = jxl_arr[mid, sample_cols]

# 4) actual bytes embedded in the UHDR SDR base
hdr_svg = (REPO / 'figures' / 'hdr_cmap_compare_hdr.svg').read_text()
jpeg_b64 = re.search(r'data:image/jpeg;base64,([A-Za-z0-9+/=]+)', hdr_svg).group(1)
parts = uhdr.decode(base64.b64decode(jpeg_b64), want_hdr=False, want_base=True)
base_arr = opencodecs.read(parts['base_compressed'])
if base_arr.shape[-1] == 4:
    base_arr = base_arr[..., :3]
H2, W2 = base_arr.shape[:2]
sample_cols2 = np.round(t * (W2 - 1)).astype(int)
embedded_uhdr = base_arr[H2 // 2, sample_cols2]

print(f"{'t':>6}  {'cmap sRGB':>16}  {'→ ocdkit P3':>18}  "
      f"{'embedded JXL':>18}  {'UHDR base':>16}")
for i, tv in enumerate(t):
    print(f"  {float(tv):.2f}  "
          f"{tuple(int(v) for v in mpl_srgb[i]):>16}  "
          f"{tuple(int(v) for v in ocdkit_p3[i]):>18}  "
          f"{tuple(int(v) for v in embedded_jxl[i]):>18}  "
          f"{tuple(int(v) for v in embedded_uhdr[i]):>16}")

# What's in the JXL header? Does opencodecs.read also surface color info?
print("\nopencodecs.read on the JXL bytes — checking for any color metadata:")
import opencodecs as oc
try:
    info = oc.jxl_open(base64.b64decode(jxl_b64))
    print("  jxl_open:", type(info).__name__, dir(info))
except Exception as e:
    print(f"  jxl_open failed: {e}")
