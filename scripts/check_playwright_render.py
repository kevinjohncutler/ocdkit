"""Render the three viridis variants in a real browser and extract pixel
values from the rasterized screenshot. Whatever Safari/WebKit paints to
the screen IS the ground truth — settles the "are they actually the same
colors after color management?" question without any speculation.

Outputs RGB at 5 sample positions (t = 0, 0.25, 0.5, 0.75, 1.0) for:
  - matplotlib PNG (sRGB-curve sRGB, no tag)
  - jxl-p3 SVG (sRGB→P3 gamut-compressed, tagged display-p3)
  - UHDR SVG (HDR with native-cmap SDR base, gamut tag display-p3)
"""
import base64, io, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'figure.facecolor':  'none',
    'axes.facecolor':    'none',
    'savefig.facecolor': 'none',
})

# Build a fresh 1024-wide gradient and the matplotlib PNG reference.
W, H = 1024, 96
grad = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))

fig, ax = plt.subplots(figsize=(10.24, 0.96), dpi=100)
ax.imshow(grad, cmap='viridis', aspect='auto', interpolation='nearest')
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
mpl_buf = io.BytesIO()
fig.savefig(mpl_buf, format='png', dpi=100, bbox_inches='tight',
            pad_inches=0, transparent=True)
plt.close(fig)
mpl_buf.seek(0)
mpl_png_b64 = base64.b64encode(mpl_buf.getvalue()).decode('ascii')

# Use existing SVGs for the JXL-P3 and UHDR tiles.
sdr_svg_text = (REPO / 'figures' / 'hdr_cmap_compare_sdr.svg').read_text()
hdr_svg_text = (REPO / 'figures' / 'hdr_cmap_compare_hdr.svg').read_text()

# Write a probe page with all three side by side at known px positions.
probe_html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  body {{ background: black; margin: 0; padding: 0; }}
  .row {{ display: flex; flex-direction: column; }}
  .label {{ color: gray; font-family: monospace; font-size: 12px; padding: 4px; }}
  img, object, svg {{ display: block; width: 1024px; height: 96px; image-rendering: pixelated; }}
</style></head>
<body>
<div class="row">
  <div class="label">matplotlib PNG (sRGB):</div>
  <img id="mpl" src="data:image/png;base64,{mpl_png_b64}">
  <div class="label">jxl-p3 (display-p3 tag):</div>
  <div id="sdr">{sdr_svg_text}</div>
  <div class="label">UHDR (display-p3 gamut):</div>
  <div id="hdr">{hdr_svg_text}</div>
</div>
</body></html>"""
probe_path = REPO / 'figures' / '_pw_probe.html'
probe_path.write_text(probe_html)

# Now render with playwright (webkit = Safari engine) and sample pixels.
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("playwright not installed: pip install playwright && playwright install webkit")
    sys.exit(1)

sample_ts = [0.02, 0.25, 0.5, 0.75, 0.98]

def sample_block_at_y(page, selector, y_offset, sample_ts):
    """Pull RGB values via canvas → getImageData from the rasterized DOM."""
    return page.evaluate("""
        async (args) => {
            const [sel, yOff, ts] = args;
            const el = document.querySelector(sel);
            const rect = el.getBoundingClientRect();
            const canvas = document.createElement('canvas');
            canvas.width = rect.width;
            canvas.height = rect.height;
            const ctx = canvas.getContext('2d');
            const img = (el.tagName === 'IMG') ? el : el.querySelector('img,svg');
            if (img.tagName === 'IMG') {
                await img.decode();
                ctx.drawImage(img, 0, 0, rect.width, rect.height);
            } else {
                // SVG: serialize → blob → image → drawImage
                const xml = new XMLSerializer().serializeToString(img);
                const blob = new Blob([xml], { type: 'image/svg+xml' });
                const url = URL.createObjectURL(blob);
                const i = new Image();
                i.src = url;
                await i.decode();
                ctx.drawImage(i, 0, 0, rect.width, rect.height);
                URL.revokeObjectURL(url);
            }
            const out = [];
            for (const t of ts) {
                const x = Math.round(t * (rect.width - 1));
                const data = ctx.getImageData(x, yOff, 1, 1).data;
                out.push([data[0], data[1], data[2]]);
            }
            return out;
        }
    """, [selector, y_offset, sample_ts])


with sync_playwright() as p:
    browser = p.webkit.launch()
    context = browser.new_context(viewport={'width': 1200, 'height': 600},
                                   device_scale_factor=1)
    page = context.new_page()
    page.goto(f"file://{probe_path}")
    page.wait_for_load_state('networkidle')

    mpl_px = sample_block_at_y(page, '#mpl', 48, sample_ts)
    sdr_px = sample_block_at_y(page, '#sdr', 48, sample_ts)
    hdr_px = sample_block_at_y(page, '#hdr', 48, sample_ts)

    # Also take a screenshot for visual inspection.
    page.screenshot(path=str(REPO / 'figures' / '_pw_probe_screenshot.png'))
    browser.close()

print(f"{'t':>6}  {'mpl PNG':>16}  {'jxl-p3':>16}  {'UHDR':>16}")
for i, t in enumerate(sample_ts):
    print(f"  {t:.2f}  {str(tuple(mpl_px[i])):>16}  "
          f"{str(tuple(sdr_px[i])):>16}  {str(tuple(hdr_px[i])):>16}")

print(f"\nScreenshot: {REPO / 'figures' / '_pw_probe_screenshot.png'}")
