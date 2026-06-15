"""Sanity check for ``image_grid(fontcolor='auto')``.

Renders two side-by-side 4x4 grids of uniform-gray tiles:

* **SDR grid** — uint8 tiles, displayed at their literal sRGB brightness.
  Labels flip at ~50% (linear ~0.5) — light on dark cells, dark on light.

* **HDR grid** — linear-light float tiles encoded as UHDR JPEG. On an
  HDR-capable browser the cells render up to ``sdr_white_nits`` (1600 nits
  by default), so a "low" linear value (~0.06) already displays at the
  same brightness as the SDR-white label. Labels flip *earlier* than 50%
  to keep contrast against the perceived display brightness.

Outputs:

* ``figures/image_grid_auto_color.html`` — interactive HTML shell.
  Open in a browser; toggle OS / browser dark mode to verify the page
  swaps but the per-cell label colours stay content-locked.
* ``figures/image_grid_auto_color.svg`` — plain SVG of the SDR grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from ocdkit.plot.image_grid import image_grid  # noqa: E402


N = 4  # 4x4 = 16 shades from 0 → 1
TILE_PX = 96


def _make_grid(*, hdr: bool):
    shades = np.linspace(0.0, 1.0, N * N)
    tiles = []
    labels = []
    for v in shades:
        if hdr:
            arr = np.full((TILE_PX, TILE_PX, 3), float(v), dtype=np.float32)
        else:
            arr = np.full((TILE_PX, TILE_PX, 3), int(round(v * 255)),
                          dtype=np.uint8)
        tiles.append(arr)
        labels.append(f'{v:.2f}')
    return image_grid(
        tiles, ncol=N, plot_labels=labels,
        fontcolor='auto',
        fontsize=10, figsize=N, lpos='top_middle',
    )


def main() -> None:
    sdr_fig = _make_grid(hdr=False)
    hdr_fig = _make_grid(hdr=True)

    fig_dir = REPO_ROOT / 'figures'
    fig_dir.mkdir(exist_ok=True)
    html_path = fig_dir / 'image_grid_auto_color.html'
    svg_path = fig_dir / 'image_grid_auto_color.svg'

    sdr_html = sdr_fig._repr_mimebundle_()['text/html']
    hdr_html = hdr_fig._repr_mimebundle_()['text/html']
    html_body = (
        '<div style="display:flex;gap:32px;align-items:flex-start;flex-wrap:wrap">'
        f'<div><h3>SDR (uint8 tiles)</h3>{sdr_html}</div>'
        f'<div><h3>HDR (linear-light float, encoded as UHDR)</h3>{hdr_html}</div>'
        '</div>'
    )
    # Standalone HTML wrapper: honour the host's color scheme but also
    # let the user force a background to verify contrast either way.
    full_html = (
        '<!DOCTYPE html>\n'
        '<html><head><meta charset="utf-8">'
        '<title>image_grid auto color check</title>'
        '<style>'
        'html { color-scheme: light dark; }'
        'body { font-family: system-ui, sans-serif; '
        '       margin: 24px; '
        '       background: light-dark(#ffffff, #1e1e1e); '
        '       color: light-dark(#1a1a1a, #f0f0f0); }'
        '.swatch { display: flex; gap: 16px; margin-bottom: 12px; }'
        '.swatch button { padding: 6px 10px; cursor: pointer; }'
        '</style></head><body>'
        '<h2>image_grid(fontcolor=\'auto\') — adaptive label check</h2>'
        '<p>Labels should stay readable against every tile shade.'
        ' Switch your OS / browser theme to flip light↔dark.</p>'
        '<div class="swatch">'
        '<button onclick="document.documentElement.style.colorScheme=\'light\'">'
        'Force light</button>'
        '<button onclick="document.documentElement.style.colorScheme=\'dark\'">'
        'Force dark</button>'
        '<button onclick="document.documentElement.style.colorScheme=\'\'">'
        'Follow system</button>'
        '</div>'
        f'{html_body}'
        '</body></html>'
    )
    html_path.write_text(full_html)
    sdr_fig.savefig(svg_path)

    print(f'HTML: {html_path}')
    print(f'SVG:  {svg_path}')


if __name__ == '__main__':
    main()
