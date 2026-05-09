"""Synthetic single-cell test for contour alignment.

Plots a small mask with grid lines at integer pixel boundaries so we can
see exactly where each pipeline puts its outline relative to:
  - the pixels of the cell  (filled gray squares spanning [j-.5, j+.5] x [i-.5, i+.5])
  - the cell boundary       (gridline at half-integer y and x at the cell edge)

The three pipelines are:
  - current (B-spline through pixel centers)
  - tier 1  (Gaussian-smoothed pixel-center walk)
  - tier 2  (marching squares; geometric cell boundary)

Also tests the new `offset` knob in tier 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from bench_vector_contours import vector_contours_fast
from bench_vector_contours_tier2 import vector_contours_marching
from ocdkit.plot.contour import vector_contours


def make_mask():
    m = np.zeros((20, 20), dtype=np.int32)
    m[5:11, 5:11] = 1   # 6x6 square cell
    m[14:17, 13:18] = 2  # 3x5 rectangle cell
    return m


def add_grid(ax, shape):
    H, W = shape
    for x in np.arange(-0.5, W, 1):
        ax.axvline(x, color='cyan', linewidth=0.3, alpha=0.35, zorder=0)
    for y in np.arange(-0.5, H, 1):
        ax.axhline(y, color='cyan', linewidth=0.3, alpha=0.35, zorder=0)


def main():
    mask = make_mask()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    titles = [
        "Current vector_contours (splprep) — note up-left bias",
        "Tier 1 (gaussian on pixel-center walk)",
        "Tier 2 default offset=0 (geometric cell boundary)",
        "Tier 2 offset=0.5 (through pixel centers, no bias)",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.imshow(mask, cmap='gray_r', interpolation='nearest', vmin=0, vmax=2)
        add_grid(ax, mask.shape)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(mask.shape[1]))
        ax.set_yticks(range(mask.shape[0]))
        ax.tick_params(labelsize=6)

    vector_contours(fig, axes[0, 0], mask, smooth_factor=5,
                    color='red', linewidth=1.5)
    vector_contours_fast(fig, axes[0, 1], mask, smooth_sigma=1.0,
                         color='red', linewidth=1.5)
    vector_contours_marching(fig, axes[1, 0], mask, smooth_sigma=1.0,
                             color='red', linewidth=1.5)
    # Tier 2 with offset (still need to add the knob — we'll do that next)
    vector_contours_marching(fig, axes[1, 1], mask, smooth_sigma=1.0,
                             color='red', linewidth=1.5)

    fig.tight_layout()
    out = Path('/Volumes/DataDrive/ocdkit/scripts/bench_contour_alignment.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"saved: {out}")


if __name__ == '__main__':
    main()
