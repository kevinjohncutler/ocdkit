"""Bench: marching-squares vector contour pipeline vs legacy.

The library implementation now lives in ``ocdkit.plot.contour``
(``vector_contours_marching`` / ``cells_to_polygons`` / ``cells_to_webgpu_mesh``)
-- this file is just a bench runner that compares it against the legacy
``vector_contours`` and the Tier-1 prototype.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io

from ocdkit.plot.contour import (
    vector_contours,
    vector_contours_marching,
)
# Tier 1 prototype kept in scripts/ for comparison only.
from bench_vector_contours import vector_contours_fast


def _time_call(fn, repeat=3):
    fn()  # warm-up (incl numba compile)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts), ts


def main():
    mask_path = os.environ.get('OCDKIT_BENCH_MASK')
    if not mask_path:
        raise SystemExit(
            "Set OCDKIT_BENCH_MASK to the path of a label-mask PNG "
            "(e.g. export OCDKIT_BENCH_MASK=/path/to/labels.png)."
        )
    mask = skimage.io.imread(mask_path)
    print(f"mask shape={mask.shape} n_labels={len(np.unique(mask)) - 1}")

    def run_current():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        vector_contours(fig, ax, mask, smooth_factor=5, color='r', linewidth=1.0)
        ax.set_axis_off()
        plt.close(fig)

    def run_tier1():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        vector_contours_fast(fig, ax, mask, smooth_sigma=2.0, color='r', linewidth=1.0)
        ax.set_axis_off()
        plt.close(fig)

    def run_tier2():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        vector_contours_marching(fig, ax, mask, smooth_sigma=2.0, color='r', linewidth=1.0)
        ax.set_axis_off()
        plt.close(fig)

    print("\nBenchmarking (best of 3) ...")
    cur_best, _ = _time_call(run_current, repeat=3)
    print(f"  current : {cur_best*1000:6.1f} ms")
    t1_best, _ = _time_call(run_tier1, repeat=3)
    print(f"  tier 1  : {t1_best*1000:6.1f} ms  ({cur_best/t1_best:.2f}x)")
    t2_best, _ = _time_call(run_tier2, repeat=3)
    print(f"  tier 2  : {t2_best*1000:6.1f} ms  ({cur_best/t2_best:.2f}x)")

    H, W = mask.shape
    crop_y = slice(H // 3, H // 3 + 80)
    crop_x = slice(W // 3, W // 3 + 80)

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    for a in axes[0]:
        a.imshow(mask, cmap='gray', interpolation='nearest')
        a.set_axis_off()
    vector_contours(fig, axes[0, 0], mask, smooth_factor=5, color='r', linewidth=1.0)
    vector_contours_fast(fig, axes[0, 1], mask, smooth_sigma=2.0, color='r', linewidth=1.0)
    vector_contours_marching(fig, axes[0, 2], mask, smooth_sigma=2.0, color='r', linewidth=1.0)
    axes[0, 0].set_title(f"Current vector_contours\n{cur_best*1000:.1f} ms", fontsize=11)
    axes[0, 1].set_title(f"Tier 1 (graph + Gaussian + LineCollection)\n"
                         f"{t1_best*1000:.1f} ms  ({cur_best/t1_best:.2f}x)", fontsize=11)
    axes[0, 2].set_title(f"Tier 2 (marching squares numba kernel)\n"
                         f"{t2_best*1000:.1f} ms  ({cur_best/t2_best:.2f}x)", fontsize=11)

    for a in axes[1]:
        a.imshow(mask[crop_y, crop_x], cmap='gray', interpolation='nearest',
                 extent=(crop_x.start, crop_x.stop, crop_y.stop, crop_y.start))
        a.set_xlim(crop_x.start, crop_x.stop)
        a.set_ylim(crop_y.stop, crop_y.start)
        a.set_axis_off()
        a.set_title("zoomed crop", fontsize=10)
    vector_contours(fig, axes[1, 0], mask, smooth_factor=5, color='r', linewidth=1.5)
    vector_contours_fast(fig, axes[1, 1], mask, smooth_sigma=2.0, color='r', linewidth=1.5)
    vector_contours_marching(fig, axes[1, 2], mask, smooth_sigma=2.0, color='r', linewidth=1.5)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / 'bench_vector_contours_tier2.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nsaved: {out}")


if __name__ == '__main__':
    main()
