"""Benchmark current vector_contours pipeline vs a vectorized prototype.

Loads the ncolor example mask and renders both the current
ocdkit.plot.contour.vector_contours output and a prototype implementation
side by side, with timings.

Run:
    python scripts/bench_vector_contours.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from matplotlib.collections import LineCollection
from numba import njit
from scipy.ndimage import gaussian_filter1d

from ocdkit.array.spatial import (
    boundary_to_masks,
    get_neighbors,
    kernel_setup,
    masks_to_affinity,
)
from ocdkit.plot.contour import vector_contours
from skimage.segmentation import find_boundaries


# ---------------------------------------------------------------------------
# Faster numba contour walker — fixes the O(N^2) `i in contour` list lookup
# in ocdkit.array.spatial.parametrize_contours by using a global boolean
# `seen` array indexed by pixel id (O(1) membership check).
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _parametrize_contours_fast(steps, labs, unique_L, neigh_inds, step_ok, csum):
    sign = np.sum(np.abs(steps), axis=1)  # noqa: F841 (kept for parity)
    s0 = 4  # center index for 2D (3**2 // 2)
    npix = neigh_inds.shape[1]
    seen = np.zeros(npix, dtype=np.bool_)

    contours = []
    for l in unique_L:
        sel = labs == l
        indices = np.argwhere(sel).flatten()
        if len(indices) == 0:
            continue
        index = indices[np.argmin(csum[sel])]

        # local list to record this contour (numba reflected-list, OK)
        contour = [np.int64(0)]
        contour.clear()

        n_iter = 0
        max_iter = len(indices) + 1
        while n_iter < max_iter:
            here = neigh_inds[s0, index]
            contour.append(here)
            seen[here] = True

            neighbor_inds = neigh_inds[:, index]
            step_ok_here = step_ok[:, index]

            # find best unseen neighbor
            best_select = -1
            best_cost = 1 << 30
            best_count = 0
            for k in range(neighbor_inds.shape[0]):
                if not step_ok_here[k]:
                    continue
                ni = neighbor_inds[k]
                if ni < 0 or seen[ni]:
                    continue
                # directional cost = sum(steps[k] * steps[3])
                cost = 0
                for d in range(steps.shape[1]):
                    cost += steps[k, d] * steps[3, d]
                if cost < best_cost:
                    best_cost = cost
                    best_select = k
                best_count += 1

            if best_select < 0:
                # closed
                break
            index = neighbor_inds[best_select]
            n_iter += 1

        # reset seen flags for THIS contour only (so other labels can reuse)
        for px in contour:
            seen[px] = False
        contours.append(contour)

    return contours


def _get_contour_fast(labels, affinity_graph, coords, neighbors, cardinal_only=True):
    """Drop-in replacement for ocdkit.array.spatial.get_contour using the
    fast walker above. Still builds the affinity graph the same way upstream."""
    from ocdkit.array.spatial import get_neigh_inds  # late import

    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    if cardinal_only:
        allowed_inds = np.concatenate(inds[1:2])
    else:
        allowed_inds = np.concatenate(inds[1:])

    indexes, neigh_inds, ind_matrix = get_neigh_inds(neighbors, coords, labels.shape)
    csum = np.sum(affinity_graph, axis=0)
    step_ok = np.zeros(affinity_graph.shape, bool)
    for s in allowed_inds:
        step_ok[s] = np.logical_and.reduce((
            affinity_graph[s] > 0,
            csum[neigh_inds[s]] < (3 ** dim - 1),
            neigh_inds[s] > -1,
        ))

    labs = labels[coords]
    import fastremap
    unique_L = fastremap.unique(labs)
    contours = _parametrize_contours_fast(
        steps, np.int32(labs), np.int32(unique_L), neigh_inds, step_ok, csum
    )
    return contours, unique_L


# ---------------------------------------------------------------------------
# Vectorized Chaikin corner-cutting (replaces per-cell scipy.splprep loop).
# ---------------------------------------------------------------------------

def _chaikin_closed(P, iterations=2):
    """Chaikin corner-cutting on a closed polygon (vectorized, no Python loop)."""
    P = np.asarray(P, dtype=np.float64)
    for _ in range(iterations):
        Pn = np.roll(P, -1, axis=0)
        Q = 0.75 * P + 0.25 * Pn
        R = 0.25 * P + 0.75 * Pn
        out = np.empty((2 * len(P), 2), dtype=np.float64)
        out[0::2] = Q
        out[1::2] = R
        P = out
    return P


def _gaussian_smooth_closed(P, sigma=2.0):
    """Periodic Gaussian smoothing of a closed polygon (low-pass filter).

    Treats x and y as 1D periodic signals along the contour parameter and
    applies a Gaussian filter — this is the linear low-pass equivalent of
    scipy.interpolate.splprep with a smoothing factor, but ~50x cheaper.
    """
    if len(P) < 3 or sigma <= 0:
        return P
    x = gaussian_filter1d(P[:, 0], sigma=sigma, mode='wrap')
    y = gaussian_filter1d(P[:, 1], sigma=sigma, mode='wrap')
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Prototype: vectorized vector outlines
# ---------------------------------------------------------------------------

def vector_contours_fast(fig, ax, mask, smooth_sigma=2.0, color='r', linewidth=1,
                         x_offset=0, y_offset=0, pad=2, zorder=1,
                         skip_despur=False):
    """Prototype.

    - Reuses the affinity-graph machinery (it is already vectorized/numba'd).
    - Uses a fast contour walker that fixes the O(N^2) Python-list membership
      check in the original parametrize_contours.
    - Replaces the per-cell scipy.splprep loop with vectorized Chaikin
      corner-cutting (2 iterations ~ visually equivalent to a B-spline).
    - Builds ONE concatenated matplotlib Path with MOVETO/LINETO/CLOSEPOLY for
      all contours, wrapped in a single PathPatch (instead of N PathPatches).
    - Optionally skips the boundary_to_masks despur step (cheap labels
      typically don't need it).
    """
    msk = np.pad(mask, pad, mode='edge')
    msk = np.pad(msk, 1, mode='constant', constant_values=0)
    dim = msk.ndim
    shape = msk.shape

    steps, inds, idx, fact, sign = kernel_setup(dim)

    if not skip_despur:
        bd = find_boundaries(msk, mode='inner', connectivity=2)
        msk, _, _ = boundary_to_masks(bd, binary_mask=msk > 0,
                                      connectivity=1, min_size=0)

    coords = np.nonzero(msk)
    neighbors = get_neighbors(tuple(coords), steps, dim, shape)
    affinity_graph = masks_to_affinity(msk, coords, steps, inds, idx,
                                       fact, sign, dim, neighbors)
    contour_list, unique_L = _get_contour_fast(
        msk, affinity_graph, coords, neighbors, cardinal_only=True
    )

    # Build a list of (N, 2) closed polylines; one LineCollection draws them
    # all at once. Much faster than a Path with N CLOSEPOLY subpaths or N
    # PathPatches in a PatchCollection.
    cy, cx = coords
    polylines = []
    for contour in contour_list:
        if len(contour) < 3:
            continue
        c = np.asarray(contour, dtype=np.int64)
        pts = np.column_stack([cx[c], cy[c]]).astype(np.float64)
        pts[:, 0] -= (pad + 1) - x_offset
        pts[:, 1] -= (pad + 1) - y_offset
        pts = _gaussian_smooth_closed(pts, sigma=smooth_sigma)
        # close the loop by repeating the first point
        polylines.append(np.vstack([pts, pts[:1]]))

    if not polylines:
        return

    def _make_lc():
        return LineCollection(polylines, colors=color, linewidths=linewidth,
                              zorder=zorder, capstyle='round')

    if isinstance(ax, list):
        for a in ax:
            a.add_collection(_make_lc())
    else:
        ax.add_collection(_make_lc())


# ---------------------------------------------------------------------------
# Benchmark + figure
# ---------------------------------------------------------------------------

def _time_call(fn, repeat=3):
    fn()  # warm-up (incl numba compile)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts), ts


def main():
    mask = skimage.io.imread('/Volumes/DataDrive/ncolor/test_files/example.png')
    print(f"mask shape={mask.shape} n_labels={len(np.unique(mask)) - 1}")

    def run_current():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        vector_contours(fig, ax, mask, smooth_factor=5, color='r', linewidth=1.0)
        ax.set_axis_off()
        plt.close(fig)

    def run_fast():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        vector_contours_fast(fig, ax, mask, smooth_sigma=2.0, color='r', linewidth=1.0)
        ax.set_axis_off()
        plt.close(fig)

    print("\nBenchmarking current pipeline (best of 3) ...")
    cur_best, cur_all = _time_call(run_current, repeat=3)
    print(f"  current: {cur_best*1000:.1f} ms (all: {[f'{t*1000:.1f}' for t in cur_all]})")

    print("Benchmarking prototype (best of 3) ...")
    fast_best, fast_all = _time_call(run_fast, repeat=3)
    print(f"  proto:   {fast_best*1000:.1f} ms (all: {[f'{t*1000:.1f}' for t in fast_all]})")
    print(f"\nspeedup: {cur_best / fast_best:.1f}x")

    # Top row: full image. Bottom row: zoomed crop so outline detail is visible.
    H, W = mask.shape
    crop_y = slice(H // 3, H // 3 + 80)
    crop_x = slice(W // 3, W // 3 + 80)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))

    for a in axes[0]:
        a.imshow(mask, cmap='gray', interpolation='nearest')
        a.set_axis_off()
    vector_contours(fig, axes[0, 0], mask, smooth_factor=5, color='r', linewidth=1.0)
    vector_contours_fast(fig, axes[0, 1], mask, smooth_sigma=2.0, color='r', linewidth=1.0)
    axes[0, 0].set_title(f"Current vector_contours\n{cur_best*1000:.1f} ms",
                         fontsize=11)
    axes[0, 1].set_title(
        f"Prototype (fast walker + Gaussian smooth + LineCollection)\n"
        f"{fast_best*1000:.1f} ms  ({cur_best/fast_best:.1f}x speedup)",
        fontsize=11)

    for a in axes[1]:
        a.imshow(mask[crop_y, crop_x], cmap='gray', interpolation='nearest',
                 extent=(crop_x.start, crop_x.stop, crop_y.stop, crop_y.start))
        a.set_xlim(crop_x.start, crop_x.stop)
        a.set_ylim(crop_y.stop, crop_y.start)
        a.set_axis_off()
    vector_contours(fig, axes[1, 0], mask, smooth_factor=5, color='r', linewidth=1.5)
    vector_contours_fast(fig, axes[1, 1], mask, smooth_sigma=2.0, color='r', linewidth=1.5)
    axes[1, 0].set_title("zoomed crop", fontsize=10)
    axes[1, 1].set_title("zoomed crop", fontsize=10)
    fig.tight_layout()

    out = Path('/Volumes/DataDrive/ocdkit/scripts/bench_vector_contours.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nsaved: {out}")


if __name__ == '__main__':
    main()
