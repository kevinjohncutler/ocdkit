"""Image pyramids for tiled / zoomable viewers.

A zoomable tile viewer keeps a coarse→fine pyramid of each layer so it can
upload a small display-sized level first and refine to full resolution on zoom.
Two reductions:

- ``mode='mean'`` — exact 2x2 block (area) mean. Continuous data; a cell edge
  maps continuously ``sigma -> sigma/2`` at every level.
- ``mode='nearest'`` — keep the top-left sample of each 2x2 block. Label-like
  data (segmentation / ncolor) whose values must NOT blend across edges (an
  averaged group index is a meaningless in-between colour).

``pyramid_dims`` gives the same level dimensions without the data, so a viewer
can lay out tiles before any projection finishes.
"""
from __future__ import annotations

import numpy as np


def image_pyramid(arr: np.ndarray, n_levels: int = 5, mode: str = "mean"):
    """Coarsest→finest pyramid ``[(lh, lw, ndarray), ...]``; finest level is
    ``arr`` itself (no copy). ``arr`` is ``(H, W)`` or ``(H, W, C)``; dtype is
    preserved. ``mode`` is ``'mean'`` (default) or ``'nearest'`` (see module
    docstring)."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    max_lv = 1
    while (h >> max_lv) >= 1 and (w >> max_lv) >= 1 and max_lv < n_levels:
        max_lv += 1
    n_levels = max(1, min(n_levels, max_lv))

    levels = [(h, w, arr)]                      # finest first; reversed below
    cur = arr
    for _ in range(n_levels - 1):
        ch, cw = cur.shape[:2]
        hc, wc = (ch // 2) * 2, (cw // 2) * 2
        if hc < 2 or wc < 2:
            break
        if mode == "nearest":
            t = cur[:hc:2, :wc:2]               # top-left of each 2x2 — no blending
        else:
            t = cur[:hc, :wc]
            if t.ndim == 3:
                t = t.reshape(hc // 2, 2, wc // 2, 2, t.shape[2]).mean(axis=(1, 3))
            else:
                t = t.reshape(hc // 2, 2, wc // 2, 2).mean(axis=(1, 3))
        t = t.astype(arr.dtype, copy=False)
        levels.append((t.shape[0], t.shape[1], t))
        cur = t
    levels.reverse()                            # coarsest -> finest
    return levels


def pyramid_dims(h: int, w: int, n_levels: int = 5):
    """Deterministic ``[(lh, lw), ...]`` coarse→fine matching
    :func:`image_pyramid`'s halving, computable without the data."""
    nl = 1
    while (h >> nl) >= 1 and (w >> nl) >= 1 and nl < n_levels:
        nl += 1
    nl = max(1, min(n_levels, nl))
    dims = [(h, w)]
    ch, cw = h, w
    for _ in range(nl - 1):
        ch, cw = ch // 2, cw // 2
        dims.append((ch, cw))
    dims.reverse()
    return dims
