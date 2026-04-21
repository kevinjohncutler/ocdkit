"""Union-find connected components via numba JIT."""

from .imports import *
from numba import njit


@njit(cache=True)
def _uf_find(parent, x):
    """Find root of *x* with path halving."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path halving
        x = parent[x]
    return x


@njit(cache=True)
def _uf_union(rows, cols, parent):
    """Union all (row, col) pairs in *parent*."""
    for i in range(len(rows)):
        rx = _uf_find(parent, rows[i])
        ry = _uf_find(parent, cols[i])
        if rx != ry:
            if rx > ry:
                rx, ry = ry, rx
            parent[ry] = rx


@njit(cache=True)
def _uf_label(parent):
    """Assign contiguous labels from the union-find *parent* array."""
    n = len(parent)
    root_lbl = np.full(n, -1, dtype=np.int32)
    labels = np.zeros(n, dtype=np.int32)
    nxt = np.int32(1)
    for i in range(n):
        r = _uf_find(parent, i)
        if root_lbl[r] < 0:
            root_lbl[r] = nxt
            nxt += 1
        labels[i] = root_lbl[r]
    return labels


def cc_union_find(rows, cols, n_nodes):
    """Connected components via numba union-find (path-halving, no rank)."""
    parent = np.arange(n_nodes, dtype=np.int32)
    _uf_union(rows.astype(np.int32), cols.astype(np.int32), parent)
    return _uf_label(parent)


# Trigger JIT compilation at import time to avoid first-call latency
cc_union_find(np.array([0], dtype=np.int32), np.array([0], dtype=np.int32), 2)
