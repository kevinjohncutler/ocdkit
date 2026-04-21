"""Spatial filtering utilities for labeled arrays."""

from .imports import *
from numba import njit

from .spatial import kernel_setup, get_neighbors


@njit
def _most_frequent(neighbor_masks):
    """Column-wise mode: for each column, return the most common value."""
    return np.array([np.bincount(row).argmax() for row in neighbor_masks.T])


def mode_filter(masks):
    """Replace each nonzero pixel with the most frequent label in its neighborhood.

    Uses ocdkit's spatial neighbor primitives and a numba-JIT inner loop
    for the per-pixel bincount. Background (0) results are replaced with
    the original label to avoid erosion.

    Parameters
    ----------
    masks : ndarray
        Integer label array (2D or ND).

    Returns
    -------
    ndarray
        Filtered label array, same shape as input.
    """
    pad = 1
    masks = np.pad(masks, pad).astype(int)
    d = masks.ndim
    shape = masks.shape
    coords = np.nonzero(masks)

    if coords[0].size == 0:
        unpad = tuple([slice(pad, -pad)] * d)
        return masks[unpad]

    steps, inds, idx, fact, sign = kernel_setup(d)
    subinds = np.concatenate(inds)
    substeps = steps[subinds]
    neighbors = get_neighbors(coords, substeps, d, shape)

    neighbor_masks = masks[tuple(neighbors)]

    mask_filt = np.zeros_like(masks)
    most_f = _most_frequent(neighbor_masks)
    z = most_f == 0
    most_f[z] = masks[coords][z]
    mask_filt[coords] = most_f

    unpad = tuple([slice(pad, -pad)] * d)
    return mask_filt[unpad]
