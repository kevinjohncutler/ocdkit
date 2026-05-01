"""Spatial primitives for ND neighbor operations, affinity graphs, and contour extraction."""

from itertools import product

from .imports import *
from numba import njit, prange
from skimage import measure
from skimage.morphology import remove_small_objects
from skimage.segmentation import expand_labels, find_boundaries as _skimage_find_boundaries

from ..io.result import Result


def kernel_setup(dim):
    """Get ND neighbor steps, grouped indices, center index, distance factors, and sign.

    Returns
    -------
    Result with attributes:

    steps : ndarray, shape (3**dim, dim)
        All neighbor offsets in the ND hypercube.
    inds : list of ndarray
        Step indices grouped by connectivity (center, cardinal, ordinal, ...).
    idx : int
        Index of the zero-step (center).
    fact : ndarray
        Euclidean distance factor per connectivity group (length dim+1).
    sign : ndarray
        Connectivity level per step (sum of absolute step components).
    """
    steps = np.array(list(product([-1, 0, 1], repeat=dim)))

    # Classify each step by connectivity level (0=center, 1=cardinal, 2=ordinal, ...)
    sign = np.sum(np.abs(steps), axis=1)
    uniq = fastremap.unique(sign)
    inds = [np.where(sign == i)[0] for i in uniq]

    idx = inds[0][0]  # center index

    # Per-group distance factor (sqrt of connectivity level)
    fact = np.sqrt(uniq)

    return Result(steps=steps, inds=inds, idx=idx, fact=fact, sign=sign)


def get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
    """Get neighbor coordinates for each foreground pixel.

    Parameters
    ----------
    coords : tuple of ndarray
        Foreground pixel coordinates, each shape ``(npix,)``.
    steps : ndarray, shape (nsteps, dim)
        Neighbor offsets from :func:`kernel_setup`.
    dim : int
        Spatial dimensionality.
    shape : tuple of int
        Image shape.
    edges : list of ndarray, optional
        Per-dimension edge indices. Default: ``[-1+pad, s-pad]`` for each dim.
    pad : int
        Edge padding.

    Returns
    -------
    neighbors : ndarray, shape (dim, nsteps, npix)
    """
    if edges is None:
        edges = [np.array([-1 + pad, s - pad]) for s in shape]

    npix = coords[0].shape[-1]
    nsteps = len(steps)

    neighbors = np.empty((dim, nsteps, npix), dtype=np.int64)

    edge_masks = []
    for d in range(dim):
        mask = np.zeros(shape[d], dtype=bool)
        valid_edges = edges[d][(edges[d] >= 0) & (edges[d] < shape[d])]
        mask[valid_edges] = True
        edge_masks.append(mask)

    for d in range(dim):
        current_mask = edge_masks[d]
        size_d = shape[d]
        for n, step_d in enumerate(steps[:, d]):
            X = coords[d] + step_d
            Xc = X.copy()
            np.clip(Xc, 0, size_d - 1, out=Xc)
            Xs = X + step_d
            np.clip(Xs, 0, size_d - 1, out=Xs)
            oob = np.logical_and(current_mask[Xc], ~current_mask[Xs])
            out = Xc
            out[oob] = coords[d][oob]
            neighbors[d, n] = out

    return neighbors


def get_neigh_inds(neighbors, coords, shape, background_reflect=False):
    """Map neighbor coordinates to pixel indices.

    Parameters
    ----------
    neighbors : ndarray, shape (dim, nsteps, npix)
    coords : tuple of ndarray
    shape : tuple of int

    Returns
    -------
    indexes : ndarray, shape (npix,)
    neigh_inds : ndarray, shape (nsteps, npix)
    ind_matrix : ndarray, same shape as image
    """
    neighbors = tuple(neighbors)
    npix = neighbors[0].shape[-1]
    indexes = np.arange(npix)
    ind_matrix = -np.ones(shape, int)
    ind_matrix[tuple(coords)] = indexes
    neigh_inds = ind_matrix[neighbors]

    if background_reflect:
        oob = np.nonzero(neigh_inds == -1)
        neigh_inds[oob] = indexes[oob[1]]
        ind_matrix[neighbors] = neigh_inds

    return indexes, neigh_inds, ind_matrix


@njit(cache=True, fastmath=True)
def _get_link_matrix(links_arr, piece_masks, inds, idx, is_link):
    """Mark (i,j) as linked if (a,b) or (b,a) is in links_arr."""
    max_label = links_arr.max() + 1
    link_set = set()
    for r in range(links_arr.shape[0]):
        a = links_arr[r, 0]
        b = links_arr[r, 1]
        if a > b:
            a, b = b, a
        link_set.add(a * max_label + b)

    for k in prange(len(inds)):
        i = inds[k]
        for j in range(piece_masks.shape[1]):
            a = piece_masks[i, j]
            b = piece_masks[idx, j]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if a * max_label + b in link_set:
                is_link[i, j] = True
    return is_link


def get_link_matrix(links, piece_masks, inds, idx, is_link):
    """Convert link tuples to array and mark linked pixels."""
    if not links:
        return is_link
    links_arr = np.array(list(links), dtype=np.int64)
    return _get_link_matrix(links_arr, piece_masks, inds, idx, is_link)


def masks_to_affinity(masks, coords, steps, inds, idx, fact, sign, dim,
                      neighbors=None, links=None, edges=None, dists=None,
                      cutoff=np.sqrt(2), spatial=False):
    """Convert label matrix to affinity graph.

    The affinity graph is an ``(3**dim, npix)`` boolean matrix where each
    entry indicates whether two adjacent pixels belong to the same object.
    """
    shape = masks.shape
    if neighbors is None:
        neighbors = get_neighbors(coords, steps, dim, shape, edges)

    is_edge = np.logical_and.reduce(
        [neighbors[d] == neighbors[d][idx] for d in range(dim)]
    )

    piece_masks = masks[tuple(neighbors)]
    is_self = piece_masks == piece_masks[idx]

    conditions = [is_self, is_edge]

    if links is not None and len(links) > 0:
        is_link = np.zeros(piece_masks.shape, dtype=np.bool_)
        is_link = get_link_matrix(links, piece_masks, np.concatenate(inds), idx, is_link)
        conditions.append(is_link)

    affinity_graph = np.logical_or.reduce(conditions)
    affinity_graph[idx] = 0  # no self connections

    if dists is not None:
        affinity_graph[is_edge] = (
            dists[tuple(neighbors)][idx][np.nonzero(is_edge)[-1]] > cutoff
        )

    return affinity_graph


def boundary_to_masks(boundaries, binary_mask=None, min_size=9,
                      dist=np.sqrt(2), connectivity=1):
    """Convert boundary map to labeled masks via expansion."""
    nlab = len(fastremap.unique(np.uint32(boundaries)))
    if binary_mask is None:
        if nlab == 3:
            inner_mask = boundaries == 1
        else:
            inner_mask = np.zeros_like(boundaries, dtype=bool)
    else:
        # skimage >= 0.26 deprecated `min_size` (strictly less than) in favour
        # of `max_size` (less than or equal).  Translate to preserve behaviour:
        # min_size=N removed sizes 1..N-1; max_size=N-1 does the same.
        inner_mask = remove_small_objects(
            measure.label((1 - boundaries) * binary_mask, connectivity=connectivity),
            max_size=max(min_size - 1, 0),
        )

    # expand_labels propagates dtype; ensure integer for label arithmetic
    if inner_mask.dtype == bool:
        inner_mask_int = inner_mask.astype(np.int32)
    else:
        inner_mask_int = inner_mask
    masks = expand_labels(inner_mask_int, dist)
    inner_bounds = (masks - inner_mask_int) > 0
    outer_bounds = _skimage_find_boundaries(masks, mode='inner', connectivity=masks.ndim)
    bounds = np.logical_or(inner_bounds, outer_bounds)
    return masks, bounds, inner_mask


@njit
def parametrize_contours(steps, labs, unique_L, neigh_inds, step_ok, csum):
    """Sort 2D contour boundaries into cyclic paths."""
    sign = np.sum(np.abs(steps), axis=1)
    contours = []
    s0 = 4  # center index for 2D (3**2 // 2)
    for l in unique_L:
        sel = labs == l
        indices = np.argwhere(sel).flatten()
        index = indices[np.argmin(csum[sel])]

        closed = 0
        contour = []
        n_iter = 0

        while not closed and n_iter < len(indices) + 1:
            contour.append(neigh_inds[s0, index])
            neighbor_inds = neigh_inds[:, index]
            step_ok_here = step_ok[:, index]
            seen = np.array([i in contour for i in neighbor_inds])
            possible_steps = np.logical_and(step_ok_here, ~seen)

            if np.sum(possible_steps) > 0:
                possible_step_indices = np.nonzero(possible_steps)[0]
                if len(possible_step_indices) == 1:
                    select = possible_step_indices[0]
                else:
                    consider_steps = steps[possible_step_indices]
                    best = np.argmin(
                        np.array([np.sum(s * steps[3]) for s in consider_steps])
                    )
                    select = possible_step_indices[best]
                neighbor_idx = neighbor_inds[select]
                index = neighbor_idx
                n_iter += 1
            else:
                closed = True
                contours.append(contour)

    return contours


def get_contour(labels, affinity_graph, coords=None, neighbors=None,
                cardinal_only=True):
    """Sort 2D boundaries into cyclic contour paths.

    Parameters
    ----------
    labels : 2D ndarray
        Label matrix.
    affinity_graph : 2D ndarray
        Pixel affinity array, ``(3**dim, npix)``.
    coords : tuple, optional
        Foreground coordinates. Computed from labels if None.
    neighbors : ndarray, optional
        Pre-computed neighbors. Computed if None.
    cardinal_only : bool
        If True, only use cardinal (face-connected) steps.

    Returns
    -------
    contour_map : ndarray — contour ordering per pixel
    contours : list of lists — pixel index sequences
    unique_L : ndarray — unique labels found
    """
    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)

    if cardinal_only:
        allowed_inds = np.concatenate(inds[1:2])
    else:
        allowed_inds = np.concatenate(inds[1:])

    shape = labels.shape
    coords = np.nonzero(labels) if coords is None else coords
    neighbors = get_neighbors(coords, steps, dim, shape) if neighbors is None else neighbors
    indexes, neigh_inds, ind_matrix = get_neigh_inds(neighbors, coords, shape)

    csum = np.sum(affinity_graph, axis=0)

    step_ok = np.zeros(affinity_graph.shape, bool)
    for s in allowed_inds:
        step_ok[s] = np.logical_and.reduce((
            affinity_graph[s] > 0,
            csum[neigh_inds[s]] < (3 ** dim - 1),
            neigh_inds[s] > -1,
        ))

    labs = labels[coords]
    unique_L = fastremap.unique(labs)

    contours = parametrize_contours(
        steps, np.int32(labs), np.int32(unique_L), neigh_inds, step_ok, csum
    )

    contour_map = np.zeros(shape, dtype=np.int32)
    for contour in contours:
        coords_t = tuple([c[contour] for c in coords])
        contour_map[coords_t] = np.arange(1, len(contour) + 1)

    return contour_map, contours, unique_L


def nd_grid_hypercube_labels(shape, side, *, center=True, dtype=np.int32):
    """Label an ND array with equal-side hypercubes of edge length *side* pixels.

    Useful as synthetic test data: tiles ``shape`` with hypercubes of size
    ``side**ndim`` and assigns each one a unique integer label. Pixels that
    fall outside the centered grid get label 0.

    Parameters
    ----------
    shape : sequence of int
        Target array shape ``(H, W, D, ...)``.
    side : int
        Edge length of each hypercube in pixels (same along all axes).
    center : bool, default True
        Center the grid inside the array; leftover margins get label 0.
    dtype : numpy dtype, default ``np.int32``
        Output dtype.

    Returns
    -------
    labels : ndarray
        Integer label map of shape *shape* with values in ``{0, 1..K}``.
    """
    shape = np.asarray(shape, dtype=int)
    if shape.ndim != 1:
        raise ValueError("shape must be 1D sequence of ints")
    if not isinstance(side, (int, np.integer)) or side < 1:
        raise ValueError("side must be a positive integer")

    counts = shape // side
    if np.any(counts <= 0):
        raise ValueError("side too large for at least one axis")

    grid_span = counts * side
    offsets = ((shape - grid_span) // 2) if center else np.zeros_like(shape)

    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    idx_axes = []
    in_bounds = np.ones(tuple(shape), dtype=bool)
    for ax, g in enumerate(grids):
        rel = g - offsets[ax]
        mask = (rel >= 0) & (rel < grid_span[ax])
        in_bounds &= mask
        idx_axes.append((rel // side).astype(int))

    # Row-major linearization
    lin = np.zeros(tuple(shape), dtype=int)
    stride = 1
    for ax in range(shape.size - 1, -1, -1):
        lin += idx_axes[ax] * stride
        stride *= counts[ax]

    labels = np.zeros(tuple(shape), dtype=dtype)
    labels[in_bounds] = lin[in_bounds] + 1
    return labels


def make_label_matrix(N, M):
    """Make a general ND quadrant/octant label matrix.

    Returns an array of shape ``(2*M,) * N`` where each axis is split into
    two halves of length ``M`` and the per-pixel label is the binary code of
    the half-indices.

    Examples
    --------
    - ``N=1`` → ``[0...0, 1...1]``
    - ``N=2`` → quadrants labeled 0..3
    - ``N=3`` → octants labeled 0..7
    - ``N=4`` → 16 hyper-quadrants labeled 0..15
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    grids = np.ogrid[tuple(slice(0, 2 * M) for _ in range(N))]
    labels = np.zeros((2 * M,) * N, dtype=int)
    for axis, g in enumerate(grids):
        half = (g // M).astype(int)     # 0 or 1
        labels += half << axis          # bit-shift
    return labels
