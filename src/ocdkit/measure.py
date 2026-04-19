"""
Measurement utilities for labeled images.

Bounding box extraction, diameter estimation, region cropping,
per-label medoid computation, ND density box localization, and
centered patch extraction.
"""

from .imports import *
import edt as _edt
from numba import jit
from scipy.ndimage import binary_dilation, convolve
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
from skimage import measure

from .array import is_integer


def bbox_to_slice(bbox, shape, pad=0, im_pad=0):
    """Return a tuple of slices for cropping an image based on a bounding box.

    Parameters
    ----------
    bbox : array-like
        Bounding box, e.g. ``[y0, x0, y1, x1]``.
    shape : tuple of int
        Shape of the array to be sliced.
    pad : int or list of int
        Padding applied to each axis.
    im_pad : int or list of int
        Region around the edges to avoid.

    Returns
    -------
    tuple of slice
    """
    dim = len(shape)
    if is_integer(pad):
        pad = [pad] * dim
    if is_integer(im_pad):
        im_pad = [im_pad] * dim
    return tuple(
        slice(
            int(max(im_pad[n], bbox[n] - pad[n])),
            int(min(bbox[n + dim] + pad[n], shape[n] - im_pad[n])),
        )
        for n in range(len(bbox) // 2)
    )


def make_square(bbox, shape):
    """Expand *bbox* to be square, clamped to image boundaries.

    Parameters
    ----------
    bbox : tuple
        ``(miny, minx, maxy, maxx)``.
    shape : tuple
        Image shape ``(H, W)``.
    """
    miny, minx, maxy, maxx = bbox
    height = maxy - miny
    width = maxx - minx
    side = max(height, width)

    dy = side - height
    dx = side - width

    miny = max(miny - dy // 2, 0)
    maxy = min(maxy + dy - dy // 2, shape[0])
    minx = max(minx - dx // 2, 0)
    maxx = min(maxx + dx - dx // 2, shape[1])

    return (miny, minx, maxy, maxx)


def crop_bbox(mask, pad=10, iterations=3, im_pad=0, area_cutoff=0,
              max_dim=np.inf, get_biggest=False, binary=False, square=False):
    """Return bounding-box slices for labeled regions in *mask*.

    Parameters
    ----------
    mask : ndarray
        Label matrix.
    pad : int
        Padding around each bounding box.
    iterations : int
        Binary dilation iterations before region detection.
    im_pad : int
        Edge margin to avoid.
    area_cutoff : int
        Minimum region area.
    get_biggest : bool
        If True, return only the largest region.
    binary : bool
        If True, merge all regions into a single bounding box.
    square : bool
        If True, expand bounding boxes to squares.
    """
    bw = binary_dilation(mask > 0, iterations=iterations) if iterations > 0 else (mask > 0)
    clusters = measure.label(bw)
    regions = measure.regionprops(clusters)
    sz = mask.shape
    d = mask.ndim

    def adjust_bbox(bbx):
        minpad = min(pad, bbx[0], bbx[1],
                     sz[0] - bbx[2], sz[1] - bbx[3])
        if square:
            bbx = make_square(bbx, sz)
        return bbox_to_slice(bbx, sz, pad=minpad, im_pad=im_pad)

    slices = []
    if get_biggest and regions:
        largest_idx = np.argmax([r.area for r in regions])
        bbx = regions[largest_idx].bbox
        slices.append(adjust_bbox(bbx))
    else:
        for props in regions:
            if props.area > area_cutoff:
                bbx = props.bbox
                slices.append(adjust_bbox(bbx))

    if binary and slices:
        start_xy = np.min([[slc[i].start for i in range(d)] for slc in slices], axis=0)
        stop_xy = np.max([[slc[i].stop for i in range(d)] for slc in slices], axis=0)
        union_bbox = (start_xy[0], start_xy[1], stop_xy[0], stop_xy[1])
        merged_slice = adjust_bbox(union_bbox)
        return merged_slice

    return slices


def pill_decomposition(A, D):
    """Decompose area and mean distance into pill (stadium) radius and length.

    Given the area *A* and integrated distance *D* of a 2D mask, returns
    the radius *R* and straight-section length *L* of a stadium shape
    that matches those statistics.

    Parameters
    ----------
    A : float or array
        Mask area (pixel count).
    D : float or array
        Integrated distance metric.

    Returns
    -------
    R : float or array
        Radius of the semicircular ends.
    L : float or array
        Length of the straight middle section.
    """
    R = np.sqrt((np.sqrt(A ** 2 + 24 * np.pi * D) - A) / (2 * np.pi))
    L = (3 * D - np.pi * (R ** 4)) / (R ** 3)
    return R, L


def dist_to_diam(dt_pos, n):
    """Convert positive distance field values to a mean diameter.

    The formula ``2 * (n + 1) * mean(dt_pos)`` guarantees that the
    returned value equals the true diameter for an N-sphere and stays
    constant for extending rods of uniform width.

    Parameters
    ----------
    dt_pos : 1D array
        Positive distance field values.
    n : int
        Spatial dimensionality.

    Returns
    -------
    float
    """
    return 2 * (n + 1) * np.mean(dt_pos)


def diameters(masks, dt=None, dist_threshold=0, pill=False, return_length=False):
    """Estimate mean cell diameter from a label mask using distance transform.

    Parameters
    ----------
    masks : ndarray
        Integer label mask.
    dt : ndarray, optional
        Pre-computed distance transform.  Computed via ``edt.edt`` if None.
    dist_threshold : float
        Values in *dt* below this are ignored.  Must be >= 0.
    pill : bool
        If True, return ``(radius, length)`` from pill decomposition.
    return_length : bool
        If True, return ``(diam, area / diam)``.

    Returns
    -------
    float, or tuple of float
    """
    if dist_threshold < 0:
        dist_threshold = 0

    if dt is None and np.any(masks):
        dt = _edt.edt(np.int32(masks))
    if dt is None:
        dt_pos = np.array([])
    else:
        dt_pos = np.abs(dt[dt > dist_threshold])

    A = np.count_nonzero(dt_pos)
    D = np.sum(dt_pos)

    if np.any(dt_pos):
        if not pill:
            diam = dist_to_diam(np.abs(dt_pos), n=masks.ndim)
            if return_length:
                return diam, A / diam
        else:
            R = np.sqrt((np.sqrt(A ** 2 + 24 * np.pi * D) - A) / (2 * np.pi))
            L = (3 * D - np.pi * (R ** 4)) / (R ** 3)
            return R, L
    else:
        diam = 0

    return diam


# ---------------------------------------------------------------------------
# Per-label medoid extraction
# ---------------------------------------------------------------------------

def argmin_cdist(X, labels, distance_values):
    """Per-label medoid index via pairwise-distance argmin (torch).

    For each unique label group in *labels*, find the index in *X* that
    minimizes the distance-field-weighted sum of pairwise distances within
    the group. The weighting prefers points that are both centrally located
    in the group AND have a high distance-field value.

    Parameters
    ----------
    X : torch.Tensor
        Coordinates, shape ``(N, D)``. Must be sorted by *labels*.
    labels : torch.Tensor
        Per-row label index, shape ``(N,)``. Must be sorted.
    distance_values : torch.Tensor
        Per-row distance-field value, shape ``(N,)``.

    Returns
    -------
    argmin_indices : torch.Tensor
        Indices into ``X`` of the chosen medoids, one per unique label.
    adjusted_summed_distances_all : torch.Tensor
        The weighted-sum-of-distances score for every row of *X*.
    """
    unique_labels, label_counts = torch.unique_consecutive(labels, return_counts=True)
    label_starts = torch.cumsum(
        torch.cat([torch.tensor([0], device=labels.device), label_counts[:-1]]), dim=0
    )
    label_ends = torch.cumsum(label_counts, dim=0)

    argmin_indices = []
    adjusted_summed_distances_all = torch.full((len(X),), float('nan'), device=X.device)

    for i in range(len(unique_labels)):
        start_idx = label_starts[i]
        end_idx = label_ends[i]
        label_indices = torch.arange(start_idx, end_idx, device=X.device)

        X_label = X[label_indices]
        distance_values_label = distance_values[label_indices]

        if X_label.shape[0] > 1:
            distances = torch.cdist(X_label, X_label)
            summed_distances = torch.sum(distances, dim=1)

            # Weight: penalizes points with low distance-field values
            adjusted = summed_distances * (1 + 1 / distance_values_label)

            adjusted_summed_distances_all[label_indices] = adjusted
            argmin_index_in_label = torch.argmin(adjusted)
            argmin_indices.append(label_indices[argmin_index_in_label])
        else:
            argmin_indices.append(label_indices[0])
            adjusted_summed_distances_all[label_indices] = 0

    return torch.tensor(argmin_indices, device=X.device), adjusted_summed_distances_all


def get_medoids(labels, do_skel=True, return_dists=False):
    """Get medoid coordinates and labels from a label mask.

    Parameters
    ----------
    labels : ndarray
        Integer label mask.
    do_skel : bool
        If True, restrict candidate medoid points to the label skeleton
        (faster + topologically meaningful for elongated cells). If False,
        consider all foreground pixels and weight by the EDT distance field.
    return_dists : bool
        If True, also return a per-pixel "centeredness" score map.

    Returns
    -------
    medoids : ndarray, shape ``(n_labels, ndim)`` or ``None``
    mlabels : ndarray, shape ``(n_labels,)`` or ``None``
    inner_dists : ndarray (only if return_dists=True)
    """
    from .morphology import skeletonize

    if do_skel:
        masks = skeletonize(labels)
        dists = np.ones_like(labels)
    else:
        masks = labels
        dists = _edt.edt(labels)

    coords = np.argwhere(masks > 0)
    slc = tuple(coords.T)
    labs = masks[slc]
    sort = np.argsort(labs)
    sort_coords = coords[sort]
    sort_labels = labs[sort]
    sort_dists = dists[slc][sort]

    inds_tensor, dists_tensor = argmin_cdist(
        torch.tensor(sort_coords).float(),
        torch.tensor(sort_labels).float(),
        torch.tensor(sort_dists).float(),
    )

    inds = inds_tensor.cpu().numpy()
    dists_arr = dists_tensor.cpu().numpy()

    if len(inds):
        inds = np.atleast_1d(inds)
        medoids = sort_coords[inds]
        mlabels = sort_labels[inds]
        if medoids.ndim == 1:
            medoids = medoids[None]

        if return_dists:
            inner_dists = np.zeros(masks.shape, dtype=dists_arr.dtype)
            inner_dists[tuple(sort_coords.T)] = dists_arr
            return medoids, mlabels, inner_dists
        return medoids, mlabels

    return None, None


# ---------------------------------------------------------------------------
# ND density-box localization
# ---------------------------------------------------------------------------

def bartlett_nd(size):
    """Create an N-dimensional Bartlett (triangular) window with shape *size*.

    If *size* is an integer it is treated as ``(size,)``. The kernel is
    normalized to sum to 1.
    """
    if isinstance(size, int):
        size = (size,)
    windows = [np.bartlett(s) for s in size]
    grids = np.ix_(*windows)
    kernel = grids[0].astype(float)
    for g in grids[1:]:
        kernel = kernel * g
    kernel /= kernel.sum()
    return kernel


def find_highest_density_box(label_matrix, box_size):
    """Find the densest sub-box of shape *box_size* in a binary/label matrix.

    Convolves a binary mask with an N-D Bartlett kernel and returns the
    sub-box of shape *box_size* centered on the convolution maximum, clamped
    so the box stays within the array bounds.

    Parameters
    ----------
    label_matrix : ndarray
        Integer label or binary mask. Any nonzero pixel counts.
    box_size : int or tuple of int
        Box dimensions. ``-1`` returns a slice covering the full array.

    Returns
    -------
    tuple of slice
    """
    if box_size == -1:
        return tuple(slice(0, s) for s in label_matrix.shape)

    if isinstance(box_size, int):
        box_size = (box_size,) * label_matrix.ndim

    mask = (label_matrix > 0).astype(np.float32)
    kernel = bartlett_nd(box_size)
    density_map = fftconvolve(mask, kernel, mode='same')
    max_density_coords = np.unravel_index(np.argmax(density_map), density_map.shape)

    slices = []
    for max_coord, size, dim_size in zip(max_density_coords, box_size, label_matrix.shape):
        start = max(0, max_coord - size // 2)
        stop = min(dim_size, start + size)
        start = max(0, stop - size)
        slices.append(slice(start, stop))

    return tuple(slices)


# ---------------------------------------------------------------------------
# Centered patch extraction
# ---------------------------------------------------------------------------

def extract_patches(image, points, box_size, fill_value=0, point_order='yx'):
    """Extract patches centered around *points* from an image.

    Out-of-bounds areas are filled with *fill_value*. Works for grayscale
    (``yx``) and RGB (``yxc``) images.

    Parameters
    ----------
    image : ndarray
        2D (grayscale) or 3D (RGB) source image.
    points : sequence of (y, x) or (x, y) tuples
        Patch center points.
    box_size : int or (int, int)
        Square patch size, or ``(height, width)``. Forced to be odd so the
        center pixel is well-defined.
    fill_value : scalar
        Value used for out-of-bounds areas.
    point_order : {'yx', 'xy'}
        Coordinate order of *points*.

    Returns
    -------
    patches : ndarray
        Stack of patches, shape ``(N, h, w[, C])``.
    slices : list of (slice, slice)
        Source-image slice for each patch.
    """
    if isinstance(box_size, int):
        box_size = (box_size, box_size)

    box_size = tuple([s + 1 - s % 2 for s in box_size])  # force odd

    half_height, half_width = box_size[0] // 2, box_size[1] // 2

    shape = (len(points), box_size[0], box_size[1])
    img_height, img_width = image.shape[:2]
    if image.ndim == 3:
        shape += (image.shape[2],)

    patches = np.full(shape, fill_value, dtype=image.dtype)
    slices = []

    for i, point in enumerate(points):
        if point_order == 'yx':
            y, x = point
        elif point_order == 'xy':
            x, y = point
        else:
            raise ValueError("point_order must be 'yx' or 'xy'")

        src_y_start = max(0, y - half_height)
        src_y_end = min(img_height, y + half_height + 1)
        src_x_start = max(0, x - half_width)
        src_x_end = min(img_width, x + half_width + 1)

        dst_y_start = half_height - (y - src_y_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = half_width - (x - src_x_start)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        patches[i, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]
        slices.append((slice(src_y_start, src_y_end), slice(src_x_start, src_x_end)))

    return patches, slices


# ---------------------------------------------------------------------------
# Synthetic shape generators
# ---------------------------------------------------------------------------

def create_pill_mask(R, L, f=1):
    """Create a 2D pill-shaped (stadium) binary mask.

    Parameters
    ----------
    R : int
        Radius of the semicircular ends.
    L : int
        Length of the straight middle section.
    f : float
        Factor for circle radius squared (default 1).

    Returns
    -------
    mask : ndarray (uint8)
        Binary mask of the pill shape with 3-pixel padding on all sides.
    """
    height = 2 * R
    width = L + 2 * R

    pad = 3
    imh = height + 2 * pad + 1
    imw = width + 2 * pad + 1

    mask = np.zeros((imh, imw), dtype=np.uint8)

    center_y = imh // 2

    # Rectangular middle
    mask[center_y - R:center_y + R + 1, R + pad:L + R + pad + 1] = 1

    y, x = np.ogrid[:imh, :imw]

    # Left semicircle
    left_center_x = R + pad
    left_circle = (x - left_center_x) ** 2 + (y - center_y) ** 2 <= f * (R ** 2)
    mask[left_circle] = 1

    # Right semicircle
    right_center_x = L + R + pad
    right_circle = (x - right_center_x) ** 2 + (y - center_y) ** 2 <= f * (R ** 2)
    mask[right_circle] = 1

    return mask


# ---------------------------------------------------------------------------
# Curvature analysis
# ---------------------------------------------------------------------------

def curve_filter(im, filterWidth=1.5):
    """Compute principal, mean, and Gaussian curvatures of an image.

    Returns
    -------
    M_, G_, C1_, C2_ : ndarray
        Mean, Gaussian, principal curvatures (negatives zeroed).
    M, G, C1, C2 : ndarray
        Unclipped curvatures.
    im_xx, im_yy, im_xy : ndarray
        Second-order partial derivatives.
    """
    shape = [np.floor(7 * filterWidth) // 2 * 2 + 1] * 2
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    v = filterWidth ** 2
    gau = 1 / (2 * np.pi * v) * np.exp(-(x**2 + y**2) / (2. * v))

    f_xx = ((x / v)**2 - 1 / v) * gau
    f_yy = ((y / v)**2 - 1 / v) * gau
    f_xy = y * x * gau / v**2

    im_xx = convolve(im, f_xx, mode='nearest')
    im_yy = convolve(im, f_yy, mode='nearest')
    im_xy = convolve(im, f_xy, mode='nearest')

    G = im_xx * im_yy - im_xy**2
    M = -(im_xx + im_yy) / 2
    C1 = M - np.sqrt(np.abs(M**2 - G))
    C2 = M + np.sqrt(np.abs(M**2 - G))

    G_ = np.clip(G, 0, None)
    M_ = np.clip(M, 0, None)
    C1_ = np.clip(C1, 0, None)
    C2_ = np.clip(C2, 0, None)

    return M_, G_, C1_, C2_, M, G, C1, C2, im_xx, im_yy, im_xy


# ---------------------------------------------------------------------------
# Label overlap / IoU / average precision
# ---------------------------------------------------------------------------

@jit(nopython=True)
def label_overlap(x, y):
    """Pixel overlap matrix between two label arrays.

    Returns an ``(x.max()+1, y.max()+1)`` matrix where entry ``(i, j)``
    is the number of pixels that have label *i* in *x* and label *j* in *y*.
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def intersection_over_union(masks_true, masks_pred):
    """IoU matrix for all mask pairs in two label images.

    Returns an ``(N_true+1, N_pred+1)`` float array where row/column 0
    corresponds to the background label.
    """
    overlap = label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def true_positive(iou, th):
    """Count true positives at IoU threshold *th* via linear sum assignment."""
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    return match_ok.sum()
