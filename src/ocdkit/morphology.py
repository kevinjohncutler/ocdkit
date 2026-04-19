"""
Morphology utilities for labeled images.

Skeletonization and boundary detection that are faster than their
skimage equivalents for labeled (instance) segmentation masks.
"""

from .imports import *
import skimage.morphology

from .spatial import kernel_setup


def find_boundaries(labels, connectivity=1, use_symmetry=False):
    """Compute boundaries of labeled instances in an N-dimensional array.

    Replicates ``skimage.segmentation.find_boundaries`` with
    ``mode='inner'``, but is much faster.
    """
    boundaries = np.zeros_like(labels, dtype=bool)
    ndim = labels.ndim
    shape = labels.shape

    steps, inds, idx, fact, sign = kernel_setup(ndim)

    if use_symmetry:
        allowed_inds = []
        for i in range(1, 1 + connectivity):
            j = inds[i][:len(inds[i]) // 2]
            allowed_inds.append(j)
        allowed_inds = np.concatenate(allowed_inds)
    else:
        allowed_inds = np.concatenate(inds[1:1 + connectivity])

    shifts = steps[allowed_inds]

    if use_symmetry:
        for shift in shifts:
            slices_main = tuple(
                slice(max(-s, 0), min(shape[d] - s, shape[d]))
                for d, s in enumerate(shift)
            )
            slices_shifted = tuple(
                slice(max(s, 0), min(shape[d] + s, shape[d]))
                for d, s in enumerate(shift)
            )
            boundary_main = (
                (labels[slices_main] != labels[slices_shifted])
                & (labels[slices_main] != 0)
            )
            boundary_shifted = (
                (labels[slices_shifted] != labels[slices_main])
                & (labels[slices_shifted] != 0)
            )
            boundaries[slices_main] |= boundary_main
            boundaries[slices_shifted] |= boundary_shifted
    else:
        for shift in shifts:
            slices_main = tuple(
                slice(max(-s, 0), min(shape[d] - s, shape[d]))
                for d, s in enumerate(shift)
            )
            slices_shifted = tuple(
                slice(max(s, 0), min(shape[d] + s, shape[d]))
                for d, s in enumerate(shift)
            )
            boundaries[slices_main] |= (
                (labels[slices_main] != labels[slices_shifted])
                & (labels[slices_main] != 0)
            )

    return boundaries.astype(np.uint8)


def skeletonize(labels, dt_thresh=1, dt=None, method='zhang'):
    """Skeletonize labeled instances, preserving label identity.

    When *dt* is provided, pixels with ``dt > dt_thresh`` are used as the
    interior mask (fast path).  Otherwise, boundaries are removed first and
    missing labels are re-attached after thinning.

    Parameters
    ----------
    labels : ndarray
        Integer label matrix.
    dt_thresh : float
        Distance-transform threshold for the fast path.
    dt : ndarray, optional
        Pre-computed distance transform.
    method : str
        Skeletonization method (``'zhang'`` or ``'lee'``).
    """
    if dt is not None:
        inner = dt > dt_thresh
        skel = skimage.morphology.skeletonize(inner, method=method)
        return skel * labels

    bd = find_boundaries(labels, connectivity=2)
    inner = np.logical_xor(labels > 0, bd)
    skel = skimage.morphology.skeletonize(inner, method=method)
    skeleton = skel * labels

    original_labels = fastremap.unique(labels)
    original_labels = original_labels[original_labels != 0]

    skeleton_labels = fastremap.unique(skeleton)
    skeleton_labels = skeleton_labels[skeleton_labels != 0]

    missing_labels = np.setdiff1d(original_labels, skeleton_labels)
    missing_labels_mask = fastremap.mask_except(labels, list(missing_labels))
    skeleton += missing_labels_mask

    return skeleton


def masks_to_outlines(masks, omni=False, mode="inner", connectivity=None):
    """Return a 0/1 outline mask for labeled instances.

    Parameters
    ----------
    masks : ndarray
        2D or 3D label matrix. 3D inputs are processed slice-by-slice.
    omni : bool
        If True, use the fast native :func:`find_boundaries` (treats the
        label matrix directly). If False, use the legacy per-label
        erosion path (slower, iterates over each label individually).
    mode : str
        Forwarded to :func:`find_boundaries` (omni=True) or
        ``skimage.segmentation.find_boundaries``-equivalent semantics
        (omni=False, only ``"inner"`` is supported).
    connectivity : int, optional
        Forwarded to :func:`find_boundaries` when ``omni=True``. Defaults
        to ``masks.ndim``.
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            f"masks_to_outlines takes 2D or 3D array, not {masks.ndim}D array"
        )

    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(
                masks[i], omni=omni, mode=mode, connectivity=connectivity,
            )
        return outlines

    if omni:
        if connectivity is None:
            connectivity = masks.ndim
        return find_boundaries(masks, connectivity=connectivity).astype(bool)

    # Legacy per-label erosion path.
    from scipy.ndimage import binary_erosion, find_objects
    slices = find_objects(masks.astype(int))
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            mask = (masks[sr, sc] == (i + 1))
            boundary = mask & ~binary_erosion(mask)
            pvr, pvc = np.nonzero(boundary)
            vr, vc = pvr + sr.start, pvc + sc.start
            outlines[vr, vc] = 1
    return outlines


def hysteresis_threshold(image, low, high):
    """PyTorch implementation of ``skimage.filters.apply_hysteresis_threshold``.

    Supports 2D and 3D spatial inputs (expects batch+channel dims, i.e.
    ``(B, C, *spatial)``).  Minor discrepancies vs. skimage occur for very
    high thresholds on thin objects.
    """
    import torch

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    mask_low = image > low
    mask_high = image > high
    thresholded = mask_low.clone()

    spatial_dims = len(image.shape) - 2
    kernel_size = [3] * spatial_dims
    hysteresis_kernel = torch.ones([1, 1] + kernel_size, device=image.device, dtype=image.dtype)

    thresholded_old = torch.zeros_like(thresholded)
    while (thresholded_old != thresholded).any():
        if spatial_dims == 2:
            hysteresis_magnitude = torch.nn.functional.conv2d(thresholded.float(), hysteresis_kernel, padding=1)
        elif spatial_dims == 3:
            hysteresis_magnitude = torch.nn.functional.conv3d(thresholded.float(), hysteresis_kernel, padding=1)
        else:
            raise ValueError(f'Unsupported number of spatial dimensions: {spatial_dims}')
        thresholded_old.copy_(thresholded)
        thresholded = ((hysteresis_magnitude > 0) & mask_low) | mask_high

    return thresholded.bool()
