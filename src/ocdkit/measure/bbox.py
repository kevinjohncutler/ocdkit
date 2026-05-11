"""Bounding box extraction and patch operations."""

from .imports import *
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from skimage import measure

from ..array import is_integer


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
