"""Diameter estimation and shape analysis."""

from .imports import *
import edt as _edt
from scipy.ndimage import convolve


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
