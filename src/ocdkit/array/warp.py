"""Affine warp utilities."""

from scipy.ndimage import affine_transform


def do_warp(A, M_inv, tyx, offset=0, order=1, mode='constant', **kwargs):
    """Apply an affine transformation to array *A*.

    Thin wrapper around :func:`scipy.ndimage.affine_transform`.

    Parameters
    ----------
    A : ndarray
        Input array.
    M_inv : ndarray
        Inverse transformation matrix.
    tyx : tuple of int
        Output shape.
    offset : float or sequence of float
        Translation offset.
    order : int
        Interpolation order (0=nearest, 1=linear).
    mode : str
        Border mode (``'constant'``, ``'nearest'``, ``'mirror'``).
    """
    return affine_transform(A, M_inv, offset=offset,
                            output_shape=tyx, order=order,
                            mode=mode, **kwargs)
