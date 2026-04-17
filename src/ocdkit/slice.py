"""Slicing utilities — builders for ``slice`` tuples used in ND indexing."""

from collections.abc import Iterable


def get_slice_tuple(start, stop, shape, axis=None):
    """Build a tuple of slice objects for ND array indexing.

    Parameters
    ----------
    start : int or iterable of int
        Starting index(es).
    stop : int or iterable of int
        Stopping index(es).
    shape : tuple of int
        Shape of the array to slice.
    axis : int, iterable of int, or None
        Axis or axes to apply slices to. Default: all axes if *start*/*stop*
        are iterable, else axis ``0``.

    Returns
    -------
    tuple of slice
        Length ``len(shape)``. Axes not addressed by *axis* get ``slice(None)``.
    """
    ndim = len(shape)
    slices = [slice(None)] * ndim

    if isinstance(start, Iterable) and isinstance(stop, Iterable):
        if axis is None:
            axis = list(range(ndim))

        if len(start) != len(stop):
            raise ValueError("start and stop must be the same length")

        if isinstance(axis, Iterable):
            if len(axis) != len(start):
                raise ValueError("axis must be the same length as start and stop")
        else:
            axis = [axis] * len(start)

        for a, s, e in zip(axis, start, stop):
            slices[a] = slice(s, e, None)
    else:
        if axis is None:
            axis = 0
        slices[axis] = slice(start, stop, None)

    return tuple(slices)
