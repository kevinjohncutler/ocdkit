"""Indexing, slicing, partitioning, and coordinate generation."""

from collections.abc import Iterable

from .imports import *


def ravel_index(b, shp):
    """Row-major flat index from multi-dim indices *b* and array shape *shp*.

    *b* has shape ``(ndim, npts)``; returns a 1D array of length ``npts``.
    """
    return np.concatenate((np.asarray(shp[1:])[::-1].cumprod()[::-1], [1])).dot(b)


def unravel_index(index, shape):
    """Multi-dim indices (row-major) from a flat *index* and array *shape*.

    Returns a tuple of per-axis index arrays.
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def border_indices(tyx):
    """Flat indices of the border pixels of an ND array with shape *tyx*.

    Use as ``A.flat[border_indices(A.shape)]``.
    """
    dim_indices = [np.arange(dim_size) for dim_size in tyx]
    dim_indices = np.meshgrid(*dim_indices, indexing='ij')
    dim_indices = [indices.ravel() for indices in dim_indices]

    indices = []
    for i in range(len(tyx)):
        for j in [0, tyx[i] - 1]:
            mask = (dim_indices[i] == j)
            indices.append(np.where(mask)[0])
    return np.concatenate(indices)


def split_array(array, parts, axes=None):
    """Split an ndarray into *parts* along specified *axes*.

    Parameters
    ----------
    array : ndarray
        The array to split.
    parts : int or tuple of int
        Number of parts to split along each axis. If an integer, applies to
        all specified axes.
    axes : int, tuple of int, or None
        The axes to split. If None, splits all axes.

    Returns
    -------
    nested list of ndarray
        Nested sub-arrays after splitting. The nesting depth equals
        ``len(axes)``.
    """
    if isinstance(parts, int):
        parts = (parts,) * array.ndim

    if axes is None:
        axes = tuple(range(array.ndim))
    elif isinstance(axes, int):
        axes = (axes,)

    if len(parts) != len(axes):
        raise ValueError("Length of 'parts' must match the number of axes specified.")

    splits = []
    warnings = []

    for ax, num_parts in zip(axes, parts):
        dim_size = array.shape[ax]
        chunk_sizes = [
            dim_size // num_parts + (1 if i < dim_size % num_parts else 0)
            for i in range(num_parts)
        ]
        if dim_size % num_parts != 0:
            warnings.append(f"Axis {ax} ({dim_size}) is not evenly divisible by {num_parts}.")
        split_indices = np.cumsum(chunk_sizes[:-1])
        splits.append(np.split(np.arange(dim_size), split_indices))

    for warning in warnings:
        print("Warning:", warning)

    def _recursive_split(array, splits, axes):
        if not splits:
            return array
        ax = axes[0]
        subarrays = []
        for idxs in splits[0]:
            sliced = np.take(array, idxs, axis=ax)
            subarrays.append(_recursive_split(sliced, splits[1:], axes[1:]))
        return subarrays

    return _recursive_split(array, splits, axes)


def reconstruct_array(nested_list, axes=None):
    """Reconstruct an ndarray from a nested list produced by :func:`split_array`.

    Parameters
    ----------
    nested_list : list of ndarray
        Nested sub-arrays to concatenate back together.
    axes : int, tuple of int, or None
        The axes used for splitting. If None, infers from the outer list's
        nesting depth.
    """
    if axes is None:
        axes = tuple(range(
            len(nested_list[0].shape) if isinstance(nested_list[0], np.ndarray)
            else len(nested_list)
        ))
    elif isinstance(axes, int):
        axes = (axes,)

    def _recursive_reconstruct(nested, level):
        if isinstance(nested, np.ndarray):
            return nested
        if level == len(axes):
            return np.array(nested)
        return np.concatenate(
            [_recursive_reconstruct(sub, level + 1) for sub in nested],
            axis=axes[level],
        )

    return _recursive_reconstruct(nested_list, 0)


def meshgrid(shape):
    """Generate a tuple of coordinate grids for an ND array of given *shape*.

    Parameters
    ----------
    shape : tuple of int
        Shape of the ND array (e.g., ``(Y, X)`` for 2D, ``(Z, Y, X)`` for 3D).

    Returns
    -------
    tuple of ndarray
        ``N`` coordinate arrays, one per dimension, in ``ij`` indexing.
    """
    ranges = [np.arange(dim) for dim in shape]
    return np.meshgrid(*ranges, indexing='ij')


def generate_flat_coordinates(shape):
    """Generate flat coordinate arrays for an ND array.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array (e.g., ``(Y, X)`` for 2D).

    Returns
    -------
    tuple of ndarray
        ``N`` 1D arrays containing the coordinates of every element of an
        array with ``shape``, in ``ij`` order.
    """
    return tuple(grid.ravel() for grid in meshgrid(shape))


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
