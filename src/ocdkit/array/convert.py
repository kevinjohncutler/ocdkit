"""Type detection, conversion, and rescaling utilities."""

from .imports import *


def get_module(x):
    """Return ``np`` or ``torch`` depending on the type of *x*."""
    if isinstance(x, da.Array):
        return np
    if isinstance(x, (np.ndarray, tuple, int, float)) or np.isscalar(x):
        return np
    if torch.is_tensor(x):
        return torch
    raise ValueError(
        "Input must be a numpy array, a tuple, a torch tensor, "
        "an integer, or a float"
    )


def safe_divide(num, den, cutoff=0):
    """Division ignoring zeros and NaNs in the denominator."""
    module = get_module(num)
    valid_den = (den > cutoff) & module.isfinite(den)

    if isinstance(num, da.Array) or isinstance(den, da.Array):
        return da.where(valid_den, num / den, 0)
    elif module == np:
        r = num.astype(np.float32, copy=False)
        r = np.divide(r, den, out=np.zeros_like(r), where=valid_den)
    elif module == torch:
        r = num.float()
        den = den.float()
        safe_den = torch.where(valid_den, den, torch.ones_like(den))
        r = torch.where(valid_den, torch.div(r, safe_den), torch.zeros_like(r))
    else:
        raise TypeError("num must be a numpy array or a PyTorch tensor")

    return r


def rescale(T, floor=None, ceiling=None, exclude_dims=None):
    """Min-max rescale to [0, 1].

    Works on numpy arrays and torch tensors.  When *exclude_dims* is
    given, normalization is applied independently along those axes.
    """
    module = get_module(T)
    if exclude_dims is not None:
        if isinstance(exclude_dims, int):
            exclude_dims = (exclude_dims,)
        axes = tuple(i for i in range(T.ndim) if i not in exclude_dims)
        newshape = [T.shape[i] if i in exclude_dims else 1 for i in range(T.ndim)]
    else:
        axes = None
        newshape = T.shape

    if ceiling is None:
        ceiling = module.amax(T, axis=axes)
        if exclude_dims is not None:
            ceiling = ceiling.reshape(*newshape)
    if floor is None:
        floor = module.amin(T, axis=axes)
        if exclude_dims is not None:
            floor = floor.reshape(*newshape)

    T = safe_divide(T - floor, ceiling - floor)

    return T


def to_16_bit(im):
    """Rescale image to [0, 2**16 - 1] and cast to uint16."""
    return np.uint16(rescale(im) * (2 ** 16 - 1))


def to_8_bit(im):
    """Rescale image to [0, 2**8 - 1] and cast to uint8."""
    return np.uint8(rescale(im) * (2 ** 8 - 1))


def is_integer(var):
    """Check whether *var* is an integer or integer-typed array/tensor."""
    if isinstance(var, int):
        return True
    if isinstance(var, np.integer):
        return True
    if isinstance(var, (np.ndarray, np.memmap)) and np.issubdtype(var.dtype, np.integer):
        return True
    if isinstance(var, da.Array) and np.issubdtype(var.dtype, np.integer):
        return True
    if isinstance(var, torch.Tensor) and not var.is_floating_point():
        return True
    return False


def to_torch(arr, device=None, dtype=None):
    """Convert *arr* to a ``torch.Tensor`` on *device* with *dtype*.

    Handles numpy arrays (incl. ``np.memmap``), dask arrays, existing torch
    tensors, and Python scalars/lists uniformly. Dask arrays are materialized
    via ``np.asarray`` — the conversion to torch is the natural boundary at
    which lazy graphs must be computed.

    Parameters
    ----------
    arr : torch.Tensor | numpy.ndarray | np.memmap | dask.array.Array | list | scalar
    device : torch.device | str | None
        Forwarded to ``.to``. ``None`` keeps the current device for tensors,
        otherwise defaults to CPU.
    dtype : torch.dtype | None
        Forwarded to ``.to``. ``None`` preserves the source dtype.

    Returns
    -------
    torch.Tensor
    """
    if torch.is_tensor(arr):
        return arr.to(device=device, dtype=dtype)
    if isinstance(arr, da.Array):
        arr = np.asarray(arr)
    if isinstance(arr, np.ndarray):
        # ``torch.from_numpy`` is zero-copy but warns on non-writeable input
        # (e.g. ``np.memmap`` opened with ``mmap_mode='r'``). When we have to
        # copy anyway, threaded ``parallel_copy`` is ~2x faster than
        # ``np.array`` for large memmaps (1 GB: 64 ms vs 123 ms) and falls
        # back to ``np.array`` automatically for tiny arrays.
        if not arr.flags["C_CONTIGUOUS"] or not arr.flags["WRITEABLE"]:
            from .parallel import parallel_copy
            arr = parallel_copy(arr)
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    return torch.tensor(arr, device=device, dtype=dtype)


def move_axis(img, axis=-1, pos="last"):
    """Move ndarray axis to a new location, preserving order of other axes."""
    if axis == -1:
        axis = img.ndim - 1
    axis = min(img.ndim - 1, axis)
    if pos in ("first", 0):
        pos = 0
    elif pos in ("last", -1):
        pos = img.ndim - 1
    perm = list(range(img.ndim))
    perm.pop(axis)
    perm.insert(pos, axis)
    return np.transpose(img, perm)


def move_min_dim(img, force=False):
    """Move the minimum-sized dimension last (as channels) if < 10."""
    if len(img.shape) > 2:
        min_dim = min(img.shape)
        if min_dim < 10 or force:
            if img.shape[-1] == min_dim:
                channel_axis = -1
            else:
                channel_axis = (img.shape).index(min_dim)
            img = move_axis(img, axis=channel_axis, pos="last")
    return img
