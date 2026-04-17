"""Array utilities — numpy/torch agnostic operations.

Provides a consistent interface for operations that work on both
numpy arrays and PyTorch tensors.
"""

import numpy as np
import torch
import dask.array as da
from scipy.ndimage import convolve1d, gaussian_filter
from skimage.transform import resize as skimage_resize

from .gpu import torch_GPU


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


# ---------------------------------------------------------------------------
# Array metadata / random helpers
# ---------------------------------------------------------------------------

def get_size(var, unit='GB'):
    """Return the in-memory size of *var* in *unit* ('B', 'KB', 'MB', 'GB').

    Works for any object with an ``nbytes`` attribute (numpy, dask, torch).
    """
    units = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3}
    return var.nbytes / (1024 ** units[unit])


def random_int(N, M=None, seed=None):
    """Generate random integers in ``[0, N)``.

    Parameters
    ----------
    N : int
        Upper bound (exclusive).
    M : int, optional
        Number of integers to generate. Scalar if None.
    seed : int, optional
        RNG seed. If None, a random seed is generated and printed to stdout.
    """
    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)
        print(f'Seed: {seed}')
    else:
        np.random.seed(seed)
    return np.random.randint(0, N, M)


# ---------------------------------------------------------------------------
# Flat / multi-dim index conversion + border index helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generic ND array partitioning
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Nested-structure iteration
# ---------------------------------------------------------------------------

def enumerate_nested(*lists, parent_indices=None):
    """Traverse matching nested lists and yield indices with corresponding values.

    Parameters
    ----------
    lists : list(s)
        One or more nested lists to traverse. All must share the same structure.
    parent_indices : list, optional
        Indices accumulated by recursive calls (internal).

    Yields
    ------
    tuple
        ``(indices, values...)`` — index path plus one value from each input.
    """
    if parent_indices is None:
        parent_indices = []

    if all(isinstance(lst[0], list) for lst in lists):
        for i, sublists in enumerate(zip(*lists)):
            current_indices = parent_indices + [i]
            yield from enumerate_nested(*sublists, parent_indices=current_indices)
    else:
        for i, values in enumerate(zip(*lists)):
            current_indices = parent_indices + [i]
            yield current_indices, *values


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


def unique_nonzero(arr):
    """Return sorted unique non-zero values of *arr*."""
    import fastremap
    u = fastremap.unique(arr)
    return u[u != 0]


def torch_norm(a, dim=0, keepdim=False):
    """Compute vector magnitude along *dim*.

    Works on numpy arrays and torch tensors. For torch, avoids
    intermediate allocations and supports autograd.
    """
    module = get_module(a)
    if module == np:
        return np.sqrt(np.sum(a**2, axis=dim, keepdims=keepdim))
    norm_sq = (a * a).sum(dim=dim, keepdim=keepdim)
    return norm_sq.sqrt_() if not norm_sq.requires_grad else norm_sq.sqrt()


def _auto_chunked_quantile(tensor, q):
    """Chunked quantile for large tensors that exceed torch.quantile limits."""
    import math
    max_elements = int(16e6 - 1)
    num_elements = tensor.nelement()
    chunk_size = max(1, math.ceil(num_elements / max_elements))
    chunks = torch.chunk(tensor, chunk_size)
    return torch.stack([torch.quantile(chunk, q) for chunk in chunks]).mean(dim=0)


def normalize99(Y, lower=0.01, upper=99.99, contrast_limits=None, dim=None, **kwargs):
    """Clip to percentile range and rescale to [0, 1].

    Works on numpy arrays and torch tensors.

    Parameters
    ----------
    Y : array or tensor
        Input data.
    lower : float
        Lower percentile (0-100).
    upper : float
        Upper percentile (0-100).
    contrast_limits : tuple of float, optional
        Explicit (low, high) values instead of computing from percentiles.
    dim : int, optional
        Normalize independently along this dimension.
    """
    module = get_module(Y)

    if contrast_limits is None:
        quantiles = np.array([lower, upper]) / 100
        if module == torch:
            quantiles = torch.tensor(quantiles, dtype=Y.dtype, device=Y.device)

        if dim is not None:
            Y_flattened = Y.reshape(Y.shape[dim], -1)
            lower_val, upper_val = module.quantile(Y_flattened, quantiles, axis=-1)
            if dim == 0:
                lower_val = lower_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
                upper_val = upper_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
            else:
                lower_val = lower_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
                upper_val = upper_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
        else:
            try:
                lower_val, upper_val = module.quantile(Y, quantiles)
            except RuntimeError:
                lower_val, upper_val = _auto_chunked_quantile(Y, quantiles)
    else:
        if module == np:
            contrast_limits = np.array(contrast_limits)
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)
        lower_val, upper_val = contrast_limits

    return module.clip(safe_divide(Y - lower_val, upper_val - lower_val), 0, 1)


def normalize_field(mu, cutoff=0, **kwargs):
    """Normalize all nonzero field vectors to unit magnitude.

    Works on numpy arrays and torch tensors (auto-detected).

    Parameters
    ----------
    mu : array or tensor
        Vector field, shape ``(D, *spatial)``.
    cutoff : float
        Vectors with magnitude below this are left unchanged.
    """
    module = get_module(mu)
    if module == torch:
        mag = torch_norm(mu, dim=0)
        return torch.where(mag > cutoff, mu / mag, mu)
    mag = np.sqrt(np.nansum(mu**2, axis=0))
    valid = mag > cutoff
    return np.where(valid, mu / np.where(valid, mag, 1.0), mu)


# ---------------------------------------------------------------------------
# Coordinate grid generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Vector calculus
# ---------------------------------------------------------------------------

def divergence(f):
    """Divergence of a vector field, dispatched on backend.

    Parameters
    ----------
    f : array or tensor
        Numpy: shape ``(D, *spatial)`` — unbatched D-vector field.
        Torch: shape ``(B, D, *spatial)`` — batched D-vector field.

    Returns
    -------
    div : array or tensor
        Numpy: shape ``(*spatial,)``.
        Torch: shape ``(B, *spatial)``.

    Notes
    -----
    Returns zeros if any spatial dimension has size < 2 (gradient undefined).
    On the CPU torch path this loops over components rather than calling
    ``torch.gradient`` on all dims at once, which would do unnecessary work.
    """
    module = get_module(f)
    if module == np:
        num_dims = len(f)
        if any(f.shape[1 + i] < 2 for i in range(num_dims)):
            return np.zeros_like(f[0])
        return np.add.reduce([np.gradient(f[i], axis=i) for i in range(num_dims)])

    # Torch path: batched (B, D, *spatial)
    B, D, *spatial = f.shape
    if any(s < 2 for s in spatial):
        return torch.zeros((B, *spatial), dtype=f.dtype, device=f.device)
    div = torch.zeros((B, *spatial), dtype=f.dtype, device=f.device)
    for d in range(D):
        div += torch.gradient(f[:, d], dim=d + 1)[0]
    return div


# ---------------------------------------------------------------------------
# Experimental normalization functions
# ---------------------------------------------------------------------------
# These are ported as-is from omnirefactor and currently support only numpy
# (or numpy + torch where the original did). They are slated for future
# generalization to dask + torch via the get_module pattern, plus statistical
# benchmarking against the existing normalize99 path.

def searchsorted(tensor, value):
    """Find indices where *value* should be inserted in *tensor* to keep order.

    Backend-agnostic via ``get_module``: works on numpy arrays, torch tensors,
    and any input where ``(tensor < value).sum()`` is meaningful.
    """
    return (tensor < value).sum()


def quantile_rescale(Y, lower=0.0001, upper=0.9999, contrast_limits=None, bins=None):
    """Sort-based quantile rescale to [0, 1].

    Slower than ``normalize99`` for large arrays (mergesort cost), but the
    explicit sort lets the caller plug in a different quantile estimator
    later. Numpy only for now.
    """
    sorted_array = np.sort(Y.flatten(), kind="mergesort")
    lower_idx = int(lower * (len(sorted_array) - 1))
    upper_idx = int(upper * (len(sorted_array) - 1))
    lower_val, upper_val = sorted_array[lower_idx], sorted_array[upper_idx]
    r = safe_divide(Y - lower_val, upper_val - lower_val)
    r[r < 0] = 0
    r[r > 1] = 1
    return r


def normalize99_hist(Y, lower=0.01, upper=99.99, contrast_limits=None, bins=None):
    """Histogram-based percentile clip and rescale to [0, 1].

    An alternative to ``normalize99`` that estimates quantiles from a CDF
    over a histogram (cheaper than torch.quantile on huge tensors).
    Works on numpy and torch.
    """
    upper = upper / 100
    lower = lower / 100

    module = get_module(Y)
    if bins is None:
        num_elements = Y.size if module == np else Y.numel()
        bins = int(np.sqrt(num_elements))

    if contrast_limits is None:
        hist, bin_edges = module.histogram(Y, bins=bins)
        cdf = module.cumsum(hist, axis=0) / module.sum(hist)
        lower_val = bin_edges[searchsorted(cdf, lower)]
        upper_val = bin_edges[searchsorted(cdf, upper)]
    else:
        if module == np:
            contrast_limits = np.array(contrast_limits)
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)
        lower_val, upper_val = contrast_limits

    r = safe_divide(Y - lower_val, upper_val - lower_val)
    r[r < 0] = 0
    r[r > 1] = 1
    return r


def qnorm(
    Y,
    nbins=100,
    bw_method=2,
    density_cutoff=None,
    density_quantile=(0.001, 0.999),
    debug=False,
    dx=None,
    log=False,
    eps=1,
):
    """Density-based quantile normalization.

    Bins the histogram, fits a symmetric KDE to the density of the histogram
    counts, and clips to the range where density exceeds *density_cutoff*.
    Useful when the brightness distribution has a heavy tail and a simple
    percentile clip is too aggressive. Numpy only for now.
    """
    import fastremap
    from scipy.stats import gaussian_kde

    if dx is not None:
        X = Y[:, ::dx, ::dx]
    else:
        X = Y

    if X.dtype not in (np.uint8, np.uint16, np.uint32, np.uint64):
        X = (rescale(X) * (2 ** 16 - 1)).astype(np.uint16)

    # bin counts
    unique_values, counts = fastremap.unique(X, return_counts=True)
    bin_edges = np.linspace(unique_values.min(), unique_values.max(), nbins + 1)
    bin_indices = np.digitize(unique_values, bin_edges) - 1
    binned_counts = np.bincount(bin_indices, weights=counts, minlength=nbins)
    bin_start = bin_edges[:-1]
    binned_counts = binned_counts[:-1]

    sel = binned_counts > 0
    counts_sel = binned_counts[sel]
    unique_sel = bin_start[sel]
    x = np.arange(len(counts_sel))
    y = np.log(counts_sel + eps) if log else counts_sel

    # symmetric KDE density
    points = np.vstack([x, y])
    kde = gaussian_kde(points, bw_method=bw_method)
    density = kde(points)
    inverted_kde = gaussian_kde(np.vstack([-x, y]), bw_method=bw_method)
    inverted_density = inverted_kde(np.vstack([-x, y]))
    d = rescale((density + inverted_density) / 2)

    if not isinstance(density_quantile, (list, tuple)):
        density_quantile = (density_quantile, density_quantile)

    if density_cutoff is None:
        density_cutoff = np.quantile(d, density_quantile)  # pragma: no cover
    elif not isinstance(density_cutoff, (list, tuple)):
        density_cutoff = (density_cutoff, density_cutoff)

    imin = np.argwhere(d > density_cutoff[0])[0][0]
    imax = np.argwhere(d > density_cutoff[1])[-1][0]
    vmin, vmax = unique_sel[imin], unique_sel[imax]

    if vmax > vmin:
        scale_factor = np.float16(1.0 / (vmax - vmin))
        r = X * scale_factor
        r[r > 1] = 1
    else:
        r = X

    if debug:
        return r, x, y, d, imin, imax, vmin, vmax
    return r


def localnormalize(im, sigma1=2, sigma2=20):
    """Local mean/std normalization via Gaussian blurs.

    Works on numpy and torch (auto-dispatched). For torch, uses
    ``torchvision.transforms.functional.gaussian_blur``; for numpy uses
    ``scipy.ndimage.gaussian_filter``.
    """
    module = get_module(im)
    if module == torch:
        import torchvision.transforms.functional as TF
        im = normalize99(im)
        ks1 = round(sigma1 * 6)
        ks1 += ks1 % 2 == 0
        blur1 = TF.gaussian_blur(im, ks1, sigma1)
        num = im - blur1
        ks2 = round(sigma2 * 6)
        ks2 += ks2 % 2 == 0
        blur2 = TF.gaussian_blur(num * num, ks2, sigma2)
        den = torch.sqrt(blur2)
        return normalize99(num / den + 1e-8)

    from scipy.ndimage import gaussian_filter
    im = normalize99(im)
    blur1 = gaussian_filter(im, sigma=sigma1)
    num = im - blur1
    blur2 = gaussian_filter(num * num, sigma=sigma2)
    den = np.sqrt(blur2)
    return normalize99(num / den + 1e-8)


# Backward-compat alias
localnormalize_GPU = localnormalize


def pnormalize(Y, p_min=-1, p_max=10):
    """Power-mean normalization to [0, 1].

    Uses ``L^p`` norms with negative *p_min* (approximating min) and positive
    *p_max* (approximating max) as soft min/max estimators. Works on numpy
    and torch (auto-dispatched).
    """
    module = get_module(Y)
    lower_val = (module.abs(Y * 1.0) ** p_min).sum() ** (1.0 / p_min)
    upper_val = (module.abs(Y * 1.0) ** p_max).sum() ** (1.0 / p_max)
    return module.clip(safe_divide(Y - lower_val, upper_val - lower_val), 0, 1)


def normalize_image(
    im,
    mask,
    target=0.5,
    foreground=False,
    iterations=1,
    scale=1,
    channel_axis=0,
    per_channel=True,
):
    """Mask-aware gamma normalization to push masked region mean to *target*.

    Numpy implementation. Optionally erodes the mask before computing the
    target mean to avoid edge contamination.
    """
    from scipy.ndimage import binary_erosion
    try:
        import numexpr as ne
    except Exception:  # pragma: no cover
        ne = None

    im = im.astype("float32") * scale
    im_min = im.min()
    im_max = im.max()
    if ne is None:
        im = (im - im_min) / (im_max - im_min)
    else:
        ne.evaluate("(im - im_min) / (im_max - im_min)", out=im)

    if im.ndim > 2:
        im = np.moveaxis(im, channel_axis, -1)
    else:
        im = np.expand_dims(im, axis=-1)

    if not isinstance(mask, list):
        mask = np.expand_dims(mask, axis=-1)
        mask = np.broadcast_to(mask, im.shape)

    bin0 = mask > 0 if foreground else mask == 0
    if iterations > 0:
        structure = np.ones((3,) * (im.ndim - 1) + (1,))
        structure[1, ...] = 0
        bin0 = binary_erosion(bin0, structure=structure, iterations=iterations)

    masked_im = im.copy()
    masked_im[~bin0] = np.nan
    source_target = np.nanmean(masked_im, axis=(0, 1) if per_channel else None)
    source_target = source_target.astype("float32")
    target = np.array(target).astype("float32")
    if ne is None:
        im = im ** (np.log(target) / np.log(source_target))
    else:
        ne.evaluate("im ** (log(target) / log(source_target))", out=im)
    return np.moveaxis(im, -1, channel_axis).squeeze()


def adjust_contrast_masked(
    img,
    masks,
    r_target=1.10,
    plo=0.01,
    phi=99.99,
    clip_output=True,
):
    """Mask-aware percentile-clip + gamma to hit a target fg/bg ratio.

    Returns ``(adjusted, gamma, (lo, hi))``. Numpy only.
    """
    x = np.asarray(img, dtype=np.float32)
    m = np.asarray(masks).astype(bool)
    bg = ~m
    fg = m

    if fg.sum() == 0 or bg.sum() == 0:
        return x.copy(), 1.0, (float(np.min(x)), float(np.max(x)))

    a = np.percentile(x[bg], plo)
    b = np.percentile(x[fg], phi)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        a = float(np.min(x))
        b = float(np.max(x))

    if not np.isfinite(b - a) or b <= a:
        return x.copy(), 1.0, (float(a), float(b))

    j = (x - a) / (b - a)
    j = np.clip(j, 0.0, 1.0)

    m_fg = float(j[fg].mean())
    m_bg = float(j[bg].mean() + 1e-12)
    r = m_fg / m_bg

    if (r >= 1.0 and r_target < 1.0) or (r <= 1.0 and r_target > 1.0):  # pragma: no cover
        return j.copy(), 1.0, (a, b)

    if abs(np.log(max(r, 1e-12))) < 1e-8 or abs((r - r_target) / max(r_target, 1e-12)) < 1e-3:  # pragma: no cover
        y = j
        gamma = 1.0
    else:
        gamma = float(np.log(max(r_target, 1e-12)) / np.log(max(r, 1e-12)))
        gamma = float(np.clip(gamma, 0.2, 5.0))
        y = np.power(j, gamma)

    if clip_output:
        y = np.clip(y, 0.0, 1.0)

    return y.astype(np.float32), gamma, (float(a), float(b))


def gamma_normalize(
    im,
    mask,
    target=1.0,
    scale=1.0,
    foreground=True,
    iterations=0,
    per_channel=True,
    channel_axis=-1,
):
    """Torch (GPU-accelerated) variant of :func:`normalize_image`.

    Uses ``ocdkit.gpu.torch_GPU`` for the device.
    """
    from scipy.ndimage import binary_erosion

    device = torch_GPU
    im = rescale(im) * scale
    if im.ndim > 2:
        im = np.moveaxis(im, channel_axis, -1)
    else:
        im = np.expand_dims(im, axis=-1)

    if not isinstance(mask, list):
        mask = np.stack([mask] * im.shape[-1], axis=-1)

    im = torch.from_numpy(im).float().to(device)
    mask = torch.from_numpy(mask).float().to(device)

    bin0 = mask > 0 if foreground else mask == 0
    if iterations > 0:
        structure = torch.ones((3,) * (im.ndim - 1) + (1,)).to(device)
        structure[1, ...] = 0
        bin0 = torch.from_numpy(
            binary_erosion(bin0.cpu().numpy(), structure=structure.cpu().numpy(), iterations=iterations)
        ).to(device)

    masked_im = im.masked_fill(~bin0, float("nan"))
    source_target = torch.nanmean(masked_im, dim=(0, 1) if per_channel else None)
    im **= (torch.log(target) / torch.log(source_target))

    return im.permute(*[channel_axis] + [i for i in range(im.ndim) if i != channel_axis]).squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Axis manipulation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Filtering / noise
# ---------------------------------------------------------------------------

def moving_average(x, w):
    """1-D moving average via convolution along axis 0."""
    return convolve1d(x, np.ones(w) / w, axis=0)


def add_poisson_noise(image):
    """Apply Poisson noise to an image and clip to [0, 1]."""
    noisy_image = np.random.poisson(image)
    return np.clip(noisy_image, 0, 1)


def correct_illumination(img, sigma=5):
    """Flatten illumination by subtracting a Gaussian-blurred background."""
    blurred = gaussian_filter(img, sigma=sigma)
    return (img - blurred) / np.std(blurred)


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=1, no_channels=False):
    """Resize an image array.

    Parameters
    ----------
    img0 : ndarray
        Image of shape ``(Y, X, C)`` or ``(Z, Y, X, C)`` or ``(Z, Y, X)``.
    Ly, Lx : int, optional
        Target spatial dimensions. If *None*, computed from *rsz*.
    rsz : float or list of float, optional
        Resize factor(s). Used when *Ly* is *None*.
    interpolation : int
        0 for nearest-neighbor, 1 for bilinear.
    no_channels : bool
        If True, treat the last spatial dim as spatial (not channels).
    """
    if Ly is None and rsz is None:
        raise ValueError('must give size to resize to or factor to use for resizing')

    if Ly is None:
        if not isinstance(rsz, (list, np.ndarray)):
            rsz = [rsz, rsz]
        if no_channels:
            Ly = int(img0.shape[-2] * rsz[-2])
            Lx = int(img0.shape[-1] * rsz[-1])
        else:
            Ly = int(img0.shape[-3] * rsz[-2])
            Lx = int(img0.shape[-2] * rsz[-1])

    order = interpolation
    if (img0.ndim > 2 and no_channels) or (img0.ndim == 4 and not no_channels):
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i, img in enumerate(img0):
            imgs[i] = skimage_resize(img, imgs[i].shape, order=order, preserve_range=True).astype(np.float32)
    else:
        out_shape = (Ly, Lx) if img0.ndim == 2 else (Ly, Lx, img0.shape[-1])
        imgs = skimage_resize(img0, out_shape, order=order, preserve_range=True).astype(np.float32)
    return imgs
