"""Normalization and contrast adjustment functions."""

from .imports import *
from .convert import get_module, safe_divide, rescale
from .ops import searchsorted

from ..utils.gpu import torch_GPU


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
