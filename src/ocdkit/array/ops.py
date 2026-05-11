"""Miscellaneous array utilities — divergence, noise, search, metadata."""

from .imports import *
from scipy.ndimage import convolve1d, gaussian_filter

from .convert import get_module


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


def searchsorted(tensor, value):
    """Find indices where *value* should be inserted in *tensor* to keep order.

    Backend-agnostic via ``get_module``: works on numpy arrays, torch tensors,
    and any input where ``(tensor < value).sum()`` is meaningful.
    """
    return (tensor < value).sum()


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


def unique_nonzero(arr):
    """Return sorted unique non-zero values of *arr*.

    Uses ``numpy.unique`` directly. Historically delegated to
    ``fastremap.unique`` because it was several × faster on integer
    label arrays; numpy 2.x's reworked unique closed that gap, so we
    drop the optional dep. ocdkit pins ``numpy>=2.0`` to keep this
    fast — older numpy will still work but loses the perf parity.
    """
    u = np.unique(arr)
    return u[u != 0]


def is_sequential(labels):
    """Whether the unique values of *labels* form a contiguous integer run.

    Returns ``True`` when ``np.unique(labels)`` is a tightly-packed
    range ``[v0, v0+1, …, vN]`` with no gaps. Useful for asking "is
    this label array already in canonical 1..N (or 0..N) form?"
    before deciding whether to compact via a renumber pass.

    Empty / single-value arrays are treated as sequential.
    """
    u = np.unique(np.asarray(labels))
    if u.size <= 1:
        return True
    return bool(np.all(np.diff(u) == 1))


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
