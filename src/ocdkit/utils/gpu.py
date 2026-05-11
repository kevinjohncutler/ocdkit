"""GPU device resolution and memory management.

Provides a single canonical way to detect the best available torch device
(CUDA -> MPS -> CPU) and to clear accelerator caches.
"""

import os
import platform

import numpy as _np
import torch
import torch.nn.functional as _F

ARM = "arm" in platform.processor() and torch.backends.mps.is_available()


# Cache of (device_type, mode) -> bool for the 3D grid_sample capability probe.
# MPS has historically lacked nearest support; landed on main but not yet in
# any released stable (as of torch 2.11). Probe once per (type, mode).
_grid3d_cap: dict = {}


def supports_grid3d(device, mode: str = 'bilinear') -> bool:
    """True if *device* supports 3D ``grid_sample`` with the given interp mode.

    Probes once per (device_type, mode) and caches. ``RuntimeError`` and
    ``NotImplementedError`` are both treated as "unsupported".
    """
    dtype = getattr(device, 'type', str(device))
    key = (dtype, mode)
    if key not in _grid3d_cap:
        try:
            dev = torch.device(dtype)
            _F.grid_sample(
                torch.zeros(1, 1, 2, 2, 2, device=dev),
                torch.zeros(1, 2, 2, 2, 3, device=dev),
                mode=mode, align_corners=True,
            )
            _grid3d_cap[key] = True
        except (NotImplementedError, RuntimeError):
            _grid3d_cap[key] = False
    return _grid3d_cap[key]


def resolve_device(device=None):
    """Return a ``torch.device`` for *device*, auto-detecting if *None*.

    When *device* is ``None`` the priority order is CUDA -> MPS -> CPU.
    A string or existing ``torch.device`` is passed through unchanged.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if ARM:
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache():
    """Clear the accelerator memory cache (CUDA or MPS)."""
    if ARM:
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()


def get_device(gpu_number=0):
    """Test whether a specific GPU device index works.

    Returns ``(device, gpu_available)``. Falls back to CPU if the requested
    GPU is not available.
    """
    try:
        if gpu_number is None:
            gpu_number = 0
        device = torch.device(f'mps:{gpu_number}') if ARM else torch.device(f'cuda:{gpu_number}')
        _ = torch.zeros([1, 2, 3]).to(device)
        return device, True
    except Exception:
        return torch_CPU, False


# Alias kept for backward compatibility.
use_gpu = get_device


def seed_all(seed: int) -> None:
    """Seed all available GPU devices plus CPU RNG, and enforce deterministic CUDA ops."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Torch tensor utilities (device dispatch, type conversion)
# ---------------------------------------------------------------------------

def ensure_torch(*arrays, device=None, dtype=None):
    """Convert numpy arrays or unbatched torch tensors to batched torch tensors.

    Each input is converted to a ``(B, ...)`` tensor on *device*:

    - numpy arrays → ``torch.tensor(...).unsqueeze(0)``
    - unbatched torch tensors (2D scalar or 3D vector field) → ``.unsqueeze(0)``
    - already-batched tensors → moved to *device*

    Non-array inputs are passed through unchanged.
    """
    if dtype is None:
        dtype = torch.float32
    result = []
    for arr in arrays:
        if isinstance(arr, _np.ndarray):
            result.append(torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0))
        elif isinstance(arr, torch.Tensor):
            arr = arr.to(device=device, dtype=dtype)
            is_spatial_unbatched = (arr.ndim == 3 and arr.shape[0] in [2, 3])
            is_scalar_unbatched = (arr.ndim == 2)
            if is_spatial_unbatched or is_scalar_unbatched:
                arr = arr.unsqueeze(0)
            result.append(arr)
        else:
            result.append(arr)
    return tuple(result)


def torch_and(tensors):
    """Element-wise logical AND across a sequence of boolean tensors.

    Automatically dispatches to the optimal implementation:

    - **CPU**: sequential ``reduce(logical_and, ...)`` (avoids kernel-launch overhead)
    - **GPU**: single ``torch.all(stack(...), dim=0)`` kernel

    Inputs are broadcast to a common shape before reduction.
    """
    from functools import reduce
    dev = tensors[0].device if tensors else torch.device('cpu')

    broadcasted = torch.broadcast_tensors(*tensors)

    if dev.type == 'cpu':
        return reduce(torch.logical_and, broadcasted)
    else:
        return torch.all(torch.stack(tuple(broadcasted), dim=0), dim=0)


def to_device(x, device):
    """Move *x* to *device*, converting numpy arrays to float32 tensors."""
    if isinstance(x, torch.Tensor):
        if device != x.device:
            return x.to(device)
        return x
    return torch.tensor(x, device=device, dtype=torch.float32)


def from_device(x):
    """Detach a tensor and return it as a numpy array."""
    return x.detach().cpu().numpy()


def torch_zoom(img, scale_factor=1.0, dim=2, size=None, mode='bilinear'):
    """Resize a torch tensor using ``F.interpolate``."""
    target_size = [int(d * scale_factor) for d in img.shape[-dim:]] if size is None else size
    return torch.nn.functional.interpolate(img, size=target_size, mode=mode, align_corners=False)


def as_torch_tensor(arr, device="cpu", dtype=None, non_blocking=False):
    """Convert an array-like to a :class:`torch.Tensor` with minimal copies.

    Accepts numpy arrays (including memmaps), dask arrays, and existing
    tensors.  Uses ``torch.from_numpy`` for zero-copy on CPU when possible
    and silences the read-only memmap warning.
    """
    import warnings

    if dtype is None:
        dtype = torch.float32

    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype, non_blocking=non_blocking)

    # Dask → numpy
    try:
        import dask.array as da
        if isinstance(arr, da.Array):
            arr = arr.compute()
    except ImportError:
        pass

    if not isinstance(arr, _np.ndarray):
        arr = _np.asanyarray(arr)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        t = torch.from_numpy(arr)

    if t.dtype != dtype:
        t = t.to(dtype)
    if str(device) != "cpu":
        t = t.to(device=device, non_blocking=non_blocking)
    return t


# Numpy ↔ torch dtype comparison
_DTYPE_CANON = {
    _np.float16: "f16", _np.float32: "f32", _np.float64: "f64",
    _np.int8: "i8", _np.int16: "i16", _np.int32: "i32", _np.int64: "i64",
    _np.uint8: "u8", _np.bool_: "b",
    torch.float16: "f16", torch.float32: "f32", torch.float64: "f64",
    torch.int8: "i8", torch.int16: "i16", torch.int32: "i32", torch.int64: "i64",
    torch.uint8: "u8", torch.bool: "b",
}


def is_same_dtype(dtype1, dtype2):
    """Check whether *dtype1* and *dtype2* represent the same numeric type.

    Works across numpy dtypes, numpy scalar types, and torch dtypes.
    """
    if isinstance(dtype1, _np.dtype):
        dtype1 = dtype1.type
    if isinstance(dtype2, _np.dtype):
        dtype2 = dtype2.type
    return _DTYPE_CANON.get(dtype1) == _DTYPE_CANON.get(dtype2)


# Backward-compatible constants (used extensively in omnirefactor).
torch_CPU = torch.device("cpu")
torch_GPU = resolve_device()
