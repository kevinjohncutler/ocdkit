"""Spectral colorization and flow-field visualization for numpy, torch, and dask arrays."""

import colorsys

import numpy as np
import torch
import dask.array as da

from ..array import get_module, safe_divide, normalize99, torch_norm


def rgb_to_hsv(arr):
    """Convert an RGB array to HSV (last axis is the channel axis)."""
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    return np.stack((h, s, v), axis=-1)


def hsv_to_rgb(arr):
    """Convert an HSV array to RGB (last axis is the channel axis)."""
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    return np.stack((r, g, b), axis=-1)


def _make_colors(C, offset=0):
    """Generate sinebow-spaced RGB colors for C channels (always numpy)."""
    angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
    angles = np.stack((angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3), axis=-1)
    return ((np.cos(angles) + 1) / 2).astype(np.float32)


def _build_weights(C, intervals, colors, color_weights, module=None, device=None):
    """Build the folded weight matrix (C, N*3) from aggregator and colors.

    The result is created in the backend matching ``module`` when provided:
    numpy arrays for NumPy, dask arrays for Dask, and device-resident tensors
    for torch. This avoids forcing accelerator tensors through ``np.asarray``.
    """
    aggregator_np = np.zeros((C, len(intervals)), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator_np[start:start + size, i] = 1.0 / size
        start += size

    if module is None or module == np:
        colors_arr = np.asarray(colors, dtype=np.float32)
        if color_weights is not None:
            colors_arr = colors_arr * np.asarray(color_weights, dtype=np.float32)[:, None]
        weights = (aggregator_np[..., None] * colors_arr[:, None, :]).reshape(C, len(intervals) * 3)
        return weights, len(intervals)

    if module is da:
        aggregator = da.asarray(aggregator_np, dtype=np.float32)
        colors_arr = colors if isinstance(colors, da.Array) else da.asarray(colors, dtype=np.float32)
        if color_weights is not None:
            cw = color_weights if isinstance(color_weights, da.Array) else da.asarray(color_weights, dtype=np.float32)
            colors_arr = colors_arr * cw[:, None]
        weights = (aggregator[..., None] * colors_arr[:, None, :]).reshape(C, len(intervals) * 3)
        return weights, len(intervals)

    if module is torch:
        aggregator = torch.as_tensor(aggregator_np, dtype=torch.float32, device=device)
        colors_arr = colors if isinstance(colors, torch.Tensor) else torch.as_tensor(colors, dtype=torch.float32, device=device)
        if not torch.is_floating_point(colors_arr):
            colors_arr = colors_arr.float()
        elif colors_arr.dtype != torch.float32:
            colors_arr = colors_arr.to(dtype=torch.float32)
        if color_weights is not None:
            cw = color_weights if isinstance(color_weights, torch.Tensor) else torch.as_tensor(color_weights, dtype=torch.float32, device=device)
            if not torch.is_floating_point(cw):
                cw = cw.float()
            elif cw.dtype != torch.float32:
                cw = cw.to(dtype=torch.float32)
            colors_arr = colors_arr * cw[:, None]
        weights = (aggregator[..., None] * colors_arr[:, None, :]).reshape(C, len(intervals) * 3)
        return weights, len(intervals)

    colors_arr = np.asarray(colors, dtype=np.float32)
    if color_weights is not None:
        colors_arr = colors_arr * np.asarray(color_weights, dtype=np.float32)[:, None]
    weights = (aggregator_np[..., None] * colors_arr[:, None, :]).reshape(C, len(intervals) * 3)
    return weights, len(intervals)


def colorize(im, colors=None, color_weights=None, intervals=None, offset=0):
    """Colorize a channel-first image.

    Works on numpy arrays, torch tensors, and dask arrays (auto-detected).
    The core operation folds ``aggregator(C, N) * colors(C, 3)`` into a
    single weight matrix ``(C, N*3)`` and applies one matmul against the
    flattened image. Benchmarked at parity with ``opt_einsum`` across a
    wide range of shapes and backends (CPU/MPS/CUDA, float32) with 50-trial
    t-tests showing no statistically significant difference. The matmul
    path avoids the ``opt_einsum`` dependency and its ~0.1ms path-finding
    overhead, which matters for small arrays and is noise for large ones.

    Parameters
    ----------
    im : array, tensor, or dask array
        Shape ``(C, *spatial)`` — channel-first image.
    colors : array-like, optional
        Shape ``(C, 3)`` — RGB color per channel. Auto-generated if None.
    color_weights : array-like, optional
        Shape ``(C,)`` — per-channel weights.
    intervals : list of int, optional
        Channel groupings for multi-excitation composites.
    offset : float
        Hue rotation offset in radians.

    Returns
    -------
    array, tensor, or dask array
        Shape ``(*spatial, 3)`` or ``(N, *spatial, 3)`` if intervals given.
    """
    module = get_module(im)
    C = im.shape[0]
    spatial_shape = im.shape[1:]

    if intervals is None:
        intervals = [C]

    if colors is None:
        colors = _make_colors(C, offset)

    device = getattr(im, "device", None)
    weights, N = _build_weights(C, intervals, colors, color_weights, module=module, device=device)

    if module == np:
        out_flat = weights.T @ im.reshape(C, -1).astype(np.float32)
        out = np.moveaxis(out_flat.reshape(N, 3, *spatial_shape), 1, -1)
        return out.squeeze(axis=0) if N == 1 else out

    # Check for dask before torch (dask arrays return np from get_module)
    if isinstance(im, da.Array):
        im_flat = im.reshape(C, -1)
        out_flat = da.dot(weights.T, im_flat)
        out = da.moveaxis(out_flat.reshape(N, 3, *spatial_shape), 1, -1)
        return out.squeeze(axis=0) if N == 1 else out

    # Torch path
    out_flat = weights.T @ im.reshape(C, -1).float()
    out = out_flat.reshape(N, 3, *spatial_shape).movedim(1, -1)
    return out.squeeze(0) if N == 1 else out




def rgb_flow(dP, transparency=True, mask=None, norm=True, device=None):
    """Convert a flow field to an RGB(A) visualization.

    Accepts batched ``(B, D, *spatial)`` or unbatched ``(D, *spatial)`` input
    as numpy arrays or torch tensors. Returns the same type as input.

    The algorithm uses complex-number sinebow coloring: each 2D flow vector
    is mapped to an angle, then projected onto three 120-degree-spaced
    cosine basis functions to produce RGB. Magnitude is encoded as either
    an alpha channel (transparency=True) or brightness modulation.

    Parameters
    ----------
    dP : array or tensor
        Flow field, shape ``(B, D, *spatial)`` or ``(D, *spatial)``.
    transparency : bool
        If True, encode magnitude as alpha channel (RGBA output).
        If False, modulate RGB brightness by magnitude.
    mask : array or tensor, optional
        Binary mask applied to the alpha channel when transparency is True.
    norm : bool
        Normalize magnitude to [0, 1] via percentile rescaling.
    device : torch.device, optional
        Target device. Auto-detected from input if not given.

    Returns
    -------
    uint8 array or tensor
        Shape ``(*spatial, 3)`` or ``(*spatial, 4)`` (with transparency).
    """
    is_numpy = isinstance(dP, np.ndarray)

    if device is None:
        device = torch.device('cpu')
    if isinstance(dP, torch.Tensor):
        device = dP.device
    else:
        dP = torch.from_numpy(np.ascontiguousarray(dP)).to(device)

    squeezed = False
    if dP.ndim == dP.shape[0] + 1:
        dP = dP.unsqueeze(0)
        squeezed = True

    mag = torch_norm(dP, dim=1)
    dP = safe_divide(dP, mag.unsqueeze(1))

    if norm:
        mag = normalize99(mag)

    vecs = dP[:, 0] + dP[:, 1] * 1j
    roots = torch.exp(1j * np.pi * (2 * torch.arange(3, device=device) / 3 + 1))
    rgb = (torch.real(vecs.unsqueeze(-1) * roots.view(1, 1, 1, -1)) + 1) / 2

    if transparency:
        im = torch.cat((rgb, mag[..., None]), dim=-1)
    else:
        im = rgb * mag[..., None]

    im = (torch.clamp(im, 0, 1) * 255).type(torch.uint8)

    if squeezed:
        im = im.squeeze(0)
    if is_numpy:
        im = im.cpu().numpy()

    return im
