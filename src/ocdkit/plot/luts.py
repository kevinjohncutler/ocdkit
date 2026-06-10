"""Colormap LUTs for GPU / web colormapping — one builder, three forms.

A linked-tile viewer colormaps intensity on the GPU by uploading a 256-entry
RGBA LUT and indexing it in the shader (changing colormap is a uniform-level
swap, not a re-colormap). This builds that LUT in whichever form the render
path needs, from the same colormap ``name``:

- ``'uint8'``     — sRGB uint8 (256x4). Plain SDR colormap (8-bit canvas).
- ``'sdr_float'`` — linear Display-P3 float, clipped to >=0 (256x4). Reproduces
  the plain colormap when rendered through an extended/HDR canvas + OETF.
- ``'hdr_float'`` — linear Display-P3 float lifted so ``1.0`` is SDR white and
  bright colours exceed 1.0 (256x4). Glows into the display headroom.

The float forms share the grid's ``1.0 == SDR white`` convention so colormapped
tiles, density rasters, etc. all match through the same OETF.
"""
from __future__ import annotations

import numpy as np
from cmap import Colormap

from .hdr_cmap import (make_hdr_cmap_lut, SDR_WHITE_NITS, HDR_PEAK_NITS_DEFAULT,
                       _XYZ_FROM_SRGB, _P3_FROM_XYZ, _srgb_to_linear)

DEFAULT_COLORMAPS = ("magma", "viridis", "gray", "plasma",
                     "inferno", "cividis", "turbo")


def colormap_lut(name: str, kind: str = "uint8") -> np.ndarray:
    """One colormap → a 256x4 RGBA LUT (see module docstring for ``kind``)."""
    x = np.linspace(0, 1, 256)
    if kind == "uint8":
        return (np.asarray(Colormap(name)(x)) * 255 + 0.5).astype(np.uint8)
    if kind == "hdr_float":
        scale = HDR_PEAK_NITS_DEFAULT / SDR_WHITE_NITS      # peak_nits -> SDR white
        lin = np.clip(make_hdr_cmap_lut(name) * scale, 0.0, None).astype(np.float32)
        return np.concatenate([lin, np.ones((lin.shape[0], 1), np.float32)], axis=1)
    if kind == "sdr_float":
        srgb = np.asarray(Colormap(name)(x))[:, :3].astype(np.float32)
        lin = np.clip(_srgb_to_linear(srgb) @ _XYZ_FROM_SRGB.T @ _P3_FROM_XYZ.T,
                      0.0, None).astype(np.float32)
        return np.concatenate([lin, np.ones((lin.shape[0], 1), np.float32)], axis=1)
    raise ValueError(f"unknown LUT kind {kind!r}")


def colormap_luts(names=DEFAULT_COLORMAPS, kind: str = "uint8", *, round_float: int = 5):
    """``{name: flat RGBA LUT list}`` for several colormaps — JSON-ready.

    uint8 LUTs become int lists; float LUTs are rounded to ``round_float``
    decimals. Colormaps that fail to build are skipped (logged)."""
    out = {}
    for n in names:
        try:
            flat = colormap_lut(n, kind).reshape(-1)
            out[n] = (flat.tolist() if kind == "uint8"
                      else [round(float(v), round_float) for v in flat])
        except Exception as e:                              # noqa: BLE001
            print(f"[luts] {n} ({kind}):", e)
    return out
