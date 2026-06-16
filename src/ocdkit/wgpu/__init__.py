"""WGPU-backed plotting primitives — generic GPU rasterizers.

Compute-shader density/line/scatter rasterization, the complement to
:mod:`ocdkit.tileserve` (which does fragment-shader *tile* rendering).

Sub-modules
-----------
- :mod:`.core`     shared GPU primitives (device + pipeline cache, LUT cache,
                   max-reduce, histogram + scan, colormap-apply, ``ShadeConfig``)
- :mod:`.scatter`  point→count→shade pipeline (drop-in for datashader scatter)
- :mod:`.aggregators`  categorical (by/count_cat) + value (max/sum/mean) scatter
- :mod:`.lines`    density-line renderer (``DensityLineRenderer``,
                   ``rasterize_spectra``) + raw shader strings

Domain decoupling
-----------------
The line renderer is colormap- and spectroscopy-agnostic: the default
alpha-ramped LUT is built internally (``lines._transparent_lut``), and
``render_spectral_rgb`` takes precomputed per-channel CIE ``xyz_weights``
rather than wavelengths, so the CIE/wavelength math lives in the caller.
"""
from __future__ import annotations

# ── core primitives ─────────────────────────────────────────────────────────
from .core import (
    ShadeConfig,
    get_cmap_lut_buffer,
    get_device,
    get_pipeline,
    pack_cmap_lut,
    readback_rgba,
    shade_count_buffer,
)

# ── scatter ────────────────────────────────────────────────────────────────
from .scatter import (
    rasterize_points,
    render_scatter_gpu,
    wgpu_scatter,
)

# ── non-count aggregators (by/count_cat, max/sum/mean of value) ────────────
from .aggregators import (
    render_scatter_by,
    render_scatter_value,
)

# ── lines / spectra ────────────────────────────────────────────────────────
from .lines import (
    COLORMAP_SHADER,
    DOWNSAMPLE_SHADER,
    DensityLineRenderer,
    HISTOGRAM_BLUR_SHADER,
    HISTOGRAM_SHADER,
    MINMAX_FIND_SHADER,
    PREFIX_SUM_SHADER,
    RASTER_TEMPLATE,
    generate_point_data,
    infer_channel_intervals,
    rasterize_spectra,
    render_spectra_wgpu,
    render_spectral_rgb,
)


def prewarm(plot_width: int = 1395, plot_height: int = 462) -> None:
    """Pre-create a ``DensityLineRenderer`` to amortize first-render cost.

    First WGPU dispatch incurs ~400 ms on Apple Silicon: adapter request,
    device creation, shader compilation (~8 pipelines), buffer allocation.
    Calling ``prewarm()`` once moves that cost off the interactive path. The
    renderer is cached in ``ocdkit.wgpu.lines._RENDERER_CACHE`` keyed by
    ``(plot_width, plot_height)``; subsequent renders at the same canvas size
    reuse it.
    """
    from .lines import DensityLineRenderer, _RENDERER_CACHE
    cache_key = (int(plot_width), int(plot_height))
    if cache_key not in _RENDERER_CACHE:
        _RENDERER_CACHE[cache_key] = DensityLineRenderer(plot_width, plot_height)


__all__ = [
    # core
    "ShadeConfig",
    "get_cmap_lut_buffer",
    "get_device",
    "get_pipeline",
    "pack_cmap_lut",
    "readback_rgba",
    "shade_count_buffer",
    # scatter
    "rasterize_points",
    "render_scatter_gpu",
    "wgpu_scatter",
    "render_scatter_by",
    "render_scatter_value",
    # lines
    "DensityLineRenderer",
    "infer_channel_intervals",
    "rasterize_spectra",
    "render_spectra_wgpu",
    "render_spectral_rgb",
    "generate_point_data",
    # raw shader strings
    "RASTER_TEMPLATE",
    "HISTOGRAM_SHADER",
    "HISTOGRAM_BLUR_SHADER",
    "PREFIX_SUM_SHADER",
    "COLORMAP_SHADER",
    "DOWNSAMPLE_SHADER",
    "MINMAX_FIND_SHADER",
    # warm-up
    "prewarm",
]
