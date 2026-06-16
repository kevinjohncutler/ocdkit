"""WGPU-based scatter rasterizer — drop-in replacement for datashader's
points + count + shade pipeline.

Pattern: rasterize points into a u32 counts buffer (compute shader with
``atomicAdd``), optionally apply circular add-pool spread (mirrors
datashader's ``tf.spread(how='add')``), then delegate post-processing
to :mod:`.core` for max-reduction, eq_hist
(histogram + scan), and colormap-apply.

Public API
----------
- :func:`render_scatter_gpu`     end-to-end full-GPU pipeline
- :func:`wgpu_scatter`           same, plus ``ax.imshow`` for matplotlib
- :func:`rasterize_points`       just the count grid, returns ndarray
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

import wgpu

from .core import (
    ShadeConfig,
    get_device,
    get_pipeline,
    readback_rgba,
    shade_count_buffer,
)


# ─────────────────────────── scatter-specific shaders ────────────────────────
_POINT_SHADER = """
struct Uniforms {
    x_min: f32, x_max: f32, y_min: f32, y_max: f32,
    width: u32, height: u32, n_points: u32, pad: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_points) { return; }
    let p = points[i];
    let nx = (p.x - u.x_min) / (u.x_max - u.x_min);
    let ny = (p.y - u.y_min) / (u.y_max - u.y_min);
    if (nx < 0.0 || nx >= 1.0 || ny < 0.0 || ny >= 1.0) { return; }
    let px = u32(nx * f32(u.width));
    let py = u32(ny * f32(u.height));
    atomicAdd(&bins[py * u.width + px], 1u);
}
"""

# Datashader's tf.spread default for numerical (count) data is `how='add'`:
# overlapping spread disks SUM together so two close points produce a
# brighter region than one isolated point.
_SPREAD_SHADER = """
struct SpreadUniforms { width: u32, height: u32, radius: u32, pad: u32 };

@group(0) @binding(0) var<uniform> u: SpreadUniforms;
@group(0) @binding(1) var<storage, read> bins_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> bins_out: array<u32>;

@compute @workgroup_size(8, 8)
fn cs_spread(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) { return; }

    let r = i32(u.radius);
    let r2 = r * r;
    let xi = i32(x);
    let yi = i32(y);

    var sum: u32 = 0u;
    for (var dy = -r; dy <= r; dy = dy + 1) {
        let ny = yi + dy;
        if (ny < 0 || ny >= i32(u.height)) { continue; }
        let dy2 = dy * dy;
        for (var dx = -r; dx <= r; dx = dx + 1) {
            if (dx * dx + dy2 > r2) { continue; }
            let nx = xi + dx;
            if (nx < 0 || nx >= i32(u.width)) { continue; }
            sum = sum + bins_in[u32(ny) * u.width + u32(nx)];
        }
    }
    bins_out[y * u.width + x] = sum;
}
"""


_VIS = wgpu.ShaderStage.COMPUTE
_BGL_RASTER = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": wgpu.BufferBindingType.storage}},
]
_BGL_SPREAD = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": wgpu.BufferBindingType.storage}},
]


# ─────────────────────────── rasterize → counts ──────────────────────────────
def rasterize_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    plot_width: int = 800,
    plot_height: int = 600,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    spread_radius: int = 0,
) -> np.ndarray:
    """Rasterize (x, y) points into a 2-D count grid via WGPU.

    Returns an ``(H, W)`` float32 ndarray of per-pixel point counts,
    aligned with output ``[origin='lower', extent=[x_min, x_max, y_min, y_max]]``.

    If ``spread_radius > 0`` a circular ADD-pool dilation is applied
    (matches datashader's ``tf.spread(px=..., how='add')``).
    """
    bins_buf, n_bins = _rasterize_to_buffer(
        x, y,
        plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range,
        spread_radius=spread_radius,
    )
    if bins_buf is None:
        return np.zeros((plot_height, plot_width), dtype=np.float32)

    device = get_device()
    rb = device.create_buffer(
        size=n_bins * 4,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    enc = device.create_command_encoder()
    enc.copy_buffer_to_buffer(bins_buf, 0, rb, 0, n_bins * 4)
    device.queue.submit([enc.finish()])
    rb.map_sync(wgpu.MapMode.READ)
    raw = rb.read_mapped()
    rb.unmap()
    return np.frombuffer(raw, dtype=np.uint32).reshape(plot_height, plot_width).astype(np.float32)


def _rasterize_to_buffer(
    x, y, *,
    plot_width, plot_height, x_range, y_range, spread_radius,
):
    """Run rasterize (+ optional spread) kernels and return the final
    GPU bins buffer + n_bins.  ``bins_buf`` is None when ``n_points == 0``."""
    n = len(x)
    if n == 0:
        return None, plot_width * plot_height
    if x_range is None:
        x_range = (float(np.min(x)), float(np.max(x)))
    if y_range is None:
        y_range = (float(np.min(y)), float(np.max(y)))

    device = get_device()
    n_bins = plot_width * plot_height

    pts = np.ascontiguousarray(np.column_stack(
        [np.asarray(x, np.float32), np.asarray(y, np.float32)]))
    points_buf = device.create_buffer_with_data(
        data=pts.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    raster_uniforms = np.zeros(8, dtype=np.float32)
    raster_uniforms[:4] = [x_range[0], x_range[1], y_range[0], y_range[1]]
    raster_uniforms.view(np.uint32)[4:7] = [plot_width, plot_height, n]
    raster_ubo = device.create_buffer_with_data(
        data=raster_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    bins_buf = device.create_buffer(
        size=n_bins * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    encoder = device.create_command_encoder()

    # Rasterize.
    pipe, bgl = get_pipeline(_POINT_SHADER, "cs_main", _BGL_RASTER)
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": raster_ubo, "offset": 0, "size": raster_uniforms.nbytes}},
        {"binding": 1, "resource": {"buffer": points_buf, "offset": 0, "size": pts.nbytes}},
        {"binding": 2, "resource": {"buffer": bins_buf, "offset": 0, "size": n_bins * 4}},
    ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe)
    p.set_bind_group(0, bg)
    p.dispatch_workgroups((n + 63) // 64)
    p.end()

    # Spread (optional, add-pool).
    final_bins = bins_buf
    if spread_radius > 0:
        sp_uniforms = np.array([plot_width, plot_height, spread_radius, 0], dtype=np.uint32)
        sp_ubo = device.create_buffer_with_data(
            data=sp_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        sp_out_buf = device.create_buffer(
            size=n_bins * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        pipe_sp, bgl_sp = get_pipeline(_SPREAD_SHADER, "cs_spread", _BGL_SPREAD)
        bg_sp = device.create_bind_group(layout=bgl_sp, entries=[
            {"binding": 0, "resource": {"buffer": sp_ubo, "offset": 0, "size": sp_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": bins_buf, "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": sp_out_buf, "offset": 0, "size": n_bins * 4}},
        ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipe_sp)
        p.set_bind_group(0, bg_sp)
        p.dispatch_workgroups((plot_width + 7) // 8, (plot_height + 7) // 8)
        p.end()
        final_bins = sp_out_buf

    device.queue.submit([encoder.finish()])
    return final_bins, n_bins


# ─────────────────────────── full GPU pipeline ───────────────────────────────
def render_scatter_gpu(
    x: np.ndarray,
    y: np.ndarray,
    *,
    plot_width: int = 800,
    plot_height: int = 600,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    spread_radius: int = 0,
    cmap_name: str = "viridis",
    transfer: str = "eq_hist",  # "linear" | "eq_hist" | "log" | "cbrt"
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> np.ndarray:
    """Rasterize + spread, then delegate max-reduce + eq_hist + colormap to
    :func:`.core.shade_count_buffer`.  Reads back only the
    final uint8 RGBA buffer."""
    bins_buf, n_bins = _rasterize_to_buffer(
        x, y,
        plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range,
        spread_radius=spread_radius,
    )
    if bins_buf is None:
        return np.zeros((plot_height, plot_width, 4), dtype=np.uint8)

    rgba_buf = shade_count_buffer(
        bins_buf, n_bins, plot_width, plot_height,
        ShadeConfig(
            transfer=transfer,
            vmin=vmin, vmax=vmax,
            alpha_min=alpha_min, alpha_max=alpha_max,
            cmap_name=cmap_name,
        ),
    )
    return readback_rgba(rgba_buf, plot_width, plot_height)


def wgpu_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ax=None,
    plot_width: int = 800,
    plot_height: int = 600,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    transfer: str = "eq_hist",
    spread_radius: int = 5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha_min: float = 0.15,
    alpha_max: float = 1.0,
):
    """matplotlib drop-in equivalent of datashader's points+shade.

    Renders to RGBA via :func:`render_scatter_gpu`, then ``ax.imshow``s
    with ``extent=`` so axes/ticks/labels behave normally.  Returns
    ``(ax, AxesImage)``.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    rgba = render_scatter_gpu(
        np.asarray(x), np.asarray(y),
        plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range,
        spread_radius=spread_radius,
        cmap_name=cmap, transfer=transfer,
        vmin=vmin, vmax=vmax,
        alpha_min=alpha_min, alpha_max=alpha_max,
    )
    if x_range is None:
        x_range = (float(np.min(x)), float(np.max(x)))
    if y_range is None:
        y_range = (float(np.min(y)), float(np.max(y)))
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    im = ax.imshow(rgba, extent=extent, origin="lower", aspect="auto", interpolation="nearest")
    return ax, im
