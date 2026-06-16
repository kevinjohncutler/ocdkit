"""Non-count aggregators on top of the WGPU scatter pipeline.

What this module covers (and why)
---------------------------------
Datashader exposes many ``Reduction`` types via ``cvs.points(df, x, y, agg=…)``.
The two patterns that matter once you go beyond plain ``count``:

  - **``by(cat, count())`` / ``count_cat``** — categorical density.  K parallel
    count grids; the per-pixel color is a weighted average of category colors,
    the per-pixel alpha follows the chosen transfer curve on the *total*.
    Used in datashader's Categorical example to show race-by-pixel in the US
    Census or species-by-pixel in synthetic 5-cluster data.

  - **``max(col)`` / ``mean(col)`` / ``sum(col)``** — value-weighted scatter.
    Each point carries a scalar; the bin holds the max / sum / count-then-divide.
    Used in datashader's Pipeline example to show "average property per
    pixel" rather than just point density.

Both are implemented on top of :mod:`.core` for shading.

Public surface
--------------
- :func:`render_scatter_by`     ``cvs.points(...).agg = by(cat, count())`` + shade
- :func:`render_scatter_value`  ``cvs.points(...).agg = max|mean|sum(col)`` + shade

Both return a uint8 ``(H, W, 4)`` ndarray, the same format as
:func:`.scatter.render_scatter_gpu`.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

import wgpu

from .core import (
    ShadeConfig,
    _BGL_HIST,
    _BGL_REDUCE,
    _BGL_REDUCE_MAX_MIN,
    _BGL_SCAN,
    _HISTOGRAM_SHADER,
    _REDUCE_MAX_MIN_SHADER,
    _REDUCE_MAX_SHADER,
    _SCAN_SHADER,
    get_device,
    get_pipeline,
    readback_rgba,
)


# ─────────────────────────── categorical (by / count_cat) ───────────────────
_POINT_BY_SHADER = """
struct Uniforms {
    x_min: f32, x_max: f32, y_min: f32, y_max: f32,
    width: u32, height: u32, n_points: u32, n_categories: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> categories: array<u32>;
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>>;

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
    let cat = categories[i];
    if (cat >= u.n_categories) { return; }
    let plane = u.width * u.height;
    atomicAdd(&bins[cat * plane + py * u.width + px], 1u);
}
"""

# Reduce K planes -> totals grid (sum across categories for every pixel).
_TOTAL_SHADER = """
struct TotalUniforms { width: u32, height: u32, n_categories: u32, p0: u32 };

@group(0) @binding(0) var<uniform> u: TotalUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;          // K*H*W
@group(0) @binding(2) var<storage, read_write> totals: array<u32>;  // H*W

@compute @workgroup_size(8, 8)
fn cs_total(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let pix = gid.y * u.width + gid.x;
    let plane = u.width * u.height;
    var s: u32 = 0u;
    for (var k: u32 = 0u; k < u.n_categories; k = k + 1u) {
        s = s + bins[k * plane + pix];
    }
    totals[pix] = s;
}
"""

# Categorical colormap: weighted-average of category colors, alpha from totals.
_CAT_COLORMAP_SHADER = """
struct CatCmapUniforms {
    width: u32, height: u32, n_categories: u32, transfer_mode: u32,
    vmin: f32, vmax: f32, alpha_min: f32, alpha_max: f32,
    auto_flags: u32, _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<uniform> u: CatCmapUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;        // K*H*W
@group(0) @binding(2) var<storage, read> totals: array<u32>;      // H*W
@group(0) @binding(3) var<storage, read> total_max: array<u32>;
@group(0) @binding(4) var<storage, read> color_key: array<u32>;   // K packed RGBA8
@group(0) @binding(5) var<storage, read> cdf: array<f32>;
@group(0) @binding(6) var<storage, read_write> rgba_out: array<u32>;
@group(0) @binding(7) var<storage, read> total_min: array<u32>;

fn unpack_rgb(packed: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(packed & 0xFFu),
        f32((packed >> 8u) & 0xFFu),
        f32((packed >> 16u) & 0xFFu),
    );
}

@compute @workgroup_size(8, 8)
fn cs_cat_colormap(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let pix = gid.y * u.width + gid.x;
    let total = totals[pix];
    if (total == 0u) { rgba_out[pix] = 0u; return; }

    let plane = u.width * u.height;
    var rgb: vec3<f32> = vec3<f32>(0.0);
    for (var k: u32 = 0u; k < u.n_categories; k = k + 1u) {
        let c = bins[k * plane + pix];
        if (c == 0u) { continue; }
        rgb = rgb + f32(c) * unpack_rgb(color_key[k]);
    }
    rgb = rgb / f32(total);

    // Alpha from totals via the chosen transfer curve.
    let mx = max(total_max[0], 1u);
    let mn_raw = total_min[0];
    let mn = select(0u, mn_raw, mn_raw != 0xFFFFFFFFu);
    let lo = select(u.vmin, f32(mn), (u.auto_flags & 1u) != 0u);
    let hi = select(u.vmax, f32(mx), (u.auto_flags & 2u) != 0u);
    let span = max(hi - lo, 1.0e-6);

    var norm: f32;
    if (u.transfer_mode == 1u) {
        let pre = clamp(f32(total) / f32(mx), 0.0, 1.0);
        let bin_f = clamp(pre * 255.0, 0.0, 255.0);
        let bin_lo = u32(floor(bin_f));
        let bin_hi = min(bin_lo + 1u, 255u);
        let t = bin_f - f32(bin_lo);
        norm = mix(cdf[bin_lo], cdf[bin_hi], t);
    } else if (u.transfer_mode == 2u) {
        let cf = clamp(f32(total), lo, hi);
        norm = log(1.0 + cf - lo) / log(1.0 + span);
    } else if (u.transfer_mode == 3u) {
        let lin = clamp((f32(total) - lo) / span, 0.0, 1.0);
        norm = pow(lin, 1.0 / 3.0);
    } else {
        norm = clamp((f32(total) - lo) / span, 0.0, 1.0);
    }

    let alpha_t = u.alpha_min + (u.alpha_max - u.alpha_min) * norm;
    let final_a = u32(clamp(alpha_t * 255.0, 0.0, 255.0));
    let r = u32(clamp(rgb.x, 0.0, 255.0));
    let g = u32(clamp(rgb.y, 0.0, 255.0));
    let b = u32(clamp(rgb.z, 0.0, 255.0));
    rgba_out[pix] = r | (g << 8u) | (b << 16u) | (final_a << 24u);
}
"""


# ─────────────────────────── value (max / mean / sum) ───────────────────────
# WGSL has no f32 atomics.  For non-negative floats, atomicMax<u32> on the bit
# pattern preserves order (IEEE 754 monotonicity), so atomicMax<u32>(bitcast<u32>(v))
# is exact and lock-free.  For sum/mean we use the standard CAS loop.
_POINT_VAL_MAX_SHADER = """
struct Uniforms {
    x_min: f32, x_max: f32, y_min: f32, y_max: f32,
    width: u32, height: u32, n_points: u32, pad: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>>;

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
    let v = values[i];
    if (v < 0.0) { return; }
    atomicMax(&bins[py * u.width + px], bitcast<u32>(v));
}
"""

# Sum aggregator via float CAS loop on bitcast<u32>(f32).
_POINT_VAL_SUM_SHADER = """
struct Uniforms {
    x_min: f32, x_max: f32, y_min: f32, y_max: f32,
    width: u32, height: u32, n_points: u32, pad: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;

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
    let v = values[i];
    let idx = py * u.width + px;
    atomicAdd(&counts[idx], 1u);
    // CAS loop to do float sum on bitcast<u32>.
    var old_bits: u32 = atomicLoad(&bins[idx]);
    loop {
        let old_f: f32 = bitcast<f32>(old_bits);
        let new_bits: u32 = bitcast<u32>(old_f + v);
        let res = atomicCompareExchangeWeak(&bins[idx], old_bits, new_bits);
        if (res.exchanged) { break; }
        old_bits = res.old_value;
    }
}
"""

# Shading kernel for value-aggregator results: bins is bitcast<u32>(f32),
# we decode back to float for normalization.  Optional ``divide_by`` lets us
# implement mean = sum / count cheaply at shade time.
_VAL_COLORMAP_SHADER = """
struct ValCmapUniforms {
    width: u32, height: u32, mode: u32, transfer_mode: u32,
    // mode=0  bins is f32 max/sum (no divide)
    // mode=1  bins is f32 sum, divide by counts -> mean
    vmin: f32, vmax: f32, alpha_min: f32, alpha_max: f32,
    auto_flags: u32, _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<uniform> u: ValCmapUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;     // f32-as-u32
@group(0) @binding(2) var<storage, read> counts: array<u32>;   // for mean
@group(0) @binding(3) var<storage, read> agg_max: array<f32>;
@group(0) @binding(4) var<storage, read> lut: array<u32>;
@group(0) @binding(5) var<storage, read_write> rgba_out: array<u32>;
@group(0) @binding(6) var<storage, read> agg_min: array<u32>;  // f32-as-u32 (sentinel u32::MAX)

@compute @workgroup_size(8, 8)
fn cs_val_colormap(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let idx = gid.y * u.width + gid.x;
    let n = counts[idx];
    if (n == 0u) { rgba_out[idx] = 0u; return; }
    var v: f32 = bitcast<f32>(bins[idx]);
    if (u.mode == 1u) { v = v / f32(n); }

    let mx = max(agg_max[0], 1.0e-6);
    let mn_raw = agg_min[0];
    let mn = select(0.0, bitcast<f32>(mn_raw), mn_raw != 0xFFFFFFFFu);
    let lo = select(u.vmin, mn, (u.auto_flags & 1u) != 0u);
    let hi = select(u.vmax, mx, (u.auto_flags & 2u) != 0u);
    let span = max(hi - lo, 1.0e-6);

    var norm: f32;
    if (u.transfer_mode == 2u) {
        let cf = clamp(v, lo, hi);
        norm = log(1.0 + cf - lo) / log(1.0 + span);
    } else if (u.transfer_mode == 3u) {
        let lin = clamp((v - lo) / span, 0.0, 1.0);
        norm = pow(lin, 1.0 / 3.0);
    } else {
        norm = clamp((v - lo) / span, 0.0, 1.0);
    }

    let lut_idx = u32(clamp(norm * 255.0, 0.0, 255.0));
    let entry = lut[lut_idx];
    let alpha_t = u.alpha_min + (u.alpha_max - u.alpha_min) * norm;
    let lut_a = f32((entry >> 24u) & 0xFFu);
    let final_a = u32(clamp(alpha_t * lut_a, 0.0, 255.0));
    rgba_out[idx] = (entry & 0x00FFFFFFu) | (final_a << 24u);
}
"""

# Float max-reduction (bitcast<u32> bin → float).
_REDUCE_VAL_MAX_SHADER = """
struct ReduceUniforms { n: u32, mode: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<uniform> u: ReduceUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;     // f32-as-u32
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> max_out: atomic<u32>;

var<workgroup> wg_max: atomic<u32>;

@compute @workgroup_size(256)
fn cs_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if (lid.x == 0u) { atomicStore(&wg_max, 0u); }
    workgroupBarrier();
    if (gid.x < u.n) {
        let n_pts = counts[gid.x];
        if (n_pts > 0u) {
            var v: f32 = bitcast<f32>(bins[gid.x]);
            if (u.mode == 1u) { v = v / f32(n_pts); }
            // bitcast monotonic for non-negative floats
            atomicMax(&wg_max, bitcast<u32>(v));
        }
    }
    workgroupBarrier();
    if (lid.x == 0u) { atomicMax(&max_out, atomicLoad(&wg_max)); }
}
"""

# Fused float max + min-nonzero reduction (single dispatch when auto_vmin).
_REDUCE_VAL_MAX_MIN_SHADER = """
struct ReduceUniforms { n: u32, mode: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<uniform> u: ReduceUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> max_out: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> min_out: atomic<u32>;

var<workgroup> wg_max: atomic<u32>;
var<workgroup> wg_min: atomic<u32>;

@compute @workgroup_size(256)
fn cs_reduce_max_min(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if (lid.x == 0u) {
        atomicStore(&wg_max, 0u);
        atomicStore(&wg_min, 0xFFFFFFFFu);
    }
    workgroupBarrier();
    if (gid.x < u.n) {
        let n_pts = counts[gid.x];
        if (n_pts > 0u) {
            var v: f32 = bitcast<f32>(bins[gid.x]);
            if (u.mode == 1u) { v = v / f32(n_pts); }
            let bits = bitcast<u32>(v);
            atomicMax(&wg_max, bits);
            atomicMin(&wg_min, bits);
        }
    }
    workgroupBarrier();
    if (lid.x == 0u) {
        atomicMax(&max_out, atomicLoad(&wg_max));
        atomicMin(&min_out, atomicLoad(&wg_min));
    }
}
"""


# ─────────────────────────── Bind group layouts ─────────────────────────────
_VIS = wgpu.ShaderStage.COMPUTE
_UNI = wgpu.BufferBindingType.uniform
_RO = wgpu.BufferBindingType.read_only_storage
_RW = wgpu.BufferBindingType.storage

_BGL_BY_RASTER = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNI}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RW}},
]
_BGL_TOTAL = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNI}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RW}},
]
_BGL_CAT_CMAP = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNI}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 4, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 5, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 6, "visibility": _VIS, "buffer": {"type": _RW}},
    {"binding": 7, "visibility": _VIS, "buffer": {"type": _RO}},
]
_BGL_VAL_RASTER_MAX = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNI}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RW}},
]
_BGL_VAL_RASTER_SUM = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNI}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RW}},
    {"binding": 4, "visibility": _VIS, "buffer": {"type": _RW}},
]
_BGL_VAL_REDUCE = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNI}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RW}},
]
_BGL_VAL_REDUCE_MAX_MIN = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNI}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RW}},
    {"binding": 4, "visibility": _VIS, "buffer": {"type": _RW}},
]
_BGL_VAL_CMAP = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNI}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 4, "visibility": _VIS, "buffer": {"type": _RO}},
    {"binding": 5, "visibility": _VIS, "buffer": {"type": _RW}},
    {"binding": 6, "visibility": _VIS, "buffer": {"type": _RO}},
]


def _pack_color_key(color_key) -> np.ndarray:
    """Pack a list of (r, g, b[, a]) floats in [0, 1] (or hex strings) into u32 RGBA8 LE."""
    import matplotlib as mpl
    rgba = np.zeros((len(color_key), 4), dtype=np.float32)
    for i, c in enumerate(color_key):
        rgba[i] = mpl.colors.to_rgba(c)
    rgba8 = (rgba * 255).astype(np.uint8)
    return (rgba8[:, 0].astype(np.uint32)
            | (rgba8[:, 1].astype(np.uint32) << 8)
            | (rgba8[:, 2].astype(np.uint32) << 16)
            | (rgba8[:, 3].astype(np.uint32) << 24))


def render_scatter_by(
    x: np.ndarray,
    y: np.ndarray,
    category_ids: np.ndarray,
    *,
    color_key: Sequence,
    plot_width: int = 800,
    plot_height: int = 600,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    transfer: str = "eq_hist",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> np.ndarray:
    """WGPU equivalent of ``cvs.points(df, x, y, by(cat, count())) → tf.shade(color_key=)``.

    ``category_ids`` must be u32-castable (already factorized to 0..K-1).
    ``color_key`` is a length-K iterable of matplotlib-recognizable colors.
    """
    transfer_mode = {"linear": 0, "eq_hist": 1, "log": 2, "cbrt": 3}.get(transfer)
    if transfer_mode is None:
        raise ValueError(f"unknown transfer: {transfer}")

    n = len(x)
    n_cat = len(color_key)
    cat = np.ascontiguousarray(np.asarray(category_ids, np.uint32))
    if cat.size and int(cat.max()) >= n_cat:
        raise ValueError("category id out of range for color_key length")

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
    cats_buf = device.create_buffer_with_data(
        data=cat.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    raster_uniforms = np.zeros(8, dtype=np.float32)
    raster_uniforms[:4] = [x_range[0], x_range[1], y_range[0], y_range[1]]
    raster_uniforms.view(np.uint32)[4:8] = [plot_width, plot_height, n, n_cat]
    raster_ubo = device.create_buffer_with_data(
        data=raster_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    bins_buf = device.create_buffer(
        size=n_bins * n_cat * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    encoder = device.create_command_encoder()

    # Pass A: rasterize per (point, category).
    pipe, bgl = get_pipeline(_POINT_BY_SHADER, "cs_main", _BGL_BY_RASTER)
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": raster_ubo, "offset": 0, "size": raster_uniforms.nbytes}},
        {"binding": 1, "resource": {"buffer": points_buf, "offset": 0, "size": pts.nbytes}},
        {"binding": 2, "resource": {"buffer": cats_buf,   "offset": 0, "size": cat.nbytes}},
        {"binding": 3, "resource": {"buffer": bins_buf,   "offset": 0, "size": n_bins * n_cat * 4}},
    ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe); p.set_bind_group(0, bg)
    p.dispatch_workgroups((n + 63) // 64)
    p.end()

    # Pass B: collapse K planes -> totals grid.
    totals_buf = device.create_buffer(
        size=n_bins * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    total_uniforms = np.array([plot_width, plot_height, n_cat, 0], dtype=np.uint32)
    total_ubo = device.create_buffer_with_data(
        data=total_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    pipe_t, bgl_t = get_pipeline(_TOTAL_SHADER, "cs_total", _BGL_TOTAL)
    bg_t = device.create_bind_group(layout=bgl_t, entries=[
        {"binding": 0, "resource": {"buffer": total_ubo,  "offset": 0, "size": total_uniforms.nbytes}},
        {"binding": 1, "resource": {"buffer": bins_buf,   "offset": 0, "size": n_bins * n_cat * 4}},
        {"binding": 2, "resource": {"buffer": totals_buf, "offset": 0, "size": n_bins * 4}},
    ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe_t); p.set_bind_group(0, bg_t)
    p.dispatch_workgroups((plot_width + 7) // 8, (plot_height + 7) // 8)
    p.end()

    # Pass C: reduce on totals.  Fused max+min when auto_vmin is on, max-only
    # otherwise (matching the cost of explicit-vmin callers pre-auto-vmin).
    auto_vmin = (vmin is None) and (transfer_mode != 1)  # eq_hist redistributes
    auto_vmax = vmax is None

    reduce_uniforms = np.array([n_bins, 0, 0, 0], dtype=np.uint32)
    reduce_ubo = device.create_buffer_with_data(
        data=reduce_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    max_buf = device.create_buffer(
        size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    min_buf = device.create_buffer_with_data(
        data=np.array([0xFFFFFFFF], dtype=np.uint32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    if auto_vmin:
        pipe_r, bgl_r = get_pipeline(_REDUCE_MAX_MIN_SHADER, "cs_reduce_max_min",
                                     _BGL_REDUCE_MAX_MIN)
        bg_r = device.create_bind_group(layout=bgl_r, entries=[
            {"binding": 0, "resource": {"buffer": reduce_ubo, "offset": 0, "size": reduce_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": totals_buf, "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": max_buf,    "offset": 0, "size": 4}},
            {"binding": 3, "resource": {"buffer": min_buf,    "offset": 0, "size": 4}},
        ])
    else:
        pipe_r, bgl_r = get_pipeline(_REDUCE_MAX_SHADER, "cs_reduce", _BGL_REDUCE)
        bg_r = device.create_bind_group(layout=bgl_r, entries=[
            {"binding": 0, "resource": {"buffer": reduce_ubo, "offset": 0, "size": reduce_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": totals_buf, "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": max_buf,    "offset": 0, "size": 4}},
        ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe_r); p.set_bind_group(0, bg_r)
    p.dispatch_workgroups((n_bins + 255) // 256)
    p.end()

    # Pass D (eq_hist only): histogram + scan on totals.
    cdf_buf = device.create_buffer(size=256 * 4, usage=wgpu.BufferUsage.STORAGE)
    if transfer_mode == 1:
        hist_uniforms = np.array([plot_width, plot_height, 0, 0], dtype=np.uint32)
        hist_ubo = device.create_buffer_with_data(
            data=hist_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        hist_buf = device.create_buffer(
            size=256 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        pipe_h, bgl_h = get_pipeline(_HISTOGRAM_SHADER, "cs_histogram", _BGL_HIST)
        bg_h = device.create_bind_group(layout=bgl_h, entries=[
            {"binding": 0, "resource": {"buffer": hist_ubo,   "offset": 0, "size": hist_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": totals_buf, "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": max_buf,    "offset": 0, "size": 4}},
            {"binding": 3, "resource": {"buffer": hist_buf,   "offset": 0, "size": 256 * 4}},
        ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipe_h); p.set_bind_group(0, bg_h)
        p.dispatch_workgroups((plot_width + 7) // 8, (plot_height + 7) // 8)
        p.end()

        pipe_s, bgl_s = get_pipeline(_SCAN_SHADER, "cs_scan", _BGL_SCAN)
        bg_s = device.create_bind_group(layout=bgl_s, entries=[
            {"binding": 0, "resource": {"buffer": hist_buf, "offset": 0, "size": 256 * 4}},
            {"binding": 1, "resource": {"buffer": cdf_buf,  "offset": 0, "size": 256 * 4}},
        ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipe_s); p.set_bind_group(0, bg_s)
        p.dispatch_workgroups(1)
        p.end()

    # Pass E: categorical colormap-apply.
    color_key_packed = _pack_color_key(color_key)
    color_key_buf = device.create_buffer_with_data(
        data=color_key_packed.tobytes(), usage=wgpu.BufferUsage.STORAGE)

    auto_flags = (1 if auto_vmin else 0) | (2 if auto_vmax else 0)

    cmap_uniforms = np.zeros(12, dtype=np.uint32)
    cmap_uniforms[0] = plot_width
    cmap_uniforms[1] = plot_height
    cmap_uniforms[2] = n_cat
    cmap_uniforms[3] = transfer_mode
    cmap_uniforms.view(np.float32)[4] = float(vmin) if vmin is not None else 0.0
    cmap_uniforms.view(np.float32)[5] = float(vmax) if vmax is not None else 0.0
    cmap_uniforms.view(np.float32)[6] = float(alpha_min)
    cmap_uniforms.view(np.float32)[7] = float(alpha_max)
    cmap_uniforms[8] = auto_flags
    cmap_ubo = device.create_buffer_with_data(
        data=cmap_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)

    rgba_buf = device.create_buffer(
        size=n_bins * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    pipe_c, bgl_c = get_pipeline(_CAT_COLORMAP_SHADER, "cs_cat_colormap", _BGL_CAT_CMAP)
    bg_c = device.create_bind_group(layout=bgl_c, entries=[
        {"binding": 0, "resource": {"buffer": cmap_ubo,      "offset": 0, "size": cmap_uniforms.nbytes}},
        {"binding": 1, "resource": {"buffer": bins_buf,      "offset": 0, "size": n_bins * n_cat * 4}},
        {"binding": 2, "resource": {"buffer": totals_buf,    "offset": 0, "size": n_bins * 4}},
        {"binding": 3, "resource": {"buffer": max_buf,       "offset": 0, "size": 4}},
        {"binding": 4, "resource": {"buffer": color_key_buf, "offset": 0, "size": color_key_packed.nbytes}},
        {"binding": 5, "resource": {"buffer": cdf_buf,       "offset": 0, "size": 256 * 4}},
        {"binding": 6, "resource": {"buffer": rgba_buf,      "offset": 0, "size": n_bins * 4}},
        {"binding": 7, "resource": {"buffer": min_buf,       "offset": 0, "size": 4}},
    ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe_c); p.set_bind_group(0, bg_c)
    p.dispatch_workgroups((plot_width + 7) // 8, (plot_height + 7) // 8)
    p.end()

    device.queue.submit([encoder.finish()])
    return readback_rgba(rgba_buf, plot_width, plot_height)


def render_scatter_value(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    op: str = "max",  # "max" | "sum" | "mean"
    plot_width: int = 800,
    plot_height: int = 600,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    cmap_name: str = "viridis",
    transfer: str = "linear",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> np.ndarray:
    """WGPU equivalent of ``cvs.points(df, x, y, max|sum|mean(col)) → tf.shade(...)``.

    Values must be non-negative (we use ``atomicMax<u32>(bitcast<u32>(v))`` and
    a CAS loop for sum, both of which assume non-negative IEEE-754 inputs).
    """
    if op not in ("max", "sum", "mean"):
        raise ValueError(f"unsupported op: {op}; choose max|sum|mean")
    transfer_mode = {"linear": 0, "log": 2, "cbrt": 3}.get(transfer)
    if transfer_mode is None:
        # eq_hist on float aggregates would need a separate float-histogram path;
        # leave it out until we actually need it.
        raise ValueError(f"transfer {transfer!r} not supported for value aggregators "
                         "(use linear, log, or cbrt)")

    n = len(x)
    if x_range is None:
        x_range = (float(np.min(x)), float(np.max(x)))
    if y_range is None:
        y_range = (float(np.min(y)), float(np.max(y)))

    device = get_device()
    n_bins = plot_width * plot_height

    pts = np.ascontiguousarray(np.column_stack(
        [np.asarray(x, np.float32), np.asarray(y, np.float32)]))
    vals = np.ascontiguousarray(np.asarray(values, np.float32))
    if vals.min(initial=0.0) < 0.0:
        raise ValueError("render_scatter_value requires values >= 0 "
                         "(WGSL atomic float trick assumes IEEE-754 non-negative monotonicity)")

    points_buf = device.create_buffer_with_data(
        data=pts.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    vals_buf = device.create_buffer_with_data(
        data=vals.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    raster_uniforms = np.zeros(8, dtype=np.float32)
    raster_uniforms[:4] = [x_range[0], x_range[1], y_range[0], y_range[1]]
    raster_uniforms.view(np.uint32)[4:7] = [plot_width, plot_height, n]
    raster_ubo = device.create_buffer_with_data(
        data=raster_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    bins_buf = device.create_buffer(
        size=n_bins * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    counts_buf = device.create_buffer(
        size=n_bins * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    encoder = device.create_command_encoder()
    if op == "max":
        pipe, bgl = get_pipeline(_POINT_VAL_MAX_SHADER, "cs_main", _BGL_VAL_RASTER_MAX)
        # We still need a counts grid (for "where any point landed").  Run a
        # separate count pass to populate it.  Reuse scatter._POINT_SHADER.
        from .scatter import _BGL_RASTER, _POINT_SHADER  # local to avoid circular at module load
        pipe_c, bgl_c = get_pipeline(_POINT_SHADER, "cs_main", _BGL_RASTER)
        bg_c = device.create_bind_group(layout=bgl_c, entries=[
            {"binding": 0, "resource": {"buffer": raster_ubo, "offset": 0, "size": raster_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": points_buf, "offset": 0, "size": pts.nbytes}},
            {"binding": 2, "resource": {"buffer": counts_buf, "offset": 0, "size": n_bins * 4}},
        ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipe_c); p.set_bind_group(0, bg_c)
        p.dispatch_workgroups((n + 63) // 64)
        p.end()

        bg = device.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": raster_ubo, "offset": 0, "size": raster_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": points_buf, "offset": 0, "size": pts.nbytes}},
            {"binding": 2, "resource": {"buffer": vals_buf,   "offset": 0, "size": vals.nbytes}},
            {"binding": 3, "resource": {"buffer": bins_buf,   "offset": 0, "size": n_bins * 4}},
        ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipe); p.set_bind_group(0, bg)
        p.dispatch_workgroups((n + 63) // 64)
        p.end()
        cmap_mode = 0  # bins is f32-as-u32, no divide
    else:  # sum or mean: same kernel; mean divides at shade time
        pipe, bgl = get_pipeline(_POINT_VAL_SUM_SHADER, "cs_main", _BGL_VAL_RASTER_SUM)
        bg = device.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": raster_ubo, "offset": 0, "size": raster_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": points_buf, "offset": 0, "size": pts.nbytes}},
            {"binding": 2, "resource": {"buffer": vals_buf,   "offset": 0, "size": vals.nbytes}},
            {"binding": 3, "resource": {"buffer": bins_buf,   "offset": 0, "size": n_bins * 4}},
            {"binding": 4, "resource": {"buffer": counts_buf, "offset": 0, "size": n_bins * 4}},
        ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipe); p.set_bind_group(0, bg)
        p.dispatch_workgroups((n + 63) // 64)
        p.end()
        cmap_mode = 1 if op == "mean" else 0

    # Reductions on the (post-divide) grid.  Fused max+min if the caller
    # wants auto-vmin, max-only otherwise (single dispatch in both cases).
    auto_vmin = vmin is None
    auto_vmax = vmax is None

    reduce_uniforms = np.array([n_bins, cmap_mode, 0, 0], dtype=np.uint32)
    reduce_ubo = device.create_buffer_with_data(
        data=reduce_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    max_buf = device.create_buffer(
        size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    min_buf = device.create_buffer_with_data(
        data=np.array([0xFFFFFFFF], dtype=np.uint32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    if auto_vmin:
        pipe_r, bgl_r = get_pipeline(_REDUCE_VAL_MAX_MIN_SHADER, "cs_reduce_max_min",
                                     _BGL_VAL_REDUCE_MAX_MIN)
        bg_r = device.create_bind_group(layout=bgl_r, entries=[
            {"binding": 0, "resource": {"buffer": reduce_ubo, "offset": 0, "size": reduce_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": bins_buf,   "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": counts_buf, "offset": 0, "size": n_bins * 4}},
            {"binding": 3, "resource": {"buffer": max_buf,    "offset": 0, "size": 4}},
            {"binding": 4, "resource": {"buffer": min_buf,    "offset": 0, "size": 4}},
        ])
    else:
        pipe_r, bgl_r = get_pipeline(_REDUCE_VAL_MAX_SHADER, "cs_reduce", _BGL_VAL_REDUCE)
        bg_r = device.create_bind_group(layout=bgl_r, entries=[
            {"binding": 0, "resource": {"buffer": reduce_ubo, "offset": 0, "size": reduce_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": bins_buf,   "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": counts_buf, "offset": 0, "size": n_bins * 4}},
            {"binding": 3, "resource": {"buffer": max_buf,    "offset": 0, "size": 4}},
        ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe_r); p.set_bind_group(0, bg_r)
    p.dispatch_workgroups((n_bins + 255) // 256)
    p.end()

    # Colormap-apply.
    from .core import get_cmap_lut_buffer
    lut_buf = get_cmap_lut_buffer(cmap_name)

    auto_flags = (1 if auto_vmin else 0) | (2 if auto_vmax else 0)

    cmap_uniforms = np.zeros(12, dtype=np.uint32)
    cmap_uniforms[0] = plot_width
    cmap_uniforms[1] = plot_height
    cmap_uniforms[2] = cmap_mode
    cmap_uniforms[3] = transfer_mode
    cmap_uniforms.view(np.float32)[4] = float(vmin) if vmin is not None else 0.0
    cmap_uniforms.view(np.float32)[5] = float(vmax) if vmax is not None else 0.0
    cmap_uniforms.view(np.float32)[6] = float(alpha_min)
    cmap_uniforms.view(np.float32)[7] = float(alpha_max)
    cmap_uniforms[8] = auto_flags
    cmap_ubo = device.create_buffer_with_data(
        data=cmap_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)

    rgba_buf = device.create_buffer(
        size=n_bins * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    pipe_c, bgl_c = get_pipeline(_VAL_COLORMAP_SHADER, "cs_val_colormap", _BGL_VAL_CMAP)
    bg_c = device.create_bind_group(layout=bgl_c, entries=[
        {"binding": 0, "resource": {"buffer": cmap_ubo,    "offset": 0, "size": cmap_uniforms.nbytes}},
        {"binding": 1, "resource": {"buffer": bins_buf,    "offset": 0, "size": n_bins * 4}},
        {"binding": 2, "resource": {"buffer": counts_buf,  "offset": 0, "size": n_bins * 4}},
        {"binding": 3, "resource": {"buffer": max_buf,     "offset": 0, "size": 4}},
        {"binding": 4, "resource": {"buffer": lut_buf,     "offset": 0, "size": 256 * 4}},
        {"binding": 5, "resource": {"buffer": rgba_buf,    "offset": 0, "size": n_bins * 4}},
        {"binding": 6, "resource": {"buffer": min_buf,     "offset": 0, "size": 4}},
    ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe_c); p.set_bind_group(0, bg_c)
    p.dispatch_workgroups((plot_width + 7) // 8, (plot_height + 7) // 8)
    p.end()

    device.queue.submit([encoder.finish()])
    return readback_rgba(rgba_buf, plot_width, plot_height)
