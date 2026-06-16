#!/usr/bin/env python3
"""
Density Line Renderer V11 — Symmetric Dual-Pass + Scatter

Single RASTER_TEMPLATE generates both col-parallel and row-parallel shaders.
Uniform-x analytical O(1) path injected via {ANALYTICAL_PATH} template parameter.
Scatter mode via separate kernel. Same render() API for lines and scatter.

V9's 4 shaders → 1 template (2 passes) + 1 scatter kernel.
"""

import argparse
import math
import time
import numpy as np

import wgpu

_RENDERER_CACHE = {}
_LUT_CACHE = {}


def _transparent_lut(cmap, n_colors=2**8, opacity_limits=(0.0, 1.0), gamma=0.5):
    """Build an RGBA colormap whose alpha ramps with perceived lightness.

    Self-contained port of the alpha-ramped colormap used by the density
    renderer: alpha=0 at index 0 (data==0), then alpha = lightness**gamma
    clipped into ``opacity_limits`` for the rest. RGB is the colormap's own
    ramp, unchanged. Depends only on generic colour tooling (``cmap`` +
    ``scikit-image``), so it carries no spectroscopy/domain coupling — callers
    needing a different ramp pass their own ``lut_fn`` to ``rasterize_spectra``
    / ``DensityLineRenderer.render``.
    """
    from cmap import Colormap
    from skimage.color import rgb2hsv
    from ..array import rescale

    if isinstance(cmap, str):
        cmap = Colormap(cmap)
    vals = np.linspace(0, 1, n_colors)
    colors = np.array(cmap(vals))  # (n_colors, 4)
    rgb = colors[:, :3].reshape((1, n_colors, 3))
    L_vals = rgb2hsv(rgb)[0, :, -1]
    L_norm = np.clip(rescale(L_vals) ** gamma, opacity_limits[0], opacity_limits[1])
    colors[:, 3] = L_norm
    return colors

# Module-level GPU context (adapter, device, all compiled pipelines).
# Built once and shared across every ``DensityLineRenderer`` instance so that
# new canvas sizes don't pay the ~700 ms shader-compile cost on first use —
# only their per-canvas buffers need to be allocated.
_GPU_CONTEXT = None
# Workgroup-shared atomic<u32> arrays in the col/row raster shaders are sized
# to MAX_DIM at compile time; runtime ``MINOR_DIM`` (=uniforms.height for col,
# uniforms.width for row) loops only touch the in-use prefix. 4096 hits the
# 32 KB Metal workgroup-memory cap (2 arrays × 4096 × 4 B = 32 KB).
_RASTER_MAX_DIM = 4096


# ============================================================================
# Unified raster template — generates both col and row pass shaders.
#
# Density is deposited per-COLUMN for all segments with |dx| >= 1 (col pass),
# and per-ROW only for near-vertical segments |dx| < 1 (row pass).
# This ensures uniform column sums for eq_hist correctness.
#
# Template parameters:
#   {SHARED_SIZE}      — shared memory array size
#   {MAJOR_DIM}        — dispatch dimension ("uniforms.width" or "uniforms.height")
#   {MINOR_DIM}        — shared memory dimension ("uniforms.height" or "uniforms.width")
#   {MAJ1},{MAJ2}      — major-axis endpoints ("x1"/"x2" or "y1"/"y2")
#   {MIN1},{MIN2}      — minor-axis endpoints ("y1"/"y2" or "x1"/"x2")
#   {DEN_COND}         — density deposit condition
#   {DEN_RANGE_INIT}   — per-line init for dedup state (row pass only)
#   {DEN_FAR_SKIP}     — dedup reset on far skip (row pass only)
#   {DEN_DEPOSIT}      — density deposit code block
#   {LOOP_SETUP}       — per-line segment loop init (binary search or 3-seg skip)
#   {LOOP_BREAK}       — early termination for monotonic lines (col pass only)
#   {ANALYTICAL_PATH}  — uniform-x O(1) fast path (col pass only, empty for row)
#   {PX_F}             — pixel x expression ("major_pos" or "minor_f")
#   {PY_F}             — pixel y expression ("minor_f" or "major_pos")
#   {GLOBAL_IDX}       — global buffer index expression
# ============================================================================

RASTER_TEMPLATE = """
struct Uniforms {{
    width: u32,
    height: u32,
    num_lines: u32,
    num_points: u32,
    half_width: f32,
    coverage_scale: f32,
    x_first: f32,
    x_step: f32,
}}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read> y_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> extent: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> segment_valid: array<u32>;

var<workgroup> s_den: array<atomic<u32>, {SHARED_SIZE}>;
var<workgroup> s_ext: array<atomic<u32>, {SHARED_SIZE}>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {{
    let major_idx = wid.x;
    if (major_idx >= {MAJOR_DIM}) {{ return; }}

    let tid = lid.x;
    let half_width = uniforms.half_width;
    let scale = uniforms.coverage_scale;
    let major_pos = f32(major_idx) + 0.5;
    let n_fill = max(1, i32(2.0 * half_width));
    let MINOR_DIM = {MINOR_DIM};

    // Init shared memory
    for (var r = tid; r < MINOR_DIM; r += 256u) {{
        atomicStore(&s_den[r], 0u);
        atomicStore(&s_ext[r], 0u);
    }}
    workgroupBarrier();

    let reach = half_width + 1.5;

    for (var line = tid; line < uniforms.num_lines; line += 256u) {{
        let base = line * uniforms.num_points;
        let num_segs = uniforms.num_points - 1u;
        {DEN_RANGE_INIT}

        // Uniform-x analytical fast path (col pass only, triggered by x_step > 0)
        {ANALYTICAL_PATH}

        // Binary search / full scan path
        var start_seg = 0u;
        {LOOP_SETUP}

        for (var seg = start_seg; seg < num_segs; seg++) {{
            if (segment_valid[seg] == 0u) {{ continue; }}
            let x1 = x_values[base + seg];
            let x2 = x_values[base + seg + 1u];
            let y1 = y_values[base + seg];
            let y2 = y_values[base + seg + 1u];
            let dx = x2 - x1;
            let dy = y2 - y1;

            let maj1 = {MAJ1}; let maj2 = {MAJ2};
            let min1 = {MIN1}; let min2 = {MIN2};
            let d_major = maj2 - maj1;
            let d_minor = min2 - min1;

            let maj_lo = min(maj1, maj2);
            let maj_hi = max(maj1, maj2);

            // Skip segments far from this major position
            if (major_pos < maj_lo - reach || major_pos > maj_hi + reach) {{
                {DEN_FAR_SKIP}
                {LOOP_BREAK}
                continue;
            }}

            let seg_len_sq = dx * dx + dy * dy;
            let seg_len = sqrt(seg_len_sq);
            if (seg_len < 0.01) {{ continue; }}

            // Density: col pass deposits for |dx| >= 1, row pass for |dx| < 1.
            let den_eligible = {DEN_COND};

            // === DENSITY ===
            {DEN_DEPOSIT}

            // === EXTENT ===
            if (den_eligible) {{
                let sec_theta = seg_len / max(abs(d_major), 0.01);
                let ext_r = i32(ceil((half_width + 0.5) * sec_theta + 0.5));
                let t_center = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
                let minor_center = min1 + t_center * d_minor;

                for (var dmi = -ext_r; dmi <= ext_r; dmi++) {{
                    let mi = i32(minor_center) + dmi;
                    if (mi < 0 || mi >= i32(MINOR_DIM)) {{ continue; }}
                    let minor_f = f32(mi) + 0.5;
                    let px_f = {PX_F};
                    let py_f = {PY_F};
                    let ax = px_f - x1;
                    let ay = py_f - y1;
                    let t_proj = clamp((ax * dx + ay * dy) / seg_len_sq, 0.0, 1.0);
                    let qx = ax - t_proj * dx;
                    let qy = ay - t_proj * dy;
                    let d_dist = sqrt(qx * qx + qy * qy);
                    let ext_cov = clamp(half_width + 0.5 - d_dist, 0.0, 1.0);
                    let ext_int = u32(ext_cov * scale);
                    if (ext_int > 0u) {{
                        atomicMax(&s_ext[u32(mi)], ext_int);
                    }}
                }}
            }}
        }}
    }}

    workgroupBarrier();

    // Flush shared memory to global buffers
    for (var mi = tid; mi < MINOR_DIM; mi += 256u) {{
        let d = atomicLoad(&s_den[mi]);
        if (d > 0u) {{
            let gi = {GLOBAL_IDX};
            atomicAdd(&density[gi], d);
        }}
        let e = atomicLoad(&s_ext[mi]);
        if (e > 0u) {{
            let gi = {GLOBAL_IDX};
            atomicMax(&extent[gi], e);
        }}
    }}
}}
"""

# ============================================================================
# Scatter kernel — 1 thread per point, deposits density + extent circle
# ============================================================================

SCATTER_SHADER = """
struct Uniforms {
    width: u32,
    height: u32,
    num_points: u32,
    _pad: u32,
    half_width: f32,
    coverage_scale: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read> y_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> extent: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> segment_valid: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.num_points) { return; }

    let px = x_values[i];
    let py = y_values[i];
    let half_width = uniforms.half_width;
    let scale = uniforms.coverage_scale;

    let col = i32(floor(px));
    let row = i32(floor(py));
    if (col < 0 || col >= i32(uniforms.width) || row < 0 || row >= i32(uniforms.height)) { return; }

    // Density: filled circle of radius half_width (the dot body)
    // Extent: antialiased edge at half_width + 0.5
    let ext_r = i32(ceil(half_width + 0.5));
    for (var dy = -ext_r; dy <= ext_r; dy++) {
        let r = row + dy;
        if (r < 0 || r >= i32(uniforms.height)) { continue; }
        for (var dx = -ext_r; dx <= ext_r; dx++) {
            let c = col + dx;
            if (c < 0 || c >= i32(uniforms.width)) { continue; }
            let d = sqrt(f32(dx * dx + dy * dy));
            let idx = u32(r) * uniforms.width + u32(c);

            // Density inside dot body
            if (d <= half_width) {
                atomicAdd(&density[idx], u32(scale));
            }

            // Extent for antialiased edge
            let ext_cov = clamp(half_width + 0.5 - d, 0.0, 1.0);
            let ext_int = u32(ext_cov * scale);
            if (ext_int > 0u) {
                atomicMax(&extent[idx], ext_int);
            }
        }
    }
}
"""

VERTEX_FIX_SHADER = """
struct Params {
    width: u32,
    height: u32,
    num_lines: u32,
    num_points: u32,
    half_width: f32,
    scale: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read> y_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> segment_valid: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.num_points < 3u) { return; }
    let num_segs = params.num_points - 1u;
    let verts_per_line = params.num_points - 2u;
    let total = params.num_lines * verts_per_line;
    let idx = gid.x;
    if (idx >= total) { return; }

    let line = idx / verts_per_line;
    let j = (idx % verts_per_line) + 1u;
    if (segment_valid[j - 1u] == 0u || segment_valid[j] == 0u) { return; }

    let base = line * params.num_points;
    let x0 = x_values[base + j - 1u];
    let y0 = y_values[base + j - 1u];
    let x1 = x_values[base + j];
    let y1 = y_values[base + j];
    let x2 = x_values[base + j + 1u];
    let y2 = y_values[base + j + 1u];

    let dx_a = x1 - x0; let dy_a = y1 - y0;
    let dx_b = x2 - x1; let dy_b = y2 - y1;

    // Pass eligibility (matching raster DEN_COND):
    //   Col pass deposits when abs(dx) >= abs(dy)
    //   Row pass deposits when abs(dy) > abs(dx)
    let a_col = abs(dx_a) >= abs(dy_a);
    let b_col = abs(dx_b) >= abs(dy_b);

    // Only fix cross-pass junctions (adjacent segments in different passes)
    if (a_col == b_col) { return; }

    let hw = params.half_width;
    let hw_sq = hw * hw;
    let scan_r = i32(ceil(hw + 1.0));
    let vx = i32(x1);
    let vy = i32(y1);

    let len_sq_a = max(dx_a * dx_a + dy_a * dy_a, 0.0001);
    let len_sq_b = max(dx_b * dx_b + dy_b * dy_b, 0.0001);
    let is_last_seg = (j == num_segs - 1u);

    for (var dpy: i32 = -scan_r; dpy <= scan_r; dpy = dpy + 1) {
        let py = vy + dpy;
        if (py < 0 || py >= i32(params.height)) { continue; }
        for (var dpx: i32 = -scan_r; dpx <= scan_r; dpx = dpx + 1) {
            let px = vx + dpx;
            if (px < 0 || px >= i32(params.width)) { continue; }

            let pxf = f32(px) + 0.5;
            let pyf = f32(py) + 0.5;

            // Distance to segment A (seg j-1)
            let ax_a = pxf - x0; let ay_a = pyf - y0;
            let t_a = clamp((ax_a * dx_a + ay_a * dy_a) / len_sq_a, 0.0, 1.0);
            let qx_a = ax_a - t_a * dx_a;
            let qy_a = ay_a - t_a * dy_a;
            let d_sq_a = qx_a * qx_a + qy_a * qy_a;
            if (d_sq_a > hw_sq) { continue; }

            // Distance to segment B (seg j)
            let ax_b = pxf - x1; let ay_b = pyf - y1;
            let t_b = clamp((ax_b * dx_b + ay_b * dy_b) / len_sq_b, 0.0, 1.0);
            let qx_b = ax_b - t_b * dx_b;
            let qy_b = ay_b - t_b * dy_b;
            let d_sq_b = qx_b * qx_b + qy_b * qy_b;
            if (d_sq_b > hw_sq) { continue; }

            // Both within half_width — verify major-axis ranges (direction-dependent)
            var col_in_range = false;
            var row_in_range = false;
            if (a_col) {
                // Seg A is col-eligible (seg j-1, never the last segment)
                if (dx_a >= 0.0) {
                    col_in_range = pxf >= min(x0, x1) && pxf < max(x0, x1);
                } else {
                    col_in_range = pxf > min(x0, x1) && pxf <= max(x0, x1);
                }
                // Seg B is row-eligible (seg j)
                if (is_last_seg) {
                    row_in_range = pyf >= min(y1, y2) && pyf <= max(y1, y2);
                } else if (dy_b >= 0.0) {
                    row_in_range = pyf >= min(y1, y2) && pyf < max(y1, y2);
                } else {
                    row_in_range = pyf > min(y1, y2) && pyf <= max(y1, y2);
                }
            } else {
                // Seg A is row-eligible (seg j-1, never the last segment)
                if (dy_a >= 0.0) {
                    row_in_range = pyf >= min(y0, y1) && pyf < max(y0, y1);
                } else {
                    row_in_range = pyf > min(y0, y1) && pyf <= max(y0, y1);
                }
                // Seg B is col-eligible (seg j)
                if (is_last_seg) {
                    col_in_range = pxf >= min(x1, x2) && pxf <= max(x1, x2);
                } else if (dx_b >= 0.0) {
                    col_in_range = pxf >= min(x1, x2) && pxf < max(x1, x2);
                } else {
                    col_in_range = pxf > min(x1, x2) && pxf <= max(x1, x2);
                }
            }

            if (col_in_range && row_in_range) {
                let didx = u32(py) * params.width + u32(px);
                var cur = atomicLoad(&density[didx]);
                loop {
                    let next = select(cur - params.scale, 0u, cur <= params.scale);
                    let res = atomicCompareExchangeWeak(&density[didx], cur, next);
                    if (res.exchanged || res.old_value == cur) { break; }
                    cur = res.old_value;
                }
            }
        }
    }
}
"""

# ============================================================================
# Utility shaders (identical to V9/V10)
# ============================================================================

HISTOGRAM_BLUR_SHADER = """
struct Params { num_bins: u32, radius: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_bins) { return; }
    var sum: f32 = 0.0;
    var total_w: f32 = 0.0;
    let r = i32(params.radius);
    for (var di = -r; di <= r; di++) {
        let ni = i32(idx) + di;
        if (ni >= 0 && ni < i32(params.num_bins)) {
            let w = f32(r + 1 - abs(di));
            sum += f32(input[u32(ni)]) * w;
            total_w += w;
        }
    }
    output[idx] = u32(sum / total_w);
}
"""

MINMAX_FIND_SHADER = """
struct Params { width: u32, height: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> density: array<u32>;
@group(0) @binding(2) var<storage, read_write> min_val: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> max_val: atomic<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.width * params.height) { return; }
    let d = density[idx];
    if (d > 0u) {
        atomicMin(&min_val, d);
        atomicMax(&max_val, d);
    }
}
"""

HISTOGRAM_SHADER = """
struct Params { width: u32, height: u32, num_bins: u32, _p0: u32, _p1: u32, _p2: u32, _p3: u32, _p4: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> density: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> min_val: array<u32>;
@group(0) @binding(4) var<storage, read> max_val: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.width * params.height) { return; }
    let d = density[idx];
    if (d == 0u) { return; }
    let min_density = min_val[0];
    let max_density = max_val[0];
    let range = f32(max(max_density - min_density, 1u));
    let normalized = f32(d - min_density) / range;
    let bin = u32(clamp(normalized, 0.0, 0.999) * f32(params.num_bins));
    atomicAdd(&histogram[bin], 1u);
}
"""

PREFIX_SUM_SHADER = """
struct Params { num_bins: u32, stride: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_bins) { return; }
    var value = input[idx];
    if (idx >= params.stride) { value += input[idx - params.stride]; }
    output[idx] = value;
}
"""

COLORMAP_SHADER = """
struct Params {
    width: u32, height: u32, num_bins: u32, _pad_d0: u32,
    transfer_fn: u32, _pad_d1: u32, _pad0: u32, _pad1: u32,
    core_floor_bits: u32, edge_floor_bits: u32, edge_alpha_scale_bits: u32, edge_alpha_gamma_bits: u32,
    core_alpha_bits: u32, color_scale_bits: u32, edge_alpha_bits: u32, _pad2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> original: array<u32>;
@group(0) @binding(2) var<storage, read> cdf: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read> extent: array<u32>;
@group(0) @binding(5) var<storage, read> lut: array<u32>;
@group(0) @binding(6) var<storage, read> min_val: array<u32>;
@group(0) @binding(7) var<storage, read> max_val: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.width * params.height) { return; }

    let d0 = original[idx];
    let e0 = extent[idx];
    if (e0 == 0u) {
        output[idx] = 0u;
        return;
    }

    let min_density = min_val[0];
    let max_density = max_val[0];
    let edge_alpha_den = bitcast<f32>(params.edge_alpha_bits);
    let core_extent_thresh = u32(round(max(edge_alpha_den, 1.0) * 0.75));
    let core = (d0 > 0u) || (e0 >= core_extent_thresh);
    let d = select(min_density, d0, core);
    let d_clamped = max(d, min_density);
    let range = f32(max(max_density - min_density, 1u));
    let norm_d = f32(d_clamped - min_density) / range;
    let core_floor = bitcast<f32>(params.core_floor_bits);
    let edge_floor = bitcast<f32>(params.edge_floor_bits);
    let edge_alpha_scale = bitcast<f32>(params.edge_alpha_scale_bits);
    let edge_alpha_gamma = bitcast<f32>(params.edge_alpha_gamma_bits);
    let core_alpha = bitcast<f32>(params.core_alpha_bits);
    let color_scale = bitcast<f32>(params.color_scale_bits);

    var v: f32;
    if (params.transfer_fn == 0u) { v = norm_d; }
    else if (params.transfer_fn == 1u) { v = log(1.0 + norm_d * 255.0) / log(256.0); }
    else if (params.transfer_fn == 2u) { v = pow(norm_d, 1.0 / 3.0); }
    else {
        let fb = clamp(norm_d, 0.0, 0.999) * f32(params.num_bins);
        let bin_lo = u32(fb);
        let bin_hi = min(bin_lo + 1u, params.num_bins - 1u);
        let frac = fb - f32(bin_lo);
        let cdf_total = cdf[params.num_bins - 1u];
        let v_lo = f32(cdf[bin_lo]) / f32(max(cdf_total, 1u));
        let v_hi = f32(cdf[bin_hi]) / f32(max(cdf_total, 1u));
        let v_raw = select(norm_d, mix(v_lo, v_hi, frac), cdf_total > 0u);
        // Rescale so CDF minimum maps to 0 (matching analytical TFs at endpoints)
        let cdf_floor = f32(cdf[0u]) / f32(max(cdf_total, 1u));
        v = (v_raw - cdf_floor) / max(1.0 - cdf_floor, 0.001);
    }
    v = clamp(v, 0.0, 1.0);
    // Keep hue/chroma consistent across core and AA edge pixels:
    // apply the same floor remap to all visible pixels, then control only alpha.
    v = core_floor + v * (1.0 - core_floor);
    v = max(v, edge_floor);
    v = pow(clamp(v, 0.0, 1.0), color_scale);
    let li = min(255u, u32(floor(v * 255.0)));
    let rgb_packed = lut[li] & 0x00FFFFFFu;

    var alpha: f32;
    if (core) {
        alpha = core_alpha;
    } else {
        let edge_raw = clamp((f32(e0) / max(edge_alpha_den, 1e-6)) * edge_alpha_scale, 0.0, 1.0);
        alpha = pow(edge_raw, edge_alpha_gamma);
    }
    alpha = clamp(alpha, 0.0, 1.0);
    let a8 = u32(round(alpha * 255.0));
    output[idx] = (a8 << 24u) | rgb_packed;
}
"""

DOWNSAMPLE_SHADER = """
struct Params {
    in_width: u32, in_height: u32, out_width: u32, out_height: u32,
    factor: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_rgba: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_rgba: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_n = params.out_width * params.out_height;
    if (idx >= out_n) { return; }
    let ox = idx % params.out_width;
    let oy = idx / params.out_width;

    var sum_a: u32 = 0u;
    var sum_ra: u32 = 0u;
    var sum_ga: u32 = 0u;
    var sum_ba: u32 = 0u;

    for (var dy: u32 = 0u; dy < params.factor; dy = dy + 1u) {
        let sy = oy * params.factor + dy;
        for (var dx: u32 = 0u; dx < params.factor; dx = dx + 1u) {
            let sx = ox * params.factor + dx;
            let p = input_rgba[sy * params.in_width + sx];
            let r = p & 255u;
            let g = (p >> 8u) & 255u;
            let b = (p >> 16u) & 255u;
            let a = (p >> 24u) & 255u;
            sum_a = sum_a + a;
            sum_ra = sum_ra + r * a;
            sum_ga = sum_ga + g * a;
            sum_ba = sum_ba + b * a;
        }
    }

    let block = max(params.factor * params.factor, 1u);
    let a_out = (sum_a + (block / 2u)) / block;
    var r_out: u32 = 0u;
    var g_out: u32 = 0u;
    var b_out: u32 = 0u;
    if (sum_a > 0u) {
        r_out = (sum_ra + (sum_a / 2u)) / sum_a;
        g_out = (sum_ga + (sum_a / 2u)) / sum_a;
        b_out = (sum_ba + (sum_a / 2u)) / sum_a;
    }
    output_rgba[idx] = (a_out << 24u) | (b_out << 16u) | (g_out << 8u) | r_out;
}
"""


def generate_point_data(num_sine_lines, num_cosine_lines, num_points, width, height, spread, seed=42):
    """Generate x and y values in pixel space."""
    t0 = time.time()
    np.random.seed(seed)
    freq = 0.08
    noise_scale = spread * 0.1

    x_vals = np.arange(num_points, dtype=np.float32)
    sin_vals = np.sin(x_vals * freq).astype(np.float32)
    cos_vals = np.cos(x_vals * freq).astype(np.float32)

    all_y = []
    min_y, max_y = np.inf, -np.inf

    for wave_type, num, vals in [("sine", num_sine_lines, sin_vals), ("cosine", num_cosine_lines, cos_vals)]:
        if num <= 0:
            continue
        amps = 1.0 + (np.random.random(num).astype(np.float32) - 0.5) * spread * 2
        noise = (np.random.random((num, num_points)).astype(np.float32) - 0.5) * noise_scale
        y_all = vals[np.newaxis, :] * amps[:, np.newaxis] + noise
        min_y = min(min_y, float(y_all.min()))
        max_y = max(max_y, float(y_all.max()))
        all_y.append(y_all)

    y_data = np.vstack(all_y) if all_y else np.zeros((0, num_points), dtype=np.float32)
    num_lines = num_sine_lines + num_cosine_lines

    pad_y = (max_y - min_y) * 0.1 if max_y > min_y else 0.1
    data_min = min_y - pad_y
    data_max = max_y + pad_y
    data_range = data_max - data_min
    y_pixels = ((y_data - data_min) / data_range * height).astype(np.float32)

    x_row = np.linspace(0, width, num_points, dtype=np.float32)
    x_pixels = np.broadcast_to(x_row[np.newaxis, :], (num_lines, num_points)).copy()

    print(f"Generated {num_lines:,} lines x {num_points} pts in {(time.time()-t0)*1000:.0f}ms")
    return x_pixels, y_pixels, num_lines


def infer_channel_intervals(num_points, channel_widths):
    """Infer contiguous [start, stop) intervals that tile ``num_points``."""
    if num_points <= 0:
        return []
    intervals = []
    start = 0
    for width in channel_widths:
        if start >= num_points:
            break
        stop = min(start + int(width), num_points)
        if stop > start:
            intervals.append((start, stop))
        start = stop
    if not intervals:
        return [(0, num_points)]
    if intervals[-1][1] < num_points:
        intervals.append((intervals[-1][1], num_points))
    return intervals


def _get_lut_rgba(cmap_name="magma", lut_fn=None):
    """RGBA (256,4) uint8 LUT for ``cmap_name``.

    ``lut_fn`` overrides the default alpha-ramped builder: a callable taking a
    colormap name and returning an (N,4) float (0..1) RGBA array. This is the
    decoupling seam — domain callers (e.g. spectroscopy code wanting a bespoke
    alpha ramp) inject their own LUT without ocdkit knowing about it.
    """
    key = (str(cmap_name), id(lut_fn))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        builder = lut_fn or _transparent_lut
        rgba = np.asarray(builder(str(cmap_name)), dtype=np.float64)
        lut = (rgba * 255.0).round().clip(0, 255).astype(np.uint8)
        _LUT_CACHE[key] = lut
    return lut


def _core_floor_from_lut_lightness(cmap_name="magma", target_lightness=0.2):
    lut = _get_lut_rgba(cmap_name)
    rgb = lut[:, :3].astype(np.float32) / 255.0
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    target = np.clip(float(target_lightness), 0.0, 1.0)
    idx = int(np.searchsorted(lum, target, side="left"))
    idx = int(np.clip(idx, 1, 255))
    return idx / 255.0


def render_spectra_wgpu(
    spectra,
    width,
    height,
    line_width=3.0,
    transfer_fn="eq_hist",
    color_scale=1.0,
    intervals=None,
    normalize=True,
    renderer=None,
    supersample=1,
    core_floor_idx=None,
    core_floor_lightness=0.2,
    edge_floor_idx=1,
    edge_alpha_scale=1.0,
    edge_alpha_gamma=1.0,
    core_alpha=1.0,
    cmap_name="magma",
    force_cpu_colormap=False,
):
    """Render spectra with optional interval splitting (no cross-interval line joins)."""
    t0 = time.perf_counter()
    y = np.asarray(spectra, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"`spectra` must be 2D (n_lines, n_points), got shape {y.shape}")
    if y.shape[1] < 2:
        raise ValueError("`spectra` must have at least 2 points per line.")

    if normalize:
        denom = np.nanmax(y, axis=1, keepdims=True)
        denom[~np.isfinite(denom) | (denom <= 0)] = 1.0
        y = y / denom
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    n_points_total = y.shape[1]
    supersample = max(1, int(supersample))
    render_width = int(width) * supersample
    render_height = int(height) * supersample

    # Map spectra domain to pixel domain to match datashader-style full-canvas usage.
    if n_points_total > 1:
        x_scale = float(render_width - 1) / float(n_points_total - 1)
        x_offset = 0.0
    else:
        x_scale = 0.0
        x_offset = (render_width - 1) / 2.0

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(y, dtype=np.float32)
    # Invert to image-row convention used by the WGPU kernel (row 0 at top).
    y_pixels_all = (1.0 - y_norm) * float(render_height - 1)

    if intervals is None:
        intervals = [(0, y.shape[1])]
    else:
        intervals = [(int(s), int(e)) for s, e in intervals]

    own_renderer = renderer is None
    if own_renderer:
        cache_key = (render_width, render_height)
        renderer = _RENDERER_CACHE.get(cache_key)
        if renderer is None:
            renderer = DensityLineRenderer(render_width, render_height)
            _RENDERER_CACHE[cache_key] = renderer
    elif renderer.width != render_width or renderer.height != render_height:
        raise ValueError(
            f"Provided renderer size ({renderer.width}, {renderer.height}) does not match "
            f"requested render size ({render_width}, {render_height})."
        )

    usable_intervals = [(start, stop) for start, stop in intervals if stop - start >= 2]

    # Build one segment-valid mask so all intervals render in one pass while
    # preserving gaps (cross-interval segments are disabled).
    seg_valid = np.zeros(max(n_points_total - 1, 0), dtype=np.uint32)
    for start, stop in usable_intervals:
        lo = max(0, start)
        hi = min(stop - 1, n_points_total - 1)
        if hi > lo:
            seg_valid[lo:hi] = 1

    x_indices = np.arange(n_points_total, dtype=np.float32)
    x_row = x_indices * x_scale + x_offset
    x_full = np.broadcast_to(x_row[None, :], y_pixels_all.shape).copy()

    if force_cpu_colormap:
        density, _extent = renderer.render(
            x_full,
            y_pixels_all,
            num_lines=y_pixels_all.shape[0],
            num_points=y_pixels_all.shape[1],
            half_width=(line_width * supersample) / 2.0,
            transfer_fn=transfer_fn,
            color_scale=color_scale,
            mode="lines",
            return_density=True,
            downsample_factor=1,
            segment_valid=seg_valid,
        )
        result = _colorize_density_extent(
            density=density,
            extent=_extent,
            coverage_scale=renderer.COVERAGE_SCALE,
            transfer_fn=transfer_fn,
            cmap_name=cmap_name,
            core_floor_idx=core_floor_idx,
            core_floor_lightness=core_floor_lightness,
            edge_floor_idx=edge_floor_idx,
            edge_alpha_scale=edge_alpha_scale,
            edge_alpha_gamma=edge_alpha_gamma,
            core_alpha=core_alpha,
            num_bins=renderer.NUM_BINS,
            color_scale=color_scale,
        )
        if supersample > 1:
            result = _downsample_rgba(result, supersample)
    else:
        result = renderer.render(
            x_full,
            y_pixels_all,
            num_lines=y_pixels_all.shape[0],
            num_points=y_pixels_all.shape[1],
            half_width=(line_width * supersample) / 2.0,
            transfer_fn=transfer_fn,
            color_scale=color_scale,
            mode="lines",
            return_density=False,
            cmap_name=cmap_name,
            core_floor_idx=core_floor_idx,
            core_floor_lightness=core_floor_lightness,
            edge_floor_idx=edge_floor_idx,
            edge_alpha_scale=edge_alpha_scale,
            edge_alpha_gamma=edge_alpha_gamma,
            core_alpha=core_alpha,
            downsample_factor=supersample,
            segment_valid=seg_valid,
        )

    return result, {
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        "num_intervals": len(usable_intervals),
        "intervals": usable_intervals,
        "owns_renderer": own_renderer,
        "supersample": supersample,
        "core_floor_idx": core_floor_idx,
        "core_floor_lightness": core_floor_lightness,
        "edge_floor_idx": edge_floor_idx,
        "edge_alpha_scale": edge_alpha_scale,
        "edge_alpha_gamma": edge_alpha_gamma,
        "core_alpha": core_alpha,
        "cmap_name": cmap_name,
        "force_cpu_colormap": force_cpu_colormap,
    }


def rasterize_spectra(
    data,
    *,
    plot_width,
    plot_height,
    x_range,
    y_range,
    line_width=3.0,
    x_coords=None,
    intervals=None,
    renderer=None,
):
    """WGPU equivalent of ``cvs.line(df, x=x_coords, y=y_cols, axis=1, agg=ds.count(),
    line_width=line_width)``.  Returns the per-pixel count grid as a float32
    ``(plot_height, plot_width)`` ndarray (matches ``ds.count()`` semantics
    and dtype expected by a downstream CPU normalize + colormap path).

    ``data`` is the (n_lines, n_points) array of y values; ``x_coords`` is the
    1D x array (defaults to ``np.linspace(x_range[0], x_range[1] - 1, n_points)``
    to match datashader's ``Canvas.line`` x-pixel convention).

    ``x_range``/``y_range`` are in *data* units; this function maps to pixel
    space the same way datashader's ``Canvas`` does.
    """
    y = np.asarray(data, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"`data` must be 2D (n_lines, n_points), got shape {y.shape}")
    n_lines, n_points = y.shape
    if n_points < 2:
        raise ValueError("`data` must have at least 2 points per line.")

    if x_coords is None:
        x_coords = np.linspace(float(x_range[0]), float(x_range[1]) - 1, n_points)
    x_coords = np.asarray(x_coords, dtype=np.float32)

    span_x = float(x_range[1]) - float(x_range[0])
    span_y = float(y_range[1]) - float(y_range[0])
    if span_x == 0 or span_y == 0:
        return np.zeros((plot_height, plot_width), dtype=np.float32)

    # Data → pixel.  Datashader's cvs.line returns the count grid with row 0 at
    # y_min (xarray-style ascending coord), so we deposit without an image-row
    # inversion — the WGSL kernel treats pixel row 0 as the y_min row in this
    # path, which yields output orientation identical to ``cvs.line``.
    x_px = (x_coords - float(x_range[0])) / span_x * float(plot_width)
    y_px_data = (y - float(y_range[0])) / span_y * float(plot_height)
    x_full = np.broadcast_to(x_px[None, :], y.shape).astype(np.float32, copy=True)
    y_pixels = y_px_data.astype(np.float32, copy=False)

    if intervals is None:
        seg_valid = None
    else:
        seg_valid = np.zeros(max(n_points - 1, 0), dtype=np.uint32)
        for start, stop in intervals:
            lo = max(0, int(start))
            hi = min(int(stop) - 1, n_points - 1)
            if hi > lo:
                seg_valid[lo:hi] = 1

    own_renderer = renderer is None
    if own_renderer:
        cache_key = (int(plot_width), int(plot_height))
        renderer = _RENDERER_CACHE.get(cache_key)
        if renderer is None:
            renderer = DensityLineRenderer(int(plot_width), int(plot_height))
            _RENDERER_CACHE[cache_key] = renderer
    elif renderer.width != plot_width or renderer.height != plot_height:
        raise ValueError(
            f"Provided renderer size ({renderer.width}, {renderer.height}) does not match "
            f"requested render size ({plot_width}, {plot_height})."
        )

    density, _extent = renderer.render(
        x_full,
        y_pixels,
        num_lines=n_lines,
        num_points=n_points,
        half_width=float(line_width) / 2.0,
        mode="lines",
        return_density=True,
        segment_valid=seg_valid,
    )
    # DensityLineRenderer counts via fixed-point COVERAGE_SCALE; rescale back
    # to "count" units so callers can treat the result like ds.count() output.
    return (density.astype(np.float32) / float(renderer.COVERAGE_SCALE))


def _transfer_normalize_density(density, transfer_fn="eq_hist"):
    """Normalize density to [0,1] similarly to datashader path before colormap lookup."""
    data = np.asarray(density, dtype=np.float32)
    valid = data > 0
    if not valid.any():
        return np.zeros_like(data, dtype=np.float32)

    if transfer_fn == "linear":
        vmin = data[valid].min()
        vmax = data[valid].max()
        out = np.zeros_like(data, dtype=np.float32)
        if np.isclose(vmax, vmin):
            out[valid] = 1.0
        else:
            out[valid] = (data[valid] - vmin) / (vmax - vmin)
        return out

    if transfer_fn == "log":
        vmin = data[valid].min()
        vmax = data[valid].max()
        denom = max(vmax - vmin, 1.0)
        n = (data - vmin) / denom
        out = np.zeros_like(data, dtype=np.float32)
        out[valid] = np.log1p(n[valid] * 255.0) / np.log(256.0)
        return np.clip(out, 0.0, 1.0)

    if transfer_fn == "cbrt":
        vmin = data[valid].min()
        vmax = data[valid].max()
        denom = max(vmax - vmin, 1.0)
        n = (data - vmin) / denom
        out = np.zeros_like(data, dtype=np.float32)
        out[valid] = np.cbrt(np.clip(n[valid], 0.0, 1.0))
        return np.clip(out, 0.0, 1.0)

    # eq_hist
    hist, bin_edges = np.histogram(data[valid], bins=256)
    cdf = hist.cumsum().astype(np.float32)
    if cdf[-1] <= 0:
        return np.zeros_like(data, dtype=np.float32)
    cdf /= cdf[-1]
    out = np.zeros_like(data, dtype=np.float32)
    out[valid] = np.interp(data[valid], bin_edges[:-1], cdf)
    pos = out > 0
    if pos.any():
        out[pos] -= out[pos].min()
        vmax_pos = out[pos].max()
        if vmax_pos > 0:
            out[pos] /= vmax_pos
    return np.clip(out, 0.0, 1.0)


def _colorize_density_extent(
    density,
    extent,
    coverage_scale,
    transfer_fn="eq_hist",
    cmap_name="magma",
    core_floor_idx=None,
    core_floor_lightness=0.2,
    edge_floor_idx=1,
    edge_alpha_scale=1.0,
    edge_alpha_gamma=1.0,
    core_alpha=1.0,
    num_bins=1024,
    blur_radius=5,
    color_scale=1.0,
):
    """
    CPU colormap that exactly replicates the GPU compute shader pipeline:
    histogram → triangular blur → prefix sum (CDF) → per-pixel colormap.

    This produces pixel-identical output to the WGSL compute shaders.
    """
    den = np.asarray(density, dtype=np.uint32).ravel()
    ext = np.asarray(extent, dtype=np.uint32).ravel()
    shape2d = np.asarray(density).shape[:2]
    n = den.size

    # --- Match MINMAX_FIND_SHADER: min/max of non-zero density ---
    nz = den > 0
    if nz.any():
        min_density = int(den[nz].min())
        max_density = int(den[nz].max())
    else:
        min_density = 0
        max_density = 0
    density_range = float(max(max_density - min_density, 1))

    # --- Match HISTOGRAM_SHADER: bin = floor(clamp(norm, 0, 0.999) * num_bins) ---
    if transfer_fn == "eq_hist":
        nz_vals = den[nz].astype(np.float64)
        norm_vals = (nz_vals - min_density) / density_range
        bins = np.floor(np.clip(norm_vals, 0.0, 0.999) * num_bins).astype(np.int32)
        hist = np.bincount(bins, minlength=num_bins).astype(np.float64)[:num_bins]

        # --- Match HISTOGRAM_BLUR_SHADER: triangular blur, radius=5 ---
        # Vectorized: convolve with triangular kernel, normalize per-position
        r = blur_radius
        kernel = np.arange(2 * r + 1, dtype=np.float64)
        kernel = (r + 1) - np.abs(kernel - r)
        # Pad hist so we can slice without boundary checks
        padded = np.zeros(num_bins + 2 * r, dtype=np.float64)
        padded[r:r + num_bins] = hist
        # Weighted sum and weight sum via 1D convolution
        weighted_sum = np.convolve(padded, kernel, mode='valid')  # length = num_bins
        # Weight normalization: convolve a mask of 1s to get per-position weight sum
        mask_padded = np.zeros(num_bins + 2 * r, dtype=np.float64)
        mask_padded[r:r + num_bins] = 1.0
        weight_sum = np.convolve(mask_padded, kernel, mode='valid')
        blurred = np.floor(weighted_sum / weight_sum).astype(np.uint64)

        # --- Match PREFIX_SUM_SHADER: Hillis-Steele inclusive scan ---
        cdf = blurred.copy()
        stride = 1
        while stride < num_bins:
            shifted = np.zeros_like(cdf)
            shifted[stride:] = cdf[:-stride]
            cdf = cdf + shifted
            stride *= 2

    # --- Match COLORMAP_SHADER ---
    edge_alpha_den = float(coverage_scale)
    core_extent_thresh = int(round(max(edge_alpha_den, 1.0) * 0.75))

    # Core mask: density > 0 OR extent >= threshold
    core = (den > 0) | (ext >= core_extent_thresh)

    # For non-core visible pixels, substitute min_density
    d = np.where(core, den, np.where(ext > 0, min_density, 0)).astype(np.float64)
    d_clamped = np.maximum(d, min_density)
    norm_d = (d_clamped - min_density) / density_range

    # Transfer function
    if transfer_fn == "linear":
        v = norm_d.copy()
    elif transfer_fn == "log":
        v = np.log1p(norm_d * 255.0) / np.log(256.0)
    elif transfer_fn == "cbrt":
        v = np.power(np.clip(norm_d, 0.0, 1.0), 1.0 / 3.0)
    else:  # eq_hist
        fb = np.clip(norm_d, 0.0, 0.999) * float(num_bins)
        bin_lo = np.floor(fb).astype(np.int32)
        bin_hi = np.minimum(bin_lo + 1, num_bins - 1)
        frac = fb - np.floor(fb)
        cdf_total = cdf[num_bins - 1]
        cdf_total_f = float(max(cdf_total, 1))
        v_lo = cdf[bin_lo].astype(np.float64) / cdf_total_f
        v_hi = cdf[bin_hi].astype(np.float64) / cdf_total_f
        v_raw = np.where(cdf_total > 0, v_lo + (v_hi - v_lo) * frac, norm_d)
        # Rescale so CDF minimum maps to 0
        cdf_floor = float(cdf[0]) / cdf_total_f
        v = (v_raw - cdf_floor) / max(1.0 - cdf_floor, 0.001)

    v = np.clip(v, 0.0, 1.0).astype(np.float32)

    # Core floor and edge floor
    if core_floor_idx is None:
        core_floor_f = _core_floor_from_lut_lightness(cmap_name, core_floor_lightness)
    else:
        core_floor_f = np.clip((float(core_floor_idx) + 0.5) / 255.0, 0.0, 1.0)
    edge_floor_f = np.clip((float(edge_floor_idx) + 0.5) / 255.0, 0.0, 1.0)

    visible = ext > 0
    v[visible] = core_floor_f + v[visible] * (1.0 - core_floor_f)
    v[visible] = np.maximum(v[visible], edge_floor_f)
    v = np.power(np.clip(v, 0.0, 1.0), float(color_scale))

    # LUT lookup — matches GPU: li = min(255, floor(v * 255))
    lut_rgba = _get_lut_rgba(cmap_name)  # (256, 4) uint8
    li = np.minimum(255, np.floor(v * 255.0).astype(np.int32))
    rgb = lut_rgba[li, :3]  # (n, 3) uint8

    # Alpha — matches GPU shader exactly
    edge_alpha_scale_f = float(max(edge_alpha_scale, 0.0))
    edge_alpha_gamma_f = float(max(edge_alpha_gamma, 1e-6))
    core_alpha_f = float(np.clip(core_alpha, 0.0, 1.0))

    edge_raw = np.clip(ext.astype(np.float32) / max(edge_alpha_den, 1e-6) * edge_alpha_scale_f, 0.0, 1.0)
    edge_a = np.power(edge_raw, edge_alpha_gamma_f)
    alpha = np.where(core, core_alpha_f, edge_a)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[ext == 0] = 0.0
    a8 = np.round(alpha * 255.0).astype(np.uint8)

    # Match GPU early-return: output[idx] = 0 for ext == 0
    bg = ext == 0
    rgba = np.zeros((n, 4), dtype=np.uint8)
    rgba[:, :3] = rgb
    rgba[:, 3] = a8
    rgba[bg] = 0
    return rgba.reshape(shape2d + (4,))


def _downsample_rgba(rgba, factor):
    """Downsample RGBA — exact match to GPU DOWNSAMPLE_SHADER integer math.

    GPU logic per output pixel:
        sum_a  = sum(a_i)
        sum_ra = sum(r_i * a_i)     (direct integer multiply, no /255)
        a_out  = (sum_a + block/2) / block
        r_out  = (sum_ra + sum_a/2) / sum_a   (if sum_a > 0, else 0)
    """
    if factor <= 1:
        return rgba
    h, w, _ = rgba.shape
    out_h = h // factor
    out_w = w // factor
    if out_h <= 0 or out_w <= 0:
        return rgba
    h2 = out_h * factor
    w2 = out_w * factor
    src = rgba[:h2, :w2].astype(np.uint32)

    # Reshape to (out_h, factor, out_w, factor, 4), then accumulate over block
    blocks = src.reshape(out_h, factor, out_w, factor, 4)
    r = blocks[..., 0]
    g = blocks[..., 1]
    b = blocks[..., 2]
    a = blocks[..., 3]

    sum_a  = a.sum(axis=(1, 3))                  # (out_h, out_w)
    sum_ra = (r * a).sum(axis=(1, 3))
    sum_ga = (g * a).sum(axis=(1, 3))
    sum_ba = (b * a).sum(axis=(1, 3))

    block = np.uint32(factor * factor)
    a_out = (sum_a + block // 2) // block

    nz = sum_a > 0
    half_a = sum_a // 2
    r_out = np.zeros_like(sum_a)
    g_out = np.zeros_like(sum_a)
    b_out = np.zeros_like(sum_a)
    r_out[nz] = (sum_ra[nz] + half_a[nz]) // sum_a[nz]
    g_out[nz] = (sum_ga[nz] + half_a[nz]) // sum_a[nz]
    b_out[nz] = (sum_ba[nz] + half_a[nz]) // sum_a[nz]

    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    out[..., 0] = np.minimum(r_out, 255).astype(np.uint8)
    out[..., 1] = np.minimum(g_out, 255).astype(np.uint8)
    out[..., 2] = np.minimum(b_out, 255).astype(np.uint8)
    out[..., 3] = np.minimum(a_out, 255).astype(np.uint8)
    return out


# Density-deposit + loop-setup templates extracted from
# ``DensityLineRenderer._make_raster_shader`` so the free
# ``_build_raster_shader`` function (used by ``_GpuContext``) and the legacy
# instance method (still consumed by ``gui/classification.py``) share the
# same source.
_COL_DEN_DEPOSIT_TEMPLATE = """\
            if (den_eligible) {{
                var in_range = major_pos >= maj_lo && major_pos < maj_hi;
                if (seg == num_segs - 1u) {{
                    in_range = major_pos >= maj_lo && major_pos <= maj_hi;
                }}
                if (in_range) {{
                    let sec_den = seg_len / max(abs(d_major), 0.01);
                    let fill_hw = half_width * sec_den;
                    let den_tc = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
                    let minor_at = min1 + den_tc * d_minor;
                    let mi_lo = i32(floor(minor_at - fill_hw + 0.5));
                    let n_fill_seg = max(1, i32(ceil(2.0 * fill_hw)));
                    let hw_sq = half_width * half_width;

                    // Precompute neighbor segment directions for miter clip
                    var ndx_v: f32 = 0.0; var ndy_v: f32 = 0.0; var nlen_sq_v: f32 = 0.0;
                    if (seg < num_segs - 1u) {{
                        if (segment_valid[seg + 1u] != 0u) {{
                            ndx_v = x_values[base + seg + 2u] - x2;
                            ndy_v = y_values[base + seg + 2u] - y2;
                            nlen_sq_v = ndx_v * ndx_v + ndy_v * ndy_v;
                        }}
                    }}
                    var pdx_v: f32 = 0.0; var pdy_v: f32 = 0.0; var plen_sq_v: f32 = 0.0;
                    if (seg > 0u) {{
                        if (segment_valid[seg - 1u] != 0u) {{
                            pdx_v = x1 - x_values[base + seg - 1u];
                            pdy_v = y1 - y_values[base + seg - 1u];
                            plen_sq_v = pdx_v * pdx_v + pdy_v * pdy_v;
                        }}
                    }}

                    for (var k = 0; k < n_fill_seg; k++) {{
                        let mi = mi_lo + k;
                        if (mi < 0 || mi >= i32(MINOR_DIM)) {{ continue; }}
                        let minor_f = f32(mi) + 0.5;
                        // Col pass pixel: px = major_pos, py = minor_f
                        let dax_c = major_pos - x1;
                        let day_c = minor_f - y1;
                        let cross_c = dax_c * dy - day_c * dx;
                        if (cross_c * cross_c <= hw_sq * seg_len_sq) {{
                            // Miter clip: reject bevel overshoot at direction changes
                            let dot_seg_c = dax_c * dx + day_c * dy;
                            var clip_c = false;
                            if (dot_seg_c > seg_len_sq) {{
                                if (nlen_sq_v > 0.01) {{
                                    let dpx_n = major_pos - x2;
                                    let dpy_n = minor_f - y2;
                                    let cross_n = dpx_n * ndy_v - dpy_n * ndx_v;
                                    if (cross_n * cross_n > hw_sq * nlen_sq_v) {{
                                        clip_c = true;
                                    }}
                                    // Miter limit: cap spike at acute junctions
                                    if (dpx_n * dpx_n + dpy_n * dpy_n > hw_sq * 9.0) {{
                                        clip_c = true;
                                    }}
                                }} else {{
                                    clip_c = true;  // flat endcap: no neighbor
                                }}
                            }}
                            if (dot_seg_c < 0.0) {{
                                if (plen_sq_v > 0.01) {{
                                    let cross_p = dax_c * pdy_v - day_c * pdx_v;
                                    if (cross_p * cross_p > hw_sq * plen_sq_v) {{
                                        clip_c = true;
                                    }}
                                    // Miter limit: cap spike at acute junctions
                                    if (dax_c * dax_c + day_c * day_c > hw_sq * 9.0) {{
                                        clip_c = true;
                                    }}
                                }} else {{
                                    clip_c = true;  // flat endcap: no neighbor
                                }}
                            }}
                            if (!clip_c) {{
                                atomicAdd(&s_den[u32(mi)], u32(scale));
                            }}
                        }}
                    }}
                }}
            }}"""

_ROW_DEN_DEPOSIT_TEMPLATE = """\
            if (den_eligible) {{
                let maj_lo_i = i32(floor(maj_lo));
                let maj_hi_i = i32(floor(maj_hi));
                if (i32(major_idx) >= maj_lo_i && i32(major_idx) <= maj_hi_i) {{
                    var skip_den = i32(major_idx) == last_den_major;
                    if (!skip_den && seg < num_segs - 1u) {{
                        let next_dx = x_values[base + seg + 2u] - x_values[base + seg + 1u];
                        if (abs(next_dx) >= 1.0 && i32(major_idx) == i32(floor(maj2))) {{
                            skip_den = true;
                        }}
                    }}
                    if (!skip_den) {{
                        let sec_den = seg_len / max(abs(d_major), 0.01);
                        let fill_hw = half_width * sec_den;
                        let den_tc = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
                        let minor_at = min1 + den_tc * d_minor;
                        let mi_lo_d = i32(floor(minor_at - fill_hw + 0.5));
                        let n_fill_seg = max(1, i32(ceil(2.0 * fill_hw)));
                        let hw_sq = half_width * half_width;

                        // Precompute neighbor segment directions for miter clip
                        var ndx_v: f32 = 0.0; var ndy_v: f32 = 0.0; var nlen_sq_v: f32 = 0.0;
                        if (seg < num_segs - 1u) {{
                            if (segment_valid[seg + 1u] != 0u) {{
                                ndx_v = x_values[base + seg + 2u] - x2;
                                ndy_v = y_values[base + seg + 2u] - y2;
                                nlen_sq_v = ndx_v * ndx_v + ndy_v * ndy_v;
                            }}
                        }}
                        var pdx_v: f32 = 0.0; var pdy_v: f32 = 0.0; var plen_sq_v: f32 = 0.0;
                        if (seg > 0u) {{
                            if (segment_valid[seg - 1u] != 0u) {{
                                pdx_v = x1 - x_values[base + seg - 1u];
                                pdy_v = y1 - y_values[base + seg - 1u];
                                plen_sq_v = pdx_v * pdx_v + pdy_v * pdy_v;
                            }}
                        }}

                        for (var k = 0; k < n_fill_seg; k++) {{
                            let mi = mi_lo_d + k;
                            if (mi < 0 || mi >= i32(MINOR_DIM)) {{ continue; }}
                            let minor_f = f32(mi) + 0.5;
                            // Row pass pixel: px = minor_f, py = major_pos
                            let dax_r = minor_f - x1;
                            let day_r = major_pos - y1;
                            let cross_r = dax_r * dy - day_r * dx;
                            if (cross_r * cross_r <= hw_sq * seg_len_sq) {{
                                // Miter clip
                                let dot_seg_r = dax_r * dx + day_r * dy;
                                var clip_r = false;
                                if (dot_seg_r > seg_len_sq) {{
                                    if (nlen_sq_v > 0.01) {{
                                        let dpx_n = minor_f - x2;
                                        let dpy_n = major_pos - y2;
                                        let cross_n = dpx_n * ndy_v - dpy_n * ndx_v;
                                        if (cross_n * cross_n > hw_sq * nlen_sq_v) {{
                                            clip_r = true;
                                        }}
                                        // Miter limit: cap spike at acute junctions
                                        if (dpx_n * dpx_n + dpy_n * dpy_n > hw_sq * 9.0) {{
                                            clip_r = true;
                                        }}
                                    }} else {{
                                        clip_r = true;  // flat endcap: no neighbor
                                    }}
                                }}
                                if (dot_seg_r < 0.0) {{
                                    if (plen_sq_v > 0.01) {{
                                        let cross_p = dax_r * pdy_v - day_r * pdx_v;
                                        if (cross_p * cross_p > hw_sq * plen_sq_v) {{
                                            clip_r = true;
                                        }}
                                        // Miter limit: cap spike at acute junctions
                                        if (dax_r * dax_r + day_r * day_r > hw_sq * 9.0) {{
                                            clip_r = true;
                                        }}
                                    }} else {{
                                        clip_r = true;  // flat endcap: no neighbor
                                    }}
                                }}
                                if (!clip_r) {{
                                    atomicAdd(&s_den[u32(mi)], u32(scale));
                                }}
                            }}
                        }}
                    }}
                }}
                last_den_major = i32(floor(maj2));
            }} else {{
                last_den_major = -999;
            }}"""

_COL_LOOP_SETUP_TEMPLATE = """\
        let maj_first = x_values[base];
            let maj_last = x_values[base + num_segs];
            var mono_dir: i32 = 0;
            if (maj_first < maj_last) {{
                mono_dir = 1;
                var bslo = 0u; var bshi = num_segs;
                while (bshi > bslo + 1u) {{
                    let mid = (bslo + bshi) >> 1u;
                    if (x_values[base + mid + 1u] < major_pos - reach) {{ bslo = mid; }}
                    else {{ bshi = mid; }}
                }}
                start_seg = bslo;
            }} else if (maj_first > maj_last) {{
                mono_dir = -1;
                var bslo = 0u; var bshi = num_segs;
                while (bshi > bslo + 1u) {{
                    let mid = (bslo + bshi) >> 1u;
                    if (x_values[base + mid] > major_pos + reach) {{ bslo = mid; }}
                    else {{ bshi = mid; }}
                }}
                start_seg = bslo;
            }}"""

_COL_LOOP_BREAK_TEMPLATE = """\
                if ((mono_dir == 1 && maj_lo > major_pos + reach) ||
                        (mono_dir == -1 && maj_hi < major_pos - reach)) {{ break; }}"""


def _build_raster_shader(is_col_pass, shared_size):
    """Generate raster shader for col or row pass from the unified template.

    Free function (was ``DensityLineRenderer._make_raster_shader``) so the
    output is independent of any particular renderer instance — let
    ``_GpuContext`` compile pipelines once with ``shared_size=_RASTER_MAX_DIM``
    and share them across renderers of any canvas size up to that max.
    """
    ANALYTICAL_PATH = """\
        if (uniforms.x_step > 0.0) {{
            let x_first = uniforms.x_first;
            let x_step = uniforms.x_step;
            let col_x = major_pos;

            let seg_f = (col_x - x_first) / x_step;
            let seg = clamp(u32(seg_f), 0u, num_segs - 1u);
            let ax1 = x_first + f32(seg) * x_step;
            let ax2 = ax1 + x_step;
            let t_col = clamp((col_x - ax1) / x_step, 0.0, 1.0);

            let ay1 = y_values[base + seg];
            let ay2 = y_values[base + seg + 1u];
            let ady = ay2 - ay1;
            let y_at_col = ay1 + t_col * ady;

            var in_range = col_x >= ax1 && col_x < ax2;
            if (seg == num_segs - 1u) {{
                in_range = col_x >= ax1 && col_x <= ax2;
            }}
            if (!in_range) {{ continue; }}

            let aseg_len_sq = x_step * x_step + ady * ady;
            let aseg_len = sqrt(aseg_len_sq);

            // Density: angle-corrected fill + perpendicular check + miter clip
            let a_fill_hw = half_width * aseg_len / x_step;
            let a_mi_lo = i32(floor(y_at_col - a_fill_hw + 0.5));
            let a_n_fill = max(1, i32(ceil(2.0 * a_fill_hw)));
            let hw_sq_a = half_width * half_width;

            // Precompute neighbor directions for miter clip
            // (all segments have dx = x_step in uniform-x).
            // Skip cross-block (invalid) neighbors so the miter doesn't pull
            // the brush toward a y-value across the gap.
            var a_ndy_v: f32 = 0.0; var a_nlen_sq_v: f32 = 0.0;
            if (seg < num_segs - 1u && segment_valid[seg + 1u] != 0u) {{
                a_ndy_v = y_values[base + seg + 2u] - ay2;
                a_nlen_sq_v = x_step * x_step + a_ndy_v * a_ndy_v;
            }}
            var a_pdy_v: f32 = 0.0; var a_plen_sq_v: f32 = 0.0;
            if (seg > 0u && segment_valid[seg - 1u] != 0u) {{
                a_pdy_v = ay1 - y_values[base + seg - 1u];
                a_plen_sq_v = x_step * x_step + a_pdy_v * a_pdy_v;
            }}

            for (var k = 0; k < a_n_fill; k++) {{
                let apy = a_mi_lo + k;
                if (apy < 0 || apy >= i32(MINOR_DIM)) {{ continue; }}
                let apy_f = f32(apy) + 0.5;
                // Perpendicular check: pixel=(col_x, apy_f), seg dir=(x_step, ady)
                let dax_a = col_x - ax1;
                let day_a = apy_f - ay1;
                let cross_a = dax_a * ady - day_a * x_step;
                if (cross_a * cross_a <= hw_sq_a * aseg_len_sq) {{
                    // Miter clip at endpoints
                    let dot_seg_a = dax_a * x_step + day_a * ady;
                    var clip_a = false;
                    if (dot_seg_a > aseg_len_sq) {{
                        if (a_nlen_sq_v > 0.01) {{
                            let dpx_n = col_x - ax2;
                            let dpy_n = apy_f - ay2;
                            let cross_n = dpx_n * a_ndy_v - dpy_n * x_step;
                            if (cross_n * cross_n > hw_sq_a * a_nlen_sq_v) {{
                                clip_a = true;
                            }}
                            // Miter limit: cap spike at acute junctions
                            if (dpx_n * dpx_n + dpy_n * dpy_n > hw_sq_a * 9.0) {{
                                clip_a = true;
                            }}
                        }} else {{
                            clip_a = true;  // flat endcap: no neighbor
                        }}
                    }}
                    if (dot_seg_a < 0.0) {{
                        if (a_plen_sq_v > 0.01) {{
                            let cross_p = dax_a * a_pdy_v - day_a * x_step;
                            if (cross_p * cross_p > hw_sq_a * a_plen_sq_v) {{
                                clip_a = true;
                            }}
                            // Miter limit: cap spike at acute junctions
                            if (dax_a * dax_a + day_a * day_a > hw_sq_a * 9.0) {{
                                clip_a = true;
                            }}
                        }} else {{
                            clip_a = true;  // flat endcap: no neighbor
                        }}
                    }}
                    if (!clip_a) {{
                        atomicAdd(&s_den[u32(apy)], u32(scale));
                    }}
                }}
            }}

            // Extent: clamped-t + slope-adaptive ext_r
            let asec_theta = aseg_len / x_step;
            let aext_r = i32(ceil((half_width + 0.5) * asec_theta + 0.5));
            let aax = col_x - ax1;
            for (var dpy = -aext_r; dpy <= aext_r; dpy++) {{
                let epy = i32(y_at_col) + dpy;
                if (epy < 0 || epy >= i32(MINOR_DIM)) {{ continue; }}
                let epy_f = f32(epy) + 0.5;
                let eay = epy_f - ay1;
                let et_proj = clamp((aax * x_step + eay * ady) / aseg_len_sq, 0.0, 1.0);
                let eqx = aax - et_proj * x_step;
                let eqy = eay - et_proj * ady;
                let ed_dist = sqrt(eqx * eqx + eqy * eqy);
                let eext_cov = clamp(half_width + 0.5 - ed_dist, 0.0, 1.0);
                let eext_int = u32(eext_cov * scale);
                if (eext_int > 0u) {{
                    atomicMax(&s_ext[u32(epy)], eext_int);
                }}
            }}

            // Adjacent segment extent: check NEXT segment near right vertex
            // Skip when seg+1 is invalid — it's a cross-block boundary; pulling
            // its endpoint y here draws a bridge across the gap.
            let vtx_reach = half_width + 1.0;
            if (seg < num_segs - 1u && segment_valid[seg + 1u] != 0u) {{
                if (ax2 - col_x < vtx_reach) {{
                    let ny3 = y_values[base + seg + 2u];
                    let nndy = ny3 - ay2;
                    let nnlen_sq = x_step * x_step + nndy * nndy;
                    let nnsec = sqrt(nnlen_sq) / x_step;
                    let nner = i32(ceil((half_width + 0.5) * nnsec + 0.5));
                    let nnbx = col_x - ax2;
                    for (var dpy = -nner; dpy <= nner; dpy++) {{
                        let npy = i32(ay2) + dpy;
                        if (npy < 0 || npy >= i32(MINOR_DIM)) {{ continue; }}
                        let npy_f = f32(npy) + 0.5;
                        let nnby = npy_f - ay2;
                        let nnt = clamp((nnbx * x_step + nnby * nndy) / nnlen_sq, 0.0, 1.0);
                        let nnqx = nnbx - nnt * x_step;
                        let nnqy = nnby - nnt * nndy;
                        let nnd = sqrt(nnqx * nnqx + nnqy * nnqy);
                        let nne = clamp(half_width + 0.5 - nnd, 0.0, 1.0);
                        let nnei = u32(nne * scale);
                        if (nnei > 0u) {{
                            atomicMax(&s_ext[u32(npy)], nnei);
                        }}
                    }}
                }}
            }}

            // Adjacent segment extent: check PREVIOUS segment near left vertex
            // Skip when seg-1 is invalid (cross-block boundary).
            if (seg > 0u && segment_valid[seg - 1u] != 0u) {{
                if (col_x - ax1 < vtx_reach) {{
                    let py0 = y_values[base + seg - 1u];
                    let ppdy = ay1 - py0;
                    let pplen_sq = x_step * x_step + ppdy * ppdy;
                    let ppsec = sqrt(pplen_sq) / x_step;
                    let pper = i32(ceil((half_width + 0.5) * ppsec + 0.5));
                    let ppx1 = ax1 - x_step;
                    let ppbx = col_x - ppx1;
                    for (var dpy = -pper; dpy <= pper; dpy++) {{
                        let ppy = i32(ay1) + dpy;
                        if (ppy < 0 || ppy >= i32(MINOR_DIM)) {{ continue; }}
                        let ppy_f = f32(ppy) + 0.5;
                        let ppby = ppy_f - py0;
                        let ppt = clamp((ppbx * x_step + ppby * ppdy) / pplen_sq, 0.0, 1.0);
                        let ppqx = ppbx - ppt * x_step;
                        let ppqy = ppby - ppt * ppdy;
                        let ppd = sqrt(ppqx * ppqx + ppqy * ppqy);
                        let ppe = clamp(half_width + 0.5 - ppd, 0.0, 1.0);
                        let ppei = u32(ppe * scale);
                        if (ppei > 0u) {{
                            atomicMax(&s_ext[u32(ppy)], ppei);
                        }}
                    }}
                }}
            }}

            continue;
        }}"""

    COL_DEN_DEPOSIT = _COL_DEN_DEPOSIT_TEMPLATE
    ROW_DEN_DEPOSIT = _ROW_DEN_DEPOSIT_TEMPLATE
    COL_LOOP_SETUP = _COL_LOOP_SETUP_TEMPLATE
    COL_LOOP_BREAK = _COL_LOOP_BREAK_TEMPLATE

    if is_col_pass:
        params = {
            'SHARED_SIZE': shared_size,
            'MAJOR_DIM': 'uniforms.width',
            'MINOR_DIM': 'uniforms.height',
            'MAJ1': 'x1', 'MAJ2': 'x2',
            'MIN1': 'y1', 'MIN2': 'y2',
            'DEN_COND': 'abs(d_major) >= 1.0',
            'DEN_RANGE_INIT': '',
            'DEN_FAR_SKIP': '',
            'DEN_DEPOSIT': COL_DEN_DEPOSIT,
            'LOOP_SETUP': COL_LOOP_SETUP,
            'LOOP_BREAK': COL_LOOP_BREAK,
            'ANALYTICAL_PATH': ANALYTICAL_PATH,
            'PX_F': 'major_pos',
            'PY_F': 'minor_f',
            'GLOBAL_IDX': 'u32(mi) * uniforms.width + major_idx',
        }
    else:
        params = {
            'SHARED_SIZE': shared_size,
            'MAJOR_DIM': 'uniforms.height',
            'MINOR_DIM': 'uniforms.width',
            'MAJ1': 'y1', 'MAJ2': 'y2',
            'MIN1': 'x1', 'MIN2': 'x2',
            'DEN_COND': 'abs(d_minor) < 1.0 && abs(d_major) >= 1.0',
            'DEN_RANGE_INIT': 'var last_den_major: i32 = -999;',
            'DEN_FAR_SKIP': 'last_den_major = -999;',
            'DEN_DEPOSIT': ROW_DEN_DEPOSIT,
            'LOOP_SETUP': '',
            'LOOP_BREAK': '',
            'ANALYTICAL_PATH': '',
            'PX_F': 'minor_f',
            'PY_F': 'major_pos',
            'GLOBAL_IDX': 'major_idx * uniforms.width + u32(mi)',
        }
    return RASTER_TEMPLATE.format(**params)


class _GpuContext:
    """Process-wide GPU state: shared adapter, device, layouts, and
    size-independent pipelines (scatter, color, hist, prefix, minmax,
    downsample, vertex_fix) — built once on first renderer creation.

    The two raster pipelines (col-pass + row-pass) compile WGSL with
    workgroup-shared atomic arrays sized at compile time to the canvas's
    minor dimension (height for col, width for row). Sized too large and
    GPU occupancy drops; sized exactly per canvas keeps occupancy optimal
    but pays compile time per new size. We compile per-canvas-size and
    cache the resulting pipeline pair in ``self._raster_cache`` keyed by
    ``(height, width)``, so multiple ``DensityLineRenderer`` instances at
    the same size share the cost.
    """

    def __init__(self):
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        print(f"GPU: {self.adapter.summary}")
        self.device = self.adapter.request_device_sync()
        self._raster_cache = {}
        self._build_size_independent_pipelines()

    def _build_size_independent_pipelines(self):
        d = self.device
        # 6-binding raster layout: uniforms, x_values, y_values, density,
        # extent, segment_valid. Layout is shared by col+row pipelines.
        self.raster_bind_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        ])
        self._raster_layout = d.create_pipeline_layout(
            bind_group_layouts=[self.raster_bind_layout])

        scatter_shader = d.create_shader_module(code=SCATTER_SHADER)
        self.scatter_pipeline = d.create_compute_pipeline(
            layout=self._raster_layout,
            compute={"module": scatter_shader, "entry_point": "main"})

        def make(code, entries):
            s = d.create_shader_module(code=code)
            l = d.create_bind_group_layout(entries=entries)
            p = d.create_compute_pipeline(
                layout=d.create_pipeline_layout(bind_group_layouts=[l]),
                compute={"module": s, "entry_point": "main"})
            return p, l

        se = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}}]
        self.hist_blur_pipeline, self.hist_blur_bind_layout = make(HISTOGRAM_BLUR_SHADER, se)
        mm = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
              {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}}]
        self.minmax_pipeline, self.minmax_bind_layout = make(MINMAX_FIND_SHADER, mm)
        he = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
              {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}]
        self.hist_pipeline, self.hist_bind_layout = make(HISTOGRAM_SHADER, he)
        self.prefix_pipeline, self.prefix_bind_layout = make(PREFIX_SUM_SHADER, se)
        ce = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
              {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 7, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}]
        self.color_pipeline, self.color_bind_layout = make(COLORMAP_SHADER, ce)
        ve = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
              {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}]
        self.vertex_fix_pipeline, self.vertex_fix_bind_layout = make(VERTEX_FIX_SHADER, ve)
        de = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}}]
        self.downsample_pipeline, self.downsample_bind_layout = make(DOWNSAMPLE_SHADER, de)

    def get_raster_pipelines(self, width, height):
        """Return ``(col_pipeline, row_pipeline)`` for the given canvas size,
        compiling and caching on first request. Workgroup-shared array size
        is set to the minor dimension (height for col, width for row) for
        optimal GPU occupancy.
        """
        key = (int(width), int(height))
        cached = self._raster_cache.get(key)
        if cached is not None:
            return cached
        d = self.device
        col_code = _build_raster_shader(is_col_pass=True, shared_size=int(height))
        col_shader = d.create_shader_module(code=col_code)
        col_pipeline = d.create_compute_pipeline(
            layout=self._raster_layout,
            compute={"module": col_shader, "entry_point": "main"})
        row_code = _build_raster_shader(is_col_pass=False, shared_size=int(width))
        row_shader = d.create_shader_module(code=row_code)
        row_pipeline = d.create_compute_pipeline(
            layout=self._raster_layout,
            compute={"module": row_shader, "entry_point": "main"})
        self._raster_cache[key] = (col_pipeline, row_pipeline)
        return col_pipeline, row_pipeline


def _get_gpu_context():
    """Return the singleton ``_GpuContext`` (lazy first-call init)."""
    global _GPU_CONTEXT
    if _GPU_CONTEXT is None:
        _GPU_CONTEXT = _GpuContext()
    return _GPU_CONTEXT


class DensityLineRenderer:
    NUM_BINS = 1024
    COVERAGE_SCALE = 256.0

    def __init__(self, width, height):
        self.width = width
        self.height = height
        assert max(width, height) <= _RASTER_MAX_DIM, (
            f"max(width={width}, height={height}) exceeds raster shader "
            f"shared-memory cap ({_RASTER_MAX_DIM})"
        )
        # Pull adapter, device, and all size-independent pipelines from the
        # process-wide singleton — pays the ~600 ms shader-compile cost
        # exactly once. The two raster pipelines (col/row) ARE size-dependent
        # (workgroup shared memory sized to canvas) and are compiled+cached
        # per (width, height) on the same context.
        ctx = _get_gpu_context()
        self.adapter = ctx.adapter
        self.device = ctx.device
        self._ctx = ctx
        self._create_resources()

    def _create_resources(self):
        self.num_pixels = self.width * self.height

        self.density_buffer = self.device.create_buffer(
            size=self.num_pixels * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self.extent_buffer = self.device.create_buffer(
            size=self.num_pixels * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self.output_buffer = self.device.create_buffer(
            size=self.num_pixels * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        self.staging_buffer = self.device.create_buffer(
            size=self.num_pixels * 4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        self.staging_density_buffer = self.device.create_buffer(
            size=self.num_pixels * 4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        self.staging_extent_buffer = self.device.create_buffer(
            size=self.num_pixels * 4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        self.min_buffer = self.device.create_buffer(
            size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self.max_buffer = self.device.create_buffer(
            size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self.hist_buffer1 = self.device.create_buffer(
            size=self.NUM_BINS * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self.hist_buffer2 = self.device.create_buffer(
            size=self.NUM_BINS * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

        # Over-allocate the x/y value buffers, but stay within the adapter's
        # limits. Metal reports a 4 GB maxBufferSize so 100 M floats (400 MB)
        # is fine there, but Vulkan adapters (e.g. NVIDIA TITAN RTX) cap
        # maxBufferSize at 256 MB and maxStorageBufferBindingSize at 128 MB,
        # so a fixed 400 MB allocation fails validation. Clamp to whichever
        # limit is smaller. The shader only ever binds the used range
        # (num_lines*num_points) so the smaller buffer is not a problem in
        # practice for spectral line counts.
        try:
            limits = self.device.limits
            max_buf = int(limits.get("max-buffer-size", 1 << 28))
            max_bind = int(limits.get("max-storage-buffer-binding-size", max_buf))
            buf_cap = min(max_buf, max_bind)
        except Exception:
            buf_cap = 1 << 27  # 128 MB fallback
        MAX_VALUES = min(100_000_000, buf_cap // 4)
        xy_bytes = MAX_VALUES * 4
        self.x_buffer = self.device.create_buffer(
            size=xy_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.y_buffer = self.device.create_buffer(
            size=xy_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

        self.raster_uniforms_buffer = self.device.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.color_lut_buffer = self.device.create_buffer(
            size=256 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.color_uniforms_buffer = self.device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.vertex_fix_uniforms_buffer = self.device.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.segment_valid_dummy_buffer = self.device.create_buffer_with_data(
            data=np.array([1], dtype=np.uint32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )

        self._zeros_bytes = np.zeros(self.num_pixels, dtype=np.uint32).tobytes()
        self._downsample_cache = {}
        # Mirror shared-context resources onto self for backwards compat with
        # code that reads ``renderer.col_pipeline`` etc. directly. Size-dep
        # raster pipelines are looked up (and compiled-on-miss) by canvas size.
        ctx = self._ctx
        self.raster_bind_layout    = ctx.raster_bind_layout
        self.col_pipeline, self.row_pipeline = ctx.get_raster_pipelines(self.width, self.height)
        self.scatter_pipeline      = ctx.scatter_pipeline
        self.hist_blur_pipeline    = ctx.hist_blur_pipeline
        self.hist_blur_bind_layout = ctx.hist_blur_bind_layout
        self.minmax_pipeline       = ctx.minmax_pipeline
        self.minmax_bind_layout    = ctx.minmax_bind_layout
        self.hist_pipeline         = ctx.hist_pipeline
        self.hist_bind_layout      = ctx.hist_bind_layout
        self.prefix_pipeline       = ctx.prefix_pipeline
        self.prefix_bind_layout    = ctx.prefix_bind_layout
        self.color_pipeline        = ctx.color_pipeline
        self.color_bind_layout     = ctx.color_bind_layout
        self.vertex_fix_pipeline   = ctx.vertex_fix_pipeline
        self.vertex_fix_bind_layout = ctx.vertex_fix_bind_layout
        self.downsample_pipeline   = ctx.downsample_pipeline
        self.downsample_bind_layout = ctx.downsample_bind_layout

    def _make_raster_shader(self, is_col_pass):
        """Generate raster shader for col or row pass from the unified template."""

        # --- Analytical fast path (col pass only) ---
        # Injected at top of line loop. When x_step > 0, computes segment
        # analytically from x_first/x_step (O(1), no x_values reads).
        # Includes adjacent-segment extent checks for exact V9 match.
        ANALYTICAL_PATH = """\
        if (uniforms.x_step > 0.0) {{
            let x_first = uniforms.x_first;
            let x_step = uniforms.x_step;
            let col_x = major_pos;

            let seg_f = (col_x - x_first) / x_step;
            let seg = clamp(u32(seg_f), 0u, num_segs - 1u);
            let ax1 = x_first + f32(seg) * x_step;
            let ax2 = ax1 + x_step;
            let t_col = clamp((col_x - ax1) / x_step, 0.0, 1.0);

            let ay1 = y_values[base + seg];
            let ay2 = y_values[base + seg + 1u];
            let ady = ay2 - ay1;
            let y_at_col = ay1 + t_col * ady;

            var in_range = col_x >= ax1 && col_x < ax2;
            if (seg == num_segs - 1u) {{
                in_range = col_x >= ax1 && col_x <= ax2;
            }}
            if (!in_range) {{ continue; }}

            let aseg_len_sq = x_step * x_step + ady * ady;
            let aseg_len = sqrt(aseg_len_sq);

            // Density: angle-corrected fill + perpendicular check + miter clip
            let a_fill_hw = half_width * aseg_len / x_step;
            let a_mi_lo = i32(floor(y_at_col - a_fill_hw + 0.5));
            let a_n_fill = max(1, i32(ceil(2.0 * a_fill_hw)));
            let hw_sq_a = half_width * half_width;

            // Precompute neighbor directions for miter clip
            // (all segments have dx = x_step in uniform-x).
            // Skip cross-block (invalid) neighbors so the miter doesn't pull
            // the brush toward a y-value across the gap.
            var a_ndy_v: f32 = 0.0; var a_nlen_sq_v: f32 = 0.0;
            if (seg < num_segs - 1u && segment_valid[seg + 1u] != 0u) {{
                a_ndy_v = y_values[base + seg + 2u] - ay2;
                a_nlen_sq_v = x_step * x_step + a_ndy_v * a_ndy_v;
            }}
            var a_pdy_v: f32 = 0.0; var a_plen_sq_v: f32 = 0.0;
            if (seg > 0u && segment_valid[seg - 1u] != 0u) {{
                a_pdy_v = ay1 - y_values[base + seg - 1u];
                a_plen_sq_v = x_step * x_step + a_pdy_v * a_pdy_v;
            }}

            for (var k = 0; k < a_n_fill; k++) {{
                let apy = a_mi_lo + k;
                if (apy < 0 || apy >= i32(MINOR_DIM)) {{ continue; }}
                let apy_f = f32(apy) + 0.5;
                // Perpendicular check: pixel=(col_x, apy_f), seg dir=(x_step, ady)
                let dax_a = col_x - ax1;
                let day_a = apy_f - ay1;
                let cross_a = dax_a * ady - day_a * x_step;
                if (cross_a * cross_a <= hw_sq_a * aseg_len_sq) {{
                    // Miter clip at endpoints
                    let dot_seg_a = dax_a * x_step + day_a * ady;
                    var clip_a = false;
                    if (dot_seg_a > aseg_len_sq) {{
                        if (a_nlen_sq_v > 0.01) {{
                            let dpx_n = col_x - ax2;
                            let dpy_n = apy_f - ay2;
                            let cross_n = dpx_n * a_ndy_v - dpy_n * x_step;
                            if (cross_n * cross_n > hw_sq_a * a_nlen_sq_v) {{
                                clip_a = true;
                            }}
                            // Miter limit: cap spike at acute junctions
                            if (dpx_n * dpx_n + dpy_n * dpy_n > hw_sq_a * 9.0) {{
                                clip_a = true;
                            }}
                        }} else {{
                            clip_a = true;  // flat endcap: no neighbor
                        }}
                    }}
                    if (dot_seg_a < 0.0) {{
                        if (a_plen_sq_v > 0.01) {{
                            let cross_p = dax_a * a_pdy_v - day_a * x_step;
                            if (cross_p * cross_p > hw_sq_a * a_plen_sq_v) {{
                                clip_a = true;
                            }}
                            // Miter limit: cap spike at acute junctions
                            if (dax_a * dax_a + day_a * day_a > hw_sq_a * 9.0) {{
                                clip_a = true;
                            }}
                        }} else {{
                            clip_a = true;  // flat endcap: no neighbor
                        }}
                    }}
                    if (!clip_a) {{
                        atomicAdd(&s_den[u32(apy)], u32(scale));
                    }}
                }}
            }}

            // Extent: clamped-t + slope-adaptive ext_r
            let asec_theta = aseg_len / x_step;
            let aext_r = i32(ceil((half_width + 0.5) * asec_theta + 0.5));
            let aax = col_x - ax1;
            for (var dpy = -aext_r; dpy <= aext_r; dpy++) {{
                let epy = i32(y_at_col) + dpy;
                if (epy < 0 || epy >= i32(MINOR_DIM)) {{ continue; }}
                let epy_f = f32(epy) + 0.5;
                let eay = epy_f - ay1;
                let et_proj = clamp((aax * x_step + eay * ady) / aseg_len_sq, 0.0, 1.0);
                let eqx = aax - et_proj * x_step;
                let eqy = eay - et_proj * ady;
                let ed_dist = sqrt(eqx * eqx + eqy * eqy);
                let eext_cov = clamp(half_width + 0.5 - ed_dist, 0.0, 1.0);
                let eext_int = u32(eext_cov * scale);
                if (eext_int > 0u) {{
                    atomicMax(&s_ext[u32(epy)], eext_int);
                }}
            }}

            // Adjacent segment extent: check NEXT segment near right vertex
            // Skip when seg+1 is invalid — it's a cross-block boundary; pulling
            // its endpoint y here draws a bridge across the gap.
            let vtx_reach = half_width + 1.0;
            if (seg < num_segs - 1u && segment_valid[seg + 1u] != 0u) {{
                if (ax2 - col_x < vtx_reach) {{
                    let ny3 = y_values[base + seg + 2u];
                    let nndy = ny3 - ay2;
                    let nnlen_sq = x_step * x_step + nndy * nndy;
                    let nnsec = sqrt(nnlen_sq) / x_step;
                    let nner = i32(ceil((half_width + 0.5) * nnsec + 0.5));
                    let nnbx = col_x - ax2;
                    for (var dpy = -nner; dpy <= nner; dpy++) {{
                        let npy = i32(ay2) + dpy;
                        if (npy < 0 || npy >= i32(MINOR_DIM)) {{ continue; }}
                        let npy_f = f32(npy) + 0.5;
                        let nnby = npy_f - ay2;
                        let nnt = clamp((nnbx * x_step + nnby * nndy) / nnlen_sq, 0.0, 1.0);
                        let nnqx = nnbx - nnt * x_step;
                        let nnqy = nnby - nnt * nndy;
                        let nnd = sqrt(nnqx * nnqx + nnqy * nnqy);
                        let nne = clamp(half_width + 0.5 - nnd, 0.0, 1.0);
                        let nnei = u32(nne * scale);
                        if (nnei > 0u) {{
                            atomicMax(&s_ext[u32(npy)], nnei);
                        }}
                    }}
                }}
            }}

            // Adjacent segment extent: check PREVIOUS segment near left vertex
            // Skip when seg-1 is invalid (cross-block boundary).
            if (seg > 0u && segment_valid[seg - 1u] != 0u) {{
                if (col_x - ax1 < vtx_reach) {{
                    let py0 = y_values[base + seg - 1u];
                    let ppdy = ay1 - py0;
                    let pplen_sq = x_step * x_step + ppdy * ppdy;
                    let ppsec = sqrt(pplen_sq) / x_step;
                    let pper = i32(ceil((half_width + 0.5) * ppsec + 0.5));
                    let ppx1 = ax1 - x_step;
                    let ppbx = col_x - ppx1;
                    for (var dpy = -pper; dpy <= pper; dpy++) {{
                        let ppy = i32(ay1) + dpy;
                        if (ppy < 0 || ppy >= i32(MINOR_DIM)) {{ continue; }}
                        let ppy_f = f32(ppy) + 0.5;
                        let ppby = ppy_f - py0;
                        let ppt = clamp((ppbx * x_step + ppby * ppdy) / pplen_sq, 0.0, 1.0);
                        let ppqx = ppbx - ppt * x_step;
                        let ppqy = ppby - ppt * ppdy;
                        let ppd = sqrt(ppqx * ppqx + ppqy * ppqy);
                        let ppe = clamp(half_width + 0.5 - ppd, 0.0, 1.0);
                        let ppei = u32(ppe * scale);
                        if (ppei > 0u) {{
                            atomicMax(&s_ext[u32(ppy)], ppei);
                        }}
                    }}
                }}
            }}

            continue;
        }}"""

        # --- Col pass density: v11 half-open dedup + perp check + miter clip ---
        # Angle-corrected fill band with perpendicular distance ≤ hw.
        # Cross product for infinite-line SDF (no cutouts at straight boundaries).
        # Miter clip: pixels projecting past an endpoint are also checked against
        # the neighbor segment's infinite line — clips bevel at direction changes
        # while preserving full width on straight/parallel sections.
        COL_DEN_DEPOSIT = """\
            if (den_eligible) {{
                var in_range = major_pos >= maj_lo && major_pos < maj_hi;
                if (seg == num_segs - 1u) {{
                    in_range = major_pos >= maj_lo && major_pos <= maj_hi;
                }}
                if (in_range) {{
                    let sec_den = seg_len / max(abs(d_major), 0.01);
                    let fill_hw = half_width * sec_den;
                    let den_tc = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
                    let minor_at = min1 + den_tc * d_minor;
                    let mi_lo = i32(floor(minor_at - fill_hw + 0.5));
                    let n_fill_seg = max(1, i32(ceil(2.0 * fill_hw)));
                    let hw_sq = half_width * half_width;

                    // Precompute neighbor segment directions for miter clip
                    var ndx_v: f32 = 0.0; var ndy_v: f32 = 0.0; var nlen_sq_v: f32 = 0.0;
                    if (seg < num_segs - 1u) {{
                        if (segment_valid[seg + 1u] != 0u) {{
                            ndx_v = x_values[base + seg + 2u] - x2;
                            ndy_v = y_values[base + seg + 2u] - y2;
                            nlen_sq_v = ndx_v * ndx_v + ndy_v * ndy_v;
                        }}
                    }}
                    var pdx_v: f32 = 0.0; var pdy_v: f32 = 0.0; var plen_sq_v: f32 = 0.0;
                    if (seg > 0u) {{
                        if (segment_valid[seg - 1u] != 0u) {{
                            pdx_v = x1 - x_values[base + seg - 1u];
                            pdy_v = y1 - y_values[base + seg - 1u];
                            plen_sq_v = pdx_v * pdx_v + pdy_v * pdy_v;
                        }}
                    }}

                    for (var k = 0; k < n_fill_seg; k++) {{
                        let mi = mi_lo + k;
                        if (mi < 0 || mi >= i32(MINOR_DIM)) {{ continue; }}
                        let minor_f = f32(mi) + 0.5;
                        // Col pass pixel: px = major_pos, py = minor_f
                        let dax_c = major_pos - x1;
                        let day_c = minor_f - y1;
                        let cross_c = dax_c * dy - day_c * dx;
                        if (cross_c * cross_c <= hw_sq * seg_len_sq) {{
                            // Miter clip: reject bevel overshoot at direction changes
                            let dot_seg_c = dax_c * dx + day_c * dy;
                            var clip_c = false;
                            if (dot_seg_c > seg_len_sq) {{
                                if (nlen_sq_v > 0.01) {{
                                    let dpx_n = major_pos - x2;
                                    let dpy_n = minor_f - y2;
                                    let cross_n = dpx_n * ndy_v - dpy_n * ndx_v;
                                    if (cross_n * cross_n > hw_sq * nlen_sq_v) {{
                                        clip_c = true;
                                    }}
                                    // Miter limit: cap spike at acute junctions
                                    if (dpx_n * dpx_n + dpy_n * dpy_n > hw_sq * 9.0) {{
                                        clip_c = true;
                                    }}
                                }} else {{
                                    clip_c = true;  // flat endcap: no neighbor
                                }}
                            }}
                            if (dot_seg_c < 0.0) {{
                                if (plen_sq_v > 0.01) {{
                                    let cross_p = dax_c * pdy_v - day_c * pdx_v;
                                    if (cross_p * cross_p > hw_sq * plen_sq_v) {{
                                        clip_c = true;
                                    }}
                                    // Miter limit: cap spike at acute junctions
                                    if (dax_c * dax_c + day_c * day_c > hw_sq * 9.0) {{
                                        clip_c = true;
                                    }}
                                }} else {{
                                    clip_c = true;  // flat endcap: no neighbor
                                }}
                            }}
                            if (!clip_c) {{
                                atomicAdd(&s_den[u32(mi)], u32(scale));
                            }}
                        }}
                    }}
                }}
            }}"""

        # --- Row pass density: v11 last_den_major dedup + perp check + miter clip ---
        ROW_DEN_DEPOSIT = """\
            if (den_eligible) {{
                let maj_lo_i = i32(floor(maj_lo));
                let maj_hi_i = i32(floor(maj_hi));
                if (i32(major_idx) >= maj_lo_i && i32(major_idx) <= maj_hi_i) {{
                    var skip_den = i32(major_idx) == last_den_major;
                    if (!skip_den && seg < num_segs - 1u) {{
                        let next_dx = x_values[base + seg + 2u] - x_values[base + seg + 1u];
                        if (abs(next_dx) >= 1.0 && i32(major_idx) == i32(floor(maj2))) {{
                            skip_den = true;
                        }}
                    }}
                    if (!skip_den) {{
                        let sec_den = seg_len / max(abs(d_major), 0.01);
                        let fill_hw = half_width * sec_den;
                        let den_tc = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
                        let minor_at = min1 + den_tc * d_minor;
                        let mi_lo_d = i32(floor(minor_at - fill_hw + 0.5));
                        let n_fill_seg = max(1, i32(ceil(2.0 * fill_hw)));
                        let hw_sq = half_width * half_width;

                        // Precompute neighbor segment directions for miter clip
                        var ndx_v: f32 = 0.0; var ndy_v: f32 = 0.0; var nlen_sq_v: f32 = 0.0;
                        if (seg < num_segs - 1u) {{
                            if (segment_valid[seg + 1u] != 0u) {{
                                ndx_v = x_values[base + seg + 2u] - x2;
                                ndy_v = y_values[base + seg + 2u] - y2;
                                nlen_sq_v = ndx_v * ndx_v + ndy_v * ndy_v;
                            }}
                        }}
                        var pdx_v: f32 = 0.0; var pdy_v: f32 = 0.0; var plen_sq_v: f32 = 0.0;
                        if (seg > 0u) {{
                            if (segment_valid[seg - 1u] != 0u) {{
                                pdx_v = x1 - x_values[base + seg - 1u];
                                pdy_v = y1 - y_values[base + seg - 1u];
                                plen_sq_v = pdx_v * pdx_v + pdy_v * pdy_v;
                            }}
                        }}

                        for (var k = 0; k < n_fill_seg; k++) {{
                            let mi = mi_lo_d + k;
                            if (mi < 0 || mi >= i32(MINOR_DIM)) {{ continue; }}
                            let minor_f = f32(mi) + 0.5;
                            // Row pass pixel: px = minor_f, py = major_pos
                            let dax_r = minor_f - x1;
                            let day_r = major_pos - y1;
                            let cross_r = dax_r * dy - day_r * dx;
                            if (cross_r * cross_r <= hw_sq * seg_len_sq) {{
                                // Miter clip
                                let dot_seg_r = dax_r * dx + day_r * dy;
                                var clip_r = false;
                                if (dot_seg_r > seg_len_sq) {{
                                    if (nlen_sq_v > 0.01) {{
                                        let dpx_n = minor_f - x2;
                                        let dpy_n = major_pos - y2;
                                        let cross_n = dpx_n * ndy_v - dpy_n * ndx_v;
                                        if (cross_n * cross_n > hw_sq * nlen_sq_v) {{
                                            clip_r = true;
                                        }}
                                        // Miter limit: cap spike at acute junctions
                                        if (dpx_n * dpx_n + dpy_n * dpy_n > hw_sq * 9.0) {{
                                            clip_r = true;
                                        }}
                                    }} else {{
                                        clip_r = true;  // flat endcap: no neighbor
                                    }}
                                }}
                                if (dot_seg_r < 0.0) {{
                                    if (plen_sq_v > 0.01) {{
                                        let cross_p = dax_r * pdy_v - day_r * pdx_v;
                                        if (cross_p * cross_p > hw_sq * plen_sq_v) {{
                                            clip_r = true;
                                        }}
                                        // Miter limit: cap spike at acute junctions
                                        if (dax_r * dax_r + day_r * day_r > hw_sq * 9.0) {{
                                            clip_r = true;
                                        }}
                                    }} else {{
                                        clip_r = true;  // flat endcap: no neighbor
                                    }}
                                }}
                                if (!clip_r) {{
                                    atomicAdd(&s_den[u32(mi)], u32(scale));
                                }}
                            }}
                        }}
                    }}
                }}
                last_den_major = i32(floor(maj2));
            }} else {{
                last_den_major = -999;
            }}"""

        # --- Col pass binary search ---
        COL_LOOP_SETUP = """\
        let maj_first = x_values[base];
            let maj_last = x_values[base + num_segs];
            var mono_dir: i32 = 0;
            if (maj_first < maj_last) {{
                mono_dir = 1;
                var bslo = 0u; var bshi = num_segs;
                while (bshi > bslo + 1u) {{
                    let mid = (bslo + bshi) >> 1u;
                    if (x_values[base + mid + 1u] < major_pos - reach) {{ bslo = mid; }}
                    else {{ bshi = mid; }}
                }}
                start_seg = bslo;
            }} else if (maj_first > maj_last) {{
                mono_dir = -1;
                var bslo = 0u; var bshi = num_segs;
                while (bshi > bslo + 1u) {{
                    let mid = (bslo + bshi) >> 1u;
                    if (x_values[base + mid] > major_pos + reach) {{ bslo = mid; }}
                    else {{ bshi = mid; }}
                }}
                start_seg = bslo;
            }}"""

        COL_LOOP_BREAK = """\
                if ((mono_dir == 1 && maj_lo > major_pos + reach) ||
                        (mono_dir == -1 && maj_hi < major_pos - reach)) {{ break; }}"""

        # Row pass processes steep segments; no heuristic skip to avoid missing
        # localized steep transitions (e.g., square-wave edges).
        ROW_LOOP_SETUP = ""

        if is_col_pass:
            params = {
                'SHARED_SIZE': self.height,
                'MAJOR_DIM': 'uniforms.width',
                'MINOR_DIM': 'uniforms.height',
                'MAJ1': 'x1', 'MAJ2': 'x2',
                'MIN1': 'y1', 'MIN2': 'y2',
                # Col pass handles segments spanning >= 1 pixel in x.
                'DEN_COND': 'abs(d_major) >= 1.0',
                'DEN_RANGE_INIT': '',
                'DEN_FAR_SKIP': '',
                'DEN_DEPOSIT': COL_DEN_DEPOSIT,
                'LOOP_SETUP': COL_LOOP_SETUP,
                'LOOP_BREAK': COL_LOOP_BREAK,
                'ANALYTICAL_PATH': ANALYTICAL_PATH,
                'PX_F': 'major_pos',
                'PY_F': 'minor_f',
                'GLOBAL_IDX': 'u32(mi) * uniforms.width + major_idx',
            }
        else:
            params = {
                'SHARED_SIZE': self.width,
                'MAJOR_DIM': 'uniforms.height',
                'MINOR_DIM': 'uniforms.width',
                'MAJ1': 'y1', 'MAJ2': 'y2',
                'MIN1': 'x1', 'MIN2': 'x2',
                # Row pass handles sub-pixel-dx segments (steep / vertical).
                'DEN_COND': 'abs(d_minor) < 1.0 && abs(d_major) >= 1.0',
                'DEN_RANGE_INIT': 'var last_den_major: i32 = -999;',
                'DEN_FAR_SKIP': 'last_den_major = -999;',
                'DEN_DEPOSIT': ROW_DEN_DEPOSIT,
                'LOOP_SETUP': ROW_LOOP_SETUP,
                'LOOP_BREAK': '',
                'ANALYTICAL_PATH': '',  # Row pass never uses analytical
                'PX_F': 'minor_f',
                'PY_F': 'major_pos',
                'GLOBAL_IDX': 'major_idx * uniforms.width + u32(mi)',
            }
        return RASTER_TEMPLATE.format(**params)

    def _create_pipelines(self):
        # 6-binding layout: uniforms, x_values, y_values, density, extent, segment_valid
        self.raster_bind_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        ])
        raster_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.raster_bind_layout])

        # Col pass (dispatch per column, shared mem = height)
        col_code = self._make_raster_shader(is_col_pass=True)
        col_shader = self.device.create_shader_module(code=col_code)
        self.col_pipeline = self.device.create_compute_pipeline(
            layout=raster_layout,
            compute={"module": col_shader, "entry_point": "main"})

        # Row pass (dispatch per row, shared mem = width)
        row_code = self._make_raster_shader(is_col_pass=False)
        row_shader = self.device.create_shader_module(code=row_code)
        self.row_pipeline = self.device.create_compute_pipeline(
            layout=raster_layout,
            compute={"module": row_shader, "entry_point": "main"})

        # Scatter pipeline
        scatter_shader = self.device.create_shader_module(code=SCATTER_SHADER)
        self.scatter_pipeline = self.device.create_compute_pipeline(
            layout=raster_layout,
            compute={"module": scatter_shader, "entry_point": "main"})

        # Utility pipelines (identical to V9/V10)
        def make(code, entries):
            s = self.device.create_shader_module(code=code)
            l = self.device.create_bind_group_layout(entries=entries)
            p = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[l]),
                compute={"module": s, "entry_point": "main"})
            return p, l

        se = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}}]
        self.hist_blur_pipeline, self.hist_blur_bind_layout = make(HISTOGRAM_BLUR_SHADER, se)
        mm = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
              {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}}]
        self.minmax_pipeline, self.minmax_bind_layout = make(MINMAX_FIND_SHADER, mm)
        he = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
              {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}]
        self.hist_pipeline, self.hist_bind_layout = make(HISTOGRAM_SHADER, he)
        self.prefix_pipeline, self.prefix_bind_layout = make(PREFIX_SUM_SHADER, se)
        ce = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
              {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 7, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}]
        self.color_pipeline, self.color_bind_layout = make(COLORMAP_SHADER, ce)
        ve = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
              {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}]
        self.vertex_fix_pipeline, self.vertex_fix_bind_layout = make(VERTEX_FIX_SHADER, ve)
        de = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
              {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
              {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}}]
        self.downsample_pipeline, self.downsample_bind_layout = make(DOWNSAMPLE_SHADER, de)

    def render(self, x_pixels, y_pixels, num_lines, num_points, half_width,
               transfer_fn="eq_hist", color_scale=1.0, mode='lines', return_density=False,
               cmap_name="magma", core_floor_idx=None, core_floor_lightness=0.2,
               edge_floor_idx=1, edge_alpha_scale=1.0, edge_alpha_gamma=1.0, core_alpha=1.0,
               downsample_factor=1, segment_valid=None):
        """Render lines or scatter points.

        Args:
            mode: 'lines' (default) or 'scatter'
        """
        t_start = time.time()

        if mode == 'scatter':
            return self._render_scatter(x_pixels, y_pixels, num_lines * num_points,
                                        half_width, transfer_fn, color_scale, t_start, return_density=return_density,
                                        cmap_name=cmap_name, core_floor_idx=core_floor_idx,
                                        core_floor_lightness=core_floor_lightness, edge_floor_idx=edge_floor_idx,
                                        edge_alpha_scale=edge_alpha_scale, edge_alpha_gamma=edge_alpha_gamma,
                                        core_alpha=core_alpha, downsample_factor=downsample_factor)

        # --- Line rendering ---
        flat_y = np.ascontiguousarray(y_pixels.ravel())
        self.device.queue.write_buffer(self.y_buffer, 0, flat_y)

        n_segs = max(num_points - 1, 0)
        if segment_valid is not None:
            seg_arr = np.ascontiguousarray(segment_valid, dtype=np.uint32).ravel()
            if seg_arr.size < n_segs:
                raise ValueError(
                    f"segment_valid length {seg_arr.size} is smaller than required {n_segs}"
                )
            seg_arr = seg_arr[:n_segs]
        else:
            seg_arr = np.ones(n_segs, dtype=np.uint32)
        all_segments_valid = True
        seg_valid_buffer = self.segment_valid_dummy_buffer
        if seg_arr.size > 0:
            all_segments_valid = bool(np.all(seg_arr != 0))
            seg_valid_buffer = self.device.create_buffer_with_data(
                data=seg_arr.tobytes(),
                usage=wgpu.BufferUsage.STORAGE,
            )

        # Detect uniform-x: all lines share same uniformly-spaced x values
        x_2d = x_pixels.reshape(num_lines, num_points) if x_pixels.ndim == 1 else x_pixels
        x_row0 = x_2d[0]
        x_diffs = np.diff(x_row0)
        is_uniform_x = False
        if len(x_diffs) > 0 and np.all(x_diffs > 0) and np.allclose(x_diffs, x_diffs[0], rtol=1e-5):
            if num_lines == 1:
                is_uniform_x = True
            else:
                check_rows = [0, num_lines - 1, num_lines // 2]
                is_uniform_x = all(np.array_equal(x_2d[r], x_row0) for r in check_rows)
        if not all_segments_valid:
            # Analytical uniform-x path assumes all segments are connected.
            is_uniform_x = False

        if is_uniform_x:
            x_first = float(x_row0[0])
            x_step = float(x_diffs[0])
            # Uniform-x can still contain steep segments; row pass is needed in that case.
            max_abs_dy = float(np.max(np.abs(np.diff(y_pixels, axis=1)))) if num_points > 1 else 0.0
            needs_row_pass = max_abs_dy > abs(x_step)
            if needs_row_pass:
                flat_x = np.ascontiguousarray(x_pixels.ravel())
                self.device.queue.write_buffer(self.x_buffer, 0, flat_x)
                # Disable analytical col path when row pass is active to avoid
                # double-deposit at steep transitions.
                x_step = 0.0
                x_first = 0.0
        else:
            flat_x = np.ascontiguousarray(x_pixels.ravel())
            self.device.queue.write_buffer(self.x_buffer, 0, flat_x)
            x_first = 0.0
            x_step = 0.0
            needs_row_pass = True

        uniforms = np.zeros(8, dtype=np.float32)
        uniforms[0] = np.array([self.width], dtype=np.uint32).view(np.float32)[0]
        uniforms[1] = np.array([self.height], dtype=np.uint32).view(np.float32)[0]
        uniforms[2] = np.array([num_lines], dtype=np.uint32).view(np.float32)[0]
        uniforms[3] = np.array([num_points], dtype=np.uint32).view(np.float32)[0]
        uniforms[4] = half_width
        uniforms[5] = self.COVERAGE_SCALE
        uniforms[6] = x_first
        uniforms[7] = x_step
        self.device.queue.write_buffer(self.raster_uniforms_buffer, 0, uniforms.tobytes())

        self.device.queue.write_buffer(self.density_buffer, 0, self._zeros_bytes)
        self.device.queue.write_buffer(self.extent_buffer, 0, self._zeros_bytes)
        t_upload = time.time()

        encoder = self.device.create_command_encoder()

        # Bind only the USED range of the (over-allocated, 400 MB) x/y buffers — the
        # full binding exceeds the 128 MB max-storage-buffer-binding-size on adapters
        # like lavapipe (Metal's 4 GB limit hid this). Shader reads index up to
        # num_lines*num_points, so that's the range it needs.
        _xy_sz = max(4, int(num_lines) * int(num_points) * 4)
        raster_bind = self.device.create_bind_group(layout=self.raster_bind_layout, entries=[
            {"binding": 0, "resource": {"buffer": self.raster_uniforms_buffer}},
            {"binding": 1, "resource": {"buffer": self.x_buffer, "offset": 0, "size": _xy_sz}},
            {"binding": 2, "resource": {"buffer": self.y_buffer, "offset": 0, "size": _xy_sz}},
            {"binding": 3, "resource": {"buffer": self.density_buffer}},
            {"binding": 4, "resource": {"buffer": self.extent_buffer}},
            {"binding": 5, "resource": {"buffer": seg_valid_buffer}},
        ])

        # Col pass: 1 workgroup per column
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self.col_pipeline)
        cp.set_bind_group(0, raster_bind)
        cp.dispatch_workgroups(self.width)
        cp.end()

        # Row pass: required for steep segments (including uniform-x data).
        if needs_row_pass:
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self.row_pipeline)
            cp.set_bind_group(0, raster_bind)
            cp.dispatch_workgroups(self.height)
            cp.end()

        # Vertex fix: disabled — v11 deposit logic handles junctions cleanly.
        if False and needs_row_pass and num_points >= 3:
            vf_params = np.zeros(8, dtype=np.uint32)
            vf_params[0] = self.width
            vf_params[1] = self.height
            vf_params[2] = num_lines
            vf_params[3] = num_points
            vf_params[4] = np.array([half_width], dtype=np.float32).view(np.uint32)[0]
            vf_params[5] = int(self.COVERAGE_SCALE)
            vf_buf = self.device.create_buffer_with_data(
                data=vf_params.tobytes(),
                usage=wgpu.BufferUsage.UNIFORM,
            )
            _vf_xy_sz = max(4, int(num_lines) * int(num_points) * 4)
            vf_bind = self.device.create_bind_group(layout=self.vertex_fix_bind_layout, entries=[
                {"binding": 0, "resource": {"buffer": vf_buf}},
                {"binding": 1, "resource": {"buffer": self.x_buffer, "offset": 0, "size": _vf_xy_sz}},
                {"binding": 2, "resource": {"buffer": self.y_buffer, "offset": 0, "size": _vf_xy_sz}},
                {"binding": 3, "resource": {"buffer": self.density_buffer}},
                {"binding": 4, "resource": {"buffer": seg_valid_buffer}},
            ])
            total_verts = num_lines * (num_points - 2)
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self.vertex_fix_pipeline)
            cp.set_bind_group(0, vf_bind)
            cp.dispatch_workgroups((total_verts + 255) // 256)
            cp.end()

        if return_density:
            return self._read_density_extent(encoder, t_start, t_upload)
        return self._colormap(
            encoder, half_width, transfer_fn, color_scale, t_start, t_upload,
            cmap_name=cmap_name, core_floor_idx=core_floor_idx, core_floor_lightness=core_floor_lightness,
            edge_floor_idx=edge_floor_idx, edge_alpha_scale=edge_alpha_scale,
            edge_alpha_gamma=edge_alpha_gamma, core_alpha=core_alpha, downsample_factor=downsample_factor,
        )

    def _render_scatter(self, x_pixels, y_pixels, num_points, half_width,
                        transfer_fn, color_scale, t_start, return_density=False, cmap_name="magma",
                        core_floor_idx=None, core_floor_lightness=0.2, edge_floor_idx=1,
                        edge_alpha_scale=1.0, edge_alpha_gamma=1.0, core_alpha=1.0,
                        downsample_factor=1):
        """Render scatter points."""
        flat_x = np.ascontiguousarray(x_pixels.ravel()[:num_points].astype(np.float32))
        flat_y = np.ascontiguousarray(y_pixels.ravel()[:num_points].astype(np.float32))
        self.device.queue.write_buffer(self.x_buffer, 0, flat_x)
        self.device.queue.write_buffer(self.y_buffer, 0, flat_y)

        uniforms = np.zeros(8, dtype=np.float32)
        uniforms[0] = np.array([self.width], dtype=np.uint32).view(np.float32)[0]
        uniforms[1] = np.array([self.height], dtype=np.uint32).view(np.float32)[0]
        uniforms[2] = np.array([num_points], dtype=np.uint32).view(np.float32)[0]
        uniforms[3] = np.array([0], dtype=np.uint32).view(np.float32)[0]  # pad
        uniforms[4] = half_width
        uniforms[5] = self.COVERAGE_SCALE
        uniforms[6] = 0.0
        uniforms[7] = 0.0
        self.device.queue.write_buffer(self.raster_uniforms_buffer, 0, uniforms.tobytes())

        self.device.queue.write_buffer(self.density_buffer, 0, self._zeros_bytes)
        self.device.queue.write_buffer(self.extent_buffer, 0, self._zeros_bytes)
        t_upload = time.time()

        encoder = self.device.create_command_encoder()

        _sc_xy_sz = max(4, int(num_points) * 4)   # scatter writes num_points floats
        scatter_bind = self.device.create_bind_group(layout=self.raster_bind_layout, entries=[
            {"binding": 0, "resource": {"buffer": self.raster_uniforms_buffer}},
            {"binding": 1, "resource": {"buffer": self.x_buffer, "offset": 0, "size": _sc_xy_sz}},
            {"binding": 2, "resource": {"buffer": self.y_buffer, "offset": 0, "size": _sc_xy_sz}},
            {"binding": 3, "resource": {"buffer": self.density_buffer}},
            {"binding": 4, "resource": {"buffer": self.extent_buffer}},
            {"binding": 5, "resource": {"buffer": self.segment_valid_dummy_buffer}},
        ])

        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self.scatter_pipeline)
        cp.set_bind_group(0, scatter_bind)
        cp.dispatch_workgroups((num_points + 255) // 256)
        cp.end()

        if return_density:
            return self._read_density_extent(encoder, t_start, t_upload)
        return self._colormap(
            encoder, half_width, transfer_fn, color_scale, t_start, t_upload,
            cmap_name=cmap_name, core_floor_idx=core_floor_idx, core_floor_lightness=core_floor_lightness,
            edge_floor_idx=edge_floor_idx, edge_alpha_scale=edge_alpha_scale,
            edge_alpha_gamma=edge_alpha_gamma, core_alpha=core_alpha, downsample_factor=downsample_factor,
        )

    def _read_density_extent(self, encoder, t_start, t_upload):
        encoder.copy_buffer_to_buffer(self.density_buffer, 0, self.staging_density_buffer, 0, self.num_pixels * 4)
        encoder.copy_buffer_to_buffer(self.extent_buffer, 0, self.staging_extent_buffer, 0, self.num_pixels * 4)
        self.device.queue.submit([encoder.finish()])

        self.staging_density_buffer.map_sync(wgpu.MapMode.READ)
        density = np.frombuffer(self.staging_density_buffer.read_mapped(), dtype=np.uint32).reshape((self.height, self.width)).copy()
        self.staging_density_buffer.unmap()

        self.staging_extent_buffer.map_sync(wgpu.MapMode.READ)
        extent = np.frombuffer(self.staging_extent_buffer.read_mapped(), dtype=np.uint32).reshape((self.height, self.width)).copy()
        self.staging_extent_buffer.unmap()

        t_end = time.time()
        print(f"  Upload: {(t_upload-t_start)*1000:.0f}ms | Raster(readback): {(t_end-t_upload)*1000:.0f}ms | Total: {(t_end-t_start)*1000:.0f}ms")
        return density, extent

    def _colormap(self, encoder, half_width, transfer_fn, color_scale, t_start, t_upload,
                  cmap_name="magma", core_floor_idx=None, core_floor_lightness=0.2,
                  edge_floor_idx=1, edge_alpha_scale=1.0, edge_alpha_gamma=1.0, core_alpha=1.0,
                  downsample_factor=1):
        """Run minmax-find, histogram, CDF, colormap pipeline. Returns RGBA image.

        Single-submit: min/max stay on GPU (no readback). The histogram and
        colormap shaders read min/max directly from storage buffers.
        """
        # Find min/max density (results stay on GPU in min_buffer/max_buffer)
        self.device.queue.write_buffer(self.min_buffer, 0, np.array([0xFFFFFFFF], dtype=np.uint32).tobytes())
        self.device.queue.write_buffer(self.max_buffer, 0, np.zeros(1, dtype=np.uint32).tobytes())
        mpb = self.device.create_buffer_with_data(
            data=np.array([self.width, self.height], dtype=np.uint32).tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        mb = self.device.create_bind_group(layout=self.minmax_bind_layout, entries=[
            {"binding": 0, "resource": {"buffer": mpb}},
            {"binding": 1, "resource": {"buffer": self.density_buffer}},
            {"binding": 2, "resource": {"buffer": self.min_buffer}},
            {"binding": 3, "resource": {"buffer": self.max_buffer}}])
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self.minmax_pipeline)
        cp.set_bind_group(0, mb)
        cp.dispatch_workgroups((self.num_pixels + 255) // 256)
        cp.end()

        # Histogram — reads min/max from GPU storage buffers (no CPU readback)
        self.device.queue.write_buffer(self.hist_buffer1, 0, np.zeros(self.NUM_BINS, dtype=np.uint32).tobytes())
        hpb = self.device.create_buffer_with_data(
            data=np.array([self.width, self.height, self.NUM_BINS, 0, 0, 0, 0, 0], dtype=np.uint32).tobytes(),
            usage=wgpu.BufferUsage.UNIFORM)
        hb = self.device.create_bind_group(layout=self.hist_bind_layout, entries=[
            {"binding": 0, "resource": {"buffer": hpb}},
            {"binding": 1, "resource": {"buffer": self.density_buffer}},
            {"binding": 2, "resource": {"buffer": self.hist_buffer1}},
            {"binding": 3, "resource": {"buffer": self.min_buffer}},
            {"binding": 4, "resource": {"buffer": self.max_buffer}}])
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self.hist_pipeline)
        cp.set_bind_group(0, hb)
        cp.dispatch_workgroups((self.num_pixels + 255) // 256)
        cp.end()

        # Blur
        blur_params = np.array([self.NUM_BINS, 5, 0, 0], dtype=np.uint32)
        blur_pb = self.device.create_buffer_with_data(data=blur_params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        blur_bg = self.device.create_bind_group(layout=self.hist_blur_bind_layout, entries=[
            {"binding": 0, "resource": {"buffer": blur_pb}},
            {"binding": 1, "resource": {"buffer": self.hist_buffer1}},
            {"binding": 2, "resource": {"buffer": self.hist_buffer2}}])
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self.hist_blur_pipeline)
        cp.set_bind_group(0, blur_bg)
        cp.dispatch_workgroups((self.NUM_BINS + 255) // 256)
        cp.end()

        # Prefix sum (CDF)
        src, dst = self.hist_buffer2, self.hist_buffer1
        stride = 1
        while stride < self.NUM_BINS:
            ppb = self.device.create_buffer_with_data(
                data=np.array([self.NUM_BINS, stride, 0, 0], dtype=np.uint32).tobytes(),
                usage=wgpu.BufferUsage.UNIFORM)
            pbg = self.device.create_bind_group(layout=self.prefix_bind_layout, entries=[
                {"binding": 0, "resource": {"buffer": ppb}},
                {"binding": 1, "resource": {"buffer": src}},
                {"binding": 2, "resource": {"buffer": dst}}])
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self.prefix_pipeline)
            cp.set_bind_group(0, pbg)
            cp.dispatch_workgroups((self.NUM_BINS + 255) // 256)
            cp.end()
            src, dst = dst, src
            stride *= 2
        cdf_buffer = src

        # Colormap — reads min/max from GPU storage buffers (no CPU readback)
        tfn_map = {"linear": 0, "log": 1, "cbrt": 2, "eq_hist": 3}
        tfn_val = tfn_map.get(transfer_fn, 3)
        edge_alpha = self.COVERAGE_SCALE
        if core_floor_idx is None:
            core_floor = _core_floor_from_lut_lightness(cmap_name, core_floor_lightness)
        else:
            core_floor = np.clip((float(core_floor_idx) + 0.5) / 255.0, 0.0, 1.0)
        edge_floor = np.clip((float(edge_floor_idx) + 0.5) / 255.0, 0.0, 1.0)

        color_params = np.zeros(16, dtype=np.uint32)
        color_params[:8] = np.array(
            [self.width, self.height, self.NUM_BINS, 0, tfn_val, 0, 0, 0],
            dtype=np.uint32,
        )
        float_bits = np.frombuffer(
            np.array(
                [core_floor, edge_floor, edge_alpha_scale, edge_alpha_gamma, core_alpha, color_scale, edge_alpha, 0.0],
                dtype=np.float32,
            ).tobytes(),
            dtype=np.uint32,
        )
        color_params[8:] = float_bits

        lut_rgba = _get_lut_rgba(cmap_name)
        lut_packed = (
            (lut_rgba[:, 3].astype(np.uint32) << 24)
            | (lut_rgba[:, 2].astype(np.uint32) << 16)
            | (lut_rgba[:, 1].astype(np.uint32) << 8)
            | lut_rgba[:, 0].astype(np.uint32)
        )
        self.device.queue.write_buffer(self.color_lut_buffer, 0, lut_packed.tobytes())
        self.device.queue.write_buffer(self.color_uniforms_buffer, 0, color_params.tobytes())

        cbg = self.device.create_bind_group(layout=self.color_bind_layout, entries=[
            {"binding": 0, "resource": {"buffer": self.color_uniforms_buffer}},
            {"binding": 1, "resource": {"buffer": self.density_buffer}},
            {"binding": 2, "resource": {"buffer": cdf_buffer}},
            {"binding": 3, "resource": {"buffer": self.output_buffer}},
            {"binding": 4, "resource": {"buffer": self.extent_buffer}},
            {"binding": 5, "resource": {"buffer": self.color_lut_buffer}},
            {"binding": 6, "resource": {"buffer": self.min_buffer}},
            {"binding": 7, "resource": {"buffer": self.max_buffer}}])
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self.color_pipeline)
        cp.set_bind_group(0, cbg)
        cp.dispatch_workgroups((self.num_pixels + 255) // 256)
        cp.end()
        read_stage = self.staging_buffer
        out_w = self.width
        out_h = self.height
        out_bytes = self.num_pixels * 4
        if downsample_factor > 1:
            factor = int(downsample_factor)
            out_w = self.width // factor
            out_h = self.height // factor
            if out_w <= 0 or out_h <= 0:
                raise ValueError(f"Invalid downsample factor {downsample_factor} for size {(self.width, self.height)}")
            cache_key = (out_w, out_h)
            cached = self._downsample_cache.get(cache_key)
            if cached is None:
                out_buffer = self.device.create_buffer(
                    size=out_w * out_h * 4,
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                )
                out_stage = self.device.create_buffer(
                    size=out_w * out_h * 4,
                    usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
                )
                self._downsample_cache[cache_key] = (out_buffer, out_stage)
            else:
                out_buffer, out_stage = cached

            ds_params = np.array(
                [self.width, self.height, out_w, out_h, factor, 0, 0, 0],
                dtype=np.uint32,
            )
            ds_param_buffer = self.device.create_buffer_with_data(
                data=ds_params.tobytes(),
                usage=wgpu.BufferUsage.UNIFORM,
            )
            ds_bg = self.device.create_bind_group(layout=self.downsample_bind_layout, entries=[
                {"binding": 0, "resource": {"buffer": ds_param_buffer}},
                {"binding": 1, "resource": {"buffer": self.output_buffer}},
                {"binding": 2, "resource": {"buffer": out_buffer}},
            ])
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self.downsample_pipeline)
            cp.set_bind_group(0, ds_bg)
            cp.dispatch_workgroups(((out_w * out_h) + 255) // 256)
            cp.end()

            read_stage = out_stage
            out_bytes = out_w * out_h * 4
            encoder.copy_buffer_to_buffer(out_buffer, 0, out_stage, 0, out_bytes)
        else:
            encoder.copy_buffer_to_buffer(self.output_buffer, 0, self.staging_buffer, 0, self.num_pixels * 4)
        self.device.queue.submit([encoder.finish()])

        read_stage.map_sync(wgpu.MapMode.READ)
        rgba = np.frombuffer(read_stage.read_mapped(), dtype=np.uint8).reshape((out_h, out_w, 4)).copy()
        read_stage.unmap()

        t_end = time.time()
        print(f"  Upload: {(t_upload-t_start)*1000:.0f}ms | GPU+readback: {(t_end-t_upload)*1000:.0f}ms | Total: {(t_end-t_start)*1000:.0f}ms")
        return rgba


# ============================================================================
# Spectral → RGB compute shader (WebGPU / wgpu-py)
#
# Converts a registered stack (C × H × W, uint16) to sRGB (H × W × 4, u8)
# using CIE 1931 2° standard observer color matching functions.
# Supports per-excitation enable mask for interactive toggling.
# ============================================================================

SPECTRAL_RGB_SHADER = """
struct Params {
    width:  u32,
    height: u32,
    num_channels: u32,
    num_excitations: u32,
    exc_mask: u32,         // bit i → excitation i enabled
    max_val: f32,          // max intensity value for normalization
    gamma: f32,            // gamma correction exponent (1.0 = linear)
    rescale_lum: u32,      // 1 = rescale luminance via HSV V channel
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stack: array<u32>;          // packed u16 pairs
@group(0) @binding(2) var<storage, read> xyz_weights: array<vec4f>;  // (C,) xyz in .xyz
@group(0) @binding(3) var<storage, read> intervals: array<vec2u>;    // (num_exc,) start/stop
@group(0) @binding(4) var<storage, read_write> output: array<u32>;   // packed RGBA u8

// Read a uint16 from the packed u32 storage
fn read_u16(channel: u32, pixel: u32) -> f32 {
    let idx = channel * (params.width * params.height) + pixel;
    let word = stack[idx >> 1u];
    let val = select(word & 0xFFFFu, word >> 16u, (idx & 1u) != 0u);
    return f32(val);
}

// sRGB companding (linear → gamma)
fn srgb_gamma(c: f32) -> f32 {
    return select(12.92 * c, 1.055 * pow(c, 1.0 / 2.4) - 0.055, c > 0.0031308);
}

// XYZ to linear sRGB (D65 illuminant)
fn xyz_to_linear_rgb(xyz: vec3f) -> vec3f {
    return vec3f(
        dot(xyz, vec3f( 3.2404542, -1.5371385, -0.4985314)),
        dot(xyz, vec3f(-0.9692660,  1.8760108,  0.0415560)),
        dot(xyz, vec3f( 0.2126729,  0.7151522,  0.0721750)),
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) { return; }

    let pixel = y * params.width + x;
    var xyz = vec3f(0.0);

    // Sum weighted contributions from enabled excitations
    for (var exc = 0u; exc < params.num_excitations; exc++) {
        if ((params.exc_mask & (1u << exc)) == 0u) { continue; }
        let start = intervals[exc].x;
        let stop  = intervals[exc].y;
        for (var c = start; c < stop; c++) {
            var val = read_u16(c, pixel) / params.max_val;
            if (params.gamma != 1.0) { val = pow(val, params.gamma); }
            let w = xyz_weights[c].xyz;
            xyz += val * w;
        }
    }

    // XYZ → linear RGB → sRGB
    var rgb = xyz_to_linear_rgb(xyz);
    rgb = max(rgb, vec3f(0.0));

    // Optional luminance rescaling (simplified: scale by max component)
    if (params.rescale_lum != 0u) {
        let mx = max(rgb.x, max(rgb.y, rgb.z));
        if (mx > 0.0) { rgb /= mx; }
    }

    // Apply sRGB gamma
    rgb = vec3f(srgb_gamma(rgb.x), srgb_gamma(rgb.y), srgb_gamma(rgb.z));
    rgb = clamp(rgb, vec3f(0.0), vec3f(1.0));

    let r = u32(rgb.x * 255.0);
    let g = u32(rgb.y * 255.0);
    let b = u32(rgb.z * 255.0);
    output[pixel] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
"""


def render_spectral_rgb(
    stack: np.ndarray,
    xyz_weights: np.ndarray,
    channel_widths: list[int],
    *,
    exc_mask: int = -1,
    gamma: float = 1.0,
    rescale_luminance: bool = True,
    device: "wgpu.GPUDevice | None" = None,
) -> np.ndarray:
    """Render a spectral stack to sRGB using a WebGPU compute shader.

    Parameters
    ----------
    stack : (C, H, W) uint16 array
    xyz_weights : (C, 3) float array of per-channel CIE XYZ weights. The caller
        supplies these (e.g. from a wavelength→XYZ conversion) so this renderer
        carries no spectroscopy/CIE coupling — it just accumulates the weights.
    channel_widths : list of int, number of channels per excitation
    exc_mask : bitmask of enabled excitations (-1 = all)
    gamma : gamma correction exponent
    rescale_luminance : if True, normalize per-pixel luminance
    device : optional wgpu device (auto-created if None)

    Returns
    -------
    (H, W, 4) uint8 RGBA array
    """
    C, H, W = stack.shape
    intervals = infer_channel_intervals(C, channel_widths)
    n_exc = len(intervals)
    if exc_mask < 0:
        exc_mask = (1 << n_exc) - 1

    # CIE XYZ weights supplied by the caller (decoupled from wavelength math)
    xyz = np.asarray(xyz_weights, dtype=np.float32)
    # Pad to vec4f alignment
    xyz_padded = np.zeros((C, 4), dtype=np.float32)
    xyz_padded[:, :3] = xyz

    # Interval pairs as vec2u
    ivl_arr = np.array(intervals, dtype=np.uint32).reshape(-1, 2)

    # Params uniform
    max_val = float(stack.max()) if stack.max() > 0 else 1.0
    params = np.zeros(8, dtype=np.uint32)
    params[0] = W
    params[1] = H
    params[2] = C
    params[3] = n_exc
    params[4] = np.uint32(exc_mask)
    params[5] = np.frombuffer(np.float32(max_val).tobytes(), dtype=np.uint32)[0]
    params[6] = np.frombuffer(np.float32(gamma).tobytes(), dtype=np.uint32)[0]
    params[7] = 1 if rescale_luminance else 0

    # Pack stack as uint16 pairs into uint32 for GPU
    stack_flat = np.ascontiguousarray(stack.ravel().astype(np.uint16))
    # Pad to even length for u32 packing
    if len(stack_flat) % 2:
        stack_flat = np.append(stack_flat, np.uint16(0))
    stack_u32 = stack_flat.view(np.uint32)

    # Create device if needed
    if device is None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()

    t0 = time.time()

    # Create buffers
    params_buf = device.create_buffer_with_data(data=params.tobytes(),
                                                 usage=wgpu.BufferUsage.UNIFORM)
    stack_buf = device.create_buffer_with_data(data=stack_u32.tobytes(),
                                                usage=wgpu.BufferUsage.STORAGE)
    xyz_buf = device.create_buffer_with_data(data=xyz_padded.tobytes(),
                                              usage=wgpu.BufferUsage.STORAGE)
    ivl_buf = device.create_buffer_with_data(data=ivl_arr.tobytes(),
                                              usage=wgpu.BufferUsage.STORAGE)
    out_size = W * H * 4
    out_buf = device.create_buffer(size=out_size,
                                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    read_buf = device.create_buffer(size=out_size,
                                     usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)

    # Pipeline
    shader_module = device.create_shader_module(code=SPECTRAL_RGB_SHADER)
    bind_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
    ])
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_layout])
    pipeline = device.create_compute_pipeline(layout=pipeline_layout,
                                               compute={"module": shader_module, "entry_point": "main"})
    bind_group = device.create_bind_group(layout=bind_layout, entries=[
        {"binding": 0, "resource": {"buffer": params_buf}},
        {"binding": 1, "resource": {"buffer": stack_buf}},
        {"binding": 2, "resource": {"buffer": xyz_buf}},
        {"binding": 3, "resource": {"buffer": ivl_buf}},
        {"binding": 4, "resource": {"buffer": out_buf}},
    ])

    # Dispatch
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(W / 16), math.ceil(H / 16))
    compute_pass.end()
    encoder.copy_buffer_to_buffer(out_buf, 0, read_buf, 0, out_size)
    device.queue.submit([encoder.finish()])

    # Readback
    read_buf.map_sync(mode=wgpu.MapMode.READ)
    data = read_buf.read_mapped()
    rgba = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 4).copy()
    read_buf.unmap()

    t_total = (time.time() - t0) * 1000
    print(f"  Spectral→RGB GPU: {t_total:.1f}ms ({W}x{H}, {C} channels, {n_exc} excitations)")
    return rgba


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=400)
    parser.add_argument('--num_sine', type=int, default=50000)
    parser.add_argument('--num_cosine', type=int, default=50000)
    parser.add_argument('--num_points', type=int, default=100)
    parser.add_argument('--spread', type=float, default=1.0)
    parser.add_argument('--line_width', type=float, default=1.0)
    parser.add_argument('--transfer_fn', default='eq_hist', choices=['linear', 'log', 'cbrt', 'eq_hist'])
    parser.add_argument('--color_scale', type=float, default=1.0)
    parser.add_argument('--output', default='output.png')
    args = parser.parse_args()

    x_data, y_data, num_lines = generate_point_data(
        args.num_sine, args.num_cosine, args.num_points, args.width, args.height, args.spread)

    renderer = DensityLineRenderer(args.width, args.height)
    rgba = renderer.render(x_data, y_data, num_lines, args.num_points, args.line_width / 2.0,
                          transfer_fn=args.transfer_fn, color_scale=args.color_scale)

    from PIL import Image
    Image.fromarray(rgba, 'RGBA').save(args.output)
    print(f"Saved {args.output}")
