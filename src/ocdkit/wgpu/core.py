"""Shared GPU primitives for WGPU-backed plot rendering.

Both consumers (scatter, lines/spectra) need:

  - device + pipeline caching
  - colormap LUT packing + buffer upload
  - parallel max-reduction over a count grid
  - 256-bin histogram + Hillis-Steele prefix-sum scan (for ``eq_hist``)
  - colormap-apply (LUT lookup + alpha ramp + vmin/vmax clamp)

This module owns the primitives so each per-renderer module only
implements its own rasterization (point splat vs line rasterize vs ...).

Public surface
--------------
- :func:`get_device`            cached :class:`wgpu.GPUDevice`
- :func:`get_pipeline`          cached compute pipeline + bind group layout
- :func:`get_cmap_lut_buffer`   cached 256-entry packed-u32 LUT buffer
- :class:`ShadeConfig`          per-call shading parameters
- :func:`shade_count_buffer`    runs post-rasterization shading on a counts buffer
- :func:`readback_rgba`         copy a packed-RGBA8 storage buffer back to numpy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import wgpu


# ─────────────────────────── Device + pipeline cache ────────────────────────
_DEVICE: Optional["wgpu.GPUDevice"] = None
# Keyed by (hash(shader_src), entry_point, _bgl_signature(bgl_entries)).
_PIPELINE_CACHE: dict = {}


def get_device() -> "wgpu.GPUDevice":
    """Return a process-wide cached GPU device.

    Cold init is ~100–300 ms; every subsequent call returns instantly.
    """
    global _DEVICE
    if _DEVICE is None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        _DEVICE = adapter.request_device_sync()
    return _DEVICE


def _bgl_signature(bgl_entries: list) -> tuple:
    out = []
    for e in bgl_entries:
        b = e.get("buffer", {})
        out.append((e["binding"], int(e["visibility"]), b.get("type", None)))
    return tuple(out)


def get_pipeline(shader_src: str, entry_point: str, bgl_entries: list):
    """Cached ``(pipeline, bind_group_layout)`` for ``(shader_src, entry, bgl)``.

    Pipeline + shader-module compilation costs ~1–5 ms each; caching them
    means the second call is essentially free.
    """
    key = (hash(shader_src), entry_point, _bgl_signature(bgl_entries))
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached
    device = get_device()
    shader = device.create_shader_module(code=shader_src)
    bgl = device.create_bind_group_layout(entries=bgl_entries)
    pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipeline = device.create_compute_pipeline(
        layout=pl, compute={"module": shader, "entry_point": entry_point}
    )
    _PIPELINE_CACHE[key] = (pipeline, bgl)
    return pipeline, bgl


# ─────────────────────────── Colormap LUT cache ─────────────────────────────
_LUT_CACHE: dict = {}


def pack_cmap_lut(cmap_name="viridis", n: int = 256) -> np.ndarray:
    """Pack a colormap into a (n,) uint32 array, RGBA8 little-endian. ``cmap_name``
    is a matplotlib colormap NAME *or* a callable Colormap object — so callers can
    pass a custom/transparent map (e.g. a single-hue scatter colour) directly."""
    import matplotlib as mpl
    cmap = mpl.colormaps.get_cmap(cmap_name) if isinstance(cmap_name, str) else cmap_name
    rgba = (np.asarray(cmap(np.linspace(0, 1, n, dtype=np.float32))) * 255).astype(np.uint8)
    return (rgba[:, 0].astype(np.uint32)
            | (rgba[:, 1].astype(np.uint32) << 8)
            | (rgba[:, 2].astype(np.uint32) << 16)
            | (rgba[:, 3].astype(np.uint32) << 24))


def get_cmap_lut_buffer(cmap_name="viridis"):
    """Return a cached 256-entry packed-u32 LUT buffer for ``cmap_name`` (a name
    or a Colormap object; objects are keyed by ``.name`` or identity)."""
    key = cmap_name if isinstance(cmap_name, str) else (getattr(cmap_name, "name", None) or id(cmap_name))
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached
    device = get_device()
    packed = pack_cmap_lut(cmap_name)
    buf = device.create_buffer_with_data(
        data=packed.tobytes(), usage=wgpu.BufferUsage.STORAGE
    )
    _LUT_CACHE[key] = buf
    return buf


# ─────────────────────────── Shaders ───────────────────────────────────────
_REDUCE_MAX_SHADER = """
struct ReduceUniforms { n: u32, p0: u32, p1: u32, p2: u32 };

@group(0) @binding(0) var<uniform> u: ReduceUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read_write> max_out: atomic<u32>;

var<workgroup> wg_max: atomic<u32>;

@compute @workgroup_size(256)
fn cs_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if (lid.x == 0u) { atomicStore(&wg_max, 0u); }
    workgroupBarrier();
    if (gid.x < u.n) { atomicMax(&wg_max, bins[gid.x]); }
    workgroupBarrier();
    if (lid.x == 0u) { atomicMax(&max_out, atomicLoad(&wg_max)); }
}
"""

# Fused max + min-nonzero reduction.  Single dispatch when the caller wants
# auto-vmin; falls back to the lighter max-only shader otherwise so callers
# with explicit ``vmin`` pay the same cost as before auto-vmin existed.
# ``min_out`` must be pre-initialized to u32::MAX (atomicMin sentinel).
_REDUCE_MAX_MIN_SHADER = """
struct ReduceUniforms { n: u32, p0: u32, p1: u32, p2: u32 };

@group(0) @binding(0) var<uniform> u: ReduceUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read_write> max_out: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> min_out: atomic<u32>;

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
        let v = bins[gid.x];
        atomicMax(&wg_max, v);
        if (v > 0u) { atomicMin(&wg_min, v); }
    }
    workgroupBarrier();
    if (lid.x == 0u) {
        atomicMax(&max_out, atomicLoad(&wg_max));
        atomicMin(&min_out, atomicLoad(&wg_min));
    }
}
"""

_HISTOGRAM_SHADER = """
struct HistUniforms { width: u32, height: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<uniform> u: HistUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> max_count: array<u32>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn cs_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let c = bins[gid.y * u.width + gid.x];
    if (c == 0u) { return; }
    let mx = max(max_count[0], 1u);
    let bin = u32(clamp(floor((f32(c) / f32(mx)) * 255.0), 0.0, 255.0));
    atomicAdd(&hist[bin], 1u);
}
"""

_SCAN_SHADER = """
@group(0) @binding(0) var<storage, read> hist_in: array<u32, 256>;
@group(0) @binding(1) var<storage, read_write> cdf_out: array<f32, 256>;

var<workgroup> wg_buf: array<u32, 256>;

@compute @workgroup_size(256)
fn cs_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    wg_buf[i] = hist_in[i];
    workgroupBarrier();
    var stride: u32 = 1u;
    while (stride < 256u) {
        var v: u32 = 0u;
        if (i >= stride) { v = wg_buf[i - stride]; }
        workgroupBarrier();
        if (i >= stride) { wg_buf[i] = wg_buf[i] + v; }
        workgroupBarrier();
        stride = stride * 2u;
    }
    let total = max(wg_buf[255], 1u);
    cdf_out[i] = f32(wg_buf[i]) / f32(total);
}
"""

_COLORMAP_SHADER = """
struct CmapUniforms {
    width: u32, height: u32,
    auto_flags: u32, transfer_mode: u32,  // bit0=auto_vmin (use min_count), bit1=auto_vmax (use max_count); transfer 0 linear, 1 eq_hist, 2 log, 3 cbrt
    vmin: f32, vmax: f32,
    alpha_min: f32, alpha_max: f32,
};

@group(0) @binding(0) var<uniform> u: CmapUniforms;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> max_count: array<u32>;
@group(0) @binding(3) var<storage, read> lut: array<u32>;
@group(0) @binding(4) var<storage, read> cdf: array<f32>;
@group(0) @binding(5) var<storage, read_write> rgba_out: array<u32>;
@group(0) @binding(6) var<storage, read> min_count: array<u32>;

@compute @workgroup_size(8, 8)
fn cs_colormap(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let idx = gid.y * u.width + gid.x;
    let c = bins[idx];
    if (c == 0u) { rgba_out[idx] = 0u; return; }

    let mx = max(max_count[0], 1u);
    // auto_vmin only fires for non-eq_hist (eq_hist already redistributes density).
    let mn_raw = min_count[0];
    let mn = select(0u, mn_raw, mn_raw != 0xFFFFFFFFu);  // sentinel = no nonzero
    let lo = select(u.vmin, f32(mn), (u.auto_flags & 1u) != 0u);
    let hi = select(u.vmax, f32(mx), (u.auto_flags & 2u) != 0u);
    let span = max(hi - lo, 1.0e-6);

    var norm: f32;
    if (u.transfer_mode == 1u) {
        let pre = clamp(f32(c) / f32(mx), 0.0, 1.0);
        let bin_f = clamp(pre * 255.0, 0.0, 255.0);
        let bin_lo = u32(floor(bin_f));
        let bin_hi = min(bin_lo + 1u, 255u);
        let t = bin_f - f32(bin_lo);
        let eq_norm = mix(cdf[bin_lo], cdf[bin_hi], t);
        let lo_n = lo / f32(mx);
        let hi_n = hi / f32(mx);
        norm = clamp((eq_norm - lo_n) / max(hi_n - lo_n, 1.0e-6), 0.0, 1.0);
    } else if (u.transfer_mode == 2u) {
        let cf = clamp(f32(c), lo, hi);
        norm = log(1.0 + cf - lo) / log(1.0 + span);
    } else if (u.transfer_mode == 3u) {
        let lin = clamp((f32(c) - lo) / span, 0.0, 1.0);
        norm = pow(lin, 1.0 / 3.0);
    } else {
        norm = clamp((f32(c) - lo) / span, 0.0, 1.0);
    }

    let lut_idx = u32(clamp(norm * 255.0, 0.0, 255.0));
    let entry = lut[lut_idx];
    let alpha_t = u.alpha_min + (u.alpha_max - u.alpha_min) * norm;
    let lut_a = f32((entry >> 24u) & 0xFFu);
    let final_a = u32(clamp(alpha_t * lut_a, 0.0, 255.0));
    rgba_out[idx] = (entry & 0x00FFFFFFu) | (final_a << 24u);
}
"""


# ─────────────────────────── Bind group layouts ─────────────────────────────
_UNIFORM = wgpu.BufferBindingType.uniform
_RO_STORAGE = wgpu.BufferBindingType.read_only_storage
_RW_STORAGE = wgpu.BufferBindingType.storage
_VIS = wgpu.ShaderStage.COMPUTE

_BGL_REDUCE = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNIFORM}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RW_STORAGE}},
]
_BGL_REDUCE_MAX_MIN = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNIFORM}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RW_STORAGE}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RW_STORAGE}},
]
_BGL_HIST = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNIFORM}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RW_STORAGE}},
]
_BGL_SCAN = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RW_STORAGE}},
]
_BGL_CMAP = [
    {"binding": 0, "visibility": _VIS, "buffer": {"type": _UNIFORM}},
    {"binding": 1, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 2, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 3, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 4, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
    {"binding": 5, "visibility": _VIS, "buffer": {"type": _RW_STORAGE}},
    {"binding": 6, "visibility": _VIS, "buffer": {"type": _RO_STORAGE}},
]


@dataclass
class ShadeConfig:
    """Per-call shading parameters.

    Analogues of datashader's ``span`` / ``min_alpha`` / ``alpha`` / ``how``.
    """
    transfer: str = "eq_hist"   # "linear" | "eq_hist" | "log" | "cbrt"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    cmap_name: str = "viridis"


def shade_count_buffer(
    bins_buf,
    n_bins: int,
    plot_width: int,
    plot_height: int,
    cfg: ShadeConfig,
    *,
    encoder=None,
):
    """Run max-reduce → (eq_hist histogram + scan) → colormap-apply.

    ``bins_buf`` is a u32 storage buffer of shape ``(plot_height, plot_width)``.
    Returns the GPU-resident packed-RGBA8 buffer (n_bins × 4 bytes); call
    :func:`readback_rgba` if you need a numpy array.

    If ``encoder`` is None a fresh :class:`CommandEncoder` is created and
    submitted; otherwise the passes are appended to the caller's encoder
    (caller is responsible for submission).
    """
    transfer_mode = {"linear": 0, "eq_hist": 1, "log": 2, "cbrt": 3}.get(cfg.transfer)
    if transfer_mode is None:
        raise ValueError(f"unknown transfer: {cfg.transfer}")

    device = get_device()
    own_encoder = encoder is None
    if own_encoder:
        encoder = device.create_command_encoder()

    # Pass A: reduction.  When auto_vmin is on (linear/log/cbrt with vmin=None)
    # we run a fused max+min shader in one dispatch; otherwise we dispatch the
    # original max-only shader, so the explicit-vmin path costs the same as
    # before auto-vmin existed.
    auto_vmin = (cfg.vmin is None) and (transfer_mode != 1)  # eq_hist redistributes
    auto_vmax = cfg.vmax is None

    reduce_uniforms = np.array([n_bins, 0, 0, 0], dtype=np.uint32)
    reduce_ubo = device.create_buffer_with_data(
        data=reduce_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    max_buf = device.create_buffer(
        size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    # min_buf must be pre-initialized to u32::MAX (sentinel).  When auto_vmin
    # is off we leave the sentinel in place and the colormap shader ignores it.
    min_buf = device.create_buffer_with_data(
        data=np.array([0xFFFFFFFF], dtype=np.uint32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    if auto_vmin:
        pipe_r, bgl_r = get_pipeline(_REDUCE_MAX_MIN_SHADER, "cs_reduce_max_min", _BGL_REDUCE_MAX_MIN)
        bg_r = device.create_bind_group(layout=bgl_r, entries=[
            {"binding": 0, "resource": {"buffer": reduce_ubo, "offset": 0, "size": reduce_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": bins_buf,   "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": max_buf,    "offset": 0, "size": 4}},
            {"binding": 3, "resource": {"buffer": min_buf,    "offset": 0, "size": 4}},
        ])
    else:
        pipe_r, bgl_r = get_pipeline(_REDUCE_MAX_SHADER, "cs_reduce", _BGL_REDUCE)
        bg_r = device.create_bind_group(layout=bgl_r, entries=[
            {"binding": 0, "resource": {"buffer": reduce_ubo, "offset": 0, "size": reduce_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": bins_buf,   "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": max_buf,    "offset": 0, "size": 4}},
        ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe_r); p.set_bind_group(0, bg_r)
    p.dispatch_workgroups((n_bins + 255) // 256)
    p.end()

    # Passes B + C (eq_hist only): histogram + Hillis-Steele scan.
    cdf_buf = device.create_buffer(size=256 * 4, usage=wgpu.BufferUsage.STORAGE)
    if transfer_mode == 1:
        hist_uniforms = np.array([plot_width, plot_height, 0, 0], dtype=np.uint32)
        hist_ubo = device.create_buffer_with_data(
            data=hist_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        hist_buf = device.create_buffer(
            size=256 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        pipe_hist, bgl_hist = get_pipeline(_HISTOGRAM_SHADER, "cs_histogram", _BGL_HIST)
        bg_hist = device.create_bind_group(layout=bgl_hist, entries=[
            {"binding": 0, "resource": {"buffer": hist_ubo, "offset": 0, "size": hist_uniforms.nbytes}},
            {"binding": 1, "resource": {"buffer": bins_buf, "offset": 0, "size": n_bins * 4}},
            {"binding": 2, "resource": {"buffer": max_buf, "offset": 0, "size": 4}},
            {"binding": 3, "resource": {"buffer": hist_buf, "offset": 0, "size": 256 * 4}},
        ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipe_hist)
        p.set_bind_group(0, bg_hist)
        p.dispatch_workgroups((plot_width + 7) // 8, (plot_height + 7) // 8)
        p.end()

        pipe_scan, bgl_scan = get_pipeline(_SCAN_SHADER, "cs_scan", _BGL_SCAN)
        bg_scan = device.create_bind_group(layout=bgl_scan, entries=[
            {"binding": 0, "resource": {"buffer": hist_buf, "offset": 0, "size": 256 * 4}},
            {"binding": 1, "resource": {"buffer": cdf_buf, "offset": 0, "size": 256 * 4}},
        ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipe_scan)
        p.set_bind_group(0, bg_scan)
        p.dispatch_workgroups(1)
        p.end()

    # Pass D: colormap-apply.  auto_vmin/auto_vmax were decided above so the
    # reduce dispatch could pick the right shader.
    auto_flags = (1 if auto_vmin else 0) | (2 if auto_vmax else 0)

    cmap_uniforms = np.zeros(8, dtype=np.uint32)
    cmap_uniforms[0] = plot_width
    cmap_uniforms[1] = plot_height
    cmap_uniforms[2] = auto_flags
    cmap_uniforms[3] = transfer_mode
    cmap_uniforms.view(np.float32)[4] = float(cfg.vmin) if cfg.vmin is not None else 0.0
    cmap_uniforms.view(np.float32)[5] = float(cfg.vmax) if cfg.vmax is not None else 0.0
    cmap_uniforms.view(np.float32)[6] = float(cfg.alpha_min)
    cmap_uniforms.view(np.float32)[7] = float(cfg.alpha_max)
    cmap_ubo = device.create_buffer_with_data(
        data=cmap_uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM)

    lut_buf = get_cmap_lut_buffer(cfg.cmap_name)
    rgba_buf = device.create_buffer(
        size=n_bins * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    pipe_cmap, bgl_cmap = get_pipeline(_COLORMAP_SHADER, "cs_colormap", _BGL_CMAP)
    bg_cmap = device.create_bind_group(layout=bgl_cmap, entries=[
        {"binding": 0, "resource": {"buffer": cmap_ubo, "offset": 0, "size": cmap_uniforms.nbytes}},
        {"binding": 1, "resource": {"buffer": bins_buf, "offset": 0, "size": n_bins * 4}},
        {"binding": 2, "resource": {"buffer": max_buf, "offset": 0, "size": 4}},
        {"binding": 3, "resource": {"buffer": lut_buf, "offset": 0, "size": 256 * 4}},
        {"binding": 4, "resource": {"buffer": cdf_buf, "offset": 0, "size": 256 * 4}},
        {"binding": 5, "resource": {"buffer": rgba_buf, "offset": 0, "size": n_bins * 4}},
        {"binding": 6, "resource": {"buffer": min_buf, "offset": 0, "size": 4}},
    ])
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipe_cmap)
    p.set_bind_group(0, bg_cmap)
    p.dispatch_workgroups((plot_width + 7) // 8, (plot_height + 7) // 8)
    p.end()

    if own_encoder:
        device.queue.submit([encoder.finish()])

    return rgba_buf


def readback_rgba(rgba_buf, plot_width: int, plot_height: int) -> np.ndarray:
    """Copy a packed-RGBA8 storage buffer back to a (H, W, 4) uint8 ndarray."""
    device = get_device()
    n_bytes = plot_width * plot_height * 4
    readback = device.create_buffer(
        size=n_bytes,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder = device.create_command_encoder()
    encoder.copy_buffer_to_buffer(rgba_buf, 0, readback, 0, n_bytes)
    device.queue.submit([encoder.finish()])
    readback.map_sync(wgpu.MapMode.READ)
    raw = readback.read_mapped()
    readback.unmap()
    return np.frombuffer(raw, dtype=np.uint8).reshape(plot_height, plot_width, 4).copy()
