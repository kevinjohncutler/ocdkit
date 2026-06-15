"""Headless wgpu-native renderer for tileserve tiles — pixel-identical, no browser.

Runs the EXACT viewer WGSL (:mod:`ocdkit.tileserve.shaders`) through Python
``wgpu`` (wgpu-native), so the live GL/WebGPU plot pipeline can produce rasters
from scripts without Playwright or a real browser. The browser viewer and this
renderer share one shader source (a drift test pins them byte-identical), so a
render here reproduces the fragment math the browser paints — the SDR path is
byte-exact against a numpy reference (see the repro harness).

Three tile modes, dispatched on the pixel array:

  - ``uint8`` HxWx{3,4}   → RGB pipeline (passthrough; byte-exact).
  - ``float32`` HxW       → intensity pipeline (normalize lo/hi → colormap LUT).
  - ``float32`` HxWx{3,4} → HDR pipeline (peak-norm linear-P3 → ``OETF(d*headroom)``;
                            ``headroom=1`` reproduces the SDR sRGB byte path).

Public surface
--------------
- :func:`get_device`   process-wide cached headless ``wgpu.GPUDevice``
- :func:`render_tile`  one tile → ``(H, W, 4)`` uint8 RGBA
- :func:`render_grid`  list of tiles → mosaiced ``(H, W, 4)`` uint8 RGBA
- :func:`save_png`     write an RGBA array to PNG (skimage; no PIL)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# ─────────────────────────── device cache ───────────────────────────────────
_DEVICE = None          # cached wgpu.GPUDevice
_FLOAT_FILT = False      # adapter has 'float32-filterable' (linear-min on r32float)
_PIPELINES: dict = {}    # (mode, target_fmt, hdr_cmap) -> (pipeline, wgpu module)
_LUT_TEX: dict = {}      # cmap name -> (texture, sampler)


def get_device():
    """Return a process-wide cached headless ``wgpu.GPUDevice``.

    Cold init is ~100–300 ms; subsequent calls are instant. ``float32-filterable``
    is requested when the adapter supports it so intensity (R32F) tiles can be
    minified with a linear sampler, matching the viewer's ``intSmp``.
    """
    global _DEVICE, _FLOAT_FILT
    if _DEVICE is None:
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        _FLOAT_FILT = "float32-filterable" in adapter.features
        feats = ["float32-filterable"] if _FLOAT_FILT else []
        _DEVICE = adapter.request_device_sync(required_features=feats)
    return _DEVICE


def _wgpu():
    import wgpu
    return wgpu


# ─────────────────────────── pipelines ──────────────────────────────────────
def _render_pipeline(mode: str, target_fmt, hdr_cmap: bool = False):
    """Cached render pipeline for ``mode`` ('rgb' | 'hdr' | 'int') + target fmt."""
    from . import shaders
    key = (mode, str(target_fmt), hdr_cmap)
    cached = _PIPELINES.get(key)
    if cached is not None:
        return cached
    wgpu = _wgpu()
    dev = get_device()
    if mode == "rgb":
        code = shaders.RGB
    elif mode == "hdr":
        code = shaders.HDR
    elif mode == "int":
        code = shaders.int_wgsl(hdr_cmap)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    mod = dev.create_shader_module(code=code)
    pipe = dev.create_render_pipeline(
        layout="auto",
        vertex={"module": mod, "entry_point": "vs"},
        fragment={"module": mod, "entry_point": "fs",
                  "targets": [{"format": target_fmt}]},
        primitive={"topology": "triangle-list"})
    _PIPELINES[key] = pipe
    return pipe


def _lut_texture(cmap: str):
    """Cached 256×1 rgba8unorm LUT texture + linear sampler for ``cmap``.

    Uses ocdkit's own ``colormap_lut(name, 'uint8')`` (cmap.Colormap), the SAME
    source the browser viewer uploads — so colormapped output matches the viewer.
    """
    cached = _LUT_TEX.get(cmap)
    if cached is not None:
        return cached
    from ocdkit.plot.luts import colormap_lut
    wgpu = _wgpu()
    dev = get_device()
    lut = np.ascontiguousarray(colormap_lut(cmap, "uint8"))      # (256,4) uint8
    tex = dev.create_texture(
        size=(256, 1, 1), format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dev.queue.write_texture({"texture": tex}, lut.tobytes(),
                            {"bytes_per_row": 256 * 4, "rows_per_image": 1}, (256, 1, 1))
    smp = dev.create_sampler(mag_filter="linear", min_filter="linear")
    _LUT_TEX[cmap] = (tex, smp)
    return tex, smp


# ─────────────────────────── helpers ────────────────────────────────────────
def _infer_mode(pixels: np.ndarray) -> str:
    a = np.asarray(pixels)
    if a.ndim == 2:
        return "int"
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return "rgb" if a.dtype == np.uint8 else "hdr"
    raise ValueError(f"cannot infer tile mode from shape {a.shape} dtype {a.dtype}")


def _upload_rgb_u8(dev, a: np.ndarray):
    """uint8 HxWx{3,4} → rgba8unorm texture (alpha forced opaque)."""
    wgpu = _wgpu()
    h, w = a.shape[:2]
    if a.shape[2] == 3:
        rgba = np.empty((h, w, 4), np.uint8)
        rgba[..., :3] = a
        rgba[..., 3] = 255
    else:
        rgba = np.ascontiguousarray(a)
    tex = dev.create_texture(
        size=(w, h, 1), format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dev.queue.write_texture({"texture": tex}, np.ascontiguousarray(rgba).tobytes(),
                            {"bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
    return tex, w, h


def _upload_rgb_f16(dev, a: np.ndarray):
    """float32 HxWx{3,4} → rgba16float texture (alpha 1.0), matching viewer HDR path."""
    wgpu = _wgpu()
    h, w = a.shape[:2]
    half = np.zeros((h, w, 4), "<f2")
    half[..., :3] = a[..., :3].astype("<f2")
    half[..., 3] = np.float16(1.0)
    tex = dev.create_texture(
        size=(w, h, 1), format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dev.queue.write_texture({"texture": tex}, np.ascontiguousarray(half).tobytes(),
                            {"bytes_per_row": w * 8, "rows_per_image": h}, (w, h, 1))
    return tex, w, h


def _upload_intensity(dev, a: np.ndarray):
    """float HxW → r32float (or r16float) scalar texture + matching sampler."""
    wgpu = _wgpu()
    h, w = a.shape[:2]
    f16 = a.dtype == np.float16
    fmt = wgpu.TextureFormat.r16float if f16 else wgpu.TextureFormat.r32float
    data = np.ascontiguousarray(a if f16 else a.astype(np.float32))
    bpp = 2 if f16 else 4
    tex = dev.create_texture(
        size=(w, h, 1), format=fmt,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dev.queue.write_texture({"texture": tex}, data.tobytes(),
                            {"bytes_per_row": w * bpp, "rows_per_image": h}, (w, h, 1))
    # r16float is always filterable; r32float needs the feature (else nearest-min).
    lin_min = f16 or _FLOAT_FILT
    smp = dev.create_sampler(mag_filter="nearest",
                             min_filter="linear" if lin_min else "nearest")
    return tex, w, h, smp


def _readback(dev, target, w: int, h: int) -> np.ndarray:
    """Copy an rgba8unorm render target back to a (h, w, 4) uint8 array."""
    wgpu = _wgpu()
    row = w * 4
    padded = ((row + 255) // 256) * 256
    buf = dev.create_buffer(size=padded * h,
                            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc = dev.create_command_encoder()
    enc.copy_texture_to_buffer(
        {"texture": target},
        {"buffer": buf, "bytes_per_row": padded, "rows_per_image": h}, (w, h, 1))
    dev.queue.submit([enc.finish()])
    buf.map_sync(wgpu.MapMode.READ)
    raw = np.frombuffer(buf.read_mapped(), np.uint8).reshape(h, padded)[:, :row].copy()
    buf.unmap()
    return raw.reshape(h, w, 4)


# ─────────────────────────── public render ──────────────────────────────────
def render_tile(pixels: np.ndarray, out_w: Optional[int] = None,
                out_h: Optional[int] = None, *, mode: Optional[str] = None,
                lo: float = 0.0, hi: float = 1.0, cmap: str = "viridis",
                headroom: float = 1.0,
                vp: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
                tr: Sequence[float] = (0.0, 0.0, 1.0, 1.0)) -> np.ndarray:
    """Render one tile through the matching viewer pipeline → ``(H, W, 4)`` uint8.

    ``pixels`` is uint8 HxWx{3,4} (RGB), float32 HxW (intensity), or float32
    HxWx{3,4} (linear-P3 RGB). ``out_w/out_h`` default to the tile's native size
    (1:1, byte-exact). ``lo``/``hi`` set the intensity normalization window;
    ``cmap`` the intensity colormap; ``headroom`` the HDR-RGB multiplier
    (``1.0`` ⇒ SDR sRGB, byte-matching the 8-bit path). ``vp``/``tr`` are the
    FOV-norm view rect and the rect this tile covers (defaults ⇒ full coverage).
    """
    wgpu = _wgpu()
    dev = get_device()
    a = np.asarray(pixels)
    mode = mode or _infer_mode(a)
    h0, w0 = a.shape[:2]
    W = int(out_w or w0)
    H = int(out_h or h0)
    target_fmt = wgpu.TextureFormat.rgba8unorm

    # uniform buffer (48B covers the widest U; RGB ignores the tail)
    U = np.zeros(12, np.float32)
    U[0:4] = vp
    if mode == "int":
        U[4:8] = [lo, hi, 0, 0]
        U[8:12] = tr
    elif mode == "hdr":
        U[4:8] = [headroom, 0, 0, 0]
        U[8:12] = tr
    else:  # rgb
        U[4:8] = tr
    ubuf = dev.create_buffer_with_data(
        data=U.tobytes(), usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    pipe = _render_pipeline(mode, target_fmt)

    if mode == "rgb":
        tex, _, _ = _upload_rgb_u8(dev, a)
        smp = dev.create_sampler(mag_filter="nearest", min_filter="linear")
        bg0 = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": tex.create_view()}, {"binding": 1, "resource": smp}])
        bg_groups = [(0, bg0)]
        ub_group = 1
    elif mode == "hdr":
        tex, _, _ = _upload_rgb_f16(dev, a)
        smp = dev.create_sampler(mag_filter="nearest", min_filter="linear")
        bg0 = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": tex.create_view()}, {"binding": 1, "resource": smp}])
        bg_groups = [(0, bg0)]
        ub_group = 1
    else:  # int
        tex, _, _, smp = _upload_intensity(dev, a)
        lut_tex, lut_smp = _lut_texture(cmap)
        bg0 = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": tex.create_view()}, {"binding": 1, "resource": smp}])
        bg1 = dev.create_bind_group(layout=pipe.get_bind_group_layout(1), entries=[
            {"binding": 0, "resource": lut_tex.create_view()}, {"binding": 1, "resource": lut_smp}])
        bg_groups = [(0, bg0), (1, bg1)]
        ub_group = 2

    ub_bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(ub_group), entries=[
        {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": U.nbytes}}])

    target = dev.create_texture(
        size=(W, H, 1), format=target_fmt,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    enc = dev.create_command_encoder()
    rp = enc.begin_render_pass(color_attachments=[{
        "view": target.create_view(), "clear_value": (0, 0, 0, 0),
        "load_op": "clear", "store_op": "store"}])
    rp.set_pipeline(pipe)
    for slot, bg in bg_groups:
        rp.set_bind_group(slot, bg)
    rp.set_bind_group(ub_group, ub_bg)
    rp.draw(3)
    rp.end()
    dev.queue.submit([enc.finish()])
    return _readback(dev, target, W, H)


def render_grid(tiles: Sequence[np.ndarray], cols: int, *, cell_w: int, cell_h: int,
                gap: int = 4, bg: Sequence[int] = (0, 0, 0, 0),
                lo: float = 0.0, hi: float = 1.0, cmap: str = "viridis",
                headroom: float = 1.0) -> np.ndarray:
    """Render each tile to ``cell_w × cell_h`` and mosaic into a ``cols``-wide grid.

    Each cell goes through :func:`render_tile` independently (the image_grid
    semantics — independent cells, no linking), then the cells are composited
    into one RGBA array with ``gap`` px between them on a ``bg`` background.
    """
    n = len(tiles)
    rows = (n + cols - 1) // cols
    H = rows * cell_h + (rows - 1) * gap
    W = cols * cell_w + (cols - 1) * gap
    out = np.empty((H, W, 4), np.uint8)
    out[:] = np.asarray(bg, np.uint8)
    for i, t in enumerate(tiles):
        cell = render_tile(t, cell_w, cell_h, lo=lo, hi=hi, cmap=cmap, headroom=headroom)
        r, c = divmod(i, cols)
        y0 = r * (cell_h + gap)
        x0 = c * (cell_w + gap)
        out[y0:y0 + cell_h, x0:x0 + cell_w] = cell
    return out


def encode_png(arr: np.ndarray) -> bytes:
    """Encode an ``(H, W, {3,4})`` uint8 array to PNG bytes using only the stdlib.

    A minimal zlib-deflate PNG writer — deliberately PIL/OpenCV-free (the
    project bans both, and imageio/skimage route PNG through Pillow). Supports
    8-bit RGB (colour type 2) and RGBA (colour type 6).
    """
    import struct
    import zlib

    a = np.ascontiguousarray(arr, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"expected (H,W,3|4) uint8, got shape {a.shape}")
    h, w, ch = a.shape
    color_type = 2 if ch == 3 else 6

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    # each scanline prefixed with filter byte 0 (None)
    raw = np.empty((h, w * ch + 1), np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = a.reshape(h, w * ch)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
            + chunk(b"IEND", b""))


def save_png(arr: np.ndarray, path: str) -> str:
    """Write an ``(H, W, {3,4})`` uint8 array to ``path`` as PNG (stdlib only). Returns path."""
    with open(path, "wb") as fh:
        fh.write(encode_png(arr))
    return path


# Anti-aliased rounded-rect alpha mask — clips tile corners to a rounded
# frame (the live viewer uses a CSS mask; headless render bakes it in).
def _rounded_alpha(w: int, h: int, r: float, inset: float = 0.0) -> np.ndarray:
    """Anti-aliased rounded-rect alpha mask (h, w) float [0,1] — used to clip each
    tile's corners to the rounded frame (the live viewer does this with a CSS mask;
    a sharp tile corner otherwise pokes out under the rounded outline)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    hx, hy = (w - 2 * inset) / 2.0, (h - 2 * inset) / 2.0
    r = float(max(0.0, min(r, hx, hy)))
    qx = np.abs(xx + 0.5 - cx) - (hx - r)
    qy = np.abs(yy + 0.5 - cy) - (hy - r)
    outx = np.maximum(qx, 0.0)
    outy = np.maximum(qy, 0.0)
    sdf = np.sqrt(outx * outx + outy * outy) + np.minimum(np.maximum(qx, qy), 0.0) - r
    return np.clip(0.5 - sdf, 0.0, 1.0).astype(np.float32)
