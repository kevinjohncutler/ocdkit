"""Bare-bones SVG document builder.

Pure string emission -- no PIL, no matplotlib, no XML library. Vector
elements (text, lines, rects, polygons, paths) stay vector so they render
crisply at any zoom. Raster content is embedded via ``<image>`` with a
base64-encoded PNG data URL.

Designed to pair with :mod:`ocdkit.plot.text_metrics` for exact
FreeType-measured glyph widths so Python-side layout matches what the
browser will eventually render.
"""
from __future__ import annotations

import base64
import io
import struct
import zlib
from html import escape
from typing import Sequence

from .imports import *


def rgba_to_css(rgba) -> str:
    """Tuple/array ``(r, g, b[, a])`` in ``[0, 1]`` or ``[0, 255]`` →
    CSS color string. ``None`` returns ``'none'``.

    Strings already valid as CSS (``#rrggbb``, ``rgb(...)``, named colors
    like ``red``, single-letter matplotlib aliases like ``'r'``) are
    forwarded through :func:`matplotlib.colors.to_rgba` so the caller
    doesn't have to think about the source format.
    """
    if rgba is None:
        return 'none'
    if isinstance(rgba, str):
        try:
            import matplotlib.colors as mcolors
            rgba = mcolors.to_rgba(rgba)
        except Exception:
            # Fall back to passing the string straight through; the
            # browser/SVG renderer will handle anything matplotlib didn't
            # recognize (e.g. ``rgba(...)`` already).
            return rgba
    r, g, b = rgba[0], rgba[1], rgba[2]
    a = rgba[3] if len(rgba) > 3 else 1.0
    if max(r, g, b, a) > 1.0001:  # already 0..255
        return f"rgba({int(r)},{int(g)},{int(b)},{a/255.0 if a > 1 else a:.3f})"
    return (f"rgba({int(round(r*255))},{int(round(g*255))},"
            f"{int(round(b*255))},{a:.3f})")


# SMPTE 2084 (PQ) constants (kept inline so ocdkit has no hard dependency
# on a separate HDR-encoding module).
_PQ_M1 = np.float32(2610.0 / 16384.0)
_PQ_M2 = np.float32(2523.0 / 4096.0 * 128.0)
_PQ_C1 = np.float32(3424.0 / 4096.0)
_PQ_C2 = np.float32(2413.0 / 4096.0 * 32.0)
_PQ_C3 = np.float32(2392.0 / 4096.0 * 32.0)


def _pq_uint16_to_p3_linear(rgb_pq_uint16: 'np.ndarray',
                            sdr_white_nits: float = 1600.0,
                            shadow_gamma: float = 1.0) -> 'np.ndarray':
    """Inverse of :func:`_p3_linear_to_pq_uint16`. Take a PQ-encoded
    uint16 array (as e.g. ``opencodecs.jxl.read`` returns from an HDR
    JXL with ``transfer=16``) and return the linear-light Display-P3
    float array.

    **Absolute SDR-white reference**: linear ``1.0`` = ``sdr_white_nits``
    nits (default 1600 = XDR HDR peak). Values >1 are HDR headroom up
    to ``10000 / sdr_white_nits``.

    ``shadow_gamma`` reverses the luminance-only perceptual pre-shaping
    applied by :func:`_p3_linear_to_pq_uint16` (default 1.5 matches the
    encode default).  Hue and saturation are preserved by construction
    — the inverse gain is computed from the encoded luminance
    ``L' = max(R',G',B')`` as ``L'^(1/gamma - 1)``. Pass
    ``shadow_gamma=1.0`` for raw inverse PQ on files that didn't have
    shadow compression applied.
    """
    if rgb_pq_uint16.dtype != np.uint16:
        raise TypeError(
            f"_pq_uint16_to_p3_linear expects uint16 PQ-encoded input, "
            f"got dtype {rgb_pq_uint16.dtype}."
        )
    arr = np.asarray(rgb_pq_uint16, dtype=np.float32) / 65535.0
    sm2 = arr ** (np.float32(1.0) / _PQ_M2)
    num = np.maximum(sm2 - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * sm2
    L_pq_norm = (num / np.maximum(den, 1e-12)) ** (np.float32(1.0) / _PQ_M1)
    linear = (L_pq_norm * (np.float32(10000.0) /
                           np.float32(sdr_white_nits))).astype(np.float32)
    if shadow_gamma != 1.0:
        linear = np.clip(linear, 0.0, None)
        L = linear.max(axis=-1, keepdims=True)
        L_safe = np.maximum(L, np.float32(1e-12))
        linear = (linear * np.power(L_safe,
                                     np.float32(1.0 / shadow_gamma - 1.0))
                  ).astype(np.float32)
    return linear


def _p3_linear_to_pq_uint16(rgb_p3_linear: 'np.ndarray',
                            sdr_white_nits: float = 1600.0,
                            shadow_gamma: float = 1.0) -> 'np.ndarray':
    """Encode an ``(H, W, 3|4)`` linear-light Display P3 float array as a
    16-bit PQ-encoded Display P3 array (primaries unchanged) for JXL
    embedding.

    **Absolute SDR-white reference**: linear value ``1.0`` →
    ``sdr_white_nits`` nits of display output (default 1600 = Apple
    Pro Display XDR HDR peak). Values >1 occupy the HDR headroom up
    to ``10000 / sdr_white_nits``.

    Applies ``shadow_gamma`` as a hue/saturation-preserving perceptual
    roll-off BEFORE the PQ OETF: ``L = max(R,G,B)`` is gamma-curved,
    RGB is scaled by ``L^(gamma-1)``. The default 1.5 is anchored to
    the SDR 8-bit noise floor (matches SDR brightness at v=1/256).
    The matching inverse in :func:`_pq_uint16_to_p3_linear` recovers
    the scene-linear values. Set ``shadow_gamma=1.0`` for raw PQ.

    Stays in Display P3 throughout. Pair with
    ``opencodecs.ColorSpec(primaries=11, transfer=16)`` (Display P3
    + PQ) and ``intensity_target=10000`` on the JXL side.
    """
    if rgb_p3_linear.dtype not in (np.float32, np.float64):
        raise TypeError(
            "_p3_linear_to_pq_uint16 expects float input "
            "(linear-light P3 RGB); got dtype "
            f"{rgb_p3_linear.dtype}. For SDR sRGB images, use the 'jxl' "
            "or 'jxl-p3' formats instead."
        )
    arr = np.asarray(rgb_p3_linear)
    has_alpha = arr.shape[-1] == 4
    rgb = np.clip(arr[..., :3].astype(np.float32, copy=False), 0.0, None)
    if shadow_gamma != 1.0:
        L = rgb.max(axis=-1, keepdims=True)
        L_safe = np.maximum(L, np.float32(1e-12))
        rgb = (rgb * np.power(L_safe, np.float32(shadow_gamma - 1.0))
               ).astype(np.float32)
    # Absolute reference: linear 1.0 -> sdr_white_nits / 10000 PQ-norm.
    L = np.clip(rgb * (np.float32(sdr_white_nits) / np.float32(10000.0)),
                0.0, 1.0)
    Lm1 = L ** _PQ_M1
    pq = (((_PQ_C1 + _PQ_C2 * Lm1) / (1.0 + _PQ_C3 * Lm1)) ** _PQ_M2)
    rgb_u16 = (np.clip(pq, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    if has_alpha:
        # Alpha is encoded linearly (no PQ — alpha is opacity, not light).
        alpha = arr[..., 3:4].astype(np.float32, copy=False)
        alpha_u16 = (np.clip(alpha, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        return np.concatenate([rgb_u16, alpha_u16], axis=-1)
    return rgb_u16


def _p3_linear_to_rec2100_pq_uint16(rgb_p3_linear: 'np.ndarray',
                                     sdr_white_nits: float = 1600.0,
                                     shadow_gamma: float = 1.0) -> 'np.ndarray':
    """Like :func:`_p3_linear_to_pq_uint16` but converts the primaries
    to BT.2020/Rec.2100 before PQ-encoding.

    The output array is rec2020-primaries, D65 white, PQ-encoded -- the
    standard HDR10/Rec.2100-PQ profile. Pair with
    ``opencodecs.ColorSpec(primaries=9, transfer=16, white_point=1)``
    (Rec.2020 primaries + PQ) on the JXL side. Most HDR-aware browser /
    OS pipelines recognise rec2100-pq as ``the`` HDR standard; some
    treat Display-P3+PQ as a non-standard combination and fall back
    to dumber color-management paths.

    The input is still scene-linear Display-P3 (P3-D65 primaries, D65
    white, linear-light). The P3->Rec.2020 conversion is a 3x3 matrix
    in linear-light space; the gamut shrinks slightly (P3's red and
    green are inside Rec.2020) so no clipping needed for in-gamut
    sources. Out-of-gamut P3 values may produce small negatives in
    Rec.2020 which the subsequent PQ step clips to 0.
    """
    if rgb_p3_linear.dtype not in (np.float32, np.float64):
        raise TypeError(
            "_p3_linear_to_rec2100_pq_uint16 expects float input "
            "(linear-light P3 RGB); got dtype "
            f"{rgb_p3_linear.dtype}."
        )
    arr = np.asarray(rgb_p3_linear)
    has_alpha = arr.shape[-1] == 4
    rgb_p3 = np.clip(arr[..., :3].astype(np.float32, copy=False), 0.0, None)
    # Linear P3-D65 -> linear Rec.2020 via a 3x3 column-vector matrix.
    # Implemented as ``rgb @ M.T`` so it broadcasts cleanly across the
    # leading H x W axes.
    rgb_2020 = rgb_p3 @ _P3_TO_REC2020_MATRIX_T
    if has_alpha:
        rgb_2020 = np.concatenate([rgb_2020, arr[..., 3:4]], axis=-1)
    return _p3_linear_to_pq_uint16(
        rgb_2020, sdr_white_nits=sdr_white_nits, shadow_gamma=shadow_gamma,
    )


_SRGB_EOTF_LUT_F32: 'np.ndarray | None' = None
_SRGB_OETF_U8_LUT: 'np.ndarray | None' = None
_SRGB_OETF_LUT_BINS = 4096   # 0..4096 inclusive → 4097 entries
_SRGB_TO_P3_MATRIX_T = np.array([
    [0.8225, 0.1775, 0.0000],
    [0.0332, 0.9669, 0.0000],
    [0.0171, 0.0724, 0.9108],
], dtype=np.float32).T

# Linear Display-P3 (P3-D65) -> linear Rec.2020 / BT.2100 RGB, D65 whitepoint.
# Computed from chromaticities:
#   P3-D65   : R(0.680, 0.320), G(0.265, 0.690), B(0.150, 0.060)
#   BT.2020  : R(0.708, 0.292), G(0.170, 0.797), B(0.131, 0.046)
# Both share D65; matrix = M_RGB2XYZ(BT.2020)^-1 @ M_RGB2XYZ(P3-D65).
# Stored as the standard column-vector matrix M, i.e. ``rgb_2020 = M @ rgb_p3``;
# np.einsum / @ apply via ``rgb_p3 @ M.T`` over the last axis.
_P3_TO_REC2020_MATRIX_T = np.array([
    [ 0.75383303,  0.19859737,  0.04756960],
    [ 0.04574385,  0.94177722,  0.01247893],
    [-0.00121034,  0.01760172,  0.98360862],
], dtype=np.float32).T

def _build_srgb_eotf_lut() -> 'np.ndarray':
    """256-entry sRGB EOTF lookup table (uint8 sRGB → float32 linear)."""
    i = np.arange(256, dtype=np.float32) / 255.0
    a = 0.055
    return np.where(i <= 0.04045,
                    i / 12.92,
                    ((i + a) / (1.0 + a)) ** 2.4).astype(np.float32)


def _build_srgb_oetf_uint8_lut() -> 'np.ndarray':
    """Fused OETF + quantize LUT: linear [0, 1] sampled at 4097 bins → uint8.

    Replaces a per-pixel ``np.where`` over the OETF (~68 ms on a 2k²
    float32 buffer) with a single uint8 fancy-index lookup (~10 ms).
    4097 bins → ≤ ±0.5 error in the encoded uint8 across the full input
    range, matching the original ``encode * 255 + 0.5 → uint8``
    precision.
    """
    n = _SRGB_OETF_LUT_BINS + 1
    x = np.arange(n, dtype=np.float32) / _SRGB_OETF_LUT_BINS
    a = 0.055
    encoded = np.where(x <= 0.0031308,
                       12.92 * x,
                       (1.0 + a) * (x ** (1.0 / 2.4)) - a)
    return np.clip(encoded * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _linear_p3_to_uint8_srgb_peaknorm(rgb_p3_linear: 'np.ndarray') -> 'np.ndarray':
    """Peak-normalize a linear-light Display-P3 array and apply the sRGB
    OETF to land at uint8 — the SDR fallback for an Ultra-HDR JPEG.

    Used by :class:`ArraySource` when ``fmt='uhdr'``: HDR-aware viewers
    composite the gain map back to the original brightness; SDR-only
    viewers see this peak-normalized base rendition.
    """
    global _SRGB_OETF_U8_LUT
    if _SRGB_OETF_U8_LUT is None:
        _SRGB_OETF_U8_LUT = _build_srgb_oetf_uint8_lut()
    rgb = np.asarray(rgb_p3_linear)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.float32, copy=False)
    peak = float(rgb.max()) if rgb.size else 1.0
    if peak > 1.0:
        rgb = rgb * np.float32(1.0 / peak)
    rgb = np.clip(rgb, 0.0, 1.0)
    idx = (rgb * _SRGB_OETF_LUT_BINS + 0.5).astype(np.int32)
    np.clip(idx, 0, _SRGB_OETF_LUT_BINS, out=idx)
    return _SRGB_OETF_U8_LUT[idx]


def _srgb_uint8_to_p3_linear(arr_u8: 'np.ndarray') -> 'np.ndarray':
    """sRGB-encoded uint8 → linear-light Display-P3 float32.

    Same first two steps as :func:`_srgb_to_display_p3_uint8` (EOTF
    LUT + sRGB→P3 matrix) but stops short of re-applying the OETF and
    quantizing. Used by the HDR thumbnail path when only an in-memory
    ``scene._rgb`` cache is available — we need to hand the PQ encoder
    *actual* linear-light values, not raw gamma-encoded uint8.

    Alpha channel is dropped (HDR thumbnails are opaque).
    """
    global _SRGB_EOTF_LUT_F32
    if _SRGB_EOTF_LUT_F32 is None:
        _SRGB_EOTF_LUT_F32 = _build_srgb_eotf_lut()
    rgb_u8 = arr_u8[..., :3] if arr_u8.shape[-1] == 4 else arr_u8
    linear_srgb = _SRGB_EOTF_LUT_F32[rgb_u8]
    return linear_srgb @ _SRGB_TO_P3_MATRIX_T


_CMS_SRGB_TO_P3 = ...   # sentinel until first call: None ⇒ unavailable, callable ⇒ in use


def _resolve_cms_path():
    """Resolve the lcms2-accelerated sRGB→P3 callable, or ``None``.

    Probed once and cached.  ``opencodecs.srgb_to_display_p3_uint8`` uses
    lcms2's ICC color-management pipeline — ~28 ms on a 2k² uint8 RGB,
    ~4× the pure-numpy LUT version below.  When liblcms2 / opencodecs
    aren't installed we silently fall back to the LUT path so the
    package keeps working without that dep.
    """
    global _CMS_SRGB_TO_P3
    if _CMS_SRGB_TO_P3 is not Ellipsis:
        return _CMS_SRGB_TO_P3
    try:
        from opencodecs._cms_codec import srgb_to_display_p3_uint8
        # One probe call to confirm liblcms2 actually loads on this host.
        srgb_to_display_p3_uint8(np.zeros((1, 1, 3), dtype=np.uint8))
        _CMS_SRGB_TO_P3 = srgb_to_display_p3_uint8
    except Exception:
        _CMS_SRGB_TO_P3 = None
    return _CMS_SRGB_TO_P3


def _srgb_to_display_p3_uint8(arr_u8: 'np.ndarray') -> 'np.ndarray':
    """Convert sRGB-encoded ``(H, W, 3|4)`` uint8 → Display P3-encoded uint8.

    Both sRGB and Display P3 use the same transfer curve (the ``sRGB``
    OETF/EOTF), so the pipeline is:

      1. Decode sRGB EOTF → linear-light sRGB (via a 256-entry LUT —
         the input is uint8, so this is exact and ~20× faster than
         ``np.where`` over a float32 view of the full image).
      2. Linear sRGB → linear P3 via the 3×3 chromaticity matrix.
      3. Apply the (same) sRGB OETF to encode for the wire / file.
      4. Quantize back to uint8.

    When liblcms2 is reachable through ``opencodecs._cms_codec``,
    delegates to its ICC-based pipeline — same ≤1 uint8-unit error vs
    the numpy LUT path but ~4× faster (28 ms vs 110 ms on a 2k² image).
    Falls back transparently when opencodecs isn't installed or lcms2
    isn't loadable on this host.

    Alpha channel passes through untouched. This keeps perceptual colour
    identity across sRGB and P3 displays — without it, P3-tagged sRGB
    pixels would render *more saturated* on a wide-gamut display than on
    an sRGB display.
    """
    cms = _resolve_cms_path()
    if cms is not None:
        return cms(arr_u8)

    global _SRGB_EOTF_LUT_F32, _SRGB_OETF_U8_LUT
    if _SRGB_EOTF_LUT_F32 is None:
        _SRGB_EOTF_LUT_F32 = _build_srgb_eotf_lut()
    if _SRGB_OETF_U8_LUT is None:
        _SRGB_OETF_U8_LUT = _build_srgb_oetf_uint8_lut()

    has_alpha = arr_u8.shape[-1] == 4
    rgb_u8 = arr_u8[..., :3]
    # Decode sRGB → linear via 256-entry LUT.  Fancy-indexing into a
    # float32 LUT materializes the (H, W, 3) float32 array directly,
    # skipping the ``astype(float32) / 255`` + ``np.where`` pair the
    # previous implementation did.  On 2k² uint8 that's ~200 ms → ~10 ms.
    linear = _SRGB_EOTF_LUT_F32[rgb_u8]
    linear_p3 = linear @ _SRGB_TO_P3_MATRIX_T
    # Map clipped linear-P3 [0, 1] → 4096-bin index → fused OETF+quantize
    # LUT.  Replaces the previous per-pixel ``np.where`` + clip + mul +
    # cast chain with a single uint8 fancy-index lookup (~68 ms → ~10 ms
    # on a 2k² buffer).
    np.clip(linear_p3, 0.0, 1.0, out=linear_p3)
    idx = (linear_p3 * _SRGB_OETF_LUT_BINS).astype(np.int32)
    out_rgb = _SRGB_OETF_U8_LUT[idx]
    if has_alpha:
        return np.concatenate([out_rgb, arr_u8[..., 3:4]], axis=-1)
    return out_rgb


_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk: ``length | tag | payload | crc(tag+payload)``."""
    return (struct.pack(">I", len(payload))
            + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def encode_png(arr, compress_level: int = 1, use_up_filter: bool = True) -> bytes:
    """Minimal PNG encoder for ``(H, W, 3|4)`` uint8 RGB(A) arrays.

    Skips Pillow / imageio overhead by writing the IHDR + IDAT + IEND
    chunks directly using ``zlib``.

    PNG supports 5 filters that pre-condition the raw bytes for better
    zlib compression. We apply the ``Up`` filter (scanline N minus
    scanline N-1, mod 256) on every row except the first, which:

    * collapses empty / uniform / smoothly-varying vertical regions to
      mostly-zero bytes (huge compression win for image-grid composites
      that have lots of transparent space and tile padding),
    * is fully vectorisable (one numpy subtract per array vs one
      iteration per scanline), so it stays cheap to apply.

    The default ``compress_level=3`` + Up filter is the sweet spot for
    interactive plot rendering: benchmarked across the realistic image-
    grid + density panel mix it's about **2× faster than
    ``imageio.imwrite``** at the cost of ~50% larger files. Bump to
    ``compress_level=6`` (matching libpng's default) to match imageio's
    size at near-imageio speed; drop to ``compress_level=1`` for fastest
    encoding if the output is going straight to a data-URL inside an
    SVG that won't be saved to disk.
    """
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(
            f"encode_png expects (H, W, 3|4) uint8, got shape={arr.shape}"
        )
    h, w, c = arr.shape
    color_type = 6 if c == 4 else 2          # 6 = RGBA, 2 = RGB
    if use_up_filter and h > 1:
        # Filter type 2 (Up): each row's bytes minus the row above,
        # mod 256. Row 0 stays raw (filter type 0).
        diff = np.empty_like(arr)
        diff[0] = arr[0]
        np.subtract(arr[1:], arr[:-1], out=diff[1:], casting='unsafe')
        filter_col = np.full((h, 1), 2, dtype=np.uint8)
        filter_col[0] = 0
        flat_bytes = np.hstack((filter_col, diff.reshape(h, w * c))).tobytes()
    else:
        # Filter type 0 (None): one null byte per scanline.
        filter_col = np.zeros((h, 1), dtype=np.uint8)
        flat_bytes = np.hstack((filter_col, arr.reshape(h, w * c))).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    idat = zlib.compress(flat_bytes, compress_level)
    return (
        _PNG_SIG
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def png_data_url(rgba_arr, compress_level: int = 1) -> str:
    """Encode an ``(H, W, 3|4)`` float/uint8 RGBA buffer as a base64 PNG data URL."""
    arr = np.asarray(rgba_arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    png_bytes = encode_png(arr, compress_level=compress_level)
    b64 = base64.b64encode(png_bytes).decode('ascii')
    return f"data:image/png;base64,{b64}"


def jxl_data_url(rgba_arr, effort: int = 1, lossless: bool = True,
                  distance: float | None = None,
                  color=None,
                  icc_profile=None,
                  intensity_target: float | None = None) -> str:
    """Encode an ``(H, W, 3|4)`` RGB(A) buffer as a base64 JPEG-XL data URL.

    This is a thin wrapper around :func:`opencodecs.jxl_encode` — there is
    no separate "ocdkit JXL codec"; encode speed = opencodecs speed.

    JXL at ``effort=1`` is **~5× faster than PNG ``compress_level=1``** on
    realistic plot rasters (a mix of empty image-grid composite and dense
    density panels) AND produces ~3× smaller output. Trade-off is browser
    support: native in Safari (16.4+) and Chromium with the decoder
    re-enabled. **Firefox stable does NOT decode JXL** (verified in
    Playwright Firefox 146 → ``naturalSize=0x0``). Use
    :func:`png_data_url` when the SVG must render in any browser.

    ``color`` / ``icc_profile`` / ``intensity_target`` pass through to
    :func:`opencodecs.jxl_encode` for wide-gamut and HDR encodes:

    * ``color='display-p3'`` — tag as Display P3 (Apple wide-gamut, ~25%
      more coverage than sRGB; cleanly displayed on any Apple device).
    * ``color='rec2020-pq'`` + ``intensity_target=`` nits — full PQ HDR;
      the data should be a 16-bit array (P3→Rec.2020 → PQ-OETF
      pre-processed upstream).

    Default is sRGB (no color tag) since our standard plot rasters are
    sRGB colormapped intensity. Pass ``lossless=False`` + ``distance=``
    (Butteraugli units, 1.0 ≈ visually lossless) for smaller-but-lossy.
    """
    import opencodecs
    arr = np.asarray(rgba_arr)
    # uint16 inputs pass through unchanged (for HDR PQ encodes). uint8
    # also passes through. Float gets quantized to uint8 [0, 255].
    if arr.dtype not in (np.uint8, np.uint16):
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    kw = dict(effort=effort, color=color, icc_profile=icc_profile)
    if intensity_target is not None:
        kw['intensity_target'] = intensity_target
    if lossless:
        kw['lossless'] = True
    else:
        kw['distance'] = distance or 1.0
    jxl_bytes = opencodecs.jxl_encode(arr, **kw)
    b64 = base64.b64encode(jxl_bytes).decode('ascii')
    return f"data:image/jxl;base64,{b64}"


def uhdr_data_url(hdr_lin_p3, sdr_u8=None, *, sdr_white_nits: float = 1600.0,
                   quality: int = 95) -> str:
    """Encode an ``(H, W, 3)`` linear-light Display-P3 float HDR buffer as
    a base64 Ultra-HDR JPEG data URL (``data:image/jpeg;base64,...``).

    Ultra-HDR is a regular JPEG with a per-pixel gain map encoded
    alongside the SDR base via ISO 21496-1 boxes. The data URL works
    everywhere a normal ``data:image/jpeg`` works — Safari and Chrome
    additionally composite the gain map to display in HDR, Firefox /
    Preview / iOS Quick Look show the SDR base.

    Pass ``sdr_u8`` (an HxWx3 uint8 P3-curve array — same primaries +
    transfer as the HDR input, just clipped/peak-normalized to [0, 1])
    to control the deterministic SDR fallback. ``None`` lets libuhdr's
    auto-tone-mapper derive the base from the HDR float — *don't* rely
    on that path for cmap content; the per-channel rolloff visibly
    desaturates / hue-shifts bright stops vs the original cmap. The
    upstream :func:`_linear_p3_to_uint8_srgb_peaknorm` produces an
    appropriate base for HDR-lifted cmaps and scene images alike.
    """
    import opencodecs.uhdr as uhdr
    hdr = np.ascontiguousarray(np.asarray(hdr_lin_p3, dtype=np.float32))
    if hdr.ndim != 3 or hdr.shape[-1] not in (3, 4):
        raise ValueError(
            f"uhdr_data_url expects (H, W, 3|4) float; got {hdr.shape}")
    if hdr.shape[-1] == 4:
        hdr = np.ascontiguousarray(hdr[..., :3])

    # Validate / coerce the SDR base layer up front so it can flow into
    # either encode path. uhdr.encode_native expects (H, W, 3) uint8
    # matching the HDR spatial dims.
    sdr_arr = None
    if sdr_u8 is not None:
        sdr_arr = np.ascontiguousarray(np.asarray(sdr_u8))
        if sdr_arr.dtype != np.uint8:
            raise TypeError(
                f"uhdr_data_url: sdr_u8 must be uint8; got {sdr_arr.dtype}")
        if sdr_arr.ndim != 3 or sdr_arr.shape[-1] not in (3, 4):
            raise ValueError(
                f"uhdr_data_url: sdr_u8 must be (H, W, 3|4); got {sdr_arr.shape}")
        if sdr_arr.shape[-1] == 4:
            sdr_arr = sdr_arr[..., :3]
        if sdr_arr.shape[:2] != hdr.shape[:2]:
            raise ValueError(
                f"uhdr_data_url: sdr_u8 spatial shape {sdr_arr.shape[:2]} "
                f"must match HDR {hdr.shape[:2]}")

    # encode_native's convention: hdr value 1.0 ≡ 203 nits (BT.2408
    # SDR-white). Our hdr buffer is normalized so 1.0 ≡ sdr_white_nits
    # — rescale before handing off so the gain map metadata matches.
    # We always use the Cython fast-path (no fall-through to libuhdr's
    # encode(), which is ~4x slower AND silently lets the auto-derived
    # min_content_boost drift > 1 — over-brightens SDR mode on EDR
    # displays, confirmed via gain-map metadata inspection 2026-05-26).
    scale = float(sdr_white_nits) / 203.0
    hdr_rescaled = (hdr.astype(np.float32, copy=False) * scale
                    ).astype(np.float32, copy=False)
    peak = float(hdr_rescaled.max()) if hdr_rescaled.size else 1.0
    max_boost = max(peak, 1.0 + 1e-6)
    data = uhdr.encode_native(
        hdr_rescaled, sdr=sdr_arr,   # sdr=None → Cython peak-norm SDR
        gamut="display-p3",
        sdr_white_nits=float(sdr_white_nits),
        quality=int(quality),
        # Hard floor: dim stops carry no boost at any display
        # headroom — uniform-Jz-scale lifts (apply_hdr_cmap) make
        # ALL stops brighter, so without this clamp SDR fallback
        # over-brightens on slight-headroom displays.
        min_content_boost=1.0,
        max_content_boost=max_boost,
    )
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _fmt(v) -> str:
    """Compact numeric formatting for SVG attribute values: drop trailing
    zeros + the decimal point if integer-valued, else keep two digits."""
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f == int(f):
        return str(int(f))
    return f"{f:.2f}"


class SVG:
    """String-based SVG document builder."""

    def __init__(self, width: int, height: int, background: str | None = None,
                 default_font_family: str = "Helvetica, Arial, 'DejaVu Sans', sans-serif",
                 viewBox: tuple[float, float, float, float] | None = None,
                 data_attrs: dict[str, str] | None = None):
        self.width = width
        self.height = height
        # ``viewBox`` lets the internal coordinate system be decoupled
        # from the outer CSS pixel size — useful for image_grid where
        # cells are sized in source-pixel units (so browser scaling
        # of embedded rasters is at integer multiples) while the SVG
        # itself renders at a smaller CSS size for the notebook.
        if viewBox is None:
            viewBox = (0, 0, width, height)
        self.viewBox = viewBox
        vb_str = " ".join(_fmt(v) for v in viewBox)
        # Optional data-* attributes on the root <svg>. Used by
        # image_grid to expose ``data-ncol`` for the SvgFigure JS shell
        # so it can navigate between cells with the arrow keys.
        data_str = (" " + " ".join(f'data-{k}="{v}"'
                                     for k, v in (data_attrs or {}).items())
                    if data_attrs else "")
        # ``max-width:100%; height:auto`` lets the SVG shrink to fit a
        # narrower container (e.g. a JupyterLab cell that's narrower than
        # ``width`` px) while keeping the viewBox aspect ratio. When the
        # container is wider than ``width``, the SVG stays at its
        # intrinsic size so it doesn't blur in standalone viewers.
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" '
            f'viewBox="{vb_str}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'style="max-width:100%;height:auto" '
            f'shape-rendering="geometricPrecision"{data_str}>'
        ]
        if background is not None:
            # Background covers the full viewBox, not the outer CSS box.
            self.rect(viewBox[0], viewBox[1], viewBox[2], viewBox[3],
                      fill=background)
        # CSS default font for text elements.
        self.parts.append(
            f"<style>text {{ font-family: {default_font_family}; }}</style>"
        )

    def add(self, raw: str):
        self.parts.append(raw)

    def text_extents(self, content: str, *, size: float,
                      weight: str = "regular") -> tuple[float, float]:
        """Return ``(visible_width, visible_height)`` for ``content`` at ``size``.

        Canonical "how big is this text" answer on the builder — uses
        the same FreeType face that produces our font metrics, no
        rasterizer involved.  Use this for layout decisions (slot
        reservation, centering, etc.) so the dimensions you reason
        about match the dimensions of the text element you'd emit
        with :meth:`text`.
        """
        from .text_metrics import measure_text_visible
        return measure_text_visible(content, size, weight=weight)

    # ── shape primitives ───────────────────────────────────────────────

    def rect(self, x, y, w, h, *, fill='none', stroke='none', stroke_width=0,
             opacity=None, class_=None):
        op = '' if opacity is None else f' opacity="{opacity}"'
        f = rgba_to_css(fill) if not isinstance(fill, str) else fill
        s = rgba_to_css(stroke) if not isinstance(stroke, str) else stroke
        cls = f' class="{escape(class_, quote=True)}"' if class_ else ''
        self.parts.append(
            f'<rect{cls} x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{f}" stroke="{s}" stroke-width="{stroke_width}"{op}/>'
        )

    def line(self, x1, y1, x2, y2, *, stroke='#666', stroke_width=1,
             linecap='butt', dasharray=None, class_=None):
        s = rgba_to_css(stroke) if not isinstance(stroke, str) else stroke
        da = f' stroke-dasharray="{dasharray}"' if dasharray else ''
        cls = f' class="{escape(class_, quote=True)}"' if class_ else ''
        self.parts.append(
            f'<line{cls} x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{s}" stroke-width="{stroke_width}" '
            f'stroke-linecap="{linecap}"{da}/>'
        )

    def text(self, x, y, content, *, fill='#888', size=10,
             anchor='start', baseline='alphabetic', weight='normal',
             family=None, transform=None, class_=None):
        # XML attribute values must escape "<", "&", and " (the surrounding
        # attribute quote). html.escape with quote=True handles all three.
        attrs = []
        if class_:
            attrs.append(f'class="{escape(class_, quote=True)}"')
        if transform:
            attrs.append(f'transform="{escape(transform, quote=True)}"')
        f = rgba_to_css(fill) if not isinstance(fill, str) else fill
        attrs.append(f'fill="{escape(f, quote=True)}"')
        attrs.append(f'font-size="{size}"')
        attrs.append(f'text-anchor="{anchor}"')
        attrs.append(f'dominant-baseline="{baseline}"')
        if weight != 'normal':
            attrs.append(f'font-weight="{escape(str(weight), quote=True)}"')
        if family:
            attrs.append(f'font-family="{escape(family, quote=True)}"')
        attr_str = ' '.join(attrs)
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" {attr_str}>{escape(content)}</text>'
        )

    def polygon(self, points: Sequence[tuple[float, float]], *,
                fill='none', stroke='none', stroke_width=0, opacity=None):
        pts = ' '.join(f"{x:.2f},{y:.2f}" for x, y in points)
        f = rgba_to_css(fill) if not isinstance(fill, str) else fill
        s = rgba_to_css(stroke) if not isinstance(stroke, str) else stroke
        op = '' if opacity is None else f' opacity="{opacity}"'
        self.parts.append(
            f'<polygon points="{pts}" fill="{f}" stroke="{s}" '
            f'stroke-width="{stroke_width}"{op}/>'
        )

    def path(self, d: str, *, stroke='#000', stroke_width=1, fill='none',
             dasharray=None, linecap='round', linejoin='round', opacity=None):
        s = rgba_to_css(stroke) if not isinstance(stroke, str) else stroke
        f = rgba_to_css(fill) if not isinstance(fill, str) else fill
        da = f' stroke-dasharray="{dasharray}"' if dasharray else ''
        op = '' if opacity is None else f' opacity="{opacity}"'
        self.parts.append(
            f'<path d="{d}" stroke="{s}" stroke-width="{stroke_width}" '
            f'fill="{f}" stroke-linecap="{linecap}" '
            f'stroke-linejoin="{linejoin}"{da}{op}/>'
        )

    def hatch_pattern(self, pattern: str, *, stroke='white',
                       stroke_width: float = 1.0,
                       alpha: float = 1.0,
                       tile: float = 6.0) -> str:
        """Register a hatch pattern in <defs>; return its element id.

        Mirrors matplotlib's small set of hatch glyphs:

          ``'/'``  → forward (45°) diagonal lines
          ``'\\'`` → backward (-45°) diagonal lines
          ``'|'``  → vertical lines
          ``'-'``  → horizontal lines
          ``'.'``  → dots
          ``'x'``  → crosshatch (45° + -45°)
          ``'+'``  → crosshatch (horizontal + vertical)
          ``''``   → no hatch (returns ``""`` so callers can
                     unconditionally ``fill=`url(#{id})``)

        Repeated characters (e.g. ``'/'*3``) request denser hatching —
        we pick a tile size proportional to ``tile / repeat_count``.

        Returns the pattern id (e.g. ``hatch-abc12345``).  Use as
        ``svg.rect(..., fill=f'url(#{id})')``.  ``stroke`` is the
        glyph color; ``alpha`` tints the whole pattern.
        """
        if not pattern:
            return ""
        # Repeat count adjusts density.
        char = pattern[0]
        repeats = max(1, len(pattern))
        t = max(2.0, tile / repeats ** 0.5)
        from hashlib import sha1
        pid = "hatch-" + sha1(
            f"{pattern}|{stroke}|{stroke_width}|{alpha}|{tile}".encode()
        ).hexdigest()[:8]
        # Build the inner glyph SVG.
        sw = stroke_width
        glyph = ""
        # All stripe glyphs: a SINGLE line per tile, drawn from edge to
        # opposite edge so neighbor tiles seam cleanly when tiled.
        # Diagonals extend slightly past the tile corners to avoid
        # sub-pixel gaps along the seam at small tile sizes.
        eps = sw  # extend by stroke-width past each end
        if char == "/":
            glyph = (f'<line x1="{-eps}" y1="{t+eps}" x2="{t+eps}" y2="{-eps}" '
                     f'stroke="{stroke}" stroke-width="{sw}" opacity="{alpha}"/>')
        elif char == "\\":
            glyph = (f'<line x1="{-eps}" y1="{-eps}" x2="{t+eps}" y2="{t+eps}" '
                     f'stroke="{stroke}" stroke-width="{sw}" opacity="{alpha}"/>')
        elif char == "|":
            glyph = (f'<line x1="{t/2}" y1="0" x2="{t/2}" y2="{t}" '
                     f'stroke="{stroke}" stroke-width="{sw}" opacity="{alpha}"/>')
        elif char == "-":
            glyph = (f'<line x1="0" y1="{t/2}" x2="{t}" y2="{t/2}" '
                     f'stroke="{stroke}" stroke-width="{sw}" opacity="{alpha}"/>')
        elif char == ".":
            r = max(0.5, sw)
            glyph = (f'<circle cx="{t/2}" cy="{t/2}" r="{r}" '
                     f'fill="{stroke}" opacity="{alpha}"/>')
        elif char == "x":
            glyph = (f'<line x1="{-eps}" y1="{-eps}" x2="{t+eps}" y2="{t+eps}" '
                     f'stroke="{stroke}" stroke-width="{sw}" opacity="{alpha}"/>'
                     f'<line x1="{-eps}" y1="{t+eps}" x2="{t+eps}" y2="{-eps}" '
                     f'stroke="{stroke}" stroke-width="{sw}" opacity="{alpha}"/>')
        elif char == "+":
            glyph = (f'<line x1="{t/2}" y1="0" x2="{t/2}" y2="{t}" '
                     f'stroke="{stroke}" stroke-width="{sw}" opacity="{alpha}"/>'
                     f'<line x1="0" y1="{t/2}" x2="{t}" y2="{t/2}" '
                     f'stroke="{stroke}" stroke-width="{sw}" opacity="{alpha}"/>')
        else:
            # Unknown — fall back to no hatch.
            return ""
        self.parts.append(
            f'<defs><pattern id="{pid}" patternUnits="userSpaceOnUse" '
            f'width="{t}" height="{t}">{glyph}</pattern></defs>'
        )
        return pid

    def image(self, x, y, w, h, rgba_arr, *,
              preserve_aspect='none', format: str = 'png',
              **encoder_kwargs):
        """Embed a raster image. ``format`` picks the encoder; any extra
        keyword arguments flow through to that encoder.

        * ``'png'`` (default) — :func:`png_data_url`, supported in every
          browser.
        * ``'jxl'`` — :func:`jxl_data_url` with no colour-space tag
          (treated as sRGB by the browser). ~5× faster to encode and
          ~3× smaller payload than PNG; renders in Safari 16.4+ and
          Chromium with the JXL decoder re-enabled. **Not Firefox stable.**
        * ``'jxl-p3'`` — JXL with the input converted **sRGB → Display
          P3** (linear-light matrix) and tagged as ``display-p3``.
          Preserves perceptual colour identity vs sRGB on sRGB-only
          displays while expanding into P3's wider gamut on capable
          displays (any recent Apple device). Same JXL browser caveats.
        * ``'jxl-hdr-pq'`` — for linear-light Display-P3 float input
          (e.g. the output of an upstream linear-P3 compositor).
          Uses the **absolute SDR-white reference**: linear ``1.0`` →
          ``sdr_white_nits`` (default 100) nits, values >1 fill HDR
          headroom up to 10000 nits (PQ peak). SDR content keeps SDR
          brightness even when HDR content shares the composite.
          Encoded via :func:`_p3_linear_to_pq_uint16` then JXL with a
          manual ``ColorSpec(primaries=11, transfer=16)`` —
          **Display P3 + PQ** (not Rec.2020). Apple displays target P3;
          tagging Rec.2020 would claim wider gamut than either source
          or display can render. **Rejects uint8 input** — for SDR
          content use ``'jxl'`` / ``'jxl-p3'`` instead.
        * ``'uhdr'`` — Ultra-HDR JPEG (ISO 21496-1). For float linear-
          P3 input: same scene-linear absolute-SDR-reference convention
          as ``'jxl-hdr-pq'``, but encoded as a JPEG with an embedded
          SDR base + gain map. Cross-browser HDR: Safari + Chrome
          composite, Firefox / Preview / non-HDR viewers see the SDR
          base. Smaller than JXL-PQ at the same quality.

        Examples::

            svg.image(0, 0, W, H, arr,           format='jxl-p3')     # sRGB → P3 JXL
            svg.image(0, 0, W, H, rgb_p3_linear, format='jxl-hdr-pq') # PQ-HDR JXL
            svg.image(0, 0, W, H, arr,           format='png', compress_level=1)
        """
        if format == 'jxl-p3':
            arr_u8 = np.asarray(rgba_arr)
            if arr_u8.dtype != np.uint8:
                arr_u8 = (np.clip(arr_u8, 0.0, 1.0) * 255).astype(np.uint8)
            arr_p3 = _srgb_to_display_p3_uint8(arr_u8)
            url = jxl_data_url(arr_p3, color='display-p3',
                               **encoder_kwargs)
        elif format == 'jxl-hdr-pq':
            import opencodecs
            arr = np.asarray(rgba_arr)
            sdr_white_nits = float(encoder_kwargs.pop('sdr_white_nits', 1600.0))
            # ``shadow_gamma`` is gone — the absolute-SDR-reference encoder
            # doesn't need a perceptual pre-OETF curve. Silently pop for
            # back-compat with the old peak-normalised signature.
            encoder_kwargs.pop('shadow_gamma', None)
            arr_pq = _p3_linear_to_pq_uint16(arr, sdr_white_nits=sdr_white_nits)
            # ColorSpec(primaries=11, transfer=16) = Display P3 + PQ.
            # opencodecs has no string alias for this combo (only
            # 'rec2020-pq' / 'display-p3' separately) but the manual
            # ColorSpec works and decoders honour it.
            p3_pq = opencodecs.ColorSpec(primaries=11, transfer=16,
                                          white_point=1,
                                          rendering_intent=1, gamma=0.0)
            # intensity_target is the JXL container's claim about the
            # brightest possible nits in this image — for absolute PQ
            # encoding that's always the PQ peak (10000 nits) regardless
            # of how bright the actual content is.
            url = jxl_data_url(arr_pq, color=p3_pq,
                               intensity_target=10000.0,
                               **encoder_kwargs)
        elif format == 'jxl':
            url = jxl_data_url(rgba_arr, **encoder_kwargs)
        elif format == 'uhdr':
            arr = np.asarray(rgba_arr)
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError(
                    "format='uhdr' requires float linear-P3 input; "
                    f"got dtype {arr.dtype}")
            url = uhdr_data_url(arr, **encoder_kwargs)
        else:
            url = png_data_url(rgba_arr, **encoder_kwargs)
        self.parts.append(
            f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'href="{url}" preserveAspectRatio="{preserve_aspect}" '
            f'image-rendering="pixelated"/>'
        )

    def group(self, *, transform=None, _open=True):
        if transform:
            self.parts.append(f'<g transform="{transform}">')
        else:
            self.parts.append('<g>')

    def end_group(self):
        self.parts.append('</g>')

    def finalize(self) -> str:
        self.parts.append('</svg>')
        return '\n'.join(self.parts)


def hdr_jxl_data_url(arr_linear_p3, *, sdr_white_nits: float = 1600.0) -> str:
    """Encode linear-light Display-P3 float → HDR P3-PQ JXL data URL.

    Inline thumbnails encoded this way match the ``jxl-hdr-pq`` raster
    format ``image_grid`` writes to disk so scene-linear HDR brightness
    survives across the inline thumb and the hi-res-on-zoom swap.
    """
    import opencodecs
    color = opencodecs.ColorSpec(
        primaries=11, transfer=16, white_point=1,
        rendering_intent=1, gamma=0.0,
    )
    arr_pq = _p3_linear_to_pq_uint16(arr_linear_p3, sdr_white_nits=sdr_white_nits)
    return jxl_data_url(arr_pq, color=color, intensity_target=10000.0)


def sdr_jxl_data_url(arr_u8) -> str:
    """Encode an sRGB-encoded uint8 RGB(A) array as a Display-P3 SDR JXL
    data URL.

    Both sRGB and Display P3 share the sRGB transfer curve, so this is a
    chromaticity-only retag: perceptual identity is preserved on sRGB
    displays while wide-gamut displays no longer over-saturate the pixels.
    """
    import numpy as np
    arr_u8 = np.asarray(arr_u8)
    if arr_u8.dtype != np.uint8:
        raise TypeError(
            f"sdr_jxl_data_url: arr must be uint8, got {arr_u8.dtype}")
    arr_p3 = _srgb_to_display_p3_uint8(arr_u8)
    return jxl_data_url(arr_p3, color='display-p3')


__all__ = ['SVG', 'rgba_to_css', 'png_data_url',
           'jxl_data_url', 'hdr_jxl_data_url', 'sdr_jxl_data_url']
