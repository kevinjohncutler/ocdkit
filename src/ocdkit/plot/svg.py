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


# SMPTE 2084 (PQ) constants — same as hiprpy.io.hdr (kept inline here so
# ocdkit doesn't depend on hiprpy).
_PQ_M1 = np.float32(2610.0 / 16384.0)
_PQ_M2 = np.float32(2523.0 / 4096.0 * 128.0)
_PQ_C1 = np.float32(3424.0 / 4096.0)
_PQ_C2 = np.float32(2413.0 / 4096.0 * 32.0)
_PQ_C3 = np.float32(2392.0 / 4096.0 * 32.0)


def _pq_oetf(linear_norm: 'np.ndarray', sdr_white_nits: float) -> 'np.ndarray':
    """SMPTE 2084 PQ OETF. ``linear_norm`` in [0, 1] (1.0 = peak content)
    maps such that 1.0 corresponds to ``sdr_white_nits`` of display
    output. Returns PQ signal in [0, 1].
    """
    L = np.clip(linear_norm * (sdr_white_nits / 10000.0),
                0.0, 1.0).astype(np.float32)
    Lm1 = L ** _PQ_M1
    return (((_PQ_C1 + _PQ_C2 * Lm1) / (1.0 + _PQ_C3 * Lm1))
            ** _PQ_M2).astype(np.float32)


def _p3_linear_to_pq_hdr_uint16(rgb_p3_linear: 'np.ndarray',
                                 sdr_white_nits: float = 1000.0,
                                 shadow_gamma: float = 2.2) -> 'np.ndarray':
    """Encode an ``(H, W, 3)`` linear-light Display P3 float array as a
    16-bit PQ-HDR Display P3 array (primaries unchanged) for JXL
    embedding.

    Note: stays in Display P3 throughout — does NOT convert to Rec.2020.
    Apple displays (XDR, recent MBP/iMac) target the P3 gamut; tagging
    as Rec.2020 would claim a wider gamut than either the source data
    or the display can render, forcing the decoder into pointless
    gamut-mapping. The :func:`hiprpy.io.hdr.encode_rgb_hdr` path uses
    Rec.2020 because BT.2100 broadcast HDR standardises on it; for
    on-screen P3 content we stay in P3.

    Pipeline: image-peak normalize → ``shadow_gamma`` pre-OETF → PQ OETF
    → uint16 quantize. Pair with a JXL ``ColorSpec(primaries=11,
    transfer=16)`` (Display P3 + PQ) at encode time.

    ``sdr_white_nits=1000`` (default) is conservative: Apple HDR displays
    (XDR, late MBP / iMac) can hit 1600 nits HDR peak but only at default
    brightness slider settings — raising the brightness slider trades SDR-
    white headroom for HDR-peak headroom, dropping the achievable HDR
    ceiling. A 1000-nit target fits inside the HDR envelope at almost any
    slider position. Bump to ``sdr_white_nits=1600`` for max XDR punch at
    default brightness; drop to ``sdr_white_nits=600`` if your viewers are
    on dimmer or non-XDR displays.

    Caller must supply **genuinely HDR** linear-light P3 RGB — this
    function does NOT promote SDR sRGB cmap output to HDR (that would
    be a lie about the source dynamic range; encode that as SDR JXL
    or PNG instead).
    """
    if rgb_p3_linear.dtype not in (np.float32, np.float64):
        raise TypeError(
            "_p3_linear_to_pq_hdr_uint16 expects float input "
            "(linear-light P3 RGB); got dtype "
            f"{rgb_p3_linear.dtype}. For SDR sRGB images, use the 'jxl' "
            "or 'jxl-p3' formats instead."
        )
    arr = np.asarray(rgb_p3_linear)
    has_alpha = arr.shape[-1] == 4
    rgb = np.clip(arr[..., :3].astype(np.float32, copy=False), 0.0, None)
    peak = float(rgb.max())
    if peak <= 0:
        peak = 1.0
    rgb_norm = rgb / peak
    perceptual = np.power(rgb_norm,
                          np.float32(shadow_gamma)).astype(np.float32)
    pq = _pq_oetf(perceptual, sdr_white_nits)
    rgb_u16 = (np.clip(pq, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    if has_alpha:
        # Alpha is encoded linearly (no PQ — alpha is opacity, not light).
        alpha = arr[..., 3:4].astype(np.float32, copy=False)
        alpha_u16 = (np.clip(alpha, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        return np.concatenate([rgb_u16, alpha_u16], axis=-1)
    return rgb_u16


def _srgb_to_display_p3_uint8(arr_u8: 'np.ndarray') -> 'np.ndarray':
    """Convert sRGB-encoded ``(H, W, 3|4)`` uint8 → Display P3-encoded uint8.

    Both sRGB and Display P3 use the same transfer curve (the ``sRGB``
    OETF/EOTF), so the pipeline is:

      1. Decode sRGB EOTF → linear-light sRGB.
      2. Linear sRGB → linear P3 via the 3×3 chromaticity matrix.
      3. Apply the (same) sRGB OETF to encode for the wire / file.
      4. Quantize back to uint8.

    Alpha channel passes through untouched. This keeps perceptual colour
    identity across sRGB and P3 displays — without it, P3-tagged sRGB
    pixels would render *more saturated* on a wide-gamut display than on
    an sRGB display.
    """
    has_alpha = arr_u8.shape[-1] == 4
    rgb = arr_u8[..., :3].astype(np.float32) / 255.0
    # sRGB EOTF (decode to linear)
    a = 0.055
    linear = np.where(rgb <= 0.04045,
                      rgb / 12.92,
                      ((rgb + a) / (1.0 + a)) ** 2.4).astype(np.float32)
    # sRGB → P3 (D65 → D65). Coefficients from the standard
    # Bradford-chromatic-adapted transform; small rotation, no clipping
    # (sRGB ⊂ P3).
    M = np.array([
        [0.8225, 0.1775, 0.0000],
        [0.0332, 0.9669, 0.0000],
        [0.0171, 0.0724, 0.9108],
    ], dtype=np.float32)
    linear_p3 = linear @ M.T
    # Re-encode with the sRGB OETF (P3's transfer is the same curve)
    encoded = np.where(linear_p3 <= 0.0031308,
                       12.92 * linear_p3,
                       (1.0 + a) * (linear_p3 ** (1.0 / 2.4)) - a)
    out_rgb = (np.clip(encoded, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
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
      the data should be a 16-bit array. See ``hiprpy.io.hdr`` for the
      P3→Rec.2020 → PQ-OETF pipeline.

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


class SVG:
    """String-based SVG document builder."""

    def __init__(self, width: int, height: int, background: str | None = None,
                 default_font_family: str = "Helvetica, Arial, 'DejaVu Sans', sans-serif"):
        self.width = width
        self.height = height
        # ``max-width:100%; height:auto`` lets the SVG shrink to fit a
        # narrower container (e.g. a JupyterLab cell that's narrower than
        # ``width`` px) while keeping the viewBox aspect ratio. When the
        # container is wider than ``width``, the SVG stays at its
        # intrinsic size so it doesn't blur in standalone viewers.
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'style="max-width:100%;height:auto" '
            f'shape-rendering="geometricPrecision">'
        ]
        if background is not None:
            self.rect(0, 0, width, height, fill=background)
        # CSS default font for text elements.
        self.parts.append(
            f"<style>text {{ font-family: {default_font_family}; }}</style>"
        )

    def add(self, raw: str):
        self.parts.append(raw)

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
        * ``'jxl-hdr-pq'`` — for **natively HDR-source data only**:
          expects a ``(H, W, 3)`` linear-light Display-P3 float array
          (e.g. ``hiprpy.io.hdr.compute_rgb_p3_linear(scene)`` output).
          Encoded via :func:`_p3_linear_to_pq_hdr_uint16` (image-peak
          normalise → ``shadow_gamma=2.2`` pre-OETF → PQ OETF →
          uint16), then JXL with a manual
          ``ColorSpec(primaries=11, transfer=16)`` — **Display P3 + PQ**
          (not Rec.2020). Apple displays target P3; tagging Rec.2020
          would claim wider gamut than either source or display can
          render. Pass ``intensity_target=`` nits (default 1600 = XDR
          peak). **Rejects uint8 input** because promoting SDR sRGB to
          PQ is meaningless — use ``'jxl'`` / ``'jxl-p3'`` for
          cmap-output / density panels instead.

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
            sdr_white_nits = float(encoder_kwargs.pop('sdr_white_nits', 1000.0))
            shadow_gamma = float(encoder_kwargs.pop('shadow_gamma', 2.2))
            arr_pq = _p3_linear_to_pq_hdr_uint16(
                arr, sdr_white_nits=sdr_white_nits,
                shadow_gamma=shadow_gamma,
            )
            # ColorSpec(primaries=11, transfer=16) = Display P3 + PQ.
            # opencodecs has no string alias for this combo (only
            # 'rec2020-pq' / 'display-p3' separately) but the manual
            # ColorSpec works and decoders honour it.
            p3_pq = opencodecs.ColorSpec(primaries=11, transfer=16,
                                          white_point=1,
                                          rendering_intent=1, gamma=0.0)
            url = jxl_data_url(arr_pq, color=p3_pq,
                               intensity_target=sdr_white_nits,
                               **encoder_kwargs)
        elif format == 'jxl':
            url = jxl_data_url(rgba_arr, **encoder_kwargs)
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


__all__ = ['SVG', 'rgba_to_css', 'png_data_url']
