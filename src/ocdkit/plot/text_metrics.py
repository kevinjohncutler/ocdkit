"""Exact text measurement via FreeType.

Python-side layout decisions (label wrapping, axis sizing, anchor positions)
need pixel-accurate text widths. We use the same FreeType engine PIL wraps --
read glyph advance widths from the font's tables, no rasterization required.

In the eventual HTML/browser target the renderer does this with
``ctx.measureText`` at runtime using the actual font it'll render with.
Until then we approximate by loading a common system sans-serif here and
emitting the matching ``font-family`` name in the SVG.

Font selection notes
--------------------
The browser renders SVG ``<text>`` with whatever the ``font-family`` value
resolves to. If freetype-py measures one font but the browser renders
another, layout will be slightly off AND the visual style will differ from
what we intended. Keep the FreeType load path and the SVG ``font-family``
aligned: prefer plain Helvetica everywhere, avoid ``-apple-system`` which
resolves to San Francisco on macOS (different metrics, different look).
"""
from __future__ import annotations

import os
from functools import lru_cache

import freetype

from .imports import *  # noqa: F401  (centralized np/etc., even if unused here)


_FONT_CANDIDATES_REGULAR = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_CANDIDATES_BOLD = [
    # Helvetica.ttc face 1 = Bold; freetype defaults to face 0 (Regular).
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_FONT_CANDIDATES_MONO = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


# SVG font-family stacks. Helvetica first to match what freetype-py loads
# below; Arial / DejaVu Sans as cross-platform fallbacks; only fall to
# generic 'sans-serif' as last resort. NO `-apple-system` -- that resolves
# to San Francisco on macOS browsers, which has different metrics + visual
# style than what we measure with.
SVG_FONT_REGULAR = "Helvetica, Arial, 'DejaVu Sans', sans-serif"
SVG_FONT_BOLD = SVG_FONT_REGULAR
SVG_FONT_MONO = "Menlo, 'SF Mono', 'DejaVu Sans Mono', monospace"


@lru_cache(maxsize=4)
def load_face(weight: str = 'regular') -> freetype.Face:
    """Return a cached ``freetype.Face`` for the given weight bucket
    (``'regular'`` / ``'bold'`` / ``'mono'``). Raises SystemExit if no
    usable system font is found."""
    paths = {'regular': _FONT_CANDIDATES_REGULAR,
             'bold': _FONT_CANDIDATES_BOLD,
             'mono': _FONT_CANDIDATES_MONO}[weight]
    for p in paths:
        if os.path.exists(p):
            return freetype.Face(p)
    raise SystemExit("no usable system font found for freetype")


def measure_text(text: str, size_px: float,
                 weight: str = 'regular') -> tuple[float, float]:
    """Return ``(width_px, height_px)`` for ``text`` at ``size_px``.

    Width is the LAYOUT advance width (sum-of-glyph-advances — where
    the next text run would start).  Height is the face's ascender +
    (-descender).  Use this for laying out sequential text.

    For visible BOUNDING-BOX geometry (e.g. centering a label in a
    fixed slot so it looks centered to the eye), prefer
    :func:`measure_text_visible` — the advance sum overshoots the
    visible ink by the leading + trailing side bearings.
    """
    if not text:
        return (0.0, 0.0)
    face = load_face(weight)
    face.set_pixel_sizes(0, int(round(size_px)))
    # FT_LOAD_NO_HINTING: the .ttc hinter divides by zero at very small
    # pixel sizes (< ~6px). We measure for layout, not rasterization, so
    # hinting offers no benefit anyway — advance widths are font-metric,
    # not rasterizer-dependent.
    flags = freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_HINTING
    w = 0
    for ch in text:
        face.load_char(ch, flags)
        w += face.glyph.advance.x
    # FreeType advances are in 26.6 fixed-point pixels.
    width = w / 64.0
    h = (face.size.ascender - face.size.descender) / 64.0
    return (width, h)


def measure_text_visible(text: str, size_px: float,
                          weight: str = 'regular') -> tuple[float, float]:
    """Return ``(ink_width_px, ink_height_px)`` — the visible bounding box.

    Same FreeType face as :func:`measure_text`, but subtracts the
    leading glyph's left side bearing (whitespace before the first
    visible ink) and the trailing glyph's right side bearing
    (whitespace after the last visible ink).  Result matches what a
    cairo / browser / PowerPoint renderer will produce at the same
    font + size, since they all use the same font outlines.

    Use this for visual-bounding-box geometry — e.g. centering a
    rotated label in a reserved slot, where you want the visible ink
    centered, not the advance box.
    """
    if not text:
        return (0.0, 0.0)
    face = load_face(weight)
    face.set_pixel_sizes(0, int(round(size_px)))
    # Hinting is disabled — see :func:`measure_text` for the rationale.
    flags = freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_HINTING
    advance = 0
    first_left_bearing = 0
    last_right_bearing = 0
    last_idx = len(text) - 1
    for i, ch in enumerate(text):
        face.load_char(ch, flags)
        g = face.glyph
        advance += g.advance.x
        if i == 0:
            # horiBearingX = horizontal distance from origin to ink left edge.
            first_left_bearing = g.metrics.horiBearingX
        if i == last_idx:
            # right bearing = advance − (left bearing + ink width).
            last_right_bearing = (g.advance.x
                                   - g.metrics.horiBearingX
                                   - g.metrics.width)
    visible = (advance - first_left_bearing - last_right_bearing) / 64.0
    h = (face.size.ascender - face.size.descender) / 64.0
    return (visible, h)


def measure_char(size_px: float, ch: str = 'M',
                 weight: str = 'regular') -> float:
    """Advance width of a representative character (default ``'M'``, the
    widest capital). Useful for quick approximations when the actual label
    text isn't available yet."""
    return measure_text(ch, size_px, weight=weight)[0]


def chunk_words(words, max_width_px: float, size_px: float,
                weight: str = 'regular', sep: str = ', ') -> list[list[str]]:
    """Greedy word-wrap using exact FreeType-measured widths.

    Returns a list of lines, each a list of the words on that line.
    Separator ``sep`` (default ``', '``) is what visually goes between
    words on the same line; its measured width is included when deciding
    line breaks but the caller is responsible for actually drawing it.
    """
    if not words:
        return []
    sep_w = measure_text(sep, size_px, weight=weight)[0]
    word_w = [measure_text(w, size_px, weight=weight)[0] for w in words]
    lines: list[list[str]] = []
    cur: list[str] = []
    cur_w = 0.0
    for w, ww in zip(words, word_w):
        add = ww + (sep_w if cur else 0)
        if cur and cur_w + add > max_width_px:
            lines.append(cur)
            cur = [w]
            cur_w = ww
        else:
            cur.append(w)
            cur_w += add
    if cur:
        lines.append(cur)
    return lines


__all__ = [
    'SVG_FONT_REGULAR', 'SVG_FONT_BOLD', 'SVG_FONT_MONO',
    'load_face', 'measure_text', 'measure_char', 'chunk_words',
]
