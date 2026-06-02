"""SVG-backed Figure + Axes.

This module is intentionally small.  It exists for three things:

* **Wrap an SVG payload** so callers (notebooks, ``figs_to_deck``) can
  pass figures around as objects with stable methods instead of raw
  strings.
* **Apply themable color schemes** in-place on the SVG XML, before
  rasterization or export — so the colors carry through to both the
  embedded vector SVG and the PNG fallback in PowerPoint.
* **Render to Jupyter** via ``_repr_mimebundle_`` (SVG + PNG bundle).

Matplotlib figures are *not* wrapped — callers that need matplotlib
should produce ``matplotlib.figure.Figure`` objects directly.  PPTX
exporters accept both ``SvgFigure`` and ``matplotlib.figure.Figure`` as
siblings.

The class is named ``SvgFigure`` (not ``Figure``) so it can sit
alongside ``matplotlib.figure.Figure`` in the same import scope without
shadowing.

Recolor targets (when present in the SVG, via the conventional
``class="fig-*"`` attribute scheme):

  class="fig-background"   — the slide-canvas background rect
  class="fig-axis"         — per-axis <g> wrapper (yields an Axes)
  class="fig-title"        — axis title text
  class="fig-axis-label"   — x/y axis label text
  class="fig-tick-label"   — tick label text
  class="fig-suptitle"     — figure-level suptitle text
  class="fig-figure-text"  — other figure-level annotations
  class="fig-spine"        — spine path / panel border
  class="fig-tick"         — tick mark line

Unstructured SVG (no ``fig-*`` classes) still gets font recoloring via
the walk-all-``<text>`` fallback in ``set_figure_text_color``; spine
recoloring requires the class marks.
"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np

# lxml (libxml2) over stdlib xml.etree.ElementTree: libxml2 handles multi-MB
# attribute values without the spurious "out of memory" Expat error that
# stdlib trips on image_grid SVGs whose <image href="data:..."> base64
# payloads add up. lxml is also ~5-10x faster on documents this size.
from lxml import etree as ET


_SVG_NS = "http://www.w3.org/2000/svg"


def _has_class(element: ET.Element, name: str) -> bool:
    return name in (element.get("class") or "").split()


class Axes:
    """Wraps an axis ``<g class="fig-axis">`` group.

    Methods mutate only that subtree.  Used for per-axis recolor when
    the SVG was authored with the class scheme; for unstructured SVG
    ``SvgFigure.get_axes()`` returns ``[]`` and the figure-level
    recolor passes do the work.
    """

    def __init__(self, group: ET.Element):
        self._g = group

    def _iter(self, class_name: str):
        for el in self._g.iter():
            if _has_class(el, class_name):
                yield el

    def set_title_color(self, color: str) -> None:
        for el in self._iter("fig-title"):
            el.set("fill", color)

    def set_label_color(self, color: str) -> None:
        for el in self._iter("fig-axis-label"):
            el.set("fill", color)

    def set_tick_color(self, color: str) -> None:
        for el in self._iter("fig-tick-label"):
            el.set("fill", color)

    def set_text_color(self, color: str) -> None:
        """Convenience: title + label + tick text in one call."""
        self.set_title_color(color)
        self.set_label_color(color)
        self.set_tick_color(color)

    def set_spine_color(self, color: str) -> None:
        for el in self._iter("fig-spine"):
            el.set("stroke", color)
        for el in self._iter("fig-tick"):
            el.set("stroke", color)


class SvgFigure:
    """SVG-backed Figure.

    Construct from an SVG string, bytes, or path::

        fig = SvgFigure(svg_str)
        fig = SvgFigure(Path('foo.svg'))

    The SVG payload is parsed into an ElementTree; all mutations
    (recolor, set_facecolor) operate on that tree.  Reads via
    ``.to_string()`` reflect the current state.
    """

    def __init__(self, payload, *, interactive: bool = True):
        if isinstance(payload, Path):
            text = payload.read_text()
        elif isinstance(payload, bytes):
            text = payload.decode("utf-8")
        elif isinstance(payload, str):
            if payload.lstrip().startswith("<"):
                text = payload  # SVG/XML content
            else:
                text = Path(payload).read_text()  # treat as path
        else:
            raise TypeError(
                f"SvgFigure expects str / bytes / Path (SVG payload); "
                f"got {type(payload).__name__}.  For matplotlib figures, "
                f"pass them directly to figs_to_deck without wrapping."
            )
        self._tree = ET.ElementTree(ET.fromstring(text))
        # SVG is resolution-independent; this is the render-to-raster default.
        self._dpi = 96
        # When True (default), ``_repr_mimebundle_`` wraps the SVG in an
        # HTML+CSS+JS shell that adds: copy + save action buttons (always),
        # hover-scale + click-to-zoom overlay (only when the SVG contains
        # ``<g class="fig-tile" data-bbox="...">`` groups). Set False for
        # plain SVG output (e.g., automated rasterization tests).
        self._interactive = bool(interactive)

    # ─── raw access ───────────────────────────────────────────────────
    @property
    def root(self) -> ET.Element:
        return self._tree.getroot()

    def to_string(self) -> str:
        return ET.tostring(self.root, encoding="unicode")

    # ─── geometry / metadata ──────────────────────────────────────────
    @property
    def dpi(self) -> int:
        return self._dpi

    def set_dpi(self, dpi: int) -> None:
        self._dpi = int(dpi)

    def _viewbox(self) -> tuple[float, float, float, float] | None:
        vb = self.root.get("viewBox")
        if vb is None:
            return None
        parts = vb.replace(",", " ").split()
        if len(parts) != 4:
            return None
        return tuple(float(p) for p in parts)  # type: ignore[return-value]

    def get_figwidth(self) -> float:
        # SVG user units → inches at 96 DPI is the convention.
        vb = self._viewbox()
        if vb is not None:
            return vb[2] / 96.0
        w = self.root.get("width")
        return float(re.sub(r"[^\d.]", "", w or "0")) / 96.0

    def get_figheight(self) -> float:
        vb = self._viewbox()
        if vb is not None:
            return vb[3] / 96.0
        h = self.root.get("height")
        return float(re.sub(r"[^\d.]", "", h or "0")) / 96.0

    @property
    def intrinsic_size_in(self) -> tuple[float, float]:
        return (self.get_figwidth(), self.get_figheight())

    # ─── background ───────────────────────────────────────────────────
    def get_facecolor(self) -> str:
        for child in list(self.root):
            tag = child.tag.split("}")[-1]
            if tag == "rect" and _has_class(child, "fig-background"):
                return child.get("fill") or "none"
        style = self.root.get("style") or ""
        m = re.search(r"background-color\s*:\s*([^;]+)", style)
        return (m.group(1).strip() if m else "none")

    def set_facecolor(self, color: str) -> None:
        for child in list(self.root):
            tag = child.tag.split("}")[-1]
            if tag == "rect" and _has_class(child, "fig-background"):
                child.set("fill", color)
                return
        vb = self._viewbox() or (0, 0, 0, 0)
        bg = ET.Element(f"{{{_SVG_NS}}}rect", {
            "class": "fig-background",
            "x": str(vb[0]), "y": str(vb[1]),
            "width": str(vb[2]), "height": str(vb[3]),
            "fill": color,
        })
        self.root.insert(0, bg)

    # ─── recolor: axis-level, figure-level, and the high-level call ────
    def get_axes(self) -> Sequence[Axes]:
        """Return per-axis wrappers for each ``<g class="fig-axis">``.

        Empty list for unstructured SVG (no class scheme); the figure-
        level recolor passes still handle those.
        """
        return [
            Axes(el) for el in self.root.iter()
            if el.tag.endswith("g") and _has_class(el, "fig-axis")
        ]

    def set_figure_text_color(self, color: str) -> None:
        """Walk every ``<text>`` / ``<tspan>`` and set its fill.

        Idempotent over the per-axis text walk, AND the natural
        fallback for unstructured SVG without ``fig-axis`` groups —
        every text element gets recolored regardless of class.
        """
        for el in self.root.iter():
            tag = el.tag.split("}")[-1]
            if tag in ("text", "tspan"):
                el.set("fill", color)

    def set_figure_spine_color(self, color: str) -> None:
        """Recolor root-level ``fig-spine`` / ``fig-tick`` elements.

        Composite SVGs (e.g. ``key_slices_svg``) put spine + tick
        elements at the root, not inside a ``fig-axis`` group, so
        ``get_axes()`` doesn't reach them.  This walk does.
        """
        for el in self.root.iter():
            classes = (el.get("class") or "").split()
            if "fig-spine" in classes or "fig-tick" in classes:
                el.set("stroke", color)

    def apply_color_scheme(self, *, font: str | None = None,
                            axes: str | None = None) -> "SvgFigure":
        """High-level recolor: text + spines, per-axis + figure-level.

        Returns ``self`` for chaining.  No-op for ``None`` arguments.
        Idempotent — calling twice with the same colors produces the
        same SVG.
        """
        if font is None and axes is None:
            return self
        for ax in self.get_axes():
            if font is not None:
                ax.set_text_color(font)
            if axes is not None:
                ax.set_spine_color(axes)
        if font is not None:
            self.set_figure_text_color(font)
        if axes is not None:
            self.set_figure_spine_color(axes)
        return self

    def _rasterizable_svg(self) -> bytes:
        """SVG bytes prepared for raster rendering (resvg).

        One normalization: transcode any embedded JPEG-XL data URLs
        to PNG (``key_slices_svg`` emits JXL tiles by default; resvg
        and most raster renderers can only decode standard PNG/JPEG).
        The in-memory tree and ``to_string()`` are unchanged — this
        only affects what the rasterizer sees.
        """
        s = self.to_string()
        s = _transcode_jxl_data_urls_to_png(s)
        return s.encode("utf-8")

    def _pptx_embeddable_svg(self) -> bytes:
        """SVG bytes prepared for native embedding inside a PPTX picture.

        Two normalizations on top of :meth:`_rasterizable_svg`:

        - Drop the responsive ``max-width:100%;height:auto`` style — it
          targets browsers/Jupyter and confuses PowerPoint's SVG layout
          (it ignores ``height:auto`` and the ``max-width`` constraint
          just adds noise to the parse).
        - Leave ``width`` / ``height`` in place; the PPTX exporter
          rewrites them to match the on-slide picture box so
          PowerPoint's convert-to-shapes uses the right physical scale
          for font-sizes (which are in SVG user units).
        """
        s = self.to_string()
        s = _transcode_jxl_data_urls_to_png(s)
        s = re.sub(r'(<svg\b[^>]*?)\s+style="[^"]*"', r'\1', s, count=1)
        return s.encode("utf-8")

    # ─── output ───────────────────────────────────────────────────────
    # SVG is the primary output.  Rasterizing to PNG/JPEG requires
    # ``resvg_py`` (optional install — ``pip install resvg-py``); it
    # is imported lazily so SVG-only workflows have no extra
    # dependency.
    def render_to_image(self, *, dpi: int | None = None) -> np.ndarray:
        png_bytes = _svg_to_png_bytes(self._rasterizable_svg(),
                                        dpi=dpi or self._dpi)
        import matplotlib.image as mpimg
        return mpimg.imread(BytesIO(png_bytes))

    def savefig(self, path, *, format: str | None = None,
                dpi: int | None = None) -> None:
        path = Path(path)
        format = (format or path.suffix.lstrip(".") or "svg").lower()
        if format == "svg":
            path.write_text(self.to_string())
            return
        if format not in ("png", "jpg", "jpeg"):
            raise ValueError(
                f"Unsupported format: {format!r}.  Supported: 'svg' "
                f"(default — no extra deps), 'png', 'jpg', 'jpeg' "
                f"(require ``pip install resvg-py``).  For PDF, convert "
                f"the .svg via rsvg-convert or Inkscape."
            )
        svg_bytes = self._rasterizable_svg()
        png_bytes = _svg_to_png_bytes(svg_bytes, dpi=dpi or self._dpi)
        if format == "png":
            Path(path).write_bytes(png_bytes)
        else:  # jpeg
            # resvg emits PNG only; transcode via matplotlib.
            import matplotlib.image as mpimg
            arr = mpimg.imread(BytesIO(png_bytes))
            mpimg.imsave(str(path), arr, format="jpeg")

    # ─── jupyter integration ──────────────────────────────────────────
    def _repr_mimebundle_(self, include=None, exclude=None):
        """Return the SVG payload (or HTML wrapping it).

        With ``interactive=True`` (default) the SVG is wrapped in a
        small HTML shell that adds:

        * **copy + save buttons** (bottom-right, fade-in on hover) for
          every SVG figure;
        * **hover-scale + click-to-zoom** for any ``<g class="fig-tile"
          data-bbox="x y w h">`` groups present in the SVG. Click a
          tile → fixed-position overlay showing just that tile (the
          overlay reuses the original SVG markup with ``viewBox`` set
          to the bbox — no raster duplication). Esc or click outside
          dismisses.

        Skipping the PNG fallback on purpose: it would force a
        rasterization round-trip on every cell display, which is
        slow and would make ``resvg`` a hard dependency for Jupyter
        display. Callers who actually want a PNG ask via
        ``render_to_image`` / ``savefig`` and handle the optional
        dep themselves.
        """
        svg_text = self.to_string()
        if not self._interactive:
            return {"image/svg+xml": svg_text}
        return {"text/html": _build_interactive_shell(svg_text)}


# Match base64 image data URLs inside SVG ``href`` / ``xlink:href`` attrs.
_DATA_URL_JXL_RE = re.compile(
    r'(?P<attr>(?:xlink:)?href)="data:image/jxl;base64,(?P<b64>[^"]+)"'
)


def _transcode_jxl_data_urls_to_png(svg_text: str) -> str:
    """Decode any embedded JXL data URLs and re-encode as PNG."""
    if "data:image/jxl;base64," not in svg_text:
        return svg_text
    from base64 import b64decode, b64encode
    try:
        import imagecodecs
    except ImportError:  # pragma: no cover
        return svg_text

    def _replace(m: re.Match) -> str:
        jxl_bytes = b64decode(m.group("b64"))
        try:
            arr = imagecodecs.jpegxl_decode(jxl_bytes)
            png_bytes = imagecodecs.png_encode(arr)
        except Exception:
            return m.group(0)  # leave the original href; let renderer complain
        png_b64 = b64encode(png_bytes).decode("ascii")
        return f'{m.group("attr")}="data:image/png;base64,{png_b64}"'

    return _DATA_URL_JXL_RE.sub(_replace, svg_text)


def _svg_to_png_bytes(svg_bytes: bytes, *, dpi: int = 96) -> bytes:
    """Rasterize SVG bytes to PNG bytes via ``resvg_py``.

    Lazy-imports ``resvg_py`` so SVG-only workflows (Jupyter display,
    .svg file save, native PowerPoint embed) don't take it as a hard
    dependency.  Raises with a helpful install hint if missing.
    """
    try:
        import resvg_py
    except ImportError as exc:
        raise ImportError(
            "Rasterizing SVG to PNG/JPEG requires the optional "
            "``resvg-py`` package.  Install with `pip install resvg-py`. "
            "SVG-only workflows (Jupyter display, .svg savefig, native "
            "SVG embedding in PowerPoint) don't need it."
        ) from exc
    out = resvg_py.svg_to_bytes(svg_string=svg_bytes.decode("utf-8"),
                                  dpi=dpi)
    # resvg_py returns either bytes or list[int] depending on version.
    if isinstance(out, list):
        out = bytes(out)
    return out


# ─── interactive HTML shell (Jupyter `_repr_mimebundle_` payload) ────


_SHELL_CSS = """
  .ocd-svgfig[data-uid="__UID__"] {
    position: relative;
    display: inline-block;
    max-width: 100%;
    /* Opt the subtree into both light + dark schemes so the
       ``light-dark()`` call below can resolve the active one. Without
       this, browsers assume ``light`` and dark-mode never fires. */
    color-scheme: light dark;
  }
  .ocd-svgfig[data-uid="__UID__"] > svg {
    display: block;
    max-width: 100%;
    height: auto;
    /* Anchor for SVG ``fill="currentColor"`` / ``stroke="currentColor"``
       (image_grid label + outline defaults). High-contrast adaptive:
       near-black on light backgrounds, near-white on dark, so labels
       stay readable when painted over arbitrary image content in
       Jupyter notebooks and embedded dashboards alike. Explicit
       ``fontcolor=...`` on ``image_grid`` still overrides via the SVG
       ``fill`` attribute (more specific than CSS). */
    color: light-dark(#1a1a1a, #f0f0f0);
  }
  .ocd-svgfig[data-uid="__UID__"] .fig-tile {
    cursor: zoom-in;
    transform-box: fill-box;
    transform-origin: center;
    transition: transform .12s ease;
  }
  .ocd-svgfig[data-uid="__UID__"] .fig-tile:hover {
    transform: scale(1.03);
  }
  /* Linked-axes mode: the cells are interactive pan/zoom viewports —
     disable the hover-scale + click-to-zoom cursor so the grab gesture
     reads cleanly. The JS controller skips the popup-zoom wiring too. */
  .ocd-svgfig[data-uid="__UID__"] svg[data-link-axes="1"] .fig-tile,
  .ocd-svgfig[data-uid="__UID__"] svg[data-link-axes="1"] .fig-tile:hover {
    cursor: grab;
    transform: none;
  }
  .ocd-svgfig[data-uid="__UID__"] svg[data-link-axes="1"] svg.ocd-linked-cell {
    cursor: grab;
  }
  /* Pointer events fire on the explicit transparent hit rect that lives
     in the OUTER svg coord system (sibling to the nested cell SVG).
     Placing it outside the nested SVG keeps its bbox stable across
     viewBox zoom/pan — putting it inside meant ``width="100%"`` was
     evaluated against the inner viewBox and only covered the top-left
     quadrant of the visible cell at most zooms. */
  .ocd-svgfig[data-uid="__UID__"] rect.ocd-linked-cell-hit {
    cursor: grab;
  }
  .ocd-svgfig[data-uid="__UID__"] rect.ocd-linked-cell-hit:active {
    cursor: grabbing;
  }
  /* Drag-pan must not start a text selection on the cell labels.
     ``user-select: none`` on the whole link-axes svg subtree
     suppresses the default selection behaviour. Re-enable on input/
     textarea descendants if any are ever embedded. */
  .ocd-svgfig[data-uid="__UID__"] svg[data-link-axes="1"],
  .ocd-svgfig[data-uid="__UID__"] svg[data-link-axes="1"] * {
    user-select: none;
    -webkit-user-select: none;
  }
  .ocd-svgfig[data-uid="__UID__"] .ocd-svgfig-actions {
    position: absolute;
    bottom: 4px; right: 4px;
    display: flex;
    gap: 8px;
    opacity: 0;
    transition: opacity .18s ease;
    pointer-events: none;
  }
  .ocd-svgfig[data-uid="__UID__"]:hover .ocd-svgfig-actions {
    opacity: 1;
    pointer-events: auto;
  }
  .ocd-svgfig[data-uid="__UID__"] .ocd-svgfig-actions button {
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    color: #808080;
    transition: transform .15s ease, color .15s ease;
  }
  .ocd-svgfig[data-uid="__UID__"] .ocd-svgfig-actions button:hover {
    transform: scale(1.2);
    color: var(--jp-ui-font-color1, #404040);
  }
  .ocd-svgfig[data-uid="__UID__"] .ocd-svgfig-actions button svg {
    width: 20px; height: 20px;
    fill: currentColor;
  }
  /* HDR toggle: when .ocd-sdr-mode is set on the wrapper (or on
     the overlay for popup zoom), clamp every <image>/<img>'s rendering
     to SDR via the CSS Color Module Level 4 dynamic-range-limit
     property. Browsers that support gain-map JPEGs (Safari 17.4+,
     Chrome 120+) honour this and skip the gain-map composition —
     identical to what a non-HDR-aware viewer would render. */
  .ocd-svgfig[data-uid="__UID__"].ocd-sdr-mode image,
  .ocd-svgfig[data-uid="__UID__"].ocd-sdr-mode img,
  .ocd-zoom-overlay[data-uid="__UID__"].ocd-sdr-mode image,
  .ocd-zoom-overlay[data-uid="__UID__"].ocd-sdr-mode img {
    dynamic-range-limit: standard;
  }
  .ocd-svgfig[data-uid="__UID__"] .ocd-hdrbtn.ocd-hdr-off {
    color: #c97a3a;  /* warm tint = SDR mode active */
  }
  .ocd-zoom-overlay[data-uid="__UID__"] {
    position: fixed;
    /* top/left/width/height set dynamically by JS to match the notebook
       pane's bbox so the overlay doesn't bleed under JupyterLab side
       panels. Default to viewport-cover for non-Jupyter hosts. */
    top: 0; left: 0; right: 0; bottom: 0;
    display: none;
    background: rgba(0, 0, 0, 0.85);
    z-index: 10000;
    cursor: zoom-out;
    /* Compositor isolation: tell the browser this subtree is layout/
       paint/style-self-contained, and force it onto its own composite
       layer.  Reduces re-composite cost on each canvas redraw — the
       browser no longer has to merge our overlay with the rest of the
       JupyterLab page on every frame.  Free perf bump for embedded
       hosts. */
    contain: strict;
    isolation: isolate;
    will-change: transform;
  }
  .ocd-zoom-overlay[data-uid="__UID__"].active { display: flex; }
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-inner {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    padding: 20px;
    box-sizing: border-box;
    min-width: 0;
    min-height: 0;
    /* Backdrop is the overlay; the inner (excluding canvas) lets clicks
       pass through so they hit the overlay's close handler. */
    pointer-events: none;
  }
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-title {
    color: #f5f5f5;
    font-family: var(--jp-ui-font-family, system-ui, sans-serif);
    font-size: 13px;
    text-align: center;
    margin: 0 0 8px 0;
    pointer-events: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-canvas {
    flex: 1 1 auto;
    position: relative;
    overflow: hidden;
    pointer-events: auto;
    /* touch-action:none disables the browser's own pan/pinch so our
       PointerEvent handlers get raw deltas — required for Safari
       multi-touch pinch zoom inside the popup. */
    touch-action: none;
    cursor: grab;
    min-height: 0;
  }
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-canvas.dragging {
    cursor: grabbing;
  }
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-fit {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transform-origin: 0 0;
    /* No will-change/translateZ here: those force a GPU layer that
       freezes the SVG raster at layout size and bilinear-composites
       the transform on top. We need the browser to RE-RASTERIZE the
       SVG on each transform change so image-rendering takes effect. */
  }
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-fit,
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-fit svg,
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-fit svg image {
    /* Nearest-neighbor on zoom. Cascade order matters: legacy values
       first, modern `pixelated` last so the modern path wins where it
       parses. Applied to the SVG, the inner <image>, AND the fit
       wrapper so each compositing pass sees the hint. */
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
    image-rendering: pixelated;
  }
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-fit svg {
    max-width: 100%;
    max-height: 100%;
    display: block;
    /* Route every pointer event to the canvas so it can do hit-testing
       against the SVG bbox itself (tap-outside-image → dismiss). With
       pointer-events:none on the SVG, the canvas always wins. */
    pointer-events: none;
    -webkit-user-select: none;
    user-select: none;
  }
  .ocd-zoom-overlay[data-uid="__UID__"] .ocd-zoom-fit svg image {
    -webkit-user-drag: none;
  }
""".strip()


_SHELL_SAVE_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640">'
    '<path fill="currentColor" d="M160 144C151.2 144 144 151.2 144 160L144 480'
    'C144 488.8 151.2 496 160 496L480 496C488.8 496 496 488.8 496 480L496 237.3'
    'C496 233.1 494.3 229 491.3 226L416 150.6L416 240C416 257.7 401.7 272 384'
    ' 272L224 272C206.3 272 192 257.7 192 240L192 144L160 144zM240 144L240 224'
    'L368 224L368 144L240 144zM96 160C96 124.7 124.7 96 160 96L402.7 96C419.7'
    ' 96 436 102.7 448 114.7L525.3 192C537.3 204 544 220.3 544 237.3L544 480'
    'C544 515.3 515.3 544 480 544L160 544C124.7 544 96 515.3 96 480L96 160z'
    'M256 384C256 348.7 284.7 320 320 320C355.3 320 384 348.7 384 384C384 419.3'
    ' 355.3 448 320 448C284.7 448 256 419.3 256 384z"/></svg>'
)

_SHELL_COPY_ICON = (
    '<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg" '
    'viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M4.75'
    ' 3A1.75 1.75 0 003 4.75v9.5c0 .966.784 1.75 1.75 1.75h1.5a.75.75 0 000'
    '-1.5h-1.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.25-.25h9.5a.25.25 0 01.25'
    '.25v1.5a.75.75 0 001.5 0v-1.5A1.75 1.75 0 0014.25 3h-9.5zm5 5A1.75 1.75'
    ' 0 008 9.75v9.5c0 .966.784 1.75 1.75 1.75h9.5A1.75 1.75 0 0021 19.25v-9.5'
    'A1.75 1.75 0 0019.25 8h-9.5zM9.5 9.75a.25.25 0 01.25-.25h9.5a.25.25 0 '
    '01.25.25v9.5a.25.25 0 01-.25.25h-9.5a.25.25 0 01-.25-.25v-9.5z"/></svg>'
)

# Stylised "HDR" badge: text-as-vector so it scales cleanly and follows
# currentColor. Toggle button shows this icon; clicking flips
# the wrapper's CSS class so embedded gain-map JPEGs render at SDR.
_SHELL_HDR_ICON = (
    '<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg" '
    'viewBox="0 0 28 16">'
    '<text x="14" y="13" text-anchor="middle" font-family="Helvetica, Arial, '
    'sans-serif" font-weight="700" font-size="12" fill="currentColor">HDR</text>'
    '</svg>'
)


_SHELL_JS = r"""
  (function() {
    const wrapper = document.querySelector('.ocd-svgfig[data-uid="__UID__"]');
    if (!wrapper) return;
    const svg = wrapper.querySelector('svg');
    const tiles = wrapper.querySelectorAll('.fig-tile');
    const overlay = document.querySelector('.ocd-zoom-overlay[data-uid="__UID__"]');
    const overlayInner = overlay && overlay.querySelector('.ocd-zoom-inner');
    // Remember where the overlay started so we can restore it on close.
    const overlayHome = overlay && overlay.parentElement;
    const xmlns = 'http://www.w3.org/2000/svg';

    // Confine the overlay to the notebook content pane so it doesn't
    // bleed under JupyterLab side panels / status bar. We prefer
    // JupyterLab's #jp-main-content-panel (excludes the bottom status
    // bar by construction); fall back to viewport-minus-status-bar.
    //
    // Re-resolves on each ``openZoom`` rather than once at IIFE time,
    // since the wrapper's containing notebook panel can be reattached
    // (e.g. JupyterLab tab moves between split panes) — the IIFE-time
    // pane reference would point at a detached node after that.
    function resolvePane() {
      // Prefer the LARGEST sensible notebook container so the dim
      // backdrop covers the whole notebook (toolbar + scrollable cell
      // area + footer), not just the inner scroll region.  ``closest``
      // returns the nearest matching ancestor, so order matters: try
      // outer-most class names first, narrowing down as fallbacks.
      // ``.jp-NotebookPanel`` is the outer container (includes the
      // cell toolbar); ``.jp-NotebookPanel-notebook`` is the inner
      // scrollable area only.
      return wrapper.closest('.jp-MainAreaWidget')
          || wrapper.closest('.jp-NotebookPanel')
          || wrapper.closest('.jp-NotebookPanel-notebook')
          || wrapper.closest('.jp-Notebook')
          || wrapper.closest('.jp-Cell')
          || document.body;
    }
    let pane = resolvePane();
    function syncOverlayToPane() {
      if (!overlay) return;
      let topLimit = 0;
      let bottomLimit = window.innerHeight;
      let leftLimit = 0;
      let rightLimit = window.innerWidth;
      const mainPanel = document.querySelector('#jp-main-content-panel');
      if (mainPanel) {
        const m = mainPanel.getBoundingClientRect();
        topLimit = Math.max(topLimit, m.top);
        bottomLimit = Math.min(bottomLimit, m.bottom);
        leftLimit = Math.max(leftLimit, m.left);
        rightLimit = Math.min(rightLimit, m.right);
      } else {
        const statusBar = document.querySelector('.jp-StatusBar');
        if (statusBar) {
          bottomLimit = Math.min(
            bottomLimit, statusBar.getBoundingClientRect().top);
        }
      }
      // No inset margin — extend the dim backdrop all the way to the
      // notebook pane's edges (clamped to the JupyterLab main content
      // area so it doesn't bleed under side panels / status bar).
      const rect = pane.getBoundingClientRect();
      const top = Math.max(rect.top, topLimit);
      const left = Math.max(rect.left, leftLimit);
      const right = Math.min(rect.right, rightLimit);
      const bottom = Math.min(rect.bottom, bottomLimit);
      overlay.style.top = top + 'px';
      overlay.style.left = left + 'px';
      overlay.style.width = Math.max(0, right - left) + 'px';
      overlay.style.height = Math.max(0, bottom - top) + 'px';
      overlay.style.right = 'auto';
      overlay.style.bottom = 'auto';
    }

    // Zoom/pan state for the currently-open tile.  Same semantics as
    // before: s=1 means "image fits within canvas"; (tx, ty) are CSS
    // pixel translations on top.  But now applied via shader uniforms
    // on a WebGL canvas instead of a CSS transform on an SVG wrapper —
    // matches the ocdkit.viewer (pywebgui) approach.  Per-fragment NN
    // texture sampling avoids the browser's SVG <image> double-resample
    // (which produced moire on downscale even with image-rendering hints
    // set) and removes the awkward auto↔pixelated threshold switch.
    // s = zoom, tx/ty = translation in canvas CSS pixels, r = rotation
    // in radians around the displayed image center. r is honored only
    // by the CSS-img viewer (HDR path) -- the WebGL viewers' vertex
    // shaders don't carry a rotation term, so r is silently ignored
    // there. Trackpad rotation is Safari-only at the browser level
    // (Chrome doesn't expose gesture events for trackpad rotate);
    // 2-finger touch rotation works in any browser via PointerEvent.
    const state = { s: 1, tx: 0, ty: 0, r: 0 };
    let canvasEl = null;       // <div class="ocd-zoom-canvas"> wrapper
    let webglViewer = null;    // see createPopupWebglViewer
    // s=1 is "image fits canvas"; s<1 zooms out beyond fit (image
    // smaller than the canvas, useful for getting full context on a
    // huge image), s>1 zooms in past 1:1.
    const MIN_S = 0.1;
    const MAX_S = 20;
    // Redraw + animation scheduling.
    //
    // Two concerns here:
    //   1. Input timing is irregular — Chrome may deliver wheel events
    //      at 30 Hz one moment and 120 Hz the next, and big mouse-wheel
    //      notches arrive as single events with large deltaY.  If we
    //      apply each event directly to the visible state, the user
    //      sees discrete jumps.
    //   2. WebGL draws faster than vsync get dropped — the browser
    //      only composites at vsync, so issuing 10 draws per frame
    //      means 9 of them get thrown away and the displayed motion
    //      doesn't match the input cadence.
    //
    // Fix: input updates a *target* state.  An rAF loop tweens the
    // *displayed* state toward the target with an ease-out curve, one
    // redraw per frame.  Smooth at 60/120 Hz regardless of input rate,
    // and never wastes a draw.
    const target = { s: 1, tx: 0, ty: 0, r: 0 };
    const TWEEN_ALPHA = 0.30;        // fraction of remaining distance per frame
    const TWEEN_EPS_S = 0.0005;
    const TWEEN_EPS_T = 0.4;          // pixels
    let _tweenRaf = 0;
    function startTween() {
      if (_tweenRaf || !webglViewer) return;
      const tick = () => {
        const ds = target.s - state.s;
        const dtx = target.tx - state.tx;
        const dty = target.ty - state.ty;
        if (Math.abs(ds) < TWEEN_EPS_S
            && Math.abs(dtx) < TWEEN_EPS_T
            && Math.abs(dty) < TWEEN_EPS_T) {
          // Snap to target and stop the loop.
          state.s = target.s;
          state.tx = target.tx;
          state.ty = target.ty;
          _tweenRaf = 0;
        } else {
          state.s += ds * TWEEN_ALPHA;
          state.tx += dtx * TWEEN_ALPHA;
          state.ty += dty * TWEEN_ALPHA;
          _tweenRaf = requestAnimationFrame(tick);
        }
        if (webglViewer) webglViewer.redraw(state);
      };
      _tweenRaf = requestAnimationFrame(tick);
    }
    // For one-shot changes that should jump (no animation): set state
    // and target to the same value, then redraw once.
    function applyTransform() {
      target.s = state.s; target.tx = state.tx; target.ty = state.ty;
      if (_tweenRaf) { cancelAnimationFrame(_tweenRaf); _tweenRaf = 0; }
      if (webglViewer) {
        requestAnimationFrame(() => webglViewer && webglViewer.redraw(state));
      }
    }
    function resetTransform() {
      if (webglViewer && webglViewer.isWorker) {
        webglViewer.reset();
        return;
      }
      state.s = 1; state.tx = 0; state.ty = 0; state.r = 0;
      target.s = 1; target.tx = 0; target.ty = 0; target.r = 0;
      if (_tweenRaf) { cancelAnimationFrame(_tweenRaf); _tweenRaf = 0; }
      applyTransform();
    }
    // zoomAboutTarget — in worker mode the worker owns the tween +
    // target; just forward.  In in-thread mode, mutate the local
    // target and kick the tween loop.
    function zoomAboutTarget(px, py, ratio) {
      if (webglViewer && webglViewer.isWorker) {
        webglViewer.applyZoomAboutTarget(px, py, ratio);
        return;
      }
      const newS = Math.max(MIN_S, Math.min(MAX_S, target.s * ratio));
      const actualRatio = newS / target.s;
      target.tx = px * (1 - actualRatio) + target.tx * actualRatio;
      target.ty = py * (1 - actualRatio) + target.ty * actualRatio;
      target.s = newS;
      startTween();
    }
    // No-op: the SVG/CSS path used ``will-change: transform`` to coax
    // the browser onto a GPU compositing layer during gestures.  WebGL
    // already lives on a GPU-composited canvas, and uniform updates +
    // a draw call are the fast path — no layer juggling needed.
    function setGestureActive(_active) {}

    // ─── Worker-thread WebGL popup viewer ─────────────────────────────
    // Renders on a Web Worker via OffscreenCanvas.  Main thread is
    // responsible only for: receiving DOM input, forwarding it to the
    // worker, and consuming back state updates for ``isPointInImage``.
    // Worker owns: WebGL context, image texture, tween rAF, draws.
    //
    // Why: inside JupyterLab the main thread is loaded with notebook /
    // widget / Comm work that competes with our rAF and WebGL
    // submission, producing visible stutter.  A worker thread has its
    // own event loop tied to display vsync, immune to main-thread
    // busyness — animation stays smooth regardless of host page load.
    //
    // Falls back to ``createPopupWebglViewer`` when OffscreenCanvas
    // isn't available (e.g. Safari pre-16.4).
    const POPUP_WORKER_SOURCE = `
      let gl = null;
      let canvas = null;
      let program = null;
      let placeholderTex = null;     // 1x1 transparent, bound at startup
      let currentTex = null;         // texture currently bound for draw()
      let imgW = 1, imgH = 1, textureLoaded = false;
      // LRU of decoded textures keyed by source URL. Switching tiles
      // in a recycled-worker popup is a Map lookup + texture rebind --
      // no refetch, no JXL re-decode, no GPU re-upload.
      //
      // Eviction is byte-budgeted so the same cap works for grids of
      // small thumbs and grids of huge hi-res tiles. 2 GB ceiling is
      // permissive enough that typical scientific-imaging workflows (25-50 FOV
      // grids, up to ~4K tiles each) never evict on a desktop GPU;
      // drivers will signal GL_OUT_OF_MEMORY long before we hit it.
      // The currently-displayed texture is never evicted regardless.
      const textureLRU = new Map();
      const TEXTURE_LRU_BYTES_MAX = 2 * 1024 * 1024 * 1024;
      let textureLRUBytes = 0;
      const state = { s: 1, tx: 0, ty: 0 };
      const target = { s: 1, tx: 0, ty: 0 };
      const MIN_S = 0.1, MAX_S = 20;
      const TWEEN_ALPHA = 0.30;
      const TWEEN_EPS_S = 0.0005;
      const TWEEN_EPS_T = 0.4;
      let tweenRaf = 0;
      let canvasSize = { w: 1, h: 1, dpr: 1 };
      let U = {};

      const VS = \`#version 300 es
in vec2 a_pos;
out vec2 v_uv;
uniform vec2 u_canvasSizePx;
uniform vec2 u_imageSizePx;
uniform float u_dpr;
uniform float u_zoom;
uniform vec2 u_translatePx;
void main() {
  vec2 canvasCSS = u_canvasSizePx / u_dpr;
  float fitScale = min(canvasCSS.x / u_imageSizePx.x,
                       canvasCSS.y / u_imageSizePx.y);
  vec2 imageFitHalf = u_imageSizePx * (fitScale * 0.5);
  vec2 canvasCenter = canvasCSS * 0.5;
  vec2 preXfmTopLeft = canvasCenter - imageFitHalf;
  vec2 imageOriginCSS = preXfmTopLeft * u_zoom + u_translatePx;
  vec2 imageSizeCSS = u_imageSizePx * (fitScale * u_zoom);
  vec2 quadCSS = imageOriginCSS + a_pos * imageSizeCSS;
  vec2 clip = (quadCSS / canvasCSS) * 2.0 - 1.0;
  clip.y = -clip.y;
  gl_Position = vec4(clip, 0.0, 1.0);
  v_uv = a_pos;
}\`;
      const FS = \`#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_tex;
void main() {
  outColor = texture(u_tex, v_uv, 0.5);
}\`;

      function compile(type, src) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
          self.postMessage({ type: 'error',
            msg: 'shader compile failed: ' + gl.getShaderInfoLog(sh) });
          return null;
        }
        return sh;
      }

      function initGL(off) {
        canvas = off;
        gl = canvas.getContext('webgl2', {
          antialias: false, alpha: true,
          premultipliedAlpha: true, preserveDrawingBuffer: true,
        });
        if (!gl) {
          self.postMessage({ type: 'unsupported' });
          return;
        }
        const vs = compile(gl.VERTEX_SHADER, VS);
        const fs = compile(gl.FRAGMENT_SHADER, FS);
        program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
          self.postMessage({ type: 'error',
            msg: 'link failed: ' + gl.getProgramInfoLog(program) });
          return;
        }
        gl.useProgram(program);

        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER,
          new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
        const aPos = gl.getAttribLocation(program, 'a_pos');
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

        U = {
          tex: gl.getUniformLocation(program, 'u_tex'),
          canvasSizePx: gl.getUniformLocation(program, 'u_canvasSizePx'),
          imageSizePx: gl.getUniformLocation(program, 'u_imageSizePx'),
          dpr: gl.getUniformLocation(program, 'u_dpr'),
          zoom: gl.getUniformLocation(program, 'u_zoom'),
          translatePx: gl.getUniformLocation(program, 'u_translatePx'),
        };
        gl.uniform1i(U.tex, 0);

        const anisoExt = gl.getExtension('EXT_texture_filter_anisotropic');
        const maxAniso = anisoExt
          ? gl.getParameter(anisoExt.MAX_TEXTURE_MAX_ANISOTROPY_EXT) : 0;

        // Save anisotropy support so per-tile textures created later
        // can apply the same filtering. (LRU entries each have their
        // own GL texture object, configured the same way.)
        self._anisoMax = (anisoExt && maxAniso > 1)
          ? Math.min(maxAniso, 16) : 0;
        self._anisoExt = anisoExt;

        placeholderTex = makeTexture_();
        gl.bindTexture(gl.TEXTURE_2D, placeholderTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0,
                       gl.RGBA, gl.UNSIGNED_BYTE,
                       new Uint8Array([0, 0, 0, 0]));
        currentTex = placeholderTex;
        self.postMessage({ type: 'ready' });
      }

      function makeTexture_() {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
                          gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        if (self._anisoExt && self._anisoMax > 1) {
          gl.texParameterf(gl.TEXTURE_2D,
            self._anisoExt.TEXTURE_MAX_ANISOTROPY_EXT, self._anisoMax);
        }
        return t;
      }

      function applyCanvasSize(w, h, dpr) {
        canvasSize = { w, h, dpr };
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        gl.viewport(0, 0, w, h);
        gl.uniform2f(U.canvasSizePx, w, h);
        gl.uniform1f(U.dpr, dpr);
      }

      function draw() {
        // Rebind on every draw -- cheap (driver no-ops if already
        // bound) and required because tile-switch via the LRU just
        // updates currentTex, not the active binding.
        if (currentTex) gl.bindTexture(gl.TEXTURE_2D, currentTex);
        gl.uniform2f(U.imageSizePx, imgW, imgH);
        gl.uniform1f(U.zoom, state.s);
        gl.uniform2f(U.translatePx, state.tx, state.ty);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }

      function setActiveTexture_(url, entry) {
        // Move to MRU position (delete + reinsert so insertion order
        // tracks recency).
        textureLRU.delete(url);
        textureLRU.set(url, entry);
        currentTex = entry.tex;
        imgW = entry.w; imgH = entry.h;
        textureLoaded = true;
      }

      function evictIfFull_() {
        // Walk insertion order (= LRU order) and drop the oldest until
        // we fit the byte budget. Skip the active texture so we never
        // pull the rug out from under the visible draw. With ~2 GL
        // mipmap overhead multiplier, a single 4K-RGBA8 texture costs
        // ~85 MB GPU memory; the 256 MB budget keeps the working set
        // bounded while allowing 3+ such tiles to stay hot.
        if (textureLRUBytes <= TEXTURE_LRU_BYTES_MAX) return;
        const keys = Array.from(textureLRU.keys());
        for (const k of keys) {
          if (textureLRUBytes <= TEXTURE_LRU_BYTES_MAX) break;
          const ev = textureLRU.get(k);
          if (!ev || ev.tex === currentTex) continue;
          textureLRU.delete(k);
          try { gl.deleteTexture(ev.tex); } catch (_) {}
          textureLRUBytes -= (ev.bytes || 0);
        }
      }

      function tweenTick() {
        const ds = target.s - state.s;
        const dtx = target.tx - state.tx;
        const dty = target.ty - state.ty;
        if (Math.abs(ds) < TWEEN_EPS_S
            && Math.abs(dtx) < TWEEN_EPS_T
            && Math.abs(dty) < TWEEN_EPS_T) {
          state.s = target.s; state.tx = target.tx; state.ty = target.ty;
          tweenRaf = 0;
          self.postMessage({ type: 'stateUpdate',
            s: state.s, tx: state.tx, ty: state.ty, settled: true });
        } else {
          state.s += ds * TWEEN_ALPHA;
          state.tx += dtx * TWEEN_ALPHA;
          state.ty += dty * TWEEN_ALPHA;
          tweenRaf = requestAnimationFrame(tweenTick);
        }
        draw();
      }
      function startTween() {
        if (tweenRaf) return;
        tweenRaf = requestAnimationFrame(tweenTick);
      }

      function zoomAboutTarget(px, py, ratio) {
        const newS = Math.max(MIN_S, Math.min(MAX_S, target.s * ratio));
        const actualRatio = newS / target.s;
        target.tx = px * (1 - actualRatio) + target.tx * actualRatio;
        target.ty = py * (1 - actualRatio) + target.ty * actualRatio;
        target.s = newS;
        startTween();
      }

      async function loadImageBlobUrl(url, isThumb) {
        // LRU hit: switch active texture, no fetch/decode/upload.
        // This is the path that makes tile-to-tile revisits flash-free.
        const cached = textureLRU.get(url);
        if (cached) {
          setActiveTexture_(url, cached);
          self.postMessage({ type: 'imageLoaded',
            imgW, imgH, isThumb, cached: true });
          startTween();
          return;
        }
        try {
          const resp = await fetch(url);
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          const blob = await resp.blob();
          const bmp = await createImageBitmap(blob);
          // A newer load for the same URL may have raced ahead while
          // we awaited the network — re-check the cache before
          // creating a redundant texture.
          const racedIn = textureLRU.get(url);
          if (racedIn) {
            setActiveTexture_(url, racedIn);
            bmp.close && bmp.close();
            self.postMessage({ type: 'imageLoaded',
              imgW, imgH, isThumb, cached: true });
            startTween();
            return;
          }
          const tex = makeTexture_();
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA,
                         gl.UNSIGNED_BYTE, bmp);
          gl.generateMipmap(gl.TEXTURE_2D);
          // Snapshot the bitmap dimensions BEFORE close(). Per spec
          // close() neuters the ImageBitmap and width/height read 0
          // afterwards — if the entry pulls those zeros, imgW/imgH in
          // the shader collapse fitScale to a degenerate quad and the
          // popup renders a fully transparent canvas (visible symptom:
          // dim backdrop, no image).
          // ~4 bytes/pixel RGBA8 + 1/3 mipmap chain = 16/3 bytes/pixel
          // for the byte-budget accounting in evictIfFull_().
          const bytes = Math.round(bmp.width * bmp.height * 16 / 3);
          const entry = { tex, w: bmp.width, h: bmp.height, bytes };
          bmp.close && bmp.close();
          textureLRU.set(url, entry);
          textureLRUBytes += bytes;
          setActiveTexture_(url, entry);
          evictIfFull_();
          self.postMessage({ type: 'imageLoaded', imgW, imgH, isThumb });
          startTween();
        } catch (e) {
          self.postMessage({ type: 'imageError',
            url, msg: String(e && e.message || e) });
        }
      }

      self.onmessage = (e) => {
        const d = e.data;
        switch (d.type) {
          case 'init':
            initGL(d.canvas);
            break;
          case 'size':
            if (gl) {
              applyCanvasSize(d.w, d.h, d.dpr);
              draw();
            }
            break;
          case 'loadImage':
            if (gl) loadImageBlobUrl(d.url, !!d.isThumb);
            break;
          case 'snapState':
            // Caller asserts the visible state should be at these
            // values right now (pan / pinch path).
            state.s = d.s; state.tx = d.tx; state.ty = d.ty;
            target.s = d.s; target.tx = d.tx; target.ty = d.ty;
            if (tweenRaf) {
              cancelAnimationFrame(tweenRaf);
              tweenRaf = 0;
            }
            if (gl) draw();
            break;
          case 'zoomAboutTarget':
            zoomAboutTarget(d.px, d.py, d.ratio);
            break;
          case 'reset':
            target.s = 1; target.tx = 0; target.ty = 0;
            startTween();
            break;
          case 'clearActive':
            // Switch active texture back to the 1x1 transparent
            // placeholder so the next draw shows nothing instead of
            // the previous tile. Called by main thread on openZoom
            // tile switch to prevent a stale-content flash before
            // the new tile's loadImage completes.
            if (gl && placeholderTex) {
              currentTex = placeholderTex;
              imgW = 1; imgH = 1;
              textureLoaded = false;
              startTween();
            }
            break;
          case 'dispose':
            if (tweenRaf) cancelAnimationFrame(tweenRaf);
            try { gl && gl.getExtension('WEBGL_lose_context')
                       && gl.getExtension('WEBGL_lose_context').loseContext(); } catch (_) {}
            self.close();
            break;
        }
      };
    `;

    function createPopupWorkerViewer(parent) {
      // Feature gate: OffscreenCanvas + transferControlToOffscreen +
      // workers w/ structured-clone of OffscreenCanvas.
      if (typeof Worker === 'undefined'
          || typeof OffscreenCanvas === 'undefined') {
        return null;
      }
      const canvas = document.createElement('canvas');
      canvas.style.display = 'block';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.touchAction = 'none';
      if (!canvas.transferControlToOffscreen) return null;
      parent.appendChild(canvas);
      let off;
      try {
        off = canvas.transferControlToOffscreen();
      } catch (_) {
        parent.removeChild(canvas);
        return null;
      }
      const blob = new Blob([POPUP_WORKER_SOURCE],
                             { type: 'application/javascript' });
      const url = URL.createObjectURL(blob);
      let worker;
      try {
        worker = new Worker(url);
      } catch (_) {
        URL.revokeObjectURL(url);
        parent.removeChild(canvas);
        return null;
      }
      URL.revokeObjectURL(url);  // worker holds reference; URL can be freed

      let imgW = 1, imgH = 1;
      let stateMirror = { s: 1, tx: 0, ty: 0 };  // last reported visible state
      let targetMirror = { s: 1, tx: 0, ty: 0 };  // what we've asked worker for
      let onTextureLoadedFns = [];

      worker.addEventListener('message', (e) => {
        const d = e.data;
        if (d.type === 'stateUpdate') {
          stateMirror.s = d.s; stateMirror.tx = d.tx; stateMirror.ty = d.ty;
        } else if (d.type === 'imageLoaded') {
          imgW = d.imgW; imgH = d.imgH;
          const fns = onTextureLoadedFns; onTextureLoadedFns = [];
          for (const fn of fns) try { fn(d.isThumb); } catch (_) {}
        } else if (d.type === 'error' || d.type === 'imageError') {
          console.warn('SvgFigure worker:', d.msg, d.url || '');
        }
      });

      worker.postMessage({ type: 'init', canvas: off }, [off]);

      function pushSize() {
        const dpr = window.devicePixelRatio || 1;
        const cssW = canvas.clientWidth;
        const cssH = canvas.clientHeight;
        const w = Math.max(1, Math.round(cssW * dpr));
        const h = Math.max(1, Math.round(cssH * dpr));
        worker.postMessage({ type: 'size', w, h, dpr });
      }
      let _sizeDirty = true;
      canvas.__invalidateSize = () => { _sizeDirty = true; };

      function redraw(s) {
        // Worker owns the tween; main only needs to push size updates
        // when canvas CSS dims change.
        if (_sizeDirty) {
          pushSize();
          _sizeDirty = false;
        }
      }

      function loadImage(url, onLoaded) {
        if (onLoaded) onTextureLoadedFns.push(() => onLoaded());
        worker.postMessage({ type: 'loadImage', url });
      }

      function isPointInImage(clientX, clientY) {
        const r = canvas.getBoundingClientRect();
        const localX = clientX - r.left;
        const localY = clientY - r.top;
        if (localX < 0 || localY < 0 || localX > r.width || localY > r.height) {
          return false;
        }
        // Use the most recent state reported by the worker.
        const s = stateMirror;
        const fitScale = Math.min(r.width / imgW, r.height / imgH);
        const halfW = imgW * fitScale * 0.5;
        const halfH = imgH * fitScale * 0.5;
        const cx = r.width * 0.5;
        const cy = r.height * 0.5;
        const imgLeft = (cx - halfW) * s.s + s.tx;
        const imgTop = (cy - halfH) * s.s + s.ty;
        const dispW = imgW * fitScale * s.s;
        const dispH = imgH * fitScale * s.s;
        return localX >= imgLeft && localX <= imgLeft + dispW
            && localY >= imgTop && localY <= imgTop + dispH;
      }

      function dispose() {
        try { worker.postMessage({ type: 'dispose' }); } catch (_) {}
        try { worker.terminate(); } catch (_) {}
        if (canvas.parentElement) canvas.parentElement.removeChild(canvas);
      }

      const viewer = {
        canvas,
        // Intercept gesture-target updates: instead of mutating
        // local-thread ``target``, post the operation to the worker
        // which runs the tween + draws.  Main thread keeps a mirror
        // for ``isPointInImage`` and for pinch math.
        applyZoomAboutTarget(px, py, ratio) {
          // Compute new targetMirror so main has it too (for pinch).
          const newS = Math.max(MIN_S, Math.min(MAX_S, targetMirror.s * ratio));
          const actualRatio = newS / targetMirror.s;
          targetMirror.tx = px * (1 - actualRatio) + targetMirror.tx * actualRatio;
          targetMirror.ty = py * (1 - actualRatio) + targetMirror.ty * actualRatio;
          targetMirror.s = newS;
          worker.postMessage({ type: 'zoomAboutTarget', px, py, ratio });
        },
        applySnapState(s, tx, ty) {
          stateMirror.s = s; stateMirror.tx = tx; stateMirror.ty = ty;
          targetMirror.s = s; targetMirror.tx = tx; targetMirror.ty = ty;
          worker.postMessage({ type: 'snapState', s, tx, ty });
        },
        reset() {
          targetMirror.s = 1; targetMirror.tx = 0; targetMirror.ty = 0;
          worker.postMessage({ type: 'reset' });
        },
        clearActive() {
          // Switch worker's active texture back to placeholder so the
          // canvas doesn't show the previous tile while the new tile's
          // loadImage is in flight.
          imgW = 1; imgH = 1;
          worker.postMessage({ type: 'clearActive' });
        },
        get stateMirror() { return stateMirror; },
        get targetMirror() { return targetMirror; },
        redraw, loadImage, isPointInImage, dispose,
        get textureLoaded() { return imgW > 1; },
        isWorker: true,
      };
      return viewer;
    }

    // ─── In-thread WebGL popup viewer (fallback) ──────────────────────
    // One instance per openZoom; discarded on closeZoom.  Same shader
    // + filter setup as the worker viewer, but state and tween live on
    // the main thread.  Used when OffscreenCanvas isn't available.
    function createPopupWebglViewer(parent) {
      const canvas = document.createElement('canvas');
      canvas.style.display = 'block';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.touchAction = 'none';
      parent.appendChild(canvas);
      let gl = null;
      try {
        gl = canvas.getContext('webgl2', {
          antialias: false, alpha: true,
          premultipliedAlpha: true, preserveDrawingBuffer: true,
        });
      } catch (_) {}
      if (!gl) {
        parent.removeChild(canvas);
        return null;
      }
      // Vertex shader computes the image quad's clip-space position
      // from the same CSS-transform model the legacy SVG path uses
      // (image pre-transform sits at canvasCenter - imageFitHalf,
      // then CSS matrix(s,0,0,s,tx,ty) with origin at fit-top-left).
      // Doing the math per-vertex (4 verts) instead of per-fragment
      // (millions) — and clipping naturally via gl_Position — means
      // we only rasterize the image region, no branch in the fragment
      // shader.  Cuts wheel-zoom stutter noticeably on big canvases.
      const VS = `#version 300 es
in vec2 a_pos;            // unit quad: (0,0) (1,0) (0,1) (1,1)
out vec2 v_uv;
uniform vec2 u_canvasSizePx;
uniform vec2 u_imageSizePx;
uniform float u_dpr;
uniform float u_zoom;
uniform vec2 u_translatePx;
void main() {
  vec2 canvasCSS = u_canvasSizePx / u_dpr;
  float fitScale = min(canvasCSS.x / u_imageSizePx.x,
                       canvasCSS.y / u_imageSizePx.y);
  vec2 imageFitHalf = u_imageSizePx * (fitScale * 0.5);
  vec2 canvasCenter = canvasCSS * 0.5;
  vec2 preXfmTopLeft = canvasCenter - imageFitHalf;
  vec2 imageOriginCSS = preXfmTopLeft * u_zoom + u_translatePx;
  vec2 imageSizeCSS = u_imageSizePx * (fitScale * u_zoom);
  vec2 quadCSS = imageOriginCSS + a_pos * imageSizeCSS;
  // CSS coords (top-left origin) → clip space [-1, +1] (bottom-left origin)
  vec2 clip = (quadCSS / canvasCSS) * 2.0 - 1.0;
  clip.y = -clip.y;
  gl_Position = vec4(clip, 0.0, 1.0);
  v_uv = a_pos;
}`;
      // Fragment shader is a single texture lookup.  The sampler's
      // MIN/MAG_FILTER settings (configured on the texture below) do
      // the actual scaling work — LINEAR_MIPMAP_LINEAR for minification
      // (smooth, mipmap-trilinear downscale, no moire on high-frequency
      // content) and NEAREST for magnification (crisp NN upscale, the
      // pixel-art look the user wants when zoomed in past 1:1).
      // Hardware picks min vs mag automatically per fragment from the
      // UV derivatives — no JS-side threshold, no jarring switch.
      // Fragment shader: one texture lookup with a small LOD bias.
      // The bias only affects the MIN (downscale) regime — hardware
      // still picks MAG (NEAREST) per fragment at upscale, so the
      // zoomed-in pixel-art crispness is untouched.
      //
      // Without bias, LINEAR_MIPMAP_LINEAR at display ratios just below
      // 1:1 blends mostly level 0 + a sliver of level 1.  Level 0 still
      // carries the source's high-frequency content (1-pixel patterns,
      // etc.), so bilinear within it produces classic moire — different
      // fragments align to different texel grids and pick up biased
      // mixtures.  A +0.5 LOD bias shifts that blend toward level 1
      // (which is the area-averaged downsample, already moire-free).
      // Trade: slight softness at near-1:1 zoom on smooth content;
      // worth it to kill moire on busy patterns.
      const FS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_tex;
void main() {
  outColor = texture(u_tex, v_uv, 0.5);
}`;
      function compile(type, src) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
          console.warn('SvgFigure WebGL shader compile failed:',
                       gl.getShaderInfoLog(sh));
          gl.deleteShader(sh);
          return null;
        }
        return sh;
      }
      const vs = compile(gl.VERTEX_SHADER, VS);
      const fs = compile(gl.FRAGMENT_SHADER, FS);
      const program = gl.createProgram();
      if (!vs || !fs) {
        parent.removeChild(canvas);
        return null;
      }
      gl.attachShader(program, vs);
      gl.attachShader(program, fs);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.warn('SvgFigure WebGL link failed:',
                     gl.getProgramInfoLog(program));
        parent.removeChild(canvas);
        return null;
      }
      gl.useProgram(program);

      // Unit quad (0..1) — vertex shader expands to clip space using
      // the image's CSS-pixel bounds at the current zoom/translate.
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER,
        new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      const aPos = gl.getAttribLocation(program, 'a_pos');
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      const U = {
        tex: gl.getUniformLocation(program, 'u_tex'),
        canvasSizePx: gl.getUniformLocation(program, 'u_canvasSizePx'),
        imageSizePx: gl.getUniformLocation(program, 'u_imageSizePx'),
        dpr: gl.getUniformLocation(program, 'u_dpr'),
        zoom: gl.getUniformLocation(program, 'u_zoom'),
        translatePx: gl.getUniformLocation(program, 'u_translatePx'),
      };
      gl.uniform1i(U.tex, 0);

      // Probe anisotropic filtering — extra-cheap quality bump at
      // oblique downscale (not crucial but visible on diagonal lines).
      const anisoExt = gl.getExtension('EXT_texture_filter_anisotropic')
                    || gl.getExtension('WEBKIT_EXT_texture_filter_anisotropic');
      const maxAniso = anisoExt
        ? gl.getParameter(anisoExt.MAX_TEXTURE_MAX_ANISOTROPY_EXT) : 0;

      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      // MIN: trilinear mipmap filtering — smooth zoom-out, no moire on
      //      high-frequency content.  Hardware picks the right mip level
      //      automatically based on UV derivatives.
      // MAG: NEAREST — crisp pixel-art look when zoomed in past 1:1.
      //      Hardware picks MIN-vs-MAG per fragment, so we get NN at
      //      upscale and trilinear at downscale with no JS toggle.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
                        gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (anisoExt && maxAniso > 1) {
        gl.texParameterf(gl.TEXTURE_2D,
                          anisoExt.TEXTURE_MAX_ANISOTROPY_EXT,
                          Math.min(maxAniso, 16));
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0,
                     gl.RGBA, gl.UNSIGNED_BYTE,
                     new Uint8Array([0, 0, 0, 0]));

      let imgW = 1, imgH = 1;
      let textureLoaded = false;

      // Cache canvas CSS dimensions to avoid layout-flushing reads on
      // every redraw.  ``canvas.clientWidth/Height`` are layout-coupled
      // properties; reading them after any style change forces a sync
      // layout pass.  During gesture redraws (60-120/sec) that adds up
      // to ms of stutter.  Invalidate on the overlay-resize hook.
      let _cssW = 0, _cssH = 0, _dpr = 1, _sizeDirty = true;
      function invalidateSize() { _sizeDirty = true; }
      canvas.__invalidateSize = invalidateSize;
      function syncSize() {
        if (_sizeDirty) {
          _cssW = canvas.clientWidth;
          _cssH = canvas.clientHeight;
          _dpr = window.devicePixelRatio || 1;
          const w = Math.max(1, Math.round(_cssW * _dpr));
          const h = Math.max(1, Math.round(_cssH * _dpr));
          if (canvas.width !== w) canvas.width = w;
          if (canvas.height !== h) canvas.height = h;
          gl.viewport(0, 0, w, h);
          gl.uniform2f(U.canvasSizePx, w, h);
          gl.uniform1f(U.dpr, _dpr);
          _sizeDirty = false;
        }
      }

      function redraw(s) {
        syncSize();
        gl.uniform2f(U.imageSizePx, imgW, imgH);
        gl.uniform1f(U.zoom, s.s);
        gl.uniform2f(U.translatePx, s.tx, s.ty);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }

      function loadImage(url, onLoaded) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.decoding = 'async';
        img.addEventListener('load', () => {
          if (viewer.disposed) return;
          imgW = img.naturalWidth || img.width;
          imgH = img.naturalHeight || img.height;
          gl.bindTexture(gl.TEXTURE_2D, texture);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          // WebGL2 sized internal format (RGBA8) so mipmap generation
          // is guaranteed across drivers; the unsized ``gl.RGBA`` form
          // silently falls back to no-mipmap on some implementations,
          // which leaves MIN sampling as effectively bilinear-of-level-0
          // (= moire on high-frequency downscale).
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA,
                         gl.UNSIGNED_BYTE, img);
          // Mipmap chain — needed for LINEAR_MIPMAP_LINEAR minification.
          // WebGL2 supports NPOT mipmaps natively, no aspect/POT constraint.
          gl.generateMipmap(gl.TEXTURE_2D);
          textureLoaded = true;
          if (onLoaded) onLoaded();
        });
        img.addEventListener('error', (e) => {
          console.warn('SvgFigure WebGL: image load failed:', url, e);
        });
        img.src = url;
      }

      function isPointInImage(clientX, clientY) {
        const r = canvas.getBoundingClientRect();
        const localX = clientX - r.left;
        const localY = clientY - r.top;
        if (localX < 0 || localY < 0 || localX > r.width || localY > r.height) {
          return false;
        }
        const fitScale = Math.min(r.width / imgW, r.height / imgH);
        // Same CSS-matrix-model math the shader uses.
        const halfW = imgW * fitScale * 0.5;
        const halfH = imgH * fitScale * 0.5;
        const cx = r.width * 0.5;
        const cy = r.height * 0.5;
        const imgLeft = (cx - halfW) * state.s + state.tx;
        const imgTop = (cy - halfH) * state.s + state.ty;
        const dispW = imgW * fitScale * state.s;
        const dispH = imgH * fitScale * state.s;
        return localX >= imgLeft && localX <= imgLeft + dispW
            && localY >= imgTop && localY <= imgTop + dispH;
      }

      function dispose() {
        viewer.disposed = true;
        try {
          gl.deleteTexture(texture);
          gl.deleteBuffer(vbo);
          gl.deleteProgram(program);
          gl.deleteShader(vs);
          gl.deleteShader(fs);
          const ext = gl.getExtension('WEBGL_lose_context');
          if (ext) ext.loseContext();
        } catch (_) {}
        if (canvas.parentElement) canvas.parentElement.removeChild(canvas);
      }

      function clearActive() {
        // Reset the single GL texture back to the 1x1 transparent
        // initial state so the canvas doesn't show the previous tile
        // while the new tile's loadImage is in flight.
        if (!gl || !texture) return;
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0,
                       gl.RGBA, gl.UNSIGNED_BYTE,
                       new Uint8Array([0, 0, 0, 0]));
        imgW = 1; imgH = 1;
        textureLoaded = false;
        draw();
      }

      const viewer = {
        canvas, redraw, loadImage, isPointInImage, dispose, clearActive,
        get textureLoaded() { return textureLoaded; },
        disposed: false,
      };
      return viewer;
    }
    // Zoom by `ratio` about the canvas-local point (px, py). Clamps to
    // [MIN_S, MAX_S]; rebases translation so the image point under the
    // cursor stays under the cursor across the scale change.
    function zoomAbout(px, py, ratio) {
      const newS = Math.max(MIN_S, Math.min(MAX_S, state.s * ratio));
      const actualRatio = newS / state.s;
      state.tx = px * (1 - actualRatio) + state.tx * actualRatio;
      state.ty = py * (1 - actualRatio) + state.ty * actualRatio;
      state.s = newS;
      applyTransform();
    }

    // Track the currently-zoomed tile so arrow-key navigation can
    // find its neighbours among the grid's siblings.
    let currentTile = null;

    function openZoom(tile) {
      // Wipe the recycled viewer's current contents BEFORE we kick off
      // the new tile's load chain. Without this, the popup briefly
      // shows the previous tile (or whatever was last clicked) until
      // the new tile's first loadImage completes -- a stale-content
      // flash because the viewer is reused across openZoom calls.
      const isTileSwitch = (currentTile !== null && currentTile !== tile);
      if (isTileSwitch && webglViewer && webglViewer.clearActive) {
        try { webglViewer.clearActive(); } catch (_) {}
      }
      currentTile = tile;
      const hiresHref = tile.getAttribute('data-hires-href');
      // Extract label (if any) for the floating title above the plot.
      const labelSrc = tile.querySelector('text.fig-figure-text, text');
      const labelText = labelSrc ? (labelSrc.textContent || '') : '';

      // Recycle the viewer + canvas across openZoom calls. The first
      // call builds them; subsequent calls (e.g. arrow navigation
      // between tiles) reuse them so the worker's texture LRU stays
      // warm — revisited tiles paint instantly from the cache instead
      // of re-fetching + re-decoding + re-uploading from scratch.
      //
      // The legacy SVG viewer is per-tile (it clones the tile's SVG)
      // and cannot be recycled; for that path we tear down and rebuild
      // on every open.
      const figRoot = tile.closest('svg');
      const viewerHint = (figRoot && figRoot.dataset
                            && figRoot.dataset.popupViewer) || 'auto';
      const needLegacyRebuild = webglViewer && webglViewer.isLegacy;
      const firstBuild = (!webglViewer || needLegacyRebuild);
      if (firstBuild) {
        if (needLegacyRebuild) {
          try { webglViewer.dispose(); } catch (_) {}
          webglViewer = null;
        }
        overlayInner.innerHTML = '';
        canvasEl = document.createElement('div');
        canvasEl.className = 'ocd-zoom-canvas';
        overlayInner.appendChild(canvasEl);

        // Default viewer is the CSS-img path: plain <img> + CSS
        // matrix3d transform on its own compositor layer. Routes the
        // raster through BitmapImage → CALayer (Safari) / Skia HDR
        // (Chrome), so P3-PQ JXLs reach the display at absolute nits
        // — the WebGL2 paths can't do that because ``texImage2D(... RGBA8
        // UNSIGNED_BYTE, bmp)`` clamps to 8-bit at upload and crushes
        // the highlights / blacks.
        //
        // ``data-popup-viewer="webgl"`` (or ``"worker"``) opts into the
        // worker-thread WebGL2 viewer, which is faster on big SDR grids
        // (texture LRU lets tile-to-tile arrow nav skip refetch /
        // decode / upload) but breaks HDR. Used for benchmarking or
        // SDR-only workloads where pan/zoom smoothness matters more
        // than correctness on HDR content.
        if (viewerHint === 'webgl' || viewerHint === 'worker') {
          webglViewer = createPopupWorkerViewer(canvasEl)
                     || createPopupWebglViewer(canvasEl);
        } else {
          webglViewer = createCssImgViewer(canvasEl);
        }
        if (!webglViewer) {
          // Legacy fallback: SVG re-raster path. Marked isLegacy so
          // we know to rebuild it on every tile switch — its DOM is
          // a clone of the specific tile and can't be repurposed.
          const bbox = tile.getAttribute('data-bbox');
          const cloned = tile.cloneNode(true);
          cloned.removeAttribute('style');
          if (labelSrc) {
            const lc = cloned.querySelector('text.fig-figure-text, text');
            if (lc) lc.remove();
          }
          const legacyFit = document.createElement('div');
          legacyFit.className = 'ocd-zoom-fit';
          canvasEl.appendChild(legacyFit);
          const oSvg = document.createElementNS(xmlns, 'svg');
          oSvg.setAttribute('xmlns', xmlns);
          oSvg.setAttribute('viewBox', bbox);
          oSvg.appendChild(cloned);
          legacyFit.appendChild(oSvg);
          webglViewer = createLegacySvgViewer(legacyFit, oSvg);
          if (webglViewer) webglViewer.isLegacy = true;
        }
        attachCanvasGestures(canvasEl);
      }

      // Title is per-tile — update on every open (lazy-create / remove
      // the element to match this tile's label).
      let titleDiv = overlayInner.querySelector('.ocd-zoom-title');
      if (labelText) {
        if (!titleDiv) {
          titleDiv = document.createElement('div');
          titleDiv.className = 'ocd-zoom-title';
          overlayInner.insertBefore(titleDiv, canvasEl);
        }
        titleDiv.textContent = labelText;
      } else if (titleDiv) {
        titleDiv.remove();
      }

      // Thumb-first: always load the small data-URL thumb before the
      // hi-res, so the popup paints something instantly even when the
      // hi-res fetch is slow. ``data-thumb-href`` is persisted by
      // image_grid and never changes (the inline <image href> may
      // have been swapped to hi-res by the hover prefetch, which is
      // why we don't read it here). Falls back to inline <image href>
      // for older SVGs / direct ``<g class="fig-tile">`` usage that
      // doesn't set data-thumb-href.
      const thumbHref = tile.getAttribute('data-thumb-href')
                     || (tile.querySelector('image')
                          && tile.querySelector('image').getAttribute('href'));
      const viewerAtOpen = webglViewer;
      // Defer the visible backdrop until the first frame of content
      // is ready. Otherwise the user sees a dark backdrop with empty
      // contents for the duration of the thumb decode (~40 ms in
      // headless, ~100 ms+ for real HDR thumbs), which reads as
      // ``click was sluggish``. With the deferred activation, click
      // produces a single visual event: backdrop + thumb appear
      // together, then the hi-res swaps in.
      let popupShown = false;
      const showPopup = () => {
        if (popupShown) return;
        popupShown = true;
        syncOverlayToPane();
        overlay.classList.add('active');
        attachOverlayResizeTracking();
      };
      const upgradeToHires = () => {
        if (hiresHref && hiresHref !== thumbHref) {
          viewerAtOpen.loadImage(hiresHref, () => {
            if (webglViewer === viewerAtOpen) applyTransform();
          });
        }
      };
      if (thumbHref) {
        viewerAtOpen.loadImage(thumbHref, () => {
          showPopup();
          if (webglViewer === viewerAtOpen) applyTransform();
          upgradeToHires();
        });
      } else if (hiresHref) {
        // No thumb available -- fall through to hires for the first
        // visible frame.
        viewerAtOpen.loadImage(hiresHref, () => {
          showPopup();
          if (webglViewer === viewerAtOpen) applyTransform();
        });
      } else {
        // Nothing to load (e.g., legacy SVG-only tile) -- show empty.
        showPopup();
      }
      // Safety net: if both loads error out, show the popup anyway
      // after 1 s so the user can dismiss it. Caps the worst-case
      // ``invisible click`` interval if a load fails silently.
      setTimeout(showPopup, 1000);

      resetTransform();
      // (Gestures attached once on first build — canvas is recycled.)

      // Move the overlay to document.body so JupyterLab CSS on
      // ancestor containers can't clip or transform it. We sync the
      // bbox to the notebook pane explicitly via syncOverlayToPane().
      if (overlay.parentElement !== document.body) {
        document.body.appendChild(overlay);
      }
      // Stash the page's overflow + scroll state so we can restore it
      // on close.  Setting ``body.overflow: hidden`` while the popup is
      // open removes the page's scrollable ancestors from under the
      // wheel target — meaning our wheel handler doesn't need to
      // ``preventDefault()`` (nothing would scroll), so it can be
      // ``passive: true`` and Chrome won't throttle it.  Wheel events
      // arrive at full 60-120 Hz instead of the 30 Hz throttled rate.
      _savedBodyOverflow = document.body.style.overflow;
      _savedHtmlOverflow = document.documentElement.style.overflow;
      _savedScrollX = window.scrollX;
      _savedScrollY = window.scrollY;
      document.body.style.overflow = 'hidden';
      document.documentElement.style.overflow = 'hidden';
      // overlay activation + resize tracking deferred until first frame
      // of content is ready (see ``showPopup`` above) so the user
      // doesn't see an empty dark backdrop while the thumb is decoding.
    }
    let _savedBodyOverflow = '';
    let _savedHtmlOverflow = '';
    let _savedScrollX = 0;
    let _savedScrollY = 0;
    // HDR-preserving viewer using a plain ``<img>`` + matrix3d CSS
    // transform on its own compositor layer.  The WebGL2 RGBA8 path
    // clamps HDR PQ JXLs to SDR at texture upload; this path keeps
    // them on the BitmapImage → CALayer (Safari) / Skia HDR (Chrome)
    // pipeline, so PQ-tagged content reaches the display at absolute
    // nits.  Transform updates ride the compositor (no re-raster)
    // thanks to ``matrix3d`` + ``will-change: transform``.
    //
    // Trade-off vs the WebGL viewer: pointer→transform mapping still
    // runs on the JS main thread, so JupyterLab's main-thread load
    // can stutter pan/zoom even though the compositing itself is
    // free.  See ``popup_viewer`` plumbing in image_grid.py for how
    // callers opt in to this path.
    function createCssImgViewer(parent) {
      const wrap = document.createElement('div');
      wrap.className = 'ocd-zoom-cssimg';
      wrap.style.cssText =
        'position:absolute; inset:0; overflow:hidden; touch-action:none;'
      + ' pointer-events:none;';
      parent.appendChild(wrap);

      const IMG_STYLE =
        'position:absolute; top:0; left:0; transform-origin:0 0;'
      + ' will-change:transform; backface-visibility:hidden;'
      + ' image-rendering:pixelated;'
      + ' user-select:none; -webkit-user-drag:none; pointer-events:none;';

      // LRU of pre-loaded <img> elements keyed by URL. Tile switching
      // in a recycled popup is a display-flip instead of a re-decode
      // + re-paint. Each cached entry keeps its <img> in the DOM
      // (display:none when inactive) so the browser keeps the
      // decoded raster ready for instant compositor swap. Eviction
      // removes the oldest entries' <img> from the DOM, freeing the
      // browser-side decoded raster.
      const imgLRU = new Map();
      const IMG_LRU_MAX = 64;
      const inFlight = new Set();  // URLs whose Image() is loading
      let img = null;     // currently visible <img>
      let imgW = 0, imgH = 0;
      let textureLoaded = false;

      function setActiveImg_(url, entry) {
        if (img && img !== entry.el) {
          img.style.display = 'none';
        }
        img = entry.el;
        img.style.display = 'block';
        imgW = entry.w;
        imgH = entry.h;
        textureLoaded = true;
        // Move to MRU position.
        imgLRU.delete(url);
        imgLRU.set(url, entry);
      }

      // Hide whatever is currently shown without disposing it (the LRU
      // entry stays cached). openZoom calls this on tile switch so the
      // popup doesn't briefly flash the previous tile while the new
      // one's first loadImage is in flight.
      function clearActive() {
        if (img) {
          img.style.display = 'none';
          img = null;
        }
        imgW = 0; imgH = 0;
        textureLoaded = false;
      }

      function evictIfFull_() {
        while (imgLRU.size > IMG_LRU_MAX) {
          const k = imgLRU.keys().next().value;
          const ev = imgLRU.get(k);
          imgLRU.delete(k);
          if (ev && ev.el && ev.el !== img && ev.el.parentElement) {
            ev.el.parentElement.removeChild(ev.el);
          }
        }
      }

      function redraw(s) {
        // Mirror the WebGL viewer's vertex-shader math exactly so
        // zoomAboutTarget()'s coordinate-anchor formula
        // ``tx_new = px(1−ratio) + tx_old·ratio`` (outer JS scope at
        // line ~755) produces the right cursor-anchored zoom.
        //
        //   preXfmTopLeft = canvasCenter − imageFitHalf
        //   imageOriginCSS = preXfmTopLeft · u_zoom + u_translatePx
        //   imageSizeCSS   = imageSize · fitScale · u_zoom
        //
        // The ``· u_zoom`` on the centering term is the critical
        // bit — without it the cursor-anchored zoom drifts toward
        // the top-left of the image.
        if (!imgW || !imgH) return;
        const rect = wrap.getBoundingClientRect();
        const fitScale = Math.min(rect.width / imgW, rect.height / imgH);
        const eff = fitScale * s.s;
        const preXfmX = (rect.width  - imgW * fitScale) * 0.5;
        const preXfmY = (rect.height - imgH * fitScale) * 0.5;
        const tx = preXfmX * s.s + s.tx;
        const ty = preXfmY * s.s + s.ty;
        // Rotation around the displayed image center. Final transform:
        //   T(tx + imgW*eff/2, ty + imgH*eff/2)  -- move center to dest
        //   * R(theta)                            -- rotate around origin
        //   * S(eff)                              -- scale
        //   * T(-imgW/2, -imgH/2)                 -- move center to origin
        // collapses to a 2x2 linear part [a b; d e] + constant (c, f).
        // When s.r is 0 the formulas reduce exactly to the previous
        // pure-scale path (a=eff, b=0, c=tx, d=0, e=eff, f=ty).
        const theta = s.r || 0;
        const cs = Math.cos(theta), sn = Math.sin(theta);
        const halfW = imgW * 0.5, halfH = imgH * 0.5;
        const a =  eff * cs;
        const b = -eff * sn;
        const d =  eff * sn;
        const e =  eff * cs;
        const c = tx + halfW * eff - eff * cs * halfW + eff * sn * halfH;
        const f = ty + halfH * eff - eff * sn * halfW - eff * cs * halfH;
        // matrix3d -> own GPU layer; transform updates skip
        // layout+paint and only touch the compositor. matrix3d is
        // column-major: matrix3d(m11, m12, m13, m14, m21, m22, ...).
        img.style.transform =
          'matrix3d(' + a + ',' + d + ',0,0, '
          + b + ',' + e + ',0,0, '
          + '0,0,1,0, '
          + c + ',' + f + ',0,1)';
      }

      function loadImage(url, onLoaded) {
        // LRU hit: instant swap, no fetch, no decode.
        const cached = imgLRU.get(url);
        if (cached) {
          setActiveImg_(url, cached);
          if (onLoaded) onLoaded();
          return;
        }
        // Already loading this URL — let the in-flight load finish.
        // (No callback chaining: openZoom's chain serializes thumb -> hires,
        //  it never re-enters for the same URL within one popup session.)
        if (inFlight.has(url)) return;
        inFlight.add(url);

        const newImg = document.createElement('img');
        newImg.draggable = false;
        newImg.style.cssText = IMG_STYLE + ' display:none;';
        wrap.appendChild(newImg);
        newImg.addEventListener('load', () => {
          inFlight.delete(url);
          const entry = {
            el: newImg,
            w: newImg.naturalWidth || 0,
            h: newImg.naturalHeight || 0,
          };
          imgLRU.set(url, entry);
          setActiveImg_(url, entry);
          evictIfFull_();
          if (onLoaded) onLoaded();
        });
        newImg.addEventListener('error', (e) => {
          inFlight.delete(url);
          if (newImg.parentElement) newImg.parentElement.removeChild(newImg);
          console.warn('SvgFigure CSS-img viewer image load failed', url, e);
        });
        newImg.src = url;
      }

      function isPointInImage(clientX, clientY) {
        const r = img.getBoundingClientRect();
        return clientX >= r.left && clientX <= r.right
            && clientY >= r.top && clientY <= r.bottom;
      }

      function dispose() {
        if (wrap.parentElement) wrap.parentElement.removeChild(wrap);
      }

      return { redraw, loadImage, isPointInImage, dispose, clearActive,
               get textureLoaded() { return textureLoaded; },
               get imgW() { return imgW; }, get imgH() { return imgH; } };
    }

    // Legacy SVG path — used only when WebGL2 isn't available.  Mirrors
    // the WebGL viewer's public interface (redraw, loadImage,
    // isPointInImage, dispose) so the rest of the popup code doesn't
    // need to branch.  Applies CSS transforms to the fit element the
    // way the original implementation did.
    function createLegacySvgViewer(fitEl, oSvg) {
      function redraw(s) {
        fitEl.style.transform =
          'matrix(' + s.s + ',0,0,' + s.s + ',' + s.tx + ',' + s.ty + ')';
      }
      function loadImage(url, onLoaded) {
        const probe = new Image();
        probe.draggable = false;
        probe.addEventListener('load', () => {
          const svgImg = oSvg.querySelector('.fig-tile image');
          if (!svgImg) return;
          svgImg.setAttribute('href', url);
          svgImg.setAttributeNS(
            'http://www.w3.org/1999/xlink', 'xlink:href', url);
          if (onLoaded) onLoaded();
        });
        probe.addEventListener('error', (e) => {
          console.warn('SvgFigure legacy viewer image load failed', url, e);
        });
        probe.src = url;
      }
      function isPointInImage(clientX, clientY) {
        const svgEl = oSvg;
        if (!svgEl) return false;
        const r = svgEl.getBoundingClientRect();
        return clientX >= r.left && clientX <= r.right
            && clientY >= r.top && clientY <= r.bottom;
      }
      function dispose() {
        if (fitEl.parentElement) fitEl.parentElement.removeChild(fitEl);
      }
      return { redraw, loadImage, isPointInImage, dispose,
               get textureLoaded() { return true; } };
    }
    // Resize tracking — keep the overlay glued to the notebook pane
    // when the user resizes the browser window, drags a JupyterLab
    // split pane, or scrolls the notebook viewport.  Attached when
    // the overlay opens; detached on close so we don't leak observers
    // across hidden figures.
    let _resizeObserver = null;
    let _onResize = null;
    let _watchdogRaf = 0;
    let _watchdogRect = null;
    function attachOverlayResizeTracking() {
      detachOverlayResizeTracking();
      // Refresh the pane reference: a notebook panel that moved between
      // JupyterLab split panes between open events would otherwise be
      // detached from layout.
      pane = resolvePane();
      _onResize = () => {
        syncOverlayToPane();
        // Canvas CSS size changed — invalidate the gesture-handler's
        // cached rect AND the viewer's cached CSS dimensions so the
        // next wheel/pointer event uses fresh coords and the next
        // redraw recomputes viewport/uniforms.
        if (canvasEl && canvasEl.__invalidateGestureRect) {
          canvasEl.__invalidateGestureRect();
        }
        if (webglViewer && webglViewer.canvas
            && webglViewer.canvas.__invalidateSize) {
          webglViewer.canvas.__invalidateSize();
        }
        // The WebGL canvas's CSS size just changed — schedule a redraw
        // so the shader picks up the new viewport / fit-scale.
        applyTransform();
      };
      window.addEventListener('resize', _onResize);
      // Scroll re-positioning matters when the notebook pane scrolls
      // under the fixed overlay; ``passive: true`` avoids blocking the
      // scroller.
      window.addEventListener('scroll', _onResize, { passive: true, capture: true });
      if (typeof ResizeObserver !== 'undefined') {
        _resizeObserver = new ResizeObserver(_onResize);
        // Wide net: observe every ancestor of the wrapper up to the
        // document body, plus a handful of well-known JupyterLab
        // containers.  Split-pane drags don't change the viewport
        // (so ``window.resize`` is silent) but DO change the size of
        // *something* in this chain.  ResizeObserver dedupes work, so
        // observing many elements is cheap.
        const seen = new Set();
        const observeIfNew = (el) => {
          if (el && !seen.has(el)) {
            seen.add(el);
            try { _resizeObserver.observe(el); } catch (_) {}
          }
        };
        let cur = wrapper;
        while (cur && cur !== document.body) {
          observeIfNew(cur);
          cur = cur.parentElement;
        }
        observeIfNew(document.body);
        observeIfNew(document.documentElement);
        for (const sel of ['#jp-main-content-panel', '.jp-MainAreaWidget',
                            '.jp-NotebookPanel', '.jp-NotebookPanel-notebook',
                            '.jp-Notebook']) {
          for (const el of document.querySelectorAll(sel)) {
            observeIfNew(el);
          }
        }
      }
      // Watchdog: ResizeObserver covers most resize triggers, but split-
      // pane drag implementations vary across JupyterLab versions /
      // classic notebook / nbclassic / VSCode webview, so we ALSO poll
      // the pane's bbox periodically as a fallback.
      //
      // setInterval @ 200 ms instead of rAF @ 60 Hz — the rAF version
      // forced a synchronous layout flush every animation frame, which
      // competes with the WebGL gesture's rAF and shows up as zoom
      // stutter on complex JupyterLab pages.  200 ms is fast enough to
      // pick up a manual split-pane drag without lag, far longer than
      // any single animation frame.
      _watchdogRect = null;
      _watchdogRaf = window.setInterval(() => {
        if (!overlay.classList.contains('active')) return;
        const r = pane.getBoundingClientRect();
        const prev = _watchdogRect;
        if (!prev || prev.top !== r.top || prev.left !== r.left
            || prev.width !== r.width || prev.height !== r.height) {
          _watchdogRect = { top: r.top, left: r.left,
                            width: r.width, height: r.height };
          syncOverlayToPane();
          applyTransform();
        }
      }, 200);
    }
    function detachOverlayResizeTracking() {
      if (_resizeObserver) {
        _resizeObserver.disconnect();
        _resizeObserver = null;
      }
      if (_onResize) {
        window.removeEventListener('resize', _onResize);
        window.removeEventListener('scroll', _onResize, { capture: true });
        _onResize = null;
      }
      if (_watchdogRaf) {
        clearInterval(_watchdogRaf);
        _watchdogRaf = 0;
      }
      _watchdogRect = null;
    }
    function closeZoom() {
      detachOverlayResizeTracking();
      // Keep webglViewer + canvasEl alive across closes. The worker
      // holds a hot texture LRU — re-opening the popup (same figure
      // or arrow-navigation re-entry) is then a Map lookup + draw,
      // not a fetch + decode + upload cycle. The worker stays GC-
      // anchored via the wrapper closure; when the figure is removed
      // from the DOM the closure becomes unreachable and the OS
      // reclaims the worker thread + GL context.
      overlay.classList.remove('active');
      overlay.style.top = overlay.style.left =
        overlay.style.width = overlay.style.height = '';
      // Restore the page's overflow + scroll position.
      document.body.style.overflow = _savedBodyOverflow;
      document.documentElement.style.overflow = _savedHtmlOverflow;
      window.scrollTo(_savedScrollX, _savedScrollY);
      // Restore the overlay to its original DOM position so a subsequent
      // wrapper-uid lookup still finds it.
      if (overlayHome && overlay.parentElement !== overlayHome) {
        overlayHome.appendChild(overlay);
      }
    }

    // Mouse-wheel zoom, 1-pointer pan, 2-pointer pinch, dbl-click reset.
    // PointerEvent unifies mouse + touch + Apple-pencil — same handler
    // works for Safari multi-touch.
    function attachCanvasGestures(canvas) {
      // Cache the canvas's bounding rect — calling getBoundingClientRect
      // inside each wheel/pointer event can force a synchronous layout
      // flush, which on complex pages costs 5-10 ms per event.  At a
      // 120 Hz trackpad event rate that's a stutter generator.  The
      // canvas's CSS box doesn't move during a gesture (overlay is
      // position:fixed); invalidate on the overlay-tracking resize
      // hook only.
      let cachedRect = canvas.getBoundingClientRect();
      function invalidateRect() { cachedRect = canvas.getBoundingClientRect(); }
      // Stash the invalidator on canvas so the resize-watchdog can call
      // it (see syncOverlayToPane → applyTransform).
      canvas.__invalidateGestureRect = invalidateRect;

      canvas.addEventListener('wheel', (e) => {
        // ctrl+wheel = trackpad pinch (Safari/Chrome). The page-zoom
        // default fires on ctrl+wheel even though we've set
        // body.overflow:hidden -- page zoom isn't a scroll, so the
        // overflow trick doesn't block it. We need ``preventDefault``
        // for those, which requires a non-passive listener. Plain
        // wheel (scroll) we also preventDefault for symmetry since
        // the page has nothing to scroll anyway.
        e.preventDefault();
        // Trackpad pinch delivers much smaller deltaY per event than
        // a mouse-scroll wheel notch -- needs a steeper exponential
        // base to feel responsive. ctrlKey distinguishes the two
        // paths in Safari/Chrome on macOS; outside that we keep the
        // gentle scroll-wheel curve.
        const base = e.ctrlKey ? 1.01 : 1.0015;
        const ratio = Math.pow(base, -e.deltaY);
        zoomAboutTarget(e.clientX - cachedRect.left,
                        e.clientY - cachedRect.top, ratio);
      }, { passive: false });

      // Safari fires gesturestart / gesturechange / gestureend on
      // trackpad pinch + rotate IN ADDITION to ctrl+wheel. Without
      // preventDefault those default to zooming the page. We also
      // use the rotation field on gesturechange to drive the popup's
      // rotation (only the CSS-img viewer renders it; the WebGL
      // viewers ignore it silently). e.rotation is degrees-since-
      // gesturestart, cumulative -- track the start value so we add
      // the delta to whatever rotation the popup had at gesture
      // beginning, not the absolute angle.
      let _gestureStartR = 0;
      canvas.addEventListener('gesturestart', (e) => {
        e.preventDefault();
        _gestureStartR = state.r || 0;
      });
      canvas.addEventListener('gesturechange', (e) => {
        e.preventDefault();
        const newR = _gestureStartR + (e.rotation || 0) * Math.PI / 180;
        state.r = newR;
        target.r = newR;
        if (_tweenRaf) { cancelAnimationFrame(_tweenRaf); _tweenRaf = 0; }
        if (webglViewer) webglViewer.redraw(state);
      });
      canvas.addEventListener('gestureend', (e) => { e.preventDefault(); });

      canvas.addEventListener('dblclick', (e) => {
        e.preventDefault();
        resetTransform();
      });

      const activePointers = new Map();
      let panLast = null;
      let pinchPrev = null;
      // Tap tracking: a single-pointer press that ends without
      // significant movement is treated as a click. If the click lands
      // OUTSIDE the SVG image bbox we dismiss the overlay — that's how
      // "click on the dark area to close" works while still letting
      // press-and-drag pan freely on the image itself.
      const TAP_MOVE_THRESHOLD_SQ = 64;  // 8 px
      let tapStart = null;
      let tapMoved = false;
      let tapPointerId = null;
      function endPointer(e) {
        const wasTapPointer = (e.pointerId === tapPointerId);
        if (activePointers.has(e.pointerId)) {
          activePointers.delete(e.pointerId);
          try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
        }
        if (activePointers.size < 2) pinchPrev = null;
        if (activePointers.size === 0) {
          panLast = null;
          canvas.classList.remove('dragging');
          // Drop the GPU compositing layer so the SVG re-rasterizes
          // at the current zoom — gets image-rendering: pixelated to
          // kick in for sharp output at rest.
          setGestureActive(false);
          // Resolve tap-vs-drag.
          if (wasTapPointer && tapStart && !tapMoved) {
            const insideImage = webglViewer
              ? webglViewer.isPointInImage(e.clientX, e.clientY)
              : false;
            if (!insideImage) {
              closeZoom();
              tapStart = null; tapPointerId = null; tapMoved = false;
              return;
            }
          }
          tapStart = null; tapPointerId = null; tapMoved = false;
        } else if (activePointers.size === 1) {
          const remaining = activePointers.values().next().value;
          panLast = { x: remaining.x, y: remaining.y };
        }
      }
      canvas.addEventListener('pointerdown', (e) => {
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
        const wasEmpty = activePointers.size === 0;
        activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        canvas.classList.add('dragging');
        if (wasEmpty) setGestureActive(true);
        if (activePointers.size === 1) {
          panLast = { x: e.clientX, y: e.clientY };
          pinchPrev = null;
          tapStart = { x: e.clientX, y: e.clientY };
          tapPointerId = e.pointerId;
          tapMoved = false;
        } else if (activePointers.size === 2) {
          panLast = null;
          tapStart = null;  // 2-finger gesture isn't a tap
          const pts = Array.from(activePointers.values());
          pinchPrev = {
            dist: Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y),
            mid: { x: (pts[0].x + pts[1].x) / 2,
                    y: (pts[0].y + pts[1].y) / 2 },
            angle: Math.atan2(pts[1].y - pts[0].y, pts[1].x - pts[0].x),
          };
        }
      });
      canvas.addEventListener('pointermove', (e) => {
        if (!activePointers.has(e.pointerId)) return;
        activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        // Tap-vs-drag movement accumulation.
        if (tapStart && e.pointerId === tapPointerId && !tapMoved) {
          const dx = e.clientX - tapStart.x;
          const dy = e.clientY - tapStart.y;
          if (dx * dx + dy * dy > TAP_MOVE_THRESHOLD_SQ) tapMoved = true;
        }
        if (activePointers.size === 1 && panLast) {
          // Pan: pointermove is already a smooth 1:1 mapping to canvas
          // CSS pixels — no rate mismatch, no need to tween.  Snap the
          // target AND the visible state together.
          const dx = e.clientX - panLast.x;
          const dy = e.clientY - panLast.y;
          panLast = { x: e.clientX, y: e.clientY };
          if (webglViewer && webglViewer.isWorker) {
            const m = webglViewer.stateMirror;
            webglViewer.applySnapState(m.s, m.tx + dx, m.ty + dy);
          } else {
            state.tx += dx; state.ty += dy;
            target.tx = state.tx; target.ty = state.ty;
            if (_tweenRaf) { cancelAnimationFrame(_tweenRaf); _tweenRaf = 0; }
            if (webglViewer) webglViewer.redraw(state);
          }
        } else if (activePointers.size === 2) {
          const pts = Array.from(activePointers.values());
          const mid = { x: (pts[0].x + pts[1].x) / 2,
                        y: (pts[0].y + pts[1].y) / 2 };
          const dist = Math.hypot(
            pts[0].x - pts[1].x, pts[0].y - pts[1].y);
          const angle = Math.atan2(
            pts[1].y - pts[0].y, pts[1].x - pts[0].x);
          if (pinchPrev) {
            // Pinch: same fingers-stick-to-image math, snap state +
            // target together — touch input is OS-smoothed, no tween.
            const prevMidLocalX = pinchPrev.mid.x - cachedRect.left;
            const prevMidLocalY = pinchPrev.mid.y - cachedRect.top;
            const newMidLocalX = mid.x - cachedRect.left;
            const newMidLocalY = mid.y - cachedRect.top;
            const curS = (webglViewer && webglViewer.isWorker)
              ? webglViewer.stateMirror : state;
            const imgX = (prevMidLocalX - curS.tx) / curS.s;
            const imgY = (prevMidLocalY - curS.ty) / curS.s;
            const ratio = dist / pinchPrev.dist;
            const newS = Math.max(MIN_S,
              Math.min(MAX_S, curS.s * ratio));
            const newTx = newMidLocalX - imgX * newS;
            const newTy = newMidLocalY - imgY * newS;
            // Rotation delta from finger-pair angle. Normalise to
            // [-PI, PI] so a wraparound across +/- PI doesn't read
            // as a full spin. Honored by the CSS-img viewer only.
            let dR = angle - pinchPrev.angle;
            if (dR >  Math.PI) dR -= 2 * Math.PI;
            if (dR < -Math.PI) dR += 2 * Math.PI;
            if (webglViewer && webglViewer.isWorker) {
              webglViewer.applySnapState(newS, newTx, newTy);
            } else {
              state.s = newS; state.tx = newTx; state.ty = newTy;
              state.r = (state.r || 0) + dR;
              target.s = state.s; target.tx = state.tx;
              target.ty = state.ty; target.r = state.r;
              if (_tweenRaf) { cancelAnimationFrame(_tweenRaf); _tweenRaf = 0; }
              if (webglViewer) webglViewer.redraw(state);
            }
          }
          pinchPrev = { dist, mid, angle };
        }
      });
      canvas.addEventListener('pointerup', endPointer);
      canvas.addEventListener('pointercancel', endPointer);
    }

    // Skip per-tile click-to-popup + hover-prefetch when the SVG is in
    // linked-axes mode: drag-pan would otherwise race the click handler
    // (and pointer events would steal focus from the linked controller).
    // The linked controller (set up earlier in this IIFE) owns all
    // pointer behaviour for those cells.
    const _linkedMode = (svg && svg.dataset && svg.dataset.linkAxes === '1');
    if (!_linkedMode) tiles.forEach(tile => {
      tile.addEventListener('click', (e) => {
        e.stopPropagation(); openZoom(tile);
      });
      // Prefetch + inline upgrade. On hover (or touchstart) we fetch
      // the hi-res bytes AND, once they decode, swap the inline tile's
      // <image> href to point at the hi-res URL. That way the user
      // sees a visible upgrade as they hover, and after a popup-close
      // the grid retains the hi-res state instead of reverting to the
      // small data-URL thumb.
      //
      // (The earlier "warm cache only, never swap inline" mode lost
      // sharpness when the browser downscaled a 2k source to a
      // few-hundred-CSS-pixel cell with no Lanczos/LP filter; the
      // tile uses ``image-rendering: pixelated`` so the downscale is
      // a clean nearest-neighbour pick. Net better than the lossy
      // 256-px thumb for HDR / scene-RGB workflows.)
      //
      // ``data-auto-upgrade="1"`` (set by ``auto_upgrade=True`` on
      // image_grid -- default for single-image imshow) fires the
      // prefetch eagerly on load, so the grid lights up to hi-res
      // without any hover required.
      const hiresHref = tile.getAttribute('data-hires-href');
      if (hiresHref) {
        const autoUpgrade = (tile.getAttribute('data-auto-upgrade') === '1');
        let prefetched = false;
        const prefetch = () => {
          if (prefetched) return;
          prefetched = true;
          const probe = new Image();
          probe.draggable = false;
          probe.addEventListener('load', () => {
            const inlineImg = tile.querySelector('image');
            if (inlineImg) {
              inlineImg.setAttribute('href', hiresHref);
              inlineImg.setAttributeNS(
                'http://www.w3.org/1999/xlink',
                'xlink:href', hiresHref);
            }
          });
          probe.addEventListener('error', (e) => {
            // Leave the thumb in place if hi-res fetch fails (e.g.
            // server unreachable, browser can't decode JXL).
            console.warn('SvgFigure hi-res upgrade failed for',
                         hiresHref, e);
          });
          probe.src = hiresHref;
        };
        tile.addEventListener('pointerenter', prefetch);
        // Touch devices: prefetch on first touchstart so a tap that
        // turns into a click already has the bytes warm.
        tile.addEventListener('touchstart', prefetch, { passive: true });
        // Auto-upgrade tiles kick the prefetch immediately on load.
        if (autoUpgrade) prefetch();
      }
    });
    if (overlay) {
      // Close on backdrop click only — clicks/drags inside the canvas
      // must not dismiss (the user is interacting with the image).
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) closeZoom();
      });
      document.addEventListener('keydown', (e) => {
        if (!overlay.classList.contains('active')) return;
        // Modifier-only presses do nothing (so Shift/Cmd/Ctrl/Alt
        // can be held without affecting the popup state).
        if (e.key === 'Shift' || e.key === 'Control' ||
            e.key === 'Alt'   || e.key === 'Meta') return;
        // Always stop the browser/Jupyter defaults: spacebar scrolls
        // the notebook underneath, Tab shifts focus, arrows move
        // cells, etc.
        e.preventDefault();
        e.stopPropagation();
        // Arrow keys navigate to adjacent tiles in the grid. Ncol is
        // read from the wrapper's data-ncol attribute (set by
        // image_grid); falls back to "left/right only" treatment if
        // ncol is unset.
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft' ||
            e.key === 'ArrowUp'    || e.key === 'ArrowDown') {
          if (!currentTile) return;
          const all = Array.from(wrapper.querySelectorAll('.fig-tile'));
          const idx = all.indexOf(currentTile);
          if (idx < 0) return;
          // image_grid stamps ``data-ncol`` on the inner root SVG; fall
          // back to 0 (= left/right only) if it isn't set.
          const innerSvg = wrapper.querySelector('svg');
          const ncol = innerSvg
            ? parseInt(innerSvg.getAttribute('data-ncol'), 10) || 0
            : 0;
          let next = idx;
          if (e.key === 'ArrowRight') next = idx + 1;
          else if (e.key === 'ArrowLeft') next = idx - 1;
          else if (e.key === 'ArrowDown' && ncol > 0) next = idx + ncol;
          else if (e.key === 'ArrowUp'   && ncol > 0) next = idx - ncol;
          if (next < 0 || next >= all.length || next === idx) return;
          openZoom(all[next]);   // re-uses overlay; no flicker close+reopen
          return;
        }
        // Any other key closes.
        closeZoom();
      }, true);  // capture phase so we beat JupyterLab's own handlers
      window.addEventListener('resize', () => {
        if (overlay.classList.contains('active')) syncOverlayToPane();
      });
    }

    // ─── linked-axes pan/zoom controller ──────────────────────────────
    // When the SVG was built with image_grid(link_axes=True), every
    // cell is a nested <svg.ocd-linked-cell viewBox="..."> over the
    // same raster shape. We wire a shared {x, y, w, h} viewport state
    // and apply it to every cell on each pointer/wheel event — drag on
    // any cell pans all cells; wheel anchored at the cursor zooms all.
    //
    // Mirrors the popup viewer's controller (target + tweened state,
    // cursor-anchored zoom, gestural pinch in capable browsers).
    // Independent of the popup overlay — the click-to-zoom path still
    // works for single-cell deep-dive on top of linked panning.
    if (svg && svg.dataset && svg.dataset.linkAxes === '1') {
      const cells = svg.querySelectorAll('svg.ocd-linked-cell');
      const hits = svg.querySelectorAll('rect.ocd-linked-cell-hit');
      if (cells.length > 0 && hits.length === cells.length) {
        const RAS_W = parseFloat(svg.dataset.linkRasterW) || 1;
        const RAS_H = parseFloat(svg.dataset.linkRasterH) || 1;
        // Parse initial ROI from data-link-roi="x y w h" (source px).
        const roiAttr = (svg.dataset.linkRoi || '').trim().split(/\s+/);
        const rawX = parseFloat(roiAttr[0]) || 0;
        const rawY = parseFloat(roiAttr[1]) || 0;
        const rawW = parseFloat(roiAttr[2]) || RAS_W;
        const rawH = parseFloat(roiAttr[3]) || RAS_H;

        // Each cell defines the locked aspect ratio. Read the SVG
        // ``width``/``height`` attributes (set in outer-viewBox units
        // at emit time) -- reliable regardless of browser layout
        // timing, unlike getBoundingClientRect which is zero before
        // the first paint. We snap the initial ROI to that aspect so
        // the viewBox is never letterboxed: the visible image fills
        // the cell precisely and pan/zoom never changes the cell's
        // clickable area.
        const cellW = parseFloat(cells[0].getAttribute('width')) || 1;
        const cellH = parseFloat(cells[0].getAttribute('height')) || 1;
        const cellAR = cellW / cellH;

        function snapToAspect(x, y, w, h) {
          // Expand the shorter axis (in viewBox units) so the resulting
          // viewBox matches the cell's aspect. Center-anchored so the
          // ROI's geometric center is preserved.
          const cx = x + w * 0.5, cy = y + h * 0.5;
          let nw = w, nh = h;
          if (w / h > cellAR) {        // ROI wider than cell -> grow h
            nh = w / cellAR;
          } else if (w / h < cellAR) { // ROI taller than cell -> grow w
            nw = h * cellAR;
          }
          return { x: cx - nw * 0.5, y: cy - nh * 0.5, w: nw, h: nh };
        }
        const init = snapToAspect(rawX, rawY, rawW, rawH);
        const initX = init.x, initY = init.y, initW = init.w, initH = init.h;

        // Two state objects mirror the popup viewer pattern:
        //   ``state``  = the viewBox we're currently rendering.
        //   ``target`` = what input is steering us toward.
        // PAN updates BOTH simultaneously (1:1 cursor tracking, no
        // perceptible lag). ZOOM updates only ``target`` and lets the
        // rAF tween ease ``state`` toward it (avoids the discrete-
        // wheel-notch jumpiness).
        const state = { x: initX, y: initY, w: initW, h: initH };
        const target = { x: initX, y: initY, w: initW, h: initH };
        // Zoom limits: max-zoom-in = 8 source-px window, max-zoom-out =
        // 8× full image. Aspect lock means we only need a single
        // ``scale`` (= w / initW) — w and h move together.
        const MIN_W = Math.max(8, initW / 200);
        const MAX_W = Math.min(RAS_W * 8, initW * 200);
        const TWEEN_ALPHA = 0.35;
        const TWEEN_EPS = 0.25;
        let _raf = 0;

        function applyViewBox() {
          const vb = state.x.toFixed(3) + ' ' + state.y.toFixed(3) + ' '
                   + state.w.toFixed(3) + ' ' + state.h.toFixed(3);
          for (let i = 0; i < cells.length; i++) {
            cells[i].setAttribute('viewBox', vb);
          }
        }

        function startTween() {
          if (_raf) return;
          const tick = () => {
            const dw = target.w - state.w;
            const dh = target.h - state.h;
            const dx = target.x - state.x;
            const dy = target.y - state.y;
            if (Math.abs(dw) < TWEEN_EPS && Math.abs(dh) < TWEEN_EPS
                && Math.abs(dx) < TWEEN_EPS && Math.abs(dy) < TWEEN_EPS) {
              state.x = target.x; state.y = target.y;
              state.w = target.w; state.h = target.h;
              _raf = 0;
            } else {
              state.w += dw * TWEEN_ALPHA;
              state.h += dh * TWEEN_ALPHA;
              state.x += dx * TWEEN_ALPHA;
              state.y += dy * TWEEN_ALPHA;
              _raf = requestAnimationFrame(tick);
            }
            applyViewBox();
          };
          _raf = requestAnimationFrame(tick);
        }

        function setViewbox(x, y, w, h, animated) {
          // Apply aspect-preserving clamp on w (and derive h).
          w = Math.max(MIN_W, Math.min(MAX_W, w));
          h = w / cellAR;
          target.x = x; target.y = y;
          target.w = w; target.h = h;
          if (animated) {
            startTween();
          } else {
            state.x = x; state.y = y;
            state.w = w; state.h = h;
            if (_raf) { cancelAnimationFrame(_raf); _raf = 0; }
            applyViewBox();
          }
        }

        // Map a client (event.clientX, clientY) into source coords
        // for ``state`` (what's currently rendered). Uses the HIT RECT's
        // bbox (= the cell's display bbox in CSS px) — the rect lives
        // in the outer SVG coord system, so its bbox doesn't change
        // when the inner viewBox pans/zooms (unlike the inner cell's
        // image bbox). With aspect lock there's no letterbox to handle.
        function clientToSource(hitEl, clientX, clientY) {
          const r = hitEl.getBoundingClientRect();
          const fracX = (clientX - r.left) / r.width;
          const fracY = (clientY - r.top)  / r.height;
          return {
            x: state.x + fracX * state.w,
            y: state.y + fracY * state.h,
          };
        }

        // Wheel zoom anchored at the cursor. Uses ``state`` (not
        // ``target``) so mid-animation wheels stay anchored to what
        // the user is *seeing*, not the lagging target.
        //
        // ctrl+wheel = trackpad pinch (Safari & Chrome on macOS).
        // Trackpad pinch delivers TINY deltaY per event (~1-5) at
        // very high event rate, vs a mouse-wheel notch's deltaY ~100.
        // Steeper exponent base for the pinch path keeps pinch
        // responsive without making scroll-wheel feel jumpy — matches
        // the popup viewer's gesture base.
        function onWheel(e) {
          e.preventDefault();
          const hit = e.currentTarget;
          // base ^ (-deltaY) === Math.exp(deltaY * Math.log(base)*-1)
          // ln(1.01) ≈ 0.00995, ln(1.0015) ≈ 0.0015
          const base = e.ctrlKey ? 1.01 : 1.0015;
          const ratio = Math.pow(base, e.deltaY);
          const newW = Math.max(MIN_W, Math.min(MAX_W, state.w * ratio));
          const actualRatio = newW / state.w;
          const newH = newW / cellAR;
          const anchor = clientToSource(hit, e.clientX, e.clientY);
          const newX = anchor.x - (anchor.x - state.x) * actualRatio;
          const newY = anchor.y - (anchor.y - state.y) * actualRatio;
          // Pinch: skip the tween for 1:1 finger-tracking responsiveness
          // (high-rate event stream tweens itself naturally). Scroll
          // wheel: tween for smoothing of discrete notch jumps.
          setViewbox(newX, newY, newW, newH, !e.ctrlKey);
        }

        // Pointer drag = pan. Updates state AND target simultaneously
        // for instant 1:1 cursor tracking (no tween lag).
        let dragId = null, dragLastX = 0, dragLastY = 0, dragHit = null;
        function onPointerDown(e) {
          if (e.button !== 0 && e.pointerType !== 'touch') return;
          dragId = e.pointerId;
          dragHit = e.currentTarget;
          dragLastX = e.clientX;
          dragLastY = e.clientY;
          dragHit.setPointerCapture(e.pointerId);
          dragHit.style.cursor = 'grabbing';
          // Snap state to target before pan so we don't fight any
          // in-flight zoom tween.
          state.x = target.x; state.y = target.y;
          state.w = target.w; state.h = target.h;
          if (_raf) { cancelAnimationFrame(_raf); _raf = 0; }
          e.preventDefault();
        }
        function onPointerMove(e) {
          if (e.pointerId !== dragId) return;
          const r = dragHit.getBoundingClientRect();
          const dxClient = e.clientX - dragLastX;
          const dyClient = e.clientY - dragLastY;
          dragLastX = e.clientX;
          dragLastY = e.clientY;
          const sx = state.w / r.width;   // source-px per client-px (X)
          const sy = state.h / r.height;  // source-px per client-px (Y)
          state.x -= dxClient * sx;
          state.y -= dyClient * sy;
          target.x = state.x; target.y = state.y;
          applyViewBox();
        }
        function onPointerUp(e) {
          if (e.pointerId !== dragId) return;
          if (dragHit) {
            try { dragHit.releasePointerCapture(e.pointerId); } catch {}
            dragHit.style.cursor = 'grab';
          }
          dragId = null; dragHit = null;
        }

        // Reset to the initial ROI (double-click on any cell OR press 'H').
        function resetView() {
          setViewbox(initX, initY, initW, initH, true);
        }
        function onDblClick(e) {
          e.preventDefault();
          resetView();
        }
        function onKey(e) {
          if (e.key === 'h' || e.key === 'H' || e.key === 'Home') {
            if (!wrapper.isConnected) return;
            const tag = (document.activeElement
                && document.activeElement.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea') return;
            e.preventDefault();
            resetView();
          }
        }
        window.addEventListener('keydown', onKey);

        // Handlers attach to the HIT RECTS (which live in the outer SVG
        // coord system at the cell's bbox). The cells' viewBox is
        // mutated by the controller but cells themselves carry no
        // listeners — events from the hit rect drive the shared state,
        // applyViewBox writes the new viewBox onto every cell SVG.
        hits.forEach(hit => {
          hit.style.cursor = 'grab';
          hit.style.touchAction = 'none';
          hit.addEventListener('wheel', onWheel, { passive: false });
          hit.addEventListener('pointerdown', onPointerDown);
          hit.addEventListener('pointermove', onPointerMove);
          hit.addEventListener('pointerup', onPointerUp);
          hit.addEventListener('pointercancel', onPointerUp);
          hit.addEventListener('dblclick', onDblClick);
        });
        // Paint initial state on every cell.
        applyViewBox();
      }
    }

    // Save: download .svg. Button may be absent when the caller built
    // the shell with save_button=False (e.g. layouts where the first
    // <svg> isn't a meaningful save target). Guard the binding so a
    // missing button doesn't throw and abort the rest of the IIFE
    // (including the tile click-to-zoom wiring below).
    const _savebtn = wrapper.querySelector('.ocd-savebtn');
    if (_savebtn) _savebtn.addEventListener('click', () => {
      const xml = new XMLSerializer().serializeToString(svg);
      const blob = new Blob([xml], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'figure.svg';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

    // HDR toggle: flip CSS class .ocd-sdr-mode on both the inline
    // wrapper and the popup overlay so the dynamic-range-limit rule
    // applies to grid thumbnails AND the zoomed hi-res view. Button
    // may be absent (caller built with hdr_button=False); guard.
    const _hdrbtn = wrapper.querySelector('.ocd-hdrbtn');
    if (_hdrbtn) _hdrbtn.addEventListener('click', () => {
      const sdr = !wrapper.classList.contains('ocd-sdr-mode');
      wrapper.classList.toggle('ocd-sdr-mode', sdr);
      if (overlay) overlay.classList.toggle('ocd-sdr-mode', sdr);
      _hdrbtn.classList.toggle('ocd-hdr-off', sdr);
      _hdrbtn.title = sdr ? 'HDR: off (showing SDR base)' : 'HDR: on';
    });

    // Copy: rasterize SVG → PNG → clipboard. Requires the browser to be
    // able to decode any embedded raster (PNG always works; embedded JXL
    // works in Safari and Chrome-with-experimental-JXL-flag).
    const _copybtn = wrapper.querySelector('.ocd-copybtn');
    if (_copybtn) _copybtn.addEventListener('click', async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      try {
        const xml = new XMLSerializer().serializeToString(svg);
        const blob = new Blob([xml], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        await new Promise((res, rej) => {
          img.onload = res; img.onerror = rej; img.src = url;
        });
        const vb = svg.viewBox && svg.viewBox.baseVal;
        const canvas = document.createElement('canvas');
        canvas.width = (vb && vb.width) || svg.clientWidth || 1000;
        canvas.height = (vb && vb.height) || svg.clientHeight || 1000;
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        const png = await new Promise(res => canvas.toBlob(res, 'image/png'));
        await navigator.clipboard.write([new ClipboardItem({ [png.type]: png })]);
      } catch (err) {
        console.error('SvgFigure copy failed:', err);
        alert('Copy failed: ' + err.message);
      } finally {
        btn.disabled = false;
      }
    });
  })();
""".strip()


def interactive_shell(content_html: str, *,
                       save_button: bool = True,
                       copy_button: bool = True,
                       hdr_button: bool = True,
                       wrapper_style: str = '') -> str:
    """Wrap arbitrary HTML in ocdkit's interactive figure shell.

    The shell adds:

    * **click-to-zoom** for every ``<g class="fig-tile" data-bbox=…>``
      group present in ``content_html`` — pointer-driven WebGL2 / worker
      / CSS-img viewer with hover-prefetch + lazy hi-res ``data-hires-href``
      streaming + ``data-auto-upgrade="1"`` in-place upgrade. Same
      behaviour as ``image_grid`` / ``imshow``.
    * optional **save / copy buttons** (bottom-right, fade-in on hover)
      that target the first ``<svg>`` found in ``content_html``.

    Use this when building a custom layout (e.g. a figure that combines
    a metadata table with one or more ``fig-tile`` rasters) that should
    pick up the same interaction model as :class:`SvgFigure`.

    Parameters
    ----------
    content_html
        HTML to embed inside the shell wrapper. Any nested
        ``<g class="fig-tile">`` elements are auto-wired by the shell.
    save_button, copy_button
        Show the save-as-SVG / copy-as-PNG actions. Default ``True``;
        pass ``False`` for layouts where the first ``<svg>`` isn't a
        meaningful save target.
    wrapper_style
        Inline ``style="..."`` value stamped on the outer ``.ocd-svgfig``
        div. Use to override the shell's default ``display:inline-block``
        — e.g. pass ``"display:block;"`` for layouts whose own internal
        flex/grid handles horizontal centring and need the wrapper to
        take full cell width so child ``max-width`` percentages resolve
        against the cell instead of the shrink-to-fit content box.
    """
    import secrets
    uid = secrets.token_hex(6)
    css = _SHELL_CSS.replace("__UID__", uid)
    js = _SHELL_JS.replace("__UID__", uid)
    actions = ''
    if save_button or copy_button or hdr_button:
        buttons = []
        if hdr_button:
            # First in the action row (leftmost). Click toggles SDR-only
            # rendering via .ocd-sdr-mode class + dynamic-range-limit CSS.
            buttons.append(
                f'<button class="ocd-hdrbtn" title="HDR: on">'
                f'{_SHELL_HDR_ICON}</button>')
        if save_button:
            buttons.append(
                f'<button class="ocd-savebtn" title="Save as SVG">'
                f'{_SHELL_SAVE_ICON}</button>')
        if copy_button:
            buttons.append(
                f'<button class="ocd-copybtn" title="Copy as PNG">'
                f'{_SHELL_COPY_ICON}</button>')
        actions = (
            f'<div class="ocd-svgfig-actions">{"".join(buttons)}</div>'
        )
    style_attr = f' style="{wrapper_style}"' if wrapper_style else ''
    return (
        f'<div class="ocd-svgfig" data-uid="{uid}"{style_attr}>'
        f'<style>{css}</style>'
        f'{content_html}'
        f'{actions}'
        f'</div>'
        f'<div class="ocd-zoom-overlay" data-uid="{uid}">'
        f'<div class="ocd-zoom-inner"></div>'
        f'</div>'
        f'<script>{js}</script>'
    )


# Internal alias kept for back-compat with any in-tree caller; new code
# should use :func:`interactive_shell`.
_build_interactive_shell = interactive_shell


__all__ = ["SvgFigure", "Axes", "interactive_shell"]
