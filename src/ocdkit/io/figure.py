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
        # huge_tree: an SVG payload can carry many large base64 textures
        # (native-res linked tiles, big FOVs), exceeding libxml2's default
        # ~10 MB per-text-node limit. Without this, ET.fromstring raises
        # "Buffer size limit exceeded, try XML_PARSE_HUGE" on big figures.
        _svg_parser = ET.XMLParser(huge_tree=True)
        self._tree = ET.ElementTree(ET.fromstring(text, parser=_svg_parser))
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
       used by callers that pass ``fontcolor='currentColor'`` (legacy
       CSS-theme path). The default ``fontcolor='auto'`` path bypasses
       this entirely — it emits per-cell fill + stroke halo computed
       from luminance under each label region, which is the only thing
       that stays readable on arbitrary image content. */
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
  /* Clickable "Masks" label (toggles the ncolor segmentation under the
     outlines). pointer-events:all overrides the cell subtree's none. */
  .ocd-svgfig[data-uid="__UID__"] text.ocd-mask-toggle {
    cursor: pointer; pointer-events: all;
    /* keep wiggle pivoting on the glyph box, not the SVG origin */
    transform-box: fill-box; transform-origin: center;
  }
  .ocd-svgfig[data-uid="__UID__"] text.ocd-mask-toggle:hover {
    font-weight: bold;
    animation: ocd-mask-wiggle__UID__ 0.4s ease-in-out;
  }
  @keyframes ocd-mask-wiggle__UID__ {
    0%   { transform: rotate(0deg); }
    25%  { transform: rotate(-4deg); }
    50%  { transform: rotate(3deg); }
    75%  { transform: rotate(-2deg); }
    100% { transform: rotate(0deg); }
  }
  .ocd-svgfig[data-uid="__UID__"] .ocd-svgfig-actions {
    /* Flow BELOW the figure (the actions div follows the content in the DOM),
       not absolutely overlaid on the plot. ``inline-block`` wrapper grows to
       include this row, so the buttons always sit clear of the lowest plot
       element instead of covering the bottom-right of the spectra panel. */
    display: flex;
    gap: 8px;
    justify-content: flex-end;
    margin-top: 6px;
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
  /* Per-cell adaptive ``image_grid`` labels. Each ``<text>`` carries
     ``--ocd-tt-hdr`` and ``--ocd-tt-sdr`` custom properties — the
     Python-side luminance sampler emits the contrast pick under each
     assumption. The rules below pick the right one for current viewing
     conditions:
       1. Default → SDR pick (works on any display, including SDR-only).
       2. ``@media (dynamic-range: high)`` → display can render HDR, so
          the UHDR gain map IS being composited; switch to the HDR pick.
       3. ``.ocd-sdr-mode`` (HDR toggle off in the shell) → user
          explicitly asked for SDR rendering even on an HDR display.
          Force the SDR pick back.
     NOTE: We use ``dynamic-range: high`` not ``standard`` because both
     queries match on HDR displays (HDR-capable monitors can also show
     standard content), so keying off ``standard`` would always fire
     and the HDR pick would never win.
     Plain SVG ``fill`` attribute is intentionally NOT set inline (only
     CSS custom props) so author-stylesheet ``fill`` rules can win the
     cascade. */
  /* Match both the parent ``<text>`` (auto-cell mode: single fill,
     CSS vars set on the text element) AND its descendant ``<tspan>``s
     (per-letter mode: each tspan has its own pair of CSS vars). The
     ``var()`` lookup happens at the element where ``fill`` is computed,
     so applying the rule to both targets makes either mode work. */
  .ocd-svgfig[data-uid="__UID__"] text.ocd-adaptive-text,
  .ocd-svgfig[data-uid="__UID__"] text.ocd-adaptive-text > tspan {
    fill: var(--ocd-tt-sdr, currentColor);
  }
  @media (dynamic-range: high) {
    .ocd-svgfig[data-uid="__UID__"] text.ocd-adaptive-text,
    .ocd-svgfig[data-uid="__UID__"] text.ocd-adaptive-text > tspan {
      fill: var(--ocd-tt-hdr, currentColor);
    }
  }
  .ocd-svgfig[data-uid="__UID__"].ocd-sdr-mode text.ocd-adaptive-text,
  .ocd-svgfig[data-uid="__UID__"].ocd-sdr-mode text.ocd-adaptive-text > tspan,
  .ocd-zoom-overlay[data-uid="__UID__"].ocd-sdr-mode text.ocd-adaptive-text,
  .ocd-zoom-overlay[data-uid="__UID__"].ocd-sdr-mode text.ocd-adaptive-text > tspan {
    fill: var(--ocd-tt-sdr, currentColor);
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
    // ─── self-pruning window listeners ─────────────────────────────────
    // Per-display ``window`` resize/scroll/keydown handlers would
    // otherwise accumulate one set per figure across a notebook session:
    // anonymous handlers can't be removed, and the wrapper carries no
    // unload hook. ``onWindow`` wraps each handler so that the first time
    // it fires after this figure's wrapper has left the DOM (cell re-run,
    // cleared output) it removes itself — bounding the live listener count
    // to on-screen figures without a document-wide MutationObserver.
    const onWindow = (type, handler, opts) => {
      const wrapped = (e) => {
        if (!wrapper.isConnected) {
          window.removeEventListener(type, wrapped, opts);
          return;
        }
        handler(e);
      };
      window.addEventListener(type, wrapped, opts);
    };
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
      let hiresHref = tile.getAttribute('data-hires-href');
      // Remote pages: route a baked-loopback hi-res URL (e.g. a tileserve
      // /attach/<sid>/<name> served on 127.0.0.1) through the Jupyter-origin
      // proxy so click-to-expand loads full-res off-machine. No-op for local
      // pages and for data:/relative hrefs.
      if (hiresHref && window.__ocdResolveTileUrl) hiresHref = window.__ocdResolveTileUrl(hiresHref);
      // Live label tile → render in the popup via createLabelViewer (same
      // LabelGLRenderer as the inline tile), so the zoom gets palette fill
      // + outlines + HDR boundary AND live hover-highlight. No <image>
      // snapshot, no <foreignObject> re-raster.
      let labelCv = tile.querySelector('canvas[data-label-tile]');
      if (!labelCv) {
        // Fallback: querySelector type-selectors for HTML elements inside
        // an SVG <foreignObject> are finicky in some engines.
        const cs = tile.getElementsByTagName('canvas');
        for (let i = 0; i < cs.length; i++) {
          if (cs[i].hasAttribute('data-label-tile')) { labelCv = cs[i]; break; }
        }
      }
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
      // Label tiles and the legacy SVG viewer are per-tile (bound to a
      // specific source) and can't be recycled — rebuild on every open.
      const needLegacyRebuild = webglViewer && webglViewer.isLegacy;
      const needLabelRebuild = !!labelCv || (webglViewer && webglViewer.isLabel);
      const firstBuild = (!webglViewer || needLegacyRebuild || needLabelRebuild);
      if (firstBuild) {
        if ((needLegacyRebuild || needLabelRebuild) && webglViewer) {
          try { webglViewer.dispose(); } catch (_) {}
          webglViewer = null;
        }
        overlayInner.innerHTML = '';
        canvasEl = document.createElement('div');
        canvasEl.className = 'ocd-zoom-canvas';
        overlayInner.appendChild(canvasEl);

        // Live label tile: render via the shared LabelGLRenderer.
        if (labelCv) {
          webglViewer = createLabelViewer(canvasEl, labelCv);
          if (webglViewer) webglViewer.isLabel = true;
        }
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
        if (!webglViewer && (viewerHint === 'webgl' || viewerHint === 'worker')) {
          webglViewer = createPopupWorkerViewer(canvasEl)
                     || createPopupWebglViewer(canvasEl);
        } else if (!webglViewer) {
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
      if (viewerAtOpen && viewerAtOpen.isLabel) {
        // Live label viewer: nothing to fetch — its data is already in
        // the renderer. Just show the popup and fit.
        showPopup();
        if (webglViewer === viewerAtOpen) applyTransform();
      } else if (thumbHref) {
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

    // Live label-tile popup viewer: renders the segmentation through the
    // SAME LabelGLRenderer as the inline tile, so the popup gets palette
    // fill + outlines + HDR boundary AND live hover-highlight (the inline
    // tile's interactivity, now in the zoom). Mirrors the WebGL image
    // viewer's transform (LabelGL.mat3ForFit) so pan/zoom feels identical.
    function createLabelViewer(parent, srcCanvas) {
      if (!self.LabelGL) return null;
      const canvas = document.createElement('canvas');
      // pointer-events:none so the popup's pan/zoom handlers (on parent)
      // own the gestures; hover is wired on ``parent`` below.
      canvas.style.cssText =
        'position:absolute; inset:0; width:100%; height:100%;'
      + ' image-rendering:pixelated; pointer-events:none;';
      parent.appendChild(canvas);
      const gl = canvas.getContext('webgl2',
        { alpha: true, premultipliedAlpha: false });
      if (!gl) { if (canvas.parentElement) canvas.parentElement.removeChild(canvas); return null; }
      // HDR float16 extended-range backbuffer (see inline controller) so >1.0
      // outline/highlight colors emit TRUE HDR. Needs EXT_color_buffer_float;
      // (re)allocated in syncSize on every resize. SDR fallback if absent.
      const _hdr = !!gl.drawingBufferStorage;
      if (_hdr) {
        try {
          gl.getExtension('EXT_color_buffer_float');
          gl.drawingBufferColorSpace = 'display-p3';
        } catch (e) {}
      }
      let cfg = srcCanvas.__labelCfg || self.LabelGL.decodeAttrs(srcCanvas);
      const imgW = cfg.w, imgH = cfg.h;
      let lastState = { s: 1, tx: 0, ty: 0 };
      // Base image: mirror the THUMBNAIL — an HTML <img> of the tile's sibling
      // SVG <image> placed BEHIND the transparent canvas, so a uhdr base keeps
      // its gain-map HDR (uploading it into the GPU as an 8-bit texture would
      // flatten it to SDR). The <img>'s CSS transform tracks the canvas's
      // pan/zoom (see redraw). The label canvas above stays transparent
      // (baseSrc is NOT passed to buildRenderer → imageVisible 0).
      let baseImg = null;
      {
        const g = srcCanvas.closest && srcCanvas.closest('g.fig-tile');
        const sib = g && g.querySelector('image');
        const href = (cfg.baseSrc) || (sib && (sib.getAttribute('href')
          || sib.getAttributeNS('http://www.w3.org/1999/xlink', 'href')));
        if (href) {
          baseImg = document.createElement('img');
          baseImg.crossOrigin = 'anonymous';
          baseImg.style.cssText = 'position:absolute; left:0; top:0;'
            + ' pointer-events:none; image-rendering:pixelated;'
            + ' transform-origin:0 0; will-change:transform;';
          baseImg.style.width = imgW + 'px';
          baseImg.style.height = imgH + 'px';
          baseImg.onload = function () { redraw(lastState); };
          parent.insertBefore(baseImg, canvas);   // behind the label canvas
          baseImg.src = href;
        } else {
          // No base image (a pure segmentation tile): give the popup the same
          // solid themed backdrop as the inline tile's <rect fill="Canvas">,
          // so the semi-transparent cells composite over it (black in dark
          // mode, white in light) instead of the bare zoom overlay. The popup
          // overlay lives on document.body and does NOT inherit the figure's
          // color-scheme, so opt this canvas in explicitly or ``Canvas``
          // resolves to its light value (white) even in dark mode.
          canvas.style.colorScheme = 'light dark';
          canvas.style.backgroundColor = 'Canvas';
        }
      }
      let r;
      try {
        // No baseSrc → buildRenderer leaves the canvas transparent (labels/
        // outlines only); the HDR <img> above provides the base.
        r = self.LabelGL.buildRenderer(gl, cfg, () => redraw(lastState));
      } catch (e) {
        console.warn('LabelGL popup:', e);
        if (baseImg && baseImg.parentElement) baseImg.parentElement.removeChild(baseImg);
        if (canvas.parentElement) canvas.parentElement.removeChild(canvas);
        return null;
      }
      // HDR toggle response (mirror the inline controller): SDR → boosts to
      // 1.0 (SDR white); HDR → configured boosts. Applied on open + on toggle.
      const _cfgOutlineHdr = (cfg.uniforms && cfg.uniforms.outlineHdrBoost) || 1.0;
      function setSdr(sdr) {
        r.setUniforms({ outlineHdrBoost: sdr ? 1.0 : _cfgOutlineHdr,
                        highlightBoost: sdr ? 1.0 : 1.8 });
        redraw(lastState);
      }
      let _cssW = 0, _cssH = 0, _dpr = 1, _dirty = true;
      function invalidateSize() { _dirty = true; }
      canvas.__invalidateSize = invalidateSize;
      function syncSize() {
        if (!_dirty) return;
        _cssW = canvas.clientWidth; _cssH = canvas.clientHeight;
        _dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.round(_cssW * _dpr));
        const h = Math.max(1, Math.round(_cssH * _dpr));
        // Always size via width/height first (full resolution + SDR fallback),
        // then upgrade the same-size buffer to float16 HDR in place. If
        // drawingBufferStorage errors it's a no-op and the SDR buffer remains,
        // so resolution is never lost.
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        if (_hdr) { try { gl.drawingBufferStorage(gl.RGBA16F, w, h); } catch (e) {} }
        _dirty = false;
      }
      function redraw(s) {
        lastState = s;
        syncSize();
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
        r.draw(self.LabelGL.mat3ForFit(s, imgW, imgH, _cssW, _cssH));
        // Track the HDR base <img> to the SAME fit+pan+zoom the shader uses
        // (mat3ForFit's geometry), so the browser-composited HDR base stays
        // pixel-aligned with the GPU label/outline overlay.
        if (baseImg) {
          const sc = (s && s.s) || 1, tx = (s && s.tx) || 0, ty = (s && s.ty) || 0;
          const fit = Math.min(_cssW / imgW, _cssH / imgH);
          const ox = (_cssW * 0.5 - imgW * fit * 0.5) * sc + tx;
          const oy = (_cssH * 0.5 - imgH * fit * 0.5) * sc + ty;
          baseImg.style.transform =
            'translate(' + ox + 'px,' + oy + 'px) scale(' + (fit * sc) + ')';
        }
      }
      function loadImage(url, onLoaded) { if (onLoaded) onLoaded(); }  // data is in cfg
      function isPointInImage(clientX, clientY) {
        const rc = canvas.getBoundingClientRect();
        return !!self.LabelGL.imagePointFromCss(lastState, imgW, imgH,
          rc.width, rc.height, clientX - rc.left, clientY - rc.top);
      }
      // hover-highlight + tooltip, wired on the gesture surface (parent)
      let tip = null, cur = 0;
      function onMove(e) {
        const rc = canvas.getBoundingClientRect();
        const p = self.LabelGL.imagePointFromCss(lastState, imgW, imgH,
          rc.width, rc.height, e.clientX - rc.left, e.clientY - rc.top);
        const id = p ? r.labelAt(p.px, p.py) : 0;
        if (id !== cur) { cur = id; r.setUniforms({ highlightLabel: id }); redraw(lastState); }
        if (id > 0) {
          if (!tip) {
            tip = document.createElement('div');
            tip.style.cssText = 'position:fixed;pointer-events:none;z-index:2147483647;'
              + 'background:rgba(20,20,20,.92);color:#eee;font:11px sans-serif;'
              + 'padding:2px 6px;border-radius:4px;';
            document.body.appendChild(tip);
          }
          tip.textContent = 'label ' + id; tip.style.display = 'block';
          tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px';
        } else if (tip) { tip.style.display = 'none'; }
      }
      function onLeave() {
        cur = 0; r.setUniforms({ highlightLabel: 0 }); redraw(lastState);
        if (tip) tip.style.display = 'none';
      }
      parent.addEventListener('pointermove', onMove);
      parent.addEventListener('pointerleave', onLeave);
      // Clear any active hover (highlight + floating tooltip) WITHOUT
      // tearing the viewer down. closeZoom keeps the viewer alive for the
      // texture LRU, so it never calls dispose() — but the ``tip`` div
      // lives on document.body and ``pointerleave`` doesn't necessarily
      // fire when the overlay is hidden via Esc / click-outside, leaving
      // the 'label N' tooltip stuck on the page. closeZoom calls this.
      function hideHover() {
        cur = 0;
        try { r.setUniforms({ highlightLabel: 0 }); redraw(lastState); } catch (_) {}
        if (tip) tip.style.display = 'none';
      }
      function dispose() {
        parent.removeEventListener('pointermove', onMove);
        parent.removeEventListener('pointerleave', onLeave);
        if (tip && tip.parentElement) tip.parentElement.removeChild(tip);
        tip = null;
        if (baseImg && baseImg.parentElement) baseImg.parentElement.removeChild(baseImg);
        baseImg = null;
        try { const ext = gl.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); } catch (_) {}
        if (canvas.parentElement) canvas.parentElement.removeChild(canvas);
      }
      function clearActive() {}  // one tile per popup session
      // Apply the figure's current HDR-toggle state to this fresh popup
      // viewer (the overlay carries .ocd-sdr-mode); setUniforms only — the
      // first real paint comes from openZoom's applyTransform.
      try { r.setUniforms({
        outlineHdrBoost: overlay.classList.contains('ocd-sdr-mode') ? 1.0 : _cfgOutlineHdr,
        highlightBoost: overlay.classList.contains('ocd-sdr-mode') ? 1.0 : 1.8 }); } catch (_) {}
      return { canvas, redraw, loadImage, isPointInImage, dispose, clearActive,
               hideHover, setSdr,
               get textureLoaded() { return true; },
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
      // Label-tile popups keep a floating 'label N' tooltip on document.body
      // and persist the viewer across closes (texture LRU). Clear the hover
      // so the tooltip + highlight don't stick to the page after close.
      if (webglViewer && webglViewer.hideHover) webglViewer.hideHover();
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
      let hiresHref = tile.getAttribute('data-hires-href');
      // remote pages: resolve the baked-loopback hi-res URL to the Jupyter proxy
      // (so the hover-prefetch / auto-upgrade swap works off-machine too)
      if (hiresHref && window.__ocdResolveTileUrl) hiresHref = window.__ocdResolveTileUrl(hiresHref);
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
      onWindow('resize', () => {
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
      // Array (not the raw NodeList): the GL refine path calls cells.every /
      // cells.some (3152/3196), which NodeList lacks → "cells.every is not a
      // function" would throw mid-refine, leaving tiles stuck on the coarse
      // level (never sharpening to full res).
      const cells = Array.from(svg.querySelectorAll('svg.ocd-linked-cell'));
      const hits = svg.querySelectorAll('rect.ocd-linked-cell-hit');
      if (cells.length > 0 && hits.length === cells.length) {
        const RAS_W = parseFloat(svg.dataset.linkRasterW) || 1;
        const RAS_H = parseFloat(svg.dataset.linkRasterH) || 1;
        // Tile pyramid levels are indexed COARSEST→FINEST (level 0 = coarsest,
        // NLEV-1 = full res), per block_mean_pyramid. NLEV lets us pick a
        // display-sized level + know which index is full res.
        const NLEV = Math.max(1, parseInt(svg.dataset.linkNlevels) || 1);
        const FULL_LEVEL = NLEV - 1;
        // Outline stroke knobs (emitter side picks the markup): cell
        // outlines use ``stroke-width: var(--ocd-osw, <image-px>)`` so they
        // scale with zoom by default. When a screen-px floor is requested
        // we set --ocd-osw per frame to max(image-px-in-vb-units, floor),
        // keeping outlines visible when zoomed far out without re-emitting.
        const _outlineBase = parseFloat(svg.dataset.linkOutlinePx) || 1;
        const _outlineMinPx = parseFloat(svg.dataset.linkOutlineMinPx) || 0;
        // Cached cell display width for the outline-min calc — invalidated on
        // resize so applyViewBox never getBoundingClientRect()s (reflow) per frame.
        let _dispWCache = 0;
        onWindow('resize', () => { _dispWCache = 0; });
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
        const state = { x: initX, y: initY, w: initW, h: initH, r: 0 };
        const target = { x: initX, y: initY, w: initW, h: initH, r: 0 };
        // Zoom limits: max-zoom-in = 8 source-px window, max-zoom-out =
        // 8× full image. Aspect lock means we only need a single
        // ``scale`` (= w / initW) — w and h move together.
        const MIN_W = Math.max(8, initW / 200);
        const MAX_W = Math.min(RAS_W * 8, initW * 200);
        const TWEEN_ALPHA = 0.35;
        const TWEEN_EPS = 0.25;
        let _raf = 0;
        let _gestureActive = false;   // a trackpad gesture owns zoom+rotate
        // Inverse-rotation fraction map: a rotated display fraction (fx,fy in
        // [0,1] over the cell) → the un-rotated viewport fraction. Used to
        // anchor rotate/zoom at the cursor instead of the viewport centre:
        // src-under-cursor = vp.xy + _unrotFrac(f, r) * vp.wh.
        function _unrotFrac(fx, fy, r) {
          const cx = (fx - 0.5) * cellAR, cy = (fy - 0.5);
          const cs = Math.cos(-r), sn = Math.sin(-r);
          return { x: (cs * cx - sn * cy) / cellAR + 0.5, y: (sn * cx + cs * cy) + 0.5 };
        }

        // ─── WebGL image layer (fast pinch) ───────────────────────────
        // The SVG-viewBox path re-rasterizes every embedded <image> on
        // each pointer/wheel event — fine for drag/scroll, but pinch's
        // high event rate makes that visibly slow on big rasters. Here a
        // single WebGL2 canvas sits BEHIND the SVG and renders each cell's
        // texture via a shared-viewport uniform (no re-raster). We hide
        // the SVG <image>s (WebGL draws them) but keep the SVG cells'
        // vector outlines/labels on top — they re-flow cheaply on viewBox
        // change. Falls back to the pure-SVG path when WebGL2 is absent.
        function createLinkedGLLayer() {
          if (!window.WebGL2RenderingContext) return null;
          const imgs = [];
          for (let i = 0; i < cells.length; i++) imgs.push(cells[i].querySelector('image'));
          if (!imgs.some(Boolean)) return null;
          const host = svg.closest('.ocd-svgfig') || svg.parentElement;
          if (!host) return null;
          const canvas = document.createElement('canvas');
          canvas.style.position = 'absolute';
          canvas.style.left = '0'; canvas.style.top = '0';
          canvas.style.pointerEvents = 'none';
          canvas.style.zIndex = '0';
          svg.style.position = 'relative';
          svg.style.zIndex = '1';
          host.insertBefore(canvas, host.firstChild);
          const gl = canvas.getContext('webgl2',
            { antialias: false, premultipliedAlpha: false, alpha: true,
              // keep the framebuffer so the copy/save compositor can drawImage()
              // this canvas after its frame (else it reads back empty/cleared).
              preserveDrawingBuffer: true });
          if (!gl) { host.removeChild(canvas); return null; }
          const VS = '#version 300 es\n'
            + 'in vec2 a_pos;out vec2 v_uv;uniform vec2 u_img;uniform vec4 u_vp;uniform float u_rot;'
            + 'void main(){vec2 s=a_pos*u_img;vec2 f0=(s-u_vp.xy)/u_vp.zw;'
            // rotate around the viewport center in aspect-correct (display) space
            + 'float ar=u_vp.z/u_vp.w;vec2 d=(f0-0.5)*vec2(ar,1.0);'
            + 'float cs=cos(u_rot),sn=sin(u_rot);'
            + 'vec2 dr=vec2(cs*d.x-sn*d.y,sn*d.x+cs*d.y);vec2 f=dr/vec2(ar,1.0)+0.5;'
            + 'gl_Position=vec4(f.x*2.0-1.0,1.0-f.y*2.0,0.0,1.0);v_uv=a_pos;}';
          // Raw-tile colormap: intensity (R32F) → normalize(lo,hi) → LUT;
          // rgb (RGBA32F) → passthrough. Matches the live grid's CFS shader so
          // the cmap picker + self/global/bit-depth readout-norm are uniforms.
          // u_mode: 0 = intensity (R32F → normalize(lo,hi) → LUT);
          //         1 = passthrough RGBA (already display-encoded: mask/alts);
          //         2 = LINEAR Display-P3 RGB → sRGB OETF (the make_rgb tile,
          //             used only for EXPORT — on screen the WebGPU exc layer
          //             owns the RGB; the GL passthrough underneath is linear).
          const FS = '#version 300 es\nprecision highp float;in vec2 v_uv;'
            + 'out vec4 o;uniform sampler2D u_tex;uniform sampler2D u_lut;'
            + 'uniform float u_lo;uniform float u_hi;uniform int u_mode;'
            + 'vec3 oetf(vec3 x){x=clamp(x,0.0,1.0);'
            + 'return mix(12.92*x, 1.055*pow(x,vec3(1.0/2.4))-0.055, step(0.0031308,x));}'
            + 'void main(){ if(u_mode==1){ o=texture(u_tex,v_uv); }'
            + ' else if(u_mode==2){ vec4 c=texture(u_tex,v_uv); o=vec4(oetf(c.rgb), c.a); }'
            + ' else { float v=texture(u_tex,v_uv).r;'
            + ' float n=clamp((v-u_lo)/max(u_hi-u_lo,1e-12),0.0,1.0);'
            + ' o=texture(u_lut, vec2(n,0.5)); } }';
          function sh(t, s) {
            const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
            if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) {
              console.warn('linkedGL shader:', gl.getShaderInfoLog(x)); return null;
            }
            return x;
          }
          const prog = gl.createProgram();
          const vs = sh(gl.VERTEX_SHADER, VS), fs = sh(gl.FRAGMENT_SHADER, FS);
          if (!vs || !fs) { host.removeChild(canvas); return null; }
          gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
          if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { host.removeChild(canvas); return null; }
          gl.useProgram(prog);
          const imgVAO = gl.createVertexArray();
          gl.bindVertexArray(imgVAO);
          const vbo = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
          gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,0,1,0,0,1,1,1]), gl.STATIC_DRAW);
          const aPos = gl.getAttribLocation(prog, 'a_pos');
          gl.enableVertexAttribArray(aPos);
          gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
          gl.bindVertexArray(null);
          const U = { img: gl.getUniformLocation(prog, 'u_img'),
                      vp: gl.getUniformLocation(prog, 'u_vp'),
                      rot: gl.getUniformLocation(prog, 'u_rot'),
                      tex: gl.getUniformLocation(prog, 'u_tex'),
                      lut: gl.getUniformLocation(prog, 'u_lut'),
                      lo: gl.getUniformLocation(prog, 'u_lo'),
                      hi: gl.getUniformLocation(prog, 'u_hi'),
                      mode: gl.getUniformLocation(prog, 'u_mode') };
          gl.useProgram(prog); gl.uniform1i(U.tex, 0); gl.uniform1i(U.lut, 1);
          gl.getExtension('OES_texture_float_linear');   // LINEAR on float tex
          // ── colormap LUT (256×1 RGBA8) on unit 1; swapped live by the picker ──
          const LUTS = (window.OCD_LUTS || {});
          let CMAP = (LUTS.magma ? 'magma' : Object.keys(LUTS)[0]) || 'magma';
          let NORMMODE = 'self';                       // self | global | bitdepth
          let EXPORT_OETF = false;                     // OETF the linear RGB tile for PNG capture
          const lutTex = gl.createTexture();
          gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, lutTex);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          function uploadLUT(name) {
            const a = LUTS[name]; if (!a) return;
            gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, lutTex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA,
                          gl.UNSIGNED_BYTE, new Uint8Array(a));
            gl.activeTexture(gl.TEXTURE0);
          }
          uploadLUT(CMAP);
          gl.activeTexture(gl.TEXTURE0);
          const tileMeta = new Array(cells.length).fill(null);  // per-cell raw hdrs
          let RGLO = Infinity, RGHI = -Infinity, BITMAX = 65535; // readout pool
          const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
          const textures = new Array(cells.length).fill(null);
          function _newTex() {
            const tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA,
                          gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,0]));
            return tex;
          }
          // Each cell's RAW tile URL: derive ``fmt=raw`` from ``data-tile-src``
          // (async) or the baked ``href``. The GL layer fetches raw float32 +
          // the X-Mode/X-Lo/X-Hi/X-Kind/X-Bitmax headers and colormaps on the
          // GPU — NO PNG encode/decode (same path + speed as the live grid).
          function _rawUrl(im, lvl) {
            let s = im.getAttribute('data-tile-src')
              || im.getAttribute('href')
              || im.getAttributeNS('http://www.w3.org/1999/xlink', 'href') || '';
            if (!s) return '';
            // set the pyramid level (the segment before ?); server clamps to
            // the available levels and reports the actual one via X-Level.
            if (lvl != null) s = s.replace(/\/(\d+)(\?|$)/, '/' + (lvl | 0) + '$2');
            const u = s.indexOf('fmt=') >= 0 ? s.replace(/fmt=[a-z0-9]+/i, 'fmt=raw')
              : (s + (s.indexOf('?') >= 0 ? '&' : '?') + 'fmt=raw');
            // remote pages: route the loopback tile URL through the Jupyter proxy
            return (window.__ocdResolveTileUrl ? window.__ocdResolveTileUrl(u) : u);
          }
          // ── Per-(cell,level) pyramid texture cache — mirrors plot_key_slices_live ──
          // The live grid keeps EVERY fetched level in a Map keyed label/level so
          // it can (a) paint the best cached level INSTANTLY (re-zoom never
          // refetches) and (b) refine ONE level at a time (progressive sharpen).
          // The old path kept a single per-cell texture and, on zoom, fetched
          // every cell's full-res level in one Promise.all (~192 MB) → multi-second
          // stalls and a full refetch on every re-zoom. This is the live-grid
          // strategy: ``textures[i]``/``tileMeta[i]``/``tileLevel[i]`` are just the
          // currently-DRAWN pointers, repointed (no upload) by _selectBest as the
          // view changes; tileCache[i] holds the uploaded textures per level.
          const tileCache = Array.from({ length: cells.length }, () => new Map()); // lvl -> {tex,t}
          const tileLevel = new Array(cells.length).fill(-1);  // currently-DRAWN level (-1 = none)
          // Coarsest level (index, width) whose width ≥ the cell's on-screen px at
          // the current zoom: FULL_LEVEL - floor(log2(rasVisible/cw)). ``state.w``
          // is the raster px currently spanning the cell width (= RAS_W at default
          // zoom, shrinks as you zoom in → finer target).
          function _targetLevel(i) {
            let cw = 0;
            try {
              cw = ((_crects && _crects[i]) ? _crects[i].w : hits[i].getBoundingClientRect().width)
                   * (window.devicePixelRatio || 1);
            } catch (e) {}
            if (cw < 1) return FULL_LEVEL;
            const span = Math.max(1, state.w);
            return Math.max(0, Math.min(FULL_LEVEL, FULL_LEVEL - Math.floor(Math.log2(span / cw))));
          }
          // Point the draw state at the finest CACHED level ≤ target — pure pointer
          // swap, no GL upload, so render() is always instant. Returns true if any
          // level is available to draw.
          function _selectBest(i, target) {
            for (let l = target; l >= 0; l--) {
              const e = tileCache[i].get(l);
              if (e) {
                if (tileLevel[i] !== l) { textures[i] = e.tex; tileMeta[i] = e.t; tileLevel[i] = l; }
                return true;
              }
            }
            return false;
          }
          function _uploadTile(t) {              // one level → a GL texture
            const tex = _newTex();
            gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            if (t.mode === 'intensity') {
              gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, t.w, t.h, 0, gl.RED,
                            gl.FLOAT, new Float32Array(t.buf));
              if (t.kind === 'readout') {
                RGLO = Math.min(RGLO, t.lo); RGHI = Math.max(RGHI, t.hi); BITMAX = t.bitmax;
              }
            } else {                                   // rgb float32 (ch 3 or 4)
              const src = new Float32Array(t.buf), px = t.w * t.h;
              let rgba;
              if (t.ch === 4) { rgba = src; }
              else {
                rgba = new Float32Array(px * 4);
                for (let p = 0, q = 0; p < px; p++) {
                  rgba[q++] = src[p*3]; rgba[q++] = src[p*3+1];
                  rgba[q++] = src[p*3+2]; rgba[q++] = 1.0;
                }
              }
              gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, t.w, t.h, 0, gl.RGBA,
                            gl.FLOAT, rgba);
            }
            return tex;
          }
          // Fetch ONE (cell, level), upload + cache it. Resolves true on success,
          // false on 204 / not-ready / error (caller retries). No-op if cached.
          function _fetchLevel(i, level) {
            if (tileCache[i].has(level)) return Promise.resolve(true);
            const im = imgs[i]; if (!im) return Promise.resolve(false);
            const _t0 = (window.performance && performance.now) ? performance.now() : 0;
            return fetch(_rawUrl(im, level)).then(r => {
              if (r.status === 204) return null;
              if (!r.ok) throw new Error('tile ' + r.status);
              const w = +r.headers.get('X-Level-Width'), h = +r.headers.get('X-Level-Height');
              const ch = +(r.headers.get('X-Channels') || '1');
              const mode = r.headers.get('X-Mode') || 'intensity';
              const lo = parseFloat(r.headers.get('X-Lo') || '0');
              const hi = parseFloat(r.headers.get('X-Hi') || '1');
              const kind = r.headers.get('X-Kind') || 'reduction';
              const bitmax = parseFloat(r.headers.get('X-Bitmax') || '65535');
              const lvl = +(r.headers.get('X-Level') || String(level));
              return r.arrayBuffer().then(buf => ({ w, h, ch, mode, lo, hi, kind, bitmax, level: lvl, buf }));
            }).then(t => {
              if (!t) return false;
              const _t1 = (window.performance && performance.now) ? performance.now() : 0;
              const tex = _uploadTile(t);
              tileCache[i].set(t.level, { tex, t });
              if (window.__ocdLog) {
                const _t2 = (window.performance && performance.now) ? performance.now() : 0;
                window.__ocdLog('tile ' + i + ' (' + t.mode + ' L' + t.level + ') fetch='
                  + Math.round(_t1 - _t0) + 'ms up=' + Math.round(_t2 - _t1) + 'ms');
              }
              if (im && im.style) im.style.display = 'none';
              return true;
            }).catch(() => false);
          }
          // Synchronous placeholder per cell so the guard + render loop have a
          // texture; tileMeta gates actual drawing until a real level lands.
          for (let i = 0; i < cells.length; i++) { if (imgs[i]) textures[i] = _newTex(); }

          // INITIAL load: fetch the COARSEST level (0) for every cell in one batch
          // → instant first paint (124² ≈ 60 KB each), exactly like the live grid's
          // poll(). Then redraw() schedules progressive refinement up to the
          // display level. 204s (layer not projected yet) retry on a single timer.
          let _tileTries = 0;
          function _pollTiles(onDrain) {
            const want = [];
            for (let i = 0; i < cells.length; i++) if (imgs[i] && !tileCache[i].has(0)) want.push(i);
            if (!want.length) { if (onDrain) onDrain(); redraw(); return; }
            Promise.all(want.map(i => _fetchLevel(i, 0))).then(res => {
              if (res.some(Boolean)) redraw();        // coalesced → 1 draw/batch
              const allReady = cells.every((c, i) => !imgs[i] || tileCache[i].has(0));
              if (allReady) { if (onDrain) onDrain(); redraw(); }
              else if (_tileTries++ < 600)
                // Backoff GROWS to ~2 s: real tiles fill on the bg thread within
                // ~1-2 s (caught by the fast early polls); cells with NO data
                // (blank readouts) 204 forever, so polling them every 150 ms for
                // 90 s just churns the HTTP/1.1 connection pool and contends with
                // the zoom refine. Slow polls = low churn, still recovers a late fill.
                setTimeout(() => _pollTiles(onDrain), Math.min(2000, 80 + _tileTries * 60));
              else if (onDrain) onDrain();
            });
          }
          _pollTiles();
          // Zoom-refine — exactly plot_key_slices_live's ``refine``: take ONE
          // progressive step toward each cell's target level (so it sharpens, not
          // pops, and a fast zoom doesn't fetch every level it flies past), cache
          // it, repaint, then re-schedule until every cell reaches its target.
          // Already-cached levels are reused with NO refetch — re-zoom is instant.
          let _refineBusy = false;
          function _refineLevels() {
            if (_refineBusy || FULL_LEVEL <= 0) return;
            const want = [];
            for (let i = 0; i < cells.length; i++) {
              if (!imgs[i] || !tileCache[i].has(0)) continue;   // not initially loaded yet
              const tgt = _targetLevel(i);
              let have = -1;
              for (let l = tgt; l >= 0; l--) { if (tileCache[i].has(l)) { have = l; break; } }
              const step = Math.min(tgt, have + 1);
              if (!tileCache[i].has(step)) want.push([i, step]);
            }
            if (!want.length) return;
            _refineBusy = true;
            // ONE paint after the whole batch lands — exactly like
            // plot_key_slices_live's refine() (fetch the level for all cells, then
            // render() once). Painting per-tile (a previous attempt) ran a FULL
            // redraw — including the ~208k-edge outline pass — after every tile,
            // i.e. N redraws per level, which blocked the main thread and inflated
            // each tile's fetch .then latency. The cached coarser level already
            // shows (scaled) via the sync redraw on zoom, so there's no blank gap
            // to fill with per-tile paints.
            Promise.all(want.map(([i, l]) => _fetchLevel(i, l))).then(res => {
              _refineBusy = false;
              if (!res.some(Boolean)) return;
              _paint();
              const more = cells.some((c, i) =>
                imgs[i] && tileCache[i].has(0) && !tileCache[i].has(_targetLevel(i)));
              if (more) scheduleRefine();
            });
          }
          let _refineTimer = null;
          function scheduleRefine() {
            if (_refineTimer) clearTimeout(_refineTimer);
            _refineTimer = setTimeout(_refineLevels, 60);
          }
          if (!textures.some(Boolean)) { host.removeChild(canvas); return null; }

          // Alt textures: the clickable "Masks" label CYCLES the cell through
          // its alt rasters. Currently only ``data-alt-href`` is emitted (the
          // pixel-exact ncolor raster), so this is a main↔ncolor toggle;
          // ``data-alt2-href`` is an optional extra-state hook (handled here so
          // adding a second raster needs no JS change). State 0 = main texture;
          // 1..N = the alts. The GPU outlines draw afterward, so they stay on
          // top of whichever is shown.
          const altTexLists = new Array(cells.length).fill(null);   // [tex,…] per cell
          const altState = new Array(cells.length).fill(0);         // 0=main; k=alt k-1
          const altLoaders = new Array(cells.length).fill(null);    // deferred fetch fns
          for (let i = 0; i < cells.length; i++) {
            const im = imgs[i]; if (!im) continue;
            const hrefs = [im.getAttribute('data-alt-href'),
                           im.getAttribute('data-alt2-href')].filter(Boolean);
            if (!hrefs.length) continue;
            const texList = [];
            for (let h = 0; h < hrefs.length; h++) {
              const altHref = hrefs[h];
              // The per-cell pixel-label raster (h>=1) wants NEAREST so the cell
              // boundaries stay pixel-crisp ("exact pixel masks"); ncolor (h==0)
              // keeps a smoothed minify so the flat fill reads cleanly.
              const minF = (h >= 1) ? gl.NEAREST : gl.LINEAR;
              const tex = gl.createTexture();
              gl.bindTexture(gl.TEXTURE_2D, tex);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minF);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
              gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                            new Uint8Array([0,0,0,0]));
              texList.push(tex);
              const idx = i, myk = h + 1;
              // The rasters fill on a bg thread → poll (204 = retry); blob URL is
              // same-origin so the Image→texture upload doesn't taint.
              let atries = 0;
              const loadAlt = () => fetch(altHref).then(r => {
                if (r.status === 204) { if (atries++ < 400) setTimeout(loadAlt, 200); return null; }
                if (!r.ok) throw new Error('alt ' + r.status);
                return r.blob();
              }).then(blob => {
                if (!blob) return;
                const url = URL.createObjectURL(blob);
                const img = new Image(); img.decoding = 'async';
                img.onload = () => {
                  gl.bindTexture(gl.TEXTURE_2D, tex);
                  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
                  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, img);
                  URL.revokeObjectURL(url);
                  if (altState[idx] === myk) redraw();
                };
                img.src = url;
              }).catch(() => { if (atries++ < 400) setTimeout(loadAlt, 300); });
              (altLoaders[i] || (altLoaders[i] = [])).push(loadAlt);  // DEFER (no eager fetch)
            }
            altTexLists[i] = texList;
          }
          function cycleAlt(i) {
            if (i < 0 || i == null || !altTexLists[i]) return 0;
            // Lazy: kick off the alt fetch(es) on the FIRST activation. This is
            // what triggers the server-side ncolor compute, so it stays deferred
            // until the user actually clicks "Masks" (default view = outlines).
            if (altLoaders[i]) { altLoaders[i].forEach(fn => fn()); altLoaders[i] = null; }
            const n = altTexLists[i].length;
            altState[i] = (altState[i] + 1) % (n + 1);
            redraw();
            return altState[i];
          }
          // back-compat alias (old callers expect a boolean toggle)
          const toggleAlt = cycleAlt;

          // ─── GPU seg outlines (instanced expanded lines) ───────────────
          // We parse the SHARED <defs> outline polygons (the same ones the
          // SVG <use> references for export / no-WebGL fallback) into (p0,p1)
          // line segments — no duplicate geometry payload — and expand each
          // into a constant-screen-px quad on the GPU, so the FULL smoothed
          // boundary stays crisp and free during zoom/rotate (no SVG
          // re-raster). The SVG <use> on the tagged cells is hidden once this
          // is live. Per-cell ncolor (filled) is not tagged → SVG handles it.
          let lineProg = null, lineVAO = null, nSeg = 0, LU = null;
          let outlineColor = [0.75, 0.75, 0.75, 1.0], outlineScreenPx = 1.0;
          // >0 → width measured in IMAGE pixels (scales with zoom); else screen-px.
          let outlineImagePx = parseFloat(svg.dataset.linkOutlineImagePx) || 0;
          const _defsId = svg.dataset.linkOutlineDefs;
          const _outStream = svg.dataset.linkOutlineStream === '1';
          const outlineCell = Array.from(cells).map(c => c && c.dataset && c.dataset.outline === '1');
          if ((_defsId || _outStream) && outlineCell.some(Boolean)) {
            try {
              let segData;
              if (_defsId) {
                const grp = document.getElementById(_defsId);
                const dpolys = grp ? grp.querySelectorAll('polygon') : [];
                const segs = [];
                for (const poly of dpolys) {
                  const ps = poly.getAttribute('points'); if (!ps) continue;
                  const nums = ps.trim().split(/[\s,]+/).map(Number);
                  const m = (nums.length / 2) | 0;
                  for (let i = 0; i < m; i++) {
                    const j = (i + 1) % m;     // closed loop
                    segs.push(nums[2*i], nums[2*i+1], nums[2*j], nums[2*j+1]);
                  }
                }
                segData = new Float32Array(segs);
                if (segData.length === 0) throw new Error('no outline segments parsed');
              } else {
                segData = new Float32Array(0);   // stream: filled after /outline fetch
                nSeg = 0;
              }
              const rgbaStr = (svg.dataset.linkOutlineRgba || '').split(',').map(Number);
              if (rgbaStr.length === 4 && rgbaStr.every(v => !isNaN(v))) outlineColor = rgbaStr;
              const spx = parseFloat(svg.dataset.linkOutlineScreenPx);
              if (!isNaN(spx) && spx > 0) outlineScreenPx = spx;
              // Miter-join ribbon: each instance is ONE edge that also knows its
              // neighbour vertices (prev,p0,p1,next). The shader offsets each end
              // along the JOINT miter (bisector) rather than the edge normal, so
              // adjacent edges share the exact mitered vertex → they abut with no
              // gap and no overlap. Result: a continuous, smooth outline whose
              // coverage is uniform (correct under alpha — no joint double-blend),
              // for the same per-edge cost as the old butt-cap segments but
              // without the end-extension overdraw. Miter is clamped (limit ~4) so
              // sharp turns truncate instead of spiking.
              const LVS = '#version 300 es\n'
                + 'in vec2 a_corner;in vec2 a_prev;in vec2 a_p0;in vec2 a_p1;in vec2 a_next;'
                + 'uniform vec4 u_vp;uniform float u_rot;uniform vec2 u_cpx;uniform float u_hw;'
                + 'out float v_perp;'
                + 'vec2 toClip(vec2 s){vec2 f0=(s-u_vp.xy)/u_vp.zw;float ar=u_vp.z/u_vp.w;'
                + 'vec2 d=(f0-0.5)*vec2(ar,1.0);float cs=cos(u_rot),sn=sin(u_rot);'
                + 'vec2 dr=vec2(cs*d.x-sn*d.y,sn*d.x+cs*d.y);vec2 f=dr/vec2(ar,1.0)+0.5;'
                + 'return vec2(f.x*2.0-1.0,1.0-f.y*2.0);}'
                + 'vec2 toPx(vec2 s){return toClip(s)*u_cpx*0.5;}'   // clip→device px (isotropic)
                + 'void main(){bool atP0=(a_corner.x<0.5);'
                + 'vec2 cur=toPx(atP0?a_p0:a_p1);'
                + 'vec2 aa=toPx(atP0?a_prev:a_p0);vec2 bb=toPx(atP0?a_p1:a_next);'
                + 'vec2 dIn=cur-aa;vec2 dOut=bb-cur;'
                + 'float lIn=length(dIn),lOut=length(dOut);'
                + 'vec2 tIn=lIn>1e-5?dIn/lIn:vec2(0.0);vec2 tOut=lOut>1e-5?dOut/lOut:vec2(0.0);'
                + 'if(lIn<=1e-5)tIn=tOut;if(lOut<=1e-5)tOut=tIn;'
                + 'vec2 nIn=vec2(-tIn.y,tIn.x);vec2 nOut=vec2(-tOut.y,tOut.x);'
                + 'vec2 mit=nIn+nOut;float ml=length(mit);'
                + 'float hwAA=u_hw+0.5;vec2 mdir;float scl;'
                + 'if(ml<1e-3){mdir=nOut;scl=1.0;}else{mdir=mit/ml;scl=1.0/max(dot(mdir,nOut),0.25);}'
                + 'vec2 outpx=cur+a_corner.y*hwAA*scl*mdir;'
                + 'v_perp=a_corner.y*hwAA;'                     // miter preserves perp width → AA ok
                + 'gl_Position=vec4(outpx/(u_cpx*0.5),0.0,1.0);}';
              // No MSAA on this context → AA in the shader: ramp the alpha over
              // the outer 1px of the (perp) line width.
              const LFS = '#version 300 es\nprecision highp float;in float v_perp;'
                + 'out vec4 o;uniform vec4 u_color;uniform float u_hw;'
                + 'void main(){float a=clamp(u_hw+0.5-abs(v_perp),0.0,1.0);'
                + 'o=vec4(u_color.rgb,u_color.a*a);}';
              const lvs = sh(gl.VERTEX_SHADER, LVS), lfs = sh(gl.FRAGMENT_SHADER, LFS);
              if (lvs && lfs) {
                lineProg = gl.createProgram();
                gl.attachShader(lineProg, lvs); gl.attachShader(lineProg, lfs);
                gl.linkProgram(lineProg);
                if (!gl.getProgramParameter(lineProg, gl.LINK_STATUS)) { lineProg = null; }
              }
              if (lineProg) {
                lineVAO = gl.createVertexArray();
                gl.bindVertexArray(lineVAO);
                const cbuf = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, cbuf);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,-1, 1,-1, 0,1, 1,1]), gl.STATIC_DRAW);
                const aC = gl.getAttribLocation(lineProg, 'a_corner');
                gl.enableVertexAttribArray(aC); gl.vertexAttribPointer(aC, 2, gl.FLOAT, false, 0, 0);
                // Per-edge instance = (prev,p0,p1,next), 8 floats / 32-byte stride.
                const ibuf = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, ibuf);
                ['a_prev', 'a_p0', 'a_p1', 'a_next'].forEach((nm, j) => {
                  const loc = gl.getAttribLocation(lineProg, nm);
                  if (loc < 0) return;
                  gl.enableVertexAttribArray(loc);
                  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 32, j * 8);
                  gl.vertexAttribDivisor(loc, 1);
                });
                gl.bindVertexArray(null);
                LU = { vp: gl.getUniformLocation(lineProg, 'u_vp'),
                       rot: gl.getUniformLocation(lineProg, 'u_rot'),
                       cpx: gl.getUniformLocation(lineProg, 'u_cpx'),
                       hw: gl.getUniformLocation(lineProg, 'u_hw'),
                       color: gl.getUniformLocation(lineProg, 'u_color') };
                // Reconstruct closed loops from the (p0,p1) segment stream — a loop
                // closes when an edge's p1 returns to the loop's first p0 — then
                // emit (prev,p0,p1,next) per edge for the miter shader. The wire /
                // <defs> format is unchanged (no Python / live-grid changes).
                function segsToLoops(s) {
                  const loops = []; let cur = null, sx = 0, sy = 0;
                  const ne = (s.length / 4) | 0;
                  for (let k = 0; k < ne; k++) {
                    const x0 = s[4*k], y0 = s[4*k+1], x1 = s[4*k+2], y1 = s[4*k+3];
                    if (!cur) { cur = [[x0, y0]]; sx = x0; sy = y0; }
                    cur.push([x1, y1]);
                    if (x1 === sx && y1 === sy) { cur.pop(); if (cur.length >= 2) loops.push(cur); cur = null; }
                  }
                  if (cur && cur.length >= 2) loops.push(cur);
                  return loops;
                }
                function loopsToMiter(loops) {
                  const o = [];
                  for (const L of loops) {
                    const m = L.length; if (m < 2) continue;
                    for (let i = 0; i < m; i++) {
                      const pv = L[(i-1+m)%m], a = L[i], b = L[(i+1)%m], nx = L[(i+2)%m];
                      o.push(pv[0], pv[1], a[0], a[1], b[0], b[1], nx[0], nx[1]);
                    }
                  }
                  return new Float32Array(o);
                }
                function uploadOutline(segArr) {
                  const inst = loopsToMiter(segsToLoops(segArr));
                  gl.bindVertexArray(lineVAO);
                  gl.bindBuffer(gl.ARRAY_BUFFER, ibuf);
                  gl.bufferData(gl.ARRAY_BUFFER, inst, gl.STATIC_DRAW);
                  gl.bindVertexArray(null);
                  nSeg = (inst.length / 8) | 0;   // instances = edges
                  if (window.__ocdLog) window.__ocdLog('outline (' + nSeg + ' edges)');
                }
                if (_defsId) uploadOutline(segData);
                // GPU outlines live → hide the SVG <use> fallback on these cells.
                for (let i = 0; i < cells.length; i++) {
                  if (!outlineCell[i]) continue;
                  cells[i].querySelectorAll('use').forEach(u => { u.style.display = 'none'; });
                }
                if (_outStream) {
                  // Deferred outline: poll /outline (packed FOV-norm p0,p1 from
                  // the bg thread), scale → RAS px, upload to the instance buffer.
                  let s = '';
                  for (const im of imgs) { if (im) { s = im.getAttribute('data-tile-src') || ''; if (s) break; } }
                  const mm = s.match(/\/tile\/([0-9a-f]+)\//);
                  const bm = s.match(/^(https?:\/\/[^/]+)/);
                  if (mm && bm) {
                    const _ob = bm[1] + '/outline/' + mm[1];
                    const ourl = (window.__ocdResolveTileUrl ? window.__ocdResolveTileUrl(_ob) : _ob);
                    let otries = 0;
                    const fetchO = () => fetch(ourl).then(r => {
                      if (r.status === 204) { if (otries++ < 600) setTimeout(fetchO, 150); return null; }
                      return r.arrayBuffer();
                    }).then(buf => {
                      if (!buf) return;
                      // Stream outline is already the miter-instance format
                      // (prev,p0,p1,next, 8 floats/edge) in RAS px — built
                      // vectorized in Python. Upload the buffer DIRECTLY; no
                      // loop-reconstruction or scaling on the main thread (that
                      // 200k+-edge JS loop stalled tile uploads).
                      const inst = new Float32Array(buf);
                      gl.bindVertexArray(lineVAO);
                      gl.bindBuffer(gl.ARRAY_BUFFER, ibuf);
                      gl.bufferData(gl.ARRAY_BUFFER, inst, gl.STATIC_DRAW);
                      gl.bindVertexArray(null);
                      nSeg = (inst.length / 8) | 0;
                      if (window.__ocdLog) window.__ocdLog('outline (' + nSeg + ' edges)');
                      redraw();
                    }).catch(() => { if (otries++ < 600) setTimeout(fetchO, 200); });
                    fetchO();
                  }
                }
              }
            } catch (e) { console.warn('linkedGL outlines init failed:', e); lineProg = null; }
          }

          // Cell positions are stable during zoom/rotate/pan (only the
          // content transforms) — cache them so redraw() doesn't call
          // getBoundingClientRect per tile per frame. That per-frame query,
          // right after the SVG viewBox writes, forced a synchronous layout
          // of the 1000s of outline polygons → the zoom-out lag. Recompute
          // only on resize.
          let _svgR = null, _crects = null;
          function _recompute() {
            _svgR = svg.getBoundingClientRect();
            _crects = [];
            for (let i = 0; i < cells.length; i++) {
              const cr = hits[i].getBoundingClientRect();
              _crects[i] = { l: cr.left - _svgR.left, t: cr.top - _svgR.top, w: cr.width, h: cr.height };
            }
          }
          // Paint SYNCHRONOUSLY on every redraw. plot_key_slices_live's render()
          // is synchronous in the wheel handler, so the cached level scales in
          // real time AS you zoom ("seamless mid-zoom"). An rAF-coalesced paint
          // (the previous impl) lagged the tween by ~1 frame and effectively
          // updated every OTHER frame → the GL canvas looked frozen until the
          // gesture ended ("waits to finish a zoom event before updating"). Tile
          // fills are already batched (one redraw per Promise.all), so a sync
          // redraw does NOT re-introduce the N-uploads→N-redraws problem.
          function redraw() {
            scheduleRefine();                 // zoom-in may need a finer pyramid level
            _redrawNow();
          }
          // Repaint WITHOUT scheduling a refine — used by the refine loop itself
          // (it re-schedules on its own; painting a freshly cached level must not
          // kick an extra refine pass).
          function _paint() { _redrawNow(); }
          function _redrawNow() {
            if (!_svgR) _recompute();
            if (!_svgR || _svgR.width < 1 || _svgR.height < 1) return;
            const dpr = window.devicePixelRatio || 1;
            const cw = Math.max(1, Math.round(_svgR.width * dpr));
            const ch = Math.max(1, Math.round(_svgR.height * dpr));
            if (canvas.width !== cw) canvas.width = cw;
            if (canvas.height !== ch) canvas.height = ch;
            canvas.style.width = _svgR.width + 'px';
            canvas.style.height = _svgR.height + 'px';
            gl.disable(gl.SCISSOR_TEST);
            gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.SCISSOR_TEST);
            // Image pass.
            gl.useProgram(prog);
            gl.bindVertexArray(imgVAO);
            gl.uniform2f(U.img, RAS_W, RAS_H);
            gl.uniform4f(U.vp, state.x, state.y, state.w, state.h);
            gl.uniform1f(U.rot, state.r || 0);
            gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, lutTex);
            gl.activeTexture(gl.TEXTURE0);
            // Point each cell at the finest cached level ≤ its on-screen target —
            // pure pointer swap (no GL upload), so pan/zoom repaints instantly from
            // whatever is cached; _refineLevels fetches finer levels in the bg.
            for (let i = 0; i < cells.length; i++) {
              if (tileCache[i] && tileCache[i].size) _selectBest(i, _targetLevel(i));
            }
            for (let i = 0; i < cells.length; i++) {
              if (!textures[i] || !_crects[i] || !tileMeta[i]) continue;
              const m = tileMeta[i];
              const _st = altState[i];
              const _altTex = (_st > 0 && altTexLists[i]) ? altTexLists[i][_st - 1] : null;
              const useAlt = !!_altTex;
              if (useAlt) {
                gl.uniform1i(U.mode, 1);            // mask RGBA → passthrough
              } else if (m.mode === 'intensity') {
                let lo = m.lo, hi = m.hi;
                if (m.kind === 'readout') {
                  if (NORMMODE === 'global' && RGLO < RGHI) { lo = RGLO; hi = RGHI; }
                  else if (NORMMODE === 'bitdepth') { lo = 0; hi = m.bitmax || BITMAX; }
                }
                gl.uniform1i(U.mode, 0); gl.uniform1f(U.lo, lo); gl.uniform1f(U.hi, hi);
              } else {
                // RGB tile: linear Display-P3 — always OETF-encode (mode 2) so it
                // displays with correct color. Passthrough (mode 1) showed the raw
                // LINEAR values = too dark/wrong (the rgb_live=0 case). When the
                // WebGPU exc layer is active (rgb_live=1) it draws on top of this
                // anyway; this is the correct SDR fallback + the export source.
                gl.uniform1i(U.mode, m.mode === 'rgb' ? 2 : 1);
              }
              const cr = _crects[i];
              const x = Math.round(cr.l * dpr);
              const wpx = Math.round(cr.w * dpr);
              const hpx = Math.round(cr.h * dpr);
              const yTop = Math.round(cr.t * dpr);
              const y = canvas.height - (yTop + hpx);
              gl.viewport(x, y, wpx, hpx);
              gl.scissor(x, y, wpx, hpx);
              gl.bindTexture(gl.TEXTURE_2D, useAlt ? _altTex : textures[i]);
              gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            }
            // Outline pass (GPU expanded lines) on tagged cells.
            if (lineProg && nSeg > 0) {
              gl.useProgram(lineProg);
              gl.bindVertexArray(lineVAO);
              gl.enable(gl.BLEND);
              gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
              gl.uniform4f(LU.vp, state.x, state.y, state.w, state.h);
              gl.uniform1f(LU.rot, state.r || 0);
              gl.uniform4f(LU.color, outlineColor[0], outlineColor[1], outlineColor[2], outlineColor[3]);
              for (let i = 0; i < cells.length; i++) {
                if (!outlineCell[i] || !_crects[i]) continue;
                const cr = _crects[i];
                const x = Math.round(cr.l * dpr);
                const wpx = Math.round(cr.w * dpr);
                const hpx = Math.round(cr.h * dpr);
                const yTop = Math.round(cr.t * dpr);
                const y = canvas.height - (yTop + hpx);
                // Image-relative half-width: ``outlineImagePx`` image px → device
                // px is (cell device width / RAS px visible), so the stroke grows
                // when zoomed in and shrinks when zoomed out (tracks the pixels).
                const hw = outlineImagePx > 0
                  ? Math.max(0.4, outlineImagePx * (wpx / Math.max(state.w, 1e-6)) * 0.5)
                  : Math.max(0.5, outlineScreenPx * dpr * 0.5);
                gl.uniform1f(LU.hw, hw);
                gl.viewport(x, y, wpx, hpx);
                gl.scissor(x, y, wpx, hpx);
                gl.uniform2f(LU.cpx, wpx, hpx);
                gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, nSeg);
              }
              gl.disable(gl.BLEND);
            }
            gl.bindVertexArray(null);
          }
          if (window.ResizeObserver) { try { new ResizeObserver(() => { _recompute(); redraw(); }).observe(svg); } catch (e) {} }
          onWindow('resize', () => { _recompute(); redraw(); });
          onWindow('scroll', () => { _recompute(); }, { passive: true });
          // Live colormap / readout-norm control for the shell's picker + toggle.
          function setCmap(name) { if (LUTS[name]) { CMAP = name; uploadLUT(name); redraw(); } }
          function setNorm(mode) { NORMMODE = mode; redraw(); }
          // EXPORT_OETF: sRGB-encode the linear RGB tile in the GL pass so a PNG
          // composite of the GL canvas reproduces what the WebGPU exc layer shows.
          function setExportOetf(b) { EXPORT_OETF = !!b; _redrawNow(); }   // sync: export reads next
          return { redraw, redrawNow: _redrawNow, toggleAlt, cycleAlt, setCmap, setNorm, setExportOetf,
                   cmaps: Object.keys(LUTS), getCmap: () => CMAP, getNorm: () => NORMMODE };
        }
        // Client render timeline logger (console: "[ocd-timing] <what> <ms>").
        // T0 = now (figure JS running ≈ when the cell output appears), so the
        // numbers are "ms after the figure showed up" — when each tile/outline/
        // spectra actually paints, which the Python backend timing can't see.
        window.__ocdT0 = performance.now();
        window.__ocdLog = function (m) {
          try { console.log('[ocd-timing]', m, (performance.now() - window.__ocdT0).toFixed(0) + 'ms'); }
          catch (e) {}
        };
        let glLayer = null;
        try { glLayer = createLinkedGLLayer(); }
        catch (e) { console.warn('linkedGL init failed:', e); glLayer = null; }

        // Live tile controls (cmap picker + readout-norm toggle), driving the
        // GL layer's uniforms — same options as the standalone grid viewer.
        if (glLayer && glLayer.setCmap && glLayer.cmaps && glLayer.cmaps.length) {
          try {
            const _host = svg.closest('.ocd-svgfig') || svg.parentElement;
            const bar = document.createElement('div');
            bar.className = 'ocd-tile-controls';
            bar.style.cssText = 'display:flex;gap:12px;align-items:center;'
              + 'margin-top:6px;font:12px system-ui,sans-serif;color:#888;'
              + 'user-select:none;flex-wrap:wrap';
            const sel = document.createElement('select');
            sel.style.cssText = 'font:12px system-ui,sans-serif;';
            glLayer.cmaps.forEach((k) => {
              const o = document.createElement('option');
              o.value = k; o.textContent = k;
              if (k === glLayer.getCmap()) o.selected = true;
              sel.appendChild(o);
            });
            sel.onchange = () => glLayer.setCmap(sel.value);
            const cwrap = document.createElement('label');
            cwrap.style.cssText = 'display:flex;gap:4px;align-items:center';
            cwrap.appendChild(document.createTextNode('cmap'));
            cwrap.appendChild(sel);
            const NM = ['self', 'global', 'bitdepth'];
            const NN = { self: 'self', global: 'global', bitdepth: 'bit-depth' };
            const btn = document.createElement('button');
            btn.style.cssText = 'font:12px system-ui,sans-serif;cursor:pointer;'
              + 'background:none;border:1px solid #888;border-radius:3px;'
              + 'color:inherit;padding:1px 7px';
            let _ni = NM.indexOf(glLayer.getNorm()); if (_ni < 0) _ni = 0;
            btn.textContent = 'key slice norm: ' + NN[NM[_ni]];
            btn.onclick = () => {
              _ni = (_ni + 1) % NM.length; glLayer.setNorm(NM[_ni]);
              btn.textContent = 'key slice norm: ' + NN[NM[_ni]];
            };
            bar.appendChild(cwrap); bar.appendChild(btn);
            _host.appendChild(bar);
          } catch (e) { console.warn('tile controls:', e); }
        }

        // Clickable "Masks" label → CYCLE its cell through its alt rasters
        // (currently: outlines ↔ ncolor pixel fill). ncolor is already a
        // pixel-exact raster, so it IS the pixel-grid view. cycleAlt handles any
        // number of alts; with one declared it's a plain toggle. A <title> child
        // documents it on hover (the wiggle/bold CSS already flags it clickable).
        const _maskToggle = svg.querySelector('text.ocd-mask-toggle');
        if (_maskToggle && glLayer && glLayer.cycleAlt) {
          _maskToggle.style.cursor = 'pointer';
          const _altIdx = Array.from(cells).findIndex(c => c.querySelector('image[data-alt-href]'));
          const _MODE_NAMES = ['outlines', 'ncolor pixel fill'];
          try {
            const _ttl = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            _ttl.textContent = 'Masks: click to toggle outlines / ncolor pixel fill';
            _maskToggle.appendChild(_ttl);
          } catch (e) {}
          _maskToggle.addEventListener('click', (e) => {
            e.preventDefault(); e.stopPropagation();
            const _st = glLayer.cycleAlt(_altIdx);
            const _tt = _maskToggle.querySelector('title');
            if (_tt) _tt.textContent = 'Masks: ' + (_MODE_NAMES[_st] || 'outlines');
          });
        }

        // ─── WebGPU HDR sub-layer (adaptive EDR for data-hdr tiles) ─────
        // A tile flagged ``data-hdr="1"`` carries a texture encoded as
        // OETF(hdr_linear) with 1.0 = XDR peak (the gain-mapped RGB). The
        // WebGL layer renders it SDR (1.0 = white). Here a per-HDR-cell
        // WebGPU canvas (rgba16float + display-p3 + toneMapping:extended)
        // re-interprets that same texture — EOTF → ×headroom → OETF — and
        // sits on top of the WebGL render, below the SVG. ``headroom`` is
        // the live display EDR headroom (screen API, polled), so highlights
        // map to the available range, never clip, and follow brightness
        // changes. No WebGPU / no adapter → the WebGL SDR render shows.
        let hdrLayer = null;
        async function createLinkedHDRLayer() {
          if (!navigator.gpu) return null;
          const hdrCells = [];
          for (let i = 0; i < cells.length; i++) {
            const im = cells[i].querySelector('image');
            if (!im) continue;
            // data-exc tiles carry per-excitation layers composited live; data-hdr
            // tiles carry a single baked gain-mapped texture. Both ride this layer.
            const isExc = im.getAttribute('data-exc') === '1';
            if (im.getAttribute('data-hdr') === '1' || isExc)
              hdrCells.push({ im, hit: hits[i], isExc });
          }
          if (!hdrCells.length) return null;
          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter) return null;
          const device = await adapter.requestDevice();
          const host = svg.closest('.ocd-svgfig') || svg.parentElement;
          if (!host) return null;
          const code = `
struct U { vp: vec4f, p: vec4f };
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var sm: sampler;
@group(0) @binding(2) var<uniform> u: U;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2f,3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  // uv.y flipped: texture row 0 is the TOP, clip y=+1 is the top.
  var uv = array<vec2f,3>(vec2f(0,1), vec2f(2,1), vec2f(0,-1));
  var o: VO; o.pos = vec4f(p[i],0,1); o.uv = uv[i]; return o;
}
fn eotf(c: f32) -> f32 { if (c <= 0.04045) { return c/12.92; } return pow((c+0.055)/1.055, 2.4); }
fn oetf(c: f32) -> f32 { let x = max(c,0.0); if (x <= 0.0031308) { return 12.92*x; } return 1.055*pow(x,1.0/2.4)-0.055; }
@fragment fn fs(in: VO) -> @location(0) vec4f {
  // Rotate the sampling around the cell center (aspect-correct), inverse
  // of the displayed rotation. u.p.y = rotation (rad), u.p.z = cell aspect.
  let ar = u.p.z;
  let p = (in.uv - vec2f(0.5)) * vec2f(ar, 1.0);
  let cs = cos(u.p.y); let sn = sin(u.p.y);
  let pr = vec2f(cs*p.x + sn*p.y, -sn*p.x + cs*p.y);     // R(-rot)
  let f = pr / vec2f(ar, 1.0) + vec2f(0.5);
  let uv = u.vp.xy + f * u.vp.zw;
  let s = textureSample(t, sm, uv).rgb;     // sample first (uniform control flow)
  let lin = vec3f(eotf(s.r), eotf(s.g), eotf(s.b)) * u.p.x;   // hdr_linear * headroom
  let c = vec3f(oetf(lin.r), oetf(lin.g), oetf(lin.b));
  // Outside the image (zoomed/panned/rotated past it) → black, not clamped edge.
  let inb = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
  return vec4f(select(vec3f(0.0), c, inb), 1.0);
}`;
          const module = device.createShaderModule({ code });
          const sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
          // Per-excitation compose shader (data-exc tiles): same vp/rotation
          // transform as the single-texture shader, but sums per-excitation
          // linear-P3 layers over a mask, max-rescales (clipHigh in LINEAR),
          // and lifts to the EDR headroom — eotf/oetf math identical in spirit.
          const composeCode = `
struct U { vp: vec4f, misc: vec4f, scales: array<vec4f,4>, ints: vec4u };
@group(0) @binding(0) var t: texture_2d_array<f32>;
@group(0) @binding(1) var sm: sampler;
@group(0) @binding(2) var<uniform> u: U;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2f,3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  var uv = array<vec2f,3>(vec2f(0,1), vec2f(2,1), vec2f(0,-1));
  var o: VO; o.pos = vec4f(p[i],0,1); o.uv = uv[i]; return o;
}
fn oetf(c: f32) -> f32 { let x = max(c,0.0); if (x <= 0.0031308) { return 12.92*x; } return 1.055*pow(x,1.0/2.4)-0.055; }
@fragment fn fs(inp: VO) -> @location(0) vec4f {
  let ar = u.misc.z;
  let p = (inp.uv - vec2f(0.5)) * vec2f(ar, 1.0);
  let cs = cos(u.misc.y); let sn = sin(u.misc.y);
  let pr = vec2f(cs*p.x + sn*p.y, -sn*p.x + cs*p.y);
  let f = pr / vec2f(ar, 1.0) + vec2f(0.5);
  let uv = u.vp.xy + f * u.vp.zw;
  let n = u.ints.x; let mask = u.ints.y;
  var lin = vec3f(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    if (k >= n) { break; }
    if ((mask & (1u << k)) == 0u) { continue; }
    lin = lin + textureSampleLevel(t, sm, uv, k, 0.0).rgb * u.scales[k/4u][k%4u];
  }
  lin = lin / f32(u.ints.z);            // /total -> mean linear
  lin = lin / max(u.misc.w, 1e-6);      // max-rescale (linear): brightest -> 1
  lin = lin * u.misc.x;                 // * headroom (EDR)
  let c = vec3f(oetf(lin.r), oetf(lin.g), oetf(lin.b));
  let inb = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
  return vec4f(select(vec3f(0.0), c, inb), 1.0);
}`;
          const composeModule = device.createShaderModule({ code: composeCode });
          let N = 4.0;   // EDR headroom — declared before the cell loop so a synchronous exc-load redraw() can read it (the await-fetch hdr path defers past this)
          for (const hc of hdrCells) {
            const canvas = document.createElement('canvas');
            canvas.style.position = 'absolute';
            canvas.style.pointerEvents = 'none';
            canvas.style.zIndex = '0';
            // Hidden until it's positioned over its cell AND has rendered —
            // otherwise it flashes at the host's top-left (0,0 / default
            // 300x150) before _placeCanvases() runs (post async texture load).
            canvas.style.visibility = 'hidden';
            host.insertBefore(canvas, svg);   // above the WebGL canvas, below the SVG
            const ctx = canvas.getContext('webgpu');
            if (!ctx) continue;
            try { ctx.configure({ device, format:'rgba16float', colorSpace:'display-p3',
                                  alphaMode:'opaque', toneMapping:{ mode:'extended' } }); }
            catch (e) { ctx.configure({ device, format:'rgba16float', colorSpace:'display-p3', alphaMode:'opaque' }); }
            const ubuf = device.createBuffer({ size: hc.isExc ? 112 : 32, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
            const _mod = hc.isExc ? composeModule : module;
            const pipe = device.createRenderPipeline({ layout:'auto',
              vertex:{ module: _mod, entryPoint:'vs' },
              fragment:{ module: _mod, entryPoint:'fs', targets:[{ format:'rgba16float' }] },
              primitive:{ topology:'triangle-list' } });
            hc.ctx = ctx; hc.ubuf = ubuf; hc.pipe = pipe; hc.canvas = canvas; hc.ready = false;
            if (hc.isExc) {
              // Per-excitation tile → texture_2d_array + CPU white point (max
              // LINEAR luminance, subsampled). Loads at the FINEST level (full res)
              // like plot_key_slices_live's loadExc — the demo RGB is "full res
              // from the start" and renders the correct color. CRITICAL: the async
              // ``scales`` (per-excitation weight) are read from /info AFTER the
              // layers are fetched (i.e. after the bg thread has filled exc{k} and
              // set its .scale meta). Reading /info BEFORE the fill returned meta
              // with no scale → scales defaulted to 1 for every excitation → wrong
              // weighting → hue shift (blue rendered PINK). vp/scales/mask are
              // written per-frame in redraw().
              (async () => {
                try {
                  const im = hc.im;
                  const total = parseInt(im.getAttribute('data-exc-total'));
                  const _base = im.getAttribute('data-exc-base'), _sid = im.getAttribute('data-exc-sid');
                  const Nl = parseInt(im.getAttribute('data-exc-n'));
                  const _async = im.getAttribute('data-exc-async') === '1';
                  hc.excN = Nl; hc.excTotal = total;
                  hc.excMask = (1 << Nl) - 1;   // all acquisitions on
                  // Fetch one excitation layer at the FINEST level (99 → server
                  // clamps). Async layers project on a bg thread (204 until ready) →
                  // retry. Dims come from the X-Level headers (= full res here).
                  const _fetchLayer = async (k) => {
                    for (let t = 0; t < 600; t++) {
                      const r = await fetch(`${_base}/tile/${_sid}/exc${k}/99?fmt=raw`, {cache:'no-store'});
                      if (r.status === 200) return { buf: new Uint8Array(await r.arrayBuffer()),
                        W: +r.headers.get('X-Level-Width'), H: +r.headers.get('X-Level-Height') };
                      if (!_async) throw new Error('exc layer ' + k + ' status ' + r.status);
                      await new Promise(s => setTimeout(s, 250));
                    }
                    throw new Error('exc layer ' + k + ' timeout');
                  };
                  const got = await Promise.all(Array.from({length: Nl}, (_, k) => _fetchLayer(k)));
                  const layers = got.map(g => g.buf), W = got[0].W, H = got[0].H;
                  // scales: emit-time attr when sync; else /info AFTER the fetch
                  // above guaranteed the layers (and their .scale meta) exist.
                  let scales;
                  if (_async) {
                    let info = {};
                    try { info = await (await fetch(`${_base}/info/${_sid}`, {cache:'no-store'})).json(); } catch (e) {}
                    scales = Array.from({length: Nl}, (_, k) => ((info.meta || {})['exc' + k] || {}).scale || 1);
                  } else {
                    scales = im.getAttribute('data-exc-scales').split(',').map(Number);
                  }
                  const tex = device.createTexture({ size:[W,H,Nl], format:'rgba8unorm',
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
                  for (let k=0;k<Nl;k++) device.queue.writeTexture({ texture: tex, origin:[0,0,k] },
                    layers[k], { bytesPerRow: W*4, rowsPerImage: H }, [W,H,1]);
                  hc.excScales = scales;
                  // White point re-fit to the VISIBLE excitations (max LINEAR
                  // luminance, subsampled) so toggling re-exposes instead of dimming.
                  hc.recomputeClip = () => {
                    let ch = 1e-6; const np = W*H, stp = Math.max(1, Math.floor(np/40000));
                    for (let px=0; px<np; px+=stp) {
                      let r=0,g=0,b=0; const base=px*4;
                      for (let k=0;k<Nl;k++){ if((hc.excMask&(1<<k))===0) continue;
                        const L=layers[k], sc=scales[k]/255; r+=L[base]*sc; g+=L[base+1]*sc; b+=L[base+2]*sc; }
                      const mm = Math.max(r,g,b)/total; if (mm>ch) ch=mm;
                    }
                    hc.excClipHigh = ch;
                  };
                  hc.recomputeClip();
                  hc.bg = device.createBindGroup({ layout: hc.pipe.getBindGroupLayout(0), entries:[
                    { binding:0, resource: tex.createView() }, { binding:1, resource: sampler },
                    { binding:2, resource:{ buffer: hc.ubuf } } ]});
                  // Per-excitation toggle chips over the tile. Bar is click-through
                  // (pointer-events:none); only the chips capture clicks, so the
                  // rest of the tile still pans/zooms. Clicking flips a mask bit.
                  const names = im.getAttribute('data-exc-names').split(',');
                  const bar = document.createElement('div');
                  bar.style.cssText = 'position:absolute;z-index:6;display:flex;gap:1px;flex-wrap:wrap;align-content:flex-start;pointer-events:none;';
                  names.forEach((nm, k) => {
                    const chip = document.createElement('button');
                    chip.textContent = nm; chip.title = 'toggle ' + nm + ' nm';
                    chip.style.cssText = 'pointer-events:auto;font:9px/1.1 system-ui;padding:1px 3px;margin:0;border:0;border-radius:2px;cursor:pointer;background:rgba(0,0,0,0.5);color:#fff;';
                    chip.onpointerdown = (e) => e.stopPropagation();
                    chip.onclick = (e) => { e.stopPropagation(); hc.excMask ^= (1 << k);
                      chip.style.opacity = (hc.excMask & (1 << k)) ? '1' : '0.3';
                      if (hc.recomputeClip) hc.recomputeClip(); redraw(); };
                    bar.appendChild(chip);
                  });
                  host.appendChild(bar); hc.toggleBar = bar;
                  hc.ready = true; hc.im.style.display = 'none'; redraw();
                } catch (e) { console.warn('linkedExc load failed:', e); }
              })();
              continue;
            }
            const href = hc.im.getAttribute('href')
              || hc.im.getAttributeNS('http://www.w3.org/1999/xlink','href');
            // Decode with colorSpaceConversion:'none' so we get the RAW
            // encoded pixels on every browser. Uploading an <img> directly
            // let Safari apply the source's ICC/P3→sRGB conversion (Chrome
            // didn't), so the same texture rendered different colors. The
            // shader owns the color math (sRGB EOTF, P3 gamut), so we must
            // hand it the untouched bytes.
            (async () => {
              try {
                const blob = await (await fetch(href)).blob();
                const bmp = await createImageBitmap(blob, {
                  colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
                const w = bmp.width, h = bmp.height;
                const tex = device.createTexture({ size:[w,h], format:'rgba8unorm',
                  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
                device.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, [w, h]);
                if (bmp.close) bmp.close();
                hc.bg = device.createBindGroup({ layout: hc.pipe.getBindGroupLayout(0), entries:[
                  { binding:0, resource: tex.createView() },
                  { binding:1, resource: sampler },
                  { binding:2, resource:{ buffer: hc.ubuf } } ]});
                hc.ready = true;
                hc.im.style.display = 'none';   // WebGPU renders this tile now
                redraw();
              } catch (e) { console.warn('linkedHDR texture load failed:', e); }
            })();
          }
          function detectHeadroom(){ const sc = window.screen || {};
            for (const k of ['highDynamicRangeHeadroom','dynamicRangeHeadroom','hdrHeadroom','currentEDRHeadroom']) {
              const v = sc[k]; if (typeof v === 'number' && v > 0) return v; }
            return null; }
          // Cache each HDR cell's screen rect (stable during transform);
          // position/size the canvas once on resize, not per frame.
          function _placeCanvases() {
            const hostR = host.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            for (const hc of hdrCells) {
              const cr = hc.hit.getBoundingClientRect();
              if (cr.width < 1 || cr.height < 1) { hc.rectOk = false; continue; }
              hc.canvas.style.left = (cr.left - hostR.left) + 'px';
              hc.canvas.style.top  = (cr.top  - hostR.top)  + 'px';
              hc.canvas.style.width  = cr.width + 'px';
              hc.canvas.style.height = cr.height + 'px';
              const bw = Math.max(1, Math.round(cr.width*dpr)), bh = Math.max(1, Math.round(cr.height*dpr));
              if (hc.canvas.width !== bw) hc.canvas.width = bw;
              if (hc.canvas.height !== bh) hc.canvas.height = bh;
              hc.cellAR = cr.width / Math.max(cr.height, 1e-6);
              if (hc.toggleBar) {                       // pin the toggle chips over the tile's top-left
                hc.toggleBar.style.left = (cr.left - hostR.left) + 'px';
                hc.toggleBar.style.top  = (cr.top  - hostR.top)  + 'px';
                hc.toggleBar.style.maxWidth = cr.width + 'px';
              }
              hc.rectOk = true;
            }
          }
          function redraw(replace) {
            // ``_placeCanvases`` reads getBoundingClientRect for every HDR cell,
            // which FORCES a synchronous SVG reflow. Calling it on every frame of
            // an interactive zoom (applyViewBox → hdrLayer.redraw) made the zoom
            // lag — the HDR cell positions DON'T change during a zoom (only the
            // content scales, via the vp uniform), so skip placement on the
            // per-frame path (``replace===false``). Init/resize/scroll still
            // re-place (default).
            if (replace !== false) _placeCanvases();
            for (const hc of hdrCells) {
              if (!hc.ready || !hc.rectOk) continue;
              if (hc.isExc) {
                const buf = new ArrayBuffer(112);
                new Float32Array(buf, 0, 8).set([ state.x/RAS_W, state.y/RAS_H, state.w/RAS_W, state.h/RAS_H,
                  N, state.r || 0, hc.cellAR, hc.excClipHigh ]);   // vp + (headroom, rot, cellAR, clipHigh)
                const sc = new Float32Array(16); sc.set(hc.excScales.slice(0, 16));
                new Float32Array(buf, 32, 16).set(sc);
                new Uint32Array(buf, 96, 4).set([ hc.excN, hc.excMask, hc.excTotal, 0 ]); // n, mask(live toggle), total
                device.queue.writeBuffer(hc.ubuf, 0, buf);
              } else {
                device.queue.writeBuffer(hc.ubuf, 0, new Float32Array(
                  [ state.x/RAS_W, state.y/RAS_H, state.w/RAS_W, state.h/RAS_H,
                    N, state.r || 0, hc.cellAR, 0 ]));
              }
              const enc = device.createCommandEncoder();
              const pass = enc.beginRenderPass({ colorAttachments:[{
                view: hc.ctx.getCurrentTexture().createView(),
                loadOp:'clear', clearValue:{ r:0, g:0, b:0, a:1 }, storeOp:'store' }]});
              pass.setPipeline(hc.pipe); pass.setBindGroup(0, hc.bg); pass.draw(3); pass.end();
              device.queue.submit([enc.finish()]);
              hc.canvas.style.visibility = 'visible';   // positioned + rendered
              if (!hc._logged && window.__ocdLog) {
                hc._logged = true;
                window.__ocdLog('webgpu ' + (hc.isExc ? 'exc/RGB' : 'hdr') + ' first render');
              }
            }
          }
          // HDR→SDR toggle: force EDR headroom to 1.0 (SDR white) on; RESTORE the
          // last HDR headroom (``_hdrN``) off. Tracking ``_hdrN`` separately is
          // essential — re-detecting on toggle-on returns 1.0 on displays/browsers
          // with no headroom API, which would strand the figure in SDR.
          let _sdr = false, _hdrN = N;          // N defaults to 4.0 above
          function setSdr(s) { _sdr = !!s; N = _sdr ? 1.0 : _hdrN; redraw(); }
          const d0 = detectHeadroom(); if (d0) { _hdrN = d0; N = d0; }
          setInterval(() => { const d = detectHeadroom(); if (d) { _hdrN = d;
            if (!_sdr && Math.abs(d - N) > 1e-3) { N = d; redraw(); } } }, 400);
          if (window.ResizeObserver) { try {
            const ro = new ResizeObserver(() => { redraw(); });
            ro.observe(svg);
            const _host = svg.closest('.ocd-svgfig'); if (_host) ro.observe(_host);
          } catch (e) {} }
          onWindow('resize', () => { redraw(); });
          onWindow('scroll', () => { _placeCanvases(); }, { passive: true });
          return { redraw, setSdr };
        }
        createLinkedHDRLayer().then(l => {
          hdrLayer = l;
          // Expose SDR toggle on the wrapper so the HDR button (different scope)
          // can switch the exc/RGB layer to SDR (EDR headroom → 1.0).
          if (l && l.setSdr) { const _w = svg.closest('.ocd-svgfig'); if (_w) _w.__hdrSetSdr = l.setSdr; }
          if (l) { l.redraw();
            // A few deferred redraws catch the responsive SVG settling its
            // size after first paint (re-places the HDR canvas over its cell).
            requestAnimationFrame(() => l.redraw());
            setTimeout(() => l.redraw(), 150); setTimeout(() => l.redraw(), 500);
          }
        })
                              .catch(e => console.warn('linkedHDR init failed:', e));

        function applyViewBox() {
          const vb = state.x.toFixed(3) + ' ' + state.y.toFixed(3) + ' '
                   + state.w.toFixed(3) + ' ' + state.h.toFixed(3);
          // SVG outline rotation matches the GPU image rotation: rotate the
          // cell content group around the viewport centre (source coords).
          const _rdeg = (state.r || 0) * 180 / Math.PI;
          const _rcx = (state.x + state.w * 0.5).toFixed(3);
          const _rcy = (state.y + state.h * 0.5).toFixed(3);
          const _rtf = _rdeg ? ('rotate(' + _rdeg.toFixed(3) + ' ' + _rcx + ' ' + _rcy + ')') : '';
          // When the GL layer is active the cell content (image + outline) is all
          // GPU-rendered (the SVG <image> is display:none, the outline is the GL
          // line pass), so the nested-SVG viewBox/rotation are VESTIGIAL — writing
          // them on every wheel event just dirties SVG layout → a full re-raster of
          // the overlay per frame, which is the manual-zoom lag the bare-canvas
          // plot_key_slices_live doesn't have. Skip them in GL mode; the SVG-image
          // FALLBACK (glLayer === null) still needs them to zoom its <image>.
          if (!glLayer) {
            for (let i = 0; i < cells.length; i++) {
              cells[i].setAttribute('viewBox', vb);
              const g = cells[i].querySelector('g.ocd-cell-rot');
              if (g) { if (_rtf) g.setAttribute('transform', _rtf); else g.removeAttribute('transform'); }
            }
          }
          // Optional outline screen-px floor: a stroke of ``sw`` viewBox
          // units renders at ``sw * dispW / state.w`` screen px, so flooring
          // that at ``_outlineMinPx`` means sw >= _outlineMinPx*state.w/dispW.
          // Below the floor we keep the image-px width (scales with zoom).
          // ``dispW`` (cell display width) only changes on RESIZE, not on zoom —
          // cache it so we don't getBoundingClientRect (forced reflow) per frame.
          if (_outlineMinPx > 0 && cells.length) {
            if (!_dispWCache) _dispWCache = cells[0].getBoundingClientRect().width || 1;
            const sw = Math.max(_outlineBase, _outlineMinPx * state.w / _dispWCache);
            svg.style.setProperty('--ocd-osw', sw.toFixed(3));
          }
          if (glLayer) glLayer.redraw();
          if (hdrLayer) hdrLayer.redraw(false);   // false = don't reflow-place; zoom only scales content
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
          // During a Safari trackpad gesture, the gesturechange handler owns
          // zoom (via e.scale). Safari ALSO emits ctrl+wheel for the same
          // pinch — ignore it here so we don't double-zoom.
          if (_gestureActive && e.ctrlKey) return;
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
          // INSTANT zoom (no tween) — matches plot_key_slices_live, which snaps
          // the view on every wheel event and renders synchronously. The tween
          // (a) lagged the GL paint and (b) kept resetting the refine debounce
          // until it settled (~250 ms), so the resolution "waited to finish the
          // zoom before updating". Instant settle → refine fires 60 ms after you
          // stop, and the sync redraw scales the cached level in real time.
          setViewbox(newX, newY, newW, newH, false);
        }

        // Pointer drag = pan. Updates state AND target simultaneously
        // for instant 1:1 cursor tracking (no tween lag).
        let dragId = null, dragLastX = 0, dragLastY = 0, dragHit = null;
        // Multitouch rotate: track active touch pointers; 2 fingers → rotate
        // by the change in the angle between them (Chrome/Windows touch).
        // Safari/Mac trackpad rotate uses gesture events (added below).
        const _ptrs = new Map();
        let _gr = null;     // { a0: start angle, r0: start rotation }
        function _snapState() {
          state.x = target.x; state.y = target.y;
          state.w = target.w; state.h = target.h;
          if (_raf) { cancelAnimationFrame(_raf); _raf = 0; }
        }
        function onPointerDown(e) {
          if (e.button !== 0 && e.pointerType !== 'touch') return;
          try { e.currentTarget.setPointerCapture(e.pointerId); } catch {}
          _ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
          if (_ptrs.size >= 2) {                       // enter 2-finger pinch+rotate
            dragId = null; if (dragHit) dragHit.style.cursor = 'grab'; dragHit = null;
            const p = [..._ptrs.values()].slice(0, 2);
            _snapState();
            const dx = p[1].x - p[0].x, dy = p[1].y - p[0].y;
            const mx = (p[0].x + p[1].x) / 2, my = (p[0].y + p[1].y) / 2;
            const rc = e.currentTarget.getBoundingClientRect();
            const fx = (mx - rc.left) / rc.width, fy = (my - rc.top) / rc.height;
            const g0 = _unrotFrac(fx, fy, state.r || 0);
            _gr = { a0: Math.atan2(dy, dx), r0: state.r || 0, d0: Math.hypot(dx, dy) || 1,
                    w0: state.w, af: { x: fx, y: fy },
                    as: { x: state.x + g0.x * state.w, y: state.y + g0.y * state.h } };
            e.preventDefault();
            return;
          }
          dragId = e.pointerId; dragHit = e.currentTarget;
          dragLastX = e.clientX; dragLastY = e.clientY;
          dragHit.style.cursor = 'grabbing';
          _snapState();
          e.preventDefault();
        }
        function onPointerMove(e) {
          if (_ptrs.has(e.pointerId)) { const p = _ptrs.get(e.pointerId); p.x = e.clientX; p.y = e.clientY; }
          if (_gr && _ptrs.size >= 2) {                // pinch zoom + rotate, anchored
            const p = [..._ptrs.values()].slice(0, 2);
            const dx = p[1].x - p[0].x, dy = p[1].y - p[0].y;
            const dist = Math.hypot(dx, dy) || 1;
            const rNew = _gr.r0 + (Math.atan2(dy, dx) - _gr.a0);
            const wNew = Math.max(MIN_W, Math.min(MAX_W, _gr.w0 * _gr.d0 / dist));
            const hNew = wNew / cellAR;
            const g1 = _unrotFrac(_gr.af.x, _gr.af.y, rNew);
            state.r = rNew; state.w = wNew; state.h = hNew;
            state.x = _gr.as.x - g1.x * wNew; state.y = _gr.as.y - g1.y * hNew;
            target.x = state.x; target.y = state.y; target.w = wNew; target.h = hNew; target.r = rNew;
            applyViewBox();
            return;
          }
          if (e.pointerId !== dragId) return;
          const r = dragHit.getBoundingClientRect();
          const dxClient = e.clientX - dragLastX;
          const dyClient = e.clientY - dragLastY;
          dragLastX = e.clientX;
          dragLastY = e.clientY;
          // Convert the SCREEN drag into source space, undoing the view
          // rotation (aspect-correct) — otherwise drag follows the rotated
          // axes after a rotation.
          const rot = state.r || 0, c = Math.cos(rot), sn = Math.sin(rot);
          const ax = (dxClient / r.width) * cellAR, ay = (dyClient / r.height);
          const d0x = (c * ax + sn * ay) / cellAR;   // R(-rot), then un-aspect
          const d0y = (-sn * ax + c * ay);
          state.x -= d0x * state.w;
          state.y -= d0y * state.h;
          target.x = state.x; target.y = state.y;
          applyViewBox();
        }
        function onPointerUp(e) {
          _ptrs.delete(e.pointerId);
          if (_gr && _ptrs.size < 2) _gr = null;
          try { e.currentTarget.releasePointerCapture(e.pointerId); } catch {}
          if (e.pointerId === dragId) {
            if (dragHit) dragHit.style.cursor = 'grab';
            dragId = null; dragHit = null;
          }
        }

        // Reset to the initial ROI + zero rotation (double-click / 'H').
        function resetView() {
          state.r = 0; target.r = 0;
          setViewbox(initX, initY, initW, initH, true);
        }
        function onDblClick(e) {
          e.preventDefault();
          resetView();
        }
        function onKey(e) {
          if (e.key === 'h' || e.key === 'H' || e.key === 'Home') {
            if (!wrapper.isConnected) return;
            // Never steal a keystroke that the user is typing into an
            // editor. Beyond native <input>/<textarea>, notebook code
            // cells are contenteditable editor surfaces — CodeMirror
            // (JupyterLab/Notebook) and Monaco (VS Code) — so a bare
            // tagName check misses them and would swallow 'h'.
            const ae = document.activeElement;
            const tag = (ae && ae.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea') return;
            if (ae && (ae.isContentEditable || ae.closest(
                '.cm-editor, .CodeMirror, .monaco-editor, [contenteditable="true"]'))) return;
            e.preventDefault();
            resetView();
          }
        }
        onWindow('keydown', onKey);

        // Handlers attach to the HIT RECTS (which live in the outer SVG
        // coord system at the cell's bbox). The cells' viewBox is
        // mutated by the controller but cells themselves carry no
        // listeners — events from the hit rect drive the shared state,
        // applyViewBox writes the new viewBox onto every cell SVG.
        // Safari (Mac) fires gesture* for trackpad pinch+rotate. e.scale and
        // e.rotation are cumulative since gesturestart. We drive BOTH zoom and
        // rotation from it, anchored at the cursor (the source point under the
        // gesture stays put), and suppress the duplicate ctrl+wheel.
        let _gStartR = 0, _gStartW = 0, _gAf = null, _gAs = null;
        hits.forEach(hit => {
          hit.style.cursor = 'grab';
          hit.style.touchAction = 'none';
          hit.addEventListener('wheel', onWheel, { passive: false });
          hit.addEventListener('pointerdown', onPointerDown);
          hit.addEventListener('pointermove', onPointerMove);
          hit.addEventListener('pointerup', onPointerUp);
          hit.addEventListener('pointercancel', onPointerUp);
          hit.addEventListener('dblclick', onDblClick);
          hit.addEventListener('gesturestart', (e) => {
            e.preventDefault();
            _snapState();
            _gestureActive = true;
            _gStartR = state.r || 0; _gStartW = state.w;
            const rc = hit.getBoundingClientRect();
            const fx = (e.clientX - rc.left) / rc.width, fy = (e.clientY - rc.top) / rc.height;
            _gAf = { x: fx, y: fy };
            const g0 = _unrotFrac(fx, fy, state.r || 0);
            _gAs = { x: state.x + g0.x * state.w, y: state.y + g0.y * state.h };
          });
          hit.addEventListener('gesturechange', (e) => {
            e.preventDefault();
            if (!_gAf) return;
            const rNew = _gStartR + (e.rotation || 0) * Math.PI / 180;
            const wNew = Math.max(MIN_W, Math.min(MAX_W, _gStartW / (e.scale || 1)));
            const hNew = wNew / cellAR;
            const g1 = _unrotFrac(_gAf.x, _gAf.y, rNew);   // keep anchor under cursor
            state.r = rNew; state.w = wNew; state.h = hNew;
            state.x = _gAs.x - g1.x * wNew; state.y = _gAs.y - g1.y * hNew;
            target.x = state.x; target.y = state.y; target.w = wNew; target.h = hNew; target.r = rNew;
            applyViewBox();
          });
          hit.addEventListener('gestureend', (e) => { e.preventDefault(); _gestureActive = false; });
        });
        // Paint initial state on every cell.
        applyViewBox();
      }
    }

    // Composite the LIVE figure (GL tiles + SVG vector + spectra/exc GPU
    // canvases) into ONE canvas → "a PNG of exactly what we see". Rasterizing
    // the SVG alone TAINTS the canvas (it has <foreignObject>) AND misses the
    // GPU layers, so we draw each layer at its on-screen rect, in z-order, and
    // replace each spectra <foreignObject> with an <image> of its density
    // canvas (keeps refs/ticks on top, drops the foreignObject → no taint).
    async function compositeFigure() {
      const host = wrapper;
      const fR = svg.getBoundingClientRect();
      // Supersample 2× over device pixels so the saved PNG is crisp (sharper
      // than the on-screen size, not below it) — vector layers re-rasterize at
      // this resolution; raster layers scale up from their device-res backing.
      const SS = (window.devicePixelRatio || 1) * 2;
      const W = Math.max(1, Math.round(fR.width * SS));
      const H = Math.max(1, Math.round(fR.height * SS));
      const out = document.createElement('canvas'); out.width = W; out.height = H;
      const ctx = out.getContext('2d');
      const bg = getComputedStyle(host).backgroundColor;
      if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') { ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H); }
      function drawCv(cv) {
        if (!cv || !cv.width || !cv.height) return;
        const r = cv.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) return;
        try { ctx.drawImage(cv, (r.left - fR.left) * SS, (r.top - fR.top) * SS, r.width * SS, r.height * SS); }
        catch (e) { console.warn('composite drawImage:', e); }
      }
      // A PNG is SDR — force the exc/RGB layer to SDR (so it's sRGB-encoded, not
      // left in HDR/extended range) for the capture, then restore the live state.
      const _wasSdr = wrapper.classList.contains('ocd-sdr-mode');
      try { if (wrapper.__hdrSetSdr) wrapper.__hdrSetSdr(true); } catch (e) {}
      // The RGB tile is linear Display-P3; on screen the WebGPU exc layer OETFs
      // it, but that HDR canvas can't be drawImage()'d to an SDR PNG without the
      // browser re-tonemapping it darker/hue-shifted. So OETF the RGB in the GL
      // pass and capture THAT (skipping the WebGPU readback below).
      const _glOetf = !!(glLayer && glLayer.setExportOetf);
      try { if (_glOetf) glLayer.setExportOetf(true); } catch (e) {}
      // synchronous draw — we read the canvas immediately below (coalesced
      // redraw() would defer to the next frame and capture a stale buffer)
      try { if (glLayer && glLayer.redrawNow) glLayer.redrawNow(); } catch (e) {}
      try { if (hdrLayer && hdrLayer.redraw) hdrLayer.redraw(); } catch (e) {}
      const glcv = host.querySelector(':scope > canvas');     // GL tiles (z0)
      drawCv(glcv);
      const clone = svg.cloneNode(true);
      clone.setAttribute('width', W); clone.setAttribute('height', H);  // rasterize at full res
      const fos = Array.prototype.slice.call(clone.getElementsByTagName('foreignObject'));
      const realFos = Array.prototype.slice.call(svg.getElementsByTagName('foreignObject'));
      for (let i = 0; i < fos.length; i++) {
        const fo = fos[i];
        const dens = realFos[i] ? realFos[i].querySelector('canvas[data-spectra-density]') : null;
        if (dens && dens.width) {
          const im = document.createElementNS('http://www.w3.org/2000/svg', 'image');
          ['x', 'y', 'width', 'height'].forEach(a => im.setAttribute(a, fo.getAttribute(a)));
          try { im.setAttribute('href', dens.toDataURL('image/png')); } catch (e) {}
          fo.parentNode.replaceChild(im, fo);
        } else if (fo.parentNode) { fo.parentNode.removeChild(fo); }
      }
      Array.prototype.slice.call(clone.querySelectorAll('image[data-tile-src], image[data-tile-async]'))
        .forEach(im => { if (im.parentNode) im.parentNode.removeChild(im); });
      const xml = new XMLSerializer().serializeToString(clone);
      const svgImg = new Image();
      await new Promise((res, rej) => {
        svgImg.onload = res; svgImg.onerror = rej;
        svgImg.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml);
      });
      ctx.drawImage(svgImg, 0, 0, W, H);                       // SVG vector + density image
      // Draw the WebGPU exc/RGB GPU canvases ONLY when the GL layer can't OETF
      // the RGB itself (no WebGL) — otherwise the GL OETF capture above is the
      // faithful SDR version and the HDR-canvas readback would just re-darken it.
      if (!_glOetf) {
        Array.prototype.slice.call(host.querySelectorAll('canvas')).forEach(cv => {
          if (cv === glcv) return;
          if (cv.closest && cv.closest('foreignObject')) return;  // spectra drawn via image
          drawCv(cv);                                             // exc/RGB GPU layer
        });
      }
      // restore the live HDR/SDR state
      try { if (_glOetf) glLayer.setExportOetf(false); } catch (e) {}
      try { if (wrapper.__hdrSetSdr && !_wasSdr) wrapper.__hdrSetSdr(false); } catch (e) {}
      return out;
    }

    // Save: PNG of exactly what we see (composited). Button may be absent.
    const _savebtn = wrapper.querySelector('.ocd-savebtn');
    if (_savebtn) _savebtn.addEventListener('click', async (e) => {
      const btn = e.currentTarget; btn.disabled = true;
      try {
        const out = await compositeFigure();
        const png = await new Promise(res => out.toBlob(res, 'image/png'));
        const url = URL.createObjectURL(png);
        const a = document.createElement('a');
        a.href = url; a.download = 'figure.png';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch (err) { console.error('SvgFigure save failed:', err); alert('Save failed: ' + err.message); }
      finally { btn.disabled = false; }
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
      // The CSS dynamic-range-limit rule handles raster <image>/<img> tiles,
      // but the GPU label canvases emit HDR via the shader — flip their
      // boosts too so the outline / hover follow the toggle.
      wrapper.querySelectorAll('canvas[data-label-tile]').forEach((cv) => {
        if (cv.__labelSetSdr) cv.__labelSetSdr(sdr);
      });
      if (webglViewer && webglViewer.setSdr) webglViewer.setSdr(sdr);
      // Linked-grid exc/RGB WebGPU layer (the streaming key-slice figure).
      if (wrapper.__hdrSetSdr) wrapper.__hdrSetSdr(sdr);
      // Live spectra density: swap the colormap LUT (HDR-lifted ↔ SDR-linear)
      // and re-render on the same extended WebGPU canvas.
      wrapper.querySelectorAll('canvas[data-spectra-density]').forEach((cv) => {
        const c = cv.__sgCfg;
        if (c && c.hdr && (c.lutHdr || c.lutSdr)) {
          c.lut = (!sdr && c.lutHdr) ? c.lutHdr : c.lutSdr;
          if (cv.__spectraDraw) cv.__spectraDraw();
        }
      });
    });

    // Copy: PNG of exactly what we see (composited GL + SVG + GPU layers) →
    // clipboard. The old SVG-rasterize path tainted the canvas (foreignObject)
    // and dropped the GPU tiles/spectra; compositeFigure() avoids both.
    const _copybtn = wrapper.querySelector('.ocd-copybtn');
    if (_copybtn) _copybtn.addEventListener('click', async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      try {
        const out = await compositeFigure();
        const png = await new Promise(res => out.toBlob(res, 'image/png'));
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


def _load_label_gl_js():
    """Read the shared WebGL2 label renderer (``ocdkit/plot/web/label_gl.js``).

    Read FRESH on every call (per figure render), NOT cached at import —
    so edits to label_gl.js take effect on the next ``imshow`` without a
    kernel restart. The file is tiny and OS-cached, so the cost is
    negligible (~0.01 ms warm).
    """
    import os
    here = os.path.dirname(os.path.dirname(__file__))  # ocdkit/
    path = os.path.join(here, "plot", "web", "label_gl.js")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""

# Per-figure controller: finds every ``<canvas data-label-tile>`` in the
# wrapper, decodes its label matrix + palette, and renders it live via the
# shared LabelGLRenderer (palette fill + outlines), with hover-highlight.
# One WebGL2 context per canvas (fine for the handful of label tiles a grid
# carries; a shared context is a future optimization).
_LABEL_CONTROLLER_JS = r"""
(function () {
  if (!self.LabelGL) return;
  var wrapper = document.querySelector('.ocd-svgfig[data-uid="__UID__"]');
  if (!wrapper) return;
  function b64bytes(s) {
    var bin = atob(s), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  var tiles = wrapper.querySelectorAll('canvas[data-label-tile]');
  tiles.forEach(function (cv) {
    if (cv.__labelWired) return;
    cv.__labelWired = true;
    var cfg = self.LabelGL.decodeAttrs(cv);  // shared decode (also used by popup)
    cv.__labelCfg = cfg;                      // popup reuses this config
    var w = cfg.w, h = cfg.h;
    cv.width = w; cv.height = h;              // native label resolution
    var gl = cv.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
    if (!gl) { console.warn('LabelGL: no WebGL2'); return; }
    // HDR: float16 extended-range backbuffer so a >1.0 outline/highlight color
    // (outline_hdr / highlight boost) emits TRUE HDR instead of clamping to
    // SDR white. Needs EXT_color_buffer_float; allocate AFTER cv.width/height
    // (set above) since setting them would reset it. SDR 8-bit fallback.
    if (gl.drawingBufferStorage) {
      try {
        gl.getExtension('EXT_color_buffer_float');
        gl.drawingBufferColorSpace = 'display-p3';
        gl.drawingBufferStorage(gl.RGBA16F, w, h);
      } catch (e) {}
    }
    var r;
    try { r = self.LabelGL.buildRenderer(gl, cfg, function () { render(); }); }
    catch (e) { console.warn('LabelGL:', e); return; }
    function render() {
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      r.draw(self.LabelGL.ortho());
    }
    render();
    cv.__labelRender = render;
    cv.__labelRenderer = r;
    // HDR toggle response: in SDR mode drop the boosts to 1.0 so the outline /
    // hover emit at SDR white (≤1.0) instead of HDR-bright; in HDR mode use
    // the configured boosts. Exposed for the shell's HDR button; applied now
    // for the figure's initial state.
    var _cfgOutlineHdr = (cfg.uniforms && cfg.uniforms.outlineHdrBoost) || 1.0;
    cv.__labelSetSdr = function (sdr) {
      r.setUniforms({ outlineHdrBoost: sdr ? 1.0 : _cfgOutlineHdr,
                      highlightBoost: sdr ? 1.0 : 1.8 });
      render();
    };
    cv.__labelSetSdr(wrapper.classList.contains('ocd-sdr-mode'));
    // hover-highlight + label-id tooltip (mirror the viewer's labelAt)
    var tip = null;
    function ensureTip() {
      if (!tip) {
        tip = document.createElement('div');
        tip.style.cssText = 'position:fixed;pointer-events:none;z-index:2147483647;'
          + 'background:rgba(20,20,20,.92);color:#eee;font:11px sans-serif;'
          + 'padding:2px 6px;border-radius:4px;display:none;';
        document.body.appendChild(tip);
      }
      return tip;
    }
    var cur = 0;
    cv.addEventListener('pointermove', function (e) {
      var rect = cv.getBoundingClientRect();
      var px = Math.floor((e.clientX - rect.left) / rect.width * w);
      var py = Math.floor((e.clientY - rect.top) / rect.height * h);
      var id = r.labelAt(px, py);
      if (id !== cur) {
        cur = id;
        r.setUniforms({ highlightLabel: id });
        render();
      }
      var t = ensureTip();
      if (id > 0) {
        t.textContent = 'label ' + id;
        t.style.display = 'block';
        t.style.left = (e.clientX + 12) + 'px';
        t.style.top = (e.clientY + 12) + 'px';
      } else { t.style.display = 'none'; }
    });
    cv.addEventListener('pointerleave', function () {
      cur = 0; r.setUniforms({ highlightLabel: 0 }); render();
      if (tip) tip.style.display = 'none';
    });
  });
})();
""".strip()


def _load_spectra_gl_js():
    """Read the shared WebGL2 spectra-density renderer
    (``ocdkit/plot/web/spectra_density_gl.js``). Read FRESH per render (like
    ``_load_label_gl_js``) so edits take effect without a kernel restart."""
    import os
    here = os.path.dirname(os.path.dirname(__file__))  # ocdkit/
    path = os.path.join(here, "plot", "web", "spectra_density_gl.js")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


# Per-figure controller: finds every ``<canvas data-spectra-density>`` in the
# wrapper and renders it live via the shared SpectraGL engine, at the canvas's
# true on-screen resolution (getBoundingClientRect × devicePixelRatio). Re-renders
# whenever the wrapper resizes (responsive SVG scaling / zoom), so the density
# is always crisp at the displayed size — no fixed-resolution baked raster.
_SPECTRA_CONTROLLER_JS = r"""
(function () {
  if (!self.SpectraGL) return;
  var wrapper = document.querySelector('.ocd-svgfig[data-uid="__UID__"]');
  if (!wrapper) return;
  var cvs = Array.prototype.slice.call(
    wrapper.querySelectorAll('canvas[data-spectra-density]'));
  if (!cvs.length) return;
  // Streamed spectra (no base64): fetch the meta JSON + binary parts from
  // ``data-spectra-src`` and assemble the SpectraGL cfg (all 3 norm variants
  // cached on the canvas for the y-axis toggle). Returns a Promise<cfg|null>
  // (null = 204 not-ready → caller retries).
  function fetchSpectraCfg(cv, src) {
    return fetch(src + '?part=meta')
      .then(function (r) { return r.status === 204 ? null : r.json(); })
      .then(function (meta) {
        if (!meta) return null;
        var parts = ['ylines_self', 'ylines_bitdepth', 'ylines_global',
                     'xpix', 'lut', 'cellids', 'lut_hdr'];
        return Promise.all(parts.map(function (p) {
          return fetch(src + '?part=' + p).then(function (r) {
            return r.ok ? r.arrayBuffer() : null; });
        })).then(function (b) {
          if (!b[0] || !b[3] || !b[4]) return null;   // a part wasn't ready → retry
          var norm = cv.getAttribute('data-norm-mode') || meta.norm_mode || 'self';
          var variants = { self: new Float32Array(b[0]),
                           bitdepth: new Float32Array(b[1]),
                           global: new Float32Array(b[2]) };
          cv.__sgVariants = variants;     // for the y-axis norm toggle
          // HDR (meta.hdr): both LUTs are linear-P3 float32 — lutSdr (≤1) and
          // lutHdr (lifted, >1). SpectraGL OETFs the active one on an extended
          // canvas; the HDR toggle swaps cfg.lut between them. SDR plot: lut is
          // uint8 sRGB (the 2D readback path), no toggle.
          var _lutSdr = meta.hdr ? new Float32Array(b[4]) : new Uint8Array(b[4]);
          var _lutHdr = (meta.hdr && b[6]) ? new Float32Array(b[6]) : null;
          var _sdrMode = wrapper.classList.contains('ocd-sdr-mode');
          return {
            numLines: meta.num_lines, numPoints: meta.num_points,
            yLines: variants[norm] || variants.self,
            xPix: new Float32Array(b[3]), plotW: meta.plot_w,
            intervals: meta.intervals, lineWidth: meta.line_width,
            yLo: meta.ylo, yHi: meta.yhi,
            lut: (meta.hdr && !_sdrMode && _lutHdr) ? _lutHdr : _lutSdr,
            lutSdr: _lutSdr, lutHdr: _lutHdr,
            hdr: !!meta.hdr,
            cellIds: b[5] ? new Int32Array(b[5]) : null,
            cellLabels: meta.cell_labels || [],
          };
        });
      });
  }
  // Shared hover tooltip (cell id + classification), position:fixed at cursor.
  function _sgTooltip() {
    var t = document.getElementById('sg-tooltip');
    if (!t) {
      t = document.createElement('div'); t.id = 'sg-tooltip';
      t.style.cssText = 'position:fixed;z-index:99999;pointer-events:none;display:none;'
        + 'background:rgba(20,20,22,0.92);color:#ddd;font:11px/1.35 system-ui,sans-serif;'
        + 'padding:4px 7px;border-radius:4px;border:1px solid #444;white-space:nowrap;'
        + 'box-shadow:0 2px 8px rgba(0,0,0,0.4)';
      document.body.appendChild(t);
    }
    return t;
  }
  cvs.forEach(function (cv) {
    if (cv.__spectraWired) return;
    cv.__spectraWired = true;
    cv.__spectraDraw = function () {
      // CRITICAL (JupyterLab): the spectra <canvas> lives in a <foreignObject>
      // that Lab can leave 0-size for a LONG time. The old loop gated the FETCH
      // on the canvas being laid out AND retried with requestAnimationFrame —
      // each tick calling getBoundingClientRect — so a 0-size canvas became a
      // ~60 forced-reflow/sec STORM that saturated the main thread, janking the
      // tile zoom and delaying the refine's setTimeout callbacks (the inter-level
      // gaps). Fix: (1) fetch the data IMMEDIATELY, decoupled from canvas size;
      // (2) wait for size with setTimeout (≈4 Hz), never rAF; (3) cache the cfg.
      var src = cv.getAttribute('data-spectra-src');
      if (src) {                              // streamed line data (no base64)
        if (!cv.__sgCfg && !cv.__sgFetching) {
          cv.__sgFetching = true;
          fetchSpectraCfg(cv, src).then(function (cfg) {
            cv.__sgFetching = false;
            if (!cfg) { setTimeout(cv.__spectraDraw, 250); return; }   // 204 → gentle retry
            cv.__sgCfg = cfg;
            if (window.__ocdLog && !cv.__sgFetchT) cv.__sgFetchT = 1, window.__ocdLog('spectra data ready');
            cv.__spectraDraw();               // data ready → try to render
          }).catch(function (e) {
            cv.__sgFetching = false;
            cv.__sgTries = (cv.__sgTries || 0) + 1;
            if (cv.__sgTries < 600) setTimeout(cv.__spectraDraw, 250);
            else console.warn('SpectraGL fetch (gave up):', e);
          });
        }
      } else {
        // Base64 fallback: re-decode each call so a norm swap (data-ylines-* →
        // data-ylines) is picked up live. Cheap (no fetch).
        try { cv.__sgCfg = self.SpectraGL.decodeAttrs(cv); }
        catch (e) { console.warn('SpectraGL decode:', e); return; }
        if (window.__ocdLog && !cv.__sgFetchT) cv.__sgFetchT = 1, window.__ocdLog('spectra data ready');
      }
      if (!cv.__sgCfg) return;                 // data not ready yet (fetch pending)
      // Render only once the canvas actually has a size — wait with setTimeout,
      // NOT requestAnimationFrame + per-frame getBoundingClientRect.
      var r = cv.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) { setTimeout(cv.__spectraDraw, 250); return; }
      Promise.resolve(self.SpectraGL.render(cv, cv.__sgCfg, r.width, r.height))
        .then(function (ok) {
          if (ok === false) console.warn('SpectraGL: WebGPU unavailable');
          else if (window.__ocdLog && !cv.__sgRenderedT) { cv.__sgRenderedT = 1; window.__ocdLog('spectra rendered'); }
        })
        .catch(function (e) { console.warn('SpectraGL render:', e); });
    };
    cv.__spectraDraw();                         // kick off the fetch immediately

    // Hover highlight on the sibling 2D overlay canvas (the density canvas is a
    // WebGPU context; the highlight stroke is 2D). Pointer events land on the
    // overlay (it's on top); we stroke the nearest spectrum there + show a
    // tooltip with the highlighted cell's id and classification label.
    var overlay = cv.parentNode ? cv.parentNode.querySelector('canvas[data-spectra-overlay]') : null;
    var hlColor = cv.getAttribute('data-highlight-color') || 'rgba(255,64,64,0.95)';
    if (overlay) {
      overlay.addEventListener('pointermove', function (e) {
        if (!cv.__sgState) return;          // density not rendered yet
        var r = overlay.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) return;
        var mx = (e.clientX - r.left) / r.width * cv.__sgState.W;
        var my = (e.clientY - r.top) / r.height * cv.__sgState.H;
        var line = -1;
        try { line = self.SpectraGL.highlight(cv, overlay, mx, my, hlColor); } catch (_) {}
        var tip = _sgTooltip();
        var cfg = cv.__sgCfg;               // current cfg (updated each draw / norm swap)
        if (line >= 0 && cfg && cfg.cellIds) {
          var lab = (cfg.cellLabels && cfg.cellLabels[line]) || '';
          tip.innerHTML = 'Cell ' + cfg.cellIds[line] + (lab ? '<br>' + lab : '');
          tip.style.display = 'block';
          tip.style.left = (e.clientX + 14) + 'px';
          tip.style.top = (e.clientY + 14) + 'px';
          // Temporarily light up THIS cell's classification readouts' reference
          // spectra (+ color their labels) while hovered.
          if (wrapper.__refSetTemp) wrapper.__refSetTemp(lab.match(/R\d+/g) || []);
        } else {
          tip.style.display = 'none';
          if (wrapper.__refClearTemp) wrapper.__refClearTemp();
        }
      });
      overlay.addEventListener('pointerleave', function () {
        try { self.SpectraGL.clearHighlight(overlay); } catch (_) {}
        var tip = document.getElementById('sg-tooltip'); if (tip) tip.style.display = 'none';
        if (wrapper.__refClearTemp) wrapper.__refClearTemp();
      });
    }
  });
  // The canvas lives in a <foreignObject> whose own box is in fixed SVG user
  // units, so observing it won't catch the SVG scaling on cell resize. Observe
  // the WRAPPER (real CSS px) and rerender every spectra canvas at the new
  // device resolution.
  if (self.ResizeObserver && !wrapper.__spectraRO) {
    var raf = 0;
    wrapper.__spectraRO = new ResizeObserver(function () {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(function () {
        cvs.forEach(function (cv) { if (cv.__spectraDraw) cv.__spectraDraw(); });
      });
    });
    wrapper.__spectraRO.observe(wrapper);
  }
})();
""".strip()


# Tile-URL resolver: tile/outline/stream URLs are baked server-side as the
# kernel loopback ``http://127.0.0.1:PORT/...`` (the tile server binds 127.0.0.1).
# That's reachable from a SAME-MACHINE browser but NOT from a remote one (there
# 127.0.0.1 is the client, not the kernel) — so a remote page's tile fetches fail
# and the figure stays on its coarse placeholder. ``window.__ocdResolveTileUrl``
# rewrites such a URL to the Jupyter-origin proxy ``{baseUrl}ocdkit-tiles/PORT/...``
# (served by ``ocdkit.tileserve.jupyter_ext``, riding the notebook's own auth+TLS)
# whenever the page is served off-machine; local pages keep the direct loopback
# UNCHANGED, so same-machine behaviour is byte-identical. Defined once (idempotent)
# and prepended FIRST so every controller + the GL layer can call it.
_TILE_BASE_RESOLVER_JS = r"""
(function () {
  if (window.__ocdResolveTileUrl) return;
  function jbase() {
    try { var el = document.getElementById('jupyter-config-data');
      if (el) { var c = JSON.parse(el.textContent || '{}'); if (c.baseUrl) return c.baseUrl; } } catch (e) {}
    var b = document.body && document.body.dataset && document.body.dataset.baseUrl; if (b) return b;
    var m = location.pathname.match(/^(.*?\/)(lab|notebooks|tree|voila|files|nbclassic)(\/|$)/);
    return m ? m[1] : '/';
  }
  window.__ocdJBase = jbase;
  window.__ocdResolveTileUrl = function (s) {
    if (!s) return s;
    var m = /^https?:\/\/(?:127\.0\.0\.1|localhost):(\d+)(\/.*)$/.exec(s);
    if (!m) return s;                       // relative / not a kernel-loopback URL
    var h = location.hostname;
    var isLocal = (h === '' || h === 'localhost' || h === '127.0.0.1' || h === '::1'
                   || location.protocol === 'vscode-webview:');
    if (isLocal) return s;                  // same machine: direct loopback works
    var jb = jbase(); if (jb.slice(-1) !== '/') jb += '/';
    return jb + 'ocdkit-tiles/' + m[1] + m[2];
  };
})();
""".strip()


# Async streamed tiles: each ``<image data-tile-async data-tile-src=URL>`` is a
# linked-grid tile whose pixels project on a background thread, so URL returns
# 204 until ready. Poll it; on 200 fetch the PNG once (blob → object URL) and set
# ``href`` so the SVG paints it. This is what lets the SVG composite return
# immediately (geometry only) and stream the heavy tile projection in after.
_TILE_STREAM_CONTROLLER_JS = r"""
(function () {
  var wrapper = document.querySelector('.ocd-svgfig[data-uid="__UID__"]');
  if (!wrapper) return;
  var XLINK = 'http://www.w3.org/1999/xlink';
  var imgs = Array.prototype.slice.call(
    wrapper.querySelectorAll('image[data-tile-async]'));
  imgs.forEach(function (im) {
    var src = (window.__ocdResolveTileUrl
               ? window.__ocdResolveTileUrl(im.getAttribute('data-tile-src'))
               : im.getAttribute('data-tile-src'));
    if (!src) return;
    var tries = 0;
    function poll() {
      fetch(src, { cache: 'no-store' }).then(function (r) {
        if (r.status === 200) return r.blob();
        if (tries++ < 400) setTimeout(poll, 250);   // 204 = not projected yet
        return null;
      }).then(function (blob) {
        if (!blob) return;
        var u = URL.createObjectURL(blob);
        im.setAttribute('href', u);
        im.setAttributeNS(XLINK, 'href', u);
        im.removeAttribute('data-tile-async');
        // The tile <image> is the full raster (rw×rh, often far larger than the
        // cell) clipped by the parent nested-<svg> overflow. Some webviews
        // (notably VS Code's) compute that clip when the cell is first laid out
        // and DON'T re-clip when a large image loads late — so the image bleeds
        // past the cell. Force a reflow of the cell so the clip re-applies now
        // that the image is present.
        var cell = im.closest('svg.ocd-linked-cell');
        if (cell) { var d = cell.style.display; cell.style.display = 'none';
                    void cell.getBoundingClientRect(); cell.style.display = d; }
      }).catch(function () { if (tries++ < 400) setTimeout(poll, 400); });
    }
    poll();
  });
})();
""".strip()


# Click-to-cycle normalization on the spectra y-axis label
# (``text.ocd-norm-toggle``): swaps the density canvas's line data between the
# three precomputed norms (self / bit-depth / global) and redraws, updating the
# label text. All three datasets ride on the canvas as data-ylines-*; no
# server round-trip, no re-render of the rest of the figure.
_NORM_TOGGLE_JS = r"""
(function () {
  var wrapper = document.querySelector('.ocd-svgfig[data-uid="__UID__"]');
  if (!wrapper) return;
  var MODES = ['self', 'bitdepth', 'global'];
  var DISP = { self: 'self-norm', bitdepth: 'bit-depth norm', global: 'global-norm' };
  wrapper.querySelectorAll('text.ocd-norm-toggle').forEach(function (lab) {
    lab.style.cursor = 'pointer';
    lab.addEventListener('click', function () {
      var cv = document.getElementById(lab.getAttribute('data-norm-target'));
      if (!cv) return;
      var cur = lab.getAttribute('data-norm-mode') || 'self';
      var nxt = MODES[(MODES.indexOf(cur) + 1) % MODES.length];
      if (cv.__sgVariants && cv.__sgVariants[nxt]) {
        // Streamed: swap the cached norm variant into the live cfg (no re-fetch).
        if (cv.__sgCfg) cv.__sgCfg.yLines = cv.__sgVariants[nxt];
      } else {
        var data = cv.getAttribute('data-ylines-' + nxt);
        if (!data) return;
        cv.setAttribute('data-ylines', data);     // base64: __spectraDraw re-decodes
      }
      cv.setAttribute('data-norm-mode', nxt);
      lab.setAttribute('data-norm-mode', nxt);
      var title = lab.getAttribute('data-norm-title') || 'Intensity';
      lab.textContent = title + ' (' + DISP[nxt] + ')';
      if (typeof cv.__spectraDraw === 'function') cv.__spectraDraw();
    });
  });
})();
""".strip()


# Clickable readout labels (``.sg-rlabel[data-sg-readout]``) that toggle the
# matching reference-spectra paths (``[data-sg-ref]``) on/off. Clicking a label
# hides that readout's references and grays the label; clicking again restores.
_REF_TOGGLE_JS = r"""
(function () {
  var wrapper = document.querySelector('.ocd-svgfig[data-uid="__UID__"]');
  if (!wrapper) return;
  var labels = wrapper.querySelectorAll('.sg-rlabel[data-sg-readout]');
  if (!labels.length) return;
  var GRAY = '#666';
  // Group reference paths + labels by readout.
  var refsBy = {}, labelBy = {};
  wrapper.querySelectorAll('[data-sg-ref]').forEach(function (r) {
    var k = r.getAttribute('data-sg-ref'); (refsBy[k] || (refsBy[k] = [])).push(r);
  });
  labels.forEach(function (l) {
    var k = l.getAttribute('data-sg-readout'); (labelBy[k] || (labelBy[k] = [])).push(l);
  });
  // Two independent "on" sources: ``pinned`` (label clicked) and ``temp`` (a
  // cell is hovered → show its classification readouts). A readout's reference
  // shows + its label colors when EITHER is set; otherwise OFF + gray.
  var pinned = {}, temp = {};
  function apply(ro) {
    var on = pinned[ro] || temp[ro];
    (refsBy[ro] || []).forEach(function (r) { r.style.display = on ? '' : 'none'; });
    (labelBy[ro] || []).forEach(function (l) {
      l.style.fill = on ? (l.getAttribute('data-sg-color') || '') : GRAY;
    });
  }
  var allRo = {};
  Object.keys(refsBy).forEach(function (k) { allRo[k] = 1; });
  Object.keys(labelBy).forEach(function (k) { allRo[k] = 1; });
  Object.keys(allRo).forEach(apply);          // default: all OFF + gray
  labels.forEach(function (lab) {
    if (lab.__sgRefWired) return; lab.__sgRefWired = true;
    var ro = lab.getAttribute('data-sg-readout');
    lab.style.cursor = 'pointer'; lab.style.userSelect = 'none';
    lab.addEventListener('click', function () { pinned[ro] = !pinned[ro]; apply(ro); });
  });
  // Hover API for the spectra controller (temporarily light up a cell's readouts).
  wrapper.__refSetTemp = function (roList) {
    var next = {}; (roList || []).forEach(function (r) { next[r] = true; });
    var changed = {};
    Object.keys(temp).forEach(function (k) { changed[k] = 1; });
    Object.keys(next).forEach(function (k) { changed[k] = 1; });
    temp = next; Object.keys(changed).forEach(apply);
  };
  wrapper.__refClearTemp = function () {
    var old = temp; temp = {}; Object.keys(old).forEach(apply);
  };
})();
""".strip()


_LUTS_JSON_CACHE = None


def _luts_json() -> str:
    """JSON ``{name: [256*4 uint8]}`` colormap LUTs for the live GPU tile
    colormap (the linked GL layer + cmap picker). Built once, cached. Mirrors
    the in-kernel tile server's ``_luts_json`` so the two viewers share maps."""
    global _LUTS_JSON_CACHE
    if _LUTS_JSON_CACHE is None:
        import json
        import numpy as np
        from cmap import Colormap
        x = np.linspace(0, 1, 256)
        out = {}
        for n in ("magma", "viridis", "gray", "plasma", "inferno", "cividis", "turbo"):
            try:
                out[n] = (Colormap(n)(x) * 255 + 0.5).astype(
                    np.uint8).reshape(-1).tolist()
            except Exception:
                pass
        _LUTS_JSON_CACHE = json.dumps(out)
    return _LUTS_JSON_CACHE


def interactive_shell(content_html: str, *,
                       save_button: bool = True,
                       copy_button: bool = True,
                       hdr_button: bool = True,
                       wrapper_style: str = '',
                       center: bool = True) -> str:
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
    center
        When ``True`` (default), wrap the shell in a
        ``<div style="text-align:center">`` so the inline-block
        ``.ocd-svgfig`` centres horizontally in its host container —
        the convention for Jupyter / dashboard figure output. The fixed-
        position zoom overlay is unaffected (it's anchored to the
        viewport). Pass ``False`` for layouts that want left alignment.
    """
    import secrets
    # lxml (SvgFigure.to_string) re-serializes empty elements self-closed
    # (``<canvas/>``). That's valid XML, but ``<canvas>`` is NOT a void
    # element in HTML — and this markup is delivered as ``text/html`` and
    # parsed by the HTML parser. There, ``<canvas/>`` doesn't self-close: the
    # parser keeps the canvas open and swallows the following ``</foreignObject>``
    # and the title ``<text>`` as canvas/foreignObject HTML content, so the
    # title renders in a leaked HTML context (no getBBox, fill forced black) =
    # invisible. Expand to an explicit close tag so HTML parsing stays correct.
    if '<canvas' in content_html:
        content_html = re.sub(r'(<canvas\b[^>]*?)\s*/>',
                              r'\1></canvas>', content_html)
    uid = secrets.token_hex(6)
    css = _SHELL_CSS.replace("__UID__", uid)
    js = _SHELL_JS.replace("__UID__", uid)
    # Colormap LUTs for the live GPU tile colormap (the linked GL layer renders
    # raw intensity tiles → normalize(lo,hi) → LUT, with a cmap picker + a
    # self/global/bit-depth readout-norm toggle — all live uniform swaps).
    if 'ocd-linked-cell' in content_html:
        js = "window.OCD_LUTS=" + _luts_json() + ";\n" + js
    # Prepend the shared WebGL2 label renderer + a per-figure controller so
    # any ``<canvas data-label-tile>`` emitted by image_grid renders live
    # (palette fill + outlines + hover-highlight), reusing the same engine
    # as the viewer. Only included when the markup actually has a label tile.
    if 'data-label-tile' in content_html:
        js = (_load_label_gl_js() + "\n"
              + _LABEL_CONTROLLER_JS.replace("__UID__", uid) + "\n" + js)
    # Same pattern for live WebGL2 spectra-density canvases (the "datashaded"
    # spectra panel rendered client-side at on-screen dpi instead of a baked
    # raster). Only included when the markup actually has one.
    if 'data-spectra-density' in content_html:
        js = (_load_spectra_gl_js() + "\n"
              + _SPECTRA_CONTROLLER_JS.replace("__UID__", uid) + "\n" + js)
    # Async streamed linked-grid tiles: poll + swap each ``data-tile-async``
    # image once its background projection lands on the tile server.
    if 'data-tile-async' in content_html:
        js = _TILE_STREAM_CONTROLLER_JS.replace("__UID__", uid) + "\n" + js
    # Click-to-cycle normalization on the spectra y-axis label.
    if 'ocd-norm-toggle' in content_html:
        js = _NORM_TOGGLE_JS.replace("__UID__", uid) + "\n" + js
    # Clickable readout labels toggling their reference spectra. Independent of
    # the live density (works for any SvgFigure that drew tagged labels/refs).
    if 'data-sg-readout' in content_html:
        js = _REF_TOGGLE_JS.replace("__UID__", uid) + "\n" + js
    # Tile-URL resolver MUST be defined before any controller/GL fetch runs, so
    # prepend it LAST (→ top of the concatenated script). Needed whenever the
    # markup carries baked loopback URLs — GL/stream tiles (data-tile-src) OR
    # click-to-expand hi-res (data-hires-href, e.g. tileserve /attach).
    if 'data-tile-src' in content_html or 'data-hires-href' in content_html:
        js = _TILE_BASE_RESOLVER_JS + "\n" + js
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
                f'<button class="ocd-savebtn" title="Save as PNG">'
                f'{_SHELL_SAVE_ICON}</button>')
        if copy_button:
            buttons.append(
                f'<button class="ocd-copybtn" title="Copy as PNG">'
                f'{_SHELL_COPY_ICON}</button>')
        actions = (
            f'<div class="ocd-svgfig-actions">{"".join(buttons)}</div>'
        )
    style_attr = f' style="{wrapper_style}"' if wrapper_style else ''
    shell_html = (
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
    if center:
        # Wrap in a centring div so the inline-block ``.ocd-svgfig``
        # centres in its parent. Jupyter / dashboard convention is to
        # show figures middle-of-cell; without this they hug the left.
        shell_html = f'<div style="text-align:center">{shell_html}</div>'
    return shell_html


# Internal alias kept for back-compat with any in-tree caller; new code
# should use :func:`interactive_shell`.
_build_interactive_shell = interactive_shell


__all__ = ["SvgFigure", "Axes", "interactive_shell"]
