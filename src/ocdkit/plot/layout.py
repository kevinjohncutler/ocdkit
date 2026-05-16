"""Building-block helpers for SVG plot construction.

Provides the small set of utilities a new plot author needs to build a
plot from scratch *without* reinventing layout math:

  * ``PlotBox``         — explicit plot-area rectangle (x0, y0, w, h).
  * ``data_to_svg``     — map (data_x, data_y) → SVG canvas coords.
  * ``linear_ticks``    — pick sensible round-number ticks for [lo, hi].
  * ``log_ticks``       — pick decade-aligned ticks for a log axis.
  * ``draw_axis``       — emit a spine + tick marks + tick labels +
                          axis label, with the standard ``fig-spine``
                          / ``fig-tick`` / ``fig-tick-label`` /
                          ``fig-axis-label`` class scheme (so
                          ``Figure.apply_color_scheme`` finds them).
  * ``measure_text``    — re-exported from :mod:`text_metrics` for
                          one-import discoverability.

Style defaults (fontsizes, paddings, stroke widths) come from
:mod:`ocdkit.plot.style`; pass overrides as kwargs to ``draw_axis``.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Sequence, Callable

from .text_metrics import measure_text  # noqa: F401 (re-exported)
from . import style as _style


# ─── plot-area rectangle ──────────────────────────────────────────────

class PlotBox(NamedTuple):
    """Axis-aligned plot rectangle in SVG canvas coordinates.

    Origin is top-left (SVG convention); ``y`` increases downward.
    """
    x0: float
    y0: float
    w: float
    h: float

    @property
    def x1(self) -> float: return self.x0 + self.w
    @property
    def y1(self) -> float: return self.y0 + self.h
    @property
    def cx(self) -> float: return self.x0 + self.w / 2
    @property
    def cy(self) -> float: return self.y0 + self.h / 2


# ─── coordinate transforms ────────────────────────────────────────────

def data_to_svg(x: float, y: float, plot_box: PlotBox,
                x_range: tuple[float, float],
                y_range: tuple[float, float],
                *, log_x: bool = False, log_y: bool = False
                ) -> tuple[float, float]:
    """Map a single data point to SVG canvas coordinates.

    ``y`` is flipped (data-up → svg-down).  Log axes take log10 of
    the data and the range before mapping; values <= 0 land at the
    log-axis floor.
    """
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    if log_x:
        x = math.log10(max(x, 1e-300))
        x_lo = math.log10(max(x_lo, 1e-300))
        x_hi = math.log10(max(x_hi, 1e-300))
    if log_y:
        y = math.log10(max(y, 1e-300))
        y_lo = math.log10(max(y_lo, 1e-300))
        y_hi = math.log10(max(y_hi, 1e-300))
    sx = plot_box.x0 + (x - x_lo) / (x_hi - x_lo) * plot_box.w
    sy = plot_box.y0 + (y_hi - y) / (y_hi - y_lo) * plot_box.h
    return sx, sy


def data_to_svg_path(xs: Sequence[float], ys: Sequence[float],
                      plot_box: PlotBox,
                      x_range: tuple[float, float],
                      y_range: tuple[float, float],
                      *, log_x: bool = False, log_y: bool = False
                      ) -> str:
    """Build an SVG ``path`` ``d`` attribute polyline from samples."""
    pts = [data_to_svg(x, y, plot_box, x_range, y_range,
                       log_x=log_x, log_y=log_y) for x, y in zip(xs, ys)]
    if not pts:
        return ""
    head = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
    rest = " ".join(f"L {x:.2f} {y:.2f}" for x, y in pts[1:])
    return head + " " + rest if rest else head


# ─── tick locators ────────────────────────────────────────────────────

def linear_ticks(lo: float, hi: float, n_target: int = 5) -> list[float]:
    """Pick ~``n_target`` round-numbered ticks covering [lo, hi].

    Returns a list of tick values, all within [lo, hi].  Uses the
    1 / 2 / 2.5 / 5 × 10^k rule so steps are visually clean.
    """
    if hi <= lo:
        return [lo]
    raw_step = (hi - lo) / max(1, n_target)
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10 ** exponent)
    # Round to a "nice" step.
    if fraction < 1.5:
        nice = 1.0
    elif fraction < 3.0:
        nice = 2.0
    elif fraction < 7.0:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * (10 ** exponent)
    # Generate ticks.
    first = math.ceil(lo / step) * step
    ticks = []
    v = first
    while v <= hi + step * 1e-9:
        # Snap tiny floating-point noise (e.g. 0.30000000000000004 → 0.3).
        ticks.append(round(v, max(0, -int(exponent) + 2)))
        v += step
    return ticks


def log_ticks(lo: float, hi: float) -> list[tuple[float, str]]:
    """Decade ticks for a log axis, returned as ``(value, label)`` pairs.

    Labels are formatted as ``10ⁿ`` using Unicode superscript digits.
    """
    if lo <= 0 or hi <= 0 or hi <= lo:
        return []
    _SUPER = str.maketrans({'-': '⁻', '0': '⁰', '1': '¹', '2': '²',
                             '3': '³', '4': '⁴', '5': '⁵', '6': '⁶',
                             '7': '⁷', '8': '⁸', '9': '⁹'})
    e_lo = math.ceil(math.log10(lo))
    e_hi = math.floor(math.log10(hi))
    return [(10.0 ** e, f"10{str(e).translate(_SUPER)}")
            for e in range(int(e_lo), int(e_hi) + 1)]


# ─── axis emission ────────────────────────────────────────────────────

# Side → (axis direction, perpendicular direction we offset into for ticks/labels).
_SIDE_INFO = {
    "bottom": dict(orient="horizontal", offset_sign=+1),
    "top":    dict(orient="horizontal", offset_sign=-1),
    "left":   dict(orient="vertical",   offset_sign=-1),
    "right":  dict(orient="vertical",   offset_sign=+1),
}


def _per_tick(option, n: int, default):
    """Normalize a scalar-or-list option to a list of length ``n``.

    ``None`` → ``[default] * n``.  Scalar → repeated ``n`` times.  List
    → returned as-is (must have length ``n``).  Lets callers say
    ``color="#000"`` for uniform OR ``colors=["#ff0000", "#00ff00"]``
    for per-tick.
    """
    if option is None:
        return [default] * n
    if isinstance(option, (str, int, float)) or option is False or option is True:
        return [option] * n
    out = list(option)
    if len(out) != n:
        raise ValueError(
            f"per-tick option has {len(out)} entries, expected {n}")
    return out


def _spine_endpoints(plot_box: PlotBox, side: str
                      ) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) for the spine on the given side."""
    if side == "bottom":
        return plot_box.x0, plot_box.y1, plot_box.x1, plot_box.y1
    if side == "top":
        return plot_box.x0, plot_box.y0, plot_box.x1, plot_box.y0
    if side == "left":
        return plot_box.x0, plot_box.y0, plot_box.x0, plot_box.y1
    if side == "right":
        return plot_box.x1, plot_box.y0, plot_box.x1, plot_box.y1
    raise ValueError(f"side must be one of {sorted(_SIDE_INFO)}; got {side!r}")


def draw_spine(svg, plot_box: PlotBox, side: str = "bottom", *,
                color: str | None = None,
                width: float | None = None) -> None:
    """Emit the spine line for one side of the plot box.

    Single ``<line class="fig-spine" />`` element.  Pick this up via
    ``Figure.apply_color_scheme(axes=...)``.
    """
    d = _style.AXIS_DEFAULTS
    color = color if color is not None else d["spine_color"]
    width = width if width is not None else d["spine_width"]
    x1, y1, x2, y2 = _spine_endpoints(plot_box, side)
    svg.line(x1, y1, x2, y2, stroke=color, stroke_width=width,
              class_="fig-spine")


def draw_tick_marks(svg, plot_box: PlotBox, side: str, *,
                     positions: Sequence[float],
                     length: float | None = None,
                     color: str | None = None,
                     colors: Sequence[str] | None = None,
                     width: float | None = None,
                     widths: Sequence[float] | None = None,
                     ) -> None:
    """Emit tick mark lines at canvas-space ``positions`` along ``side``.

    Positions are **canvas coordinates** along the side's axis (x for
    horizontal sides, y for vertical) — *not* data values.  Callers
    that have data values should map them via :func:`data_to_svg`
    first.  This keeps the primitive backend-agnostic w.r.t. tick
    locators / data ranges.

    Per-tick variants (``colors``, ``widths``) override the uniform
    scalars (``color``, ``width``) when given.

    Each emitted line carries ``class="fig-tick"`` so
    ``apply_color_scheme(axes=...)`` finds it.
    """
    info = _SIDE_INFO[side]
    sign = info["offset_sign"]
    orient = info["orient"]
    d = _style.AXIS_DEFAULTS
    length = length if length is not None else d["tick_length"]
    n = len(positions)
    if n == 0:
        return
    color_list = _per_tick(colors, n, color if color is not None else d["spine_color"])
    width_list = _per_tick(widths, n, width if width is not None else d["spine_width"])

    if orient == "horizontal":
        ty = plot_box.y1 if side == "bottom" else plot_box.y0
        for x, c, w in zip(positions, color_list, width_list):
            svg.line(x, ty, x, ty + sign * length,
                      stroke=c, stroke_width=w, class_="fig-tick")
    else:
        tx = plot_box.x0 if side == "left" else plot_box.x1
        for y, c, w in zip(positions, color_list, width_list):
            svg.line(tx, y, tx + sign * length, y,
                      stroke=c, stroke_width=w, class_="fig-tick")


def draw_tick_labels(svg, plot_box: PlotBox, side: str, *,
                      positions: Sequence[float],
                      labels: Sequence[str],
                      offset: float | None = None,
                      size: float | None = None,
                      sizes: Sequence[float] | None = None,
                      color: str | None = None,
                      colors: Sequence[str] | None = None,
                      weight: str = "normal",
                      weights: Sequence[str] | None = None,
                      family: str | None = None,
                      families: Sequence[str | None] | None = None,
                      anchor: str | None = None,
                      baseline: str | None = None,
                      ) -> None:
    """Emit tick label text at canvas-space ``positions`` along ``side``.

    ``offset`` is the perpendicular distance from the spine to the
    label baseline (defaults to ``tick_length + tick_label_padding``
    from :mod:`ocdkit.plot.style`).  Per-tick variants for size,
    color, weight, family override the uniform scalars.

    Default ``anchor`` / ``baseline`` are chosen per side so labels
    sit naturally outside the spine; callers can override.

    Each emitted text carries ``class="fig-tick-label"``.
    """
    info = _SIDE_INFO[side]
    sign = info["offset_sign"]
    orient = info["orient"]
    d = _style.AXIS_DEFAULTS
    size = size if size is not None else d["tick_label_size"]
    if offset is None:
        offset = d["tick_length"] + d["tick_label_padding"]
    n = len(positions)
    if n == 0:
        return
    if len(labels) != n:
        raise ValueError(
            f"labels has {len(labels)} entries, expected {n}")
    size_list = _per_tick(sizes, n, size)
    color_list = _per_tick(colors, n, color if color is not None else d["text_color"])
    weight_list = _per_tick(weights, n, weight)
    family_list = _per_tick(families, n, family)

    # Choose sensible defaults for anchor + baseline per side.
    if orient == "horizontal":
        default_anchor = "middle"
        default_baseline = "hanging" if sign > 0 else "alphabetic"
    else:
        default_anchor = "start" if sign > 0 else "end"
        default_baseline = "middle"
    anchor = anchor if anchor is not None else default_anchor
    baseline = baseline if baseline is not None else default_baseline

    if orient == "horizontal":
        ty = plot_box.y1 if side == "bottom" else plot_box.y0
        label_y = ty + sign * offset
        for x, lab, s, c, w, fam in zip(positions, labels, size_list,
                                          color_list, weight_list, family_list):
            kwargs = dict(fill=c, size=s, anchor=anchor, baseline=baseline,
                          class_="fig-tick-label")
            if w != "normal":
                kwargs["weight"] = w
            if fam:
                kwargs["family"] = fam
            svg.text(x, label_y, lab, **kwargs)
    else:
        tx = plot_box.x0 if side == "left" else plot_box.x1
        label_x = tx + sign * offset
        for y, lab, s, c, w, fam in zip(positions, labels, size_list,
                                          color_list, weight_list, family_list):
            kwargs = dict(fill=c, size=s, anchor=anchor, baseline=baseline,
                          class_="fig-tick-label")
            if w != "normal":
                kwargs["weight"] = w
            if fam:
                kwargs["family"] = fam
            svg.text(label_x, y, lab, **kwargs)


def draw_axis(svg, plot_box: PlotBox, side: str = "bottom", *,
              data_range: tuple[float, float] | None = None,
              ticks: Sequence[float] | None = None,
              tick_labels: Sequence[str] | None = None,
              tick_locator: Callable[[float, float], list[float]] | None = None,
              tick_label_fmt: Callable[[float], str] | None = None,
              axis_label: str | None = None,
              log: bool = False,
              draw_spine: bool = True,
              # style overrides (default → ocdkit.plot.style)
              tick_length: float | None = None,
              tick_label_size: float | None = None,
              tick_label_padding: float | None = None,
              axis_label_size: float | None = None,
              axis_label_padding: float | None = None,
              spine_width: float | None = None,
              spine_color: str | None = None,
              text_color: str | None = None,
              color: str | None = None,
              ) -> None:
    """Emit a complete axis (spine + ticks + tick labels + axis label).

    All elements carry the ``fig-*`` class scheme expected by
    :meth:`hiprpy._figure.Figure.apply_color_scheme`, so downstream
    recolor calls (``apply_color_scheme(font=..., axes=...)``) target
    the right things.

    Parameters
    ----------
    svg : ocdkit.plot.SVG
        Builder to emit into.
    plot_box : PlotBox
        The plot-area rectangle this axis hugs.
    side : {'bottom', 'top', 'left', 'right'}
        Which side of the plot box this axis sits on.
    data_range : (lo, hi), optional
        Data-axis range; required unless ``ticks`` is supplied.
    ticks : sequence of float, optional
        Explicit tick positions in data coordinates.  If not given,
        ``tick_locator`` (or ``linear_ticks`` / ``log_ticks`` by
        default) picks them from ``data_range``.
    tick_labels : sequence of str, optional
        One label per tick.  Defaults to ``tick_label_fmt(v)``.
    tick_locator : callable, optional
        ``(lo, hi) -> list[float]``.  Default: ``linear_ticks`` (or
        ``log_ticks`` if ``log=True``).
    tick_label_fmt : callable, optional
        ``float -> str`` formatter.  Default: ``str``.
    axis_label : str, optional
        Axis title (e.g. "x" or "amplitude").
    log : bool
        Log-scale this axis.
    color : str, optional
        Single override for all axis colors (spine + ticks + text).
        Pass ``spine_width`` etc. separately for finer control.
    """
    info = _SIDE_INFO[side]
    orient = info["orient"]
    sign = info["offset_sign"]

    d = _style.AXIS_DEFAULTS
    tick_length = tick_length if tick_length is not None else d["tick_length"]
    tick_label_size = tick_label_size if tick_label_size is not None else d["tick_label_size"]
    tick_label_padding = (tick_label_padding if tick_label_padding is not None
                           else d["tick_label_padding"])
    axis_label_size = axis_label_size if axis_label_size is not None else d["axis_label_size"]
    axis_label_padding = (axis_label_padding if axis_label_padding is not None
                           else d["axis_label_padding"])
    spine_width = spine_width if spine_width is not None else d["spine_width"]
    # ``color`` is a convenience override for both spine + text; individual
    # ``spine_color`` / ``text_color`` win over it.
    spine_color = (spine_color if spine_color is not None
                   else color if color is not None
                   else d["spine_color"])
    text_color = (text_color if text_color is not None
                  else color if color is not None
                  else d["text_color"])

    # Spine.
    if draw_spine:
        draw_spine_fn = draw_spine_  # alias to the module-level primitive
        draw_spine_fn(svg, plot_box, side, color=spine_color, width=spine_width)

    # Resolve ticks (data values + labels).
    if ticks is None:
        if data_range is None:
            raise ValueError("draw_axis needs either ticks= or data_range=")
        locator = tick_locator or (log_ticks if log else linear_ticks)
        located = locator(*data_range)
        if log and located and isinstance(located[0], tuple):
            ticks = [v for v, _ in located]
            tick_labels = [lab for _, lab in located]
        else:
            ticks = located
    if tick_labels is None:
        fmt = tick_label_fmt or (lambda v: f"{v:g}")
        tick_labels = [fmt(v) for v in ticks]

    # Map data-space tick values to canvas positions.
    if data_range is None:
        raise ValueError("draw_axis needs data_range= to position the ticks")
    lo, hi = data_range
    if hi == lo:
        return  # degenerate
    if log:
        lo, hi = math.log10(max(lo, 1e-300)), math.log10(max(hi, 1e-300))
    def _frac(v):
        v_eff = math.log10(max(v, 1e-300)) if log else v
        return (v_eff - lo) / (hi - lo)
    if orient == "horizontal":
        positions = [plot_box.x0 + _frac(v) * plot_box.w for v in ticks]
    else:
        positions = [plot_box.y0 + (1 - _frac(v)) * plot_box.h for v in ticks]

    # Compose: tick marks + labels via the atomic primitives.
    draw_tick_marks(svg, plot_box, side, positions=positions,
                     length=tick_length, color=spine_color, width=spine_width)
    draw_tick_labels(svg, plot_box, side, positions=positions,
                      labels=[str(lab) for lab in tick_labels],
                      offset=tick_length + tick_label_padding,
                      size=tick_label_size, color=text_color)

    # Axis label.
    if axis_label is None:
        return
    if orient == "horizontal":
        lx = plot_box.cx
        sy1 = plot_box.y1 if side == "bottom" else plot_box.y0
        ly = sy1 + sign * (tick_length + tick_label_padding +
                            tick_label_size + axis_label_padding +
                            (axis_label_size if sign > 0 else 0))
        svg.text(lx, ly, axis_label, fill=text_color,
                  size=axis_label_size, anchor="middle",
                  baseline="hanging" if sign > 0 else "alphabetic",
                  class_="fig-axis-label")
    else:
        sx1 = plot_box.x0 if side == "left" else plot_box.x1
        lx = sx1 + sign * (tick_length + tick_label_padding +
                            axis_label_padding)
        ly = plot_box.cy
        transform = f"rotate(-90 {lx:.2f} {ly:.2f})"
        svg.text(lx, ly, axis_label, fill=text_color,
                  size=axis_label_size, anchor="middle",
                  baseline="middle", transform=transform,
                  class_="fig-axis-label")


# Alias so draw_axis can refer to the spine primitive without a recursive name.
draw_spine_ = draw_spine
