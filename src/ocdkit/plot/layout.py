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

    # Spine endpoints.
    if side == "bottom":
        sx1, sy1, sx2, sy2 = plot_box.x0, plot_box.y1, plot_box.x1, plot_box.y1
    elif side == "top":
        sx1, sy1, sx2, sy2 = plot_box.x0, plot_box.y0, plot_box.x1, plot_box.y0
    elif side == "left":
        sx1, sy1, sx2, sy2 = plot_box.x0, plot_box.y0, plot_box.x0, plot_box.y1
    elif side == "right":
        sx1, sy1, sx2, sy2 = plot_box.x1, plot_box.y0, plot_box.x1, plot_box.y1
    else:
        raise ValueError(f"side must be one of {sorted(_SIDE_INFO)}; got {side!r}")
    if draw_spine:
        svg.line(sx1, sy1, sx2, sy2, stroke=spine_color,
                  stroke_width=spine_width, class_="fig-spine")

    # Resolve ticks.
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

    # Map tick values to canvas positions along the spine.
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

    # Tick marks + labels.
    for v, lab in zip(ticks, tick_labels):
        f = _frac(v)
        if orient == "horizontal":
            tx = plot_box.x0 + f * plot_box.w
            ty = sy1
            svg.line(tx, ty, tx, ty + sign * tick_length,
                      stroke=spine_color, stroke_width=spine_width,
                      class_="fig-tick")
            label_x = tx
            label_y = ty + sign * (tick_length + tick_label_padding +
                                    (tick_label_size if sign > 0 else 0))
            baseline = "hanging" if sign > 0 else "alphabetic"
            svg.text(label_x, label_y, str(lab), fill=text_color,
                      size=tick_label_size, anchor="middle",
                      baseline=baseline, class_="fig-tick-label")
        else:  # vertical
            tx = sx1
            ty = plot_box.y0 + (1 - f) * plot_box.h
            svg.line(tx, ty, tx + sign * tick_length, ty,
                      stroke=spine_color, stroke_width=spine_width,
                      class_="fig-tick")
            label_x = tx + sign * (tick_length + tick_label_padding)
            # baseline="middle" centers the text on ty — no extra y-offset needed.
            anchor = "start" if sign > 0 else "end"
            svg.text(label_x, ty, str(lab), fill=text_color,
                      size=tick_label_size, anchor=anchor,
                      baseline="middle", class_="fig-tick-label")

    # Axis label.
    if axis_label is None:
        return
    if orient == "horizontal":
        lx = plot_box.cx
        ly = sy1 + sign * (tick_length + tick_label_padding +
                            tick_label_size + axis_label_padding +
                            (axis_label_size if sign > 0 else 0))
        svg.text(lx, ly, axis_label, fill=text_color,
                  size=axis_label_size, anchor="middle",
                  baseline="hanging" if sign > 0 else "alphabetic",
                  class_="fig-axis-label")
    else:
        # Rotated -90 around the anchor point.
        lx = sx1 + sign * (tick_length + tick_label_padding +
                            axis_label_padding)
        ly = plot_box.cy
        transform = f"rotate(-90 {lx:.2f} {ly:.2f})"
        svg.text(lx, ly, axis_label, fill=text_color,
                  size=axis_label_size, anchor="middle",
                  baseline="middle", transform=transform,
                  class_="fig-axis-label")
