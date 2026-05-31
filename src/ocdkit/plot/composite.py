"""Layout primitives for multi-band/multi-channel composite spectra plots.

Domain-agnostic helpers: given a list of integer column intervals (one per
channel/band), compute per-sample pixel x-positions and per-band pixel
boundaries with optional gaps, padding, and width-mode reweighting. Also
includes a helper for sizing the stacked-readout-label top axis.
"""
from __future__ import annotations

from typing import Iterable

from .imports import *
from .text_metrics import chunk_words, measure_text


def compute_x_pixels_and_bands(
    n_points: int,
    intervals: Iterable[tuple[int, int]],
    plot_width: float,
    *,
    x_margin_px: float = 0,
    interval_width_mode: str = 'proportional',
    interval_gap_px: float = 0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Map per-column data indices to per-pixel x-positions across a
    multi-band plot.

    Parameters
    ----------
    n_points
        Total number of columns of source data.
    intervals
        Iterable of ``(start, stop)`` column ranges, one per band.
    plot_width
        Total available pixel width for the data area.
    x_margin_px
        Padding on the LEFT and RIGHT edges of the data area.
    interval_width_mode
        ``'proportional'`` — width proportional to column count (default).
        ``'equal'``        — every band gets the same width.
        ``'log'``          — log-weighted.
    interval_gap_px
        Total visual gap between adjacent bands in pixels (split half on
        either side of the band).

    Returns
    -------
    x_pixels
        ``(n_points,)`` float32 array, the x-pixel center for each source
        column (0 for columns not covered by any interval).
    band_bounds
        List of ``(x0, x1)`` per band giving the OUTER band rectangles,
        flush with the plot's left/right edges at the extremes.
    """
    intervals = list(intervals)
    n = len(intervals)
    if n == 0 or n_points == 0:
        return np.array([], dtype=np.float32), []
    avail_w = max(1.0, plot_width - 2.0 * x_margin_px)
    cols_per_ch = [stop - start for start, stop in intervals]
    if interval_width_mode == 'equal':
        weights = np.ones(n, dtype=float)
    elif interval_width_mode == 'log':
        weights = np.array([max(1.0, float(np.log(c + 1))) for c in cols_per_ch])
    else:
        weights = np.array([float(c) for c in cols_per_ch])
    if weights.sum() <= 0:
        weights = np.ones(n, dtype=float)
    weights = weights / weights.sum()
    channel_widths = weights * avail_w
    half_gap = max(0.0, interval_gap_px / 2.0)

    x_pixels = np.zeros(n_points, dtype=np.float32)
    band_bounds: list[tuple[float, float]] = []
    cur_x = float(x_margin_px)
    for i, (i_start, i_stop) in enumerate(intervals):
        cw = channel_widths[i]
        cols = i_stop - i_start
        if cols > 0:
            data_w = max(0.0, cw - 2.0 * half_gap)
            inner_left = cur_x + half_gap
            if cols == 1:
                x_pixels[i_start] = inner_left + data_w / 2.0
            else:
                inner_step = data_w / (cols - 1)
                x_pixels[i_start:i_stop] = (
                    inner_left + np.arange(cols) * inner_step
                ).astype(np.float32)
        x0 = 0.0 if i == 0 else float(cur_x)
        x1 = float(plot_width) if i == n - 1 else float(cur_x + cw)
        band_bounds.append((x0, x1))
        cur_x += cw
    return x_pixels, band_bounds


def measure_top_axis_height(
    channels_for_top: list,
    ch_to_band: dict,
    ch_to_readout_clean: dict,
    *,
    bold_fontsize_px: float,
    inter_line_gap_px: int,
    bottom_pad_to_plot: int,
    weight: str = 'regular',
) -> tuple[int, int, int]:
    """Compute the pixel height needed for a stacked per-channel top-axis
    label block.

    For each channel, the band's available width determines how many
    text lines (greedy-wrapped readout names) are required; the axis
    height is set by the channel that needs the most lines.

    Returns ``(top_axis_h, max_lines, text_h)`` where:

    - ``top_axis_h``: total pixel height (0 if no channels would have any
      labels).
    - ``max_lines``: lines required by the most-cramped channel.
    - ``text_h``: pixel height of a single line of label text.
    """
    _, text_h = measure_text('Rg', bold_fontsize_px, weight=weight)
    text_h = int(round(text_h))
    max_lines = 0
    for ch in channels_for_top:
        bx0, bx1 = ch_to_band[ch]
        words = ch_to_readout_clean.get(ch, [])
        lines = chunk_words(words, max(1, int(bx1 - bx0)),
                            bold_fontsize_px, weight=weight)
        max_lines = max(max_lines, len(lines))
    if max_lines > 0:
        top_axis_h = (max_lines * text_h
                      + max(0, max_lines - 1) * inter_line_gap_px
                      + bottom_pad_to_plot)
    else:
        top_axis_h = 0
    return top_axis_h, max_lines, text_h


__all__ = [
    'compute_x_pixels_and_bands',
    'measure_top_axis_height',
]
