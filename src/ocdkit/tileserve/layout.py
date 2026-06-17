"""Tile-grid + panel layout geometry — the single source of truth.

Ported verbatim (geometry only, no drawing) from the viewer's JS ``layout()``
(``viewer.py`` ``_GRID_HTML``). The browser overlay and the headless wgpu
compositor BOTH consume this so the interactive figure and a script-side export
are pixel-identical — one layout, not one-per-backend.

The figure is designed at a reference width ``Wref`` and uniformly scaled by
``k = container_w / Wref`` (width-driven: the aspect is fixed by cols/rows +
whether a panel is present, so an embedding iframe fits with no letterbox and
fonts scale with the figure). Everything below mirrors the JS constants and math
exactly; a parity test pins it to the live viewer (≤1 px).

:func:`compute_layout` returns a dict of pixel geometry; it does NOT draw.
"""
from __future__ import annotations

import re

# Reference-width design constants. This is the SINGLE source of truth for the
# figure geometry — the viewer (viewer.py layout()) consumes compute_layout's
# output as `_G` and no longer re-derives any of these.
WREF = 1000.0
PAD = 0.05
YAXW0 = 50.0      # y-axis strip reserved left of the grid (when a panel is present)
RM0 = 8.0         # right margin (matches, so tiles + panel share x-extent)
XLAB_H0 = 46.0    # x-label strip height below the panel
TOPAX0 = 22.0     # legacy top-axis height when no ported strip is supplied
# Viewer CHROME strip heights (Wref units). The viewer's _sizeBars() sizes the
# real DOM strips from these (× k) and the Python embed reserves them in the
# iframe aspect — both read them from compute_layout's output, so they live ONCE.
CTL_H0 = 30.0     # colormap/controls bar ABOVE the figure
HUD_H0 = 22.0     # debug hud strip BELOW the figure
TITLE_H0 = 26.0   # global title bar (only when a title is advertised)


def _label_pos(x, y, w, h, pos, pad):
    """Port of the viewer's ``_labelPos`` → (tx, ty, text_anchor, baseline)."""
    if pos == "top_left":
        return (x + pad, y + pad, "start", "hanging")
    if pos == "bottom_middle":
        return (x + w / 2, y + h - pad, "middle", "alphabetic")
    if pos == "bottom_left":
        return (x + pad, y + h - pad, "start", "alphabetic")
    if pos == "above_middle":
        return (x + w / 2, y - pad, "middle", "alphabetic")
    # 'top_middle' (default)
    return (x + w / 2, y + pad, "middle", "hanging")


def _placed(grid, layers):
    """Resolve the 2D ``grid`` (or a flat wrap over ``layers``) into placed cells
    and (cols, rows). Mirrors the head of layout(): exc* layers are auxiliary
    (RGB-compose inputs), never their own cells."""
    if grid:
        rows = len(grid)
        cols = max((len(r) for r in grid), default=0)
        placed = []
        for r, grow in enumerate(grid):
            for col, lbl in enumerate(grow):
                if isinstance(lbl, str):
                    placed.append({"label": lbl, "r": r, "col": col})
                elif isinstance(lbl, dict) and lbl.get("empty"):
                    placed.append({"empty": lbl["empty"], "r": r, "col": col})
        return placed, cols, rows
    order = [l for l in (layers or {}) if not re.match(r"^exc\d+$", l)]
    cols = len(order) if len(order) <= 6 else 7
    rows = (len(order) + cols - 1) // cols if cols else 0
    placed = [{"label": lbl, "r": i // cols, "col": i % cols}
              for i, lbl in enumerate(order)]
    return placed, cols, rows


def compute_layout(grid, layers, panel_axes, container_w, *,
                   label_pos="top_middle", label_pad=4.0):
    """Pixel geometry for the tile grid + optional panel, at ``container_w``.

    Parameters mirror the viewer's ``info``:
      ``grid``        — 2D list[list[str | {'empty': label} | None]] (or None).
      ``layers``      — ``info.layers`` dict (used only for the flat-wrap fallback).
      ``panel_axes``  — ``info.panel_axes`` or ``info.spectra_axes`` (or None).
      ``container_w`` — the container width in CSS px (the JS ``window.innerWidth``).

    Returns a dict with: ``k``, ``cols``, ``rows``, ``cells`` (each
    ``{label|empty, x, y, w, h, label_x, label_y, anchor, baseline}``),
    ``canvas_w``/``canvas_h`` (the tile-grid canvas), ``content_left``,
    ``content_w``, ``grid_h`` (tile span), ``has_panel``, ``is_xy``,
    ``top_ax_top``, ``top_ax_h``, ``panel_top``, ``panel_left``, ``panel_w``,
    ``panel_h``, ``full_h``. Geometry only — drawing is the caller's job.
    """
    placed, cols, rows = _placed(grid, layers)

    AX = panel_axes
    is_xy = bool(AX and AX.get("kind") == "xy")
    has_ax = bool(AX and (is_xy or AX.get("bands")))

    W0 = float(container_w)
    k = W0 / WREF

    padL0 = YAXW0 if has_ax else 0.0
    padR0 = RM0 if has_ax else 0.0
    contentW0 = max(20.0, WREF - padL0 - padR0)
    cw0 = contentW0 / (cols + (cols + 1) * PAD) if cols else contentW0
    gap0 = PAD * cw0
    totH0 = rows * cw0 + gap0 * (rows + 1)
    gridSpanH0 = rows * cw0 + max(0, rows - 1) * gap0

    TA = (AX.get("top_axis") if (has_ax and not is_xy) else None)
    # ── PANEL-ONLY mode (no image grid): the spectra panel IS the whole figure.
    # Used by ``scope.plot_spectra(backend='live')`` — a barcode has no FOV
    # images, just the spectra panel that normally sits below the scene grid.
    # Without a grid there's no ``gridSpanH0`` to size the panel, so the plot-box
    # height comes from ``AX['panel_h']`` (Wref units; default = content_w/3.2,
    # a wide single panel). The grid canvas collapses to zero; the top/bottom
    # axes + panel stack exactly as in the gridded case.
    panel_only = bool(has_ax and rows == 0 and cols == 0)
    if panel_only:
        totH0 = 0.0
        gap0 = 0.0
        # No grid cell, but the border/tick width is bw=cw*0.012 (viewer) — use a
        # nominal cell width so the spine/ticks stay ~1.5px, not the full panel
        # width (which would draw a ~14px-thick border).
        cw0 = WREF / 8.0
        plotW0 = contentW0
        plotH0 = float(AX.get("panel_h") or (contentW0 / 3.2))
        vgap0 = 0.0
        TOPAX_H0 = (TA["top_axis_h"] if (TA and TA.get("top_axis_h") is not None)
                    else (TOPAX0 if not is_xy else 0.0))
        fullH0 = TOPAX_H0 + plotH0 + XLAB_H0
    else:
        vgap0 = 0.0 if (TA or is_xy) else max(6.0, PAD * cw0)
        plotW0 = max(20.0, contentW0 - 2 * gap0)
        plotH0 = gridSpanH0 if has_ax else 0.0
        TOPAX_H0 = (TA["top_axis_h"] if (TA and TA.get("top_axis_h") is not None)
                    else (TOPAX0 if (has_ax and not is_xy) else 0.0))
        fullH0 = (totH0 + vgap0 + TOPAX_H0 + plotH0 + XLAB_H0) if has_ax else totH0

    # scale Wref-units → px
    padL = padL0 * k
    cw = cw0 * k
    gap = gap0 * k
    totH = totH0 * k
    vgap = vgap0 * k
    XLAB_H = XLAB_H0 * k
    TOPAX_H = TOPAX_H0 * k
    content_left = padL                       # xoff = 0 (fill width, no centering)
    content_w = contentW0 * k
    top_ax_top = (totH + vgap) if has_ax else 0.0
    panel_top = (totH + vgap + TOPAX_H) if has_ax else 0.0
    plotW = plotW0 * k if has_ax else 0.0
    plotH = plotH0 * k if has_ax else 0.0
    fullH = fullH0 * k

    cells = []
    lpad = label_pad * k
    for p in placed:
        x = content_left + gap + p["col"] * (cw + gap)
        y = gap + p["r"] * (cw + gap)
        lx, ly, anchor, baseline = _label_pos(x, y, cw, cw, label_pos, lpad)
        cell = {"x": x, "y": y, "w": cw, "h": cw,
                "label_x": lx, "label_y": ly, "anchor": anchor, "baseline": baseline}
        if "empty" in p:
            cell["empty"] = p["empty"]
        else:
            cell["label"] = p["label"]
        cells.append(cell)

    return {
        "k": k, "cols": cols, "rows": rows, "cells": cells,
        "cw": cw, "gap": gap,                       # cell side + gap (px) — viewer needs both
        "canvas_w": W0, "canvas_h": totH,
        "content_left": content_left, "content_w": content_w, "grid_h": gridSpanH0 * k,
        "has_panel": has_ax, "is_xy": is_xy,
        "top_ax_top": top_ax_top, "top_ax_h": TOPAX_H,
        "panel_top": panel_top, "panel_left": content_left + gap,
        "panel_w": plotW, "panel_h": plotH,
        "full_h": fullH,
        # chrome strip heights (Wref units; unscaled) — single source of truth
        # for both the viewer's _sizeBars and the Python embed aspect.
        "ctl_h": CTL_H0, "hud_h": HUD_H0, "title_h": TITLE_H0,
    }
