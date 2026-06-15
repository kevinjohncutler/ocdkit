from __future__ import annotations
import os, math
from typing import List, Tuple
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.text import Text
from matplotlib.patches import Rectangle

__all__ = ["axes_union_bbox_inches", "save_figure_axes_pngs", "debug_overlay"]


# ----- internals -----

def _lw_px(art, dpi: float) -> float:
    lw_pt = 0.0
    if hasattr(art, "get_linewidth"):
        try:
            lw_pt = float(np.max(art.get_linewidth()))
        except Exception:
            try:
                lw_pt = float(art.get_linewidth() or 0.0)
            except Exception:
                lw_pt = 0.0
    elif hasattr(art, "get_linewidths"):
        try:
            arr = art.get_linewidths()
            if arr is not None and len(arr):
                lw_pt = float(np.max(arr))
        except Exception:
            pass
    return (lw_pt * dpi / 72.0) if lw_pt else 0.0


def _is_line_like(art) -> bool:
    try:
        from matplotlib.lines import Line2D
        from matplotlib.spines import Spine
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Patch
        if isinstance(art, (Line2D, Spine, LineCollection)):
            return True
        if isinstance(art, Patch) and not art.get_fill():
            return (getattr(art, "get_linewidth", lambda: 0)() or 0) > 0
    except Exception:
        pass
    return hasattr(art, "get_linewidth") and (art.get_linewidth() or 0) > 0


def _artist_extent_px(art, renderer) -> Bbox | None:
    """Prefer stroke-inclusive window extents. If unavailable, use path extents and add half-linewidth for line-like."""
    fig = art.figure
    dpi = fig.dpi

    # 1) Try stroke-inclusive window extent.
    b = None
    if hasattr(art, "get_window_extent"):
        try:
            b = art.get_window_extent(renderer)
        except Exception:
            b = None
        if b is not None:
            arr = np.array([b.x0, b.y0, b.x1, b.y1], float)
            if not np.isfinite(arr).all() or b.width <= 0 or b.height <= 0:
                b = None

    # 2) Fallback to geometric path extent, then grow by half linewidth for line-like.
    if b is None and hasattr(art, "get_path") and hasattr(art, "get_transform"):
        try:
            pext = art.get_path().transformed(art.get_transform()).get_extents()
        except Exception:
            pext = None
        if pext is not None:
            arr = np.array([pext.x0, pext.y0, pext.x1, pext.y1], float)
            if np.isfinite(arr).all() and pext.width > 0 and pext.height > 0:
                if _is_line_like(art):
                    pad = 0.5 * _lw_px(art, dpi)
                    if pad:
                        b = Bbox.from_extents(pext.x0 - pad, pext.y0 - pad, pext.x1 + pad, pext.y1 + pad)
                    else:
                        b = pext
                else:
                    b = pext

    if b is None:
        return None
    # Post-adjustments: ensure stroke and small text halos are included
    try:
        if _is_line_like(art):
            pad = 0.5 * _lw_px(art, dpi)
            if pad:
                b = Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)
        elif isinstance(art, Text):
            # Tiny halo to avoid antialias clipping
            b = Bbox.from_extents(b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1)
    except Exception:
        pass
    return b


def _gather_axes_extents(
    ax,
    renderer,
    include_legend: bool = True,
    include_tick_marks: bool = False,
    tick_label_clip_margin_px: int | None = 32,
) -> List[Bbox]:
    boxes: List[Bbox] = []

    if ax.patch is not None and ax.patch.get_visible():
        b = _artist_extent_px(ax.patch, renderer);  boxes += [b] if b else []

    for sp in ax.spines.values():
        if sp.get_visible():
            b = _artist_extent_px(sp, renderer);  boxes += [b] if b else []

    texts = [ax.xaxis.get_label(), ax.yaxis.get_label()] + list(ax.texts)
    for attr in ("title", "_left_title", "_right_title"):
        t = getattr(ax, attr, None)
        if t is not None:
            texts.append(t)
    for t in texts:
        if t is not None and t.get_visible():
            b = _artist_extent_px(t, renderer);  boxes += [b] if b else []

    ticks = (
        list(ax.xaxis.get_major_ticks()) + list(ax.xaxis.get_minor_ticks()) +
        list(ax.yaxis.get_major_ticks()) + list(ax.yaxis.get_minor_ticks())
    )
    # Precompute axes window bbox once for optional clipping of tick-related elements
    ax_win = ax.get_window_extent(renderer)
    for tk in ticks:
        comps = []
        if include_tick_marks:
            comps += [tk.tick1line, tk.tick2line]
        comps += [tk.gridline, tk.label1, tk.label2]
        for comp in comps:
            if comp is None or not comp.get_visible():
                continue
            # Skip empty text labels (e.g., pruned or NullFormatter)
            try:
                from matplotlib.text import Text
                if isinstance(comp, Text) and (not comp.get_text() or comp.get_text().strip() == ""):
                    continue
            except Exception:
                pass
            b = _artist_extent_px(comp, renderer)
            if b is None:
                continue
            # Optionally ignore tick-related boxes that are far outside the axes + margin
            if tick_label_clip_margin_px is not None:
                m = float(tick_label_clip_margin_px)
                clip_bb = Bbox.from_extents(ax_win.x0 - m, ax_win.y0 - m, ax_win.x1 + m, ax_win.y1 + m)
                inter = Bbox.intersection(b, clip_bb)
                if inter is None or inter.width <= 0 or inter.height <= 0:
                    continue
            boxes.append(b)

    for coll in ax.collections:
        if coll.get_visible():
            b = _artist_extent_px(coll, renderer);  boxes += [b] if b else []
    for ln in ax.lines:
        if ln.get_visible():
            b = _artist_extent_px(ln, renderer);  boxes += [b] if b else []
    for im in ax.images:
        if im.get_visible():
            b = _artist_extent_px(im, renderer);  boxes += [b] if b else []
    for p in ax.patches:
        if p.get_visible():
            b = _artist_extent_px(p, renderer);  boxes += [b] if b else []
    for art in ax.artists:
        if art.get_visible():
            b = _artist_extent_px(art, renderer);  boxes += [b] if b else []

    if include_legend:
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            b = _artist_extent_px(leg, renderer);  boxes += [b] if b else []

    return boxes


# ----- API -----

def axes_union_bbox_inches(
    ax,
    renderer=None,
    pad_px: int = 0,
    shave_px: int = 0,
    include_legend: bool = True,
    include_tick_marks: bool = False,
    tick_label_clip_margin_px: int | None = 32,
    snap_to_pixel: bool = True,
) -> Bbox:
    fig = ax.figure
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    boxes = _gather_axes_extents(
        ax,
        renderer,
        include_legend=include_legend,
        include_tick_marks=include_tick_marks,
        tick_label_clip_margin_px=tick_label_clip_margin_px,
    )
    if not boxes:
        raise ValueError("no visible artists on axes")

    x0 = min(b.x0 for b in boxes);  y0 = min(b.y0 for b in boxes)
    x1 = max(b.x1 for b in boxes);  y1 = max(b.y1 for b in boxes)

    # Apply padding in display pixels
    x0 = x0 - float(pad_px)
    y0 = y0 - float(pad_px)
    x1 = x1 + float(pad_px)
    y1 = y1 + float(pad_px)

    # Optionally snap outward to pixel edges
    if snap_to_pixel:
        x0 = math.floor(x0)
        y0 = math.floor(y0)
        x1 = math.ceil(x1)
        y1 = math.ceil(y1)

    if shave_px:
        x0 += int(shave_px);  y0 += int(shave_px)
        x1 -= int(shave_px);  y1 -= int(shave_px)

    return Bbox.from_extents(x0/fig.dpi, y0/fig.dpi, x1/fig.dpi, y1/fig.dpi)


def save_figure_axes_pngs(
    fig,
    out_dir: str,
    prefix: str = "ax",
    dpi: int | None = None,
    pad_px: int = 0,
    shave_px: int = 0,
    transparent: bool = False,
    include_legend: bool = True,
    isolate: bool = True,
    hide_figure_text: bool = True,
) -> List[Tuple[object, str, Bbox]]:
    os.makedirs(out_dir, exist_ok=True)
    axes = list(fig.axes)
    n = len(axes);  nd = max(2, len(str(n)))
    results: List[Tuple[object, str, Bbox]] = []

    axes_vis = [a.get_visible() for a in axes] if isolate else None
    fig_text_vis = [t.get_visible() for t in fig.texts] if hide_figure_text else None
    fig_legend_vis = [lg.get_visible() for lg in getattr(fig, "legends", [])] if hide_figure_text else None

    for i, ax in enumerate(axes, 1):
        if isolate:
            for a in axes:
                a.set_visible(a is ax)
        if hide_figure_text:
            for t in fig.texts: t.set_visible(False)
            for lg in getattr(fig, "legends", []): lg.set_visible(False)

        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bb = axes_union_bbox_inches(ax, renderer=r, pad_px=pad_px, shave_px=shave_px,
                                    include_legend=include_legend)
        path = os.path.join(out_dir, f"{prefix}_{i:0{nd}d}.png")
        fig.savefig(path, dpi=dpi or fig.dpi, bbox_inches=bb, pad_inches=0,
                    transparent=transparent, facecolor=fig.get_facecolor())
        results.append((ax, path, bb))

    if isolate and axes_vis is not None:
        for a, vis in zip(axes, axes_vis): a.set_visible(vis)
    if hide_figure_text and fig_text_vis is not None:
        for t, vis in zip(fig.texts, fig_text_vis): t.set_visible(vis)
    if hide_figure_text and fig_legend_vis is not None:
        for lg, vis in zip(getattr(fig, "legends", []), fig_legend_vis): lg.set_visible(vis)
    fig.canvas.draw()
    return results


def debug_overlay(fig, ax, renderer=None, out_path=None, include_legend=True):
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    boxes = _gather_axes_extents(ax, renderer, include_legend=include_legend)
    union = axes_union_bbox_inches(ax, renderer=renderer, pad_px=0, shave_px=0, include_legend=include_legend)

    ov = fig.add_axes([0, 0, 1, 1], label=f"__overlay_{id(ax)}", zorder=9999)
    ov.set_axis_off()
    for b in boxes:
        bf = b.transformed(fig.transFigure.inverted())
        r = Rectangle((bf.x0, bf.y0), bf.width, bf.height, transform=fig.transFigure,
                      facecolor=(1, 0, 0, 0.15), edgecolor=(1, 0, 0, 0.6), linewidth=1)
        ov.add_patch(r)
    uf = union.transformed(fig.transFigure.inverted())
    ur = Rectangle((uf.x0, uf.y0), uf.width, uf.height, transform=fig.transFigure,
                   facecolor=(0, 1, 1, 0.08), edgecolor=(0, 1, 1, 0.8), linewidth=2)
    ov.add_patch(ur)

    if out_path:
        fig.savefig(out_path, dpi=fig.dpi)
        fig.delaxes(ov); fig.canvas.draw()
        return out_path, boxes, union
