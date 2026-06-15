"""
Thin wrappers around a robust Matplotlib-only axes cropper, exposing the
interface used by the PPTX exporter (:mod:`ocdkit.io.pptx`).

Key functions preserved for pptx integration:
- export_axes_to_buffers: return per-axes PNG buffers and pixel bboxes
- save_full_figure_tight: save full figure with tight bbox (comparison)
- save_full_figure_overview: draw cyan boxes over full figure for crops
- save_full_figure_component_overlay: draw red component boxes + cyan unions
- _ensure_debug_dir: centralized debug artifact folder at project root
- FigureCanvas: Agg canvas re-export

The core cropping is delegated to mpl_axes_cropper.axes_union_bbox_inches.
"""
from __future__ import annotations
import io
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches

from .mpl_axes_cropper import axes_union_bbox_inches as _axes_union_bbox_inches
from .mpl_axes_cropper import _gather_axes_extents as _gather_axes_extents  # type: ignore
from .mpl_axes_cropper import _artist_extent_px as _artist_extent_px  # type: ignore
from .mpl_axes_cropper import _artist_extent_px as _artist_extent_px  # type: ignore

# Public constant retained
DPI = 600


def _project_root(start: Optional[Path] = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    cur = start
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return p if p.name != "src" else p.parent
    return Path(__file__).resolve().parents[3]


def _ensure_debug_dir() -> Path:
    root = _project_root()
    debug_dir = root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _inches_to_disp_bbox(bb_in: Bbox, dpi: float) -> Bbox:
    return Bbox.from_extents(bb_in.x0 * dpi, bb_in.y0 * dpi, bb_in.x1 * dpi, bb_in.y1 * dpi)


def export_axes_to_buffers(
    fig: Figure,
    dpi: int = DPI,
    pad_inches: float = 0.0,
    log_path: Optional[os.PathLike] = None,
    *,
    include_legend: bool = True,
    pad_px: int = 1,
    shave_px: int = 0,
    hide_figure_text: bool = True,
    include_tick_marks: bool = False,
    tick_label_clip_margin_px: int | None = 32,
    snap_to_pixel: bool = True,
    skip_empty_axes: bool = True,
):
    """Export each visible axes of a figure to an in-memory PNG buffer.

    Returns a list of dicts with keys:
      - index: subplot index (1-based by fig.axes order)
      - bbox_disp: Bbox in display (pixel) coords on the figure canvas
      - bbox_inches: Bbox in inches for use with savefig's bbox_inches
      - png: bytes of the PNG image (RGBA)
      - size_px: (width, height) of the crop in pixels
    """
    orig_dpi = fig.dpi
    fig.set_dpi(dpi)
    canvas = FigureCanvas(fig)
    canvas.draw()

    axes = list(fig.axes)
    axes_vis = [ax.get_visible() for ax in axes]
    fig_text_vis = [t.get_visible() for t in fig.texts] if hide_figure_text else None
    fig_legend_vis = [lg.get_visible() for lg in getattr(fig, "legends", [])] if hide_figure_text else None

    log_fh = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")

    results = []
    try:
        for i, ax in enumerate(axes, start=1):
            # Optionally skip axes that have no meaningful content
            if skip_empty_axes:
                def _axes_is_empty(ax_) -> bool:
                    # data-like content
                    if any(getattr(ln, "get_visible", lambda: False)() for ln in ax_.lines):
                        return False
                    if any(getattr(im, "get_visible", lambda: False)() for im in ax_.images):
                        return False
                    if any(getattr(co, "get_visible", lambda: False)() for co in ax_.collections):
                        return False
                    # patches excluding the axes background patch
                    for p in ax_.patches:
                        if p is getattr(ax_, "patch", None):
                            continue
                        if getattr(p, "get_visible", lambda: False)():
                            return False
                    # other artists
                    if any(getattr(ar, "get_visible", lambda: False)() for ar in ax_.artists):
                        return False
                    # legends
                    lg = ax_.get_legend()
                    if lg is not None and lg.get_visible():
                        return False
                    # axis-owned texts (titles/labels/ax.texts)
                    texts = [ax_.xaxis.get_label(), ax_.yaxis.get_label(),
                             getattr(ax_, "title", None), getattr(ax_, "_left_title", None), getattr(ax_, "_right_title", None)]
                    texts += list(getattr(ax_, "texts", []))
                    for t in texts:
                        if t is None:
                            continue
                        if t.get_visible() and str(getattr(t, "get_text", lambda: "")()).strip():
                            return False
                    return True

                if _axes_is_empty(ax):
                    if log_fh is not None:
                        print(f"[Axes {i:02d}] skipped: empty", file=log_fh)
                    continue

            # Isolate target axes during export to avoid bleed-through
            for a in axes:
                a.set_visible(a is ax)
            if hide_figure_text:
                for t in fig.texts:
                    t.set_visible(False)
                for lg in getattr(fig, "legends", []):
                    lg.set_visible(False)

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # Union in inches; convert to display pixels for placement
            bb_in = _axes_union_bbox_inches(
                ax,
                renderer=renderer,
                pad_px=pad_px,
                shave_px=shave_px,
                include_legend=include_legend,
                include_tick_marks=include_tick_marks,
                tick_label_clip_margin_px=tick_label_clip_margin_px,
                snap_to_pixel=snap_to_pixel,
            )
            bb_px = _inches_to_disp_bbox(bb_in, fig.dpi)

            try:
                def _union_bb(base, extra):
                    if extra is None:
                        return base
                    return Bbox.from_extents(
                        min(base.x0, extra.x0),
                        min(base.y0, extra.y0),
                        max(base.x1, extra.x1),
                        max(base.y1, extra.y1),
                    )

                debug_info = []
                pad_left = pad_right = pad_bottom = pad_top = 0.0

                for spine in getattr(ax, "spines", {}).values():
                    if not spine.get_visible():
                        continue

                    try:
                        path = spine.get_path().transformed(spine.get_transform())
                        path_bb = path.get_extents()
                    except Exception:
                        path = None
                        path_bb = None

                    try:
                        stroke_bb = spine.get_window_extent(renderer)
                    except Exception:
                        stroke_bb = None

                    if path_bb is None and stroke_bb is None:
                        continue

                    if path_bb is not None:
                        bb_px = _union_bb(bb_px, path_bb)
                    if stroke_bb is not None:
                        bb_px = _union_bb(bb_px, stroke_bb)

                    try:
                        lw_pt = spine.get_linewidth() or 0.0
                    except Exception:
                        lw_pt = 0.0

                    if lw_pt <= 0:
                        continue

                    half_px = (lw_pt * fig.dpi / 72.0) / 2.0
                    cap = None
                    if hasattr(spine, "get_capstyle"):
                        try:
                            cap = spine.get_capstyle()
                        except Exception:
                            cap = None
                    tangent_pad = half_px if cap in {"round", "projecting", "square"} else 0.0

                    if path_bb is not None and stroke_bb is not None:
                        extra_left = max(0.0, bb_px.x0 - stroke_bb.x0)
                        extra_right = max(0.0, stroke_bb.x1 - bb_px.x1)
                        extra_bottom = max(0.0, bb_px.y0 - stroke_bb.y0)
                        extra_top = max(0.0, stroke_bb.y1 - bb_px.y1)
                        need_left = max(0.0, half_px - extra_left)
                        need_right = max(0.0, half_px - extra_right)
                        need_bottom = max(0.0, half_px - extra_bottom)
                        need_top = max(0.0, half_px - extra_top)
                    else:
                        need_left = need_right = need_bottom = need_top = half_px

                    need_left = max(need_left, tangent_pad)
                    need_right = max(need_right, tangent_pad)
                    need_bottom = max(need_bottom, tangent_pad)
                    need_top = max(need_top, tangent_pad)

                    pad_left = max(pad_left, need_left)
                    pad_right = max(pad_right, need_right)
                    pad_bottom = max(pad_bottom, need_bottom)
                    pad_top = max(pad_top, need_top)
                    debug_info.append((getattr(spine, "spine_type", None), lw_pt, cap, need_left, need_right, need_bottom, need_top))

                if pad_left or pad_right or pad_bottom or pad_top:
                    bb_px = Bbox.from_extents(
                        bb_px.x0 - pad_left,
                        bb_px.y0 - pad_bottom,
                        bb_px.x1 + pad_right,
                        bb_px.y1 + pad_top,
                    )

                bb_in = Bbox.from_extents(
                    bb_px.x0 / fig.dpi,
                    bb_px.y0 / fig.dpi,
                    bb_px.x1 / fig.dpi,
                    bb_px.y1 / fig.dpi,
                )
            except Exception:
                pass

            # Save into memory
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bb_in, pad_inches=pad_inches,
                        facecolor=fig.get_facecolor())
            w_px, h_px = int(round(bb_px.width)), int(round(bb_px.height))
            results.append({
                "index": i,
                "bbox_disp": bb_px,
                "bbox_inches": bb_in,
                "png": buf.getvalue(),
                "size_px": (w_px, h_px),
            })

            if log_fh is not None:
                print(f"[Axes {i:02d}] disp=({bb_px.x0:.1f},{bb_px.y0:.1f})→({bb_px.x1:.1f},{bb_px.y1:.1f}) px={w_px}x{h_px}", file=log_fh)
                if debug_info:
                    for info in debug_info:
                        ori, lw_pt, cap, pL, pR, pB, pT = info
                        print(
                            f"   spine '{ori}': lw_pt={lw_pt:.2f} cap={cap} pads=({pL:.2f},{pR:.2f},{pB:.2f},{pT:.2f})",
                            file=log_fh,
                        )
                # Debug: axis label bboxes and whether they extend beyond canvas
                W, H = fig.canvas.get_width_height()
                bx = _artist_extent_px(ax.xaxis.get_label(), renderer)
                by = _artist_extent_px(ax.yaxis.get_label(), renderer)
                def _fmt(bb):
                    if bb is None:
                        return "None"
                    flags = []
                    if bb.x0 < 0: flags.append("x0<0")
                    if bb.y0 < 0: flags.append("y0<0")
                    if bb.x1 > W: flags.append("x1>W")
                    if bb.y1 > H: flags.append("y1>H")
                    suf = (" oob=" + ",".join(flags)) if flags else ""
                    return f"({bb.x0:.1f},{bb.y0:.1f})→({bb.x1:.1f},{bb.y1:.1f}){suf}"
                print(f"   x-label: {_fmt(bx)}", file=log_fh)
                print(f"   y-label: {_fmt(by)}", file=log_fh)
    finally:
        # Restore visibilities
        for a, vis in zip(axes, axes_vis):
            a.set_visible(vis)
        if hide_figure_text and fig_text_vis is not None:
            for t, vis in zip(fig.texts, fig_text_vis):
                t.set_visible(vis)
        if hide_figure_text and fig_legend_vis is not None:
            for lg, vis in zip(getattr(fig, "legends", []), fig_legend_vis):
                lg.set_visible(vis)
        if log_fh is not None:
            log_fh.flush(); log_fh.close()
        try:
            fig.set_dpi(orig_dpi)
        except Exception:
            pass

    return results


def export_figure_texts_to_buffers(
    fig: Figure,
    dpi: int = DPI,
    pad_inches: float = 0.0,
):
    """Export each visible figure-level Text (fig.text) as an RGBA PNG buffer.

    The export isolates one Text at a time (hides other texts and axes), computes
    its window extent in display pixels, converts to inches, and uses savefig with
    bbox_inches to rasterize just that element.

    Returns list of dicts with keys: bbox_disp, bbox_inches, png, size_px, text, index
    """
    # Ensure canvas is ready
    orig_dpi = fig.dpi
    fig.set_dpi(dpi)
    canvas = FigureCanvas(fig)
    canvas.draw()

    axes = list(fig.axes)
    axes_vis = [ax.get_visible() for ax in axes]
    texts = list(getattr(fig, "texts", []))
    texts_vis = [t.get_visible() for t in texts]

    results = []
    try:
        for i, t in enumerate(texts, start=1):
            try:
                if not t.get_visible():
                    continue
                s = (t.get_text() or "").strip()
                if not s:
                    continue
                # Only export pure figure-level annotations
                if t.get_transform() is not getattr(fig, "transFigure", None):
                    continue

                # Isolate this text: hide axes and other fig texts
                for ax, vis in zip(axes, axes_vis):
                    ax.set_visible(False)
                for tj in texts:
                    tj.set_visible(tj is t)

                fig.canvas.draw(); renderer = fig.canvas.get_renderer()
                bb_px = _artist_extent_px(t, renderer)
                if bb_px is None:
                    continue
                # Convert to inches bbox for savefig
                bb_in = Bbox.from_extents(bb_px.x0 / fig.dpi, bb_px.y0 / fig.dpi, bb_px.x1 / fig.dpi, bb_px.y1 / fig.dpi)

                # Save into memory
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bb_in, pad_inches=pad_inches,
                            facecolor=fig.get_facecolor())

                w_px, h_px = int(round(bb_px.width)), int(round(bb_px.height))
                results.append({
                    "index": i,
                    "text": s,
                    "bbox_disp": bb_px,
                    "bbox_inches": bb_in,
                    "png": buf.getvalue(),
                    "size_px": (w_px, h_px),
                })
            except Exception:
                continue
    finally:
        # Restore visibilities
        for ax, vis in zip(axes, axes_vis):
            ax.set_visible(vis)
        for t, vis in zip(texts, texts_vis):
            t.set_visible(vis)
        try:
            fig.set_dpi(orig_dpi)
        except Exception:
            pass

    return results

def save_full_figure(fig: Figure, path: str, dpi: int = DPI) -> None:
    FigureCanvas(fig).draw()
    out_path = _ensure_debug_dir() / Path(path).name
    fig.savefig(out_path, dpi=dpi)


def save_full_figure_tight(fig: Figure, path: str, dpi: int = DPI, pad_inches: float = 0.02) -> None:
    FigureCanvas(fig).draw()
    out_path = _ensure_debug_dir() / Path(path).name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)


def save_full_figure_overview(fig: Figure, path: str, dpi: int, axis_bboxes: List[Tuple[int, Bbox]]) -> None:
    """Render the full figure and draw cyan rectangles for each axes crop.
    axis_bboxes: list of (index, Bbox in pixel/display coords)
    """
    _orig_dpi = fig.dpi
    try:
        fig.set_dpi(dpi)
        canvas = FigureCanvas(fig); canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        overlay = Figure(figsize=fig.get_size_inches(), dpi=fig.dpi); _ = FigureCanvas(overlay)
        try:
            overlay.patch.set_facecolor(fig.get_facecolor())
        except Exception:
            pass
        ax = overlay.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.imshow(img, origin="upper", extent=[0, w, h, 0])
        for idx, bb in axis_bboxes:
            x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
            y0_img, y1_img = h - y0, h - y1
            rect = mpatches.Rectangle((x0, y1_img), x1 - x0, y0_img - y1_img,
                                      fill=False, edgecolor="cyan", linewidth=2, transform=ax.transData)
            ax.add_patch(rect)
            ax.text(x0 + 4, y1_img + 14, f"{idx:02d}", color="cyan", fontsize=10,
                    ha="left", va="bottom", transform=ax.transData,
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))
        out_path = _ensure_debug_dir() / Path(path).name
        overlay.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    finally:
        try:
            fig.set_dpi(_orig_dpi)
        except Exception:
            pass


def save_full_figure_component_overlay(
    fig: Figure,
    path: str,
    dpi: int,
    *,
    draw_final_bboxes: Optional[List[Tuple[int, Bbox]]] = None,
    red_alpha_fill: float = 0.15,
    red_alpha_edge: float = 0.5,
) -> None:
    """Overlay per-artist red rectangles and optional final cyan bboxes."""
    _orig_dpi = fig.dpi
    try:
        fig.set_dpi(dpi)
        canvas = FigureCanvas(fig); canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        overlay = Figure(figsize=fig.get_size_inches(), dpi=fig.dpi); _ = FigureCanvas(overlay)
        try:
            overlay.patch.set_facecolor(fig.get_facecolor())
        except Exception:
            pass
        ax = overlay.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.imshow(img, origin="upper", extent=[0, w, h, 0])

        renderer = canvas.get_renderer()
        for ax_src in fig.axes:
            boxes = _gather_axes_extents(ax_src, renderer, include_legend=True, include_tick_marks=False)
            for bb in boxes:
                x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
                y0_img, y1_img = h - y0, h - y1
                rect = mpatches.Rectangle((x0, y1_img), x1 - x0, y0_img - y1_img,
                                          facecolor=(1, 0, 0, red_alpha_fill),
                                          edgecolor=(1, 0, 0, red_alpha_edge), linewidth=0.8,
                                          transform=ax.transData)
                ax.add_patch(rect)

        if draw_final_bboxes:
            for idx, bb in draw_final_bboxes:
                x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
                y0_img, y1_img = h - y0, h - y1
                rect = mpatches.Rectangle((x0, y1_img), x1 - x0, y0_img - y1_img,
                                          fill=False, edgecolor="cyan", linewidth=2.0, transform=ax.transData)
                ax.add_patch(rect)
                ax.text(x0 + 4, y1_img + 14, f"{idx:02d}", color="cyan", fontsize=10,
                        ha="left", va="bottom", transform=ax.transData,
                        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))

        out_path = _ensure_debug_dir() / Path(path).name
        overlay.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    finally:
        try:
            fig.set_dpi(_orig_dpi)
        except Exception:
            pass


if __name__ == "__main__":
    # Design a multi-panel figure with diverse elements to test cropping
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, MaxNLocator

    fig = plt.figure(figsize=(10, 6), dpi=DPI, facecolor="white")
    fig.suptitle("Global Figure Title — Complex Layout", fontsize=16)

    gs = fig.add_gridspec(2, 3, left=0.08, right=0.98, bottom=0.08, top=0.88, wspace=0.35, hspace=0.45)
    x = np.linspace(0, 2*np.pi, 400)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, np.sin(x), label="sin"); ax1.plot(x, np.cos(x), label="cos"); ax1.legend(loc="upper right")
    ax1.set_title("Top x-ticks"); ax1.xaxis.tick_top(); ax1.xaxis.set_label_position('top'); ax1.set_xlabel("angle (rad)")
    ax1.set_ylabel("value")

    ax2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0); y = np.sin(3*x) + 0.2*rng.standard_normal(len(x))
    ax2.scatter(x, y, s=8, alpha=0.6, label="pts"); ax2.legend(loc="lower left")
    ax2.set_title("Bottom x-ticks"); ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.xaxis.set_major_locator(MultipleLocator(np.pi/2))

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x, np.tan(x), lw=0.8)
    ax3.set_title("No ticks"); ax3.set_xticks([]); ax3.set_yticks([])
    ax3.text(0.02, 0.1, "axis-local note", transform=ax3.transAxes, fontsize=10, color="tab:purple")

    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(x, np.sin(2*x), color="tab:green", label="sin 2x")
    ax4.set_title("Dense y-ticks"); ax4.set_xlabel("time"); ax4.set_ylabel("amp")
    ax4.yaxis.set_major_locator(MaxNLocator(7))
    ax4.grid(True, alpha=0.2)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(np.outer(np.sin(x), np.cos(x)), extent=[0, 10, 0, 10], origin="lower")
    ax5.set_title("Image panel")

    # Figure-level text not associated to any axes
    fig.text(0.5, 0.02, "Figure footer note (not axis-owned)", ha="center", va="bottom", fontsize=9, color="gray")

    debug_dir = _ensure_debug_dir()
    # Export buffers and save crops to disk for quick inspection
    crops = export_axes_to_buffers(fig, dpi=DPI, pad_inches=0.0, log_path=debug_dir / "export_bboxes.txt")
    for item in crops:
        with open(debug_dir / f"demo_ax_{item['index']:02d}.png", "wb") as fh:
            fh.write(item["png"]) 

    save_full_figure_tight(fig, "demo_full_tight.png", dpi=DPI)
    save_full_figure_overview(fig, "demo_overlay.png", dpi=DPI, axis_bboxes=[(c["index"], c["bbox_disp"]) for c in crops])
    save_full_figure_component_overlay(fig, "demo_components.png", dpi=DPI, draw_final_bboxes=[(c["index"], c["bbox_disp"]) for c in crops])
