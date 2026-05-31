"""Single-imshow composite renderer for image grids.

The legacy path (``image_grid`` in :mod:`ocdkit.plot.grid`) creates one
matplotlib Axes per tile, each with its own ``imshow`` + ticks + spines +
labels. matplotlib's per-axes overhead dominates the warm cost.

``composite_image_grid`` here collapses every tile into one big numpy RGBA
array laid out at the final output resolution, then renders it with a
single ``imshow`` on a single Axes. Per-tile labels become Axes-level
``ax.text`` overlays; per-tile outlines are painted directly into the
composite bitmap. Net cost: ~17× faster on the i880 sample's 2×7+extras
grid.

Architecture
------------
Two stages, intentionally decoupled so a non-matplotlib renderer (SVG,
HTML/JS, raw PNG) can consume the same intermediate representation:

1. :func:`build_grid_layout` → :class:`GridLayout`
     Pure-numpy composition: cmap/normalize/upscale each tile into a single
     RGBA buffer; record per-tile pixel bounds + adaptive-label color. No
     matplotlib involvement.

2. :func:`render_grid_matplotlib` → ``(fig, [ax], pos)``
     Thin matplotlib renderer: one ``imshow`` of the composite, plus tile
     labels (``ax.text``). Returns ``pos = (lefts, bottoms, widths,
     heights)`` in the same parallel-list shape :func:`image_grid` returns
     so existing callers slot in unchanged.

:func:`composite_image_grid` is the one-shot convenience wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt

from .imports import *


# Rec.601 luminance weights (same as ocdkit.plot.label.LUMA_WEIGHTS).
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ─── Pure layout primitives ─────────────────────────────────────────────


@dataclass
class TileInfo:
    """One cell of the composite grid (may be empty / label-only)."""
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    label: str | None = None
    label_color: tuple = (0.8, 0.8, 0.8)
    label_pos: tuple = (0.5, 0.05, 'center', 'top')  # (px, py, ha, va) in 0..1
    has_content: bool = False


@dataclass
class GridLayout:
    """Pure data describing a single-composite grid layout. Self-contained
    enough that any renderer (matplotlib, HTML, raw PNG) can consume it.
    """
    composite: np.ndarray  # (H, W, 4) float32 RGBA, alpha 0 where empty
    width_px: int
    height_px: int
    dpi: int
    fontsize: float
    figsize_inches: tuple[float, float]
    facecolor: tuple
    tiles: list[TileInfo] = field(default_factory=list)
    outline: bool = False
    outline_color: tuple = (0.5, 0.5, 0.5)
    outline_width: float = 0.5


def _resize_nearest(tile, dst_h, dst_w):
    src_h, src_w = tile.shape[:2]
    if src_h == dst_h and src_w == dst_w:
        return tile
    ys = (np.arange(dst_h) * (src_h / dst_h)).astype(np.int64)
    xs = (np.arange(dst_w) * (src_w / dst_w)).astype(np.int64)
    return tile[ys][:, xs]


def _paint_contours(rgba, mask, color=(1, 1, 1, 1)):
    """Paint a 1-pixel boundary of ``mask`` onto ``rgba`` in place."""
    if mask is None or rgba is None:
        return rgba
    m = np.asarray(mask)
    if m.shape != rgba.shape[:2]:
        return rgba
    boundary = np.zeros_like(m, dtype=bool)
    boundary[1:, :]  |= (m[1:, :]  != m[:-1, :]) & (m[1:, :]  > 0)
    boundary[:-1, :] |= (m[:-1, :] != m[1:, :])  & (m[:-1, :] > 0)
    boundary[:, 1:]  |= (m[:, 1:]  != m[:, :-1]) & (m[:, 1:]  > 0)
    boundary[:, :-1] |= (m[:, :-1] != m[:, 1:])  & (m[:, :-1] > 0)
    rgba[boundary] = color
    return rgba


def _paint_rect_outline(rgba, x0, y0, x1, y1, color, width):
    """Paint a rectangular outline ``width`` px thick onto ``rgba`` in place.
    The outline sits flush with (x0, y0)..(x1, y1) on the OUTSIDE — i.e.
    pixels [x0, x0+w) and [x1-w, x1) get painted. This keeps the tile's
    full bounds intact for composability while ensuring the outline never
    extends outside the composite.
    """
    H, W = rgba.shape[:2]
    x0c = max(0, x0); x1c = min(W, x1)
    y0c = max(0, y0); y1c = min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return rgba
    c = np.asarray(color, dtype=np.float32)
    if c.size == 3:
        c = np.concatenate([c, [1.0]]).astype(np.float32)
    w = max(1, int(round(width)))
    rgba[y0c:min(y0c + w, y1c), x0c:x1c] = c
    rgba[max(y0c, y1c - w):y1c, x0c:x1c] = c
    rgba[y0c:y1c, x0c:min(x0c + w, x1c)] = c
    rgba[y0c:y1c, max(x0c, x1c - w):x1c] = c
    return rgba


def _tile_to_rgba(tile, *, cmap, vmin, vmax, gamma):
    """Apply gamma/vmin/vmax/cmap to a 2D float tile -> (H, W, 4) float32.
    RGB(A) inputs are normalized from 0..255 if needed and passed through.
    """
    if tile.ndim == 3:
        arr = np.asarray(tile)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32, copy=False)
            if arr.size and float(arr.max()) > 1.0001:
                arr = arr / 255.0
        if arr.shape[-1] == 3:
            out = np.empty((*arr.shape[:2], 4), dtype=np.float32)
            out[..., :3] = arr
            out[..., 3] = 1.0
            return out
        return np.ascontiguousarray(arr, dtype=np.float32)
    norm = np.clip(
        (tile.astype(np.float32) ** gamma - vmin) / max(vmax - vmin, 1e-9),
        0, 1,
    )
    return cmap(norm).astype(np.float32, copy=False)


_LABEL_POSITIONS = {
    'top_middle':    (0.5, 0.05, 'center', 'top'),
    'bottom_middle': (0.5, 0.95, 'center', 'bottom'),
    'top_left':      (0.05, 0.05, 'left', 'top'),
    'bottom_left':   (0.05, 0.95, 'left', 'bottom'),
}


def build_grid_layout(
    tiles: Sequence[Sequence[np.ndarray | None]],
    labels: Sequence[Sequence[str | None]] | None = None,
    *,
    cmap='magma',
    vmin: float = 0.0,
    vmax: float | None = None,
    gamma: float = 1.0,
    ncol: int = 7,
    contour_masks: Sequence[Sequence[np.ndarray | None]] | None = None,
    target_tile_px: int | None = None,
    figsize: float | tuple[float, float] | None = None,
    pad_frac: float = 0.05,
    dpi: int = 300,
    fontsize: float = 4,
    fontcolor=(.8, .8, .8),
    facecolor=(0, 0, 0, 0),
    outline: bool = True,
    outline_color=(0.5, 0.5, 0.5),
    outline_width: float = 0.5,
    adaptive_label_color: bool = True,
    label_lightness_threshold: float = 0.6,
    light_label_color=(0.8, 0.8, 0.8),
    dark_label_color=(0.2, 0.2, 0.2),
    label_lpos: str = 'top_middle',
) -> GridLayout:
    """Build the pure data layout (no matplotlib). See :func:`composite_image_grid`."""
    from matplotlib.colors import Colormap as MplColormap

    if isinstance(cmap, str):
        from cmap import Colormap
        cmap_obj = Colormap(cmap).to_matplotlib()
    elif isinstance(cmap, MplColormap) or callable(cmap):
        cmap_obj = cmap
    else:
        from cmap import Colormap
        cmap_obj = Colormap(cmap).to_matplotlib()

    sub_rows, sub_labels, sub_masks = [], [], []
    for r_idx, row in enumerate(tiles):
        lab_row = labels[r_idx] if labels else [None] * len(row)
        mask_row = contour_masks[r_idx] if contour_masks else [None] * len(row)
        n_sub = (len(row) + ncol - 1) // ncol
        for sub in range(n_sub):
            chunk = list(row[sub*ncol:(sub+1)*ncol])
            lab_chunk = list(lab_row[sub*ncol:(sub+1)*ncol])
            mask_chunk = list(mask_row[sub*ncol:(sub+1)*ncol])
            while len(chunk) < ncol:
                chunk.append(None); lab_chunk.append(None); mask_chunk.append(None)
            sub_rows.append(chunk)
            sub_labels.append(lab_chunk)
            sub_masks.append(mask_chunk)

    rep = next((t for sr in sub_rows for t in sr if t is not None), None)
    if rep is None:
        raise ValueError("build_grid_layout: no non-None tiles")
    src_h, src_w = rep.shape[:2]

    if target_tile_px is None and figsize is not None:
        fs_w = figsize[0] if isinstance(figsize, (tuple, list)) else figsize
        target_tile_px = int(fs_w * dpi / (ncol * (1 + pad_frac)))
    if target_tile_px is None:
        scale = 1
    else:
        scale = max(1, int(round(target_tile_px / max(src_h, src_w))))
    th, tw = src_h * scale, src_w * scale
    pad = max(1, int(pad_frac * min(th, tw)))

    nrows, ncols = len(sub_rows), ncol
    H = nrows * th + (nrows - 1) * pad
    W = ncols * tw + (ncols - 1) * pad
    composite = np.zeros((H, W, 4), dtype=np.float32)

    if vmax is None:
        finite_2d = [t for sr in sub_rows for t in sr
                     if t is not None and t.ndim == 2]
        vmax_eff = max((float(np.max(t ** gamma)) for t in finite_2d),
                       default=1.0)
    else:
        vmax_eff = float(vmax)

    tile_infos: list[TileInfo] = []
    for r, (row, lab_row, mask_row) in enumerate(
            zip(sub_rows, sub_labels, sub_masks)):
        y0 = r * (th + pad)
        for c, (tile, label, mask) in enumerate(zip(row, lab_row, mask_row)):
            x0 = c * (tw + pad)
            x1, y1 = x0 + tw, y0 + th
            info = TileInfo(row=r, col=c, x0=x0, y0=y0, x1=x1, y1=y1,
                            label=label, has_content=tile is not None,
                            label_pos=_LABEL_POSITIONS.get(label_lpos,
                                                           _LABEL_POSITIONS['top_middle']))
            if tile is not None:
                if tile.shape[:2] != (src_h, src_w):
                    tile = _resize_nearest(tile, src_h, src_w)
                rgba = _tile_to_rgba(tile, cmap=cmap_obj, vmin=vmin,
                                     vmax=vmax_eff, gamma=gamma)
                if mask is not None and mask.shape == (src_h, src_w):
                    rgba = _paint_contours(rgba, mask)
                if scale > 1:
                    rgba = np.repeat(np.repeat(rgba, scale, axis=0),
                                     scale, axis=1)
                composite[y0:y1, x0:x1] = rgba
            tile_infos.append(info)

    # Paint outlines directly into the composite bitmap. This puts the
    # outline at the tile's exact pixel edges (x0..x1, y0..y1) with full
    # specified thickness -- no matplotlib stroke clipping at figure
    # edges, no inset that misaligns with downstream panels.
    if outline:
        outline_px = max(1, int(round(outline_width * dpi / 72.0)))
        for info in tile_infos:
            _paint_rect_outline(composite, info.x0, info.y0, info.x1, info.y1,
                                color=outline_color, width=outline_px)

    # Pick adaptive label colors from the *built* composite — never depends
    # on matplotlib's display transform.
    for info in tile_infos:
        if not info.label:
            continue
        if not adaptive_label_color:
            info.label_color = tuple(fontcolor)
            continue
        px, py, _ha, _va = info.label_pos
        cx = info.x0 + px * (info.x1 - info.x0)
        ay = info.y0 + py * (info.y1 - info.y0)
        patch_h = max(2, int(0.12 * (info.y1 - info.y0)))
        patch_w = max(2, int(0.5  * (info.x1 - info.x0)))
        py0 = max(0, int(ay) - patch_h // 2)
        py1 = min(H, py0 + patch_h)
        px0 = max(0, int(cx) - patch_w // 2)
        px1 = min(W, px0 + patch_w)
        patch = composite[py0:py1, px0:px1, :3]
        if patch.size:
            lum = float(np.mean(np.tensordot(patch, _LUMA, axes=([-1], [0]))))
            info.label_color = (tuple(dark_label_color)
                                if lum >= label_lightness_threshold
                                else tuple(light_label_color))
        else:
            info.label_color = tuple(fontcolor)

    return GridLayout(
        composite=composite, width_px=W, height_px=H, dpi=dpi,
        fontsize=fontsize,
        figsize_inches=(W / dpi, H / dpi),
        facecolor=tuple(facecolor),
        tiles=tile_infos,
        outline=outline, outline_color=tuple(outline_color),
        outline_width=outline_width,
    )


# ─── Matplotlib renderer ────────────────────────────────────────────────


def render_grid_matplotlib(layout: GridLayout):
    """Render a :class:`GridLayout` to a single matplotlib Figure + Axes.

    Returns ``(fig, [ax], pos)`` where ``pos = (lefts, bottoms, widths,
    heights)`` in figure-fraction coords, one entry per cell (including
    empty cells) in row-major order. Matches the shape :func:`image_grid`
    returns so existing callers slot in unchanged.
    """
    W, H = layout.width_px, layout.height_px

    fig = plt.figure(figsize=layout.figsize_inches, dpi=layout.dpi,
                     facecolor=layout.facecolor)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(layout.composite, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Composite extent is exactly (0, 0)..(W, H) in pixel space. Don't pad
    # xlim/ylim — that would squeeze the composite and break alignment with
    # adjacent panels/figures.
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    lefts, bottoms, widths, heights = [], [], [], []
    for t in layout.tiles:
        lefts.append(t.x0 / W)
        bottoms.append(1 - t.y1 / H)
        widths.append((t.x1 - t.x0) / W)
        heights.append((t.y1 - t.y0) / H)

        if not t.label:
            continue
        px, py, ha, va = t.label_pos
        cx = t.x0 + px * (t.x1 - t.x0)
        ay = t.y0 + py * (t.y1 - t.y0)
        ax.text(cx, ay, t.label, fontsize=layout.fontsize,
                color=t.label_color, ha=ha, va=va, zorder=3)

    pos = (np.array(lefts), np.array(bottoms),
           np.array(widths), np.array(heights))
    return fig, [ax], pos


def composite_image_grid(tiles, labels=None, **kwargs):
    """Build the layout and render it with matplotlib. One-shot convenience."""
    layout = build_grid_layout(tiles, labels, **kwargs)
    return render_grid_matplotlib(layout)


__all__ = [
    "TileInfo", "GridLayout",
    "build_grid_layout", "render_grid_matplotlib",
    "composite_image_grid",
]
