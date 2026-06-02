"""Unified image-grid plotter — SVG (HDR-capable) by default, matplotlib
on opt-in.

The default ``backend='svg'`` produces an :class:`ocdkit.io.SvgFigure`
with one composite JXL raster + vector outlines/labels. HDR is
automatic when inputs are float linear-light Display-P3 arrays (or
Scene-like objects carrying ``_rgb_linear_p3``); ``uint8`` inputs fall
back to SDR ``jxl-p3`` (wider gamut than sRGB, same bit depth).

``backend='matplotlib'`` delegates to
:func:`image_grid_matplotlib` for the legacy per-axes layout — SDR
sRGB output, ``matplotlib.figure.Figure`` return type.

Input shapes accepted:

* Flat list of items: ``[item_0, item_1, ...]`` with ``ncol=`` kwarg.
* Nested list ``[[item_0, item_1, ...], [item_5, ...]]`` — ``ncol`` is
  inferred from the first row's length.

Each item is one of:

* ``numpy.ndarray`` — a per-cell raster (any aspect ratio).
* A Scene-like object with ``_rgb_linear_p3`` and/or ``.rgb``
  attributes; ``_rgb_linear_p3`` is preferred for HDR.

``vmin`` / ``vmax`` are accepted for matplotlib-API compatibility but
ignored on the SVG backend — the absolute-SDR-reference PQ encoder
uses linear ``1.0`` ⇒ ``sdr_white_nits`` regardless.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .grid import image_grid_matplotlib
from .figure import split_list
from .svg import SVG
from ..io.figure import SvgFigure


def image_grid(
    items,
    *,
    ncol: int | None = None,
    backend: str = 'svg',
    # SVG-backend layout
    figsize: float | None = None,         # inches; canvas width = figsize * 96
    target_tile_px: int | None = None,    # uniform per-cell PIXEL height; None ⇒ ocdkit.plot.setup default (512)
    gap_px: int = 6,                       # uniform gap between cells
    margin_px: int = 4,                    # outer margin around the grid
    dx: int = 1,                           # raster stride (downsample)
    plot_labels=None,
    fontsize: float = 8,
    # Default ``'currentColor'`` makes labels inherit the host's CSS
    # ``color`` — JupyterLab swaps that with the active theme, so labels
    # render light on dark backgrounds and dark on light without re-encoding
    # the SVG. Pass an explicit color string (``'lightgray'``, ``'#808080'``)
    # to pin a specific shade regardless of theme.
    fontcolor: str = 'currentColor',
    lpos: str = 'top_middle',
    facecolor: str | None = None,
    outline: bool = False,
    # Same theme-adaptive default as ``fontcolor`` — outlines follow the
    # host text color via CSS ``currentColor`` unless explicitly overridden.
    outline_color: str = 'currentColor',
    outline_width: float = 0.5,
    raster_format: str | None = None,     # None → autodetect from dtype
    sdr_white_nits: float = 1600.0,
    # Linked pan/zoom: when True, all cells share one viewport state.
    # Drag/wheel on any cell pans/zooms every cell in lockstep — useful
    # for spectral key-slice grids where every panel shows the same FOV
    # at a different channel. Requires every cell's raster to have the
    # same source dimensions; raises ``ValueError`` if shapes diverge.
    # ``roi`` (optional) seeds the initial viewport as
    # ``(y, x, h, w)`` in source-pixel coords; default is the full
    # image. Mirrors the popup viewer's controller (cursor-anchored
    # wheel zoom + drag pan + rAF tween).
    link_axes: bool = False,
    roi: tuple | None = None,
    # When True (only meaningful with ``link_axes=True``), draw a red
    # dashed outline around each cell's clickable/draggable bbox. Use
    # to debug aspect-lock issues — anything inside the red rect should
    # capture pointer events regardless of where the image is rendered.
    link_axes_debug: bool = False,
    # Per-cell vector overlay: list of (N, 2) float32 polygons in
    # source-pixel coords (the same coord system as the cell viewBox).
    # Each polygon is drawn as a stroked, non-filled <polygon> inside
    # the cell's nested SVG so it pans/zooms with the image at zero
    # rasterization cost (vector forever). Use to overlay seg outlines
    # / contours / any structured annotation. Only meaningful with
    # ``link_axes=True`` AND uniform-shape cells; same polygons drawn
    # on every cell.
    seg_polygons: list | None = None,
    seg_stroke: str = '#ffffff',
    seg_stroke_opacity: float = 0.5,
    seg_stroke_width: float = 0.5,
    auto_upgrade: bool = False,           # eagerly stream hi-res into inline
    # Popup zoom viewer choice. ``None`` (default) uses the CSS-img
    # viewer — a plain <img> + CSS matrix3d transform that keeps the
    # raster on the browser's BitmapImage → CALayer (Safari) / Skia HDR
    # (Chrome) compositor pipeline, so P3-PQ JXLs render at absolute
    # nits with no RGBA8 clamp. Pass ``'webgl'`` to opt into the
    # worker-thread WebGL2 viewer instead: faster pan/zoom on big SDR
    # grids (texture LRU lets arrow-nav between tiles skip refetch +
    # decode + upload) but ``texImage2D(... RGBA, UNSIGNED_BYTE, bmp)``
    # clamps to 8-bit at upload — never use it on HDR content.
    popup_viewer: 'str | None' = None,
    # quietly-accepted matplotlib-API kwargs (no-op on SVG)
    vmin=None, vmax=None, dpi=None,
    # mpl-backend passthrough
    **mpl_kwargs,
):
    """Image grid with auto-selected SVG (default) or matplotlib backend."""

    # ── normalize input shape (flat list vs nested list) ────────────────
    items = list(items)
    if items and isinstance(items[0], (list, tuple)):
        inferred_ncol = max(len(row) for row in items)
        items = [x for row in items for x in row]
        if ncol is None:
            ncol = inferred_ncol
    if ncol is None:
        ncol = 5

    # plot_labels can mirror the nested shape; flatten the same way.
    if plot_labels is not None:
        plot_labels = list(plot_labels)
        if plot_labels and isinstance(plot_labels[0], (list, tuple)):
            plot_labels = [x for row in plot_labels for x in row]

    if backend == 'matplotlib':
        # Re-split for the legacy nested-list API.
        images = split_list(_to_array_only(items, dx=dx), ncol)
        labels = (split_list(list(plot_labels), ncol)
                  if plot_labels is not None else None)
        # ``currentColor`` is a CSS keyword the SVG backend resolves at
        # render time; matplotlib has no equivalent. Translate the SVG
        # default back to neutral gray so matplotlib renders cleanly.
        mpl_fontcolor = '#808080' if fontcolor == 'currentColor' else fontcolor
        return image_grid_matplotlib(
            images,
            plot_labels=labels,
            figsize=figsize if figsize is not None else ncol,
            fontsize=fontsize, fontcolor=mpl_fontcolor, lpos=lpos,
            dpi=dpi if dpi is not None else 300,
            **mpl_kwargs,
        )

    if backend != 'svg':
        raise ValueError(f"backend must be 'svg' or 'matplotlib', got {backend!r}")

    if not items:
        raise ValueError("image_grid: items is empty")

    if target_tile_px is None:
        from . import _config
        target_tile_px = _config.target_tile_px

    # Resolve each item to a per-cell array + pick a raster_format if
    # not explicitly set. Mixed-dtype grids stay on the float HDR path
    # so SDR cells survive next to HDR ones (uint8 cells are promoted
    # to linear-P3 float by the resolver below).
    arrays, auto_fmt = _resolve_items(items, dx=dx,
                                       target_px=int(target_tile_px))
    if raster_format is None:
        raster_format = auto_fmt

    # ── source-native cell dimensions ──────────────────────────────────
    # Each cell's SVG bbox is the source's native pixel dims (NOT the
    # thumb's, NOT target_tile_px). For homogeneous grids this makes
    # the browser's scaling of thumb→cell and hires→cell both an exact
    # integer multiple — zero fractional sampling offset between the
    # inline thumb and the hires-on-zoom. Eliminates the "thumb shifts
    # when hi-res lands" artifact the centred-NN resize used to mask.
    src_dims = [_native_dims(it, arr) for it, arr in zip(items, arrays)]
    cell_h_pxs = [h for h, _ in src_dims]
    cell_w_pxs = [w for _, w in src_dims]

    # We do NOT resize the array to match the cell. The SVG <image>
    # element gives the browser the full bbox (cell dims) and the small
    # thumb raster (250² etc); the browser nearest-neighbor-upscales by
    # an integer factor. Saves the Python upscale cost completely.

    nrow = math.ceil(len(items) / ncol)
    # Row height = tallest source in that row (for non-uniform grids).
    row_heights = [
        max(cell_h_pxs[r * ncol:(r + 1) * ncol], default=0)
        for r in range(nrow)
    ]
    # Display sizing decision: target_tile_px is the on-screen height
    # we want a typical (max-height) cell to occupy in CSS pixels at
    # the SVG's default rendering size. Everything else scales off
    # that. ``figsize`` overrides by stating an absolute CSS width.
    src_cell_h = max(cell_h_pxs) if cell_h_pxs else 1
    effective_dpi = float(dpi if dpi is not None else 96)
    # source-pixel units per CSS pixel (= the inverse display scale).
    vb_per_css = src_cell_h / float(target_tile_px)
    gap_vb = gap_px * vb_per_css
    margin_vb = margin_px * vb_per_css
    # Recompute row widths now that gap_vb is known.
    row_widths = [
        sum(cell_w_pxs[r * ncol:(r + 1) * ncol]) +
        gap_vb * (min(ncol, len(items) - r * ncol) - 1)
        for r in range(nrow)
    ]
    canvas_w_vb = max(row_widths) + 2 * margin_vb
    canvas_h_vb = sum(row_heights) + (nrow - 1) * gap_vb + 2 * margin_vb

    # Outer SVG CSS size: prefer figsize × dpi, else use target_tile_px
    # per cell as the implied CSS scale.
    if figsize is not None:
        svg_w_css = int(round(float(figsize) * effective_dpi))
        # Preserve aspect (viewBox does this, but pin the height too).
        svg_h_css = int(round(svg_w_css * canvas_h_vb / canvas_w_vb))
    else:
        svg_w_css = int(round(canvas_w_vb / vb_per_css))
        svg_h_css = int(round(canvas_h_vb / vb_per_css))

    # Font / outline are SPECIFIED as matplotlib-style points but RENDERED
    # in viewBox units. Points → CSS pixels via dpi/72; CSS px → viewBox
    # via vb_per_css.
    fontsize_uu = fontsize * effective_dpi / 72.0 * vb_per_css
    outline_width_uu = outline_width * effective_dpi / 72.0 * vb_per_css

    data_attrs = {"ncol": str(ncol)}
    if popup_viewer:
        # Mirrored 1:1 to ``data-popup-viewer`` on the root <svg>. The
        # figure.py JS reads ``tile.closest('svg').dataset.popupViewer``
        # in the open-zoom handler to pick a viewer. Default (unset) is
        # the CSS-img viewer, which preserves HDR PQ; pass
        # ``popup_viewer='webgl'`` to opt into the worker-thread WebGL2
        # viewer (faster pan/zoom on big SDR grids, SDR-only).
        data_attrs["popup-viewer"] = str(popup_viewer)

    # Linked-axes mode: every cell must share the same raster shape.
    # We emit a nested ``<svg viewBox>`` per cell whose viewBox is the
    # current ROI; the figure.py JS controller updates every linked
    # cell's viewBox in lockstep on pointer/wheel input.
    #
    # Use ``_native_dims`` (= the SOURCE pixel size) rather than the
    # resolved array's shape: ``_resolve_items`` may have decoded at a
    # smaller downsample ratio for the inline thumb, but the ``roi``
    # parameter is in source-pixel coordinates and the <image> bbox
    # must declare the source dims for the viewBox math to line up.
    # The encoded thumb under the hood may still be sub-resolution;
    # the browser nearest-neighbour-upscales (image-rendering:pixelated).
    linked_raster_shape = None
    if link_axes:
        shapes = [_native_dims(it, a) for it, a in zip(items, arrays)]
        if len(set(shapes)) != 1:
            raise ValueError(
                "image_grid: link_axes=True requires every cell raster to "
                f"share the same source (H, W); got {sorted(set(shapes))}"
            )
        linked_raster_shape = shapes[0]   # (H, W) source-pixel dims
        ras_h, ras_w = linked_raster_shape
        if roi is None:
            # Default ROI = full image.
            roi_y, roi_x, roi_h, roi_w = 0.0, 0.0, float(ras_h), float(ras_w)
        else:
            if len(roi) != 4:
                raise ValueError(
                    f"image_grid: roi must be (y, x, h, w); got {roi!r}")
            roi_y, roi_x, roi_h, roi_w = (float(v) for v in roi)
        data_attrs["link-axes"] = "1"
        data_attrs["link-raster-h"] = str(int(ras_h))
        data_attrs["link-raster-w"] = str(int(ras_w))
        # Stamp the initial viewport as ``data-link-roi="x y w h"`` so
        # the JS controller picks up the same values on first paint.
        data_attrs["link-roi"] = (
            f"{roi_x:.4f} {roi_y:.4f} {roi_w:.4f} {roi_h:.4f}")
    svg = SVG(width=svg_w_css, height=svg_h_css,
              viewBox=(0, 0, canvas_w_vb, canvas_h_vb),
              background=facecolor,
              data_attrs=data_attrs)
    encoder_kwargs = (
        {'sdr_white_nits': sdr_white_nits}
        if raster_format in ('jxl-hdr-pq', 'uhdr') else {}
    )

    # Hi-res streaming: register each original ``item`` with the
    # per-kernel figure server so SvgFigure's zoom overlay can fetch
    # full-resolution bytes on demand. ``resolve_source`` prefers
    # ``rgb_path`` (zero re-encode), then ``_rgb_linear_p3`` (linear-P3
    # HDR), then ``.rgb`` (SDR), then raw ndarrays. Returns ``None`` if
    # there's no useful source — that tile's zoom falls back to the
    # inline thumbnail.
    try:
        from ..io.figure_server import register, resolve_source
        hires_urls = [
            (register(src) if (src := resolve_source(it)) else None)
            for it in items
        ]
    except Exception:
        hires_urls = [None] * len(items)

    # Build per-tile data URLs in parallel, caching the encoded bytes on
    # each scene-like item so subsequent renders (same scenes, same
    # format) skip the encode entirely. Each cached URL is paired with
    # a fingerprint of the source state (rgb_path mtime + id of any
    # in-memory RGB caches) so a regenerated ``scene._rgb`` or a
    # touched ``rgb_path`` invalidates the URL and forces a re-encode.
    cache_attr = f'_thumb_url_{raster_format}'
    fp_attr = f'_thumb_url_{raster_format}_fp'

    def _get_or_build_url(idx):
        it = items[idx]
        if not isinstance(it, np.ndarray):
            fp_now = _source_fingerprint(it)
            if getattr(it, fp_attr, None) == fp_now:
                cached = getattr(it, cache_attr, None)
                if cached is not None:
                    return cached
        url = _encode_thumb_url(arrays[idx], raster_format, sdr_white_nits)
        if not isinstance(it, np.ndarray):
            try:
                setattr(it, cache_attr, url)
                setattr(it, fp_attr, fp_now)
            except Exception:
                pass
        return url

    if len(arrays) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(arrays), 16)) as ex:
            urls = list(ex.map(_get_or_build_url, range(len(arrays))))
    else:
        urls = [_get_or_build_url(0)]

    # Pre-compute each row's vertical offset (cumulative row heights +
    # gaps) since rows can vary in height for non-uniform grids.
    row_y_offsets = []
    y_cursor = margin_vb
    for r in range(nrow):
        row_y_offsets.append(y_cursor)
        y_cursor += row_heights[r] + gap_vb

    # Pre-build the seg-polygon SVG overlay string. Pasted verbatim
    # into every linked cell so the same vector outlines pan/zoom with
    # the image. Each polygon is a closed loop in source-pixel coords
    # (matches the cell viewBox), drawn as a stroked SVG <polygon>
    # with pointer-events="none" so the hit rect still owns gestures.
    _link_axes_seg_overlay = ''
    if link_axes and seg_polygons:
        # Use a single <g> with shared stroke attrs so the SVG payload
        # stays compact even with thousands of cells. stroke-width is
        # in inner-viewBox units (= source pixels); since the cell
        # scales viewBox-to-CSS by ~cell_css / vb_w, a stroke-width of
        # 0.5 source-px renders thinner as the user zooms IN (the
        # opposite of what most users want). We compensate by using
        # vector-effect="non-scaling-stroke" so the stroke stays at
        # the same CSS pixel width regardless of viewBox zoom.
        parts = [
            f'<g class="ocd-seg-outlines" pointer-events="none" '
            f'fill="none" stroke="{seg_stroke}" '
            f'stroke-opacity="{float(seg_stroke_opacity):.3f}" '
            f'stroke-width="{float(seg_stroke_width):.3f}" '
            f'vector-effect="non-scaling-stroke">'
        ]
        for poly in seg_polygons:
            # poly: (N, 2) float32 in (x, y) order
            if poly is None or len(poly) < 3:
                continue
            pts = ' '.join(f'{x:.2f},{y:.2f}' for x, y in poly)
            parts.append(f'<polygon points="{pts}"/>')
        parts.append('</g>')
        _link_axes_seg_overlay = ''.join(parts)

    # Per-tile emission: one ``<image>`` per cell, wrapped in
    # ``<g class="fig-tile" data-bbox="x y w h"
    # data-hires-href="...">`` so SvgFigure's interactive shell can
    # attach hover-scale + click-to-zoom + lazy hi-res fetch. Coords
    # are in viewBox units (= source-pixel coords) — the browser
    # scales the whole SVG element to its CSS size for display, but
    # the relative positions of cells and the integer scaling of
    # raster→cell-bbox stay fixed.
    for i, arr in enumerate(arrays):
        r, c = divmod(i, ncol)
        x = margin_vb + sum(cell_w_pxs[r * ncol:r * ncol + c]) + gap_vb * c
        y = row_y_offsets[r]
        w, h = cell_w_pxs[i], cell_h_pxs[i]
        hires_attr = (f' data-hires-href="{hires_urls[i]}"'
                      if hires_urls[i] else '')
        upgrade_attr = (' data-auto-upgrade="1"'
                        if auto_upgrade and hires_urls[i] else '')
        # Persist the original thumb URL separately from the inline
        # <image href>, which may get swapped to the hi-res URL by the
        # hover prefetch. The popup viewer reads data-thumb-href so it
        # can always show the cheap thumb first (instant feedback),
        # then chain the hi-res load for a visible upgrade.
        thumb_attr = f' data-thumb-href="{urls[i]}"'
        svg.add(
            f'<g class="fig-tile" data-bbox="{x:.2f} {y:.2f} {w:.2f} {h:.2f}"'
            f'{thumb_attr}{hires_attr}{upgrade_attr}>'
        )
        if link_axes:
            seg_overlay_inner = _link_axes_seg_overlay or ''
            # Nested <svg viewBox> turns each cell into its own clipped
            # viewport. The viewBox is the current ROI in source-pixel
            # coords; the JS controller mutates it on pointer/wheel
            # input to pan/zoom synchronously across all linked cells.
            #
            # ``preserveAspectRatio="xMidYMid slice"`` (NOT "meet") ensures
            # the image always fills the cell — any tiny aspect drift
            # from the JS aspect-lock crops a hair instead of leaving
            # letterbox bars.
            #
            # Hi-res image source: prefer the figure_server HTTP URL
            # (full source resolution, browser caches, no inline base64)
            # over the inline downsampled thumb. The downsampled thumb
            # was sharp enough for image_grid's STATIC display, but in
            # link_axes mode the cell IS the popup — high zoom would
            # show pixelated NN upscale of the thumb. The hires URL is
            # the original encoded raster.
            #
            # Pointer hit-test: a transparent <rect> at the FRONT of the
            # nested SVG fires pointer events for the cell's full
            # viewport. The image gets pointer-events="none" so events
            # fall through to the rect — robust across Safari, Chrome,
            # Firefox (pointer-events: bounding-box has spotty support).
            ras_h, ras_w = linked_raster_shape
            img_href = hires_urls[i] if hires_urls[i] else urls[i]
            svg.add(
                f'<svg class="ocd-linked-cell" '
                f'x="{x:.2f}" y="{y:.2f}" '
                f'width="{w:.2f}" height="{h:.2f}" '
                f'viewBox="{roi_x:.4f} {roi_y:.4f} {roi_w:.4f} {roi_h:.4f}" '
                f'preserveAspectRatio="xMidYMid slice" '
                f'overflow="hidden">'
                f'<image x="0" y="0" '
                f'width="{ras_w}" height="{ras_h}" '
                f'href="{img_href}" preserveAspectRatio="none" '
                f'image-rendering="pixelated" '
                f'pointer-events="none"/>'
                f'{seg_overlay_inner}'
                f'</svg>'
                # Hit rect lives OUTSIDE the nested SVG — in the OUTER
                # viewBox coord system — so its bbox stays the cell's
                # bbox regardless of the inner viewBox's zoom/pan state.
                # Putting it inside the nested SVG would mean width="100%"
                # resolves to the inner viewBox's width (here, the ROI
                # width 400 px in source coords), and the rect would
                # cover only a fraction of the visible cell — pan would
                # work in that fraction only. ``data-cell-index`` links
                # this rect to its corresponding ocd-linked-cell SVG
                # (same DOM order, 1:1 correspondence) so the JS
                # controller can route events to the right cell.
                f'<rect class="ocd-linked-cell-hit" '
                f'data-cell-index="{i}" '
                f'x="{x:.2f}" y="{y:.2f}" '
                f'width="{w:.2f}" height="{h:.2f}" '
                f'fill="transparent" pointer-events="all"/>'
            )
            if link_axes_debug:
                # Red dashed outline at the cell's actual bbox in outer-
                # viewBox units. Anything inside this rect is clickable
                # and pannable; the image painted underneath may be
                # smaller (at high zoom) or letterboxed (at large zoom-
                # out). Width is one viewBox unit so it's visible at any
                # zoom of the outer SVG.
                svg.add(
                    f'<rect x="{x:.2f}" y="{y:.2f}" '
                    f'width="{w:.2f}" height="{h:.2f}" '
                    f'fill="none" stroke="#e63946" '
                    f'stroke-width="{outline_width_uu * 2:.2f}" '
                    f'stroke-dasharray="{outline_width_uu * 6:.2f} '
                    f'{outline_width_uu * 4:.2f}" '
                    f'pointer-events="none"/>'
                )
        else:
            # No half-pixel compensation: tried both ±0.5 and both moved
            # the image visibly on swap. Empirical browser behaviour
            # doesn't match the DC-frame-centroid math; without a way to
            # test the actual sampling convention, every non-zero shift
            # trades one artifact for another. Block edges aligned with
            # hires pixel boundaries is the cleanest swap behaviour.
            svg.add(
                f'<image x="{x:.2f}" y="{y:.2f}" '
                f'width="{w:.2f}" height="{h:.2f}" '
                f'href="{urls[i]}" preserveAspectRatio="none" '
                f'image-rendering="pixelated"/>'
            )
        if outline:
            svg.rect(x, y, w, h,
                     fill='none', stroke=outline_color,
                     stroke_width=outline_width_uu)
        label = (plot_labels[i]
                 if plot_labels and i < len(plot_labels) else None)
        if label:
            tx, ty, anchor, baseline = _label_position(x, y, w, h, lpos)
            svg.text(tx, ty, str(label),
                     fill=fontcolor, size=fontsize_uu,
                     anchor=anchor, baseline=baseline,
                     class_='fig-figure-text')
        svg.add('</g>')

    return SvgFigure(svg.finalize())


# ─── helpers ────────────────────────────────────────────────────────


def _native_dims(item, arr):
    """Best estimate of an item's source-resolution dimensions in (h, w).

    Used as the SVG <image> bbox so the browser can do an integer-
    multiple scale of the embedded thumb (or 1:1 of the hires) into
    that bbox — no fractional sampling offset.

    Order of preference:
      1. ``item.shape`` for caller-supplied :class:`Source` items
         (``NpySliceSource``, ``CziSliceSource``, etc.)
      2. ``item._rgb_jxl_size`` (cached header peek of the source JXL)
      3. ``item.rgb_path`` peeked now (and stashed for next call)
      4. shape of the resolved array (for raw ndarray items)
    """
    from ..io.figure_server import Source
    if isinstance(item, Source):
        shape = getattr(item, 'shape', None)
        if shape is not None and len(shape) >= 2:
            return (int(shape[0]), int(shape[1]))
    if isinstance(item, np.ndarray):
        # Use the original ndarray's shape, not the resolved (possibly
        # downsampled) thumb's shape — the cell bbox must match the
        # source so the browser's thumb→bbox upscale is integer-exact
        # and the hires-on-zoom lands without sub-pixel shift.
        return item.shape[:2]
    size = getattr(item, "_rgb_jxl_size", None)
    if size is not None:
        return size
    rgb_path = getattr(item, "rgb_path", None)
    if rgb_path:
        from ..io.figure_server import _peek_jxl_size
        import os as _os
        if _os.path.exists(str(rgb_path)):
            size = _peek_jxl_size(rgb_path)
            if size is not None:
                try:
                    item._rgb_jxl_size = size
                except Exception:
                    pass
                return size
    return arr.shape[:2]


def _source_fingerprint(item):
    """Tuple that changes whenever an item's underlying RGB source
    changes — used to invalidate ``_thumb_url_*`` caches.

    Includes:
      * mtime of ``rgb_path`` if it exists (covers disk re-saves).
      * ``id()`` of any in-memory cached arrays (``_rgb``,
        ``_rgb_linear_p3``, ``_rgb_linear_p3_dsN``) — these get new
        identity when the user regenerates them via ``scene.make_rgb``
        or assigns directly, so a stale cached URL invalidates.
    """
    import os as _os
    fp = []
    rgb_path = getattr(item, "rgb_path", None)
    if rgb_path:
        try:
            fp.append(("mtime", _os.path.getmtime(str(rgb_path))))
        except OSError:
            pass
    for attr in ("_rgb", "_rgb_linear_p3",
                  "_rgb_linear_p3_ds2", "_rgb_linear_p3_ds4",
                  "_rgb_linear_p3_ds8"):
        v = getattr(item, attr, None)
        if v is not None:
            fp.append((attr, id(v)))
    return tuple(fp)


def _encode_thumb_url(arr, raster_format, sdr_white_nits):
    """Encode a per-tile array into a ``data:image/...;base64,...`` URL.

    Mirrors :meth:`SVG.image` for the ``jxl-hdr-pq`` / ``jxl-p3`` /
    ``uhdr`` paths but returns the URL string directly so callers
    (image_grid) can cache it on the scene object. The cached URL is
    reusable across re-renders of the same scenes with the same
    target/HDR settings, so warm renders skip both decode and encode
    entirely.
    """
    # Fast path: the array carries a pre-built UHDR JPEG thumbnail
    # (layer-IDCT subsample of the source, original gainmap metadata
    # preserved bit-exact). Used by ``_resolve_items`` whenever the
    # scene has ``_rgb_uhdr`` or a ``.jpg`` ``rgb_path`` — no float
    # roundtrip, no gain-map recompute.
    thumb_bytes = getattr(arr, '_uhdr_thumb_bytes', None)
    if thumb_bytes is not None:
        import base64
        return ("data:image/jpeg;base64,"
                + base64.b64encode(thumb_bytes).decode('ascii'))

    from .svg import (jxl_data_url, uhdr_data_url, _p3_linear_to_pq_uint16,
                      _srgb_to_display_p3_uint8,
                      _linear_p3_to_uint8_srgb_peaknorm,
                      _srgb_uint8_to_p3_linear)
    import opencodecs

    # ``effort=1, lossless=True`` (the ``jxl_data_url`` defaults).  Bench
    # on a 256² uint8 thumbnail: lossy ``distance=1.0`` → 40 ms / 30 kB,
    # lossless effort=1 → 2 ms / ~150 kB.  The ~120 kB inline payload
    # bump per tile is acceptable for the speed win (and matches the
    # ArraySource hi-res encoder, so the auto-upgrade swap is now a
    # bit-identical raster instead of a slight lossy→lossless shift).
    if raster_format == 'jxl-hdr-pq':
        arr_pq = _p3_linear_to_pq_uint16(arr, sdr_white_nits=sdr_white_nits)
        p3_pq = opencodecs.ColorSpec(
            primaries=11, transfer=16, white_point=1,
            rendering_intent=1, gamma=0.0,
        )
        return jxl_data_url(arr_pq, color=p3_pq, intensity_target=10000.0)
    if raster_format == 'jxl-p3':
        arr_u8 = arr if (hasattr(arr, 'dtype') and arr.dtype == np.uint8) \
            else (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        arr_p3 = _srgb_to_display_p3_uint8(arr_u8)
        return jxl_data_url(arr_p3, color='display-p3')
    if raster_format == 'uhdr':
        # Ultra-HDR JPEG: cross-browser HDR (Safari + Chrome composite,
        # Firefox sees the SDR base). For uint8 cells in a mixed grid,
        # round-trip through linear-P3 so the gain map still encodes
        # (otherwise the SDR base + HDR are identical and no gain rides
        # on top — the file is still valid, just SDR-equivalent).
        if hasattr(arr, 'dtype') and np.issubdtype(arr.dtype, np.floating):
            hdr = arr
        else:
            arr_u8 = arr if (hasattr(arr, 'dtype') and arr.dtype == np.uint8) \
                else (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            hdr = _srgb_uint8_to_p3_linear(arr_u8)
        # Prefer a caller-supplied deterministic SDR base (e.g. the
        # native non-lifted cmap from :func:`hdr_cmap.apply_hdr_cmap`)
        # — libuhdr's auto-tonemap of cmap data desaturates bright
        # stops. Peak-norm fallback for plain HDR scene images.
        sdr_u8 = getattr(arr, '_sdr_base_p3_u8', None)
        if sdr_u8 is None:
            sdr_u8 = _linear_p3_to_uint8_srgb_peaknorm(hdr)
        return uhdr_data_url(hdr, sdr_u8, sdr_white_nits=sdr_white_nits)
    # Generic JXL (no colour-space tag).
    return jxl_data_url(arr)


def _resolve_items(items, *, dx, target_px=None):
    """Return ``(arrays, raster_format)``. For scene-like inputs, prefer
    the on-disk HDR JXL via :func:`ocdkit.io.figure_server.resolve_linear_p3`
    so thumbnails preserve the same scene-referred HDR brightness as the
    full-resolution PathSource hi-res stream. Any HDR-bearing item flips
    the whole grid to ``jxl-hdr-pq`` so SDR cells survive next to HDR
    ones.

    ``target_px`` (the grid's ``target_tile_px``) lets the resolver pick
    a libjxl-native downsample ratio per item — the longest source side
    stays ≥ ``target_px`` after decode, then ``_resize_nearest`` does the
    final exact-fit shrink. Cuts per-thumb decode cost a lot for ≥4K
    images via the DC progressive path.
    """
    from concurrent.futures import ThreadPoolExecutor
    from ..io.figure_server import (
        resolve_linear_p3, resolve_uhdr_thumb_bytes, Source,
    )

    def _resolve_one(it):
        if isinstance(it, Source):
            # Source items (NpySliceSource, CziSliceSource, PathSource,
            # BytesSource, …) provide their bytes themselves via the
            # figure_server HTTP path — we do NOT need an in-memory
            # array thumb. Return a tiny placeholder so the rest of the
            # array-shaped layout pipeline doesn't crash; the cell's
            # actual <image href> in link_axes mode points at the
            # source's registered URL, bypassing this thumb entirely.
            shape = getattr(it, 'shape', (1, 1))
            h = int(shape[0]) if len(shape) >= 1 else 1
            w = int(shape[1]) if len(shape) >= 2 else 1
            return np.zeros((h, w, 3), dtype=np.float32)
        if isinstance(it, np.ndarray):
            arr = it
            # For raw ndarrays, downsample to ~target_px on the longest
            # side so the inline thumbnail stays small.  ``_native_dims``
            # still reads ``item.shape[:2]`` for the cell bbox, so the
            # browser's upscale from this thumb to the full bbox is an
            # exact integer multiple (zero sampling shift when hi-res
            # later replaces the thumb via auto-upgrade).
            # Scene-like inputs already downsample via libjxl's ds=8
            # decode in ``resolve_linear_p3`` — this branch keeps raw
            # ndarrays on the same footing.
            #
            # ``_resize_nearest`` does area-averaging (box filter) when
            # both axes scale by an integer factor — important for
            # high-frequency content where a plain ``arr[::s, ::s]``
            # stride would alias / moire before encoding. The browser
            # then nearest-neighbor upscales the small filtered thumb
            # to the cell bbox, integer-ratio, with no further moire.
            if target_px is not None:
                longest = max(arr.shape[0], arr.shape[1])
                if longest > target_px:
                    s = longest // target_px
                    if s > 1:
                        new_h = arr.shape[0] // s
                        new_w = arr.shape[1] // s
                        arr = _resize_nearest(arr, new_h, new_w)
        else:
            # Prefer the layer-IDCT subsample of the source UHDR (when
            # available): extract base + gain map, stride both layers
            # in lockstep, re-pack with the ORIGINAL gainmap metadata.
            # Preserves ``max_content_boost`` bit-exact so the thumb's
            # HDR brightness matches the on-disk file. Skips the float
            # roundtrip + gain-map recompute entirely.
            # Pick a downsample factor from the (legacy) target_px:
            # source-longest // target_px gives an integer stride that
            # produces ≥ target_px output. Clamped at 1.
            if target_px is not None:
                src_size = (getattr(it, '_rgb_jxl_size', None)
                            or _native_dims(it, np.zeros((1, 1, 3))))
                src_longest = max(int(src_size[0]), int(src_size[1]))
                ds = max(1, src_longest // int(target_px))
            else:
                ds = 4
            uhdr_resolved = resolve_uhdr_thumb_bytes(it, downsample=ds)
            if uhdr_resolved is not None:
                # Placeholder float array sized to match the actual
                # thumb pixel dims so ``_native_dims`` reports the cell
                # bbox at the same resolution the bytes encode — browser
                # renders the embedded UHDR JPEG 1:1 inside the bbox.
                thumb_bytes, (h, w) = uhdr_resolved
                from .hdr_cmap import HdrCmapArray
                arr = HdrCmapArray(np.zeros((h, w, 3), dtype=np.float32))
                arr._uhdr_thumb_bytes = thumb_bytes
                return arr  # bypass the dx-stride below
            # Fall back to float decode + re-encode for items without
            # UHDR-decodable bytes (rare: scenes with only ``_rgb``
            # set and no on-disk JPG / in-memory UHDR cache).
            arr = resolve_linear_p3(it, target_px=target_px)
            if arr is None:
                rgb = getattr(it, 'rgb', None)
                if rgb is None:
                    raise TypeError(
                        f"image_grid: item {type(it).__name__} has neither "
                        f"`_rgb_linear_p3`, an `rgb_path` we can decode, "
                        f"nor an `rgb` attribute."
                    )
                arr = np.asarray(rgb)
        if dx != 1:
            arr = arr[::dx, ::dx]
        return arr

    # Parallelize across items: each tile's resolve work — JXL header
    # peek + ds=8 decode + PQ inversion — releases the GIL via opencodecs
    # so a thread pool gets real wall-clock speedup. NAS reads scale with
    # concurrent SMB ops too. Cap workers so we don't oversubscribe
    # libjxl's own thread runner (which already uses CPU-count threads
    # per decode call).
    n = len(items)
    if n > 1:
        # Decode at ds=8 is mostly NAS I/O on real workloads; libjxl's
        # internal thread runner mops up the CPU slack while reads
        # block. So bump workers to N or 16, whichever is smaller, to
        # keep all the SMB ops in flight at once.
        n_workers = min(n, 16)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            arrays = list(ex.map(_resolve_one, items))
    else:
        arrays = [_resolve_one(items[0])]

    any_hdr = any(np.issubdtype(a.dtype, np.floating) for a in arrays)
    # Default HDR codec is Ultra-HDR JPEG (ISO 21496-1): cross-browser HDR
    # (Safari + Chrome composite the gain map, Firefox / Preview / Quick
    # Look fall back to the SDR base cleanly) AND smaller than JXL-PQ at
    # the same quality. Pass ``raster_format='jxl-hdr-pq'`` to force JXL.
    raster_format = 'uhdr' if any_hdr else 'jxl-p3'

    if any_hdr:
        # Promote uint8 cells to float so they composite alongside HDR
        # at the absolute SDR-white reference (uint8 1.0 → linear 1.0).
        arrays = [
            (a.astype(np.float32) / 255.0) if a.dtype == np.uint8 else a
            for a in arrays
        ]
    return arrays, raster_format


def _to_array_only(items, *, dx):
    """Coerce a flat ``items`` list to arrays for the matplotlib path
    (which doesn't know about scene-like objects)."""
    out = []
    for it in items:
        if isinstance(it, np.ndarray):
            arr = it
        elif hasattr(it, 'rgb'):
            arr = np.asarray(it.rgb)
        else:
            raise TypeError(
                f"image_grid(backend='matplotlib'): item {type(it).__name__} "
                f"has no `.rgb` and isn't an ndarray."
            )
        if dx != 1:
            arr = arr[::dx, ::dx]
        out.append(arr)
    return out


def _resize_nearest(arr, dst_h, dst_w):
    """Downsample to (dst_h, dst_w). Prefers area-averaging (mean over
    integer blocks) when both axes scale by an integer factor — that's
    a perfect zero-offset filter for HDR linear-light input and a fine
    approximation for SDR uint8. Falls back to centered nearest-
    neighbor sampling (sampled at bin centers, not top-left corners) so
    the downsampled image aligns visually with the source — no half-bin
    shift when the hi-res replacement lands.

    Never upscales: if the source is already ≤ the target, returns the
    source unchanged. The SVG <image> element embeds the smaller raster
    at the cell's bbox size and the browser handles display scaling via
    image-rendering: pixelated. Pre-upscaling here would just waste
    CPU + bytes for an identical visual result.
    """
    # Carry along a paired SDR base layer (used by HdrCmapArray) so the
    # UHDR encoder downstream sees matched shapes. Strip the subclass
    # for the actual resize math to avoid surprising ufunc dispatch.
    sdr_base = getattr(arr, '_sdr_base_p3_u8', None)
    if sdr_base is not None:
        plain = np.asarray(arr)
        hdr_resized = _resize_nearest(plain, dst_h, dst_w)
        if hdr_resized is plain:
            return arr  # no resize happened — return original subclass
        sdr_resized = _resize_nearest(sdr_base, dst_h, dst_w)
        from .hdr_cmap import HdrCmapArray
        return HdrCmapArray(hdr_resized, sdr_base_p3_u8=sdr_resized)

    src_h, src_w = arr.shape[:2]
    if src_h <= dst_h and src_w <= dst_w:
        return arr

    # Area-averaging when both dimensions are integer multiples — cheap
    # via reshape+mean; gives perfect bin-center alignment.
    if src_h >= dst_h and src_w >= dst_w \
            and src_h % dst_h == 0 and src_w % dst_w == 0:
        fy = src_h // dst_h
        fx = src_w // dst_w
        if arr.ndim == 2:
            return arr.reshape(dst_h, fy, dst_w, fx).mean(axis=(1, 3)).astype(arr.dtype, copy=False)
        c = arr.shape[2]
        return (
            arr.reshape(dst_h, fy, dst_w, fx, c).mean(axis=(1, 3))
            .astype(arr.dtype, copy=False)
        )

    # Centered nearest-neighbor: sample the source pixel whose CENTER
    # is closest to each output pixel's center. ``+ 0.5`` shifts to bin
    # centers; ``- 0.5`` re-aligns to integer source coords.
    yi = (np.arange(dst_h) + 0.5) * (src_h / dst_h) - 0.5
    xi = (np.arange(dst_w) + 0.5) * (src_w / dst_w) - 0.5
    yi = np.clip(np.rint(yi), 0, src_h - 1).astype(np.int64)
    xi = np.clip(np.rint(xi), 0, src_w - 1).astype(np.int64)
    if arr.ndim == 2:
        return arr[np.ix_(yi, xi)]
    return arr[yi[:, None], xi[None, :]]


def _label_position(x, y, w, h, lpos):
    pad = 3.0
    if lpos == 'top_middle':
        return x + w / 2, y + pad, 'middle', 'hanging'
    if lpos == 'top_left':
        return x + pad, y + pad, 'start', 'hanging'
    if lpos == 'bottom_middle':
        return x + w / 2, y + h - pad, 'middle', 'alphabetic'
    if lpos == 'bottom_left':
        return x + pad, y + h - pad, 'start', 'alphabetic'
    if lpos == 'above_middle':
        return x + w / 2, y - pad, 'middle', 'alphabetic'
    raise ValueError(f"unknown lpos {lpos!r}")


__all__ = ['image_grid']
