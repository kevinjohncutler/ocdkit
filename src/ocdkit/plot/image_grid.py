"""Unified image-grid plotter — SVG (HDR-capable) by default, matplotlib
on opt-in.

The default ``backend='svg'`` produces an :class:`ocdkit.io.SvgFigure`
with one composite raster per tile + vector outlines/labels. The codec
is chosen per cell: float linear-light Display-P3 arrays (or Scene-like
objects carrying ``_rgb_linear_p3``) encode as Ultra-HDR JPEG; ``uint8``
inputs encode as lossless PNG (bit-exact, decodes everywhere).

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


def _srgb_oetf(x):
    """Linear-light → sRGB-gamma (the inverse of the shader's ``eotf``). Used to
    encode HDR-nearest tile textures so the WebGPU controller's
    ``eotf→×headroom→oetf`` reconstructs the linear values exactly."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 0, None), 1 / 2.4) - 0.055)


class _RawF16Source:
    """Raw float16 RGBA tile for DIRECT GPU upload — NO image codec at all. The
    static-HDR controller fetches these bytes and does ``device.queue.writeTexture``
    into an ``rgba16float`` texture; the shader then skips ``eotf`` because the data
    is already linear-light. Zero encode (vs ~280-820ms PNG) and full HDR precision
    (no 8-bit boost-banding). Bytes = an 8-byte header ``struct('<II', w, h)`` then
    ``w*h*4`` float16 (RGBA, alpha=1). For a Scene it resolves linear-P3 lazily; for
    an ndarray it uses it directly. ``downsample`` decimates (``arr[::ds,::ds]``,
    ~free) for the progressive low-res tier that paints before the full-res lands."""
    content_type = 'application/octet-stream'

    def __init__(self, it, downsample=1):
        import threading
        self._it = it
        self._ds = max(1, int(downsample))
        self._bytes = None
        self._lock = threading.Lock()

    def get_bytes(self):
        with self._lock:
            if self._bytes is None:
                import struct
                it = self._it
                if isinstance(it, np.ndarray):
                    lin = it
                else:
                    from ..io.figure_server import resolve_linear_p3
                    lin = resolve_linear_p3(it, target_px=None)
                a = np.asarray(lin, np.float32)
                if self._ds > 1:
                    a = a[::self._ds, ::self._ds]
                h, w = a.shape[:2]
                rgb = np.clip(a[..., :3], 0.0, None)
                rgba = np.dstack([rgb, np.ones((h, w), np.float32)]).astype(np.float16)
                self._bytes = (struct.pack('<II', int(w), int(h))
                               + np.ascontiguousarray(rgba).tobytes())
        return self._bytes


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
    # ``fontcolor`` accepts four sentinel keywords plus any literal CSS color:
    #   * ``'auto'`` (default): per-cell content-adaptive — sample image
    #     luminance under each label and pick a near-white fill on dark
    #     cells, near-black on light. Fill only; no stroke / halo.
    #   * ``'light'`` / ``'dark'``: fixed near-white / near-black across
    #     every cell. Use when content stays on one side of 50% luminance.
    #   * ``'currentColor'``: CSS-driven path that inherits from the
    #     host's ``color`` (JupyterLab / dashboard theme).
    #   * Any other string: literal color (``'lightgray'``,
    #     ``'#808080'``, ``'rgb(...)'`` etc.) passed through verbatim.
    fontcolor: str = 'auto',
    lpos: str = 'top_middle',
    facecolor: str | None = None,
    outline: bool = False,
    # ``outline_color`` accepts ``'currentColor'`` for CSS-theme driven
    # outlines or any literal color. Per-cell adaptation doesn't apply
    # (the outline spans the whole cell border).
    outline_color: str = 'currentColor',
    outline_width: float = 0.5,
    raster_format: str | None = None,     # None → autodetect from dtype
    # Float tiles within [0,1] encode as an sRGB-OETF PNG + ``data-hdr`` and
    # render through the WebGPU rgba16float canvas with a NEAREST sampler (true
    # nearest HDR glow, no gain-map interpolation cross) — matching how the
    # scene's key-slice RGB tile feeds createLinkedHDRLayer. >1 true-HDR tiles
    # stay on uhdr (per-pixel gain map; large enough that interpolation is
    # invisible). Default True; set False to force the native-uhdr path.
    hdr_nearest: bool = True,
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
    auto_upgrade='auto',                  # eagerly stream hi-res into inline;
                                          # 'auto' = on for small/on-disk grids
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

    # Resolve the fixed-shade sentinels up front. ``'auto'`` is handled
    # per-cell in the label-emission loop because it needs to sample each
    # tile's pixel content; the other two are just nicely-named aliases.
    if fontcolor == 'light':
        fontcolor = _LIGHT_TEXT
    elif fontcolor == 'dark':
        fontcolor = _DARK_TEXT
    if outline_color == 'auto':
        # No content to sample for the whole-cell border — fall through
        # to the CSS-theme path.
        outline_color = 'currentColor'

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

    # ── layered tiles: ``{'base': <img>, 'overlay': <RGBA uint8>}`` ──────
    # A dict item is a two-layer cell. The overlay (a pre-rasterized RGBA
    # array, e.g. ``outline_view(..., layered=True)`` boundaries) is
    # alpha-composited INTO the base, and the combined tile is then forced
    # to a *lossless* encode (lossless uhdr for HDR float, PNG for SDR
    # uint8). Baking-then-lossless is what makes a colored outline both
    # bit-exact/crisp AND bright over an HDR base: a separate SDR PNG
    # overlay renders at SDR-white and looks dim against an HDR-boosted
    # base, and a transparent HDR overlay is impossible (JPEG/uhdr carry
    # no alpha). With the outline in the lossless SDR base, the gain map
    # rides it up to the same HDR brightness as the rest of the tile —
    # pure color, no blend desaturation. A dict can't collide with the
    # nested-row syntax above (only list/tuple trigger that). The
    # matplotlib backend has no compositing — it sees the bare base.
    #
    # A second dict shape — ``{'labels': int2d, 'palette':…, …}`` — is a
    # LIVE GPU segmentation tile: the integer label matrix + palette ship
    # to the browser and the shared WebGL2 LabelGLRenderer (the same engine
    # as the ocdkit viewer) colorizes + outlines + hover-highlights it.
    # No raster, no codec — lossless by construction, HDR-capable, free
    # crisp outlines. Live-only (a <canvas>, not embeddable in a saved
    # static SVG/.ipynb).
    baked_lossless = [False] * len(items)
    label_tiles = [None] * len(items)
    if any(isinstance(it, dict) for it in items):
        base_items = []
        # ncolor.label is the dominant per-label-tile CPU cost, and a grid
        # often shows the SAME segmentation in more than one tile (e.g. a fill
        # view + an outline-over-image view). Cache the relabel by source-array
        # identity so each distinct array is ncolor'd once, not once per tile.
        _ncolor_cache = {}
        for i, it in enumerate(items):
            if isinstance(it, dict) and 'labels' in it:
                inst = np.asarray(it['labels'])   # original per-cell instance ids
                lbl = inst                        # color ids (== instance unless ncolor)
                pal_opt = it.get('palette')
                # Default coloring = ncolor relabel + sinebow (matches
                # apply_ncolor): adjacent cells get distinct colors. Done in
                # Python; the GPU then just looks up the small palette. Pass
                # an explicit palette (array) to skip ncolor, or
                # ``palette='raw'`` to color the raw label ids via the
                # shader's procedural sinebow (no ncolor relabel).
                #
                # The ncolor relabel collapses many cells onto a few GROUP
                # ids (that's how adjacent cells differ in color), so the
                # group id can't drive hover — every cell sharing the group
                # would light up. We keep the original INSTANCE ids (`inst`)
                # separately so the GPU highlights a single cell, mirroring
                # the viewer's nColorInstanceMask.
                if pal_opt is None or (isinstance(pal_opt, str)
                                       and pal_opt in ('ncolor', 'sinebow')):
                    from .ncolor import ncolor_labels_and_palette
                    _ck = id(it['labels'])   # same array in >1 tile → relabel once
                    if _ck in _ncolor_cache:
                        lbl, pal_opt = _ncolor_cache[_ck]
                    else:
                        lbl, pal_opt = ncolor_labels_and_palette(inst)
                        _ncolor_cache[_ck] = (lbl, pal_opt)
                elif isinstance(pal_opt, str) and pal_opt == 'raw':
                    pal_opt = 'sinebow'   # shader procedural sinebow on raw ids
                label_tiles[i] = {
                    'labels': lbl,        # color/group ids (palette lookup)
                    'instances': inst,    # original ids (hover/picking + outlines)
                    'palette': pal_opt,
                    'image': it.get('image'),   # optional base under the labels
                    'opacity': float(it.get('opacity', 0.6)),
                    'style': it.get('style', 'both'),
                    'outlines': bool(it.get('outlines', True)),
                    # Boundary brightness boost. >1 makes the per-cell outline
                    # EMIT HDR (>1.0) on the float16 extended-range label canvas, so
                    # the contour glows like the HDR base instead of reading flat /
                    # SDR. Default 4.0 (clearly HDR; the base content is ~8x, so this
                    # glows without overpowering it). Tune via ``outline_hdr=`` — e.g.
                    # ~8 to match the base, 1.0 for plain SDR. On an SDR display (or
                    # the figure's SDR toggle) it clamps to a brighter edge.
                    'outline_hdr': float(it.get('outline_hdr', 4.0)),
                    # Uniform contour color (matplotlib color spec or hex), or
                    # None = per-cell palette color. None is the DEFAULT for every
                    # style (incl. ``style='outline'``): the perimeter inherits each
                    # cell's own color so two TOUCHING cells show distinct-coloured
                    # boundaries instead of merging into one red line. Pass a color
                    # (e.g. 'red', 'cyan', '#00ff88', (1,0,0)) to force a uniform contour.
                    'outline_color': it.get('outline_color', None),
                    # Whether a base image rides BEHIND the (transparent) GPU
                    # label canvas as a normal raster tile (see below).
                    'has_base': it.get('image') is not None,
                }
                # The base image (e.g. max projection) rides through the NORMAL
                # tile raster pipeline — registered + HDR-encoded (uhdr/jxl) +
                # served exactly like every other tile — so a float base keeps
                # its gain-map HDR. It's painted as a plain <image> BEHIND the
                # transparent GPU label canvas (the canvas draws only labels/
                # outlines). Uploading it into the GPU canvas flattened it to
                # SDR (the reported regression). No base → a zero placeholder
                # (kept for layout; the label tile just won't emit the image).
                _base = it.get('image')
                if _base is not None:
                    _base = np.asarray(_base)
                    if _base.ndim == 2:
                        _base = np.repeat(_base[..., None], 3, axis=2)
                    elif _base.ndim == 3 and _base.shape[2] == 1:
                        _base = np.repeat(_base, 3, axis=2)
                    base_items.append(_base)
                else:
                    base_items.append(np.zeros(lbl.shape[:2] + (3,), np.uint8))
            elif isinstance(it, dict) and 'base' in it:
                base = np.asarray(it['base'])
                ov = it.get('overlay')
                if ov is not None:
                    base = _composite_overlay(base, np.asarray(ov))
                base_items.append(base)
                baked_lossless[i] = True
            else:
                base_items.append(it)
        items = base_items

    if backend == 'matplotlib':
        # Re-split for the legacy nested-list API.
        images = split_list(_to_array_only(items, dx=dx), ncol)
        labels = (split_list(list(plot_labels), ncol)
                  if plot_labels is not None else None)
        # The SVG-only sentinels (``'auto'``, ``'currentColor'``) have no
        # matplotlib equivalent. Map them to a neutral gray so the legacy
        # backend renders cleanly; pinned ``'light'`` / ``'dark'`` were
        # already resolved to literals above.
        mpl_fontcolor = (
            '#808080' if fontcolor in ('auto', 'currentColor') else fontcolor
        )
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
    # not explicitly set. Mixed-dtype grids keep HDR float cells on the
    # uhdr path and route SDR uint8 cells (seg/outline tiles) to lossless
    # PNG via ``cell_fmts`` — see ``_resolve_items``.
    # Unify HDR rendering when a grid MIXES scene-like (native-uhdr gain-map JPEG)
    # cells with raw float-array (data-hdr/WebGPU) cells — e.g. imshow([scene,
    # scene.rgb]). Left split, the two render through different stacks (native
    # <image> vs WebGPU canvas) → different brightness, zoom viewer, and sharpness
    # = the seam/flash. Routing the scene cells through the SAME data-hdr path
    # (resolve to a real linear-P3 float → OETF-PNG + WebGPU) makes both identical.
    # Scoped to the MIX so an all-scene grid keeps the fast embedded-thumb path.
    _has_float_arr = any(isinstance(it, np.ndarray)
                         and np.issubdtype(it.dtype, np.floating) for it in items)
    _has_scene = any((not isinstance(it, np.ndarray))
                     and (hasattr(it, 'rgb_path') or hasattr(it, '_rgb_linear_p3')
                          or hasattr(it, 'rgb')) for it in items)
    _hdr_unify = bool(hdr_nearest and _has_float_arr and _has_scene)

    arrays, auto_fmt, cell_fmts = _resolve_items(items, dx=dx,
                                                 target_px=int(target_tile_px),
                                                 hdr_unify=_hdr_unify)
    if raster_format is None:
        raster_format = auto_fmt
    else:
        # An explicit caller-supplied format overrides the per-cell
        # auto-routing for every tile.
        cell_fmts = [raster_format] * len(arrays)

    # Baked layered tiles force a lossless encode so the composited
    # outline survives byte-exact. Float bases → lossless uhdr (gain map
    # still lifts the outline to HDR brightness); uint8 bases are already
    # routed to lossless PNG, so nothing to change there.
    for i, is_baked in enumerate(baked_lossless):
        if is_baked and np.issubdtype(arrays[i].dtype, np.floating):
            cell_fmts[i] = 'uhdr-lossless'

    # ── HDR-nearest routing ─────────────────────────────────────────────
    # ``cell_hdr[i]`` flags tiles that render through the WebGPU rgba16float
    # canvas with a NEAREST sampler (the ``data-hdr`` static controller): a plain
    # sRGB PNG whose glow comes from the shader's ``eotf×headroom×oetf``. The PNG
    # carries the REAL sRGB pixels, so nearest sampling shows them exactly — no
    # gain map, so no bilinear-gain-map cross on small/flat tiles. Only float
    # tiles fully within [0,1] qualify (no >1 headroom to lose); >1 true-HDR
    # tiles stay on ``uhdr`` (per-pixel gain map; large enough that the
    # interpolation is invisible). Opt-in via ``hdr_nearest`` so existing grids'
    # brightness model is unchanged.
    cell_hdr = [False] * len(arrays)
    if hdr_nearest:
        for i, a in enumerate(arrays):
            if (np.issubdtype(a.dtype, np.floating)
                    and cell_fmts[i] in ('uhdr', 'uhdr-lossless')
                    and float(np.nanmax(a)) <= 1.0 + 1e-6):
                cell_fmts[i] = 'png'   # plain sRGB texture for the nearest shader
                cell_hdr[i] = True

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

    # Hi-res for click-to-expand: encode each item's full-resolution bytes and
    # host them as tileserve ATTACHMENTS, NOT on the standalone figure_server.
    # Why tileserve: it's the engine that ships with the ``ocdkit-tiles`` Jupyter
    # proxy, so its ``/attach/<sid>/<name>`` URLs are reachable from a REMOTE
    # browser (the figure shell's __ocdResolveTileUrl rewrites the baked
    # 127.0.0.1 attach URL to {baseUrl}ocdkit-tiles/<port>/...). figure_server
    # binds a random un-proxied port → unreachable off-machine, which is why
    # remote zoom never got past the inline thumbnail. ``resolve_source`` prefers
    # ``rgb_path`` (zero re-encode), then ``_rgb_linear_p3`` (HDR), ``.rgb``
    # (SDR), then raw ndarrays; ``None`` → that cell's zoom falls back to the
    # inline thumbnail. Each cell stays INDEPENDENT (its own attachment + its own
    # click-to-expand popup) — no linked viewport.
    _tbase = _tsid = _tsattach = None
    _n_encode_hires = 0   # hi-res cells whose bytes need an in-memory encode (ArraySource)
    try:
        from ..io.figure_server import resolve_source, ArraySource
        from ..tileserve.server import ensure_server, register_pending, attach
        _tsattach = attach

        def _hires_source(it, fmt, is_hdr=False):
            if fmt == 'png' and is_hdr:
                # data-hdr cell — unified SCENE, raw float ARRAY, OR a label tile's
                # HDR base (e.g. a max projection under outlines). Ship the linear
                # float as RAW float16 straight to the GPU (no image codec): the
                # static-HDR controller (inline) and LabelGL.setBaseRaw (label zoom)
                # both upload it into rgba16float and skip eotf. Zero encode (was
                # ~280-820ms PNG) + full HDR precision, and the label zoom upgrades
                # as fast as the standalone cells — no PNG bottleneck. cell_hdr is
                # set only when a scene already resolved to linear, so it's known-good.
                return _RawF16Source(it)
            if fmt == 'png' and isinstance(it, np.ndarray):
                # non-HDR png array: raw bytes, no OETF.
                return ArraySource(it, fmt='png')
            if fmt == 'uhdr-lossless' and isinstance(it, np.ndarray):
                return ArraySource(it, fmt='uhdr', lossless=True)
            return resolve_source(it)

        _tbase = ensure_server()
        # attachment-only source (1×1 placeholder layer never requested); the
        # hi-res bytes ride as named attachments on it.
        _tsid = register_pending(1, 1, ['_hires'])
        # These attachments (hi-res, display thumb, label matrices) are IMMUTABLE
        # for the life of this source id, so serve them browser-CACHEABLE instead
        # of the /attach default ``no-store``. Without this, every click-to-zoom
        # re-fetched the full-res tile from scratch ("always has to upgrade"),
        # whereas a Scene's on-disk PathSource is cached by its mtime URL — the
        # whole reason ``imshow(scene)`` felt instant and ``imshow(scene.rgb)``
        # (in-memory array) re-upgraded every time.
        _IMMUTABLE_HDRS = {"Cache-Control": "private, max-age=86400, immutable"}
        # Encode + attach the hi-res bytes SYNCHRONOUSLY. (Backgrounding this on
        # a daemon thread was tried to shave the encode off first paint, but the
        # uhdr/jxl C encoders are not reliably thread-safe — the encode could
        # fail silently in the worker so the attachment never landed and the tile
        # never upgraded. The dominant interactive cost is the ncolor relabel,
        # now content-cached, so the eager encode is an acceptable one-time cost
        # for a RELIABLE upgrade.) The thumb is already inline, so the figure
        # still paints immediately; the browser swaps to hi-res once fetched.
        # Attach the hi-res OFF the critical path. ArraySource already encodes in
        # its own daemon thread; the only reason imshow(scene.rgb) was ~10x slower
        # than imshow(scene) (a PathSource file reference, zero encode) is that we
        # BLOCKED on get_bytes() here — the full-res float→PNG/uhdr encode (~130ms
        # PNG / ~470ms uhdr, measured). Wait + attach on a per-cell daemon thread
        # instead: imshow returns immediately (thumb + SVG, ~scene speed), the
        # /attach endpoint serves 204 until the encode lands, and the tile
        # controllers poll/retry → swap to hi-res when ready. (A PathSource's
        # get_bytes is a fast file read, so its waiter finishes ~instantly too.)
        import threading as _threading
        hires_urls = []
        raw_disp_urls = []   # data-hdr raw path: decimated f16 disp tier (progressive)
        # Attach the hi-res bytes OFF the critical path: ``imshow`` returns
        # immediately with the thumb + URLs, and each ``/attach/.../hiresN`` lands
        # when its encode finishes (the inline prefetch + popup both 204-retry until
        # ready). ``ArraySource`` already encodes in its own daemon thread, and a
        # PathSource (Scene file) is a zero-encode read — so this is non-blocking.
        def _attach_hires_bg(_src, _name):
            try:
                attach(_tsid, _name, _src.get_bytes(), headers=_IMMUTABLE_HDRS,
                       media=_src.content_type)
            except Exception:
                pass   # encode/attach failed → controllers keep the thumb
        for i, it in enumerate(items):
            src = _hires_source(it, cell_fmts[i], cell_hdr[i])
            if src is None:
                hires_urls.append(None)
                raw_disp_urls.append(None)
                continue
            if isinstance(src, (ArraySource, _RawF16Source)):
                _n_encode_hires += 1   # in-memory encode (vs zero-cost PathSource file)
            _name = f'hires{i}'
            hires_urls.append(f'{_tbase}/attach/{_tsid}/{_name}')
            _threading.Thread(target=_attach_hires_bg, args=(src, _name),
                              daemon=True, name=f'ocd-hires-{i}').start()
            # data-hdr raw path: also attach a DECIMATED raw-f16 disp so the WebGPU
            # cell paints a cheap low-res first, then upgrades to the full-res raw.
            # All raw, no codec; arr[::4,::4] is ~free.
            if cell_hdr[i] and isinstance(src, _RawF16Source):
                _dname = f'rawdisp{i}'
                raw_disp_urls.append(f'{_tbase}/attach/{_tsid}/{_dname}')
                _threading.Thread(target=_attach_hires_bg,
                                  args=(_RawF16Source(it, downsample=4), _dname),
                                  daemon=True, name=f'ocd-rawdisp-{i}').start()
            else:
                raw_disp_urls.append(None)
        # Stream LARGE label/instance matrices as raw-uint16 attachments instead
        # of inlining them as multi-MB base64 (a 2048² seg is ~11 MB/matrix; two
        # tiles sharing one seg ballooned the SVG to ~45 MB and stalled the parse).
        # The browser fetch()es the bytes straight to the GPU — no base64 inflation,
        # no giant XML. Dedup by array identity so a seg reused across tiles ships
        # once. Small tiles stay inline (self-contained, no fetch latency).
        _STREAM_MIN_BYTES = 1_000_000          # ~> 700² ; below this inline is fine
        _mat_src_cache = {}                     # id(arr) -> attachment URL

        def _attach_matrix(arr, tag):
            key = id(arr)
            u = _mat_src_cache.get(key)
            if u is not None:
                return u
            a = np.asarray(arr)
            if int(a.max(initial=0)) > 65535:   # uint16 wire: renumber if overflow
                import fastremap
                a = fastremap.renumber(a, in_place=False)[0]
            raw = np.ascontiguousarray(a.astype('<u2')).tobytes()
            name = f'{tag}{len(_mat_src_cache)}'
            attach(_tsid, name, raw, headers=_IMMUTABLE_HDRS,
                   media='application/octet-stream')
            u = f'{_tbase}/attach/{_tsid}/{name}'
            _mat_src_cache[key] = u
            return u

        for i, lt in enumerate(label_tiles):
            if lt is None:
                continue
            lbl = np.asarray(lt['labels'])
            if lbl.size * 2 < _STREAM_MIN_BYTES:
                continue                        # small → keep inline base64
            inst = lt.get('instances')
            lt['labels_src'] = _attach_matrix(lbl, 'lbl')
            if inst is not None and not np.array_equal(np.asarray(inst), lbl):
                lt['instances_src'] = _attach_matrix(inst, 'inst')
    except Exception:
        hires_urls = [None] * len(items)

    # Build per-tile data URLs in parallel, caching the encoded bytes on
    # each scene-like item so subsequent renders (same scenes, same
    # format) skip the encode entirely. Each cached URL is paired with
    # a fingerprint of the source state (rgb_path mtime + id of any
    # in-memory RGB caches) so a regenerated ``scene._rgb`` or a
    # touched ``rgb_path`` invalidates the URL and forces a re-encode.
    def _get_or_build_url(idx):
        it = items[idx]
        fmt = cell_fmts[idx]
        cache_attr = f'_thumb_url_{fmt}'
        fp_attr = f'_thumb_url_{fmt}_fp'
        if not isinstance(it, np.ndarray):
            fp_now = _source_fingerprint(it)
            if getattr(it, fp_attr, None) == fp_now:
                cached = getattr(it, cache_attr, None)
                if cached is not None:
                    return cached
        # HDR-nearest tiles: the source float is linear-light P3 (image_grid's
        # convention). The WebGPU controller's shader does eotf→×headroom→oetf,
        # so the texture must be sRGB-OETF (gamma) encoded — exactly how the
        # scene's key-slice RGB tile feeds createLinkedHDRLayer. Gamma-encode
        # here; the shader eotf's it back to linear before the headroom boost.
        _enc_arr = arrays[idx]
        if cell_hdr[idx]:
            _enc_arr = _srgb_oetf(np.clip(_enc_arr, 0.0, 1.0))
        url = _encode_thumb_url(_enc_arr, fmt, sdr_white_nits)
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

    # Kill the heavy inline base64 thumb for a PURE scene grid (no linked
    # viewport, no label tiles — the ``image_grid([scenes])`` case): host each
    # cell's display thumbnail as a tileserve ATTACHMENT and load it via
    # ``data-tile-async`` (the stream controller routes it through the
    # __ocdResolveTileUrl proxy → reachable remotely). The SVG then carries a
    # short URL per cell instead of a multi-KB base64 blob — a fraction of the
    # payload — and the cell display works off-machine. The link_axes / label
    # paths keep base64 (they have their own GL / overlay wiring). Export still
    # works: _rasterizable_svg inlines these tiles server-side for resvg.
    _disp_async = False
    if (_tsid is not None and _tsattach is not None
            and not link_axes
            and not any(lt is not None for lt in label_tiles)):
        try:
            import base64 as _b64
            import re
            _du = []
            for _i, _u in enumerate(urls):
                # HDR-nearest tiles stay INLINE: the static HDR controller needs
                # the texture href immediately (the async stream controller swaps
                # href later, and the two don't coordinate → the tile would show
                # as the bare SDR PNG with no WebGPU glow). Small glow swatches
                # don't benefit from streaming anyway.
                _mm = re.match(r'data:([^;]+);base64,(.*)$', _u or '', re.S)
                if _mm and not cell_hdr[_i]:
                    _tsattach(_tsid, f'disp{_i}', _b64.b64decode(_mm.group(2)),
                              headers=_IMMUTABLE_HDRS, media=_mm.group(1))
                    _du.append(f'{_tbase}/attach/{_tsid}/disp{_i}')
                else:
                    _du.append(_u)
            urls = _du
            _disp_async = True
        except Exception:
            _disp_async = False

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
    # Resolve the ``auto_upgrade='auto'`` default: eagerly stream hi-res into the
    # inline preview on page load (so a native-uhdr/scene cell isn't stuck showing
    # the blurry disp thumb until you hover) — but ONLY when warming is cheap. That
    # is: a small grid, OR a grid with no in-memory encodes (all on-disk PathSource
    # files, e.g. a wall of Scenes — /attach serves cached bytes, no encode). A
    # large all-in-memory grid stays hover-gated to avoid an N-way parallel uhdr
    # encode storm at first paint.
    if auto_upgrade == 'auto':
        _n_hires = sum(1 for u in hires_urls if u)
        auto_upgrade = (_n_hires <= 16) or (_n_encode_hires == 0)

    for i, arr in enumerate(arrays):
        r, c = divmod(i, ncol)
        x = margin_vb + sum(cell_w_pxs[r * ncol:r * ncol + c]) + gap_vb * c
        y = row_y_offsets[r]
        w, h = cell_w_pxs[i], cell_h_pxs[i]
        hires_attr = (f' data-hires-href="{hires_urls[i]}"'
                      if hires_urls[i] else '')
        # data-hdr raw path: the decimated raw-f16 disp tier (progressive first paint).
        raw_disp_attr = (f' data-raw-disp-href="{raw_disp_urls[i]}"'
                         if raw_disp_urls[i] else '')
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
            f'{thumb_attr}{hires_attr}{raw_disp_attr}{upgrade_attr}>'
        )
        if label_tiles[i] is not None:
            # Behind the (transparent) GPU label canvas:
            #  * with a base image (e.g. max projection) — a plain <image>
            #    using the normal registered + HDR-encoded raster URL, so a
            #    float base keeps its gain-map HDR;
            #  * without one — a SOLID themed backdrop so the semi-transparent
            #    ncolor cells composite over a consistent surface instead of
            #    the bare page. ``light-dark()`` → light in light mode, dark in
            #    dark mode (matches the figure's canvas).
            if label_tiles[i].get('has_base'):
                # The base under the GPU label canvas (e.g. a gray max-projection)
                # is an HDR-nearest float tile too — tag it ``data-hdr`` so the
                # static HDR controller glows it (OETF PNG + WebGPU canvas). The
                # canvas sits below the SVG; the opacity:0 image lets it show, and
                # the transparent label canvas (outlines) composites on top.
                _base_hdr = (f' data-hdr="1" data-hdr-headroom="{sdr_white_nits / 203.0:.4f}"'
                             if cell_hdr[i] else '')
                svg.add(
                    f'<image x="{x:.2f}" y="{y:.2f}" '
                    f'width="{w:.2f}" height="{h:.2f}" href="{urls[i]}"{_base_hdr} '
                    f'preserveAspectRatio="none" image-rendering="pixelated"/>')
            else:
                # ``Canvas`` = the CSS system page-background color: WHITE in
                # light mode, BLACK (the page color) in dark mode. A real solid
                # opaque fill that matches the page — so it blocks anything
                # behind it (the earlier light-dark() fill rendered UNFILLED =
                # see-through in some renderers; system colors always paint and
                # auto-adapt to the color scheme).
                svg.add(
                    f'<rect x="{x:.2f}" y="{y:.2f}" '
                    f'width="{w:.2f}" height="{h:.2f}" '
                    f'fill="Canvas"/>')
            # Live GPU segmentation tile — a transparent <canvas> the figure
            # shell's LabelGLRenderer colorizes/outlines/highlights from the
            # label matrix (see io/figure.py controller). The title is emitted
            # as a normal SVG <text> AFTER the foreignObject (below) so it
            # paints on top; SVG text lays out where an HTML <div> inside the
            # foreignObject would collapse to zero size.
            lt_title = (plot_labels[i]
                        if plot_labels and i < len(plot_labels) else None)
            svg.add(_label_tile_svg(label_tiles[i], x, y, w, h,
                                    title=lt_title, fontsize_uu=fontsize_uu,
                                    lpos=lpos))
        elif link_axes:
            # Converge on the canonical linked-cell emitter (shared with the
            # scene key-slice grid): image_grid's per-cell data maps to a
            # minimal link_ctx (full-image texture per cell + shared raster +
            # ROI) and a TileInfo-shaped display rect; ``seg_polygons`` rides
            # as the extra_inner overlay; ``data-cell-index`` is dropped (the
            # controller pairs cells<->hits by NodeList index). HDR float cells
            # carry data-hdr so the linked WebGPU HDR layer glows them.
            from types import SimpleNamespace as _NS
            from .linked_cell import emit_linked_cell as _emit_linked_cell
            ras_h, ras_w = linked_raster_shape
            img_href = hires_urls[i] if hires_urls[i] else urls[i]
            _lc = {
                'tiles': [_NS(x0=0, y0=0, x1=ras_w, y1=ras_h)],
                'raster_w': ras_w, 'raster_h': ras_h,
                'roi_str': f'{roi_x:.4f} {roi_y:.4f} {roi_w:.4f} {roi_h:.4f}',
                'tile_urls': [img_href],
            }
            _ti = _NS(x0=x, y0=y, x1=x + w, y1=y + h,
                      label=(plot_labels[i] if plot_labels and i < len(plot_labels) else ''),
                      has_content=True)
            _emit_linked_cell(
                svg, 0, _ti, _lc, 0,
                False, 'outline', 0,
                raster_format='png', linked_outlines=False, tile_box=False,
                hdr=cell_hdr[i], extra_inner=(_link_axes_seg_overlay or ''))
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
            # ``data-hdr`` (WebGPU-nearest path) is set ONLY for ``hdr-nearest``
            # tiles — a plain sRGB PNG whose glow comes from the shader's
            # ``eotf×headroom×oetf`` (the texture carries the REAL sRGB pixels, so
            # nearest sampling shows them exactly). A native ``uhdr`` tile must
            # NOT get data-hdr: its decoded base is the tonemapped/peak-normalized
            # SDR layer, and boosting THAT by headroom ignores the per-pixel gain
            # map (wrong colors). uhdr stays on native browser compositing.
            # ``data-hdr-headroom`` = the content's encoded headroom
            # (``sdr_white_nits/203``, ≈7.88 for the 1600-nit default). The HDR
            # controller boosts the sRGB PNG by THIS (not the flaky JS-detected
            # display headroom, which Chrome won't report → falls to 4.0 and
            # renders DIMMER than the matching native-uhdr cell). The
            # ``toneMapping:'extended'`` canvas then caps at the real display
            # headroom in hardware — so the glow matches the uhdr path exactly.
            _hdr_attr = (f' data-hdr="1" data-hdr-headroom="{sdr_white_nits / 203.0:.4f}"'
                         if cell_hdr[i] else '')
            if _disp_async and not cell_hdr[i]:
                # tileserve-hosted display tile: no inline href — the stream
                # controller fetches data-tile-src (via the proxy resolver) and
                # sets href. data-tile-async lets the SVG compose immediately.
                # (HDR-nearest tiles stay inline — see the _disp_async block.)
                svg.add(
                    f'<image x="{x:.2f}" y="{y:.2f}" '
                    f'width="{w:.2f}" height="{h:.2f}" '
                    f'data-tile-async="1" data-tile-src="{urls[i]}"{_hdr_attr} '
                    # ``auto`` (smooth), NOT ``pixelated``: this is photographic content
                    # scaled into a cell. Nearest (pixelated) at a non-integer ratio drops
                    # columns irregularly = the "columns of pixels missing" seam, worst on
                    # the low-res disp thumb when the hi-res upgrade is slow/stalled. Smooth
                    # scaling never drops columns; the hi-res swap keeps auto too.
                    f'preserveAspectRatio="none" image-rendering="auto"/>'
                )
            else:
                # HDR-nearest tiles carry ``data-hdr`` here (inline sRGB PNG); the
                # static HDR controller overlays a host WebGPU canvas (NEAREST +
                # glow) and sets the image to ``opacity:0`` (invisible but still
                # hit-testable, so click-to-zoom keeps working). A foreignObject
                # canvas was tried but the SVG compositor clamps it to SDR (no
                # glow) — the host-overlay is the proven HDR path.
                svg.add(
                    f'<image x="{x:.2f}" y="{y:.2f}" '
                    f'width="{w:.2f}" height="{h:.2f}" '
                    f'href="{urls[i]}"{_hdr_attr} preserveAspectRatio="none" '
                    # smooth scaling (see the data-tile-async branch): photographic
                    # content, nearest would drop columns = the seam. data-hdr cells
                    # hide this <image> (opacity:0) so it only affects native tiles.
                    f'image-rendering="auto"/>'
                )
        if outline:
            svg.rect(x, y, w, h,
                     fill='none', stroke=outline_color,
                     stroke_width=outline_width_uu)
        label = (plot_labels[i]
                 if plot_labels and i < len(plot_labels) else None)
        if label and label_tiles[i] is not None and not label_tiles[i].get('has_base'):
            # No-base segmentation tile sits on a SOLID ``Canvas`` (page-bg)
            # backdrop (added behind the canvas above). The title uses
            # ``CanvasText`` — the CSS system page-text color, the guaranteed
            # contrast partner of ``Canvas`` (black on the light-mode white
            # backdrop, white on the dark-mode black backdrop). System colors
            # are universally honored as SVG fills and auto-adapt to the color
            # scheme, unlike ``light-dark()`` (which some renderers drop).
            tx, ty, anchor, baseline = _label_position(x, y, w, h, lpos)
            from html import escape as _html_escape
            svg.add(
                f'<text x="{tx:.2f}" y="{ty:.2f}" class="fig-figure-text" '
                f'font-size="{fontsize_uu}" text-anchor="{anchor}" '
                f'dominant-baseline="{baseline}" fill="CanvasText">'
                f'{_html_escape(str(label))}</text>'
            )
            label = None
        # Label tiles WITH a base image fall through to the adaptive `if label:`
        # path below — its luminance sampler reads the tile's base raster
        # (``arr`` = the base image), giving the same adaptive light/dark title
        # as raster tiles. Placed AFTER the foreignObject, the <text> paints on
        # top of the GPU canvas (SVG text lays out where a foreignObject HTML
        # div would collapse to zero size; see _label_tile_svg).
        if label:
            tx, ty, anchor, baseline = _label_position(x, y, w, h, lpos)
            if fontcolor in ('auto', 'auto-cell', 'auto-letter'):
                # Two adaptive modes — both pick from local 90th-percentile
                # luminance so a few outlier pixels under the label don't
                # blend a light fill into a bright spot:
                #
                #   * ``'auto'`` / ``'auto-letter'`` (default): per-LETTER.
                #     Each character samples its own glyph footprint and
                #     gets its own dark / light fill via a ``<tspan>``.
                #     Best on labels that cross a dark→bright boundary.
                #   * ``'auto-cell'``: single fill for the whole label,
                #     picked from the full label-region 90th percentile.
                #     Cheaper SVG payload; previous behaviour.
                #
                # Each emitted text / tspan carries inline CSS custom
                # properties for both an SDR-render pick and an HDR-
                # render pick; the shell stylesheet flips between them
                # at render time so the label tracks the gain-map state
                # without a Python round-trip.
                #
                # HDR detection is PER CELL via ``cell_fmts[i]``, not the
                # grid-level ``raster_format``. PNG cells have no gain
                # map so their displayed brightness is identical in HDR
                # and SDR rendering — using the grid raster_format
                # incorrectly flips SDR-only cell labels (e.g. a uint8
                # segmentation in a mixed grid) when the HDR toggle
                # changes, even though that cell never actually renders
                # differently.
                # ``cell_hdr`` tiles render through the WebGPU HDR canvas (glow
                # ×headroom), so the label contrast pick must use the HDR-aware
                # luminance sampler too — even though the tile encodes as PNG
                # (cell_fmts == 'png'), not uhdr/jxl.
                is_hdr_cell = cell_fmts[i] in ('uhdr', 'jxl-hdr-pq') or cell_hdr[i]
                label_str = str(label)
                per_letter = fontcolor != 'auto-cell'
                from html import escape as _html_escape
                if per_letter and label_str:
                    picks = _per_letter_picks(
                        arr, lpos, fontsize_uu, w, h, label_str,
                        hdr_white_nits=(sdr_white_nits if is_hdr_cell else None),
                    )
                    tspans = ''.join(
                        f'<tspan style="--ocd-tt-hdr:{hdr_p};--ocd-tt-sdr:{sdr_p}">'
                        f'{_html_escape(ch)}</tspan>'
                        for ch, (sdr_p, hdr_p) in zip(label_str, picks)
                    )
                    svg.add(
                        f'<text x="{tx:.2f}" y="{ty:.2f}" '
                        f'class="fig-figure-text ocd-adaptive-text" '
                        f'font-size="{fontsize_uu}" '
                        f'text-anchor="{anchor}" '
                        f'dominant-baseline="{baseline}">'
                        f'{tspans}</text>'
                    )
                else:
                    sdr_lum = _label_region_luminance(
                        arr, lpos, fontsize_uu, w, h, len(label_str),
                        hdr_white_nits=None,
                    )
                    sdr_pick = _DARK_TEXT if sdr_lum > 0.5 else _LIGHT_TEXT
                    if is_hdr_cell:
                        hdr_lum = _label_region_luminance(
                            arr, lpos, fontsize_uu, w, h, len(label_str),
                            hdr_white_nits=sdr_white_nits,
                        )
                        hdr_pick = _DARK_TEXT if hdr_lum > 0.5 else _LIGHT_TEXT
                    else:
                        hdr_pick = sdr_pick
                    style_attr = f'--ocd-tt-hdr:{hdr_pick};--ocd-tt-sdr:{sdr_pick}'
                    svg.add(
                        f'<text x="{tx:.2f}" y="{ty:.2f}" '
                        f'class="fig-figure-text ocd-adaptive-text" '
                        f'style="{style_attr}" '
                        f'font-size="{fontsize_uu}" '
                        f'text-anchor="{anchor}" '
                        f'dominant-baseline="{baseline}">'
                        f'{_html_escape(label_str)}</text>'
                    )
            else:
                svg.text(tx, ty, str(label),
                         fill=fontcolor, size=fontsize_uu,
                         anchor=anchor, baseline=baseline,
                         class_='fig-figure-text')
        svg.add('</g>')

    return SvgFigure(svg.finalize())


# ─── helpers ────────────────────────────────────────────────────────


# Fixed text shades used by ``fontcolor='auto'`` / ``'auto-cell'`` /
# ``'auto-letter'`` / ``'light'`` / ``'dark'``. Pure black and pure white
# — the contrast pick is binary (each glyph or label goes one way or the
# other), so there's no reason to leave headroom; pure values give the
# strongest contrast against any background.
_DARK_TEXT = '#000000'
_LIGHT_TEXT = '#ffffff'


def _label_region_luminance(arr, lpos, fontsize_uu, w_vb, h_vb, label_len,
                              *, hdr_white_nits=None):
    """90th-percentile luminance under a label region, rescaled so the
    caller's ``> 0.5`` threshold corresponds to "the pixel renders
    brighter than an SDR label" — i.e. whichever side of 0.5 picks the
    more contrasty fill.

    Why 90th percentile, not mean: a label that sits over mostly-dark
    pixels with a few bright outliers (a sky strip, a hot specular)
    averages to "dark" but the light SDR label visibly blends into
    those bright pixels. The 90th percentile is robust to a *few* dark
    outliers but still catches significant bright content under the
    label — which is what determines whether a light label is legible.

    SDR (``hdr_white_nits=None``): Rec. 709 luminance, 90th percentile,
    clamped to [0, 1].

    HDR (``hdr_white_nits`` set, e.g. 1600 for the ``image_grid``
    default): linear-light region values are converted to absolute
    display nits (``nits = linear * hdr_white_nits``) then re-normalised
    against ``SDR_REF_NITS = 500`` — empirically the SDR-reference
    brightness an Apple display picks when compositing CSS/SVG colours
    on top of an HDR gain-map layer. (The OS sets this; libuhdr / our
    encoder don't control it. ``1600`` is what our encoder targets for
    HDR peak — see ``svg.py:_p3_linear_to_pq_uint16`` — so the ratio
    ``500 / 1600`` matches what shows up on screen.) With that
    calibration, ``0.5`` lands at HDR-displayed brightness == SDR-label
    brightness, where the contrast pick should flip.

    Returns ``0.5`` (neutral) when sampling is not possible — placeholder
    arrays from ``Source`` items, empty regions, or label positions that
    sit outside the image (``'above_middle'``).
    """
    if arr is None or arr.size == 0:
        return 0.5
    ah, aw = arr.shape[:2]
    if ah == 0 or aw == 0:
        return 0.5

    # Glyph width is roughly 0.6 × font-size for proportional fonts at
    # typical aspect ratios; clamp to the cell so very long labels don't
    # blow past the cell edges and skew the sample.
    label_w_vb = min(0.6 * fontsize_uu * max(label_len, 1), w_vb)
    label_h_vb = fontsize_uu * 1.2
    pad = 3.0  # mirrors ``_label_position``

    if lpos == 'top_middle':
        x0_vb = (w_vb - label_w_vb) / 2
        y0_vb = pad
    elif lpos == 'top_left':
        x0_vb = pad
        y0_vb = pad
    elif lpos == 'bottom_middle':
        x0_vb = (w_vb - label_w_vb) / 2
        y0_vb = h_vb - pad - label_h_vb
    elif lpos == 'bottom_left':
        x0_vb = pad
        y0_vb = h_vb - pad - label_h_vb
    else:
        # ``'above_middle'`` etc.: label is outside the image, no content
        # to sample.
        return 0.5

    x0_px = max(0, int(x0_vb / w_vb * aw))
    y0_px = max(0, int(y0_vb / h_vb * ah))
    x1_px = min(aw, int(np.ceil((x0_vb + label_w_vb) / w_vb * aw)))
    y1_px = min(ah, int(np.ceil((y0_vb + label_h_vb) / h_vb * ah)))
    if x1_px <= x0_px or y1_px <= y0_px:
        return 0.5

    region = arr[y0_px:y1_px, x0_px:x1_px]
    if region.ndim == 3:
        rgb = region[..., :3].astype(np.float32, copy=False)
        if region.dtype == np.uint8:
            rgb = rgb / 255.0
        # Rec. 709 luminance — close enough to perceptual for a
        # binary pick-light-or-dark decision.
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        lum = region.astype(np.float32, copy=False)
        if region.dtype == np.uint8:
            lum = lum / 255.0

    if hdr_white_nits is not None and np.issubdtype(arr.dtype, np.floating):
        # HDR: convert the 90th-percentile linear value to display nits
        # then renormalise so the decision flips at SDR-label brightness
        # (~500 nits on Apple HDR displays — the OS-chosen reference for
        # CSS/SVG colours composited over a gain-map layer). Above-SDR-
        # white linear values stay unclipped so an HDR highlight properly
        # biases the pick toward a dark label.
        SDR_REF_NITS = 500.0
        p90_nits = float(np.percentile(lum, 90)) * float(hdr_white_nits)
        return p90_nits / (2.0 * SDR_REF_NITS)

    # SDR: clamp so an accidental out-of-range value doesn't bias the
    # percentile. 90th percentile is robust to dark outliers but still
    # weights bright-under-label content correctly.
    lum = np.clip(lum, 0.0, 1.0)
    return float(np.percentile(lum, 90))


def _per_letter_picks(arr, lpos, fontsize_uu, w_vb, h_vb, label_text,
                       *, hdr_white_nits=None):
    """Return ``[(sdr_pick, hdr_pick), ...]`` — one fill-colour pair per
    character in ``label_text``, picked from the 90th-percentile
    luminance under that glyph's slice of the label region.

    Each character's footprint is approximated as an equal-width slice
    of the total label bbox (``label_w_vb / n`` per character). For
    proportional fonts this is off by a small amount per glyph, but the
    visual win — every letter contrasts its own background — is robust
    to the slop because the percentile is taken over a multi-pixel
    region that includes neighbouring content anyway.

    Whitespace returns the ``LIGHT`` fallback (fill is invisible regardless).
    """
    n = len(label_text)
    fallback = (_LIGHT_TEXT, _LIGHT_TEXT)
    if n == 0 or arr is None or arr.size == 0:
        return [fallback] * max(n, 0)
    ah, aw = arr.shape[:2]
    if ah == 0 or aw == 0:
        return [fallback] * n

    label_w_vb = min(0.6 * fontsize_uu * n, w_vb)
    label_h_vb = fontsize_uu * 1.2
    pad = 3.0
    if lpos == 'top_middle':
        x0_vb = (w_vb - label_w_vb) / 2
        y0_vb = pad
    elif lpos == 'top_left':
        x0_vb = pad
        y0_vb = pad
    elif lpos == 'bottom_middle':
        x0_vb = (w_vb - label_w_vb) / 2
        y0_vb = h_vb - pad - label_h_vb
    elif lpos == 'bottom_left':
        x0_vb = pad
        y0_vb = h_vb - pad - label_h_vb
    else:
        return [fallback] * n

    y0_px = max(0, int(y0_vb / h_vb * ah))
    y1_px = min(ah, int(np.ceil((y0_vb + label_h_vb) / h_vb * ah)))

    SDR_REF_NITS = 500.0
    use_hdr = (
        hdr_white_nits is not None
        and np.issubdtype(arr.dtype, np.floating)
    )

    picks = []
    for i, ch in enumerate(label_text):
        if ch.isspace():
            picks.append(fallback)
            continue
        cx0_vb = x0_vb + i * label_w_vb / n
        cx1_vb = x0_vb + (i + 1) * label_w_vb / n
        x0_px = max(0, int(cx0_vb / w_vb * aw))
        x1_px = min(aw, int(np.ceil(cx1_vb / w_vb * aw)))
        if x1_px <= x0_px or y1_px <= y0_px:
            picks.append(fallback)
            continue
        region = arr[y0_px:y1_px, x0_px:x1_px]
        if region.ndim == 3:
            rgb = region[..., :3].astype(np.float32, copy=False)
            if region.dtype == np.uint8:
                rgb = rgb / 255.0
            lum = (
                0.2126 * rgb[..., 0]
                + 0.7152 * rgb[..., 1]
                + 0.0722 * rgb[..., 2]
            )
        else:
            lum = region.astype(np.float32, copy=False)
            if region.dtype == np.uint8:
                lum = lum / 255.0

        sdr_p90 = float(np.percentile(np.clip(lum, 0.0, 1.0), 90))
        sdr_p = _DARK_TEXT if sdr_p90 > 0.5 else _LIGHT_TEXT

        if use_hdr:
            p90_nits = float(np.percentile(lum, 90)) * float(hdr_white_nits)
            hdr_p = (
                _DARK_TEXT
                if (p90_nits / (2.0 * SDR_REF_NITS)) > 0.5
                else _LIGHT_TEXT
            )
        else:
            hdr_p = sdr_p
        picks.append((sdr_p, hdr_p))
    return picks


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
    # In-memory render (scene._rgb, cached by make_rgb_scenes / make_rgb) is the
    # source of truth and the CURRENT size — prefer it over the on-disk header
    # peek, whose size gets cached on the Scene (``_rgb_jxl_size``) and goes
    # stale when the file is re-saved at a different resolution.
    _rgb_mem = getattr(item, "_rgb", None)
    if _rgb_mem is not None and getattr(_rgb_mem, "ndim", 0) >= 2:
        return (int(_rgb_mem.shape[0]), int(_rgb_mem.shape[1]))
    # On-disk header peek, cached on the item but mtime-invalidated, so a
    # re-saved RGB (new resolution) re-peeks instead of returning a stale size.
    from ..io.figure_server import _peek_jxl_size_cached
    size = _peek_jxl_size_cached(item)
    if size is not None:
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


def _is_discrete_color(arr, *, max_colors=64, sample=8192):
    """Heuristic: does ``arr`` hold a small palette of discrete colors?

    Used to route float tiles that are really categorical label maps
    (e.g. ``apply_ncolor`` output, which returns float RGBA in [0, 1])
    to lossless PNG instead of lossy uhdr. Quantizes to 8-bit and counts
    unique colors over a strided sample.

    The threshold sits below 256 on purpose: an n-colored label map uses
    a tiny palette (~4-20 colors), while a continuous *grayscale* tile
    spans up to 256 distinct levels — counting unique RGB rows can't tell
    those apart at 256, but 64 cleanly separates the label palette from
    any real gradient.
    """
    # Stride-sample FIRST (a cheap view), then quantize only the sample —
    # never touch every pixel of a large tile just to peek at the palette.
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1, 1)
    if flat.shape[0] > sample:
        flat = flat[:: flat.shape[0] // sample]
    if not (hasattr(flat, 'dtype') and flat.dtype == np.uint8):
        flat = (np.clip(flat, 0.0, 1.0) * 255).astype(np.uint8)
    return np.unique(np.ascontiguousarray(flat), axis=0).shape[0] <= max_colors


def _label_tile_svg(lt, x, y, w, h, title=None, fontsize_uu=0.0, lpos='top_middle'):
    """Build a ``<foreignObject>`` holding a live GPU segmentation tile.

    The foreignObject wraps a relative ``<div>`` containing (a) a
    ``<canvas data-label-tile>`` absolutely positioned to fill the cell and
    (b) — when ``title`` is given — a normal-flow HTML title ``<div>`` laid
    over it. The title MUST be HTML inside the same foreignObject: an SVG
    ``<text>`` placed after a foreignObject is parsed in the leaked HTML
    context (no ``getBBox``, fill forced to ``currentColor`` = black) and
    renders invisible.

    The canvas carries, as ``data-`` attrs (uint16 LE, base64):
      * ``data-labels``    — COLOR/group ids (palette lookup)
      * ``data-instances`` — original per-cell ids for hover/picking +
        outline boundaries (omitted when identical to ``data-labels``)
    plus the palette + display options. The figure shell's controller
    instantiates a LabelGLRenderer over it. Live-only — a canvas isn't in
    the static SVG.
    """
    import base64

    labels = np.asarray(lt['labels'])
    lh, lw = labels.shape[:2]
    # Large matrices are STREAMED (``labels_src`` set by the caller): emit a
    # ``data-labels-src`` URL and omit the multi-MB inline base64; the figure
    # shell's controller fetches the raw uint16 bytes. Small matrices stay
    # inline (self-contained, no fetch).
    _lbl_src = lt.get('labels_src')
    if _lbl_src:
        lbl_data_attr = f' data-labels-src="{_lbl_src}"'
    else:
        lbl_b64 = base64.b64encode(
            np.ascontiguousarray(labels.astype('<u2')).tobytes()).decode('ascii')
        lbl_data_attr = f' data-labels="{lbl_b64}"'

    # Instance ids for hover/picking. Only shipped when they differ from the
    # color ids (i.e. ncolor relabel happened); otherwise the color ids
    # double as instances on the GPU side, saving the extra payload. Clamp
    # to 16-bit (renumber if a crop somehow exceeds 65535 cells).
    inst_attr = ''
    _inst_src = lt.get('instances_src')
    inst = lt.get('instances')
    if _inst_src:
        inst_attr = f' data-instances-src="{_inst_src}"'
    elif not _lbl_src and inst is not None:
        # Inline path (streamed labels never carry inline instances).
        inst = np.asarray(inst)
        if not np.array_equal(inst, labels):
            if int(inst.max(initial=0)) > 65535:
                import fastremap
                inst = fastremap.renumber(inst, in_place=False)[0]
            inst_b64 = base64.b64encode(
                np.ascontiguousarray(inst.astype('<u2')).tobytes()).decode('ascii')
            inst_attr = f' data-instances="{inst_b64}"'

    pal = lt.get('palette')
    if pal is None or (isinstance(pal, str) and pal == 'sinebow'):
        pal_attr = 'sinebow'
    else:
        pal = np.asarray(pal)
        if pal.dtype != np.uint8:
            pal = (np.clip(pal, 0, 1) * 255).astype(np.uint8)
        if pal.shape[-1] == 3:                       # RGB → RGBA
            a = np.full(pal.shape[:-1] + (1,), 255, np.uint8)
            pal = np.concatenate([pal, a], axis=-1)
        pal_attr = base64.b64encode(
            np.ascontiguousarray(pal).tobytes()).decode('ascii')

    # The base image (max projection etc.) is NOT handled here: the caller
    # paints it as a plain <image> behind this (transparent) canvas using the
    # normal registered + HDR-encoded raster URL, so a float base keeps its
    # gain-map HDR. The GPU canvas only draws labels/outlines.

    # Uniform outline color (#rrggbb) — overrides per-cell palette on edges.
    oc_attr = ''
    oc = lt.get('outline_color')
    if oc is not None:
        import matplotlib.colors as _mcolors
        oc_attr = f' data-outline-color="{_mcolors.to_hex(oc)}"'

    # ``data-title`` lets the popup/expanded viewer label the tile (it can't
    # read the GPU canvas; see io/figure.py openZoom). The visible inline
    # title is a normal SVG <text> emitted by the CALLER right after this
    # foreignObject — see the comment there. (An HTML title <div> inside the
    # foreignObject is NOT used: a non-replaced child of a foreignObject
    # collapses to a zero-size box in Blink when the figure SVG is sized via
    # viewBox + CSS height:auto, so it never paints. SVG <text> always lays
    # out and, placed after the foreignObject, paints on top.)
    from html import escape as _html_escape2
    title_attr = f' data-title="{_html_escape2(str(title))}"' if title else ''

    return (
        f'<foreignObject x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}">'
        f'<canvas xmlns="http://www.w3.org/1999/xhtml" data-label-tile="1" '
        f'data-w="{lw}" data-h="{lh}"{title_attr}'
        f'{lbl_data_attr}{inst_attr} data-palette="{pal_attr}" '
        f'data-opacity="{lt["opacity"]:.3f}" data-style="{lt["style"]}" '
        f'data-outlines="{1 if lt["outlines"] else 0}" '
        f'data-outline-hdr="{lt.get("outline_hdr", 1.0):.3f}"{oc_attr} '
        f'style="width:100%;height:100%;image-rendering:pixelated;'
        f'display:block"></canvas>'
        f'</foreignObject>'
    )


def _composite_overlay(base, overlay_rgba):
    """Alpha-composite an ``(H, W, 4)`` uint8 overlay onto ``base``.

    Returns an array the same dtype/range as ``base`` so the combined
    tile flows through the normal codec routing (float → lossless uhdr,
    uint8 → PNG). The overlay's RGB is mapped onto the base's scale: for
    a float base the 8-bit color is divided by 255 (so an opaque red
    outline lands at ``1.0`` — the top of the base's range — and the
    gain map lifts it to HDR brightness alongside everything else); for a
    uint8 base it's used directly. ``out = base*(1-a) + color*a``.
    """
    base = np.asarray(base)
    ov = np.asarray(overlay_rgba)
    if ov.ndim != 3 or ov.shape[-1] != 4:
        raise ValueError(
            f"overlay must be (H, W, 4) RGBA; got {ov.shape}")
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)
    if base.shape[:2] != ov.shape[:2]:
        raise ValueError(
            f"overlay shape {ov.shape[:2]} must match base {base.shape[:2]}")

    alpha = (ov[..., 3:4].astype(np.float32)) / 255.0
    is_float = np.issubdtype(base.dtype, np.floating)
    base_f = base[..., :3].astype(np.float32)
    color = ov[..., :3].astype(np.float32)
    if is_float:
        color = color / 255.0          # 8-bit color → base's [0,1]+ scale
    out = base_f * (1.0 - alpha) + color * alpha
    if is_float:
        return out.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _encode_thumb_url(arr, raster_format, sdr_white_nits):
    """Encode a per-tile array into a ``data:image/...;base64,...`` URL.

    Mirrors :meth:`SVG.image` for the ``jxl-hdr-pq`` / ``uhdr`` / ``png``
    paths but returns the URL string directly so callers
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

    if raster_format == 'png':
        # Lossless SDR tile (segmentation / outline overlay) in a mixed
        # HDR grid. No gain map → no HDR boost → identical brightness to
        # the matching PNG hi-res (no hover-dim), and bit-exact discrete
        # colors / thin lines (no JPEG ringing). Tiny on flat content.
        import base64
        import imagecodecs
        a = arr if (hasattr(arr, 'dtype') and arr.dtype == np.uint8) \
            else (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[..., :3]
        png = imagecodecs.png_encode(np.ascontiguousarray(a))
        return "data:image/png;base64," + base64.b64encode(png).decode('ascii')

    from .svg import (jxl_data_url, uhdr_data_url, _p3_linear_to_pq_uint16,
                      _linear_p3_to_uint8_srgb_peaknorm,
                      _srgb_uint8_to_p3_linear)
    import opencodecs

    if raster_format == 'jxl-hdr-pq':
        arr_pq = _p3_linear_to_pq_uint16(arr, sdr_white_nits=sdr_white_nits)
        p3_pq = opencodecs.ColorSpec(
            primaries=11, transfer=16, white_point=1,
            rendering_intent=1, gamma=0.0,
        )
        return jxl_data_url(arr_pq, color=p3_pq, intensity_target=10000.0)
    if raster_format in ('uhdr', 'uhdr-lossless'):
        # Ultra-HDR JPEG: cross-browser HDR (Safari + Chrome composite,
        # Firefox sees the SDR base). For uint8 cells in a mixed grid,
        # round-trip through linear-P3 so the gain map still encodes
        # (otherwise the SDR base + HDR are identical and no gain rides
        # on top — the file is still valid, just SDR-equivalent).
        # ``uhdr-lossless`` is a baked layered tile: the SDR base is
        # encoded losslessly so the composited outline stays bit-exact
        # while the gain map still lifts it to HDR brightness.
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
        return uhdr_data_url(hdr, sdr_u8, sdr_white_nits=sdr_white_nits,
                             lossless=(raster_format == 'uhdr-lossless'))
    # Fallback (unrecognized format): lossless PNG. Universally decodable
    # — we don't emit untagged JXL, which Chrome/Electron won't render
    # without the experimental flag.
    import base64
    import imagecodecs
    a = arr if (hasattr(arr, 'dtype') and arr.dtype == np.uint8) \
        else (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    png = imagecodecs.png_encode(np.ascontiguousarray(a))
    return "data:image/png;base64," + base64.b64encode(png).decode('ascii')


def _resolve_items(items, *, dx, target_px=None, hdr_unify=False):
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
                from ..io.figure_server import _peek_jxl_size_cached
                src_size = (_peek_jxl_size_cached(it)
                            or _native_dims(it, np.zeros((1, 1, 3))))
                src_longest = max(int(src_size[0]), int(src_size[1]))
                ds = max(1, src_longest // int(target_px))
            else:
                ds = 4
            # hdr_unify routes scene cells through the data-hdr/WebGPU path, which
            # needs REAL linear-P3 pixels — so try resolve_linear_p3 first and SKIP
            # the embedded-uhdr-thumb fast-path (it returns a zeros placeholder for
            # the native <image>). If the scene can't decode to linear, fall back to
            # the native embedded thumb so it still renders (just not unified).
            arr = resolve_linear_p3(it, target_px=target_px) if hdr_unify else None
            if arr is None:
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
    # Two codecs, picked per cell by dtype — float vs uint8:
    #   * float (HDR) → Ultra-HDR JPEG (ISO 21496-1): cross-browser HDR
    #     (Safari + Chrome composite the gain map, Firefox / Preview fall
    #     back to the SDR base cleanly). ``raster_format='jxl-hdr-pq'``
    #     forces JXL-PQ for the explicit Safari-native-HDR opt-in.
    #   * uint8 (SDR) → lossless PNG: bit-exact discrete label colors and
    #     thin outlines (no JPEG ringing), no gain map (so the inline
    #     thumb and the hi-res render at identical brightness — no
    #     hover-dim), decodes in every browser, and tiny on flat content.
    # PNG replaces the old ``jxl-p3`` SDR path: JXL needs an experimental
    # flag in Chrome/Electron (VS Code's renderer), so it never reliably
    # displayed there — not worth the extra encode branch for a format we
    # don't ship.
    raster_format = 'uhdr' if any_hdr else 'png'

    def _fmt_for(a):
        if not np.issubdtype(a.dtype, np.floating):
            return 'png'          # uint8 SDR → lossless PNG
        # Float tiles default to uhdr, EXCEPT discrete-color content with
        # no HDR headroom (e.g. ``apply_ncolor`` label maps, which return
        # float RGBA in [0, 1]). Lossy uhdr would ring at the hard color
        # boundaries; route those to lossless PNG instead. A continuous
        # SDR-range projection still goes to uhdr so the encoder can lift
        # it to HDR linear light.
        hi = float(np.nanmax(a)) if a.size else 0.0
        if hi <= 1.0 + 1e-6 and _is_discrete_color(a):
            return 'png'
        return raster_format

    cell_fmts = [_fmt_for(a) for a in arrays]
    return arrays, raster_format, cell_fmts


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
