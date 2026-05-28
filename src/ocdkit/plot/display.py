"""Generic image display helpers (axes chrome, multi-image grids)."""

import inspect

from .imports import *

from .figure import figure
from .color import colorize
from ..array.normalize import normalize99
from ..utils.kwargs import split_kwargs


_COLORIZE_PARAMS = frozenset(inspect.signature(colorize).parameters) - {"im"}


# Kwargs that ``image_grid`` accepts beyond what ``imshow``'s legacy
# matplotlib signature already exposes — passed through verbatim when
# present.  Excludes overlapping names (``figsize``, ``dpi``, etc.) that
# get translated explicitly in ``_imshow_svg``.
_IMAGE_GRID_PASSTHROUGH = frozenset({
    'target_tile_px', 'gap_px', 'margin_px', 'dx', 'lpos',
    'facecolor', 'raster_format', 'sdr_white_nits', 'ncol',
    'popup_viewer', 'link_axes', 'roi', 'link_axes_debug',
})


def _to_css_color(c):
    """Accept matplotlib-style RGB tuple or CSS string; return CSS string."""
    if isinstance(c, str):
        return c
    from matplotlib.colors import to_hex
    return to_hex(c)


def _coerce_to_rgb(arr):
    """Reshape any input array to ``(H, W, 3|4)`` for ``image_grid``,
    *without* losing precision.

    - uint8 RGB(A) pass through (the SDR JXL path expects uint8).
    - float RGB(A) pass through as float32 — image_grid will pick the HDR
      P3-PQ encoder. Floats outside [0, 1] get min/max normalized to
      [0, 1] but stay float.
    - Integer dtypes (uint16, int16, etc.) become float32 in [0, 1].
    - 2D grayscale broadcasts to (H, W, 3) preserving dtype.
    - (H, W, 1) is squeezed to 2D first.

    Bit-depth-lossy quantization to uint8 only happens when the caller
    handed us uint8 to begin with. Everything else flows to image_grid
    at full precision, so HDR ``scene.rgb`` lands in the HDR-PQ encoder.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # 2D grayscale → broadcast to 3-channel, preserve dtype.
    if arr.ndim == 2:
        prepared = _prepare_channel(arr)
        return np.ascontiguousarray(
            np.broadcast_to(prepared[..., None], prepared.shape + (3,))
        )

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.dtype == np.uint8:
            return arr
        return _prepare_channel(arr)

    raise ValueError(
        f"imshow: cannot display array of shape {arr.shape}; expected "
        f"2D, (H, W, 1), (H, W, 3), or (H, W, 4)."
    )


def _prepare_channel(arr):
    """Return ``arr`` as float32. Float [0, 1] passes through; out-of-range
    floats and integer/bool dtypes are min/max normalized to [0, 1]."""
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size == 0:
            return arr.astype(np.float32, copy=False)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if 0.0 <= lo and hi <= 1.0 + 1e-6:
            return arr.astype(np.float32, copy=False)
    else:
        # integer dtypes
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((arr.astype(np.float32) - np.float32(lo))
            / np.float32(hi - lo))


# Back-compat alias — older callers used the old name. Both refer to the
# same precision-preserving coercion now.
_coerce_to_uint8_rgb = _coerce_to_rgb


def _normalize_to_uint8(arr):
    """Map any dtype/range to ``uint8`` in [0, 255].

    Float in [0, 1] is the common matplotlib convention — pass through
    (just ×255).  Anything else: min/max normalize.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.floating):
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if 0.0 <= lo and hi <= 1.0 + 1e-6:
            return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


class _CenteredSvgFigure:
    """Thin shim wrapping an :class:`SvgFigure` to center it in the cell.

    Jupyter renders ``_repr_mimebundle_`` output left-aligned by default.
    Matplotlib inline-PNG output happens to look centered because the
    inline backend wraps the PNG in markup the JupyterLab CSS centers.
    SVG output gets no such centering, so wrap the bundle's HTML in a
    ``text-align: center`` block to match the visual convention users
    expect from ``imshow``.

    Delegates all other attribute access to the underlying figure so
    ``savefig``, ``to_string``, etc. still work on the return value.
    """

    def __init__(self, inner):
        self._inner = inner

    def _repr_mimebundle_(self, include=None, exclude=None):
        bundle = self._inner._repr_mimebundle_(include=include, exclude=exclude)
        if isinstance(bundle, tuple):
            data, metadata = bundle
        else:
            data, metadata = bundle, {}
        if 'text/html' in data:
            data['text/html'] = (
                '<div style="text-align:center">'
                + data['text/html']
                + '</div>'
            )
        elif 'image/svg+xml' in data:
            # Non-interactive path: SVG-only bundle. Wrap in HTML so
            # we can center it; Jupyter prefers HTML over raw SVG.
            data['text/html'] = (
                '<div style="text-align:center">'
                + data['image/svg+xml']
                + '</div>'
            )
        return data, metadata

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _apply_cmap_item(arr, cmap, *, hdr, vmin, vmax):
    """Apply ``cmap`` to a 2D ndarray. HDR path returns ``(H,W,3)
    float32`` linear-P3 (routes to ``jxl-hdr-pq`` downstream); SDR path
    returns ``(H,W,3) uint8`` (routes to ``jxl-p3``).
    """
    if hdr:
        from .hdr_cmap import apply_hdr_cmap
        return apply_hdr_cmap(arr, cmap, vmin=vmin, vmax=vmax)
    from cmap import Colormap
    cm = Colormap(cmap) if isinstance(cmap, str) else cmap
    a = arr.astype(np.float32, copy=False)
    vmin_eff = float(np.nanmin(a)) if vmin is None else float(vmin)
    vmax_eff = float(np.nanmax(a)) if vmax is None else float(vmax)
    if vmax_eff <= vmin_eff:
        vmax_eff = vmin_eff + 1e-9
    t = np.clip((a - vmin_eff) / (vmax_eff - vmin_eff), 0.0, 1.0)
    rgba = np.asarray(cm(t))
    return np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)


def _imshow_svg(imgs, *, figsize, titles, title_size, textcolor,
                outline_color, outline_width, dpi,
                cmap=None, hdr=True, vmin=None, vmax=None, **kwargs):
    """SVG-backend imshow — delegates to :func:`image_grid`.

    Translates the imshow per-panel ``figsize`` (legacy: scalar = width
    *per panel*, total = ``figsize × n_panels``) to image_grid's total-
    grid-width semantics, and remaps color/label kwargs.

    When ``cmap`` is given, 2D ndarray items are run through the
    colormap before reaching the grid: ``hdr=True`` (default) yields a
    float linear-P3 tile that the existing jxl-hdr-pq encoder picks up;
    ``hdr=False`` falls back to a uint8 SDR tile.
    """
    from .image_grid import image_grid

    is_list = isinstance(imgs, (list, tuple))
    raw_items = list(imgs) if is_list else [imgs]

    def _prep(it):
        if not isinstance(it, np.ndarray):
            return it
        if cmap is not None and it.ndim == 2:
            return _apply_cmap_item(it, cmap, hdr=hdr, vmin=vmin, vmax=vmax)
        return _coerce_to_rgb(it)

    items = [_prep(it) for it in raw_items]
    n_panels = len(items)

    if isinstance(figsize, (list, tuple, np.ndarray)):
        target_w = float(figsize[0]) if len(figsize) >= 1 else 2.0
    else:
        target_w = float(figsize) * n_panels

    # imshow's dpi default (100) is tuned for the matplotlib path's PNG
    # size; the SVG dispatch uses dpi as a display-scale multiplier
    # (``target_w_px = figsize × dpi``), so 100 would render a single
    # ``figsize=2`` image at only ~200 px wide.  Floor at 300 so single-
    # image imshow matches the legacy matplotlib ``dpi=300`` appearance.
    effective_dpi = max(float(dpi), 300.0)

    ig_kwargs = dict(
        ncol=n_panels,
        figsize=target_w,
        fontsize=float(title_size),
        fontcolor=_to_css_color(textcolor),
        dpi=effective_dpi,
        # Single-image imshow: trigger the hi-res upgrade on load
        # instead of waiting for hover, since the user always wants
        # the full-res view of an explicitly-shown image.  For wider
        # grids (multi-image) the caller can opt in via image_grid
        # directly.
        auto_upgrade=(n_panels == 1),
    )
    if titles is not None:
        ig_kwargs['plot_labels'] = (
            list(titles)
            if isinstance(titles, (list, tuple, np.ndarray))
            else [titles]
        )
    if outline_color is not None and outline_width > 0:
        ig_kwargs['outline'] = True
        ig_kwargs['outline_color'] = _to_css_color(outline_color)
        ig_kwargs['outline_width'] = float(outline_width)

    for k in _IMAGE_GRID_PASSTHROUGH:
        if k in kwargs:
            ig_kwargs[k] = kwargs.pop(k)

    return _CenteredSvgFigure(image_grid(items, **ig_kwargs))


def set_outline(ax, outline_color=None, outline_width=0):
    """Hide ticks; optionally show colored spines as a border.

    - Always hide axis ticks.
    - If ``outline_color`` is not None and ``outline_width > 0``,
      show spines with that color/width.
    - Otherwise, hide spines entirely.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)

    if outline_color is not None and outline_width > 0:
        for spine in ax.spines.values():
            spine.set_edgecolor(outline_color)
            spine.set_linewidth(outline_width)
    else:
        for s in ax.spines.values():
            s.set_visible(False)


def imshow(
    imgs,
    figsize=2,
    ax=None,
    hold=False,
    titles=None,
    title_size=8,
    spacing=0.05,
    textcolor=(0.5, 0.5, 0.5),
    dpi=100,
    text_scale=1,
    outline_color=None,
    outline_width=0.5,
    show=False,
    *,
    backend='svg',
    cmap=None,
    hdr=True,
    vmin=None,
    vmax=None,
    **kwargs,
):
    """Display one or more images in a horizontal strip.

    Default ``backend='svg'`` routes through :func:`image_grid` — HDR-
    capable, hover-zoom, click-to-expand, ~instant for the common
    ``imshow(uint8_rgb)`` case (the matplotlib path round-trips through
    a libpng encode that's seconds-slow for multi-megapixel images).

    ``ax=`` or ``hold=True`` auto-fall back to the matplotlib path
    (compositing into a caller-owned axes has no SVG equivalent).
    Set ``backend='matplotlib'`` to force the legacy path explicitly.

    Parameters
    ----------
    imgs : array or list of arrays
        Single image or list to render side by side.
    figsize : float or tuple
        Per-panel figsize; scalar scales width by panel count.
    ax : matplotlib axis, optional
        Existing axis to draw into. Implies ``hold=True``. Forces the
        matplotlib backend.
    titles : str or list of str, optional
        Per-panel title.
    outline_color, outline_width
        Optional border around each panel (see :func:`set_outline`).
    backend : {'svg', 'matplotlib'}
        Rendering backend. SVG is default; matplotlib is auto-selected
        when ``ax=`` or ``hold=True`` is passed.
    cmap : str | Colormap, optional
        Apply this colormap to 2D ndarray items before display. Defaults
        to ``None`` (2D arrays broadcast as grayscale).
    hdr : bool, default True
        With ``cmap`` set, lift the colormap into HDR Display-P3 (see
        :mod:`ocdkit.plot.hdr_cmap`) and emit a float linear-P3 tile —
        Safari / Chrome decode the resulting jxl-hdr-pq natively, tone-
        mapping to SDR on non-EDR displays. Set ``False`` to emit a
        uint8 SDR tile instead. Ignored when ``cmap`` is ``None``.
    vmin, vmax : float, optional
        Colormap normalization range; defaults to per-image min/max.

    Notes
    -----
    Imports ``IPython.display.display`` lazily so the function is usable
    outside of Jupyter (in scripts) without forcing the dependency.
    """
    # SVG fast-path — only when the caller isn't compositing into an
    # existing matplotlib axis.  ``hold=True`` historically meant "give
    # me back the Figure"; preserve that by routing to mpl.
    if backend == 'svg' and ax is None and not hold:
        fig = _imshow_svg(
            imgs,
            figsize=figsize,
            titles=titles,
            title_size=title_size,
            textcolor=textcolor,
            outline_color=outline_color,
            outline_width=outline_width,
            dpi=dpi,
            cmap=cmap,
            hdr=hdr,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        # In a Jupyter kernel, multiple ``imshow(...)`` calls in one
        # cell each need to render — without an explicit ``display()``
        # only the cell's last return value auto-displays and earlier
        # calls disappear silently. Match the matplotlib backend's
        # ``display(fig)`` semantics. Return ``None`` to suppress the
        # second auto-display of the return value.
        #
        # In scripts (no kernel), return the figure unchanged so callers
        # can save it / inspect it. We key off ``ipykernel`` being
        # imported — that module loads as part of Jupyter kernel
        # startup but is not pulled in by ``import IPython`` alone, so
        # this distinguishes interactive Jupyter from CLI scripts even
        # when IPython is available.
        import sys
        if 'ipykernel' in sys.modules:
            from IPython.display import display as _ipd
            _ipd(fig)
            return None
        return fig

    from IPython.display import display

    if isinstance(imgs, list):
        if titles is None:
            titles = [None] * len(imgs)
        if title_size is None:
            title_size = figsize / len(imgs) * text_scale

        if isinstance(figsize, (list, tuple, np.ndarray)):
            if len(figsize) >= 2:
                fig_w, fig_h = float(figsize[0]), float(figsize[1])
            elif len(figsize) == 1:
                fig_w = fig_h = float(figsize[0])
            else:
                fig_w = fig_h = 2.0
        else:
            fig_w = float(figsize) * len(imgs)
            fig_h = float(figsize)
        figsize_list = (fig_w, fig_h)

        fig, axes = figure(
            nrow=1, ncol=len(imgs),
            figsize=figsize_list,
            dpi=dpi,
            frameon=False,
            facecolor=[0, 0, 0, 0],
        )

        mpl_imshow_kw = dict(kwargs)
        if cmap is not None:
            mpl_imshow_kw['cmap'] = cmap
        if vmin is not None:
            mpl_imshow_kw['vmin'] = vmin
        if vmax is not None:
            mpl_imshow_kw['vmax'] = vmax

        for this_ax, img, ttl in zip(axes, imgs, titles):
            this_ax.imshow(img, **mpl_imshow_kw)
            set_outline(this_ax, outline_color, outline_width)
            this_ax.set_facecolor([0, 0, 0, 0])

            if ttl is not None:
                this_ax.set_title(ttl, fontsize=title_size, color=textcolor)

    else:
        if not isinstance(figsize, (list, tuple, np.ndarray)):
            figsize = (figsize, figsize)
        elif len(figsize) == 2:
            figsize = (figsize[0], figsize[1])
        else:
            figsize = (figsize[0], figsize[0])
        if title_size is None:
            title_size = figsize[0] * text_scale

        if ax is None:
            subplot_args = {
                'frameon': False,
                'figsize': figsize,
                'facecolor': [0, 0, 0, 0],
                'dpi': dpi,
            }
            fig, ax = figure(**subplot_args)
        else:
            hold = True
            fig = ax.get_figure()

        mpl_imshow_kw = dict(kwargs)
        if cmap is not None:
            mpl_imshow_kw['cmap'] = cmap
        if vmin is not None:
            mpl_imshow_kw['vmin'] = vmin
        if vmax is not None:
            mpl_imshow_kw['vmax'] = vmax

        ax.imshow(imgs, **mpl_imshow_kw)
        set_outline(ax, outline_color, outline_width)
        ax.set_facecolor([0, 0, 0, 0])

        if titles is not None:
            ax.set_title(titles, fontsize=title_size, color=textcolor)

    if not hold:
        display(fig)
    else:
        return fig


def outline_view(img, masks, *, boundaries=None, color=(1, 0, 0),
                 mode="inner", connectivity=2, channel_axis=-1, **kwargs):
    """Stamp label-mask boundaries as a flat color over an image.

    Channel handling is automatic:

    - 2D (or N=1) → tiled to RGB as grayscale
    - N=3         → assumed to already be RGB (unless a :func:`colorize`
      kwarg like ``colors`` is passed, in which case it is composited)
    - other N     → composited via :func:`colorize`

    Any keyword argument matching :func:`colorize`'s signature
    (``colors``, ``color_weights``, ``intervals``, ``offset``) is routed
    to colorize via :func:`split_kwargs`.

    Parameters
    ----------
    img : array
        2D grayscale or 3D with channels along ``channel_axis``.
    masks : array
        Integer label image (same spatial shape as ``img``).
    boundaries : array, optional
        Precomputed boolean boundary mask. If ``None``, computed via
        :func:`skimage.segmentation.find_boundaries`.
    color : tuple of 3 floats
        Outline color. Floats in ``[0, 1]`` are scaled to ``[0, 255]``;
        any value above 1 is taken as already in 8-bit range.
    mode, connectivity
        Forwarded to :func:`find_boundaries`.
    channel_axis : int
        Channel axis for multi-channel input. Default ``-1``.
    """
    from skimage.segmentation import find_boundaries

    user_colorize_kw = bool(_COLORIZE_PARAMS & kwargs.keys())
    nchan = 1 if img.ndim == 2 else img.shape[channel_axis]

    if not user_colorize_kw and nchan == 3:
        rgb = img if channel_axis in (-1, img.ndim - 1) else np.moveaxis(img, channel_axis, -1)
    elif not user_colorize_kw and nchan == 1:
        src = img if img.ndim == 2 else img.squeeze(axis=channel_axis)
        rgb = np.stack([src] * 3, axis=-1)
    else:
        if img.ndim == 2:
            im_first = img[None]
        elif channel_axis != 0:
            im_first = np.moveaxis(img, channel_axis, 0)
        else:
            im_first = img
        rgb = colorize(im_first, **split_kwargs([colorize], kwargs, strict=False))

    if rgb.dtype != np.uint8:
        rgb = (np.clip(normalize99(rgb), 0, 1) * 255).astype(np.uint8)

    color_arr = np.asarray(color, dtype=np.float64)
    if color_arr.max() <= 1:
        color_arr = color_arr * 255
    color_arr = color_arr.astype(np.uint8)

    if boundaries is None:
        boundaries = find_boundaries(masks, mode=mode, connectivity=connectivity)

    out = rgb.copy()
    out[np.asarray(boundaries, dtype=bool)] = color_arr
    return out
