"""Generic image display helpers (axes chrome, multi-image grids)."""

import inspect

from .imports import *

from .figure import figure
from .color import colorize
from ..array.normalize import normalize99
from ..utils.kwargs import split_kwargs


_COLORIZE_PARAMS = frozenset(inspect.signature(colorize).parameters) - {"im"}


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
    dpi=300,
    text_scale=1,
    outline_color=None,
    outline_width=0.5,
    show=False,
    **kwargs,
):
    """Display one or more images in a horizontal strip.

    Parameters
    ----------
    imgs : array or list of arrays
        Single image or list to render side by side.
    figsize : float or tuple
        Per-panel figsize; scalar scales width by panel count.
    ax : matplotlib axis, optional
        Existing axis to draw into. Implies ``hold=True``.
    titles : str or list of str, optional
        Per-panel title.
    outline_color, outline_width
        Optional border around each panel (see :func:`set_outline`).

    Notes
    -----
    Imports ``IPython.display.display`` lazily so the function is usable
    outside of Jupyter (in scripts) without forcing the dependency.
    """
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

        for this_ax, img, ttl in zip(axes, imgs, titles):
            this_ax.imshow(img, **kwargs)
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

        ax.imshow(imgs, **kwargs)
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
