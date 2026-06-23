"""N-color label visualization."""

import inspect
import hashlib as _hashlib
import threading as _threading

from .imports import *
import ncolor
from cmap import Colormap

from ..array import rescale


def sinebow(N,bg_color=[0,0,0,0], offset=0):
    """ Generate a color dictionary for use in visualizing N-colored labels. Background color
    defaults to transparent black.

    Parameters
    ----------
    N: int
        number of distinct colors to generate (excluding background)

    bg_color: ndarray, list, or tuple of length 4
        RGBA values specifying the background color at the front of the  dictionary.

    Returns
    --------------
    Dictionary with entries {int:RGBA array} to map integer labels to RGBA colors.

    """
    colordict = {0:bg_color}
    for j in range(N):
        k = j+offset
        angle = k*2*np.pi / (N)
        r = ((np.cos(angle)+1)/2)
        g = ((np.cos(angle+2*np.pi/3)+1)/2)
        b = ((np.cos(angle+4*np.pi/3)+1)/2)
        colordict.update({j+1:[r,g,b,1]})
    return colordict


_NCP_CACHE = {}                 # content hash -> (m, palette)
_NCP_LOCK = _threading.Lock()
_NCP_MAX = 32                   # bounded; evict oldest


def ncolor_labels_and_palette(masks, offset=0, max_depth=20, expand=True, greedy=False):
    """N-color relabel + sinebow palette for GPU label tiles.

    Returns ``(m, palette)`` where ``m`` is the masks relabeled to a small
    set of adjacent-distinct group ids (uint16, background 0) via
    :func:`ncolor.label`, and ``palette`` is an ``(n+1, 4)`` uint8 RGBA LUT
    (index 0 = transparent background) from :func:`sinebow`. Indexing the
    palette by ``m`` reproduces :func:`apply_ncolor`'s coloring exactly,
    but as data the GPU ``LabelGLRenderer`` can render directly — so the
    nice ncolor look (adjacent cells differ) survives the live path.

    Result is cached by masks CONTENT (not object id) so re-running an
    ``imshow``/``image_grid`` on the same segmentation — the common
    interactive-iteration case — skips the ~100 ms ``ncolor.label`` entirely
    (blake2b of the array is ~1000× cheaper than the relabel).
    """
    a = np.ascontiguousarray(masks)
    key = (_hashlib.blake2b(a.tobytes(), digest_size=16).digest(),
           a.shape, int(offset), int(max_depth), bool(expand), bool(greedy))
    with _NCP_LOCK:
        hit = _NCP_CACHE.get(key)
    if hit is not None:
        return hit
    kwargs = dict(max_depth=max_depth, return_n=True, conn=2, expand=expand)
    if 'greedy' in inspect.signature(ncolor.label).parameters:
        kwargs['greedy'] = greedy
    m, n = ncolor.label(masks, **kwargs)
    colors = np.array(list(sinebow(n, offset=offset).values()), dtype=np.float64)
    if colors.shape[-1] == 3:
        colors = np.concatenate(
            [colors, np.ones(colors.shape[:-1] + (1,))], axis=-1)
    palette = np.clip(colors, 0.0, 1.0)
    palette = (palette * 255.0 + 0.5).astype(np.uint8)
    res = (np.ascontiguousarray(m).astype(np.uint16), palette)
    with _NCP_LOCK:
        _NCP_CACHE[key] = res
        if len(_NCP_CACHE) > _NCP_MAX:
            _NCP_CACHE.pop(next(iter(_NCP_CACHE)))
    return res


def apply_ncolor(masks,offset=0,cmap=None,max_depth=20,expand=True, maxv=1, greedy=False):
    cmap = Colormap(cmap) if isinstance(cmap, str) else cmap

    kwargs = dict(max_depth=max_depth, return_n=True, conn=2, expand=expand)
    # greedy was removed in newer ncolor versions
    if 'greedy' in inspect.signature(ncolor.label).parameters:
        kwargs['greedy'] = greedy
    m,n = ncolor.label(masks, **kwargs)
    if cmap is None:
        c = sinebow(n,offset=offset)
        colors = np.array(list(c.values()))
        cmap = mpl.colors.ListedColormap(colors)
        return cmap(m)
    else:
        return cmap(rescale(m)/maxv)
