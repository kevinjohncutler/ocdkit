"""N-color label visualization."""

import inspect

import numpy as np
import matplotlib as mpl
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
