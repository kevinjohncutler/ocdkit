"""Common imports for ocdkit.plot subpackage.

Note: ``matplotlib.figure.Figure`` and ``FigureCanvasAgg`` are deliberately
NOT re-exported here.  Callers that need them should import explicitly::

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

Reason: ``ocdkit.io.figure.Figure`` (the SVG-backed figure abstraction)
will land in this package soon — re-exporting matplotlib's ``Figure``
under the same name would create a name clash.  Keeping the matplotlib
import path explicit at call sites makes the two unambiguous.
"""
from ..imports import *

import matplotlib as mpl
import matplotlib.patches as mpatches

__all__ = ['np', 'torch', 'da', 'mpl', 'mpatches']
