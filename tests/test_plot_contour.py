"""Tests for ocdkit.plot.contour — vector contour rendering."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from ocdkit.plot.contour import vector_contours


class TestVectorContours:
    def _make_mask(self):
        """Simple 2-cell mask."""
        mask = np.zeros((32, 32), dtype=np.int32)
        mask[5:15, 5:15] = 1
        mask[18:28, 18:28] = 2
        return mask

    def test_basic(self):
        mask = self._make_mask()
        fig = Figure(figsize=(4, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(mask)
        vector_contours(fig, ax, mask)
        # Should have added patch collections
        assert len(ax.collections) > 0

    def test_single_cell(self):
        mask = np.zeros((16, 16), dtype=np.int32)
        mask[4:12, 4:12] = 1
        fig = Figure(figsize=(4, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(mask)
        vector_contours(fig, ax, mask)
        assert len(ax.collections) > 0

    def test_multiple_axes(self):
        mask = self._make_mask()
        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(mask)
        ax2.imshow(mask)
        vector_contours(fig, [ax1, ax2], mask)
        assert len(ax1.collections) > 0
        assert len(ax2.collections) > 0
