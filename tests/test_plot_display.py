"""Tests for ocdkit.plot.display — set_outline (no IPython needed)."""

import numpy as np
from matplotlib.figure import Figure

from ocdkit.plot.display import set_outline


class TestSetOutline:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(111)

    def test_hides_ticks(self):
        ax = self._make_ax()
        set_outline(ax)
        assert ax.get_xticks().tolist() == []
        assert ax.get_yticks().tolist() == []

    def test_hides_spines_by_default(self):
        ax = self._make_ax()
        set_outline(ax)
        for spine in ax.spines.values():
            assert not spine.get_visible()

    def test_shows_colored_spines(self):
        ax = self._make_ax()
        set_outline(ax, outline_color='red', outline_width=2)
        for spine in ax.spines.values():
            assert spine.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)  # red
            assert spine.get_linewidth() == 2

    def test_zero_width_hides(self):
        ax = self._make_ax()
        set_outline(ax, outline_color='blue', outline_width=0)
        for spine in ax.spines.values():
            assert not spine.get_visible()
