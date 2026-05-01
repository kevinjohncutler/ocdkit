"""Tests for ocdkit.plot.figure and ocdkit.plot.defaults."""

import matplotlib
import matplotlib as mpl
from matplotlib.figure import Figure

from ocdkit.plot import figure, split_list
from ocdkit.plot.defaults import apply_mpl_defaults


class TestFigure:
    def test_single_ax(self):
        fig, ax = figure()
        assert isinstance(fig, Figure)
        assert hasattr(ax, 'imshow')

    def test_grid(self):
        fig, axs = figure(nrow=2, ncol=3)
        assert isinstance(fig, Figure)
        assert len(axs) == 6

    def test_figsize_scalar(self):
        fig, ax = figure(figsize=4)
        w, h = fig.get_size_inches()
        assert h == 4

    def test_figsize_tuple(self):
        fig, ax = figure(figsize=(6, 3))
        w, h = fig.get_size_inches()
        assert w == 6
        assert h == 3

    def test_aspect(self):
        fig, ax = figure(figsize=2, aspect=2)
        w, h = fig.get_size_inches()
        assert w == 4
        assert h == 2


class TestSplitList:
    def test_even(self):
        assert split_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven(self):
        assert split_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_empty(self):
        assert split_list([], 3) == []


class TestApplyMplDefaults:
    def test_sets_rcparams(self):
        apply_mpl_defaults()
        assert mpl.rcParams['svg.fonttype'] == 'none'
        assert mpl.rcParams['text.usetex'] is False
