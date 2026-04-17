"""Tests for ocdkit.plot.grid — image_grid layout and rendering."""

import numpy as np
from matplotlib.figure import Figure

from ocdkit.plot.grid import image_grid


class TestImageGrid:
    def _rand_imgs(self, nrow, ncol, h=8, w=8):
        """Create a grid of random RGB images."""
        return [[np.random.rand(h, w, 3).astype(np.float32)
                 for _ in range(ncol)] for _ in range(nrow)]

    def test_basic_grid(self):
        imgs = self._rand_imgs(2, 3)
        fig = image_grid(imgs)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 6

    def test_single_row(self):
        imgs = self._rand_imgs(1, 4)
        fig = image_grid(imgs)
        assert len(fig.get_axes()) == 4

    def test_single_image(self):
        imgs = self._rand_imgs(1, 1)
        fig = image_grid(imgs)
        assert len(fig.get_axes()) == 1

    def test_column_titles(self):
        imgs = self._rand_imgs(2, 2)
        fig = image_grid(imgs, column_titles=['A', 'B'])
        assert isinstance(fig, Figure)

    def test_row_titles(self):
        imgs = self._rand_imgs(2, 2)
        fig = image_grid(imgs, row_titles=['Row1', 'Row2'])
        assert isinstance(fig, Figure)

    def test_plot_labels(self):
        imgs = self._rand_imgs(1, 3)
        labels = [['a', 'b', 'c']]
        fig = image_grid(imgs, plot_labels=labels)
        assert isinstance(fig, Figure)

    def test_return_axes(self):
        imgs = self._rand_imgs(2, 2)
        fig, axes, pos = image_grid(imgs, return_axes=True)
        assert isinstance(fig, Figure)
        assert len(axes) == 4

    def test_xy_order(self):
        imgs = self._rand_imgs(2, 3)
        fig = image_grid(imgs, order='xy')
        assert isinstance(fig, Figure)

    def test_outline(self):
        imgs = self._rand_imgs(1, 2)
        fig = image_grid(imgs, outline=True, outline_color=[1, 0, 0])
        assert isinstance(fig, Figure)

    def test_ragged_grid(self):
        imgs = [[np.random.rand(8, 8, 3).astype(np.float32) for _ in range(3)],
                [np.random.rand(8, 8, 3).astype(np.float32) for _ in range(2)]]
        fig = image_grid(imgs)
        assert isinstance(fig, Figure)
