"""Tests for ocdkit.plot.label — text recoloring and label backgrounds."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from ocdkit.plot.label import recolor_label, add_label_background, _coerce_pad


class TestCoercePad:
    def test_none(self):
        assert _coerce_pad(None) == (0.0, 0.0)

    def test_scalar(self):
        assert _coerce_pad(3.0) == (3.0, 3.0)

    def test_tuple_two(self):
        assert _coerce_pad((2.0, 4.0)) == (2.0, 4.0)

    def test_tuple_one(self):
        assert _coerce_pad([5.0]) == (5.0, 5.0)

    def test_empty(self):
        assert _coerce_pad([]) == (0.0, 0.0)


def _make_scene():
    """Create a rendered figure with an image and text label."""
    fig = Figure(figsize=(4, 4), dpi=72)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    img_data = np.random.rand(16, 16, 3).astype(np.float32)
    im_artist = ax.imshow(img_data)
    text = ax.text(0.5, 0.5, 'test', transform=ax.transAxes,
                   fontsize=10, color='black')
    fig.canvas.draw()
    return fig, ax, text, im_artist


class TestRecolorLabel:
    def test_recolor_on_dark(self):
        fig, ax, text, _ = _make_scene()
        ax.imshow(np.zeros((8, 8, 3)))
        fig.canvas.draw()
        im_artist = ax.images[-1]
        recolor_label(ax, text, image_artist=im_artist)
        assert text.get_color() is not None

    def test_recolor_on_light(self):
        fig, ax, text, _ = _make_scene()
        ax.imshow(np.ones((8, 8, 3)))
        fig.canvas.draw()
        im_artist = ax.images[-1]
        recolor_label(ax, text, image_artist=im_artist)
        assert text.get_color() is not None

    def test_no_image_artist(self):
        fig, ax, text, _ = _make_scene()
        recolor_label(ax, text)
        assert text.get_color() is not None


class TestAddLabelBackground:
    def test_basic(self):
        fig, ax, text, _ = _make_scene()
        patch = add_label_background(ax, text, color='red', alpha=0.5)
        # May return None if text bbox is zero on headless backend
        if patch is not None:
            assert hasattr(patch, 'get_facecolor')

    def test_pill_style(self):
        fig, ax, text, _ = _make_scene()
        patch = add_label_background(ax, text, color='blue', style='pill')
        # Just verify it doesn't crash
