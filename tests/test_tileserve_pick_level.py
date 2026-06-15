"""Tests for TileSource.pick_level and the geometry helpers relocated into
tileserve/plot (rounded-rect alpha mask + contour-to-rect clip)."""
import numpy as np

from ocdkit.tileserve.server import TileSource
from ocdkit.tileserve.headless import _rounded_alpha
from ocdkit.plot.contour import _clip_contour_to_rect


class TestPickLevel:
    def test_coarsest_covering_tier(self):
        ts = TileSource(width=1000, height=800, n_levels=4)
        dims = ts.level_dims("fov")          # [[h, w], ...] coarse -> fine

        # tiny target -> coarsest level (0) already covers it
        assert ts.pick_level("fov", 1, 1) == 0
        # larger than every level -> finest level (n_level - 1)
        assert ts.pick_level("fov", 10_000, 10_000) == ts.n_level("fov") - 1

        # exact-cover at a mid tier -> that tier (coarsest that still covers)
        mid = len(dims) // 2
        h, w = dims[mid]
        li = ts.pick_level("fov", w, h)
        lh, lw = dims[li]
        assert lw >= w and lh >= h
        if li > 0:                           # the next-coarser tier must NOT cover
            ph, pw = dims[li - 1]
            assert not (pw >= w and ph >= h)

    def test_single_level_label(self):
        ts = TileSource(width=512, height=512, n_levels=5)
        ts.declare("rgb", single_level=True)
        assert ts.n_level("rgb") == 1
        assert ts.pick_level("rgb", 99_999, 99_999) == 0


class TestMovedGeometryHelpers:
    def test_rounded_alpha_shape_and_range(self):
        m = _rounded_alpha(20, 16, r=4.0)
        assert m.shape == (16, 20) and m.dtype == np.float32
        assert m.min() >= 0.0 and m.max() <= 1.0
        assert m[8, 10] > m[0, 0]            # interior opaque, corner masked

    def test_clip_contour_inside_rect(self):
        sq = [(2, 2), (8, 2), (8, 8), (2, 8)]          # fully inside
        polylines = _clip_contour_to_rect(sq, 0, 0, 10, 10)
        assert polylines and all(len(p) >= 2 for p in polylines)

    def test_clip_contour_outside_rect(self):
        sq = [(20, 20), (28, 20), (28, 28), (20, 28)]  # fully outside
        assert _clip_contour_to_rect(sq, 0, 0, 10, 10) == []
