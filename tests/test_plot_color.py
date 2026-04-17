"""Tests for ocdkit.plot.color — verify numpy/torch/dask parity for colorization and flow viz."""

import numpy as np
import pytest
import torch
import dask.array as da

from ocdkit.plot.color import colorize, rgb_flow, rgb_to_hsv, hsv_to_rgb
from ocdkit.plot import sinebow


# ---------------------------------------------------------------------------
# colorize — numpy/torch parity
# ---------------------------------------------------------------------------

class TestColorize:
    def test_numpy_shape(self):
        im = np.random.rand(3, 32, 32).astype(np.float32)
        rgb = colorize(im)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.float32

    def test_torch_shape(self):
        im = torch.rand(3, 32, 32)
        rgb = colorize(im)
        assert rgb.shape == (32, 32, 3)

    def test_numpy_torch_parity(self):
        np.random.seed(0)
        im_np = np.random.rand(4, 16, 16).astype(np.float32)
        im_t = torch.from_numpy(im_np)

        rgb_np = colorize(im_np)
        rgb_t = colorize(im_t).numpy()
        np.testing.assert_allclose(rgb_np, rgb_t, atol=1e-4)

    def test_intervals(self):
        im = torch.rand(6, 8, 8)
        rgb = colorize(im, intervals=[3, 3])
        assert rgb.shape == (2, 8, 8, 3)

    def test_intervals_numpy(self):
        im = np.random.rand(6, 8, 8).astype(np.float32)
        rgb = colorize(im, intervals=[3, 3])
        assert rgb.shape == (2, 8, 8, 3)

    def test_intervals_parity(self):
        np.random.seed(1)
        im_np = np.random.rand(4, 8, 8).astype(np.float32)
        im_t = torch.from_numpy(im_np)

        rgb_np = colorize(im_np, intervals=[2, 2])
        rgb_t = colorize(im_t, intervals=[2, 2]).numpy()
        np.testing.assert_allclose(rgb_np, rgb_t, atol=1e-4)

    def test_custom_colors(self):
        im = np.random.rand(2, 8, 8).astype(np.float32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rgb = colorize(im, colors=colors)
        assert rgb.shape == (8, 8, 3)

    def test_color_weights(self):
        im = np.ones((2, 4, 4), dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        weights = np.array([1.0, 0.0])
        rgb = colorize(im, colors=colors, color_weights=weights)
        # Only red channel should be nonzero
        assert rgb[:, :, 1].max() == 0.0
        assert rgb[:, :, 0].max() > 0.0

    def test_offset(self):
        im = np.random.rand(3, 8, 8).astype(np.float32)
        rgb0 = colorize(im, offset=0)
        rgb1 = colorize(im, offset=np.pi)
        # Different offsets should produce different colors
        assert not np.allclose(rgb0, rgb1)

    def test_dask_shape(self):
        im = da.from_array(np.random.rand(3, 32, 32).astype(np.float32))
        rgb = colorize(im)
        assert rgb.shape == (32, 32, 3)

    def test_dask_intervals(self):
        im = da.from_array(np.random.rand(6, 16, 16).astype(np.float32))
        rgb = colorize(im, intervals=[3, 3])
        assert rgb.shape == (2, 16, 16, 3)

    def test_dask_numpy_parity(self):
        np.random.seed(10)
        im_np = np.random.rand(4, 16, 16).astype(np.float32)
        im_da = da.from_array(im_np)

        rgb_np = colorize(im_np)
        rgb_da = np.asarray(colorize(im_da))
        np.testing.assert_allclose(rgb_np, rgb_da, atol=1e-5)

    def test_dask_intervals_parity(self):
        np.random.seed(11)
        im_np = np.random.rand(6, 16, 16).astype(np.float32)
        im_da = da.from_array(im_np)

        rgb_np = colorize(im_np, intervals=[3, 3])
        rgb_da = np.asarray(colorize(im_da, intervals=[3, 3]))
        np.testing.assert_allclose(rgb_np, rgb_da, atol=1e-5)


# ---------------------------------------------------------------------------
# rgb_flow — numpy/torch parity + unbatched/batched
# ---------------------------------------------------------------------------

class TestRgbFlow:
    def test_torch_batched(self):
        dP = torch.rand(2, 2, 16, 16)  # (B=2, D=2, H, W)
        im = rgb_flow(dP, transparency=False)
        assert im.shape == (2, 16, 16, 3)
        assert im.dtype == torch.uint8

    def test_torch_unbatched(self):
        dP = torch.rand(2, 16, 16)  # (D=2, H, W)
        im = rgb_flow(dP, transparency=False)
        assert im.shape == (16, 16, 3)
        assert im.dtype == torch.uint8

    def test_numpy_returns_numpy(self):
        dP = np.random.rand(2, 16, 16).astype(np.float32)
        im = rgb_flow(dP, transparency=False)
        assert isinstance(im, np.ndarray)
        assert im.shape == (16, 16, 3)
        assert im.dtype == np.uint8

    def test_torch_returns_torch(self):
        dP = torch.rand(2, 16, 16)
        im = rgb_flow(dP, transparency=False)
        assert isinstance(im, torch.Tensor)

    def test_transparency(self):
        dP = np.random.rand(2, 8, 8).astype(np.float32)
        im = rgb_flow(dP, transparency=True)
        assert im.shape == (8, 8, 4)  # RGBA

        im_no = rgb_flow(dP, transparency=False)
        assert im_no.shape == (8, 8, 3)  # RGB

    def test_numpy_torch_parity(self):
        np.random.seed(42)
        dP_np = np.random.rand(2, 16, 16).astype(np.float32)
        dP_t = torch.from_numpy(dP_np)

        im_np = rgb_flow(dP_np, transparency=False, norm=False)
        im_t = rgb_flow(dP_t, transparency=False, norm=False).cpu().numpy()
        # uint8 outputs may differ by 1 due to rounding
        np.testing.assert_allclose(im_np.astype(float), im_t.astype(float), atol=1.0)

    def test_batched_unbatched_parity(self):
        dP = torch.rand(2, 8, 8)
        im_unbatched = rgb_flow(dP, transparency=False, norm=False)
        im_batched = rgb_flow(dP.unsqueeze(0), transparency=False, norm=False)
        assert torch.equal(im_unbatched, im_batched.squeeze(0))


# ---------------------------------------------------------------------------
# rgb_to_hsv / hsv_to_rgb
# ---------------------------------------------------------------------------

class TestRgbHsvRoundtrip:
    def test_roundtrip(self):
        rgb = np.array(
            [
                [[0.1, 0.2, 0.3], [0.8, 0.1, 0.2]],
                [[0.9, 0.9, 0.1], [0.0, 0.4, 0.7]],
            ],
            dtype=np.float32,
        )
        hsv = rgb_to_hsv(rgb)
        out = hsv_to_rgb(hsv)
        assert np.allclose(out, rgb, atol=1e-6)


# ---------------------------------------------------------------------------
# sinebow palette (lives in ocdkit.plot.ncolor but exposed via ocdkit.plot)
# ---------------------------------------------------------------------------

class TestSinebow:
    def test_palette(self):
        bg = [0.1, 0.2, 0.3, 0.4]
        palette = sinebow(3, bg_color=bg, offset=1)
        assert palette[0] == bg
        assert len(palette) == 4
        for k in range(1, 4):
            r, g, b, a = palette[k]
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0
            assert a == 1
