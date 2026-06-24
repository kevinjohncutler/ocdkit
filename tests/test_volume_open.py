"""Phase A: server-side volume detection in the open-image flow.

A 3-D (Z, Y, X) stack is detected as a volume, stored whole, and exposed to the
2D pipeline one slice at a time; config carries volume metadata and a slice route
serves individual planes for 2.5D navigation.
"""
import io

import numpy as np
import pytest

pytest.importorskip("imageio")
tifffile = pytest.importorskip("tifffile")
from imageio import v2 as imageio

from ocdkit.viewer.session import SESSION_MANAGER


def _write_volume(tmp_path, shape=(20, 40, 40)):
    p = tmp_path / "vol.tif"
    arr = (np.random.default_rng(0).random(shape) * 1000).astype(np.uint16)
    tifffile.imwrite(str(p), arr)
    return p


def test_volume_detected_and_metadata(tmp_path):
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, _write_volume(tmp_path, (20, 40, 40)))
    cfg = SESSION_MANAGER.build_config(state, embed_image=False)
    assert cfg["isVolume"] is True
    assert cfg["volumeDepth"] == 20
    assert cfg["currentSlice"] == 10          # middle slice
    assert (cfg["width"], cfg["height"]) == (40, 40)
    assert state.current_volume.shape == (20, 40, 40)
    # the 2D view shows the middle slice
    assert state.current_image.shape == (40, 40)


def test_2d_image_not_volume(tmp_path):
    p = tmp_path / "img.tif"
    tifffile.imwrite(str(p), (np.random.default_rng(1).random((48, 48)) * 255).astype(np.uint8))
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, p)
    cfg = SESSION_MANAGER.build_config(state, embed_image=False)
    assert cfg["isVolume"] is False
    assert state.current_volume is None


def test_rgb_image_not_volume(tmp_path):
    p = tmp_path / "rgb.tif"
    tifffile.imwrite(str(p), (np.random.default_rng(2).random((40, 40, 3)) * 255).astype(np.uint8))
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, p)
    cfg = SESSION_MANAGER.build_config(state, embed_image=False)
    assert cfg["isVolume"] is False


def test_slice_route_returns_png_and_tracks_index(tmp_path):
    from ocdkit.viewer.routers.session_routes import api_volume_slice
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, _write_volume(tmp_path, (12, 32, 32)))
    resp = api_volume_slice(state.session_id, z=5)
    assert resp.media_type == "image/png"
    img = np.asarray(imageio.imread(io.BytesIO(resp.body)))
    assert img.shape == (32, 32)
    assert state.volume_slice == 5            # slice index tracked on the session
    # out-of-range clamps
    api_volume_slice(state.session_id, z=999)
    assert state.volume_slice == 11
