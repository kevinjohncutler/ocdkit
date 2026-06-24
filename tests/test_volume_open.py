"""Phase A: server-side volume detection in the open-image flow.

A 3-D (Z, Y, X) stack is detected as a volume, stored whole, and exposed to the
2D pipeline one slice at a time; config carries volume metadata and a slice route
serves individual planes for 2.5D navigation.
"""
import base64
import gzip
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


def test_volume_bundle_intensity_only(tmp_path):
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, _write_volume(tmp_path, (10, 24, 24)))
    bundle = SESSION_MANAGER.encode_volume_bundle(state)
    assert bundle["meta"] == {"dim": 3, "axes": ["t", "y", "x"],
                              "depth": 10, "height": 24, "width": 24}
    assert "mask" not in bundle           # intensity-only until segmentation
    img = bundle["image"]
    raw = gzip.decompress(base64.b64decode(img["b64"]))
    arr = np.frombuffer(raw, dtype=np.dtype(img["dtype"])).reshape(img["shape"])
    assert arr.shape == (10, 24, 24)
    np.testing.assert_array_equal(arr, state.current_volume)


def test_volume_bundle_none_for_2d(tmp_path):
    p = tmp_path / "img.tif"
    tifffile.imwrite(str(p), (np.random.default_rng(3).random((30, 30)) * 255).astype(np.uint8))
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, p)
    assert SESSION_MANAGER.encode_volume_bundle(state) is None


def test_auto_mask_from_sidecar(tmp_path):
    vol = _write_volume(tmp_path, (10, 24, 24))            # writes vol.tif
    masks = np.zeros((10, 24, 24), np.uint8)
    masks[2:6, 4:10, 4:10] = 3
    tifffile.imwrite(str(tmp_path / "vol_masks.tif"), masks)   # *_masks sidecar
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    assert state.current_mask_volume is not None
    assert state.current_mask_volume.shape == (10, 24, 24)
    bundle = SESSION_MANAGER.encode_volume_bundle(state)
    assert "mask" in bundle
    m = bundle["mask"]
    arr = np.frombuffer(gzip.decompress(base64.b64decode(m["b64"])),
                        dtype=np.dtype(m["dtype"])).reshape(m["shape"])
    assert arr.shape == (10, 24, 24) and int(arr.max()) >= 1   # bundle mask = ncolor groups


def test_open_mask_route_and_shape_mismatch(tmp_path):
    from ocdkit.viewer.routers.session_routes import api_open_mask
    from ocdkit.viewer.schemas import OpenMaskPayload
    vol = _write_volume(tmp_path, (8, 20, 20))            # no sidecar this dir
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    assert state.current_mask_volume is None
    good = tmp_path / "m.tif"
    tifffile.imwrite(str(good), np.ones((8, 20, 20), np.uint8))
    api_open_mask(OpenMaskPayload(sessionId=state.session_id, path=str(good)), state)
    assert state.current_mask_volume is not None
    bad = tmp_path / "bad.tif"
    tifffile.imwrite(str(bad), np.ones((4, 10, 10), np.uint8))
    with pytest.raises(Exception):
        api_open_mask(OpenMaskPayload(sessionId=state.session_id, path=str(bad)), state)


def test_mask_slice_route(tmp_path):
    from ocdkit.viewer.routers.session_routes import api_mask_slice
    vol = _write_volume(tmp_path, (12, 32, 32))
    masks = np.zeros((12, 32, 32), np.uint8)
    masks[5, 8:16, 8:16] = 7
    tifffile.imwrite(str(tmp_path / "vol_masks.tif"), masks)
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    # instance = identity labels
    resp_i = api_mask_slice(state.session_id, z=5, kind="instance")
    assert resp_i.headers["X-Mask-Width"] == "32" and resp_i.headers["X-Mask-Height"] == "32"
    arr_i = np.frombuffer(resp_i.body, dtype=np.dtype(resp_i.headers["X-Mask-Dtype"])).reshape(32, 32)
    assert int(arr_i.max()) == 7
    # default (group) = ncolor groups for display
    resp_g = api_mask_slice(state.session_id, z=5)
    arr_g = np.frombuffer(resp_g.body, dtype=np.dtype(resp_g.headers["X-Mask-Dtype"])).reshape(32, 32)
    assert int(arr_g.max()) >= 1
    cfg = SESSION_MANAGER.build_config(state, embed_image=False)
    assert cfg["hasVolumeMask"] is True


def test_set_mask_slice_roundtrip_and_create(tmp_path):
    vol = _write_volume(tmp_path, (6, 16, 16))
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    assert state.current_mask_volume is None              # no sidecar here
    labels = np.zeros((16, 16), np.uint32)
    labels[2:5, 2:5] = 9
    SESSION_MANAGER.set_mask_slice(state, 3, labels.tobytes(), "uint32")   # creates the volume
    assert state.current_mask_volume.shape == (6, 16, 16)
    data, w, h, dtype = SESSION_MANAGER.mask_slice(state, 3, kind="instance")
    back = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(h, w)
    assert int(back.max()) == 9
    b = SESSION_MANAGER.encode_volume_bundle(state)        # bundle = ncolor groups for the 3D render
    m = b["mask"]
    marr = np.frombuffer(gzip.decompress(base64.b64decode(m["b64"])),
                         dtype=np.dtype(m["dtype"])).reshape(m["shape"])
    assert marr.shape == (6, 16, 16) and int(marr.max()) >= 1


def test_orthogonal_slicing_dims(tmp_path):
    from ocdkit.viewer.routers.session_routes import api_volume_slice
    vol = _write_volume(tmp_path, (10, 24, 30))           # D=10, H=24, W=30
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    shapes = {
        0: (24, 30),   # Z slice → (H, W)
        1: (10, 30),   # Y slice → (D, W)
        2: (10, 24),   # X slice → (D, H)
    }
    for axis, want in shapes.items():
        r = api_volume_slice(state.session_id, z=3, axis=axis)
        img = np.asarray(imageio.imread(io.BytesIO(r.body)))
        assert img.shape == want, (axis, img.shape)
    cfg = SESSION_MANAGER.build_config(state, embed_image=False)
    assert cfg["volumeShape"] == [10, 24, 30]


def test_mask_slice_write_along_axis(tmp_path):
    vol = _write_volume(tmp_path, (8, 20, 16))           # D, H, W
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    labels = np.zeros((8, 16), np.uint32)               # Y-axis slice shape = (D, W)
    labels[1:4, 2:5] = 4
    SESSION_MANAGER.set_mask_slice(state, 7, labels.tobytes(), "uint32", axis=1)
    data, w, h, dtype = SESSION_MANAGER.mask_slice(state, 7, axis=1, kind="instance")
    arr = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(h, w)
    assert arr.shape == (8, 16) and int(arr.max()) == 4
    assert int(state.current_mask_volume[:, 7, :].max()) == 4   # landed at y=7


def test_ncolor_groups_separate_adjacent_cells(tmp_path):
    """Two touching cells must get different ncolor groups (no shared color)."""
    vol = _write_volume(tmp_path, (5, 20, 20))   # depth>4 so it's read as a volume
    masks = np.zeros((5, 20, 20), np.uint8)
    masks[:, 4:10, 4:16] = 1          # cell 1
    masks[:, 10:16, 4:16] = 2         # cell 2 (touches cell 1 along y=10)
    tifffile.imwrite(str(tmp_path / "vol_masks.tif"), masks)
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    g = SESSION_MANAGER.ensure_ncolor(state)
    assert g is not None
    g1 = int(np.unique(g[masks == 1])[0])
    g2 = int(np.unique(g[masks == 2])[0])
    assert g1 != g2 and g1 > 0 and g2 > 0     # adjacent cells → different groups
