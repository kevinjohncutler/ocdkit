"""Navigation + load-masks auto-detection for volumes.

Switching the open image (incl. to a 3D stack) and the load-masks button finding
a sibling ``*_masks`` / ``*_masks_edited`` next to the source image.
"""
import numpy as np
import pytest

pytest.importorskip("imageio")
pytest.importorskip("scipy")
tifffile = pytest.importorskip("tifffile")

from ocdkit.viewer.session import SESSION_MANAGER


def _img(tmp_path, name, shape):
    p = tmp_path / name
    tifffile.imwrite(str(p), (np.random.default_rng(0).random(shape) * 255).astype(np.uint8))
    return p


def test_auto_mask_base_is_source_edited_is_loaded(tmp_path):
    """With both present, the BASE mask is the source (so it's preserved + future
    autosaves target *_edited), but the EDITED content is what loads/resumes."""
    img = _img(tmp_path, "a.tif", (8, 20, 20))
    tifffile.imwrite(str(tmp_path / "a_masks.tif"), np.zeros((8, 20, 20), np.uint8))
    edited = np.zeros((8, 20, 20), np.uint8); edited[:, 4:8, 4:16] = 1
    tifffile.imwrite(str(tmp_path / "a_masks_edited.tif"), edited)
    assert SESSION_MANAGER._auto_mask_path(img).name == "a_masks.tif"   # base is the source
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, img)
    assert state.mask_source_path.name == "a_masks.tif"
    assert int((state.current_mask_volume == 1).sum()) > 0              # edited content loaded


def test_auto_mask_finds_standalone_edited(tmp_path):
    """If ONLY a *_masks_edited exists (no base), it's still found."""
    img = _img(tmp_path, "z.tif", (8, 20, 20))
    tifffile.imwrite(str(tmp_path / "z_masks_edited.tif"), np.zeros((8, 20, 20), np.uint8))
    assert SESSION_MANAGER._auto_mask_path(img).name == "z_masks_edited.tif"


def test_auto_mask_path_finds_plain_masks(tmp_path):
    img = _img(tmp_path, "b.tif", (8, 20, 20))
    tifffile.imwrite(str(tmp_path / "b_masks.tif"), np.zeros((8, 20, 20), np.uint8))
    assert SESSION_MANAGER._auto_mask_path(img).name == "b_masks.tif"
    assert SESSION_MANAGER._auto_mask_path(_img(tmp_path, "none.tif", (8, 20, 20))) is None


def test_try_auto_mask_loads_sibling_for_volume(tmp_path):
    img = _img(tmp_path, "c.tif", (8, 30, 30))
    m = np.zeros((8, 30, 30), np.uint8); m[:, 4:10, 4:24] = 1
    tifffile.imwrite(str(tmp_path / "c_masks.tif"), m)
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, img)              # auto-loads on open...
    state.current_mask_volume = None                   # ...clear, then exercise the button path
    state.label_group = None
    path = SESSION_MANAGER.try_auto_mask(state)
    assert path is not None and path.endswith("c_masks.tif")
    assert int((state.current_mask_volume == 1).sum()) > 0


def test_try_auto_mask_none_when_missing(tmp_path):
    img = _img(tmp_path, "d.tif", (8, 30, 30))
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, img)              # no sidecar in this dir
    assert SESSION_MANAGER.try_auto_mask(state) is None


def test_open_image_switches_between_2d_and_3d(tmp_path):
    """Switching the open image to a 3D stack flips the config to a volume so the
    client knows to (re)initialise the 3D view."""
    flat = tmp_path / "flat.tif"
    tifffile.imwrite(str(flat), (np.random.default_rng(1).random((64, 64)) * 255).astype(np.uint8))
    stack = _img(tmp_path, "stack.tif", (12, 40, 40))
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, flat)
    assert not SESSION_MANAGER.build_config(state, embed_image=False)["isVolume"]
    SESSION_MANAGER.set_image(state, stack)
    cfg = SESSION_MANAGER.build_config(state, embed_image=False)
    assert cfg["isVolume"] and cfg["volumeDepth"] == 12
    SESSION_MANAGER.set_image(state, flat)             # back to 2D
    assert not SESSION_MANAGER.build_config(state, embed_image=False)["isVolume"]
