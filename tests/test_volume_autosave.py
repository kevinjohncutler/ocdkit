"""Debounced autosave of the edited mask to disk.

Edits never touch the uploaded original — they go to a ``*_edited`` sibling — and
reopening the volume resumes from that edited file. This is what makes work
durable across page refresh / server restart / disconnect (the file is on disk).
"""
import time

import numpy as np
import pytest

pytest.importorskip("imageio")
pytest.importorskip("scipy")
tifffile = pytest.importorskip("tifffile")
from imageio import v2 as imageio

from ocdkit.viewer.session import SESSION_MANAGER


def _setup(tmp_path, shape=(8, 30, 30)):
    """Volume + an original *_masks.tif sidecar; returns (state, image_path, masks_path)."""
    img = tmp_path / "vol.tif"
    tifffile.imwrite(str(img), (np.random.default_rng(0).random(shape) * 255).astype(np.uint8))
    masks = tmp_path / "vol_masks.tif"
    m = np.zeros(shape, np.uint8)
    m[:, 4:10, 4:26] = 1
    tifffile.imwrite(str(masks), m)
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, img)            # auto-attaches vol_masks.tif
    return state, img, masks


def _disk(shape2d, cy, cx, r):
    yy, xx = np.mgrid[0:shape2d[0], 0:shape2d[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r


def test_edited_path_suffix_and_no_double_suffix(tmp_path):
    p = tmp_path / "foo_masks.tif"
    assert SESSION_MANAGER._edited_mask_path(p).name == "foo_masks_edited.tif"
    # an already-edited file maps to itself (no foo_masks_edited_edited.tif)
    e = tmp_path / "foo_masks_edited.tif"
    assert SESSION_MANAGER._edited_mask_path(e) == e


def test_autosave_writes_edited_and_leaves_original_untouched(tmp_path):
    state, img, masks = _setup(tmp_path)
    original_before = np.asarray(imageio.imread(masks)).copy()
    SESSION_MANAGER.paint_sphere(state, 4, 0, 2, 0, _disk((30, 30), 22, 15, 4))
    dest = SESSION_MANAGER.save_edited_mask(state)
    edited = masks.with_name("vol_masks_edited.tif")
    assert dest == str(edited) and edited.exists()
    # the ORIGINAL upload is byte-for-byte unchanged
    assert np.array_equal(np.asarray(imageio.imread(masks)), original_before)
    # the EDITED file reflects the edit (a new region appeared)
    saved = np.asarray(imageio.imread(edited))
    assert int(saved[4, 22, 15]) != 0
    assert not edited.with_name(edited.name + ".tmp").exists()   # atomic: no temp left behind


def test_debounce_timer_fires_without_manual_save(tmp_path, monkeypatch):
    state, img, masks = _setup(tmp_path)
    monkeypatch.setattr(SESSION_MANAGER, "SAVE_DEBOUNCE_SECONDS", 0.15)
    edited = masks.with_name("vol_masks_edited.tif")
    assert not edited.exists()
    SESSION_MANAGER.paint_sphere(state, 4, 0, 3, 0, _disk((30, 30), 22, 15, 4))  # arms the timer
    deadline = time.time() + 3.0
    while not edited.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert edited.exists(), "debounced autosave timer did not write the edited mask"


def test_debounce_coalesces_rapid_edits(tmp_path, monkeypatch):
    """Many quick strokes arm-and-reset one timer → a single write of the final state."""
    state, img, masks = _setup(tmp_path)
    monkeypatch.setattr(SESSION_MANAGER, "SAVE_DEBOUNCE_SECONDS", 0.3)
    edited = masks.with_name("vol_masks_edited.tif")
    for cx in range(12, 24, 2):
        SESSION_MANAGER.paint_sphere(state, 4, 0, 2, 0, _disk((30, 30), 22, cx, 3))
        time.sleep(0.05)                                  # faster than the debounce → coalesce
    assert not edited.exists()                            # not written yet (timer kept resetting)
    time.sleep(0.5)
    assert edited.exists()
    saved = np.asarray(imageio.imread(edited))
    assert int(saved[4, 22, 22]) != 0                     # the LAST stroke is present


def test_reopen_resumes_from_edited_file(tmp_path):
    state, img, masks = _setup(tmp_path)
    SESSION_MANAGER.paint_sphere(state, 5, 0, 2, 0, _disk((30, 30), 22, 15, 4))
    SESSION_MANAGER.save_edited_mask(state)
    edited_voxel = int(state.current_mask_volume[5, 22, 15])
    assert edited_voxel != 0

    # a brand-new session reopening the SAME image must pick up the edited mask
    state2 = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state2, img)
    assert int(state2.current_mask_volume[5, 22, 15]) == edited_voxel   # resumed ✓
    # and it keeps saving to the SAME edited file (source stays the original)
    assert state2.mask_source_path == masks.resolve()
    assert SESSION_MANAGER._edited_mask_path(state2.mask_source_path).name == "vol_masks_edited.tif"


def test_no_source_path_no_autosave(tmp_path):
    """A volume with no mask file (nothing to derive an edited path from) doesn't crash."""
    img = tmp_path / "vol.tif"
    tifffile.imwrite(str(img), (np.random.default_rng(1).random((8, 30, 30)) * 255).astype(np.uint8))
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, img)                 # no sidecar
    assert state.mask_source_path is None
    # painting creates a mask but there's no source file → save is a no-op (returns None)
    SESSION_MANAGER.paint_sphere(state, 4, 0, 1, 0, _disk((30, 30), 15, 15, 4))
    assert SESSION_MANAGER.save_edited_mask(state) is None
