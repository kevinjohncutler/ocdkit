"""3D-view brush: a stroke of balls painted into the mask volume."""
import numpy as np
import pytest

pytest.importorskip("scipy")
tifffile = pytest.importorskip("tifffile")

from ocdkit.viewer.session import SESSION_MANAGER, SessionManager


def _state(tmp_path, shape=(20, 30, 40)):
    img = tmp_path / "vol.tif"
    tifffile.imwrite(str(img), (np.random.default_rng(0).random(shape) * 255).astype(np.uint8))
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, img)
    return state


def test_ball_mask_is_the_voxels_within_radius():
    mask, origin = SessionManager.ball_stroke_mask((20, 30, 40), [[20.5, 15.5, 10.5]], 3)
    zz, yy, xx = np.nonzero(mask)
    cz, cy, cx = zz + origin[0] + 0.5, yy + origin[1] + 0.5, xx + origin[2] + 0.5
    d2 = (cx - 20.5) ** 2 + (cy - 15.5) ** 2 + (cz - 10.5) ** 2
    assert d2.max() <= 9 and mask.sum() == 123            # every voxel center within R; a radius-3 voxel ball


def test_snap_moves_the_center_to_its_voxel():
    a, oa = SessionManager.ball_stroke_mask((20, 30, 40), [[20.9, 15.1, 10.6]], 3, snap=True)
    b, ob = SessionManager.ball_stroke_mask((20, 30, 40), [[20.5, 15.5, 10.5]], 3)
    assert oa == ob and np.array_equal(a, b)


def test_stroke_is_swept_without_gaps():
    mask, origin = SessionManager.ball_stroke_mask((20, 30, 40), [[5.5, 15.5, 10.5], [35.5, 15.5, 10.5]], 2)
    row = mask[10 - origin[0], 15 - origin[1]]              # along x through both centers
    xs = np.nonzero(row)[0] + origin[2]
    assert xs.min() <= 5 and xs.max() >= 35 and len(xs) == xs.max() - xs.min() + 1


def test_clipped_to_the_volume():
    mask, origin = SessionManager.ball_stroke_mask((20, 30, 40), [[0.5, 0.5, 0.5]], 4)
    assert origin == (0, 0, 0) and mask.shape == (5, 5, 5)          # voxel 4 (center 4.5) is exactly R from 0.5
    assert SessionManager.ball_stroke_mask((20, 30, 40), [[-50, -50, -50]], 2) == (None, None)


def test_paint_balls_paints_merges_and_undoes(tmp_path):
    state = _state(tmp_path)
    lab = SESSION_MANAGER.paint_balls(state, [[10.5, 10.5, 10.5]], 2, group=3)
    mv = state.current_mask_volume
    assert lab > 0 and mv[10, 10, 10] == lab and mv[10, 10, 13] == 0
    assert state.label_group[lab] == 3
    # a touching stroke of the same colour extends that cell (no new label)
    lab2 = SESSION_MANAGER.paint_balls(state, [[13.5, 10.5, 10.5]], 2, group=3)
    assert lab2 == lab and mv[10, 10, 15] == lab
    # erase, then undo the erase
    SESSION_MANAGER.paint_balls(state, [[10.5, 10.5, 10.5]], 1, group=0)
    assert mv[10, 10, 10] == 0
    assert SESSION_MANAGER.undo(state) and state.current_mask_volume[10, 10, 10] == lab
