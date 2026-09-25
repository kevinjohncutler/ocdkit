"""3D ray-pick + fill-by-label — the picker/fill acting on the 3D render.

A world-space ray (from the camera) is marched through the label volume with the
SAME coordinate math as the render shader, so the picked cell is the one drawn
under the cursor. fill_label then deletes / merges that whole cell.
"""
import numpy as np
import pytest

pytest.importorskip("imageio")
pytest.importorskip("scipy")
tifffile = pytest.importorskip("tifffile")

from ocdkit.viewer.session import SESSION_MANAGER

BOX_MIN = [-10.0, -10.0, -10.0]
BOX_MAX = [10.0, 10.0, 10.0]   # centered box for a 20^3 volume, zScale=1


def _session(tmp_path):
    vol = tmp_path / "vol.tif"
    tifffile.imwrite(str(vol), (np.random.default_rng(0).random((20, 20, 20)) * 255).astype(np.uint8))
    m = np.zeros((20, 20, 20), np.uint8)
    m[6:14, 6:14, 6:14] = 5            # a block cell centered in the volume
    m[6:14, 6:14, 0:4] = 7             # a second cell near the low-x wall
    tifffile.imwrite(str(tmp_path / "vol_masks.tif"), m)
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    return state


def test_pick_ray_hits_cell_under_cursor(tmp_path):
    state = _session(tmp_path)
    # ray down the +z axis through the centre → hits the central cell (label 5)
    lab, grp, vc = SESSION_MANAGER.pick_ray(state, [0, 0, 30], [0, 0, -1], BOX_MIN, BOX_MAX)
    assert lab == 5
    assert grp == SESSION_MANAGER.ncolor_map(state)[5]
    assert 6 <= vc[0] < 14 and 6 <= vc[1] < 14            # voxel inside the cell (x,y)


def test_pick_ray_misses_empty_space(tmp_path):
    state = _session(tmp_path)
    # ray through a corner column that has no labels
    lab, grp, vc = SESSION_MANAGER.pick_ray(state, [-9, -9, 30], [0, 0, -1], BOX_MIN, BOX_MAX)
    assert lab == 0 and vc is None


def test_pick_ray_then_fill_label_deletes(tmp_path):
    state = _session(tmp_path)
    lab, _, _ = SESSION_MANAGER.pick_ray(state, [0, 0, 30], [0, 0, -1], BOX_MIN, BOX_MAX)
    assert lab == 5
    assert (state.current_mask_volume == 5).sum() > 0
    out = SESSION_MANAGER.fill_label(state, lab, erase=True)
    assert out == 0
    assert (state.current_mask_volume == 5).sum() == 0    # whole 3D cell deleted
    assert (state.current_mask_volume == 7).sum() > 0    # other cell untouched


def test_fill_ray_only_hits_contiguous_component(tmp_path):
    """3D-view fill via a ray deletes only the contiguous component under the cursor,
    not a disconnected blob that shares the label."""
    state = _session(tmp_path)
    mv = state.current_mask_volume
    mv[mv == 5] = 0                                       # clear the central cell
    mv[8:12, 8:12, 8:12] = 9                              # blob A (front, high z)
    mv[8:12, 8:12, 0:3] = 9                               # blob B, disconnected, same label
    state.current_ncolor_volume = None
    state.label_group = None
    # ray straight down +z through (x,y)=(10,10) hits blob A first
    out = SESSION_MANAGER.fill_ray(state, [0, 0, 30], [0, 0, -1], BOX_MIN, BOX_MAX, erase=True)
    assert out == 0
    assert (state.current_mask_volume[8:12, 8:12, 8:12] == 9).sum() == 0   # blob A erased
    assert (state.current_mask_volume[8:12, 8:12, 0:3] == 9).sum() > 0     # blob B kept


def test_fill_label_identity_merge_and_undo(tmp_path):
    state = _session(tmp_path)
    n5 = int((state.current_mask_volume == 5).sum())
    n7 = int((state.current_mask_volume == 7).sum())
    before = state.current_mask_volume.copy()
    out = SESSION_MANAGER.fill_label(state, 7, target_label=5)   # merge cell 7 into cell 5
    assert out == 5
    assert (state.current_mask_volume == 7).sum() == 0
    assert int((state.current_mask_volume == 5).sum()) == n5 + n7
    assert SESSION_MANAGER.undo(state) is True
    assert np.array_equal(state.current_mask_volume, before)


def _first_label_fine(mv, ro, rd, box_min, box_max, step=1e-3):
    """Reference: sample the ray every `step` voxels and return the first
    labeled voxel (z, y, x). Much finer than any voxel, so only slivers thinner
    than `step` can be missed."""
    NZ, NY, NX = mv.shape
    dims = np.array([NX, NY, NZ], float)
    ro, rd = np.asarray(ro, float), np.asarray(rd, float) / np.linalg.norm(rd)
    bmin, bmax = np.asarray(box_min, float), np.asarray(box_max, float)
    span = bmax - bmin
    inv = 1.0 / np.where(rd == 0, 1e-12, rd)
    t1, t2 = (bmin - ro) * inv, (bmax - ro) * inv
    tn, tf = max(0.0, np.minimum(t1, t2).max()), np.maximum(t1, t2).min()
    if tn > tf:
        return None
    t = np.arange(tn, tf, step * span.max() / dims.max())
    vc = np.clip(np.floor((ro + rd * t[:, None] - bmin) / span * dims).astype(int), 0, dims.astype(int) - 1)
    hit = mv[vc[:, 2], vc[:, 1], vc[:, 0]] > 0
    if not hit.any():
        return None
    x, y, z = vc[np.argmax(hit)]
    return (int(z), int(y), int(x))


def test_pick_ray_enters_a_grazed_voxel_corner(tmp_path):
    """A ray clipping a 0.02-voxel corner of a one-voxel cell still picks it (the
    render's DDA draws that corner; a fixed-step march can step over it)."""
    state = _session(tmp_path)
    mv = np.zeros((20, 20, 20), np.uint8)
    mv[5, 5, 5] = 9                                   # z, y, x
    state.current_mask_volume = mv
    state.current_ncolor_volume = None
    state.label_group = None
    # world = voxel - 10 (20^3 volume in a centered 20-unit box); the ray runs along
    # x + y = 11.98 in voxel units at z = 5.5, clipping voxel (x5, y5)'s corner
    ro = [0.98 - 10.0, 11.0 - 10.0, 5.5 - 10.0]
    rd = [1.0, -1.0, 0.0]
    lab, _, vc = SESSION_MANAGER.pick_ray(state, ro, rd, BOX_MIN, BOX_MAX)
    assert lab == 9 and vc == [5, 5, 5]


def test_pick_ray_matches_fine_sampling_on_random_rays(tmp_path):
    """The picked voxel is the first labeled voxel along the true ray."""
    state = _session(tmp_path)
    rng = np.random.default_rng(1)
    mv = (rng.random((20, 20, 20)) < 0.03).astype(np.uint8) * rng.integers(1, 200, (20, 20, 20)).astype(np.uint8)
    state.current_mask_volume = mv
    agree = total = 0
    for _ in range(1000):
        ro = rng.uniform(-30, 30, 3)
        rd = rng.normal(size=3) - ro / 30.0           # roughly toward the box
        v = SESSION_MANAGER._march_ray(state, ro, rd, BOX_MIN, BOX_MAX)
        ref = _first_label_fine(mv, ro, rd, BOX_MIN, BOX_MAX)
        total += 1
        agree += v == ref
    # the DDA matched 1000/1000 when written; the old fixed-step march 986-988
    assert agree / total >= 0.998, f"{agree}/{total}"
