"""3D colour picker + 3D fill (whole-cell merge / delete).

These power the "merge cells / delete cells" workflow: the picker captures a
cell's identity from one click on any slice; the fill then deletes a whole 3D
cell or merges one whole cell into the picked one — all server-side on the volume,
undoable, and autosaved.
"""
import numpy as np
import pytest

pytest.importorskip("imageio")
pytest.importorskip("scipy")
tifffile = pytest.importorskip("tifffile")

from ocdkit.viewer.session import SESSION_MANAGER


def _session(tmp_path, shape=(8, 30, 40)):
    vol = tmp_path / "vol.tif"
    tifffile.imwrite(str(vol), (np.random.default_rng(0).random(shape) * 255).astype(np.uint8))
    masks = np.zeros(shape, np.uint8)
    masks[:, 4:10, 4:36] = 1          # cell 1 (a bar through all z)
    masks[:, 14:20, 4:36] = 2          # cell 2
    masks[:, 24:28, 4:36] = 3          # cell 3
    tifffile.imwrite(str(tmp_path / "vol_masks.tif"), masks)
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    return state


def test_label_at_picks_cell_and_group(tmp_path):
    state = _session(tmp_path)
    cmap = SESSION_MANAGER.ncolor_map(state)
    lab, grp = SESSION_MANAGER.label_at(state, 4, 0, 16, 20)   # inside cell 2
    assert lab == 2 and grp == cmap[2]
    assert SESSION_MANAGER.label_at(state, 4, 0, 0, 0) == (0, 0)   # background


def test_fill_delete_removes_whole_3d_cell(tmp_path):
    state = _session(tmp_path)
    assert (state.current_mask_volume == 2).sum() > 0
    out = SESSION_MANAGER.fill_cell(state, 4, 0, 16, 20, erase=True)   # click cell 2, erase
    assert out == 0
    assert (state.current_mask_volume == 2).sum() == 0            # gone from EVERY slice
    assert (state.current_mask_volume == 1).sum() > 0            # neighbours untouched
    assert (state.current_mask_volume == 3).sum() > 0


def test_fill_identity_merge_joins_whole_cell_into_picked(tmp_path):
    state = _session(tmp_path)
    n1 = int((state.current_mask_volume == 1).sum())
    n3 = int((state.current_mask_volume == 3).sum())
    g1 = SESSION_MANAGER.ncolor_map(state)[1]
    # pick cell 1, then fill cell 3 → cell 3 merges into cell 1 even though they don't touch
    picked, grp = SESSION_MANAGER.label_at(state, 4, 0, 6, 20)
    assert picked == 1
    out = SESSION_MANAGER.fill_cell(state, 4, 0, 26, 20, target_label=picked)
    assert out == 1
    assert (state.current_mask_volume == 3).sum() == 0            # cell 3 absorbed
    assert int((state.current_mask_volume == 1).sum()) == n1 + n3  # cell 1 grew by all of cell 3
    assert SESSION_MANAGER.ncolor_map(state)[1] == g1            # cell 1 kept its colour


def test_fill_colour_merge_joins_touching_same_colour(tmp_path):
    """Fill with a colour: a cell merges into the TOUCHING cell of that colour."""
    state = _session(tmp_path)
    # make cells 1 and 2 touch (remove the gap) so a colour-fill can merge them
    mv = state.current_mask_volume
    mv[:, 10:14, 4:36] = 2                                        # extend cell 2 up to touch cell 1
    state.current_ncolor_volume = None
    state.label_group = None
    g1 = SESSION_MANAGER.ncolor_map(state)[1]                     # cell 1's colour
    n1 = int((mv == 1).sum()); n2 = int((mv == 2).sum())
    out = SESSION_MANAGER.fill_cell(state, 4, 0, 16, 20, group=g1)   # fill cell 2 with cell 1's colour
    assert out == 1                                               # merged into the touching cell 1
    assert (state.current_mask_volume == 2).sum() == 0
    assert int((state.current_mask_volume == 1).sum()) == n1 + n2


def test_fill_colour_recolours_isolated_cell(tmp_path):
    """Fill a non-touching cell with a colour → it just takes that colour (kept as its own cell)."""
    state = _session(tmp_path)
    g1 = SESSION_MANAGER.ncolor_map(state)[1]
    out = SESSION_MANAGER.fill_cell(state, 4, 0, 26, 20, group=g1)   # cell 3 (isolated) → colour g1
    assert out == 3                                               # stays its own cell
    assert SESSION_MANAGER.ncolor_map(state)[3] == g1            # recoloured
    assert (state.current_mask_volume == 3).sum() > 0


def test_fill_background_is_noop(tmp_path):
    state = _session(tmp_path)
    before = state.current_mask_volume.copy()
    assert SESSION_MANAGER.fill_cell(state, 4, 0, 0, 0, erase=True) == 0   # clicked background
    assert np.array_equal(state.current_mask_volume, before)
    assert not SESSION_MANAGER.can_undo(state)                    # nothing recorded


def test_fill_is_undoable(tmp_path):
    state = _session(tmp_path)
    before = state.current_mask_volume.copy()
    SESSION_MANAGER.fill_cell(state, 4, 0, 16, 20, erase=True)   # delete cell 2
    assert (state.current_mask_volume == 2).sum() == 0
    assert SESSION_MANAGER.undo(state) is True
    assert np.array_equal(state.current_mask_volume, before)      # cell 2 restored exactly


def test_fill_merge_then_undo_restores_colours(tmp_path):
    state = _session(tmp_path)
    g3 = SESSION_MANAGER.ncolor_map(state)[3]
    SESSION_MANAGER.fill_cell(state, 4, 0, 26, 20, target_label=1)   # merge 3 into 1
    assert (state.current_mask_volume == 3).sum() == 0
    SESSION_MANAGER.undo(state)
    assert (state.current_mask_volume == 3).sum() > 0
    assert SESSION_MANAGER.ncolor_map(state)[3] == g3            # colour map restored
