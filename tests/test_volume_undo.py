"""Server-owned undo/redo for volume mask editing.

The mask volume lives on the server (that's where the edit compute lives: ncolor,
the EDT ball, merge-by-colour), so its undo/redo history is server-owned too — no
split client/server history to drift out of sync. These tests validate that
history under explicit scenarios AND under long randomised permutations, checking
the mask volume AND the stable label→group map at every step.
"""
import numpy as np
import pytest

pytest.importorskip("imageio")
pytest.importorskip("scipy")
tifffile = pytest.importorskip("tifffile")

from ocdkit.viewer.session import SESSION_MANAGER


def _session(tmp_path, shape=(8, 40, 40), cells=2):
    """A session with a volume + an initial mask of `cells` horizontal bars."""
    vol = tmp_path / "vol.tif"
    tifffile.imwrite(str(vol), (np.random.default_rng(0).random(shape) * 255).astype(np.uint8))
    masks = np.zeros(shape, np.uint8)
    for c in range(cells):
        y0 = 4 + c * 8
        masks[:, y0:y0 + 5, 4:shape[2] - 4] = c + 1
    tifffile.imwrite(str(tmp_path / "vol_masks.tif"), masks)
    state = SESSION_MANAGER.get_or_create(None)
    SESSION_MANAGER.set_image(state, vol)
    SESSION_MANAGER.ensure_ncolor(state)   # populate the stable label→group map
    return state


def _disk(shape2d, cy, cx, r):
    yy, xx = np.mgrid[0:shape2d[0], 0:shape2d[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r


def _snapshot(state):
    return (state.current_mask_volume.copy(), dict(state.label_group or {}))


def _same(state, snap):
    return np.array_equal(state.current_mask_volume, snap[0]) and (dict(state.label_group or {}) == snap[1])


# ----- explicit scenarios -------------------------------------------------


def test_paint_then_undo_then_redo(tmp_path):
    state = _session(tmp_path)
    base = _snapshot(state)
    fp = _disk((40, 40), 30, 20, 5)
    SESSION_MANAGER.paint_sphere(state, 4, 0, 3, 0, fp)     # paint colour 3 in empty space
    after_paint = _snapshot(state)
    assert not _same(state, base)
    assert SESSION_MANAGER.undo(state) is True
    assert _same(state, base)                               # exactly back to start
    assert SESSION_MANAGER.redo(state) is True
    assert _same(state, after_paint)                        # exactly forward again


def test_the_reported_bug_undo_is_not_reverted_by_next_draw(tmp_path):
    """Draw a 'bad' stroke, undo it, draw a 'good' stroke elsewhere: the bad
    stroke must NOT resurface (the original report)."""
    state = _session(tmp_path)
    bad_fp = _disk((40, 40), 32, 10, 4)
    SESSION_MANAGER.paint_sphere(state, 4, 0, 2, 0, bad_fp)     # bad colour at region A
    assert state.current_mask_volume[4, 32, 10] != 0
    SESSION_MANAGER.undo(state)                                 # undo the bad stroke
    assert state.current_mask_volume[4, 32, 10] == 0           # region A cleared
    good_fp = _disk((40, 40), 32, 30, 4)
    SESSION_MANAGER.paint_sphere(state, 4, 0, 3, 0, good_fp)    # good colour at region B
    # ending the new draw must NOT bring the bad stroke back
    assert state.current_mask_volume[4, 32, 10] == 0           # region A still clear ✓
    assert state.current_mask_volume[4, 32, 30] != 0           # region B painted


def test_new_edit_after_undo_truncates_redo(tmp_path):
    state = _session(tmp_path)
    SESSION_MANAGER.paint_sphere(state, 4, 0, 2, 0, _disk((40, 40), 30, 10, 4))   # edit 1
    SESSION_MANAGER.paint_sphere(state, 4, 0, 3, 0, _disk((40, 40), 30, 30, 4))   # edit 2
    SESSION_MANAGER.undo(state)                                                   # back to after edit 1
    assert SESSION_MANAGER.can_redo(state) is True
    SESSION_MANAGER.paint_sphere(state, 5, 0, 4, 0, _disk((40, 40), 10, 20, 4))   # edit 3 forks history
    assert SESSION_MANAGER.can_redo(state) is False                              # edit 2's redo is gone
    assert SESSION_MANAGER.redo(state) is False                                  # nothing to redo


def test_deep_stack_undo_all_then_redo_all(tmp_path):
    state = _session(tmp_path)
    snaps = [_snapshot(state)]
    for i in range(8):
        SESSION_MANAGER.paint_sphere(state, i % 8, 0, (i % 4) + 1, 0, _disk((40, 40), 5 + i * 3, 20, 3))
        snaps.append(_snapshot(state))
    for i in range(8):                                        # undo everything
        assert SESSION_MANAGER.undo(state) is True
        assert _same(state, snaps[8 - i - 1])
    assert SESSION_MANAGER.can_undo(state) is False
    assert SESSION_MANAGER.undo(state) is False               # past the bottom: no-op
    for i in range(8):                                        # redo everything
        assert SESSION_MANAGER.redo(state) is True
        assert _same(state, snaps[i + 1])
    assert SESSION_MANAGER.redo(state) is False               # past the top: no-op


def test_merge_then_undo_restores_split_cells_and_colours(tmp_path):
    """Painting cell 1's colour onto cell 2 (same group → merge) then undoing
    restores both cells AND their colours (label_group)."""
    state = _session(tmp_path, cells=2)
    cmap = SESSION_MANAGER.ncolor_map(state)
    g1 = cmap[1]
    before = _snapshot(state)
    # force cell 2 to share cell 1's colour so painting g1 across them merges
    state.label_group[2] = g1
    state.current_ncolor_volume = None
    merged_before = _snapshot(state)
    SESSION_MANAGER.paint_sphere(state, 4, 0, g1, 0, _disk((40, 40), 11, 20, 6))  # bridge 1↔2
    assert (state.current_mask_volume == 2).sum() == 0       # cell 2 merged into cell 1
    assert SESSION_MANAGER.undo(state) is True
    assert _same(state, merged_before)                        # split cells + colours restored
    assert (state.current_mask_volume == 2).sum() > 0


def test_erase_then_undo(tmp_path):
    state = _session(tmp_path)
    before = _snapshot(state)
    n1 = int((state.current_mask_volume == 1).sum())
    SESSION_MANAGER.paint_sphere(state, 4, 0, 0, 0, _disk((40, 40), 6, 20, 4))   # erase part of cell 1
    assert int((state.current_mask_volume == 1).sum()) < n1
    SESSION_MANAGER.undo(state)
    assert _same(state, before)


def test_noop_edit_does_not_record_history(tmp_path):
    state = _session(tmp_path)
    SESSION_MANAGER.paint_sphere(state, 4, 0, 1, 0, _disk((40, 40), 6, 20, 3))   # paint cell-1 colour into cell 1
    # (this may or may not change pixels; an edit that changes nothing must not push history)
    depth = len(state.undo_stack or [])
    SESSION_MANAGER.paint_sphere(state, 4, 0, 0, 0, np.zeros((40, 40), bool))    # empty footprint = no-op
    assert len(state.undo_stack or []) == depth                                  # no spurious entry


# ----- model-based randomised permutations --------------------------------


@pytest.mark.parametrize("seed", [1, 7, 42, 123, 2024])
def test_undo_redo_random_permutations(tmp_path, seed):
    """Apply a long random sequence of paint/erase/undo/redo and verify the server
    state matches an independent timeline model at every single step."""
    state = _session(tmp_path, shape=(8, 36, 36), cells=2)
    rng = np.random.default_rng(seed)
    timeline = [_snapshot(state)]   # timeline[ptr] == expected current state
    ptr = 0
    counts = {"paint": 0, "erase": 0, "undo": 0, "redo": 0, "noop": 0}

    for _ in range(60):
        op = rng.choice(["paint", "paint", "erase", "undo", "undo", "redo"])
        if op in ("paint", "erase"):
            group = 0 if op == "erase" else int(rng.integers(1, 5))
            z = int(rng.integers(0, 8))
            fp = _disk((36, 36), int(rng.integers(6, 30)), int(rng.integers(6, 30)), int(rng.integers(2, 6)))
            before = state.current_mask_volume.copy()
            SESSION_MANAGER.paint_sphere(state, z, 0, group, 0, fp)
            if np.array_equal(before, state.current_mask_volume):
                counts["noop"] += 1                          # changed nothing → no history entry
            else:
                timeline = timeline[:ptr + 1]                # a new edit forks history
                timeline.append(_snapshot(state))
                ptr += 1
                counts[op] += 1
        elif op == "undo":
            ok = SESSION_MANAGER.undo(state)
            assert ok == (ptr > 0)
            if ok:
                ptr -= 1
                counts["undo"] += 1
        else:  # redo
            ok = SESSION_MANAGER.redo(state)
            assert ok == (ptr < len(timeline) - 1)
            if ok:
                ptr += 1
                counts["redo"] += 1

        # invariants checked EVERY step
        assert _same(state, timeline[ptr]), f"state diverged from model at ptr={ptr}"
        assert SESSION_MANAGER.can_undo(state) == (ptr > 0)
        assert SESSION_MANAGER.can_redo(state) == (ptr < len(timeline) - 1)
        # the rendered ncolor map is always derivable + consistent with the labels
        cmap = SESSION_MANAGER.ncolor_map(state)
        present = [int(x) for x in np.unique(state.current_mask_volume) if x > 0]
        assert all(cmap[l] > 0 for l in present)

    # the random walk actually exercised editing + undo (redo landing is
    # probabilistic per-seed; it's covered deterministically in the explicit tests)
    assert counts["paint"] > 0 and counts["undo"] > 0
