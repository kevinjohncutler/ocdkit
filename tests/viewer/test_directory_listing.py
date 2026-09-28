"""The file list next to an image: sidecars are folded into their image, and a
symlinked image stays in the folder it was opened from."""

from __future__ import annotations

import numpy as np
import tifffile

from ocdkit.viewer.session import SessionManager


def _img(path, shape=(16, 16)):
    tifffile.imwrite(path, (np.arange(np.prod(shape)) % 251).astype(np.uint8).reshape(shape))
    return path


def test_sidecars_of_listed_images_are_hidden(tmp_path):
    for name in ("a.tif", "a_masks.tif", "a_flows.tif", "a_cp_masks_edited.tif", "b.tif", "orphan_masks.tif"):
        _img(tmp_path / name)
    names = [p.name for p in SessionManager()._list_directory_images(tmp_path)]
    assert names == ["a.tif", "b.tif", "orphan_masks.tif"]   # a sidecar without its image stays


def test_symlinked_image_keeps_its_folder(tmp_path):
    real = tmp_path / "data"; real.mkdir()
    samples = tmp_path / "samples"; samples.mkdir()
    _img(real / "stack.tif"); _img(real / "stack_masks.tif"); _img(real / "other.tif")
    _img(samples / "sample.tif")
    (samples / "stack.tif").symlink_to(real / "stack.tif")
    (samples / "stack_masks.tif").symlink_to(real / "stack_masks.tif")
    mgr = SessionManager()
    state = mgr.get_or_create(None)
    mgr.set_image(state, samples / "stack.tif")
    assert state.current_path == samples / "stack.tif"
    assert state.directory == samples
    assert [p.name for p in state.files] == ["sample.tif", "stack.tif"]
    assert mgr.navigate(state, -1) == samples / "sample.tif"      # can go back to the sample
