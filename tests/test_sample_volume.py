"""The built-in 3D sample (OCDKIT_VIEWER_SAMPLE_IMAGE=3d / --sample 3d)."""
import numpy as np
import pytest

pytest.importorskip("imageio")
pytest.importorskip("scipy")
pytest.importorskip("tifffile")

from ocdkit.viewer import sample_image as si


def test_widefield_psf_is_an_hourglass():
    psf = si.widefield_psf()
    nz, ny, nx = psf.shape
    assert psf.sum() == pytest.approx(1.0, rel=1e-4)
    assert np.unravel_index(psf.argmax(), psf.shape) == (nz // 2, ny // 2, nx // 2)
    half = psf.max() / 2
    axial = int((psf[:, ny // 2, nx // 2] > half).sum())
    lateral = int((psf[nz // 2, ny // 2, :] > half).sum())
    assert axial > 2 * lateral                          # widefield: elongated along z
    # symmetric about focus (no aberrations), and no FFT wraparound: the defocused
    # cone is brighter near the axis than at the crop edge
    np.testing.assert_allclose(psf[nz // 2 - 10], psf[nz // 2 + 10], rtol=1e-3, atol=1e-9)
    far = psf[0]
    assert far[ny // 2, nx // 4] > far[ny // 2, 0]


def test_sample_volume_is_deterministic_with_matching_labels():
    img, lab = si.generate_sample_volume()
    img2, lab2 = si.generate_sample_volume()
    np.testing.assert_array_equal(img, img2)
    np.testing.assert_array_equal(lab, lab2)
    assert img.shape == lab.shape == (64, 128, 128)
    assert img.dtype == np.uint8
    assert set(np.unique(lab)) == set(range(7))          # background + 6 cells
    assert img[lab > 0].mean() > 3 * img[lab == 0].mean()


def test_viewer_opens_the_3d_sample_with_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))            # sample cache goes under ~/.ocdkit
    monkeypatch.setenv("OCDKIT_VIEWER_SAMPLE_IMAGE", "3d")
    path = si.get_preload_image_path()
    assert path.is_file() and path.with_name(path.stem + "_masks.tif").is_file()
    assert tmp_path in path.parents

    from ocdkit.viewer.session import SESSION_MANAGER
    state = SESSION_MANAGER.get_or_create(None)
    assert state.current_volume is not None and state.current_volume.shape == (64, 128, 128)
    assert state.current_mask_volume is not None
    assert int(state.current_mask_volume.max()) == 6


def test_cli_sample_flag_sets_the_env(monkeypatch):
    from ocdkit.viewer import cli
    monkeypatch.delenv("OCDKIT_VIEWER_SAMPLE_IMAGE", raising=False)
    assert cli.parse_args(["serve", "--sample", "3d"]).sample == "3d"
    assert cli.parse_args(["desktop", "--sample", "3d"]).sample == "3d"
    assert cli.parse_args(["serve"]).sample is None


def test_voxel_shapes_volume_is_exact():
    """The renderer test volume: exact shapes on a flat background, one label each."""
    from ocdkit.viewer.sample_image import generate_voxel_shapes_volume

    img, lab = generate_voxel_shapes_volume()
    assert img.shape == lab.shape == (32, 48, 64) and img.dtype == np.uint8
    assert set(np.unique(img[lab == 0]).tolist()) == {20}               # flat background, no noise
    sizes = np.bincount(lab.ravel())[1:]
    assert sizes.tolist() == [1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 27, 64, 216, 64]
    zz, yy, xx = np.nonzero(lab == 8)                                    # the line along z
    assert len(set(zz.tolist())) == 8 and len(set(yy.tolist())) == 1 and len(set(xx.tolist())) == 1
    assert sorted(set(img[lab == 14].tolist())) == [80, 130, 180, 230]  # the ramp block
