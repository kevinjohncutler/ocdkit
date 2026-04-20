"""Tests for ocdkit.io — file I/O round-trips and path utilities."""

import ntpath
import os
from pathlib import Path

import numpy as np
import pytest

import ocdkit.io as ocdkit_io
from ocdkit.io import check_dir, getname, imread, imwrite, findbetween, get_image_files, find_files, adjust_file_path


# ---------------------------------------------------------------------------
# check_dir
# ---------------------------------------------------------------------------

class TestCheckDir:
    def test_creates_new_directory(self, tmp_path):
        new_dir = tmp_path / "new"
        assert not new_dir.exists()
        check_dir(str(new_dir))
        assert new_dir.is_dir()

    def test_creates_nested_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        check_dir(str(nested))
        assert nested.is_dir()

    def test_existing_directory_no_error(self, tmp_path):
        existing = tmp_path / "exists"
        existing.mkdir()
        check_dir(str(existing))  # should not raise
        assert existing.is_dir()


# ---------------------------------------------------------------------------
# getname
# ---------------------------------------------------------------------------

class TestGetname:
    def test_basic(self):
        assert getname("/path/to/file.tif") == "file"

    def test_no_extension(self):
        assert getname("/path/to/file") == "file"

    def test_strip_suffix(self):
        assert getname("/path/img_masks.tif", suffix="_masks") == "img"

    def test_strip_prefix(self):
        assert getname("/path/pre_name.tif", prefix="pre_") == "name"

    def test_strip_both(self):
        assert getname("/path/pre_name_suf.tif", prefix="pre_", suffix="_suf") == "name"

    def test_padding(self):
        assert getname("/path/3.tif", padding=5) == "00003"

    def test_padding_no_truncation(self):
        # Padding doesn't shorten — just left-pads
        assert getname("/path/12345.tif", padding=3) == "12345"

    def test_pathlib_input(self):
        assert getname(Path("/path/to/file.tif")) == "file"


# ---------------------------------------------------------------------------
# imread / imwrite round-trips
# ---------------------------------------------------------------------------

class TestImageRoundtrips:
    def test_tiff_roundtrip(self, tmp_path):
        arr = (np.arange(16, dtype=np.uint16).reshape(4, 4))
        path = tmp_path / "a.tif"
        imwrite(str(path), arr)
        loaded = imread(str(path))
        np.testing.assert_array_equal(loaded, arr)

    def test_tiff_uppercase_extension(self, tmp_path):
        arr = np.ones((4, 4), dtype=np.uint16)
        path = tmp_path / "a.TIFF"
        imwrite(str(path), arr)
        loaded = imread(str(path))
        np.testing.assert_array_equal(loaded, arr)

    def test_npy_roundtrip(self, tmp_path):
        arr = np.random.rand(4, 4).astype(np.float32)
        path = tmp_path / "a.npy"
        imwrite(str(path), arr)
        loaded = imread(str(path))
        np.testing.assert_array_equal(loaded, arr)

    def test_npz_read(self, tmp_path):
        arr = np.arange(20).reshape(4, 5)
        path = tmp_path / "a.npz"
        np.savez(str(path), arr)
        loaded = imread(str(path))
        np.testing.assert_array_equal(loaded, arr)

    def test_png_roundtrip(self, tmp_path):
        arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
        path = tmp_path / "a.png"
        imwrite(str(path), arr)
        loaded = imread(str(path))
        assert loaded.shape == arr.shape

    def test_bmp_roundtrip(self, tmp_path):
        arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        path = tmp_path / "a.bmp"
        imwrite(str(path), arr)
        loaded = imread(str(path))
        assert loaded.shape == arr.shape

    def test_czi_roundtrip(self):
        fixture = Path(__file__).parent / "fixtures" / "tiny_8x8.czi"
        if not fixture.exists():
            pytest.skip("CZI fixture not found")
        data = imread(str(fixture))
        assert data is not None
        assert data.shape[-2:] == (8, 8)

    def test_czi_multichannel(self):
        fixture = Path(__file__).parent / "fixtures" / "multichan_3c_4x4.czi"
        if not fixture.exists():
            pytest.skip("CZI fixture not found")
        data = imread(str(fixture))
        assert data is not None
        assert data.shape[-2:] == (4, 4)

    def test_jpeg_roundtrip(self, tmp_path):
        arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        path = tmp_path / "a.jpg"
        imwrite(str(path), arr)
        loaded = imread(str(path))
        assert loaded is not None
        assert loaded.shape[:2] == arr.shape[:2]

    def test_webp_roundtrip(self, tmp_path):
        arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        path = tmp_path / "a.webp"
        imwrite(str(path), arr)
        loaded = imread(str(path))
        assert loaded is not None
        assert loaded.shape[:2] == arr.shape[:2]

    def test_webp_quality_kwarg(self, tmp_path):
        arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        path = tmp_path / "a.webp"
        imwrite(str(path), arr, quality=50)
        loaded = imread(str(path))
        assert loaded is not None

    def test_jxl_roundtrip(self, tmp_path):
        arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        path = tmp_path / "a.jxl"
        imwrite(str(path), arr)
        loaded = imread(str(path))
        assert loaded is not None
        assert loaded.shape[:2] == arr.shape[:2]

    def test_fallback_extension_uses_png(self, tmp_path):
        arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
        path = tmp_path / "a.unknown_ext"
        imwrite(str(path), arr)
        # Should have written PNG-encoded bytes
        assert path.stat().st_size > 0

    def test_imread_unknown_format_returns_none(self, tmp_path):
        path = tmp_path / "garbage.xyz"
        path.write_bytes(b"not an image")
        result = imread(str(path))
        assert result is None


# ---------------------------------------------------------------------------
# download_url_to_file
# ---------------------------------------------------------------------------

class TestDownloadUrlToFile:
    def test_importable(self):
        from ocdkit.io import download_url_to_file
        assert callable(download_url_to_file)


# ---------------------------------------------------------------------------
# findbetween
# ---------------------------------------------------------------------------

class TestFindbetween:
    def test_basic(self):
        assert findbetween("a[bc]d") == "bc"

    def test_no_brackets(self):
        assert findbetween("abcd") == ""


# ---------------------------------------------------------------------------
# adjust_file_path
# ---------------------------------------------------------------------------

class TestAdjustFilePath:
    def test_darwin_home_to_volumes(self, monkeypatch):
        monkeypatch.setattr(ocdkit_io.path.platform, "system", lambda: "Darwin")
        assert adjust_file_path("/home/alice/project") == "/Volumes/project"

    def test_linux_volumes_to_home(self, monkeypatch):
        monkeypatch.setattr(ocdkit_io.path.platform, "system", lambda: "Linux")
        home = os.path.expanduser("~")
        assert adjust_file_path("/Volumes/data").startswith(home)

    def test_linux_rewrites_volumes(self, monkeypatch):
        monkeypatch.setattr(ocdkit_io.path.platform, "system", lambda: "Linux")
        monkeypatch.setattr(ocdkit_io.path.os.path, "expanduser", lambda _: "/home/tester")
        assert adjust_file_path("/Volumes/datasets/run1") == "/home/tester/datasets/run1"

    def test_windows_from_home(self, monkeypatch):
        monkeypatch.setattr(ocdkit_io.path.platform, "system", lambda: "Windows")
        monkeypatch.setattr(ocdkit_io.path.os.path, "expanduser", lambda _: r"C:\Users\Tester")
        monkeypatch.setattr(ocdkit_io.path.os.path, "normpath", ntpath.normpath)
        adjusted = adjust_file_path("/home/alice/project/data.txt")
        expected = ntpath.normpath(r"C:\Users\Tester\project\data.txt")
        assert adjusted == expected

    def test_windows_from_volumes(self, monkeypatch):
        monkeypatch.setattr(ocdkit_io.path.platform, "system", lambda: "Windows")
        monkeypatch.setattr(ocdkit_io.path.os.path, "expanduser", lambda _: r"C:\Users\Tester")
        monkeypatch.setattr(ocdkit_io.path.os.path, "normpath", ntpath.normpath)
        adjusted = adjust_file_path("/Volumes/share/results")
        expected = ntpath.normpath(r"C:\Users\Tester\share\results")
        assert adjusted == expected

    def test_unknown_os_passthrough(self, monkeypatch):
        monkeypatch.setattr(ocdkit_io.path.platform, "system", lambda: "UnknownOS")
        assert adjust_file_path("/custom/path") == "/custom/path"


# ---------------------------------------------------------------------------
# find_files
# ---------------------------------------------------------------------------

class TestFindFiles:
    def test_exclude_suffixes(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "img_mask.tif").touch()
        (tmp_path / "img_mask_backup.tif").touch()
        (tmp_path / "sub" / "img_mask.tif").touch()
        matches = find_files(str(tmp_path), "_mask", exclude_suffixes=["_mask_backup"])
        assert len(matches) == 2

    def test_basic(self, tmp_path):
        (tmp_path / "a_seg.tif").touch()
        (tmp_path / "b_seg.tif").touch()
        (tmp_path / "c.tif").touch()
        matches = find_files(str(tmp_path), "_seg")
        assert len(matches) == 2


# ---------------------------------------------------------------------------
# get_image_files
# ---------------------------------------------------------------------------

class TestGetImageFiles:
    def test_filters_masks(self, tmp_path):
        img = np.zeros((8, 8), dtype=np.uint8)
        imwrite(str(tmp_path / "sample.tif"), img)
        imwrite(str(tmp_path / "sample_masks.tif"), img)
        imwrite(str(tmp_path / "other_img.tif"), img)

        files = get_image_files(str(tmp_path), mask_filter="_masks", img_filter="")
        names = {os.path.basename(p) for p in files}
        assert "sample.tif" in names
        assert "sample_masks.tif" not in names
        assert "other_img.tif" in names

    def test_pattern_filter(self, tmp_path):
        img = np.zeros((8, 8), dtype=np.uint8)
        imwrite(str(tmp_path / "alpha_img.tif"), img)
        imwrite(str(tmp_path / "beta_img.tif"), img)

        files = get_image_files(str(tmp_path), img_filter="_img", pattern="alpha_img")
        assert len(files) == 1
        assert files[0].endswith("alpha_img.tif")

    def test_look_one_level_down(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        img = np.zeros((4, 4), dtype=np.uint8)
        imwrite(str(sub / "deep.tif"), img)
        imwrite(str(tmp_path / "top.tif"), img)

        files = get_image_files(str(tmp_path), look_one_level_down=True, img_filter="")
        names = {os.path.basename(p) for p in files}
        assert "deep.tif" in names
        assert "top.tif" in names

    def test_no_images_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no images"):
            get_image_files(str(tmp_path))
