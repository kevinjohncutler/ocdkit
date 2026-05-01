"""Tests for ocdkit.utils.paths.migrate_legacy_dotfolder."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocdkit.utils import paths as paths_mod
from ocdkit.utils.paths import migrate_legacy_dotfolder


@pytest.fixture
def iso_home(tmp_path, monkeypatch):
    """Redirect Home AND platformdirs-user_data to a clean temp dir.

    Patching ``Path.home`` directly is portable: on Windows, ``Path.home()``
    reads ``USERPROFILE``/``HOMEDRIVE``/``HOMEPATH`` rather than ``HOME``,
    so just setting ``HOME`` doesn't redirect it.
    """
    fake_home = tmp_path / "home"
    fake_data = tmp_path / "data"
    fake_home.mkdir()
    fake_data.mkdir()

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(
        paths_mod.platformdirs, "user_data_dir",
        lambda app, appauthor=False: str(fake_data / app),
    )
    return fake_home, fake_data


def test_noop_when_legacy_missing(iso_home):
    fake_home, fake_data = iso_home
    assert migrate_legacy_dotfolder("myapp") is None


def test_moves_contents_and_writes_marker(iso_home):
    fake_home, fake_data = iso_home
    legacy = fake_home / ".myapp"
    legacy.mkdir()
    (legacy / "models").mkdir()
    (legacy / "models" / "cell.pth").write_bytes(b"fake-model")
    (legacy / "config.txt").write_text("hello")

    dest = migrate_legacy_dotfolder("myapp")

    assert dest == fake_data / "myapp"
    assert (dest / "models" / "cell.pth").read_bytes() == b"fake-model"
    assert (dest / "config.txt").read_text() == "hello"
    assert (dest / ".migrated").exists()
    assert not legacy.exists()


def test_marker_makes_subsequent_calls_noop(iso_home):
    fake_home, fake_data = iso_home
    legacy = fake_home / ".myapp"
    legacy.mkdir()
    (legacy / "file").write_text("x")

    migrate_legacy_dotfolder("myapp")
    assert not legacy.exists()

    # Recreate legacy and verify it's left alone.
    legacy.mkdir()
    (legacy / "new_file").write_text("y")
    migrate_legacy_dotfolder("myapp")
    assert legacy.exists()
    assert (legacy / "new_file").exists()
    assert not (fake_data / "myapp" / "new_file").exists()


def test_custom_legacy_name(iso_home):
    fake_home, fake_data = iso_home
    (fake_home / ".cellpose").mkdir()
    (fake_home / ".cellpose" / "model.pt").write_bytes(b"m")

    dest = migrate_legacy_dotfolder("omnipose", legacy="cellpose")

    assert (dest / "model.pt").exists()
    assert not (fake_home / ".cellpose").exists()


def test_does_not_clobber_existing_destination(iso_home, caplog):
    fake_home, fake_data = iso_home
    legacy = fake_home / ".myapp"
    legacy.mkdir()
    (legacy / "old.txt").write_text("legacy")

    dst = fake_data / "myapp"
    dst.mkdir()
    (dst / "new.txt").write_text("already here")

    with caplog.at_level("WARNING", logger="ocdkit.utils.paths"):
        result = migrate_legacy_dotfolder("myapp")

    assert result == dst
    assert (dst / "new.txt").read_text() == "already here"
    assert not (dst / "old.txt").exists()
    assert (dst / ".migrated").exists()
    assert legacy.exists()
    assert any("already has contents" in r.message for r in caplog.records)


def test_empty_legacy_dir_still_records_marker(iso_home):
    fake_home, fake_data = iso_home
    (fake_home / ".myapp").mkdir()

    dest = migrate_legacy_dotfolder("myapp")

    assert dest == fake_data / "myapp"
    assert (dest / ".migrated").exists()
    assert not (fake_home / ".myapp").exists()
