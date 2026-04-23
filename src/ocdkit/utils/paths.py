"""Cross-platform user directories (config, data, cache, state, logs).

Thin wrapper around :mod:`platformdirs` that:

* returns :class:`pathlib.Path` instead of :class:`str`,
* creates the directory on access (unless ``create=False``),
* accepts trailing path parts so callers don't glue them by hand.

Platform resolution summary::

    user_config("myapp")
      macOS   → ~/Library/Application Support/myapp
      Linux   → $XDG_CONFIG_HOME/myapp          (default ~/.config/myapp)
      Windows → %APPDATA%\\myapp\\myapp

    user_data("myapp")
      macOS   → ~/Library/Application Support/myapp
      Linux   → $XDG_DATA_HOME/myapp            (default ~/.local/share/myapp)
      Windows → %LOCALAPPDATA%\\myapp\\myapp

    user_cache("myapp")
      macOS   → ~/Library/Caches/myapp
      Linux   → $XDG_CACHE_HOME/myapp           (default ~/.cache/myapp)
      Windows → %LOCALAPPDATA%\\myapp\\myapp\\Cache

    user_state("myapp")
      macOS   → ~/Library/Application Support/myapp
      Linux   → $XDG_STATE_HOME/myapp           (default ~/.local/state/myapp)
      Windows → %LOCALAPPDATA%\\myapp\\myapp

    user_log("myapp")
      macOS   → ~/Library/Logs/myapp
      Linux   → $XDG_STATE_HOME/myapp/log       (default ~/.local/state/myapp/log)
      Windows → %LOCALAPPDATA%\\myapp\\myapp\\Logs

Usage::

    from ocdkit.utils.paths import user_data, user_config

    models_dir = user_data("omnipose", "models")
    prefs_file = user_config("hiprpy") / "preferences.toml"
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import platformdirs

_log = logging.getLogger(__name__)
_MIGRATION_MARKER = ".migrated"


def _resolve(fn: Callable[..., str], app: str, parts: tuple[str, ...], create: bool) -> Path:
    # ``appauthor=False`` suppresses platformdirs' default Windows behaviour of
    # inserting an "author" path segment between %APPDATA% and the app name
    # (so we get ``%APPDATA%\myapp\`` rather than ``%APPDATA%\myapp\myapp\``).
    base = Path(fn(app, appauthor=False))
    path = base.joinpath(*parts) if parts else base
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def user_config(app: str, *parts: str, create: bool = True) -> Path:
    """Directory for user-editable configuration files."""
    return _resolve(platformdirs.user_config_dir, app, parts, create)


def user_data(app: str, *parts: str, create: bool = True) -> Path:
    """Directory for persistent app data (models, saved work, etc.)."""
    return _resolve(platformdirs.user_data_dir, app, parts, create)


def user_cache(app: str, *parts: str, create: bool = True) -> Path:
    """Directory for disposable caches."""
    return _resolve(platformdirs.user_cache_dir, app, parts, create)


def user_state(app: str, *parts: str, create: bool = True) -> Path:
    """Directory for volatile state (history, undo, recent files)."""
    return _resolve(platformdirs.user_state_dir, app, parts, create)


def user_log(app: str, *parts: str, create: bool = True) -> Path:
    """Directory for log files."""
    return _resolve(platformdirs.user_log_dir, app, parts, create)


def migrate_legacy_dotfolder(
    app: str,
    legacy: Optional[str] = None,
    *,
    marker: str = _MIGRATION_MARKER,
) -> Optional[Path]:
    """Move ``~/.<legacy>/`` contents into :func:`user_data` ``(app)`` once.

    Designed to be called every import — the per-call overhead after the
    first run is a single ``stat()`` on the destination marker file. Safe
    to invoke from package ``__init__`` modules.

    Semantics:

    * If ``~/.<legacy>`` does not exist → no-op, returns ``None``.
    * If ``<destination>/<marker>`` exists → already migrated, returns
      destination.
    * If the destination already has **other content** (not just the marker) →
      logs a warning and skips the move; user's existing data is never
      clobbered.  Still writes the marker so future calls are fast.
    * Otherwise → moves every child of ``~/.<legacy>/`` into the destination
      (preserves metadata via :func:`shutil.move`), writes the marker, and
      removes the now-empty legacy folder.

    Parameters
    ----------
    app:
        Application name (platformdirs key) — e.g. ``"omnipose"``.
    legacy:
        Dotfolder basename (without the leading dot).  Defaults to *app*.
    marker:
        Sentinel filename written into the destination to record that the
        migration already ran.  Default ``".migrated"``.

    Returns
    -------
    pathlib.Path or None
        The destination directory, or ``None`` if no legacy folder existed.
    """
    if legacy is None:
        legacy = app
    src = Path.home() / f".{legacy}"
    dst = _resolve(platformdirs.user_data_dir, app, (), create=False)
    marker_path = dst / marker

    # Fast path: already migrated.
    if marker_path.exists():
        return dst

    # Nothing to migrate.
    if not src.exists() or not src.is_dir():
        return None

    dst.mkdir(parents=True, exist_ok=True)

    # Refuse to clobber if the destination already has real content.
    existing = [p for p in dst.iterdir() if p.name != marker]
    if existing:
        _log.warning(
            "migrate_legacy_dotfolder: %s already has contents; "
            "leaving %s in place for manual review.", dst, src,
        )
        try:
            marker_path.touch()
        except OSError:
            pass
        return dst

    # Move every child (preserves metadata, handles cross-device moves).
    moved_any = False
    for child in list(src.iterdir()):
        target = dst / child.name
        try:
            shutil.move(str(child), str(target))
            moved_any = True
        except OSError as exc:
            _log.error("migrate_legacy_dotfolder: could not move %s → %s: %s",
                       child, target, exc)

    # Record completion even if nothing moved (empty legacy dir).
    try:
        marker_path.touch()
    except OSError:
        pass

    # Remove the now-empty legacy dir (best-effort).
    try:
        src.rmdir()
    except OSError:
        pass

    if moved_any:
        _log.info("migrate_legacy_dotfolder: moved %s → %s", src, dst)
    return dst
