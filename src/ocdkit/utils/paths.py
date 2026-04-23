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

from pathlib import Path
from typing import Callable

import platformdirs


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
