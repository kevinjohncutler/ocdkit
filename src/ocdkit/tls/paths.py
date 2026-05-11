"""Paths to ocdkit TLS config, CA, and leaf cert/key storage."""

from __future__ import annotations

from pathlib import Path

try:
    from platformdirs import user_config_dir
except ImportError:
    import os

    def user_config_dir(appname: str, appauthor: str = "") -> str:
        return os.path.expanduser(f"~/.config/{appname}")


def config_dir() -> Path:
    """Directory holding ocdkit TLS config, CA, and leaf cert/key files."""
    return Path(user_config_dir("ocdkit")) / "tls"


def config_path() -> Path:
    return config_dir() / "config.json"


def ca_dir() -> Path:
    return config_dir() / "ca"
