"""Persistent registry for user-added custom segmentation model files.

Stores entries as a JSON array under ``~/.ocdkit/models/<plugin>/custom_models.json``
or, if no plugin is given, ``~/.ocdkit/models/custom_models.json``. Each entry
is ``{"name": "...", "path": "..."}``.

Per-plugin namespacing prevents Cellpose models from colliding with StarDist
models, etc.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

_BASE_DIR = Path.home() / ".ocdkit" / "models"
_lock = threading.Lock()


def _registry_dir(plugin: Optional[str]) -> Path:
    if plugin:
        return _BASE_DIR / plugin
    return _BASE_DIR


def _registry_file(plugin: Optional[str]) -> Path:
    return _registry_dir(plugin) / "custom_models.json"


def _read_entries(plugin: Optional[str]) -> list[dict[str, str]]:
    try:
        data = json.loads(_registry_file(plugin).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [
                e for e in data if isinstance(e, dict) and "name" in e and "path" in e
            ]
    except Exception:
        pass
    return []


def _write_entries(entries: list[dict[str, str]], plugin: Optional[str]) -> None:
    target_dir = _registry_dir(plugin)
    target_dir.mkdir(parents=True, exist_ok=True)
    _registry_file(plugin).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def list_models(plugin: Optional[str] = None) -> list[dict[str, Any]]:
    with _lock:
        entries = _read_entries(plugin)
    return [
        {"name": e["name"], "path": e["path"], "exists": Path(e["path"]).exists()}
        for e in entries
    ]


def add_model(name: str, path: str, plugin: Optional[str] = None) -> dict[str, Any]:
    with _lock:
        entries = _read_entries(plugin)
        for entry in entries:
            if entry["name"] == name:
                entry["path"] = path
                _write_entries(entries, plugin)
                return {"name": name, "path": path, "exists": Path(path).exists()}
        entries.append({"name": name, "path": path})
        _write_entries(entries, plugin)
    return {"name": name, "path": path, "exists": Path(path).exists()}


def remove_model(
    name: str, *, plugin: Optional[str] = None, delete_file: bool = False
) -> bool:
    with _lock:
        entries = _read_entries(plugin)
        before = len(entries)
        removed_path = None
        kept: list[dict[str, str]] = []
        for e in entries:
            if e["name"] == name:
                removed_path = e["path"]
            else:
                kept.append(e)
        if len(kept) == before:
            return False
        _write_entries(kept, plugin)
    if delete_file and removed_path:
        try:
            Path(removed_path).unlink(missing_ok=True)
        except Exception:
            pass
    return True
