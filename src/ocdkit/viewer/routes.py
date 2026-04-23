"""HTTP route helpers + DebugAPI for the viewer.

Hosts the routes that depend on the active plugin and the in-process mask
cache. Generic mask ops (n-color, format-labels) live in :mod:`ocdkit.viewer.masks`.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .assets import append_gui_log
from .masks import compute_ncolor_mask, format_labels as _format_labels
from .segmentation import ACTIVE_PLUGIN, run_mask_update, run_segmentation
from .session import SESSION_COOKIE_NAME, SESSION_MANAGER, SessionState

WEBGL_LOG_PATH = Path("/tmp/ocdkit_viewer_webgl_log.txt")
try:
    WEBGL_LOG_PATH.write_text("", encoding="utf-8")
except OSError:
    pass


class DebugAPI:
    """Helper used by the desktop launcher (pywebview js_api) and the FastAPI
    routes. Wraps the plugin-aware operations into a class with a single log
    sink and consistent error envelopes."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or WEBGL_LOG_PATH

    def log(self, message: str) -> None:
        message = str(message)
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except OSError:
            return

    # -- segmentation -------------------------------------------------------

    def segment(self, settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
        mode = None
        if isinstance(settings, Mapping):
            mode = settings.get("mode")
        state = SESSION_MANAGER.get_or_create(None)
        if mode == "recompute":
            return run_mask_update(settings, state=state)
        return run_segmentation(settings, state=state)

    def resegment(self, settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
        state = SESSION_MANAGER.get_or_create(None)
        return run_mask_update(settings, state=state)

    # -- mask post-processing ----------------------------------------------

    def get_ncolor(self) -> dict[str, Any]:
        # Use cached encoding (issue #7). ACTIVE_PLUGIN.get_encoded_ncolor()
        # only re-encodes once per stored result.
        return {"nColorMask": ACTIVE_PLUGIN.get_encoded_ncolor()}

    def ncolor_from_mask(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}
        expand = bool(payload.get("expand", True))
        ncm = compute_ncolor_mask(mask, expand=expand)
        if ncm is None:
            return {"nColorMask": None}
        encoded = base64.b64encode(
            np.ascontiguousarray(ncm.astype(np.uint32, copy=False)).tobytes()
        ).decode("ascii")
        return {"nColorMask": encoded, "width": int(width), "height": int(height)}

    def format_labels(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}
        try:
            formatted = _format_labels(mask, clean=False, min_area=1)
        except Exception as exc:
            return {"error": f"format_labels failed: {exc}"}
        encoded = base64.b64encode(
            np.ascontiguousarray(formatted.astype(np.uint32, copy=False)).tobytes()
        ).decode("ascii")
        return {"mask": encoded, "width": int(width), "height": int(height)}

    # -- affinity-graph editing (delegates to plugin if it supports it) ----

    def relabel_from_affinity(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        plugin = ACTIVE_PLUGIN.current()
        if plugin is None or plugin.relabel_from_affinity is None:
            return {"error": "active plugin does not support relabel_from_affinity"}
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}

        ag = payload.get("affinityGraph")
        if not isinstance(ag, Mapping):
            return {"error": "affinityGraph required"}
        try:
            w = int(ag.get("width"))
            h = int(ag.get("height"))
            steps_list = ag.get("steps")
            enc = ag.get("encoded")
            if w <= 0 or h <= 0:
                return {"error": "invalid affinityGraph size"}
            if not isinstance(steps_list, list) or not isinstance(enc, str):
                return {"error": "invalid affinityGraph payload"}
            raw_aff = base64.b64decode(enc)
            arr_aff = np.frombuffer(raw_aff, dtype=np.uint8)
            s = len(steps_list)
            if arr_aff.size != s * h * w:
                return {"error": "affinityGraph data size mismatch"}
            spatial = arr_aff.reshape((s, h, w))
            steps = np.asarray(steps_list, dtype=np.int16)
        except Exception as exc:
            return {"error": f"invalid affinityGraph: {exc}"}

        try:
            new_labels = plugin.relabel_from_affinity(mask, spatial, steps)
        except Exception as exc:
            import traceback
            print("[relabel_from_affinity] EXCEPTION:", file=sys.stderr)
            traceback.print_exc()
            return {"error": f"{type(exc).__name__}: {exc}"}

        new_labels = np.asarray(new_labels, dtype=np.int32)
        ncm = compute_ncolor_mask(new_labels, expand=True)
        ACTIVE_PLUGIN.store_result(new_labels, ACTIVE_PLUGIN.get_last_extras(), ncolor=ncm)
        encoded_mask = base64.b64encode(
            np.ascontiguousarray(new_labels.astype(np.uint32, copy=False)).tobytes()
        ).decode("ascii")
        encoded_ncolor = None
        if ncm is not None:
            encoded_ncolor = base64.b64encode(
                np.ascontiguousarray(ncm.astype(np.uint32, copy=False)).tobytes()
            ).decode("ascii")
        return {
            "mask": encoded_mask,
            "nColorMask": encoded_ncolor,
            "affinityGraph": ACTIVE_PLUGIN.get_last_extras().get("affinityGraph"),
            "width": int(width),
            "height": int(height),
        }


def _choose_path_osascript(kind: str) -> Optional[str]:
    try:
        import subprocess
    except Exception:
        return None
    script = (
        'POSIX path of (choose file with prompt "Select image")'
        if kind == "file"
        else 'POSIX path of (choose folder with prompt "Select image folder")'
    )
    result = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    path_value = (result.stdout or "").strip()
    return path_value or None


__all__ = [
    "DebugAPI",
    "WEBGL_LOG_PATH",
    "_choose_path_osascript",
    "SESSION_COOKIE_NAME",
    "SessionState",
    "SESSION_MANAGER",
]
