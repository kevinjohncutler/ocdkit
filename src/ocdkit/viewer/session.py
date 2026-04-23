"""Per-browser session state for the ocdkit viewer.

Tracks the current image, working directory, file list, and per-file saved
viewer state across navigations within one session.

Sessions are evicted via a combination of LRU cap and TTL, so a long-running
server cannot grow memory unboundedly as new browsers connect (issue #2).
"""

from __future__ import annotations

import base64
import io
import json
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Eviction limits — overridable via env vars in deployment.
import os as _os
SESSION_MAX_COUNT = int(_os.environ.get("OCDKIT_VIEWER_MAX_SESSIONS", "100"))
SESSION_TTL_SECONDS = float(_os.environ.get("OCDKIT_VIEWER_SESSION_TTL", "3600"))

import numpy as np
from imageio import v2 as imageio

from .sample_image import (
    _ensure_spatial_last,
    _normalize_uint8,
    get_instance_color_table,
    get_preload_image_path,
    load_image_uint8,
)

SUPPORTED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}

SESSION_COOKIE_NAME = "OCDSESSION"


def _session_path_key(path: Optional[Path]) -> str:
    return str(path.resolve()) if path else "__sample__"


@dataclass
class SessionState:
    session_id: str
    current_path: Optional[Path]
    directory: Optional[Path]
    files: list[Path] = field(default_factory=list)
    saved_states: dict[str, Any] = field(default_factory=dict)
    current_image: Optional[np.ndarray] = None
    image_is_rgb: bool = False
    encoded_image: Optional[str] = None
    encoded_image_bytes: Optional[bytes] = None
    encoded_image_mime: str = "image/png"
    last_seen: float = 0.0  # unix time of most recent access (for TTL eviction)

    def path_key(self, path: Optional[Path] = None) -> str:
        return _session_path_key(path if path is not None else self.current_path)


class SessionManager:
    """Thread-safe LRU+TTL bounded store of per-browser sessions.

    The store has two eviction policies that run together:
      * LRU cap (``SESSION_MAX_COUNT``) — drops least-recently-used entries
        when the cap is exceeded.
      * TTL (``SESSION_TTL_SECONDS``) — drops entries idle for longer than TTL.
    """

    def __init__(
        self,
        *,
        max_count: int = SESSION_MAX_COUNT,
        ttl_seconds: float = SESSION_TTL_SECONDS,
    ) -> None:
        # OrderedDict preserves insertion order → cheap LRU bump via move_to_end
        self._sessions: "OrderedDict[str, SessionState]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_count = max_count
        self._ttl_seconds = ttl_seconds

    # -- eviction helpers --------------------------------------------------

    def _evict_unlocked(self) -> None:
        """Drop expired (TTL) and excess (LRU) sessions. Caller holds lock."""
        if self._ttl_seconds > 0:
            now = time.time()
            stale = [sid for sid, st in self._sessions.items()
                     if now - st.last_seen > self._ttl_seconds]
            for sid in stale:
                self._sessions.pop(sid, None)
        while len(self._sessions) > self._max_count:
            self._sessions.popitem(last=False)

    def _touch_unlocked(self, state: SessionState) -> None:
        state.last_seen = time.time()
        self._sessions.move_to_end(state.session_id, last=True)

    # -- inspection (used by tests) ----------------------------------------

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _create_session_unlocked(self) -> SessionState:
        session_id = secrets.token_urlsafe(16)
        initial_path = get_preload_image_path()
        if initial_path and initial_path.exists():
            image, is_rgb = self._load_image_from_path(initial_path)
            directory = initial_path.parent
            files = self._list_directory_images(directory)
        else:
            image = load_image_uint8(as_rgb=True)
            is_rgb = image.ndim == 3 and image.shape[-1] >= 3
            directory = None
            files = []
            initial_path = None
        state = SessionState(
            session_id=session_id,
            current_path=initial_path,
            directory=directory,
            files=files,
            current_image=np.ascontiguousarray(image, dtype=np.uint8),
            image_is_rgb=is_rgb,
            encoded_image=None,
        )
        state.last_seen = time.time()
        self._sessions[session_id] = state
        raw_bytes = self._encode_image_bytes(state.current_image, is_rgb=is_rgb)
        state.encoded_image_bytes = raw_bytes
        state.encoded_image_mime = "image/png"
        state.encoded_image = (
            "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")
        )
        return state

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        with self._lock:
            self._evict_unlocked()
            if session_id and session_id in self._sessions:
                state = self._sessions[session_id]
                self._touch_unlocked(state)
                return state
            state = self._create_session_unlocked()
            self._evict_unlocked()  # respect cap if creation pushed us over
            return state

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions[session_id]
            self._touch_unlocked(state)
            return state

    def clear_saved_states(self, state: SessionState) -> None:
        with self._lock:
            existing = self._sessions.get(state.session_id)
            if existing:
                existing.saved_states.clear()

    def _load_image_from_path(self, path: Path) -> tuple[np.ndarray, bool]:
        arr = imageio.imread(path)
        arr = _ensure_spatial_last(arr)
        arr = _normalize_uint8(arr)
        is_rgb = arr.ndim == 3 and arr.shape[-1] >= 3
        return arr, is_rgb

    def _list_directory_images(self, directory: Path) -> list[Path]:
        try:
            return [
                p
                for p in sorted(directory.iterdir())
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
            ]
        except FileNotFoundError:
            return []

    def set_image(self, state: SessionState, path: Optional[Path]) -> None:
        if path is not None:
            path = path.expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            image, is_rgb = self._load_image_from_path(path)
            directory = path.parent
            files = self._list_directory_images(directory)
        else:
            image = load_image_uint8(as_rgb=True)
            is_rgb = image.ndim == 3 and image.shape[-1] >= 3
            directory = None
            files = []
        state.current_path = path
        state.directory = directory
        state.files = files
        state.current_image = np.ascontiguousarray(image, dtype=np.uint8)
        state.image_is_rgb = is_rgb
        raw_bytes = self._encode_image_bytes(state.current_image, is_rgb=is_rgb)
        state.encoded_image_bytes = raw_bytes
        state.encoded_image_mime = "image/png"
        state.encoded_image = (
            "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")
        )

    def build_config(
        self, state: SessionState, *, embed_image: bool = True
    ) -> dict[str, Any]:
        image = state.current_image if state.current_image is not None else load_image_uint8(as_rgb=True)
        is_rgb = state.image_is_rgb
        height, width = image.shape[:2]
        if not state.encoded_image:
            state.encoded_image = self._encode_image(image, is_rgb=is_rgb)
        directory_entries: list[dict[str, Any]] = []
        index = None
        if state.current_path and state.files:
            for i, item in enumerate(state.files):
                is_current = item == state.current_path
                if is_current:
                    index = i
                directory_entries.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "isCurrent": is_current,
                    }
                )
        config: dict[str, Any] = {
            "sessionId": state.session_id,
            "width": int(width),
            "height": int(height),
            "colorTable": get_instance_color_table().tolist(),
            "maskOpacity": 0.8,
            "maskThreshold": -2.0,
            "flowThreshold": 0.0,
            "cluster": True,
            "affinitySeg": True,
            "imagePath": str(state.current_path) if state.current_path else None,
            "imageName": state.current_path.name if state.current_path else "Sample Image",
            "directoryEntries": directory_entries,
            "directoryIndex": index,
            "directoryPath": str(state.directory) if state.directory else None,
            "hasPrev": bool(index is not None and index > 0),
            "hasNext": bool(index is not None and index < len(state.files) - 1),
            "isRgb": is_rgb,
            "useWebglPipeline": True,
        }
        if embed_image:
            config["imageDataUrl"] = state.encoded_image
        else:
            config["imageUrl"] = f"/api/image/{state.session_id}?t={int(time.time() * 1000)}"
        saved_state = state.saved_states.get(state.path_key())
        if saved_state:
            try:
                sanitized = json.loads(json.dumps(saved_state))
            except Exception:
                sanitized = saved_state
            config["savedViewerState"] = sanitized
            state.saved_states[state.path_key()] = sanitized
        return config

    def _encode_image(self, array: np.ndarray, *, is_rgb: bool) -> str:
        raw_bytes = self._encode_image_bytes(array, is_rgb=is_rgb)
        return "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")

    def _encode_image_bytes(self, array: np.ndarray, *, is_rgb: bool) -> bytes:
        buffer = io.BytesIO()
        if is_rgb and array.ndim == 3 and array.shape[-1] == 2:
            rgb = np.empty((*array.shape[:-1], 3), dtype=array.dtype)
            rgb[..., :2] = array
            rgb[..., 2] = 0
            imageio.imwrite(buffer, rgb, format="png", compress_level=1)
        else:
            imageio.imwrite(buffer, array, format="png", compress_level=1)
        return buffer.getvalue()

    def navigate(self, state: SessionState, delta: int) -> Optional[Path]:
        if not state.files or state.current_path is None:
            return None
        try:
            idx = state.files.index(state.current_path)
        except ValueError:
            return None
        target = idx + delta
        if target < 0 or target >= len(state.files):
            return None
        return state.files[target]

    def save_viewer_state(
        self,
        state: SessionState,
        image_path: Optional[Path],
        viewer_state: dict[str, Any],
    ) -> None:
        key = _session_path_key(image_path if image_path is not None else state.current_path)
        try:
            state.saved_states[key] = json.loads(json.dumps(viewer_state))
        except Exception:
            state.saved_states[key] = viewer_state


SESSION_MANAGER = SessionManager()
