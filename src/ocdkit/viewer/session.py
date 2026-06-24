"""Per-browser session state for the ocdkit viewer.

Tracks the current image, working directory, file list, and per-file saved
viewer state across navigations within one session.

Sessions are evicted via a combination of LRU cap and TTL, so a long-running
server cannot grow memory unboundedly as new browsers connect (issue #2).
"""

from __future__ import annotations

import base64
import gzip
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


def _encode_array(arr: np.ndarray, *, level: int = 1) -> dict[str, Any]:
    """Pack an ndarray as ``{dtype, shape, gzip, b64}`` (C-order bytes).

    Matches omnipose ``_volume3d.encode_array`` / the JS ``decodeField`` so the
    3D viewer can consume a bundle built straight from session state.
    """
    arr = np.ascontiguousarray(arr)
    raw = gzip.compress(arr.tobytes(), level)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "gzip": True,
        "b64": base64.b64encode(raw).decode("ascii"),
    }


def _narrow_labels(arr: np.ndarray) -> np.ndarray:
    """Downcast a label volume to the smallest uint that holds its max label."""
    arr = np.asarray(arr)
    m = int(arr.max()) if arr.size else 0
    dt = np.uint8 if m <= 0xFF else (np.uint16 if m <= 0xFFFF else np.uint32)
    return arr.astype(dt, copy=False)


@dataclass
class SessionState:
    session_id: str
    current_path: Optional[Path]
    directory: Optional[Path]
    files: list[Path] = field(default_factory=list)
    saved_states: dict[str, Any] = field(default_factory=dict)
    current_image: Optional[np.ndarray] = None
    image_is_rgb: bool = False
    current_volume: Optional[np.ndarray] = None  # (Z, Y, X) uint8 when a 3D stack is loaded
    volume_slice: int = 0  # index of the slice currently shown in the 2D view
    current_mask_volume: Optional[np.ndarray] = None  # (Z, Y, X) label volume, if loaded
    current_ncolor_volume: Optional[np.ndarray] = None  # (Z, Y, X) ncolor group volume (cache)
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

    def _apply_image(self, state: SessionState, image: np.ndarray,
                     is_rgb: bool, is_volume: bool) -> None:
        """Store a loaded image/volume on the session + (re)encode the 2D view.

        For a volume the full ``(Z, Y, X)`` stack is kept and the middle slice
        feeds the 2D pipeline. Shared by the create and set-image paths so a
        preloaded volume behaves the same as one opened later.
        """
        if is_volume:
            state.current_volume = np.ascontiguousarray(image, dtype=np.uint8)
            state.volume_slice = int(state.current_volume.shape[0] // 2)
            state.current_image = np.ascontiguousarray(
                state.current_volume[state.volume_slice], dtype=np.uint8
            )
            state.image_is_rgb = False
        else:
            state.current_volume = None
            state.volume_slice = 0
            state.current_image = np.ascontiguousarray(image, dtype=np.uint8)
            state.image_is_rgb = is_rgb
        raw_bytes = self._encode_image_bytes(state.current_image, is_rgb=state.image_is_rgb)
        state.encoded_image_bytes = raw_bytes
        state.encoded_image_mime = "image/png"
        state.encoded_image = (
            "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")
        )

    def _create_session_unlocked(self) -> SessionState:
        session_id = secrets.token_urlsafe(16)
        initial_path = get_preload_image_path()
        if initial_path and initial_path.exists():
            image, is_rgb, is_volume = self._load_image_from_path(initial_path)
            directory = initial_path.parent
            files = self._list_directory_images(directory)
        else:
            image = load_image_uint8(as_rgb=True)
            is_rgb = image.ndim == 3 and image.shape[-1] >= 3
            is_volume = False
            directory = None
            files = []
            initial_path = None
        state = SessionState(
            session_id=session_id,
            current_path=initial_path,
            directory=directory,
            files=files,
            encoded_image=None,
        )
        self._apply_image(state, image, is_rgb, is_volume)
        self._maybe_auto_mask(state)
        state.last_seen = time.time()
        self._sessions[session_id] = state
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

    def _load_image_from_path(self, path: Path) -> tuple[np.ndarray, bool, bool]:
        """Read an image or volume. Returns ``(array, is_rgb, is_volume)``.

        A 3-D array is treated as a ``(Z, Y, X)`` volume only when neither the
        first nor the last axis is a small channel count — so ``(Y, X, 3)`` RGB
        and ``(3, Y, X)`` channels-first stay 2-D images.
        """
        arr = np.asarray(imageio.imread(path))
        if (arr.ndim == 3 and arr.shape[0] not in (1, 3, 4)
                and arr.shape[-1] not in (1, 2, 3, 4)):
            vol = _normalize_uint8(arr)  # global normalize across the whole stack
            return vol, False, True
        arr = _ensure_spatial_last(arr)
        arr = _normalize_uint8(arr)
        is_rgb = arr.ndim == 3 and arr.shape[-1] >= 3
        return arr, is_rgb, False

    def set_mask(self, state: SessionState, path: Path) -> None:
        """Load a label volume from *path* and attach it to the session.

        Masks are kept as integer labels (not normalized); their spatial shape
        must match the loaded volume. Feeds the 3D bundle's label layer.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if state.current_volume is None:
            raise ValueError("masks require a loaded volume")
        arr = np.asarray(imageio.imread(path))
        if tuple(arr.shape) != tuple(state.current_volume.shape):
            raise ValueError(
                f"mask shape {tuple(arr.shape)} does not match "
                f"volume shape {tuple(state.current_volume.shape)}"
            )
        state.current_mask_volume = _narrow_labels(arr)
        state.current_ncolor_volume = None

    def _auto_mask_path(self, image_path: Optional[Path]) -> Optional[Path]:
        """Find a default mask: env override, else a ``*_masks`` / ``*_cp_masks`` sidecar."""
        env = _os.environ.get("OCDKIT_VIEWER_SAMPLE_MASKS")
        if env:
            p = Path(env).expanduser()
            if p.is_file():
                return p
        if image_path is None:
            return None
        for suffix in ("_masks", "_cp_masks"):
            cand = image_path.with_name(image_path.stem + suffix + image_path.suffix)
            if cand.is_file():
                return cand
        return None

    def _maybe_auto_mask(self, state: SessionState) -> None:
        """Auto-attach a sidecar/env mask to a freshly loaded volume (best effort)."""
        state.current_mask_volume = None
        state.current_ncolor_volume = None
        if state.current_volume is None:
            return
        mp = self._auto_mask_path(state.current_path)
        if mp is not None:
            try:
                self.set_mask(state, mp)
            except Exception:
                pass  # mismatched or unreadable sidecar → leave unmasked

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
            image, is_rgb, is_volume = self._load_image_from_path(path)
            directory = path.parent
            files = self._list_directory_images(directory)
        else:
            image = load_image_uint8(as_rgb=True)
            is_rgb = image.ndim == 3 and image.shape[-1] >= 3
            is_volume = False
            directory = None
            files = []
        state.current_path = path
        state.directory = directory
        state.files = files
        self._apply_image(state, image, is_rgb, is_volume)
        self._maybe_auto_mask(state)

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
        if state.current_volume is not None:
            config["isVolume"] = True
            config["volumeDepth"] = int(state.current_volume.shape[0])
            config["volumeShape"] = [int(s) for s in state.current_volume.shape]  # [D, H, W]
            config["currentSlice"] = int(state.volume_slice)
            config["hasVolumeMask"] = state.current_mask_volume is not None
        else:
            config["isVolume"] = False
            config["hasVolumeMask"] = False
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

    def encode_slice_png(self, state: SessionState, z: int, axis: int = 0) -> Optional[bytes]:
        """PNG bytes for slice ``z`` along ``axis`` (0=Z, 1=Y, 2=X), or None.

        Updates ``state.volume_slice`` for the Z axis so the config stays in sync.
        """
        vol = state.current_volume
        if vol is None:
            return None
        axis = int(axis) % 3
        z = max(0, min(int(z), vol.shape[axis] - 1))
        if axis == 0:
            state.volume_slice = z
        sl = np.ascontiguousarray(np.take(vol, z, axis=axis))
        return self._encode_image_bytes(sl, is_rgb=False)

    def ensure_ncolor(self, state: SessionState) -> Optional[np.ndarray]:
        """Compute + cache the volume ncolor group volume (adjacent cells get
        different small group IDs, computed once on the whole 3D volume so the
        coloring is consistent across slices and matches the 3D render)."""
        if state.current_ncolor_volume is not None:
            return state.current_ncolor_volume
        mv = state.current_mask_volume
        if mv is None:
            return None
        try:
            import ncolor
            g = _narrow_labels(np.asarray(ncolor.label(mv)))
        except Exception:
            g = _narrow_labels(mv)   # fallback: raw labels
        state.current_ncolor_volume = g
        return g

    def ncolor_map(self, state: SessionState):
        """label → ncolor group as a list indexed by label (0 unused). Lets the
        2D view color each label by its volume-consistent group while keeping the
        label itself as the editable source of truth."""
        mv = state.current_mask_volume
        if mv is None:
            return None
        g = self.ensure_ncolor(state)
        maxl = int(mv.max())
        out = np.zeros(maxl + 1, dtype=np.int32)
        flat_l = mv.reshape(-1)
        nz = flat_l > 0
        out[flat_l[nz]] = g.reshape(-1)[nz]   # ncolor is per-label, so last-wins is fine
        return out.tolist()

    def paint_sphere(self, state: SessionState, z: int, axis: int, label: int,
                     radius: int, footprint: np.ndarray) -> None:
        """Extrude a 2D stroke ``footprint`` (bool, slice-shaped) into a ball:
        paint ``label`` on slices ``z±k`` eroded by k, for the 3D brush."""
        vol = state.current_volume
        if vol is None:
            raise ValueError("no volume loaded")
        from scipy import ndimage
        axis = int(axis) % 3
        D = vol.shape[axis]
        mv = state.current_mask_volume
        if mv is None:
            mv = np.zeros(vol.shape, np.uint32)
            state.current_mask_volume = mv
        if int(label) > int(np.iinfo(mv.dtype).max):
            mv = state.current_mask_volume = mv.astype(np.uint32)
        radius = max(0, int(radius))
        for k in range(0, radius + 1):
            fk = footprint if k == 0 else ndimage.binary_erosion(footprint, iterations=k)
            if not fk.any():
                break
            for zz in ({z} if k == 0 else {z - k, z + k}):
                if 0 <= zz < D:
                    plane = np.take(mv, zz, axis=axis)
                    plane[fk] = label
                    idx = [slice(None)] * 3
                    idx[axis] = zz
                    mv[tuple(idx)] = plane
        state.current_ncolor_volume = None

    def mask_slice(self, state: SessionState, z: int, axis: int = 0, kind: str = "group"):
        """Raw label bytes for mask slice ``z`` along ``axis`` →
        ``(bytes, width, height, dtype)``. ``kind='group'`` returns the ncolor
        group values (for display); ``kind='instance'`` returns identity labels."""
        src = state.current_mask_volume if kind == "instance" else self.ensure_ncolor(state)
        if src is None:
            return None
        axis = int(axis) % 3
        z = max(0, min(int(z), src.shape[axis] - 1))
        sl = np.ascontiguousarray(np.take(src, z, axis=axis))
        return sl.tobytes(), int(sl.shape[1]), int(sl.shape[0]), str(sl.dtype)

    def set_mask_slice(self, state: SessionState, z: int, data: bytes, dtype: str,
                       axis: int = 0) -> None:
        """Write edited label bytes back into mask slice ``z`` along ``axis``
        (creates the mask volume on the first edit; widens dtype as needed)."""
        vol = state.current_volume
        if vol is None:
            raise ValueError("no volume loaded")
        axis = int(axis) % 3
        shp = list(vol.shape)
        z = max(0, min(int(z), shp[axis] - 1))
        sshape = [shp[a] for a in range(3) if a != axis]   # 2D slice shape for this axis
        incoming = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(sshape)
        mv = state.current_mask_volume
        if mv is None:
            mv = np.zeros(tuple(shp), np.uint32)
            state.current_mask_volume = mv
        if int(incoming.max(initial=0)) > int(np.iinfo(mv.dtype).max):
            mv = state.current_mask_volume = mv.astype(np.uint32)
        idx = [slice(None)] * 3
        idx[axis] = z
        mv[tuple(idx)] = incoming.astype(mv.dtype, copy=False)
        state.current_ncolor_volume = None    # mask changed → recompute ncolor

    def encode_volume_bundle(self, state: SessionState) -> Optional[dict[str, Any]]:
        """Intensity-only 3D viewer bundle from the loaded volume, or None.

        Lets a raw (Z, Y, X) stack render in the 3D view before any
        segmentation — masks/overlays are added later via the plugin bundle.
        """
        vol = state.current_volume
        if vol is None:
            return None
        D, H, W = vol.shape
        bundle = {
            "meta": {
                "dim": 3,
                "axes": ["t", "y", "x"],
                "depth": int(D),
                "height": int(H),
                "width": int(W),
            },
            "image": _encode_array(vol),
        }
        if state.current_mask_volume is not None:
            # color the 3D volume by the same ncolor groups as the 2D slices
            g = self.ensure_ncolor(state)
            bundle["mask"] = _encode_array(g if g is not None
                                           else _narrow_labels(state.current_mask_volume))
        return bundle

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
