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
    label_group: Optional[dict] = None  # stable label→group map (keeps colors fixed across edits)
    undo_stack: Optional[list] = None  # diff-based mask edit history (server is the source of truth)
    redo_stack: Optional[list] = None
    mask_source_path: Optional[Path] = None  # original mask file; edits autosave to its *_edited sibling
    save_timer: Any = field(default=None, compare=False, repr=False)  # debounce timer for autosave
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

    def set_mask(self, state: SessionState, path: Path,
                 source_path: Optional[Path] = None) -> None:
        """Load a label volume from *path* and attach it to the session.

        Masks are kept as integer labels (not normalized); their spatial shape
        must match the loaded volume. Feeds the 3D bundle's label layer.

        ``source_path`` is the ORIGINAL mask file edits are derived from (used to
        compute the ``*_edited`` autosave target); defaults to *path*. When
        resuming from an already-edited file we load *path* (the edited one) but
        keep ``source_path`` pointing at the original so we keep one edited file.
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
        state.label_group = None   # fresh mask → fresh stable color map
        state.undo_stack = None
        state.redo_stack = None
        state.mask_source_path = Path(source_path or path).expanduser().resolve()

    def _edited_mask_path(self, source: Path) -> Path:
        """``foo_masks.tif`` → ``foo_masks_edited.tif`` (never overwrites the
        original). An already-``_edited`` source maps to itself, so resuming +
        re-saving keeps a single edited file (no ``_edited_edited``)."""
        source = Path(source)
        if source.stem.endswith("_edited"):
            return source
        return source.with_name(source.stem + "_edited" + source.suffix)

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
        """Auto-attach a sidecar/env mask to a freshly loaded volume (best effort).
        If a prior ``*_edited`` autosave exists, resume from it instead of the
        original — so reopening the volume continues your edits automatically."""
        state.current_mask_volume = None
        state.current_ncolor_volume = None
        state.label_group = None
        state.undo_stack = None
        state.redo_stack = None
        state.mask_source_path = None
        if state.current_volume is None:
            return
        mp = self._auto_mask_path(state.current_path)
        if mp is not None:
            edited = self._edited_mask_path(mp)
            load = edited if edited.exists() else mp
            try:
                self.set_mask(state, load, source_path=mp)   # resume from edited; keep original as source
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
        """Stable ncolor group volume. The label→group map is cached and kept
        FIXED across edits, so existing cells never change color when you draw;
        only NEW labels are assigned a (non-conflicting) group incrementally.
        Computed once on the whole 3D volume so 2D slices and 3D match."""
        if state.current_ncolor_volume is not None:
            return state.current_ncolor_volume
        mv = state.current_mask_volume
        if mv is None:
            return None
        maxl = int(mv.max())
        present = set(int(x) for x in np.unique(mv) if x > 0)
        cache = dict(state.label_group) if state.label_group else {}
        if not cache:
            # first time: a full ncolor pass seeds the stable map
            try:
                import ncolor
                g = np.asarray(ncolor.label(mv))
                lut0 = np.zeros(maxl + 1, dtype=np.int64)
                flat, gf = mv.reshape(-1), g.reshape(-1)
                nz = flat > 0
                lut0[flat[nz]] = gf[nz]
                for lab in present:
                    cache[lab] = int(lut0[lab]) or 1
            except Exception:
                for lab in present:
                    cache[lab] = ((lab - 1) % 7) + 1
        else:
            new_labels = present - set(cache.keys())
            if new_labels:
                from scipy import ndimage
                for nl in sorted(new_labels):
                    region = mv == nl
                    nbrs = np.unique(mv[ndimage.binary_dilation(region)])
                    used = {cache.get(int(x)) for x in nbrs if int(x) > 0 and int(x) != nl}
                    grp = 1
                    while grp in used:
                        grp += 1
                    cache[nl] = grp
        state.label_group = cache
        lut = np.zeros(maxl + 1, dtype=np.int64)
        for lab, grp in cache.items():
            if 0 < lab <= maxl:
                lut[lab] = grp
        g = _narrow_labels(lut[mv])
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

    def paint_sphere(self, state: SessionState, z: int, axis: int, group: int,
                     radius: int, footprint: np.ndarray) -> int:
        """Paint a stroke in ncolor COLOR space. The footprint is painted with
        ncolor ``group`` (the selected colour) and MERGED into adjacent cells of
        the same group (extending them) — "merge by visible colour". An isolated
        stroke becomes a region of that *same* colour (never auto-recoloured).
        ``group == 0`` erases.

        When ``radius > 0`` the stroke is extruded into a TRUE Euclidean ball
        (cross-section radius ``sqrt(R²-k²)`` at slice offset k), not the diamond
        the old iterative cityblock erosion produced. Returns the affected label.
        """
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
        self.ensure_ncolor(state)
        lg = state.label_group if state.label_group is not None else {}
        radius = max(0, int(radius))
        group = int(group)

        # Slices to paint: a TRUE Euclidean ball. Find the stroke's medial axis
        # (centreline) and, at slice offset k, dilate it by radius sqrt(R²-k²) —
        # cross-sections shrink as a sphere, giving a ball (or a swept ball / capsule
        # for a dragged stroke), not the octahedral diamond cityblock erosion gives.
        R = float(radius)
        if radius > 0:
            edt = ndimage.distance_transform_edt(footprint)
            # The medial axis = the RIDGE of the EDT (its local maxima), so EVERY
            # section of the stroke contributes to the z-extrusion: thick dabs, thin
            # connectors from fast motion, and clipped slivers at the canvas edge
            # alike. A global depth threshold missed anything shallower than it,
            # leaving those sections a single 2D plane.
            core = (edt >= 0.5) & (edt >= ndimage.maximum_filter(edt, size=3))
            dist_to_core = ndimage.distance_transform_edt(~core)
        slices = []
        for k in range(0, radius + 1):
            if k == 0:
                fk = footprint
            else:
                rk2 = R * R - k * k
                if rk2 <= 0:
                    break
                fk = dist_to_core <= rk2 ** 0.5                # disk of radius sqrt(R²-k²)
            if not fk.any():
                break
            for zz in ({z} if k == 0 else {z - k, z + k}):
                if 0 <= zz < D:
                    slices.append((zz, fk))
        if not slices:
            return 0

        if int(mv.max()) + 1 > int(np.iinfo(mv.dtype).max):
            mv = state.current_mask_volume = mv.astype(np.uint32)

        before_mv = mv.copy()                          # snapshot for undo (diffed below)
        before_lg = dict(lg)

        if group <= 0:
            target = 0
        else:
            # adjacent cells already of this colour → extend/merge them
            adj = set()
            for zz, fk in slices:
                plane = np.take(mv, zz, axis=axis)
                border = ndimage.binary_dilation(fk) & ~fk
                for l in np.unique(plane[border]):
                    li = int(l)
                    if li > 0 and lg.get(li) == group:
                        adj.add(li)
            if adj:
                same = sorted(adj)
                target = same[0]
                for l in same[1:]:                 # a stroke bridging two same-colour cells merges them
                    mv[mv == l] = target
                    lg.pop(l, None)
            else:
                target = int(mv.max()) + 1         # isolated → new region, SAME colour
            lg[target] = group                     # keep the chosen colour (never recoloured)
            state.label_group = lg

        for zz, fk in slices:
            idx = [slice(None)] * 3
            idx[axis] = zz
            plane = mv[tuple(idx)]
            plane[fk] = target
            mv[tuple(idx)] = plane
        state.current_ncolor_volume = None         # rebuild group volume (label_group kept)
        self._record_edit(state, before_mv, before_lg)
        self.schedule_autosave(state)
        return int(target)

    # ----- whole-cell ops: 3D colour picker + 3D fill (merge / delete) -----

    def label_at(self, state: SessionState, z: int, axis: int, y: int, x: int):
        """The (label, group) at a voxel — the 3D colour picker. Picks a whole
        cell's identity + colour from one click on any slice/axis."""
        mv = state.current_mask_volume
        if mv is None:
            return 0, 0
        plane = np.take(mv, int(z), axis=int(axis) % 3)
        H, W = plane.shape
        if not (0 <= int(y) < H and 0 <= int(x) < W):
            return 0, 0
        lab = int(plane[int(y), int(x)])
        self.ensure_ncolor(state)
        return lab, int((state.label_group or {}).get(lab, 0))

    def fill_cell(self, state: SessionState, z: int, axis: int, y: int, x: int,
                  group: int = -1, target_label: int = 0, erase: bool = False) -> int:
        """Fill from a 2D-slice click — acts on the CONNECTED COMPONENT under the
        cursor (the contiguous region), not every voxel that merely shares the
        label (a spacetime label can span disconnected blobs across time)."""
        mv = state.current_mask_volume
        if mv is None:
            return 0
        axis = int(axis) % 3
        idx = [0, 0, 0]
        idx[axis] = int(z)
        others = [i for i in range(3) if i != axis]
        idx[others[0]] = int(y)
        idx[others[1]] = int(x)
        return self._fill_component(state, tuple(idx), group=group,
                                    target_label=target_label, erase=erase)

    def fill_ray(self, state: SessionState, ro, rd, box_min, box_max,
                 group: int = -1, target_label: int = 0, erase: bool = False) -> int:
        """3D-view fill: ray-pick the cell under the cursor, then fill its connected
        component (contiguous region only)."""
        voxel = self._march_ray(state, ro, rd, box_min, box_max)
        if voxel is None:
            return 0
        return self._fill_component(state, voxel, group=group,
                                    target_label=target_label, erase=erase)

    def _fill_component(self, state: SessionState, voxel, group: int = -1,
                        target_label: int = 0, erase: bool = False) -> int:
        """Core op restricted to the connected component of the clicked label that
        contains ``voxel`` = (vz, vy, vx). Only the contiguous region is deleted /
        merged / recoloured. An isolated recolour splits the component into its own
        new cell so other components of the label keep their identity."""
        from scipy import ndimage
        mv = state.current_mask_volume
        if mv is None:
            return 0
        vz, vy, vx = int(voxel[0]), int(voxel[1]), int(voxel[2])
        if not (0 <= vz < mv.shape[0] and 0 <= vy < mv.shape[1] and 0 <= vx < mv.shape[2]):
            return 0
        L = int(mv[vz, vy, vx])
        if L <= 0:
            return 0
        self.ensure_ncolor(state)
        lg = state.label_group if state.label_group is not None else {}
        comp, _ = ndimage.label(mv == L, structure=np.ones((3, 3, 3), dtype=int))
        cell = comp == comp[vz, vy, vx]                # the contiguous piece only
        before_mv = mv.copy()
        before_lg = dict(lg)
        if erase:
            target = 0
        elif int(target_label) > 0 and int(target_label) != L:
            target = int(target_label)                 # identity merge into the picked cell
        else:
            g = int(group)
            if g <= 0:
                return L
            border = ndimage.binary_dilation(cell) & ~cell
            adj = [int(n) for n in np.unique(mv[border]) if n > 0 and lg.get(int(n)) == g and int(n) != L]
            if adj:
                target = min(adj)                      # colour-merge into a touching same-colour cell
            elif int(cell.sum()) == int((mv == L).sum()):
                lg[L] = g                              # whole cell is this one component → recolour in place
                state.label_group = lg
                state.current_ncolor_volume = None
                self._record_edit(state, before_mv, before_lg)
                self.schedule_autosave(state)
                return L
            else:
                target = int(mv.max()) + 1             # only a piece → split it into a new cell of that colour
                lg[target] = g
        mv[cell] = target
        if not (mv == L).any():
            lg.pop(L, None)                            # label fully consumed
        if target > 0 and target not in lg:
            lg[target] = int(group) if int(group) > 0 else 1
        state.label_group = lg
        state.current_ncolor_volume = None
        self._record_edit(state, before_mv, before_lg)
        self.schedule_autosave(state)
        return target

    def fill_label(self, state: SessionState, label: int, group: int = -1,
                   target_label: int = 0, erase: bool = False) -> int:
        """Whole-cell 3D op on label ``label`` (its entire extent across the volume):

        - ``erase`` → delete the cell.
        - ``target_label > 0`` → merge into that label (identity merge; non-touching OK).
        - else ``group >= 0`` → recolour, colour-merging into a touching same-colour cell."""
        from scipy import ndimage
        mv = state.current_mask_volume
        L = int(label)
        if mv is None or L <= 0:
            return 0
        cell = mv == L
        if not cell.any():
            return 0
        self.ensure_ncolor(state)
        lg = state.label_group if state.label_group is not None else {}
        before_mv = mv.copy()
        before_lg = dict(lg)
        if erase:
            target = 0
        elif int(target_label) > 0 and int(target_label) != L:
            target = int(target_label)                 # identity merge into the picked cell
        else:
            g = int(group)
            if g <= 0:
                return L                               # no colour/target → nothing to do
            border = ndimage.binary_dilation(cell) & ~cell
            adj = [int(n) for n in np.unique(mv[border]) if n > 0 and lg.get(int(n)) == g and int(n) != L]
            if adj:
                target = min(adj)                      # colour-merge into a touching same-colour cell
            else:
                lg[L] = g                              # isolated → just recolour the cell
                state.label_group = lg
                state.current_ncolor_volume = None
                self._record_edit(state, before_mv, before_lg)
                self.schedule_autosave(state)
                return L
        mv[cell] = target
        lg.pop(L, None)
        if target > 0 and target not in lg:
            lg[target] = 1
        state.label_group = lg
        state.current_ncolor_volume = None
        self._record_edit(state, before_mv, before_lg)
        self.schedule_autosave(state)
        return target

    def _march_ray(self, state: SessionState, ro, rd, box_min, box_max):
        """March a world-space ray exactly as the render shader (n = (p-boxMin)/span,
        sample mv[z,y,x]); return the first labelled voxel ``(vz, vy, vx)`` or None.
        Identical coordinate math → the hit is the cell drawn under the cursor."""
        mv = state.current_mask_volume
        if mv is None:
            return None
        NZ, NY, NX = mv.shape
        dims = np.array([NX, NY, NZ], dtype=np.float64)
        ro = np.asarray(ro, dtype=np.float64)
        rd = np.asarray(rd, dtype=np.float64)
        n = np.linalg.norm(rd)
        if n == 0:
            return None
        rd = rd / n
        bmin = np.asarray(box_min, dtype=np.float64)
        bmax = np.asarray(box_max, dtype=np.float64)
        span = bmax - bmin
        inv = 1.0 / np.where(rd == 0, 1e-9, rd)
        t1 = (bmin - ro) * inv
        t2 = (bmax - ro) * inv
        tnear = max(0.0, float(np.max(np.minimum(t1, t2))))
        tfar = float(np.min(np.maximum(t1, t2)))
        if tnear > tfar:
            return None
        nsteps = int(max(64, 2 * max(NX, NY, NZ)))
        dt = (tfar - tnear) / nsteps
        t = tnear + dt * 0.5
        for _ in range(nsteps):
            p = ro + rd * t
            vc = np.clip(np.floor(((p - bmin) / span) * dims).astype(int), 0, dims.astype(int) - 1)
            if int(mv[vc[2], vc[1], vc[0]]) > 0:       # vc = (x, y, z) → mv[z, y, x]
                return (int(vc[2]), int(vc[1]), int(vc[0]))
            t += dt
        return None

    def pick_ray(self, state: SessionState, ro, rd, box_min, box_max):
        """3D pick (read-only): the first labelled cell under the cursor →
        ``(label, group, [x, y, z])`` (or 0s for a miss)."""
        voxel = self._march_ray(state, ro, rd, box_min, box_max)
        if voxel is None:
            return 0, 0, None
        self.ensure_ncolor(state)
        vz, vy, vx = voxel
        lab = int(state.current_mask_volume[vz, vy, vx])
        return lab, int((state.label_group or {}).get(lab, 0)), [vx, vy, vz]

    # ----- undo / redo (server is the single source of truth) -------------

    UNDO_LIMIT = 100

    def _record_edit(self, state: SessionState, before_mv: np.ndarray, before_lg: dict) -> None:
        """Record a compact diff (changed voxels + label_group before/after) of the
        edit that just mutated ``current_mask_volume``. Pushes onto the undo stack
        and clears the redo stack (a new edit forks history)."""
        after = state.current_mask_volume
        changed = np.flatnonzero(before_mv.reshape(-1) != after.reshape(-1))
        if changed.size == 0:
            return                                  # no-op edit: don't pollute history
        entry = {
            "idx": changed,
            "old": before_mv.reshape(-1)[changed].copy(),
            "new": after.reshape(-1)[changed].copy(),
            "lg_before": before_lg,
            "lg_after": dict(state.label_group or {}),
        }
        if state.undo_stack is None:
            state.undo_stack = []
        state.undo_stack.append(entry)
        if len(state.undo_stack) > self.UNDO_LIMIT:
            state.undo_stack.pop(0)
        state.redo_stack = []

    def _apply_entry(self, state: SessionState, entry: dict, *, redo: bool) -> None:
        flat = state.current_mask_volume.reshape(-1)
        flat[entry["idx"]] = entry["new"] if redo else entry["old"]
        state.label_group = dict(entry["lg_after"] if redo else entry["lg_before"])
        state.current_ncolor_volume = None

    def undo(self, state: SessionState) -> bool:
        if not state.undo_stack:
            return False
        entry = state.undo_stack.pop()
        self._apply_entry(state, entry, redo=False)
        if state.redo_stack is None:
            state.redo_stack = []
        state.redo_stack.append(entry)
        self.schedule_autosave(state)
        return True

    def redo(self, state: SessionState) -> bool:
        if not state.redo_stack:
            return False
        entry = state.redo_stack.pop()
        self._apply_entry(state, entry, redo=True)
        if state.undo_stack is None:
            state.undo_stack = []
        state.undo_stack.append(entry)
        self.schedule_autosave(state)
        return True

    def can_undo(self, state: SessionState) -> bool:
        return bool(state.undo_stack)

    def can_redo(self, state: SessionState) -> bool:
        return bool(state.redo_stack)

    # ----- debounced autosave of the edited mask to disk ------------------

    SAVE_DEBOUNCE_SECONDS = float(_os.environ.get("OCDKIT_VIEWER_AUTOSAVE_DEBOUNCE", "1.2"))

    def schedule_autosave(self, state: SessionState) -> None:
        """(Re)arm a debounce timer; the edited mask is written once edits settle.
        Server-driven so it survives client refresh/disconnect. No source path
        (e.g. no mask loaded from a file) → nothing to autosave."""
        if state.mask_source_path is None:
            return
        try:
            if state.save_timer is not None:
                state.save_timer.cancel()
        except Exception:
            pass
        t = threading.Timer(self.SAVE_DEBOUNCE_SECONDS, self._autosave_fire, args=(state.session_id,))
        t.daemon = True
        state.save_timer = t
        t.start()

    def _autosave_fire(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.current_mask_volume is None or state.mask_source_path is None:
                return
            arr = state.current_mask_volume.copy()       # snapshot under lock; write outside
            dest = self._edited_mask_path(state.mask_source_path)
        self._write_mask_atomic(dest, arr)

    def save_edited_mask(self, state: SessionState) -> Optional[str]:
        """Force an immediate save (e.g. on page unload). Returns the path written."""
        if state.current_mask_volume is None or state.mask_source_path is None:
            return None
        dest = self._edited_mask_path(state.mask_source_path)
        self._write_mask_atomic(dest, state.current_mask_volume.copy())
        return str(dest)

    @staticmethod
    def _write_mask_atomic(dest: Path, arr: np.ndarray) -> None:
        """Write the label volume to a temp file then atomically replace, so a
        crash mid-write can never corrupt the saved mask."""
        tmp = dest.with_name(dest.name + ".tmp")
        try:
            import tifffile                       # volume-aware writer (imageio.imwrite is 2D-only)
            tifffile.imwrite(str(tmp), arr)
            _os.replace(tmp, dest)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

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
