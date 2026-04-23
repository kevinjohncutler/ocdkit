"""Server-side segmentation dispatch — bridges HTTP routes to the active plugin.

The viewer keeps a small per-process cache of the most recent segmentation
result so that routes like ``GET /api/ncolor`` and the n-color/affinity-graph
helpers can return data without re-running the plugin.
"""

from __future__ import annotations

import base64
import threading
from typing import Any, Mapping, Optional, TYPE_CHECKING

import numpy as np

from .assets import append_gui_log
from .masks import compute_ncolor_mask
from .plugins.base import SegmentationPlugin, split_run_result
from .plugins.registry import REGISTRY
from .sample_image import load_image_uint8

if TYPE_CHECKING:
    from .session import SessionState


class ActivePlugin:
    """Singleton tracking which plugin is currently active and the last result.

    The frontend can switch plugins at runtime via ``POST /api/plugin/select``.
    All segmentation requests dispatch through ``current()``.

    Caches base64 encodings of the last mask + ncolor so repeated GET requests
    (e.g. polling /api/ncolor) don't re-encode the same arrays (issue #7).
    """

    def __init__(self) -> None:
        self._name: Optional[str] = None
        self._lock = threading.Lock()
        self._last_mask: Optional[np.ndarray] = None
        self._last_extras: dict[str, Any] = {}
        self._last_ncolor: Optional[np.ndarray] = None
        self._encoded_mask: Optional[str] = None
        self._encoded_ncolor: Optional[str] = None

    def select(self, name: Optional[str]) -> Optional[SegmentationPlugin]:
        """Set the active plugin by name. ``None`` clears it."""
        with self._lock:
            if name is None:
                self._name = None
                self._reset_cache()
                return None
            plugin = REGISTRY.get(name)
            if plugin.warmup is not None:
                # Best effort — model warmup is handled separately by the route
                pass
            self._name = name
            self._reset_cache()
            return plugin

    def name(self) -> Optional[str]:
        with self._lock:
            return self._name

    def current(self) -> Optional[SegmentationPlugin]:
        """Return the currently selected plugin, auto-selecting if exactly one is registered."""
        with self._lock:
            if self._name is not None:
                try:
                    return REGISTRY.get(self._name)
                except KeyError:
                    self._name = None
        all_plugins = REGISTRY.all()
        if len(all_plugins) == 1:
            with self._lock:
                self._name = all_plugins[0].name
            return all_plugins[0]
        return None

    def require(self) -> SegmentationPlugin:
        plugin = self.current()
        if plugin is None:
            raise RuntimeError(
                "no active plugin — register one or POST /api/plugin/select first"
            )
        return plugin

    def has_cache(self) -> bool:
        with self._lock:
            return self._last_mask is not None

    def get_last_mask(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_mask is None else self._last_mask.copy()

    def get_last_extras(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_extras)

    def get_last_ncolor(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_ncolor is None else self._last_ncolor.copy()

    def _reset_cache(self) -> None:
        # Caller already holds the lock.
        self._last_mask = None
        self._last_extras = {}
        self._last_ncolor = None
        self._encoded_mask = None
        self._encoded_ncolor = None

    def reset_cache(self) -> None:
        with self._lock:
            self._reset_cache()

    def store_result(
        self,
        mask: np.ndarray,
        extras: Mapping[str, Any],
        *,
        ncolor: Optional[np.ndarray] = None,
    ) -> None:
        with self._lock:
            self._last_mask = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
            self._last_extras = dict(extras)
            self._last_ncolor = ncolor
            # Invalidate base64 cache; lazily re-encoded on first read.
            self._encoded_mask = None
            self._encoded_ncolor = None

    def get_encoded_mask(self) -> Optional[str]:
        """Return base64-encoded uint32 mask bytes, computing once and caching."""
        with self._lock:
            if self._encoded_mask is not None:
                return self._encoded_mask
            if self._last_mask is None:
                return None
            self._encoded_mask = base64.b64encode(self._last_mask.tobytes()).decode("ascii")
            return self._encoded_mask

    def get_encoded_ncolor(self) -> Optional[str]:
        """Return base64-encoded uint32 ncolor mask, computing once and caching."""
        with self._lock:
            if self._encoded_ncolor is not None:
                return self._encoded_ncolor
            if self._last_ncolor is None:
                return None
            arr = np.ascontiguousarray(self._last_ncolor.astype(np.uint32, copy=False))
            self._encoded_ncolor = base64.b64encode(arr.tobytes()).decode("ascii")
            return self._encoded_ncolor


ACTIVE_PLUGIN = ActivePlugin()


def _build_segment_payload(
    mask: np.ndarray,
    extras: Mapping[str, Any],
    *,
    plugin: SegmentationPlugin,
    ncolor_mask: Optional[np.ndarray],
) -> dict[str, Any]:
    """Assemble the JSON response from the cached (already-encoded) state."""
    h, w = mask.shape
    payload: dict[str, Any] = {
        "plugin": plugin.name,
        # Pull from cache instead of re-encoding (issue #7).
        "mask": ACTIVE_PLUGIN.get_encoded_mask(),
        "width": int(w),
        "height": int(h),
        "canRebuild": plugin.resegment is not None,
        "nColorMask": ACTIVE_PLUGIN.get_encoded_ncolor() if ncolor_mask is not None else None,
    }
    # Pass through plugin-supplied extras (flow overlay, affinity graph, points,
    # etc.). Names are conventional and rendered by the frontend if present:
    # flowOverlay, distanceOverlay, affinityGraph, points.
    for key, value in extras.items():
        if key in payload:
            continue
        payload[key] = value
    return payload


def run_segmentation(
    settings: Mapping[str, Any] | None = None,
    *,
    state: "SessionState | None" = None,
) -> dict[str, Any]:
    """Dispatch a full segment to the active plugin and cache the result."""
    from .session import SESSION_MANAGER

    plugin = ACTIVE_PLUGIN.require()
    if state is None:
        state = SESSION_MANAGER.get_or_create(None)
    image = state.current_image if state.current_image is not None else load_image_uint8(as_rgb=True)

    if isinstance(settings, Mapping) and "use_gpu" in settings and plugin.set_use_gpu is not None:
        try:
            plugin.set_use_gpu(bool(settings["use_gpu"]))
        except Exception as exc:  # pragma: no cover
            append_gui_log(f"[segment] set_use_gpu failed: {exc}")

    params = dict(plugin.defaults())
    if isinstance(settings, Mapping):
        for k, v in settings.items():
            if k != "use_gpu":
                params[k] = v

    raw = plugin.run(image, params)
    mask, extras = split_run_result(raw)
    if mask.ndim != 2:
        raise RuntimeError(f"plugin {plugin.name!r} returned mask with ndim={mask.ndim}")

    ncolor_mask = compute_ncolor_mask(mask, expand=True)
    ACTIVE_PLUGIN.store_result(mask, extras, ncolor=ncolor_mask)
    return _build_segment_payload(mask, extras, plugin=plugin, ncolor_mask=ncolor_mask)


def run_mask_update(
    settings: Mapping[str, Any] | None = None,
    *,
    state: "SessionState | None" = None,
) -> dict[str, Any]:
    """Re-run mask reconstruction using the plugin's resegment hook (or full re-run)."""
    plugin = ACTIVE_PLUGIN.require()
    if plugin.resegment is None or not ACTIVE_PLUGIN.has_cache():
        return run_segmentation(settings, state=state)

    if isinstance(settings, Mapping) and "use_gpu" in settings and plugin.set_use_gpu is not None:
        try:
            plugin.set_use_gpu(bool(settings["use_gpu"]))
        except Exception as exc:  # pragma: no cover
            append_gui_log(f"[rebuild] set_use_gpu failed: {exc}")

    params = dict(plugin.defaults())
    if isinstance(settings, Mapping):
        for k, v in settings.items():
            if k != "use_gpu":
                params[k] = v

    raw = plugin.resegment(params)
    mask, extras = split_run_result(raw)
    if mask.ndim != 2:
        raise RuntimeError(f"plugin {plugin.name!r}.resegment returned ndim={mask.ndim}")

    ncolor_mask = compute_ncolor_mask(mask, expand=True)
    ACTIVE_PLUGIN.store_result(mask, extras, ncolor=ncolor_mask)
    return _build_segment_payload(mask, extras, plugin=plugin, ncolor_mask=ncolor_mask)
