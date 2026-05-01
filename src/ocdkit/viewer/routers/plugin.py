"""Plugin discovery, selection, warmup, and one-shot stateless segmentation."""

from __future__ import annotations

import asyncio
import base64
import io
import logging

import numpy as np
from fastapi import APIRouter, Depends
from imageio import v2 as imageio

from ..dependencies import (
    get_active_plugin,
    get_plugin_by_name,
    require_plugin_capability,
)
from ..exceptions import BadRequest
from ..plugins.base import SegmentationPlugin, split_run_result
from ..plugins.registry import REGISTRY
from ..schemas import OkBody, OneShotSegmentPayload, SelectPluginPayload, WarmupPayload
from ..segmentation import ACTIVE_PLUGIN

router = APIRouter(prefix="/api")
logger = logging.getLogger("ocdkit.viewer")


# ----- helpers ------------------------------------------------------------


def _decode_data_url_image(data_url_or_b64: str) -> np.ndarray:
    payload = data_url_or_b64
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    raw = base64.b64decode(payload)
    arr = imageio.imread(io.BytesIO(raw))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr -= arr.min()
        peak = arr.max()
        if peak > 0:
            arr = arr / peak
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


# ----- routes -------------------------------------------------------------


@router.get("/plugins")
def api_list_plugins() -> dict:
    # Trigger auto-select if nothing is active yet, so the picker UI sees a
    # default highlighted on first open.
    ACTIVE_PLUGIN.current()
    return {
        "plugins": [p.manifest() for p in REGISTRY.all()],
        "active": ACTIVE_PLUGIN.name(),
    }


@router.get("/plugins/{name}")
def api_plugin_manifest(plugin: SegmentationPlugin = Depends(get_plugin_by_name)) -> dict:
    return plugin.manifest()


@router.post("/plugin/select")
def api_select_plugin(payload: SelectPluginPayload) -> dict:
    try:
        plugin = ACTIVE_PLUGIN.select(payload.name)
    except KeyError as exc:
        from ..exceptions import PluginNotRegistered
        raise PluginNotRegistered(detail=payload.name) from exc
    return {
        "ok": True,
        "active": ACTIVE_PLUGIN.name(),
        "manifest": plugin.manifest() if plugin else None,
    }


@router.post("/plugin/warmup", response_model=OkBody)
def api_plugin_warmup(
    payload: WarmupPayload,
    plugin: SegmentationPlugin = Depends(require_plugin_capability("warmup")),
) -> OkBody:
    plugin.warmup(payload.model)
    return OkBody()


@router.post("/plugin/clear_cache", response_model=OkBody)
def api_plugin_clear_cache(
    plugin: SegmentationPlugin = Depends(require_plugin_capability("clear_cache")),
) -> OkBody:
    plugin.clear_cache()
    ACTIVE_PLUGIN.reset_cache()
    return OkBody()


@router.post("/plugins/{name}/segment")
async def api_plugins_segment_oneshot(
    payload: OneShotSegmentPayload,
    plugin: SegmentationPlugin = Depends(get_plugin_by_name),
) -> dict:
    """Stateless single-call: send image bytes, get a mask back. Phase A compat."""
    try:
        image = _decode_data_url_image(payload.image)
    except Exception as exc:
        raise BadRequest("failed to decode image", detail=str(exc)) from exc
    params = dict(plugin.defaults())
    params.update(payload.params)
    mask, extras = split_run_result(
        await asyncio.to_thread(plugin.run, image, params)
    )
    h, w = mask.shape
    return {
        "ok": True,
        "plugin": plugin.name,
        "params": params,
        "mask": {
            "width": int(w),
            "height": int(h),
            "dtype": "int32",
            "data": base64.b64encode(
                np.ascontiguousarray(mask.astype(np.int32)).tobytes()
            ).decode("ascii"),
            "maxLabel": int(mask.max()) if mask.size else 0,
        },
        "extras": extras,
    }
