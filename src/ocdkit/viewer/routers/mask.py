"""Mask post-processing routes (n-color, format-labels, affinity-graph relabel)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import require_plugin_capability
from ..exceptions import BadRequest
from ..plugins.base import SegmentationPlugin
from ..routes import DebugAPI, WEBGL_LOG_PATH
from ..schemas import MaskOpPayload, NColorFromMaskPayload, RelabelFromAffinityPayload

router = APIRouter(prefix="/api")
_DEBUG_API = DebugAPI(log_path=WEBGL_LOG_PATH)


@router.get("/ncolor")
def api_ncolor() -> dict:
    return _DEBUG_API.get_ncolor()


@router.post("/ncolor_from_mask")
def api_ncolor_from_mask(payload: NColorFromMaskPayload) -> dict:
    return _DEBUG_API.ncolor_from_mask(payload.model_dump())


@router.post("/format_labels")
def api_format_labels(payload: MaskOpPayload) -> dict:
    return _DEBUG_API.format_labels(payload.model_dump())


@router.post("/relabel_from_affinity")
def api_relabel_from_affinity(
    payload: RelabelFromAffinityPayload,
    plugin: SegmentationPlugin = Depends(
        require_plugin_capability("relabel_from_affinity")
    ),
) -> dict:
    # DebugAPI handles the heavy lifting; the dependency just guarantees the
    # active plugin actually supports the operation.
    result = _DEBUG_API.relabel_from_affinity(payload.model_dump())
    if "error" in result:
        # DebugAPI returns dict-with-error for legacy reasons; surface it
        # through the standard envelope.
        raise BadRequest(result["error"])
    return result
