"""Segmentation routes — full segment + cached recompute, both off-loop."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ..dependencies import get_session_state
from ..schemas import SegmentPayload
from ..segmentation import run_mask_update, run_segmentation
from ..session import SessionState

router = APIRouter(prefix="/api")


def _payload_dict(payload: SegmentPayload) -> dict:
    """Pass plugin-specific extras through model_dump(); drop None-only fields."""
    return payload.model_dump(exclude_none=True)


@router.post("/segment")
async def api_segment(
    payload: SegmentPayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    body = _payload_dict(payload)
    if payload.mode == "recompute":
        return await asyncio.to_thread(run_mask_update, body, state=state)
    return await asyncio.to_thread(run_segmentation, body, state=state)


@router.post("/resegment")
async def api_resegment(
    payload: SegmentPayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    body = _payload_dict(payload)
    return await asyncio.to_thread(run_mask_update, body, state=state)
