"""Frontend logging sink: ``POST /api/log``."""

from __future__ import annotations

import json

from fastapi import APIRouter

from ..assets import append_gui_log
from ..routes import DebugAPI, WEBGL_LOG_PATH
from ..schemas import LogPayload, OkBody

router = APIRouter(prefix="/api")

_DEBUG_API = DebugAPI(log_path=WEBGL_LOG_PATH)


@router.post("/log", response_model=OkBody)
def api_log(payload: LogPayload) -> OkBody:
    if payload.entries is not None:
        for entry in payload.entries:
            try:
                line = json.dumps(entry, ensure_ascii=False)
            except Exception:
                line = str(entry)
            _DEBUG_API.log(line)
            append_gui_log(line)
    elif payload.messages is not None:
        for raw in payload.messages:
            line = str(raw)
            _DEBUG_API.log(line)
            append_gui_log(line)
    elif payload.message:
        _DEBUG_API.log(payload.message)
        append_gui_log(payload.message)
    return OkBody()
