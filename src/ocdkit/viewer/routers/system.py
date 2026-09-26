"""System info, GPU toggle, cache reset."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from ..schemas import OkBody, UseGpuPayload
from ..segmentation import ACTIVE_PLUGIN
from ..session import SESSION_COOKIE_NAME, SESSION_MANAGER
from ..system import get_system_info

router = APIRouter(prefix="/api")


@router.get("/system_info")
def api_system_info() -> dict:
    return get_system_info(ACTIVE_PLUGIN.current())


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class _QuietHeadroomPoll(logging.Filter):
    """Keep the page's periodic headroom poll out of the access log (it would
    print a line every couple of seconds for as long as the viewer is open)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/display_headroom" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietHeadroomPoll())


@router.get("/display_headroom")
def api_display_headroom(request: Request) -> dict:
    """The live HDR headroom of this machine's display (macOS), for HDR
    rendering in a browser, which may not read it itself. Only answered for a
    client on this same machine: a remote browser's display is a different one."""
    host = request.client.host if request.client else ""
    if host not in _LOOPBACK:
        return {"available": False, "reason": "not a local client"}
    from ..edr_bridge import read_edr_headroom_info
    info = read_edr_headroom_info()
    if not info or not info.get("headroom") or info["headroom"] <= 1.0:
        return {"available": False, "reason": "no EDR display"}
    return {"available": True, **info}


@router.post("/use_gpu")
def api_use_gpu(payload: UseGpuPayload) -> dict:
    plugin = ACTIVE_PLUGIN.current()
    if plugin is not None and plugin.set_use_gpu is not None:
        plugin.set_use_gpu(payload.use_gpu)
    return get_system_info(plugin)


@router.post("/clear_cache", response_model=OkBody)
def api_clear_cache(request: Request) -> OkBody:
    session_cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if session_cookie:
        try:
            state = SESSION_MANAGER.get(session_cookie)
            SESSION_MANAGER.clear_saved_states(state)
        except KeyError:
            pass
    ACTIVE_PLUGIN.reset_cache()
    plugin = ACTIVE_PLUGIN.current()
    if plugin and plugin.clear_cache is not None:
        plugin.clear_cache()
    return OkBody()
