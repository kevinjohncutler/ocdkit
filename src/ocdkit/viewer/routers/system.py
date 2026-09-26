"""System info, GPU toggle, cache reset."""

from __future__ import annotations

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
    if not info or not info.get("current"):
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
