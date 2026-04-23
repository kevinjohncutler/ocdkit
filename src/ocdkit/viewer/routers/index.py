"""GET / — renders the viewer index HTML and mints the session cookie."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from ..assets import build_html
from ..dependencies import get_or_create_session
from ..plugins.registry import REGISTRY
from ..segmentation import ACTIVE_PLUGIN
from ..session import SESSION_COOKIE_NAME, SESSION_MANAGER, SessionState

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(state: SessionState = Depends(get_or_create_session)) -> HTMLResponse:
    config = SESSION_MANAGER.build_config(state)
    config["activePlugin"] = ACTIVE_PLUGIN.name()
    config["plugins"] = [p.manifest() for p in REGISTRY.all()]
    html = build_html(config, inline_assets=False)
    response = HTMLResponse(html)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        state.session_id,
        max_age=7 * 24 * 60 * 60,
        secure=False,
        httponly=False,
        samesite="Lax",
    )
    return response
