"""GET / — renders the viewer index HTML and mints the session cookie."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..assets import build_html
from ..dependencies import get_or_create_session
from ..plugins.registry import REGISTRY
from ..segmentation import ACTIVE_PLUGIN
from ..session import SESSION_COOKIE_NAME, SESSION_MANAGER, SessionState

router = APIRouter()


_ALLOWED_UI_MODES = {"browser", "desktop"}


def _detect_ui_mode(request: Request) -> str:
    """Pick a UI mode from the request.

    Resolution order:
      1. ``?ui=desktop`` query string (set by the pywebview launcher).
      2. ``?ui=`` value from the cookie set by the launcher (future use).
      3. Default ``browser``.
    """
    qs_mode = request.query_params.get("ui")
    if qs_mode in _ALLOWED_UI_MODES:
        return qs_mode
    return "browser"


@router.get("/", response_class=HTMLResponse)
def render_index(
    request: Request,
    state: SessionState = Depends(get_or_create_session),
) -> HTMLResponse:
    config = SESSION_MANAGER.build_config(state)
    config["activePlugin"] = ACTIVE_PLUGIN.name()
    config["plugins"] = [p.manifest() for p in REGISTRY.all()]
    ui_mode = _detect_ui_mode(request)
    config["uiMode"] = ui_mode
    html = build_html(config, inline_assets=False, ui_mode=ui_mode)
    response = HTMLResponse(html)
    # Never cache the HTML shell — the trust-install banner JS and probe
    # origins are inlined here, and stale HTML caused the "banner persists
    # after install" debugging session. Static assets (under /static/) are
    # still cached aggressively via mtime cache-bust query strings.
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.set_cookie(
        SESSION_COOKIE_NAME,
        state.session_id,
        max_age=7 * 24 * 60 * 60,
        secure=False,
        httponly=False,
        samesite="Lax",
    )
    return response
