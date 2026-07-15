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
    # Revalidate the HTML shell every load but ALLOW the browser to keep the last
    # render — `no-cache` (not `no-store`) means "store, but revalidate before
    # use". The server has no ETag so revalidation always returns fresh 200 HTML
    # (so the inlined trust-banner JS / probe origins are never stale — the bug
    # that motivated no-store), while letting the browser paint-hold the previous
    # frame during the reload fetch instead of flashing a blank/white backdrop
    # (Safari especially discards the page under no-store). Static assets keep
    # their aggressive mtime cache-bust.
    response.headers["Cache-Control"] = "no-cache"
    response.set_cookie(
        SESSION_COOKIE_NAME,
        state.session_id,
        max_age=7 * 24 * 60 * 60,
        secure=False,
        httponly=False,
        samesite="Lax",
    )
    return response
