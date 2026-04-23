"""Exception types + handlers giving every endpoint a uniform error envelope.

Routes raise the typed exceptions below instead of returning ``JSONResponse({"error": ...})``.
The handlers translate them into ``{"ok": false, "error": ..., "detail": ...}``
JSON with the right HTTP status code, and log unexpected failures via the
standard logger (so uvicorn/k8s ingress them correctly).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("ocdkit.viewer")


# ----- exception classes --------------------------------------------------


class APIError(Exception):
    """Base class for application-defined HTTP errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error: str = "internal error"

    def __init__(self, error: Optional[str] = None, detail: Any = None) -> None:
        super().__init__(error or self.error)
        if error is not None:
            self.error = error
        self.detail = detail


class BadRequest(APIError):
    status_code = status.HTTP_400_BAD_REQUEST
    error = "bad request"


class NotFound(APIError):
    status_code = status.HTTP_404_NOT_FOUND
    error = "not found"


class UnknownSession(NotFound):
    error = "unknown session"


class PluginNotRegistered(NotFound):
    error = "plugin not registered"


class NoActivePlugin(BadRequest):
    error = "no active plugin"


class PluginCapabilityMissing(BadRequest):
    error = "plugin capability missing"


# ----- envelope helper ----------------------------------------------------


def envelope(error: str, *, detail: Any = None, status_code: int = 500) -> JSONResponse:
    body: dict[str, Any] = {"ok": False, "error": error}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(body, status_code=status_code)


# ----- handlers (registered in app.py) ------------------------------------


def install(app: FastAPI) -> None:
    """Wire every error path to the standard envelope shape."""

    @app.exception_handler(APIError)
    async def _on_api_error(request: Request, exc: APIError) -> JSONResponse:
        return envelope(exc.error, detail=exc.detail, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Reshape FastAPI's default 422 to match our envelope. Strip
        # pydantic's non-JSON fields (ctx.error / url) so JSONResponse can
        # serialize cleanly.
        details = []
        for err in exc.errors():
            safe = {
                k: v for k, v in err.items()
                if k not in {"ctx", "url"}
            }
            ctx = err.get("ctx")
            if isinstance(ctx, dict):
                safe["ctx"] = {k: str(v) for k, v in ctx.items()}
            details.append(safe)
        return envelope(
            "validation error",
            detail=details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return envelope(
            str(exc.detail) if exc.detail is not None else "http error",
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Last-resort: log the full traceback and return a generic 500. Don't
        # leak the exception message to the client unless we know it's safe.
        logger.exception("unhandled exception in route %s %s", request.method, request.url.path)
        return envelope(
            "internal server error",
            detail=f"{type(exc).__name__}: {exc}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
