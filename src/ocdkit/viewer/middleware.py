"""Custom ASGI middleware."""

from __future__ import annotations

import json
from typing import Awaitable, Callable, MutableMapping


class _BodyTooLarge(Exception):
    def __init__(self, size: int) -> None:
        self.size = size


class BodySizeLimitMiddleware:
    """ASGI middleware rejecting POST/PUT/PATCH bodies above ``max_bytes``.

    Prefer this over framework body limits because it short-circuits **before**
    FastAPI/Pydantic try to parse the body. Without it, a 1 GB JSON upload
    would be fully buffered + decoded before validation reports the error.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: MutableMapping, receive: Callable[[], Awaitable[MutableMapping]], send: Callable[[MutableMapping], Awaitable[None]]) -> None:
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        # Fast path: trust Content-Length when present.
        content_length: int | None = None
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    content_length = int(v.decode("ascii"))
                except (ValueError, UnicodeDecodeError):
                    content_length = None
                break
        if content_length is not None and content_length > self.max_bytes:
            await _send_413(send, content_length, self.max_bytes)
            return

        # Streaming path: count bytes as they arrive (handles chunked uploads).
        total = 0

        async def _wrapped_receive() -> MutableMapping:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"") or b""
                total += len(body)
                if total > self.max_bytes:
                    raise _BodyTooLarge(total)
            return message

        try:
            await self.app(scope, _wrapped_receive, send)
        except _BodyTooLarge as exc:
            await _send_413(send, exc.size, self.max_bytes)


async def _send_413(send: Callable[[MutableMapping], Awaitable[None]], size: int, limit: int) -> None:
    body = json.dumps(
        {
            "ok": False,
            "error": "payload too large",
            "detail": {"size": size, "limit": limit},
        }
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
