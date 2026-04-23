"""Tests for the standard error envelope, Pydantic validation, body-size limit.

These document the new (Batch B + C) contract: every error response has the
shape ``{"ok": false, "error": str, "detail": Optional[Any]}``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient  # noqa: E402

from ocdkit.viewer.app import _autoload_plugins, create_app  # noqa: E402
from ocdkit.viewer.middleware import BodySizeLimitMiddleware  # noqa: E402
from ocdkit.viewer.plugins.registry import REGISTRY  # noqa: E402
from ocdkit.viewer.segmentation import ACTIVE_PLUGIN  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry():
    REGISTRY.clear()
    _autoload_plugins()
    ACTIVE_PLUGIN.select("threshold")
    yield
    ACTIVE_PLUGIN.reset_cache()
    ACTIVE_PLUGIN.select(None)
    REGISTRY.clear()


@pytest.fixture
def client():
    return TestClient(create_app())


# -- standard envelope -----------------------------------------------------


def test_validation_error_uses_standard_envelope(client):
    """Pydantic validation failures come back as the standard error shape."""
    r = client.post("/api/use_gpu", json={})  # missing required field `use_gpu`
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "validation error"
    assert isinstance(body["detail"], list) and len(body["detail"]) > 0


def test_unknown_plugin_uses_standard_envelope(client):
    r = client.get("/api/plugins/no_such_plugin")
    assert r.status_code == 404
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "plugin not registered"
    assert body["detail"] == "no_such_plugin"


def test_unknown_session_uses_standard_envelope(client):
    """Posting with an unknown sessionId yields the unified 404 envelope."""
    r = client.post("/api/save_state",
                    json={"sessionId": "bogus", "viewerState": {}})
    assert r.status_code == 404
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "unknown session"


def test_success_uses_ok_envelope_where_no_payload(client):
    r = client.post("/api/clear_cache")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# -- /api/log accepts the three documented shapes ----------------------------


def test_log_accepts_entries(client):
    r = client.post("/api/log", json={"entries": [{"k": "v"}, "string"]})
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_log_accepts_messages(client):
    r = client.post("/api/log", json={"messages": ["one", "two"]})
    assert r.status_code == 200


def test_log_accepts_message(client):
    r = client.post("/api/log", json={"message": "hello"})
    assert r.status_code == 200


def test_log_rejects_empty_payload(client):
    r = client.post("/api/log", json={})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation error"


# -- body-size limit -------------------------------------------------------


def test_body_size_limit_returns_413(client):
    """A POST larger than the configured limit gets 413 before reaching routes."""
    # Build a payload bigger than the test limit. Default cap is 64 MB; most
    # tests will use the default. We use a small custom app for this case.
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from ocdkit.viewer.middleware import BodySizeLimitMiddleware

    small_app = FastAPI()
    small_app.add_middleware(BodySizeLimitMiddleware, max_bytes=1024)

    @small_app.post("/echo")
    def _echo(payload: dict) -> dict:
        return {"size": len(str(payload))}

    c = TestClient(small_app)
    big_payload = {"x": "A" * 4096}
    r = c.post("/echo", json=big_payload)
    assert r.status_code == 413
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "payload too large"
    assert body["detail"]["limit"] == 1024


def test_body_size_limit_passes_small_request(client):
    """A normal-sized request is unaffected by the size middleware."""
    r = client.post("/api/log", json={"message": "tiny"})
    assert r.status_code == 200


# -- middleware contract ---------------------------------------------------


def test_middleware_is_a_class():
    """Sanity: BodySizeLimitMiddleware is the ASGI class form (max_bytes kwarg)."""
    assert callable(BodySizeLimitMiddleware)
    # Constructible with required kwargs
    inst = BodySizeLimitMiddleware(lambda *a, **kw: None, max_bytes=1)
    assert inst.max_bytes == 1
