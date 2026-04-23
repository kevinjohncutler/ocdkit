"""End-to-end tests for the Phase A FastAPI app.

Skipped if FastAPI is not installed (viewer extras not present).
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("imageio")

from imageio import v2 as imageio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ocdkit.viewer.app import _autoload_builtin_plugins, create_app  # noqa: E402
from ocdkit.viewer.plugins.registry import REGISTRY  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry():
    REGISTRY.clear()
    _autoload_builtin_plugins()
    yield
    REGISTRY.clear()


@pytest.fixture
def client():
    return TestClient(create_app())


def _png_b64(image: np.ndarray) -> str:
    buf = io.BytesIO()
    imageio.imwrite(buf, image, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ocdkit.viewer" in r.text


def test_list_plugins_returns_threshold(client):
    r = client.get("/api/plugins")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()["plugins"]]
    assert "threshold" in names


def test_get_plugin_manifest_shape(client):
    r = client.get("/api/plugins/threshold")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "threshold"
    assert any(w["name"] == "method" for w in body["widgets"])
    assert "capabilities" in body


def test_unknown_plugin_404(client):
    r = client.get("/api/plugins/no_such_thing")
    assert r.status_code == 404


def test_segment_round_trip(client):
    pytest.importorskip("skimage")
    image = np.zeros((48, 48), dtype=np.uint8)
    image[8:16, 8:16] = 220
    image[30:42, 30:42] = 240

    r = client.post(
        "/api/plugins/threshold/segment",
        json={"image": _png_b64(image), "params": {"method": "otsu", "min_size": 1}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["mask"]["width"] == 48
    assert body["mask"]["height"] == 48
    assert body["mask"]["maxLabel"] >= 2


def test_segment_invalid_image(client):
    """A non-base64 image string passes validation but fails decode → 400 envelope."""
    r = client.post(
        "/api/plugins/threshold/segment",
        json={"image": "not-base64", "params": {}},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert "decode" in body["error"].lower() or "decode" in str(body.get("detail", "")).lower()


def test_segment_missing_image(client):
    """Pydantic validation rejects a payload missing required `image` field."""
    r = client.post("/api/plugins/threshold/segment", json={"params": {}})
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "validation error"


def test_warmup_unsupported(client):
    """threshold plugin does not declare warmup → /api/plugin/warmup returns 400."""
    sel = client.post("/api/plugin/select", json={"name": "threshold"})
    assert sel.status_code == 200
    r = client.post("/api/plugin/warmup", json={"model": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert "plugin capability missing" in body["error"]
    assert "warmup" in str(body["detail"])


def test_clear_cache_endpoint(client):
    """clear_cache always succeeds and returns the standard OK envelope."""
    r = client.post("/api/clear_cache")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_select_plugin(client):
    r = client.post("/api/plugin/select", json={"name": "threshold"})
    assert r.status_code == 200
    body = r.json()
    assert body["active"] == "threshold"
    assert body["manifest"]["name"] == "threshold"


def test_select_unknown_plugin_404(client):
    r = client.post("/api/plugin/select", json={"name": "nope"})
    assert r.status_code == 404


def test_segment_session_route_with_threshold(client):
    """The session-aware /api/segment dispatches through the active plugin."""
    pytest.importorskip("skimage")
    # Hit / first to mint a session cookie + auto-select the lone plugin.
    home = client.get("/")
    assert home.status_code == 200
    cookie = client.cookies.get("OCDSESSION")
    assert cookie

    r = client.post(
        "/api/segment",
        json={"sessionId": cookie, "method": "otsu", "min_size": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["plugin"] == "threshold"
    assert body["width"] > 0 and body["height"] > 0
    assert body["mask"]  # base64-encoded
