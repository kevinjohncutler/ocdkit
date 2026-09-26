"""GET /api/display_headroom: the live HDR headroom of this machine's display,
answered only for a client on the same machine."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from ocdkit.viewer import edr_bridge  # noqa: E402
from ocdkit.viewer.app import create_app  # noqa: E402


@pytest.fixture
def fake_display(monkeypatch):
    monkeypatch.setattr(edr_bridge, "read_edr_headroom_info", lambda: {"current": 16.0, "potential": 20.3})


def test_local_client_gets_the_headroom(fake_display):
    with TestClient(create_app(), client=("127.0.0.1", 50000)) as c:
        assert c.get("/api/display_headroom").json() == {"available": True, "current": 16.0, "potential": 20.3}


def test_remote_client_is_refused(fake_display):
    # a browser on another machine has a different display: never send this one
    with TestClient(create_app(), client=("10.0.0.7", 50000)) as c:
        body = c.get("/api/display_headroom").json()
    assert body["available"] is False and "current" not in body


def test_no_edr_display(monkeypatch):
    monkeypatch.setattr(edr_bridge, "read_edr_headroom_info", lambda: None)
    with TestClient(create_app(), client=("127.0.0.1", 50000)) as c:
        assert c.get("/api/display_headroom").json()["available"] is False
