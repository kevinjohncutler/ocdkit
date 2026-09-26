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
    # an idle desktop: nothing on screen uses EDR yet, so 'current' is 1.0
    monkeypatch.setattr(edr_bridge, "read_edr_headroom_info",
                        lambda: {"headroom": 16.0, "potential": 16.0, "current": 1.0})


def test_local_client_gets_the_headroom(fake_display):
    with TestClient(create_app(), client=("127.0.0.1", 50000)) as c:
        body = c.get("/api/display_headroom").json()
    # the page must be given the potential, never the idle 'current' of 1.0
    assert body["available"] is True and body["headroom"] == 16.0


def test_remote_client_is_refused(fake_display):
    # a browser on another machine has a different display: never send this one
    with TestClient(create_app(), client=("10.0.0.7", 50000)) as c:
        body = c.get("/api/display_headroom").json()
    assert body["available"] is False and "headroom" not in body


def test_no_edr_display(monkeypatch):
    monkeypatch.setattr(edr_bridge, "read_edr_headroom_info", lambda: None)
    with TestClient(create_app(), client=("127.0.0.1", 50000)) as c:
        assert c.get("/api/display_headroom").json()["available"] is False
