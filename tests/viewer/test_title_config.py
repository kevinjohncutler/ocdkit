"""Tests for the configurable viewer title (OCDKIT_VIEWER_TITLE / --title)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from ocdkit.viewer.app import _autoload_plugins, create_app, viewer_title  # noqa: E402
from ocdkit.viewer.plugins.registry import REGISTRY  # noqa: E402
from ocdkit.viewer.segmentation import ACTIVE_PLUGIN  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.delenv("OCDKIT_VIEWER_TITLE", raising=False)
    REGISTRY.clear()
    _autoload_plugins()
    ACTIVE_PLUGIN.select("threshold")
    yield
    ACTIVE_PLUGIN.reset_cache()
    ACTIVE_PLUGIN.select(None)
    REGISTRY.clear()


def test_default_title_when_env_unset():
    assert viewer_title() == "ocdkit.viewer"


def test_env_var_overrides_title(monkeypatch):
    monkeypatch.setenv("OCDKIT_VIEWER_TITLE", "Omnipose")
    assert viewer_title() == "Omnipose"


def test_empty_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OCDKIT_VIEWER_TITLE", "")
    assert viewer_title() == "ocdkit.viewer"


def test_html_title_uses_default_when_env_unset():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>ocdkit.viewer</title>" in r.text


def test_html_title_picks_up_env_var(monkeypatch):
    monkeypatch.setenv("OCDKIT_VIEWER_TITLE", "Omnipose")
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>Omnipose</title>" in r.text
    assert "<title>ocdkit.viewer</title>" not in r.text


def test_fastapi_app_title_picks_up_env_var(monkeypatch):
    monkeypatch.setenv("OCDKIT_VIEWER_TITLE", "Custom-Title")
    app = create_app()
    assert app.title == "Custom-Title"


def test_html_title_escapes_special_chars(monkeypatch):
    """A title containing < or & should be HTML-escaped."""
    monkeypatch.setenv("OCDKIT_VIEWER_TITLE", "Acme <Beta> & Co")
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    # The < / > / & characters should be escaped, not raw HTML.
    assert "<title>Acme &lt;Beta&gt; &amp; Co</title>" in r.text
