"""Tests for the ?ui=desktop query parameter and the early-paint background."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from ocdkit.viewer.app import _autoload_plugins, create_app  # noqa: E402
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


# -- Early-paint background (white-flash fix) -----------------------------


def test_index_inlines_early_background_style(client):
    """The OS-native background is set via an inline <style> at the top of
    <head>, BEFORE the external CSS link, so the browser's first paint
    matches the OS appearance instead of flashing white."""
    r = client.get("/")
    assert r.status_code == 200
    text = r.text
    # The early-bg style block must appear before the external CSS link.
    early_pos = text.find('id="early-bg"')
    css_pos = text.find('href="/static/css/layout.css')
    assert early_pos > 0, "early-bg <style> not found"
    assert css_pos > 0, "external CSS link not found"
    assert early_pos < css_pos, "early-bg must appear BEFORE external CSS link"
    # The body must be painted with the OS-native window background color.
    assert "background: Canvas" in text


# -- ?ui=desktop activates the translucent path ---------------------------


def test_default_browser_mode_no_html_attribute(client):
    """Without ?ui=desktop, the <html> tag has no data-ui attribute."""
    r = client.get("/")
    # The CSS rules inside the inline style still mention 'data-ui="desktop"'
    # as selectors — we want to verify the attribute is NOT on <html> itself.
    assert '<html lang="en">' in r.text
    assert '<html lang="en" data-ui=' not in r.text


def test_desktop_mode_sets_html_attribute(client):
    r = client.get("/?ui=desktop")
    assert r.status_code == 200
    assert '<html lang="en" data-ui="desktop">' in r.text


def test_unknown_ui_mode_falls_back_to_browser(client):
    r = client.get("/?ui=jetpack")
    assert r.status_code == 200
    assert '<html lang="en" data-ui=' not in r.text


def test_ui_mode_in_embedded_config(client):
    """The frontend can also read the mode from window.__VIEWER_CONFIG__.uiMode."""
    r_browser = client.get("/")
    r_desktop = client.get("/?ui=desktop")
    assert '"uiMode": "browser"' in r_browser.text
    assert '"uiMode": "desktop"' in r_desktop.text


# -- Background style adapts to OS dark/light mode (regression guard) ----


def test_early_bg_uses_system_colors(client):
    """The inline early-bg style uses CSS system colors (``Canvas``,
    ``CanvasText``) with ``color-scheme: light dark``, so the OS's native
    window chrome colors are used (not hardcoded black/white)."""
    r = client.get("/?ui=desktop")
    text = r.text
    assert "color-scheme: light dark" in text
    assert "background: Canvas" in text
    assert "color: CanvasText" in text


# -- Identifier rename smoke test ------------------------------------------


def test_window_config_uses_viewer_namespace(client):
    """After the rename pass __OMNI_CONFIG__ should not appear; __VIEWER_CONFIG__ should."""
    r = client.get("/")
    text = r.text
    assert "__VIEWER_CONFIG__" in text
    assert "__OMNI_CONFIG__" not in text
    assert "__VIEWER_WEBGL_LOGGING__" in text
    assert "__OMNI_WEBGL_LOGGING__" not in text
