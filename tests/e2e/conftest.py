"""Shared fixtures for end-to-end browser tests.

Spins up a real uvicorn server in a background thread for the duration of the
test session, then yields a Playwright browser/page that's already pointed at
it. Use ``pytest -m e2e tests/e2e`` to run.

Tests that don't actually need the browser (purely server-side) should live
under ``tests/viewer/`` — this directory is for cases where we need to drive
the JS frontend.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import closing

import pytest

pytest.importorskip("playwright")
pytest.importorskip("uvicorn")

from playwright.sync_api import sync_playwright  # noqa: E402

from ocdkit.viewer.app import _autoload_plugins, create_app  # noqa: E402
from ocdkit.viewer.plugins.registry import REGISTRY  # noqa: E402
from ocdkit.viewer.segmentation import ACTIVE_PLUGIN  # noqa: E402
from ocdkit.viewer.session import SESSION_MANAGER  # noqa: E402


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=0.5)):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server at {host}:{port} did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# Server fixture (session-scoped — one server for the whole test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def viewer_server():
    import uvicorn

    REGISTRY.clear()
    _autoload_plugins()  # registers `threshold`
    ACTIVE_PLUGIN.select("threshold")

    port = _free_port()
    config = uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="ViewerE2E", daemon=True)
    thread.start()
    _wait_for_port("127.0.0.1", port, timeout=10.0)

    yield {"host": "127.0.0.1", "port": port, "url": f"http://127.0.0.1:{port}"}

    server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Per-test reset of mutable singletons so tests are independent
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_active_plugin():
    ACTIVE_PLUGIN.reset_cache()
    yield
    ACTIVE_PLUGIN.reset_cache()


# ---------------------------------------------------------------------------
# Playwright fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def playwright_instance():
    with sync_playwright() as pw:
        yield pw


@pytest.fixture(scope="session")
def browser(playwright_instance):
    # Headless Chromium covers ~99% of GUI behavior; webview-specific quirks
    # are tested separately in tests/e2e/test_pywebview_snapshot.py.
    browser = playwright_instance.chromium.launch(headless=True)
    yield browser
    browser.close()


@pytest.fixture
def page(browser, viewer_server):
    """A fresh page (and context) per test, pointed at the running viewer."""
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    pg = context.new_page()
    # Surface JS errors as test failures rather than silent fails.
    pg.on("pageerror", lambda exc: pytest.fail(f"JS pageerror: {exc}"))
    pg.on(
        "console",
        lambda msg: print(f"[browser console:{msg.type}] {msg.text}")
        if msg.type in ("error", "warning")
        else None,
    )
    pg.goto(viewer_server["url"])
    pg.wait_for_load_state("networkidle", timeout=10000)
    yield pg
    context.close()
