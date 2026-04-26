"""Tests for the asyncio.to_thread offload of inference (issue #3).

Confirms that the FastAPI event loop stays responsive even when a plugin's
``run()`` function takes a long time. Without ``await asyncio.to_thread``,
the event loop would be blocked and a concurrent ``GET /api/plugins``
request would queue behind segmentation.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ocdkit.viewer import SegmentationPlugin, WidgetSpec  # noqa: E402
from ocdkit.viewer.app import _autoload_plugins, create_app  # noqa: E402
from ocdkit.viewer.plugins.registry import REGISTRY  # noqa: E402
from ocdkit.viewer.segmentation import ACTIVE_PLUGIN  # noqa: E402


def _slow_plugin(sleep_seconds: float = 0.5):
    def _run(image, params):
        time.sleep(sleep_seconds)
        return np.zeros(image.shape[:2], dtype=np.int32)
    return SegmentationPlugin(
        name="slow_test",
        version="0.1.0",
        widgets=[WidgetSpec("noop", "Noop", "toggle", default=False)],
        run=_run,
    )


@pytest.fixture
def slow_client():
    REGISTRY.clear()
    REGISTRY.register(_slow_plugin())
    ACTIVE_PLUGIN.select("slow_test")
    yield TestClient(create_app())
    ACTIVE_PLUGIN.reset_cache()
    ACTIVE_PLUGIN.select(None)
    REGISTRY.clear()
    _autoload_plugins()


def test_event_loop_stays_responsive_during_segmentation(slow_client):
    """While a slow segment runs, /api/plugins should still return promptly."""
    # Spin up the session.
    home = slow_client.get("/")
    assert home.status_code == 200
    cookie = slow_client.cookies.get("OCDSESSION")

    # Async client lets us issue concurrent requests via httpx.
    import httpx
    base = "http://testserver"
    async def _runner():
        transport = httpx.ASGITransport(app=slow_client.app)
        async with httpx.AsyncClient(transport=transport, base_url=base) as ac:
            ac.cookies.set("OCDSESSION", cookie)
            t0 = time.perf_counter()
            seg_task = asyncio.create_task(
                ac.post("/api/segment", json={"sessionId": cookie})
            )
            # Give the segmentation a moment to start.
            await asyncio.sleep(0.05)
            t_concurrent_start = time.perf_counter()
            list_resp = await ac.get("/api/plugins")
            t_concurrent_done = time.perf_counter()
            seg_resp = await seg_task
            return {
                "list_status": list_resp.status_code,
                "concurrent_latency": t_concurrent_done - t_concurrent_start,
                "total": time.perf_counter() - t0,
                "seg_status": seg_resp.status_code,
            }

    # Newer starlette TestClient uses anyio under the hood, which leaves a
    # running event loop on the calling thread by the time the fixture
    # yields. ``asyncio.run`` refuses to nest. Allow nesting.
    import nest_asyncio
    nest_asyncio.apply()
    result = asyncio.run(_runner())
    assert result["list_status"] == 200
    assert result["seg_status"] == 200
    # The concurrent /api/plugins call should not have waited for the slow
    # segmentation. With to_thread it returns in ~ms; without it would
    # block for 0.5s. Allow a 100ms safety margin.
    assert result["concurrent_latency"] < 0.1, (
        f"Concurrent request blocked for {result['concurrent_latency']:.3f}s; "
        "asyncio.to_thread offload may not be working."
    )
