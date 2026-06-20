"""Headless FastAPI tests for POST /api/volume (3D bundle route).

Uses a fake plugin + dependency override so the route wiring + capability
dispatch are tested without omnipose/torch. The actual bundle building is
covered by omnipose/tests/test_volume3d.py.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocdkit.viewer import exceptions
from ocdkit.viewer.dependencies import get_active_plugin
from ocdkit.viewer.plugins.base import SegmentationPlugin
from ocdkit.viewer.routers import volume


def _app(plugin):
    app = FastAPI()
    exceptions.install(app)
    app.include_router(volume.router)
    app.dependency_overrides[get_active_plugin] = lambda: plugin
    return app


def test_route_dispatches_to_capability():
    captured = {}

    def build(masks_path, raw_path=None, links_path=None, **flags):
        captured.update(masks_path=masks_path, raw_path=raw_path,
                        links_path=links_path, flags=flags)
        return {"meta": {"dim": 3, "depth": 7}, "ok": True}

    plugin = SegmentationPlugin(name="fake", version="1", widgets=[],
                               run=lambda i, p: i, build_volume_bundle=build)
    client = TestClient(_app(plugin))
    r = client.post("/api/volume", json={
        "masksPath": "/x/m.tif", "rawPath": "/x/r.tif",
        "doFlow": False, "doAffinity": False})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["depth"] == 7 and body["ok"] is True
    assert captured["masks_path"] == "/x/m.tif"
    assert captured["raw_path"] == "/x/r.tif"
    assert captured["flags"]["do_flow"] is False
    assert captured["flags"]["do_affinity"] is False
    assert captured["flags"]["do_trajectories"] is True   # default
    assert captured["flags"]["embed_volumes"] is True


def test_route_requires_capability():
    plain = SegmentationPlugin(name="plain", version="1", widgets=[],
                              run=lambda i, p: i)  # no build_volume_bundle
    client = TestClient(_app(plain))
    r = client.post("/api/volume", json={"masksPath": "/x/m.tif"})
    assert r.status_code == 400


def test_manifest_advertises_capability():
    p = SegmentationPlugin(name="fake", version="1", widgets=[],
                          run=lambda i, p: i, build_volume_bundle=lambda *a, **k: {})
    assert p.manifest()["capabilities"]["build_volume_bundle"] is True
    plain = SegmentationPlugin(name="plain", version="1", widgets=[], run=lambda i, p: i)
    assert plain.manifest()["capabilities"]["build_volume_bundle"] is False


import os  # noqa: E402

_SPACE_MASKS = "/Volumes/DataDrive/3D_spacetime/linked/a_baylii/dnaA_xy1_crop_masks.tif"


@pytest.mark.skipif(not os.path.exists(_SPACE_MASKS), reason="spacetime stack not mounted")
def test_route_end_to_end_with_omnipose_plugin():
    """Full chain: HTTP -> omnipose plugin -> Segmenter -> _volume3d bundle."""
    pytest.importorskip("omnipose")
    from omnipose.gui.ocdkit_plugin import plugin as omni_plugin

    client = TestClient(_app(omni_plugin))
    r = client.post("/api/volume", json={
        "masksPath": _SPACE_MASKS,
        "doFlow": False, "doAffinity": False, "embedVolumes": False,
    })
    assert r.status_code == 200
    b = r.json()
    assert b["meta"] == {"dim": 3, "axes": ["t", "y", "x"],
                         "depth": 133, "height": 302, "width": 302, "nLabels": 40}
    assert len(b["trajectories"]["edges"]) == 36
    assert len(b["trajectories"]["tracks"]) == 40
