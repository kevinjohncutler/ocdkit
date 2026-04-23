"""Unit tests for the plugin contract (base + registry + schema)."""

from __future__ import annotations

import numpy as np
import pytest

from ocdkit.viewer import SegmentationPlugin, WidgetSpec, register_plugin
from ocdkit.viewer.plugins.registry import PluginRegistry


# --- WidgetSpec validation --------------------------------------------------

def test_slider_requires_min_max():
    with pytest.raises(ValueError, match="requires both min and max"):
        WidgetSpec(name="x", label="X", kind="slider", default=0.0)


def test_slider_requires_max_greater_than_min():
    with pytest.raises(ValueError, match="max > min"):
        WidgetSpec(name="x", label="X", kind="slider", default=0.0, min=1.0, max=0.5)


def test_slider_log_requires_positive_min():
    with pytest.raises(ValueError, match="min > 0"):
        WidgetSpec(name="x", label="X", kind="slider_log", default=1.0, min=0.0, max=10.0)


def test_dropdown_requires_choices():
    with pytest.raises(ValueError, match="requires choices"):
        WidgetSpec(name="x", label="X", kind="dropdown", default="a")


def test_dropdown_default_must_be_in_choices():
    with pytest.raises(ValueError, match="not in choices"):
        WidgetSpec(name="x", label="X", kind="dropdown", default="z", choices=["a", "b"])


def test_toggle_default_must_be_bool():
    with pytest.raises(ValueError, match="default must be bool"):
        WidgetSpec(name="x", label="X", kind="toggle", default=1)


def test_widget_to_dict_round_trip():
    w = WidgetSpec(
        name="threshold", label="Threshold", kind="slider",
        default=0.5, min=0.0, max=1.0, step=0.01,
        help="help text", group="Detection",
        visible_when={"method": "manual"},
    )
    d = w.to_dict()
    assert d["name"] == "threshold"
    assert d["min"] == 0.0
    assert d["step"] == 0.01
    assert d["visibleWhen"] == {"method": "manual"}


# --- SegmentationPlugin validation ------------------------------------------

def _mk_plugin(name="test", widgets=None, **kw) -> SegmentationPlugin:
    return SegmentationPlugin(
        name=name,
        version="0.0.1",
        widgets=widgets or [
            WidgetSpec("t", "Threshold", "slider", default=0.5, min=0.0, max=1.0)
        ],
        run=kw.pop("run", lambda img, params: np.zeros(img.shape[:2], dtype=np.int32)),
        **kw,
    )


def test_plugin_requires_name():
    with pytest.raises(ValueError, match="name is required"):
        _mk_plugin(name="")


def test_plugin_duplicate_widget_name_rejected():
    with pytest.raises(ValueError, match="duplicate widget name"):
        _mk_plugin(widgets=[
            WidgetSpec("t", "T", "slider", default=0.5, min=0.0, max=1.0),
            WidgetSpec("t", "T2", "slider", default=0.3, min=0.0, max=1.0),
        ])


def test_plugin_manifest_is_json_serializable():
    import json

    plugin = _mk_plugin(description="desc")
    manifest = plugin.manifest()
    assert manifest["name"] == "test"
    assert manifest["widgets"][0]["name"] == "t"
    json.dumps(manifest)  # must not raise


def test_plugin_defaults_extracts_widget_defaults():
    plugin = _mk_plugin(widgets=[
        WidgetSpec("a", "A", "slider", default=0.5, min=0, max=1),
        WidgetSpec("b", "B", "toggle", default=True),
    ])
    assert plugin.defaults() == {"a": 0.5, "b": True}


# --- Registry ----------------------------------------------------------------

def test_registry_register_and_get():
    reg = PluginRegistry()
    p = _mk_plugin(name="thing")
    reg.register(p)
    assert "thing" in reg
    assert reg.get("thing") is p


def test_registry_duplicate_rejected():
    reg = PluginRegistry()
    reg.register(_mk_plugin(name="a"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_mk_plugin(name="a"))


def test_registry_replace_allowed():
    reg = PluginRegistry()
    a1 = _mk_plugin(name="a")
    a2 = _mk_plugin(name="a")
    reg.register(a1)
    reg.register(a2, replace=True)
    assert reg.get("a") is a2


def test_registry_names_sorted():
    reg = PluginRegistry()
    reg._discovered = True  # opt out of entry-point discovery for unit isolation
    for n in ("z", "a", "m"):
        reg.register(_mk_plugin(name=n))
    assert reg.names() == ["a", "m", "z"]


def test_registry_unregister():
    reg = PluginRegistry()
    reg.register(_mk_plugin(name="a"))
    reg.unregister("a")
    assert "a" not in reg
    reg.unregister("a")  # no-op


def test_registry_len_and_iter():
    reg = PluginRegistry()
    reg._discovered = True  # opt out of entry-point discovery for unit isolation
    reg.register(_mk_plugin(name="x"))
    reg.register(_mk_plugin(name="y"))
    assert len(reg) == 2
    assert [p.name for p in reg] == ["x", "y"]


def test_registry_register_type_checked():
    reg = PluginRegistry()
    with pytest.raises(TypeError):
        reg.register("not a plugin")  # type: ignore[arg-type]


# --- Schema ------------------------------------------------------------------

def test_schema_modules_are_dicts():
    from ocdkit.viewer.plugins.schema import (
        example_widget_specs,
        plugin_manifest_schema,
        widget_spec_schema,
    )
    ws = widget_spec_schema()
    pm = plugin_manifest_schema()
    assert isinstance(ws, dict) and "properties" in ws
    assert isinstance(pm, dict) and "properties" in pm
    examples = example_widget_specs()
    assert all(isinstance(w, WidgetSpec) for w in examples)


# --- Built-in threshold plugin ----------------------------------------------

def test_threshold_plugin_runs_on_synthetic_image():
    pytest.importorskip("skimage")
    from ocdkit.viewer.plugins.threshold import plugin

    rng = np.random.default_rng(0)
    image = (rng.random((64, 64)) * 255).astype(np.uint8)
    # Make two bright blobs
    image[10:20, 10:20] = 250
    image[40:55, 40:55] = 240

    params = dict(plugin.defaults())
    mask = plugin.run(image, params)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.int32
    assert mask.max() >= 2


def test_threshold_plugin_manual_mode():
    pytest.importorskip("skimage")
    from ocdkit.viewer.plugins.threshold import plugin

    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:16, 8:16] = 200
    mask = plugin.run(image, {"method": "manual", "threshold": 0.5, "invert": False, "min_size": 0})
    assert mask.max() == 1
    assert mask[0, 0] == 0
    assert mask[12, 12] == 1
