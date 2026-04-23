"""Plugin contract for ocdkit.viewer."""

from .base import SegmentationPlugin, WidgetSpec, WidgetKind, split_run_result
from .registry import (
    PluginRegistry,
    REGISTRY,
    get_plugin,
    list_plugins,
    register_plugin,
)
from .schema import widget_spec_schema, plugin_manifest_schema

__all__ = [
    "SegmentationPlugin",
    "WidgetSpec",
    "WidgetKind",
    "split_run_result",
    "PluginRegistry",
    "REGISTRY",
    "get_plugin",
    "list_plugins",
    "register_plugin",
    "widget_spec_schema",
    "plugin_manifest_schema",
]
