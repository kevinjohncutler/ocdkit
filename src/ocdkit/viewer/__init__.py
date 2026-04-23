"""ocdkit.viewer — generic image-viewer + segmentation-editor toolkit.

Plugins register a :class:`SegmentationPlugin` via the ``ocdkit.plugins`` entry
point group. The viewer discovers them at startup, builds a settings pane from
each plugin's :class:`WidgetSpec` list, and dispatches segmentation calls back
to the plugin's ``run()`` callable.

See ``docs/plugin-authoring.md`` for the LLM-facing plugin contract.
"""

from .plugins.base import SegmentationPlugin, WidgetSpec, WidgetKind
from .plugins.registry import (
    PluginRegistry,
    REGISTRY,
    get_plugin,
    list_plugins,
    register_plugin,
)

__all__ = [
    "SegmentationPlugin",
    "WidgetSpec",
    "WidgetKind",
    "PluginRegistry",
    "REGISTRY",
    "get_plugin",
    "list_plugins",
    "register_plugin",
]
