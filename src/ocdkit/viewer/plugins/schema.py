"""JSON schemas describing plugin manifests and widget specs.

Useful for:

* Documentation tools (rendering the contract from the schema).
* Validating LLM-authored plugins before registration.
* Generating frontend type definitions.
"""

from __future__ import annotations

from typing import Any

from .base import WidgetSpec


def widget_spec_schema() -> dict[str, Any]:
    """Return the JSON Schema for one :class:`WidgetSpec`."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "ocdkit.viewer.WidgetSpec",
        "type": "object",
        "required": ["name", "label", "kind", "default"],
        "additionalProperties": False,
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "Stable parameter key passed to plugin.run(image, params).",
            },
            "label": {
                "type": "string",
                "minLength": 1,
                "description": "Human-readable label rendered above/beside the widget.",
            },
            "kind": {
                "type": "string",
                "enum": [
                    "slider",
                    "slider_log",
                    "number",
                    "toggle",
                    "dropdown",
                    "text",
                    "file",
                    "color",
                    "colormap",
                ],
            },
            "default": {
                "description": "Default value. Type depends on kind."
            },
            "min": {"type": "number"},
            "max": {"type": "number"},
            "step": {"type": "number"},
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required for kind=dropdown.",
            },
            "help": {
                "type": "string",
                "description": "Tooltip / help text shown on hover.",
            },
            "group": {
                "type": "string",
                "description": "Optional subsection header.",
            },
            "visibleWhen": {
                "type": "object",
                "description": "Map of {other_param: required_value}. Widget hides until all match.",
            },
        },
        "allOf": [
            {
                "if": {"properties": {"kind": {"enum": ["slider", "slider_log", "number"]}}},
                "then": {"required": ["min", "max"]},
            },
            {
                "if": {"properties": {"kind": {"const": "dropdown"}}},
                "then": {"required": ["choices"]},
            },
            {
                "if": {"properties": {"kind": {"const": "slider_log"}}},
                "then": {"properties": {"min": {"exclusiveMinimum": 0}}},
            },
            {
                "if": {"properties": {"kind": {"const": "toggle"}}},
                "then": {"properties": {"default": {"type": "boolean"}}},
            },
        ],
    }


def plugin_manifest_schema() -> dict[str, Any]:
    """Return the JSON Schema for a plugin manifest dict."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "ocdkit.viewer.PluginManifest",
        "type": "object",
        "required": ["name", "version", "widgets"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "version": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "homepage": {"type": "string"},
            "widgets": {
                "type": "array",
                "items": widget_spec_schema(),
            },
            "models": {
                "type": "array",
                "items": {"type": "string"},
            },
            "capabilities": {
                "type": "object",
                "properties": {
                    "warmup": {"type": "boolean"},
                    "set_use_gpu": {"type": "boolean"},
                    "clear_cache": {"type": "boolean"},
                },
            },
        },
    }


def example_widget_specs() -> list[WidgetSpec]:
    """Return a non-empty list of WidgetSpec for tests and documentation."""
    return [
        WidgetSpec(
            name="threshold",
            label="Threshold",
            kind="slider",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            help="Pixels above this value become foreground.",
            group="Detection",
        ),
        WidgetSpec(
            name="min_size",
            label="Min size (px)",
            kind="number",
            default=20,
            min=0,
            max=10000,
            step=1,
            group="Filtering",
        ),
        WidgetSpec(
            name="invert",
            label="Invert",
            kind="toggle",
            default=False,
            help="Swap foreground/background polarity.",
        ),
        WidgetSpec(
            name="method",
            label="Threshold method",
            kind="dropdown",
            default="otsu",
            choices=["otsu", "li", "yen", "triangle", "manual"],
            group="Detection",
        ),
        WidgetSpec(
            name="manual_value",
            label="Manual value",
            kind="slider",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            visible_when={"method": "manual"},
            group="Detection",
        ),
    ]
