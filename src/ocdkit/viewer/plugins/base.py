"""Plugin contract for ocdkit.viewer.

A plugin is a :class:`SegmentationPlugin` that declares:

* a list of user-facing parameters (:class:`WidgetSpec`),
* a ``run(image, params) -> mask`` callable,
* optional lifecycle hooks (model loading, GPU toggle, cache clearing,
  resegment-from-cache, affinity-graph relabeling).

The viewer renders the widget specs as a settings pane in the browser, ships
the form values back to the server on each segmentation request, and forwards
them as ``params`` to the plugin's ``run`` function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

WidgetKind = Literal[
    "slider",        # numeric range with handle (continuous)
    "slider_log",    # log-scaled numeric slider
    "number",        # plain number input
    "toggle",        # boolean checkbox / switch
    "dropdown",      # single-select from `choices`
    "text",          # free-form text input
    "file",          # file path picker
    "color",         # hex color picker
    "colormap",      # colormap name (uses ocdkit.cmap registry)
]

# Type aliases for clarity
# A plugin's run() may return either:
#   - just the mask: np.ndarray (shape HxW, integer dtype)
#   - or (mask, extras): (np.ndarray, dict[str, Any]) — extras are forwarded
#     to the frontend as-is (e.g. flow overlay PNG data URLs, affinity graph
#     payloads, points, etc.)
RunResult = Union[np.ndarray, Tuple[np.ndarray, Mapping[str, Any]]]
RunCallable = Callable[[np.ndarray, Mapping[str, Any]], RunResult]
ResegmentCallable = Callable[[Mapping[str, Any]], RunResult]
RelabelFromAffinityCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
LoadModelsCallable = Callable[[], Sequence[str]]
WarmupCallable = Callable[[str], None]
SetUseGPUCallable = Callable[[bool], None]
GetUseGPUCallable = Callable[[], bool]
ClearCacheCallable = Callable[[], None]


@dataclass(frozen=True)
class WidgetSpec:
    """Declarative spec for one parameter knob.

    The ``name`` is the stable key passed to ``plugin.run(image, params)``.
    The ``kind`` selects which frontend widget is rendered.
    """

    name: str
    label: str
    kind: WidgetKind
    default: Any
    # numeric kinds (slider, slider_log, number)
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    # dropdown kind
    choices: Optional[Sequence[str]] = None
    # all kinds
    help: Optional[str] = None
    group: Optional[str] = None  # subsection header in the pane
    # conditional visibility: this widget shows only when other params match
    # e.g. {"use_advanced": True} — multiple keys are AND-ed
    visible_when: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for the frontend."""
        out: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
        }
        for attr in ("min", "max", "step", "help", "group"):
            value = getattr(self, attr)
            if value is not None:
                out[attr] = value
        if self.choices is not None:
            out["choices"] = list(self.choices)
        if self.visible_when is not None:
            out["visibleWhen"] = dict(self.visible_when)
        return out

    def __post_init__(self) -> None:
        _validate_widget(self)


def _validate_widget(w: WidgetSpec) -> None:
    """Sanity-check a widget spec at construction time."""
    if not w.name or not isinstance(w.name, str):
        raise ValueError("WidgetSpec.name must be a non-empty string")
    if not w.label:
        raise ValueError(f"WidgetSpec({w.name}).label is required")
    if w.kind in ("slider", "slider_log", "number"):
        if w.min is None or w.max is None:
            raise ValueError(
                f"WidgetSpec({w.name}, kind={w.kind}) requires both min and max"
            )
        if w.max <= w.min:
            raise ValueError(
                f"WidgetSpec({w.name}) requires max > min"
            )
        if w.kind == "slider_log" and w.min <= 0:
            raise ValueError(
                f"WidgetSpec({w.name}, kind=slider_log) requires min > 0"
            )
    if w.kind == "dropdown":
        if not w.choices:
            raise ValueError(f"WidgetSpec({w.name}, kind=dropdown) requires choices")
        if w.default not in w.choices:
            raise ValueError(
                f"WidgetSpec({w.name}) default {w.default!r} not in choices"
            )
    if w.kind == "toggle" and not isinstance(w.default, bool):
        raise ValueError(f"WidgetSpec({w.name}, kind=toggle) default must be bool")


@dataclass
class SegmentationPlugin:
    """A registered segmentation tool.

    Required:
        name: stable plugin id (used in URLs, e.g. "omnipose").
        version: plugin version string.
        widgets: ordered list of user-facing parameter specs.
        run: ``(image, params) -> mask`` — see contract below.

    Optional lifecycle hooks:
        load_models: returns the available model id strings.
        warmup: preload the named model (called on plugin selection).
        set_use_gpu / get_use_gpu: toggle / probe GPU on/off.
        clear_cache: drop any cached state (called by the user-facing button).
        resegment: re-run mask reconstruction from cached intermediates with
            new threshold/cluster settings — returns same shape as run().
        relabel_from_affinity: rebuild instance labels from a user-edited
            spatial affinity graph; (mask, spatial, steps) -> new labels.

    Run contract:
        ``image`` is a uint8 numpy array, shape (H, W) for grayscale or
        (H, W, C) for multichannel/RGB. ``params`` is a dict whose keys match
        the plugin's WidgetSpec names. The return value is either:
            - a 2D integer mask (shape HxW; 0 = background; >0 = instance id), or
            - a tuple ``(mask, extras)`` where ``extras`` is a JSON-serializable
              dict of per-plugin payloads (flow overlay PNG data URLs, affinity
              graph encodings, etc.). The viewer forwards extras unchanged to
              the frontend.
        The viewer handles n-coloring and RGB rendering of the mask itself.
    """

    name: str
    version: str
    widgets: Sequence[WidgetSpec]
    run: RunCallable
    load_models: Optional[LoadModelsCallable] = None
    warmup: Optional[WarmupCallable] = None
    set_use_gpu: Optional[SetUseGPUCallable] = None
    get_use_gpu: Optional[GetUseGPUCallable] = None
    clear_cache: Optional[ClearCacheCallable] = None
    resegment: Optional[ResegmentCallable] = None
    relabel_from_affinity: Optional[RelabelFromAffinityCallable] = None
    description: str = ""
    homepage: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SegmentationPlugin.name is required")
        if not self.version:
            raise ValueError(f"SegmentationPlugin({self.name}).version is required")
        if not callable(self.run):
            raise TypeError(f"SegmentationPlugin({self.name}).run must be callable")
        seen: set[str] = set()
        for w in self.widgets:
            if not isinstance(w, WidgetSpec):
                raise TypeError(
                    f"SegmentationPlugin({self.name}).widgets must contain WidgetSpec instances"
                )
            if w.name in seen:
                raise ValueError(
                    f"SegmentationPlugin({self.name}): duplicate widget name {w.name!r}"
                )
            seen.add(w.name)

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable description of this plugin."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "homepage": self.homepage,
            "widgets": [w.to_dict() for w in self.widgets],
            "models": list(self.load_models()) if self.load_models else [],
            "capabilities": {
                "warmup": self.warmup is not None,
                "set_use_gpu": self.set_use_gpu is not None,
                "get_use_gpu": self.get_use_gpu is not None,
                "clear_cache": self.clear_cache is not None,
                "resegment": self.resegment is not None,
                "relabel_from_affinity": self.relabel_from_affinity is not None,
            },
        }

    def defaults(self) -> dict[str, Any]:
        """Return a dict of widget-name → default value."""
        return {w.name: w.default for w in self.widgets}


def split_run_result(result: RunResult) -> Tuple[np.ndarray, dict[str, Any]]:
    """Normalize a plugin run() return value to ``(mask, extras_dict)``."""
    if isinstance(result, tuple):
        if len(result) != 2:
            raise TypeError(
                "plugin run() tuple must have exactly 2 elements (mask, extras)"
            )
        mask, extras = result
        if not isinstance(extras, Mapping):
            raise TypeError(
                f"plugin run() extras must be a Mapping, got {type(extras).__name__}"
            )
        return np.asarray(mask), dict(extras)
    return np.asarray(result), {}
