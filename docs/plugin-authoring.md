# Authoring an ocdkit.viewer plugin

This document is the source-of-truth contract for adding a new segmentation
backend to the ocdkit viewer. It is written so an LLM can read it and produce
a working plugin in one shot.

## What a plugin is

A plugin wires a segmentation tool (Cellpose, StarDist, SAM, your own model,
…) into the ocdkit viewer. The viewer handles:

- Image loading, display, panning, zooming
- Mask rendering (n-coloring, opacity, color tables)
- Manual mask editing (brush, fill, split, merge)
- Affinity-graph editing
- Session persistence

Your plugin only needs to:

1. Declare the parameters the user can tune.
2. Provide a `run(image, params) -> mask` function.
3. (Optionally) declare lifecycle hooks: model loading, GPU toggle, cache.

## The `SegmentationPlugin` contract

```python
from ocdkit.viewer import SegmentationPlugin, WidgetSpec

plugin = SegmentationPlugin(
    name="my_tool",                # stable plugin id (lowercase, no spaces)
    version="0.1.0",
    description="Brief one-line description.",
    homepage="https://github.com/you/my_tool",
    widgets=[
        WidgetSpec(
            name="threshold",        # key passed into params
            label="Threshold",       # UI label
            kind="slider",           # widget kind (see below)
            default=0.5,
            min=0.0, max=1.0, step=0.01,
            help="Pixels above this become foreground.",
            group="Detection",        # subsection header
        ),
        # ... more WidgetSpec entries ...
    ],
    run=my_segmentation_function,
    # optional:
    load_models=lambda: ["model_a", "model_b"],
    warmup=lambda model_id: None,
    set_use_gpu=lambda enabled: None,
    clear_cache=lambda: None,
)
```

## `run(image, params) -> mask` contract

```python
def my_segmentation_function(image: np.ndarray, params: dict) -> np.ndarray:
    """
    image: uint8 numpy array.
        - Shape (H, W) for grayscale.
        - Shape (H, W, C) for multichannel/RGB. C is 1, 2, 3, or 4.
    params: dict whose keys are the WidgetSpec.name strings you declared.
        Values are typed: numbers for slider/number, bool for toggle, str for
        dropdown/text/file/color/colormap.
    returns: 2D int32 (or smaller int) numpy array, shape (H, W).
        0 = background. Positive integers are instance ids.
        The viewer takes care of n-coloring and rendering.
    """
    ...
```

## Widget kinds

| `kind`        | Type    | Required fields            | Notes                                           |
| ------------- | ------- | -------------------------- | ----------------------------------------------- |
| `slider`      | float   | `min`, `max` (`step`)      | Continuous range with handle                    |
| `slider_log`  | float   | `min` > 0, `max` (`step`)  | Log-scaled slider                               |
| `number`      | float   | `min`, `max` (`step`)      | Plain number input box                          |
| `toggle`      | bool    | `default` must be bool     | Checkbox / switch                               |
| `dropdown`    | str     | `choices`                  | Single-select from list of strings              |
| `text`        | str     | —                          | Free-form text input                            |
| `file`        | str     | —                          | File path picker                                |
| `color`       | str     | —                          | Hex color string (`"#ff0000"`)                  |
| `colormap`    | str     | —                          | Colormap name from ocdkit.cmap                  |

### Conditional visibility

A widget can be hidden until other widgets have specific values:

```python
WidgetSpec(
    name="manual_value",
    label="Manual value",
    kind="slider", default=0.5, min=0.0, max=1.0,
    visible_when={"method": "manual"},   # show only when method == "manual"
)
```

Multiple keys are AND-ed: the widget shows only when all entries match.

### Grouping

`group="Detection"` puts widgets under a collapsible "Detection" section in
the pane. Widgets without `group` go into a default top section.

## Lifecycle hooks

All optional. Skip the ones you don't need.

| Hook            | Signature              | When called                                          |
| --------------- | ---------------------- | ---------------------------------------------------- |
| `load_models`   | `() -> list[str]`      | Once on plugin selection — populates the model list. |
| `warmup`        | `(model_id) -> None`   | When the user picks a model — preload it.            |
| `set_use_gpu`   | `(enabled: bool) -> None` | When the user toggles the GPU switch.             |
| `clear_cache`   | `() -> None`           | When the user clicks "Clear cache".                  |

If `load_models` is provided, the viewer renders a model dropdown at the top
of the pane and passes the chosen `model` value in `params["model"]` on each
`run()` call — you do not need to declare a `WidgetSpec` for it.

## Registering the plugin

### Option A — entry point (recommended for installed packages)

In your `pyproject.toml`:

```toml
[project.entry-points."ocdkit.plugins"]
my_tool = "my_tool.ocdkit_plugin:plugin"
```

Where `my_tool/ocdkit_plugin.py` contains:

```python
from ocdkit.viewer import SegmentationPlugin, WidgetSpec
plugin = SegmentationPlugin(name="my_tool", ..., run=...)
```

The viewer auto-discovers all `ocdkit.plugins` entry points at startup.

### Option B — explicit registration (for in-process or tests)

```python
from ocdkit.viewer import register_plugin
register_plugin(plugin)
```

## Validation

Both `WidgetSpec(...)` and `SegmentationPlugin(...)` validate their arguments
in `__post_init__`. Common errors raised at construction time:

- `WidgetSpec.name` empty or non-string → `ValueError`
- numeric kinds without `min`/`max` → `ValueError`
- `slider_log` with `min <= 0` → `ValueError`
- `dropdown` without `choices` → `ValueError`
- `dropdown` whose `default` is not in `choices` → `ValueError`
- `toggle` with non-bool `default` → `ValueError`
- duplicate widget `name` within one plugin → `ValueError`

Programmatic schemas for tools and tests:

```python
from ocdkit.viewer.plugins.schema import (
    widget_spec_schema,        # JSON Schema for one WidgetSpec
    plugin_manifest_schema,    # JSON Schema for plugin.manifest()
)
```

## Minimal complete example

```python
# my_tool/ocdkit_plugin.py
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from ocdkit.viewer import SegmentationPlugin, WidgetSpec


def _run(image: np.ndarray, params: dict) -> np.ndarray:
    if image.ndim == 3:
        image = image.mean(axis=-1).astype(image.dtype)
    cutoff = float(params["threshold"]) * 255.0
    if params.get("method") == "otsu":
        cutoff = float(threshold_otsu(image))
    binary = (image > cutoff)
    if params.get("invert"):
        binary = ~binary
    return label(binary).astype(np.int32)


plugin = SegmentationPlugin(
    name="simple_threshold",
    version="0.1.0",
    description="Otsu / manual threshold + connected components.",
    widgets=[
        WidgetSpec("method", "Method", "dropdown",
                   default="otsu", choices=["otsu", "manual"]),
        WidgetSpec("threshold", "Threshold", "slider",
                   default=0.5, min=0.0, max=1.0, step=0.01,
                   visible_when={"method": "manual"}),
        WidgetSpec("invert", "Invert", "toggle", default=False),
    ],
    run=_run,
)
```

That's the entire plugin. Once entry-pointed, it appears in the viewer's
plugin dropdown automatically.

## Conventions

- Plugin `name` is lowercase, snake_case. It appears in URLs.
- Widget `name`s are also snake_case and are the keys in `params`.
- Don't mutate `image` in `run()`; return a new array.
- Return masks as 2D `int32` (or `uint16`/`int64` work too). Background = 0.
- Lifecycle hooks should be cheap or fork to a worker thread internally.
- Heavy imports (torch, your model library) belong **inside** `run()` or
  inside `warmup()`, not at module top — keeps viewer startup fast.
