"""Built-in reference plugin: skimage threshold + connected components.

Ships with ocdkit so the viewer is usable out of the box and so the plugin
contract has a tested reference implementation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SegmentationPlugin, WidgetSpec

_METHODS = ("otsu", "li", "yen", "triangle", "mean", "minimum", "manual")


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        # Simple mean over channels; viewer hands us uint8 already.
        return image.mean(axis=-1).astype(image.dtype)
    raise ValueError(f"threshold plugin expects 2D or 3D image, got ndim={image.ndim}")


def _run(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    from skimage import filters, measure, morphology

    gray = _to_grayscale(image)
    method = str(params.get("method", "otsu"))

    if method == "manual":
        cutoff = float(params.get("threshold", 0.5)) * 255.0
    else:
        fn = {
            "otsu": filters.threshold_otsu,
            "li": filters.threshold_li,
            "yen": filters.threshold_yen,
            "triangle": filters.threshold_triangle,
            "mean": filters.threshold_mean,
            "minimum": filters.threshold_minimum,
        }[method]
        cutoff = float(fn(gray))

    binary = gray > cutoff
    if bool(params.get("invert", False)):
        binary = ~binary

    min_size = int(params.get("min_size", 20))
    if min_size > 0:
        binary = morphology.remove_small_objects(binary, min_size=min_size)

    return measure.label(binary, connectivity=2).astype(np.int32)


plugin = SegmentationPlugin(
    name="threshold",
    version="0.1.0",
    description="Classical threshold + connected components (skimage).",
    homepage="https://scikit-image.org/docs/stable/api/skimage.filters.html",
    widgets=[
        WidgetSpec(
            name="method",
            label="Method",
            kind="dropdown",
            default="otsu",
            choices=_METHODS,
            help="Automatic threshold selector, or 'manual' to set a cutoff by hand.",
            group="Detection",
        ),
        WidgetSpec(
            name="threshold",
            label="Manual threshold",
            kind="slider",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            help="Pixel cutoff (0–1, relative to full uint8 range).",
            group="Detection",
            visible_when={"method": "manual"},
        ),
        WidgetSpec(
            name="invert",
            label="Invert",
            kind="toggle",
            default=False,
            help="Swap foreground/background polarity.",
            group="Detection",
        ),
        WidgetSpec(
            name="min_size",
            label="Min object size (px)",
            kind="number",
            default=20,
            min=0,
            max=100000,
            step=1,
            help="Drop connected components smaller than this area.",
            group="Filtering",
        ),
    ],
    run=_run,
)
