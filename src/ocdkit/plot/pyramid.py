"""Backward-compat re-export.

The pyramid builder moved to :mod:`ocdkit.tileserve._pyramid` — a torch-free
module — so the out-of-process tile server (and its child process) can import it
WITHOUT pulling in the heavy ``ocdkit.plot`` package init (which eagerly imports
figure + image_grid, dragging in torch/dask/pandas). New code should import from
there; this shim keeps ``ocdkit.plot.pyramid`` working for existing callers.
"""
from ..tileserve._pyramid import (  # noqa: F401
    image_pyramid, pyramid_dims)
