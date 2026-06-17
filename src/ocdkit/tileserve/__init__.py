"""Generic in-kernel tile server for zoomable GPU-colormapped linked viewers.

See :mod:`ocdkit.tileserve.server`. A host application:
  * declares sources + layers (``register_pending`` / ``fill`` / ``register_lazy``),
  * ``attach``es opaque named blobs (overlays, per-object geometry, panel data),
  * mounts its own routes (viewer HTML, domain endpoints) via ``register_extension``,
  * starts the daemon with ``ensure_server``.
"""
from .server import (
    TileSource,
    register, register_pending, register_array, fill, register_lazy, attach,
    get_source, drop,
    register_extension, register_reset_hook,
    make_app, ensure_server, reset_server,
)
from .embed import embed_viewer, figure_embed_height

__all__ = [
    "TileSource", "register", "register_pending", "register_array", "fill",
    "register_lazy", "attach", "get_source", "drop", "register_extension",
    "register_reset_hook", "make_app", "ensure_server", "reset_server",
    "embed_viewer", "figure_embed_height",
]
