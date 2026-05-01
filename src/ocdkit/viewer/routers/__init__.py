"""HTTP route modules.

Each submodule defines an :class:`fastapi.APIRouter`; the application factory
in :mod:`ocdkit.viewer.app` includes them all.
"""

from . import index, log, mask, plugin, segment, session_routes, system, trust

__all__ = ["index", "log", "mask", "plugin", "segment", "session_routes", "system", "trust"]
