"""Plotting utilities — figure creation, image grids, label styling, and colorization."""

from ..load import enable_submodules

enable_submodules(__name__, expose=True)

# Explicit re-export so ``from ocdkit.plot import figure`` returns the
# function (the ``figure`` submodule of the same name would otherwise
# shadow it when Python's import machinery sets ``ocdkit.plot.figure =
# <module>``). NumPy uses the same pattern in its ``__init__`` files.
from .figure import figure  # noqa: E402,F401
# Same shadow-avoidance pattern for image_grid — both a module name and
# the dispatcher live under ``ocdkit.plot.image_grid``.
from .image_grid import image_grid  # noqa: E402,F401


# ─── application-level configuration ────────────────────────────────


class _PlotConfig:
    """Process-level defaults for ocdkit.plot. Mutated via
    :func:`setup`; read by ``image_grid`` and friends."""

    # Low default so the thumbnail → hi-res handoff is visibly obvious
    # during testing. Production callers bump via setup(target_tile_px=)
    # for sharper thumbs when the hi-res transport isn't reachable.
    target_tile_px: int = 256


_config = _PlotConfig()


def setup(*, registry=None, target_tile_px=None):
    """Configure ocdkit.plot defaults for the current process.

    Optional — auto-detection handles the common cases without a
    ``setup`` call (Jupyter + jupyter-server-proxy → JupyterProxy; else
    LocalHttp). Use ``setup`` to inject a custom transport or to bump
    default tile resolution.

    Parameters
    ----------
    registry
        A :class:`ocdkit.io.figure_server.FigureSourceRegistry`
        instance to use for hi-res tile streaming. Passing ``None``
        leaves the active registry unchanged (call
        :func:`ocdkit.io.figure_server.set_registry(None)` to reset
        to auto-detect).
    target_tile_px
        Default pixel height for ``image_grid`` tiles. Higher = sharper
        thumbnails and a sharper fallback when no hi-res transport is
        reachable; larger SVG payload.
    """
    if registry is not None:
        from ..io.figure_server import set_registry
        set_registry(registry)
    if target_tile_px is not None:
        _config.target_tile_px = int(target_tile_px)
