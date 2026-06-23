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

    Parameters
    ----------
    registry
        Deprecated / ignored. The per-kernel hi-res HTTP registry has
        been retired — ``image_grid`` hosts hi-res + display tiles as
        ocdkit.tileserve attachments (reachable remotely via the
        ``ocdkit-tiles`` Jupyter proxy). Accepted for backward
        compatibility but has no effect.
    target_tile_px
        Default pixel height for ``image_grid`` tiles. Higher = sharper
        thumbnails; larger payload.
    """
    if registry is not None:
        import warnings
        warnings.warn(
            "ocdkit.plot.setup(registry=...) is retired and ignored; image_grid "
            "now serves tiles via ocdkit.tileserve attachments.",
            DeprecationWarning, stacklevel=2)
    if target_tile_px is not None:
        _config.target_tile_px = int(target_tile_px)

    # Configure the matplotlib theme + (inside a kernel) the Jupyter inline
    # environment. This used to be a separate ``ocdkit.plot.defaults.setup()``
    # that the name ``setup`` here shadowed, so notebooks calling
    # ``ocdkit.plot.setup()`` silently got NO dark-mode rcParams (white
    # figure/axes backgrounds). Fold it in so the single public entry point
    # applies the theme.
    from . import defaults
    defaults.setup()

    # Pre-warm the tileserve server in the background. The first interactive
    # figure (or rich ``_repr_html_``) otherwise pays the ~0.7s FastAPI+uvicorn
    # import and socket startup on its own critical path, because ``register()``
    # → ``ensure_server()`` blocks until the port accepts connections. ``setup()``
    # is an explicit "I'm about to plot" signal and runs early (the notebook's
    # import cell), so kicking the import+startup onto a daemon thread here means
    # the server is up by the time the first display happens — the later
    # ``ensure_server()`` call then returns instantly (it is idempotent).
    # Daemon thread → never blocks ``setup()``. Disable with
    # ``OCDKIT_TILESERVE_PREWARM=0``. Only inside an interactive kernel, where
    # displays actually happen (a headless render-to-file script needs no server).
    import os
    if os.environ.get("OCDKIT_TILESERVE_PREWARM", "1") != "0":
        try:
            from IPython import get_ipython
            in_kernel = get_ipython() is not None
        except Exception:
            in_kernel = False
        if in_kernel:
            import threading

            def _prewarm_tileserve():
                try:
                    from ..tileserve.server import ensure_server
                    ensure_server()
                except Exception:
                    pass

            threading.Thread(target=_prewarm_tileserve, daemon=True,
                             name="ocdkit-tileserve-prewarm").start()
