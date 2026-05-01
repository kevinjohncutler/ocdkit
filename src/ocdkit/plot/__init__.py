"""Plotting utilities — figure creation, image grids, label styling, and colorization."""

from ..load import enable_submodules

enable_submodules(__name__, expose=True)

# Explicit re-export so ``from ocdkit.plot import figure`` returns the
# function (the ``figure`` submodule of the same name would otherwise
# shadow it when Python's import machinery sets ``ocdkit.plot.figure =
# <module>``). NumPy uses the same pattern in its ``__init__`` files.
from .figure import figure  # noqa: E402,F401
