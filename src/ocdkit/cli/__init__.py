"""Top-level command-line interface for ocdkit.

Subcommands live in sibling modules and are wired into the dispatcher
in :mod:`ocdkit.cli.main`.
"""

from ..load import enable_submodules

enable_submodules(__name__)

# Explicit re-export so ``from ocdkit.cli import main`` returns the entry
# point function (without this, the same name resolves to the submodule
# because the import system sets ``ocdkit.cli.main = <module>``).
from .main import main  # noqa: E402,F401
