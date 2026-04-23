"""Top-level command-line interface for ocdkit.

Subcommands live in sibling modules and are wired into the dispatcher
in :mod:`ocdkit.cli.main`.
"""

from ..load import enable_submodules

enable_submodules(__name__)
