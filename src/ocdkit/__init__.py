"""
ocdkit — Obsessive Coder's Dependency Toolkit.

Python utilities for array manipulation, GPU dispatch, image I/O,
morphology, and plotting. Designed for use across multiple projects. 
"""

from .load import enable_submodules

# Exclude ``__main__`` from auto-discovery: it's a ``python -m ocdkit`` entry
# point, not a package submodule, and eager import here would trigger a
# double-execution warning when Python later runs it as ``__main__``.
enable_submodules(__name__, exclude=["__main__"])
