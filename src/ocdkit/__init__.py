"""
ocdkit — Obsessive Coder's Dependency Toolkit.

Python utilities for array manipulation, GPU dispatch, image I/O,
morphology, and plotting. Designed for use across multiple projects. 
"""

from .load import enable_submodules

# Top-level: lazy. Sub-packages (ocdkit.array, ocdkit.io, ocdkit.plot, …) load
# only when accessed, keeping bare ``import ocdkit`` near-instant.
# ``__main__`` is excluded because it's a ``python -m ocdkit`` entry point,
# not a sub-module — eager import would trigger a double-execution warning.
enable_submodules(__name__, exclude=["__main__"], expose=False)
