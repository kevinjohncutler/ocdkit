"""
ocdkit — Obsessive Coder's Dependency Toolkit.

Python utilities for array manipulation, GPU dispatch, image I/O,
morphology, and plotting. Designed for use across multiple projects.
"""

import os as _os
import pathlib as _pathlib

# Numba JIT cache: pin to local home so SMB latency on NAS-mounted source
# trees doesn't add seconds to every fresh subprocess that imports a
# numba-using module. Compiled artifacts are machine-local anyway (CPU /
# Python version specific) — they shouldn't live on a shared NAS. ``setdefault``
# yields to any explicit user override (env var or shell profile).
_os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(_pathlib.Path.home() / ".cache" / "numba")
)

from .load import enable_submodules

# Top-level: lazy. Sub-packages (ocdkit.array, ocdkit.io, ocdkit.plot, …) load
# only when accessed, keeping bare ``import ocdkit`` near-instant.
# ``__main__`` is excluded because it's a ``python -m ocdkit`` entry point,
# not a sub-module — eager import would trigger a double-execution warning.
enable_submodules(__name__, exclude=["__main__"], expose=False)
