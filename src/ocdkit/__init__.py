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

# Numba threading layer: leave numba's autoselection alone in normal use.
# Earlier benchmarks on a Threadripper PRO 3995WX appeared to show that
# ``omp`` (libgomp) had a 14–43 ms per-parallel-region launch cost — but
# that turned out to be load contention from ~75 cores' worth of orphan
# multiprocessing workers, not a libgomp pathology. With a clean machine,
# omp on the same threadripper completes ncolor.label on a 109k-px image
# in 1.07 ms vs workqueue's 5.86 ms (5× faster). Mac defaults to workqueue
# because that's the only layer numba builds there. Windows defaults to omp
# (5 µs prange launch). The earlier ``workqueue`` override is removed; users
# who hit a real omp issue can still ``NUMBA_THREADING_LAYER=workqueue ...``
# explicitly. (Note: ``OMP_PROC_BIND=spread OMP_PLACES=cores`` further
# improves omp by ~15% on threadripper but is not set here because it can
# hurt non-OMP code paths sharing the process — leave it to invocation-time.)

from .load import enable_submodules

# Top-level: lazy. Sub-packages (ocdkit.array, ocdkit.io, ocdkit.plot, …) load
# only when accessed, keeping bare ``import ocdkit`` near-instant.
# ``__main__`` is excluded because it's a ``python -m ocdkit`` entry point,
# not a sub-module — eager import would trigger a double-execution warning.
enable_submodules(__name__, exclude=["__main__"], expose=False)
