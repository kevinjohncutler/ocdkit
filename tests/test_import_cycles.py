"""Regression test: no top-level import cycles in ocdkit.

The check itself lives in :mod:`ocdkit.testing.imports` so downstream
packages (hiprpy, omnipose, …) can opt in with the same one-liner — see
their respective ``tests/test_import_cycles.py``.
"""
import ocdkit
from ocdkit.testing import make_import_cycles_test

test_no_import_cycles = make_import_cycles_test(ocdkit)
