"""Regression test: ``pkgutil.walk_packages`` discovers every ocdkit
module without silently swallowing import errors.

The check itself lives in :mod:`ocdkit.testing.imports` so downstream
packages (hiprpy, omnipose, …) can opt in with the same one-liner — see
their respective ``tests/test_module_discovery.py``.
"""
import ocdkit
from ocdkit.testing import make_no_silent_discovery_test

test_no_silent_discovery_errors = make_no_silent_discovery_test(ocdkit)
