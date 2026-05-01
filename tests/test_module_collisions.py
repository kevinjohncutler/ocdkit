"""Regression test: no submodule shadows a same-named public callable.

The check itself lives in :mod:`ocdkit.testing.collisions` so downstream
packages (hiprpy, omnipose, …) can opt in with the same one-liner — see
their respective ``tests/test_module_collisions.py``.
"""
import ocdkit
from ocdkit.testing import make_module_collision_test

test_no_module_callable_collisions = make_module_collision_test(ocdkit)
