"""Reusable test utilities for ocdkit and downstream packages.

Each helper is a pytest-test factory: pass the root package and assign
the returned function to a ``test_*`` name in a test file. Downstream
packages (omnipose, hiprpy, …) opt into each check with one line.

Available factories:

- :func:`make_module_collision_test` — guards against submodules that
  shadow same-named public callables (``foo.py`` next to ``def foo()``).
  Such pairs make ``from pkg import foo`` ambiguous; this catches the
  pattern at test time. Allowlist intentional cases via ``ALLOWED``.
- :func:`make_import_cycles_test` — asserts there are no top-level
  import cycles within the package. Function-body imports are excluded
  by design (intentional cycle-breakers).
- :func:`make_no_silent_discovery_test` — wraps
  ``pkgutil.walk_packages`` with an ``onerror`` callback so modules
  that fail to import during discovery surface as test failures
  instead of being silently skipped.

Typical opt-in for a downstream repo (one file each in ``tests/``)::

    import mypkg
    from ocdkit.testing import (
        make_module_collision_test,
        make_import_cycles_test,
        make_no_silent_discovery_test,
    )

    test_no_module_callable_collisions = make_module_collision_test(mypkg)
    test_no_import_cycles              = make_import_cycles_test(mypkg)
    test_no_silent_discovery_errors    = make_no_silent_discovery_test(mypkg)

The lower-level ``find_*`` functions are exposed for ad-hoc audits.
"""

from .collisions import (
    find_module_callable_collisions,
    check_no_unannotated_module_collisions,
    make_module_collision_test,
)
from .imports import (
    find_import_cycles,
    make_import_cycles_test,
    make_no_silent_discovery_test,
)

__all__ = [
    "find_module_callable_collisions",
    "check_no_unannotated_module_collisions",
    "make_module_collision_test",
    "find_import_cycles",
    "make_import_cycles_test",
    "make_no_silent_discovery_test",
]
