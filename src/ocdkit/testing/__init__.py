"""Reusable test utilities for ocdkit and downstream packages.

Currently exposes :func:`check_no_unannotated_module_collisions` so any
package that uses ``ocdkit.load.enable_submodules`` (or just shares the
"submodule named like the function inside it" anti-pattern risk) can add
a one-line regression test:

    from ocdkit.testing.collisions import check_no_unannotated_module_collisions
    import mypkg

    ALLOWED_COLLISIONS = {("mypkg.foo", "bar")}  # parent_dotted, leaf_name

    def test_no_unannotated_module_callable_collisions():
        check_no_unannotated_module_collisions(mypkg, ALLOWED_COLLISIONS)
"""

from .collisions import (
    find_module_callable_collisions,
    check_no_unannotated_module_collisions,
    make_module_collision_test,
)

__all__ = [
    "find_module_callable_collisions",
    "check_no_unannotated_module_collisions",
    "make_module_collision_test",
]
