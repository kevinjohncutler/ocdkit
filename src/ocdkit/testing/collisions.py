"""AST-based detector for submodule/callable name collisions.

Background
----------
A package with a submodule ``foo.py`` and a public ``foo()`` function inside
creates an ambiguity: ``from pkg import foo`` may return either the module or
the function depending on how the package is loaded. The figure-export bug in
``ocdkit.plot.figure`` (May 2026) was caused exactly by this pattern.

Resolution options when a new collision appears:

* **Rename** the submodule with a leading underscore (``foo.py`` → ``_foo.py``)
  and update internal callers. This eliminates the ambiguity at the file
  system level and signals "implementation detail not for direct
  ``from pkg.foo import X``".
* **Allowlist** by adding the entry to ``ALLOWED_COLLISIONS`` in the calling
  test, with a one-line rationale. Use this when the submodule is intentionally
  public (e.g. a console-script entry point or an explicit re-export pattern).

Usage
-----
::

    from ocdkit.testing.collisions import check_no_unannotated_module_collisions
    import mypkg

    ALLOWED_COLLISIONS: set[tuple[str, str]] = {
        # ("mypkg.subpkg", "name"),  # one-line reason
    }

    def test_no_unannotated_module_callable_collisions():
        check_no_unannotated_module_collisions(mypkg, ALLOWED_COLLISIONS)
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType


def _parent_init_reexports_leaf(pkg_root: Path, parent_parts: list[str], leaf: str) -> bool:
    """Return True if the parent package's ``__init__.py`` explicitly re-exports
    a same-named symbol from the ``leaf`` submodule.

    The pattern looked for is the standard NumPy-style mitigation::

        from .leaf import leaf, ...   # promotes leaf-the-function over
                                       # leaf-the-submodule on the parent

    When this is in place, ``from pkg import leaf`` returns the function,
    and the structural file/function name collision is consciously handled.
    """
    init_path = pkg_root.joinpath(*parent_parts) / "__init__.py" if parent_parts else pkg_root / "__init__.py"
    if not init_path.is_file():
        return False
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        # Relative ``from .leaf import …`` has node.module == "leaf";
        # absolute ``from pkg.parent.leaf import …`` ends with ".leaf".
        mod = node.module or ""
        if mod != leaf and not mod.endswith("." + leaf):
            continue
        for alias in node.names:
            if alias.name == leaf:
                # ``from .leaf import leaf`` (or aliased to the same name)
                effective = alias.asname or alias.name
                if effective == leaf:
                    return True
    return False


def find_module_callable_collisions(pkg: ModuleType) -> set[tuple[str, str]]:
    """Walk *pkg*'s source tree for submodules whose name shadows a public callable.

    Returns the set of ``(parent_pkg_dotted, leaf_name)`` tuples — one per
    ``<parent>/leaf.py`` file that defines a top-level ``def leaf`` or
    ``class leaf`` (with ``leaf`` not starting with an underscore).

    Two patterns are recognized as already-mitigated and skipped:

    1. **Mixin / method functions** — ``def leaf(self, ...)`` is intended
       to be attached to a class via ``load_submodules`` and never
       imported via ``from pkg import leaf`` (it would be useless without
       its bound ``self``). The structural collision exists but is never
       observable.

    2. **Explicit NumPy-style re-export** — when the parent's
       ``__init__.py`` contains ``from .leaf import leaf, ...``, the
       function correctly shadows the submodule on the parent's namespace
       and ``from pkg import leaf`` returns the function. Standard
       Python-stdlib + NumPy + sklearn pattern for handling these
       collisions.

    Anything else is a real, unmitigated collision and gets reported.

    The scan is purely static (AST), so it is fast and side-effect-free —
    no submodule imports are triggered.
    """
    pkg_root = Path(pkg.__file__).parent
    pkg_name = pkg.__name__
    found: set[tuple[str, str]] = set()
    for py in pkg_root.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        if py.name.startswith("._"):  # macOS resource forks
            continue
        leaf = py.stem
        if leaf.startswith("_"):
            continue  # private modules can't be shadowed via ``from pkg import``
        rel = py.relative_to(pkg_root).with_suffix("")
        parts = list(rel.parts)
        parent_parts = parts[:-1]
        parent_dotted = ".".join([pkg_name] + parent_parts)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in tree.body:
            collision = False
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == leaf:
                # Skip mixin/method patterns: ``def leaf(self, ...)`` is
                # meant to be attached to a class, not imported as a free
                # function from the package namespace.
                args = node.args.args
                if args and args[0].arg == "self":
                    break
                collision = True
            elif isinstance(node, ast.ClassDef) and node.name == leaf:
                collision = True
            if collision:
                # Skip if mitigated by an explicit re-export in the parent's __init__.py.
                if not _parent_init_reexports_leaf(pkg_root, parent_parts, leaf):
                    found.add((parent_dotted, leaf))
                break
    return found


def check_no_unannotated_module_collisions(
    pkg: ModuleType,
    allowed: set[tuple[str, str]],
) -> None:
    """Assert *pkg* has no module/callable collisions outside *allowed*.

    Fails on either:

    * **New collisions** — a submodule that shadows a same-named public
      callable was added without being audited. Rename to ``_<name>.py``
      or add to *allowed* with a rationale.
    * **Stale allowlist entries** — a previously-allowed collision no
      longer exists (file renamed/removed). Drop the entry from *allowed*
      so the allowlist stays a tight, useful audit trail.

    Designed to be called from a single ``test_*`` function inside any
    package's test suite.
    """
    found = find_module_callable_collisions(pkg)
    new_collisions = found - allowed
    stale_allowed = allowed - found

    msgs = []
    if new_collisions:
        bullets = "\n".join(f"  ({p!r}, {n!r})" for p, n in sorted(new_collisions))
        msgs.append(
            "New module/callable name collisions detected:\n"
            f"{bullets}\n"
            "Either rename the submodule with a leading underscore "
            "(foo.py → _foo.py) or add the (parent, name) tuple to "
            "ALLOWED_COLLISIONS in the calling test with a one-line rationale."
        )
    if stale_allowed:
        bullets = "\n".join(f"  ({p!r}, {n!r})" for p, n in sorted(stale_allowed))
        msgs.append(
            "ALLOWED_COLLISIONS contains entries that no longer apply "
            "(file renamed or symbol removed):\n"
            f"{bullets}\n"
            "Remove these tuples from ALLOWED_COLLISIONS."
        )
    assert not msgs, "\n\n".join(msgs)


def make_module_collision_test(pkg, allowed=frozenset()):
    """Return a pytest-discoverable test function for *pkg*.

    Use this in any package's test suite to add the collision check with
    minimal boilerplate::

        # tests/test_module_collisions.py
        import mypkg
        from ocdkit.testing import make_module_collision_test

        test_no_module_callable_collisions = make_module_collision_test(mypkg)

    Pytest discovers the module-level ``test_*`` name automatically.

    *allowed* is for the rare case where a collision is intentionally
    preserved (e.g. backwards-compat with an externally-published API);
    in most projects, the right move is to rename the colliding submodule
    with a leading underscore and leave *allowed* empty.
    """
    def test_no_module_callable_collisions():
        check_no_unannotated_module_collisions(pkg, set(allowed))
    test_no_module_callable_collisions.__doc__ = (
        f"ocdkit.testing-provided: assert no submodule shadows a "
        f"same-named public callable in {pkg.__name__}."
    )
    return test_no_module_callable_collisions
