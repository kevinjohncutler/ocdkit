"""Import-graph regression tests for ocdkit and downstream packages.

Two pytest-test factories that any package using ``ocdkit.load.enable_submodules``
can opt into with one line:

    from ocdkit.testing import (
        make_import_cycles_test,
        make_no_silent_discovery_test,
    )
    import mypkg

    test_no_import_cycles = make_import_cycles_test(mypkg)
    test_no_silent_discovery_errors = make_no_silent_discovery_test(mypkg)

The cycle detector walks each module's AST for **top-level** import
statements only — function-body / method-body imports are intentional
cycle-breakers and are excluded. The discovery test catches modules
that ``pkgutil.walk_packages`` silently skips when they fail to import.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Iterable


def _module_top_level_imports(file_path: Path, package_name: str) -> set[str]:
    """Return fully-qualified imports from *file_path* at module scope only."""
    try:
        tree = ast.parse(file_path.read_text())
    except (SyntaxError, OSError):
        return set()

    pkg_parts = package_name.split('.')
    # For a regular module ``pkg.subpkg.mod`` the relative-import anchor is
    # ``pkg.subpkg``; for ``pkg/__init__.py`` the anchor is the package itself
    # (``pkg``). Drop the module stem only when it isn't an __init__.
    is_pkg_init = file_path.name == '__init__.py'
    base_parts = pkg_parts if is_pkg_init else pkg_parts[:-1]
    out: set[str] = set()

    for node in tree.body:  # body only — top-level statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                drop = node.level - 1
                anchor = base_parts[: len(base_parts) - drop] if drop else base_parts
                if not anchor:
                    continue
                resolved_pkg = '.'.join(anchor)
                if node.module:
                    # ``from .X import Y, Z`` — single edge to ``pkg.X``
                    out.add(f"{resolved_pkg}.{node.module}")
                else:
                    # ``from . import A, B, C`` — one edge per name
                    for alias in node.names:
                        out.add(f"{resolved_pkg}.{alias.name}")
            elif node.module:
                out.add(node.module)
                # ``from pkg.sub import A, B`` — A/B may be submodules too,
                # but the test resolves edges by walking up dotted prefixes
                # in find_import_cycles, so adding only the explicit module
                # is sufficient here.
    return out


def _collect_package_modules(package: ModuleType) -> dict[str, Path]:
    """Map ``pkg.sub.mod`` → file path for every module in *package*."""
    out: dict[str, Path] = {}
    pkg_path = Path(package.__file__).parent
    out[package.__name__] = pkg_path / "__init__.py"

    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            continue
        if spec is None or spec.origin is None or not spec.origin.endswith('.py'):
            continue
        out[name] = Path(spec.origin)
    return out


def find_import_cycles(package: ModuleType) -> list[list[str]]:
    """Build the top-level import graph for *package* and return all cycles.

    Each cycle is returned as a list of dotted module names ending where
    it began. Imports outside *package* are ignored. Lazy imports
    (inside functions or methods) are excluded by design.
    """
    modules = _collect_package_modules(package)
    pkg_name = package.__name__
    pkg_prefix = pkg_name + "."

    # Build adjacency: only edges to modules inside the package
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for name, path in modules.items():
        for imp in _module_top_level_imports(path, name):
            # Prefer exact match (most specific) over package-prefix match.
            if imp in modules:
                target = imp
            else:
                # ``from pkg.subpkg import leaf`` resolves to ``pkg.subpkg``.
                # Walk up dotted prefixes to find the closest enclosing module.
                target = None
                parts = imp.split('.')
                while parts:
                    candidate = '.'.join(parts)
                    if candidate in modules:
                        target = candidate
                        break
                    parts.pop()
            if target and target != name:
                graph[name].add(target)

    # Tarjan-style DFS for all cycles via back-edge detection.
    cycles: list[list[str]] = []
    color: dict[str, int] = {}  # 0=unseen, 1=in stack, 2=done
    stack: list[str] = []

    def visit(node: str) -> None:
        color[node] = 1
        stack.append(node)
        for nbr in sorted(graph.get(node, ())):
            c = color.get(nbr, 0)
            if c == 1:
                idx = stack.index(nbr)
                cycles.append(stack[idx:] + [nbr])
            elif c == 0:
                visit(nbr)
        stack.pop()
        color[node] = 2

    for n in sorted(graph):
        if color.get(n, 0) == 0:
            visit(n)

    # Deduplicate by canonical rotation
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for cyc in cycles:
        body = tuple(cyc[:-1])  # drop closing repeat
        rot = min(body[i:] + body[:i] for i in range(len(body)))
        if rot not in seen:
            seen.add(rot)
            unique.append(list(rot) + [rot[0]])
    return unique


def make_import_cycles_test(package: ModuleType, *, allow: Iterable[tuple[str, ...]] = ()):
    """Return a pytest function asserting *package* has no top-level import cycles.

    Parameters
    ----------
    package
        The imported root package (e.g. ``import omnipose; omnipose``).
    allow
        Iterable of cycles (as tuples of dotted module names, in canonical
        rotation, NOT including the trailing repeat) to allow. Use sparingly.
    """
    allowed = {tuple(t) for t in allow}

    def test_no_import_cycles() -> None:
        cycles = find_import_cycles(package)
        offenders = [c for c in cycles if tuple(c[:-1]) not in allowed]
        if offenders:
            msg = "Top-level import cycles detected:\n"
            for c in offenders:
                msg += f"  {' -> '.join(c)}\n"
            raise AssertionError(msg)

    test_no_import_cycles.__doc__ = (
        f"Assert that ``{package.__name__}`` has no top-level import cycles."
    )
    return test_no_import_cycles


def make_no_silent_discovery_test(package: ModuleType):
    """Return a pytest function asserting ``pkgutil.walk_packages`` reports no errors.

    ``walk_packages`` silently skips modules that raise during import discovery
    by default. This test re-runs the walk with an ``onerror`` callback so
    any silently-skipped failure surfaces as a test failure.
    """
    def test_no_silent_discovery_errors() -> None:
        errors: list[tuple[str, BaseException | None]] = []

        def _on_error(name: str) -> None:
            errors.append((name, sys.exc_info()[1]))

        # Drain the iterator
        for _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + ".", onerror=_on_error
        ):
            pass

        if errors:
            lines = [f"  {name}: {exc!r}" for name, exc in errors]
            raise AssertionError(
                "Module discovery silently failed for:\n" + "\n".join(lines)
            )

    test_no_silent_discovery_errors.__doc__ = (
        f"Assert that ``pkgutil.walk_packages`` discovers every module in "
        f"``{package.__name__}`` without errors."
    )
    return test_no_silent_discovery_errors
