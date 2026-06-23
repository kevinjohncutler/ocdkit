"""Lazy sub-module discovery and attribute forwarding for packages.

Usage inside any package's ``__init__.py``::

    from ocdkit.load import enable_submodules
    enable_submodules(__name__)

After this call, every ``.py`` file (or sub-package) inside the package
directory becomes accessible by name **and** every public attribute
defined in those sub-modules is accessible directly on the package
without an explicit import.
"""

from types import ModuleType
import ast
import importlib
import os
import pkgutil
import sys


def _scan_export_map(pkg) -> dict:
    """Map ``public_name -> submodule_name`` for a package, by **AST-parsing**
    each submodule file (no import).

    Lets the lazy ``__getattr__`` import only the *one* submodule that defines a
    requested name, instead of import-scanning every submodule to find it (which
    defeats laziness — a single ``from pkg import helper`` would drag in the
    whole package, e.g. wgpu/datashader behind a key-slice helper).

    Names come from a literal ``__all__`` when present, else top-level
    ``def``/``class``/assignment names. Re-exported (imported) names without an
    ``__all__`` aren't visible to the AST and fall through to the runtime scan,
    so correctness is preserved. Submodules are visited in sorted order; the
    first to claim a name wins (deterministic).
    """
    paths = list(getattr(pkg, "__path__", []) or [])
    name_map: dict = {}
    if not paths:
        return name_map
    base = paths[0]
    for info in sorted(pkgutil.iter_modules([base]), key=lambda i: i.name):
        sub = info.name
        modfile = os.path.join(base, sub, "__init__.py") if info.ispkg \
            else os.path.join(base, sub + ".py")
        try:
            with open(modfile, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), modfile)
        except (OSError, SyntaxError):
            continue
        explicit = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                try:
                    explicit = [n for n in ast.literal_eval(node.value) if isinstance(n, str)]
                except (ValueError, TypeError, SyntaxError):
                    explicit = None
        if explicit is not None:
            names = explicit
        else:
            # Include private (``_``-prefixed) top-level names too: the runtime
            # scan fallback resolves them (it doesn't filter on ``_``), so the
            # map must cover them or such lookups regress to scan-import-all.
            names = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            names.append(t.id)
        for n in names:
            name_map.setdefault(n, sub)
    return name_map


def enable_submodules(
    pkg_name: str,
    *,
    include=None,
    exclude=None,
    expose: bool = False,
) -> None:
    """Discover sub-modules and expose them on the package.

    Default mode (``expose=False``) is fully lazy (PEP 562): sub-modules
    are *not* imported at call time, only their names are enumerated.
    Each sub-module loads on first attribute access via ``__getattr__``.
    ``import pkg`` stays near-instant and circular sub-package imports
    don't blow up at package load. Cross-attribute lookups
    (``pkg.some_func`` where ``some_func`` lives in a sub-module) still
    work via the ``__getattr__`` fallback that scans sub-modules on
    miss.

    ``expose=True`` eagerly imports each sub-module and promotes its
    public non-module attributes onto the parent. Use for gateway
    sub-packages whose consumers expect ``from foo import *`` to pull
    in re-exported sub-module attrs (PEP 328 wildcard imports do *not*
    invoke ``__getattr__``, so promoted attrs must already exist on the
    package). Eager mode also makes ``dir(pkg)`` complete without
    triggering loads. Avoid at top-level packages — the transitive
    dependency cost (torch, numba, …) is paid on bare ``import pkg``.

    ``__all__`` lists sub-module names (lazy mode) or sub-module names
    plus promoted attributes (expose mode), suitable for ``from pkg
    import *`` and ``dir(pkg)``.

    Parameters
    ----------
    pkg_name : str
        Fully-qualified package name (typically ``__name__``).
    include : collection of str, optional
        If given, only these sub-module names are exposed. Mutually
        exclusive with *exclude*.
    exclude : collection of str, optional
        If given, these sub-module names are skipped.
    expose : bool, default False
        If False (default), defer all sub-module loading until first
        access. If True, eagerly load and promote sub-module attrs —
        opt in for wildcard-import gateways.
    """
    pkg: ModuleType = sys.modules[pkg_name]
    submods = {info.name for info in pkgutil.iter_modules(pkg.__path__)}

    if include is not None:
        submods &= set(include)
    elif exclude is not None:
        submods -= set(exclude)

    promoted: set = set()
    if expose:
        for sub in sorted(submods):
            try:
                mod = importlib.import_module(f"{pkg_name}.{sub}")
            except ImportError:
                continue
            # Same-name disambiguation: if the submodule exports a
            # public callable named the same as the submodule itself
            # (a common ``foo.py`` → ``def foo()`` pattern), prefer the
            # callable. Otherwise the submodule shadows the function and
            # ``from pkg import foo`` returns the module — almost never
            # what the consumer intended.
            same_named = getattr(mod, sub, None)
            if (same_named is not None
                    and not isinstance(same_named, ModuleType)
                    and callable(same_named)):
                setattr(pkg, sub, same_named)
            else:
                setattr(pkg, sub, mod)
            # Prefer the submodule's ``__all__`` when defined — gives
            # authors a way to keep helper imports (e.g.
            # ``from matplotlib.figure import Figure`` used internally)
            # out of the parent-package namespace.  Falls back to
            # ``dir(mod)`` filtered to public names when ``__all__`` is
            # absent (preserves legacy behavior for modules that haven't
            # opted in).
            if hasattr(mod, "__all__"):
                names = list(mod.__all__)
            else:
                names = [n for n in dir(mod) if not n.startswith("_")]
            for name in names:
                if name.startswith("_"):
                    continue
                if hasattr(pkg, name):
                    continue
                attr = getattr(mod, name, None)
                if attr is None or isinstance(attr, ModuleType):
                    continue
                setattr(pkg, name, attr)
                promoted.add(name)

    pkg.__all__ = sorted(submods | promoted)

    # Built lazily on first attribute miss (keeps bare ``import pkg`` instant);
    # AST-derived public_name -> submodule map, used to import ONLY the owning
    # submodule instead of scan-importing all of them.
    name_map: dict = {}
    map_built = False

    def _getattr(name: str):
        nonlocal map_built
        # Direct sub-module access: load that one only and cache.
        if name in submods:
            mod = importlib.import_module(f"{pkg_name}.{name}")
            # Same-name disambiguation (lazy mode): if the submodule
            # defines a public callable with the same name as itself,
            # prefer the callable. See the ``expose=True`` branch above.
            same_named = getattr(mod, name, None)
            if (same_named is not None
                    and not isinstance(same_named, ModuleType)
                    and callable(same_named)):
                setattr(pkg, name, same_named)
                return same_named
            setattr(pkg, name, mod)
            return mod

        # Fast path: import ONLY the submodule the AST map says defines this
        # name. This is what makes lazy mode actually lazy — without it the scan
        # below imports every submodule (dragging in heavy deps) to find one
        # function. Respects the submodule's runtime ``__all__`` too.
        if not map_built:
            name_map.update(_scan_export_map(pkg))
            map_built = True
        owner = name_map.get(name)
        if owner is not None:
            try:
                mod = importlib.import_module(f"{pkg_name}.{owner}")
            except ImportError:
                mod = None
            if mod is not None and not (hasattr(mod, "__all__") and name not in mod.__all__) \
                    and hasattr(mod, name):
                attr = getattr(mod, name)
                setattr(pkg, name, attr)
                return attr

        # Fallback scan: catches names the AST map can't see — re-exported
        # (imported) names without an ``__all__``, or runtime-defined attrs.
        # Respects each submodule's ``__all__`` when defined (same contract as
        # the eager-promotion pass): a name imported internally but omitted from
        # ``__all__`` will NOT leak through.
        for sub in submods:
            try:
                mod = importlib.import_module(f"{pkg_name}.{sub}")
            except ImportError:
                continue
            if hasattr(mod, "__all__") and name not in mod.__all__:
                continue
            if hasattr(mod, name):
                attr = getattr(mod, name)
                setattr(pkg, name, attr)
                return attr
        raise AttributeError(f"module {pkg_name!r} has no attribute {name!r}")

    pkg.__getattr__ = _getattr


def enable_attr_map(pkg_name: str, attr_map: dict) -> None:
    """Install lazy attribute loading driven by an explicit name-to-module mapping.

    Unlike :func:`enable_submodules` which auto-discovers everything,
    this function exposes only the attributes listed in *attr_map* and
    loads them on first access.

    Parameters
    ----------
    pkg_name : str
        Fully-qualified package name (typically ``__name__``).
    attr_map : dict
        Maps exposed attribute names to their source. Values can be:

        - A string — interpreted as a relative module path; the attribute
          name in that module is assumed to match the key.
          E.g. ``{'Scene': '.scene'}`` loads ``Scene`` from ``.scene``.
        - A tuple ``(module_path, attr_name)`` — for when the exposed name
          differs from the name in the source module.
          E.g. ``{'MyScene': ('.scene', 'Scene')}``.

    Examples
    --------
    ::

        from ocdkit.load import enable_attr_map

        enable_attr_map(__name__, {
            'Collection': '.collection',
            'Scene': '.scene',
            'PseudoScene': '.scene',
        })
    """
    pkg: ModuleType = sys.modules[pkg_name]
    pkg.__all__ = list(attr_map)

    def _getattr(name: str):
        if name not in attr_map:
            raise AttributeError(f"module {pkg_name!r} has no attribute {name!r}")
        entry = attr_map[name]
        if isinstance(entry, tuple):
            module_path, attr_name = entry
        else:
            module_path, attr_name = entry, name
        mod = importlib.import_module(f"{pkg_name}{module_path}")
        value = getattr(mod, attr_name)
        setattr(pkg, name, value)
        return value

    def _dir():
        return sorted(set(pkg.__dict__) | set(attr_map))

    pkg.__getattr__ = _getattr
    pkg.__dir__ = _dir
