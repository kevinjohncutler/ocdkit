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
import importlib
import pkgutil
import sys


def enable_submodules(
    pkg_name: str,
    *,
    include=None,
    exclude=None,
    expose: bool = True,
) -> None:
    """Discover sub-modules and expose them on the package.

    Default mode (``expose=True``) eagerly imports each sub-module and
    promotes its public non-module attributes onto the parent. Suitable
    for gateway sub-packages that re-export their sub-modules' functions
    (``from foo import some_func`` and ``from foo import *`` semantics).
    This matches the historical behavior.

    ``expose=False`` is fully lazy (PEP 562): sub-modules are *not*
    imported at call time, only their names are enumerated. Each
    sub-module loads on first attribute access via ``__getattr__``.
    ``import pkg`` becomes near-instant. Use for *top-level* packages
    (``ocdkit``, ``omnipose``) where the cost of eagerly loading every
    sub-package's transitive dependencies (torch, numba, …) on bare
    ``import pkg`` is unacceptable.

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
    expose : bool, default True
        If True (default), eagerly load and promote sub-module attrs.
        If False, defer all sub-module loading until first access.
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
            setattr(pkg, sub, mod)
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                if hasattr(pkg, name):
                    continue
                attr = getattr(mod, name)
                if isinstance(attr, ModuleType):
                    continue
                setattr(pkg, name, attr)
                promoted.add(name)

    pkg.__all__ = sorted(submods | promoted)

    def _getattr(name: str):
        # Direct sub-module access: load that one only and cache.
        if name in submods:
            mod = importlib.import_module(f"{pkg_name}.{name}")
            setattr(pkg, name, mod)
            return mod
        # Fallback: scan sub-modules for the attribute. Catches private
        # names and module-typed attrs (e.g. stdlib ``platform``) that
        # the eager pass skips. Sub-modules are already loaded under
        # ``expose=True``, so this scan is just dict lookups.
        for sub in submods:
            try:
                mod = importlib.import_module(f"{pkg_name}.{sub}")
            except ImportError:
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
