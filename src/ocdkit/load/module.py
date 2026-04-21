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


def enable_submodules(pkg_name: str, *, include=None, exclude=None) -> None:
    """Discover sub-modules and register their public names on the package.

    Imports all sub-modules at call time, then copies every public
    function, class, and non-module attribute into the parent package's
    namespace.  This makes both ``from pkg import func`` and
    ``from pkg import *`` work without hand-maintained export lists.

    Parameters
    ----------
    pkg_name : str
        Fully-qualified package name (typically ``__name__``).
    include : collection of str, optional
        If given, only these submodule names are loaded. Mutually exclusive
        with *exclude*.
    exclude : collection of str, optional
        If given, these submodule names are skipped.
    """
    pkg: ModuleType = sys.modules[pkg_name]
    submods = {info.name for info in pkgutil.iter_modules(pkg.__path__)}

    if include is not None:
        submods &= set(include)
    elif exclude is not None:
        submods -= set(exclude)

    for sub in submods:
        try:
            mod = importlib.import_module(f"{pkg_name}.{sub}")
        except ImportError:
            continue
        setattr(pkg, sub, mod)
        for name in dir(mod):
            if name.startswith("_"):
                continue
            attr = getattr(mod, name)
            if isinstance(attr, ModuleType):
                continue
            if not hasattr(pkg, name):
                setattr(pkg, name, attr)

    pkg.__all__ = [n for n in vars(pkg) if not n.startswith("_")]

    # Fallback __getattr__ for names not eagerly registered (e.g. private names)
    def _getattr(name: str):
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
