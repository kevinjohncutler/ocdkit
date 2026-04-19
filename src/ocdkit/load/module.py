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


def enable_submodules(pkg_name: str) -> None:
    """Attach ``__all__``, ``__getattr__``, and ``__dir__`` to *pkg_name*.

    Discovers sub-modules via :func:`pkgutil.iter_modules`, then installs
    a module-level ``__getattr__`` that lazily imports them on first access.
    Attribute lookups that don't match a sub-module name are forwarded into
    each sub-module in turn, so ``pkg.some_function`` works even when
    ``some_function`` lives in ``pkg/_internal.py``.
    """
    pkg: ModuleType = sys.modules[pkg_name]
    submods = {info.name for info in pkgutil.iter_modules(pkg.__path__)}
    pkg.__all__ = list(submods)

    def _getattr(name: str):
        if name in submods:
            mod = importlib.import_module(f"{pkg_name}.{name}")
            setattr(pkg, name, mod)
            return mod
        for sub in submods:
            mod = importlib.import_module(f"{pkg_name}.{sub}")
            if hasattr(mod, name):
                attr = getattr(mod, name)
                setattr(pkg, name, attr)
                return attr
        raise AttributeError(f"module {pkg_name!r} has no attribute {name!r}")

    def _dir():
        items = set(pkg.__dict__) | submods
        for sub in submods:
            try:
                mod = importlib.import_module(f"{pkg_name}.{sub}")
            except ImportError:
                continue
            items.update(n for n in dir(mod) if not n.startswith("_"))
        return sorted(items)

    pkg.__getattr__ = _getattr
    pkg.__dir__ = _dir
