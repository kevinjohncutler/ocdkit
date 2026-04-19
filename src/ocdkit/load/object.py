"""Attach functions from submodules onto classes or module namespaces.

Provides helpers for the pattern where a class (or dict) collects public
functions from a set of submodules at import time, supporting decorator
tags for properties and classmethods.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path


def attach_helpers(cls, modules_or_packages, *, skip_subpackages=True, exclude_modules=None):
    """Import helper modules/packages and copy their public helpers onto *cls*.

    Parameters
    ----------
    cls : type
        Class that will receive the helpers.
    modules_or_packages : str or sequence of str
        Each entry can be either a fully-qualified module name
        (e.g. ``"mypackage.helpers.chans"``) or a package name
        (e.g. ``"mypackage.helpers"``).  When a package is given,
        every top-level ``.py`` file in that directory is treated as
        a helper module.
    skip_subpackages : bool, default True
        Ignore nested subpackages when scanning a package directory.
    exclude_modules : container of str or None
        Module basenames to skip when scanning a package directory.
    """
    if exclude_modules is None:
        exclude_modules = set()
    elif not isinstance(exclude_modules, set):
        exclude_modules = set(exclude_modules)

    if isinstance(modules_or_packages, str):
        modules_or_packages = [modules_or_packages]

    for name in modules_or_packages:
        mod = importlib.import_module(name)
        if getattr(mod, "__file__", None) is not None:
            pkg_dir = Path(mod.__file__).parent
            pkg_name = mod.__package__ or name.split(".")[0]
        elif hasattr(mod, "__path__"):
            pkg_dir = Path(next(iter(mod.__path__)))
            pkg_name = name
        else:
            continue

        load_submodules(
            cls,
            package_dir=str(pkg_dir),
            package_name=pkg_name,
            exclude_modules=exclude_modules,
            skip_subpackages=skip_subpackages,
        )


def attach_function_to_object(cls, name, func):
    """Attach *func* to *cls*, honoring decorator tags for properties/classmethods."""
    if isinstance(cls, dict):
        cls[name] = func
    elif getattr(func, "__is_property__", False):
        setattr(cls, name, property(func))
    elif getattr(func, "__is_classmethod__", False):
        setattr(cls, name, classmethod(func))
    else:
        setattr(cls, name, func)


def load_submodules(
    cls,
    package_dir,
    package_name,
    exclude_modules=None,
    skip_subpackages=False,
    attach_vars=False,
):
    """Load all Python submodules in *package_dir* and attach their functions to *cls*.

    Parameters
    ----------
    cls : type or dict
        Target object that receives the functions.
    package_dir : str
        Filesystem path to the package directory.
    package_name : str
        Fully-qualified package name (for ``importlib``).
    exclude_modules : container of str or None
        Module basenames to skip.
    skip_subpackages : bool, default False
        If True, ignore nested sub-packages.
    attach_vars : bool, default False
        If True, also attach non-callable public variables.
    """
    if exclude_modules is None:
        exclude_modules = set()
    else:
        exclude_modules = set(exclude_modules)

    for module_info in pkgutil.iter_modules([package_dir]):
        if module_info.ispkg and skip_subpackages:
            continue

        mod_name = module_info.name
        if mod_name in exclude_modules or mod_name == "__init__":
            continue

        full_name = f"{package_name}.{mod_name}"
        mod = importlib.import_module(full_name)

        for attr_name, obj in inspect.getmembers(mod, inspect.isfunction):
            orig = inspect.unwrap(obj)
            if getattr(orig, "__module__", None) == mod.__name__:
                attach_function_to_object(cls, attr_name, obj)

        if attach_vars:
            for name, val in vars(mod).items():
                if not name.startswith("_") and not callable(val):
                    if isinstance(cls, dict):
                        cls[name] = val
                    else:
                        setattr(cls, name, val)
