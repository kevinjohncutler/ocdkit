"""Plugin discovery and dispatch.

Plugins register themselves either:

* By calling :func:`register_plugin` at import time, or
* By being declared as an ``ocdkit.plugins`` entry point that resolves to a
  :class:`SegmentationPlugin` instance (preferred for distribution).

Example ``pyproject.toml``::

    [project.entry-points."ocdkit.plugins"]
    my_tool = "my_tool.ocdkit_plugin:plugin"
"""

from __future__ import annotations

import threading
from importlib import metadata
from typing import Iterator

from .base import SegmentationPlugin

_ENTRY_POINT_GROUP = "ocdkit.plugins"


class PluginRegistry:
    """Thread-safe registry of segmentation plugins.

    Lookups by plugin name. Discovery via ``importlib.metadata`` entry points
    is lazy — the first call to :meth:`names`, :meth:`get`, :meth:`all`, etc.
    triggers a one-time scan, after which the registry is fully populated.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, SegmentationPlugin] = {}
        self._lock = threading.RLock()
        self._discovered = False

    # -- registration --------------------------------------------------------

    def register(self, plugin: SegmentationPlugin, *, replace: bool = False) -> None:
        """Add ``plugin`` to the registry.

        Raises ``ValueError`` if a plugin with the same name is already
        registered and ``replace`` is False.
        """
        if not isinstance(plugin, SegmentationPlugin):
            raise TypeError(
                f"register() expected SegmentationPlugin, got {type(plugin).__name__}"
            )
        with self._lock:
            if plugin.name in self._plugins and not replace:
                raise ValueError(f"plugin {plugin.name!r} already registered")
            self._plugins[plugin.name] = plugin

    def unregister(self, name: str) -> None:
        """Remove the named plugin (no-op if absent)."""
        with self._lock:
            self._plugins.pop(name, None)

    # -- discovery -----------------------------------------------------------

    def discover(self) -> list[str]:
        """Load any plugins published via entry points.

        Returns the list of plugin names discovered (newly added or
        previously registered). Safe to call repeatedly.
        """
        with self._lock:
            if self._discovered:
                return list(self._plugins.keys())
            try:
                eps = metadata.entry_points(group=_ENTRY_POINT_GROUP)
            except TypeError:
                # Python <3.10 metadata.entry_points returns a dict
                eps = metadata.entry_points().get(_ENTRY_POINT_GROUP, [])
            for ep in eps:
                try:
                    obj = ep.load()
                except Exception as exc:  # pragma: no cover - logged for visibility
                    import sys
                    print(
                        f"[ocdkit.viewer] failed to load plugin entry point {ep.name!r}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if not isinstance(obj, SegmentationPlugin):
                    import sys
                    print(
                        f"[ocdkit.viewer] entry point {ep.name!r} resolved to "
                        f"{type(obj).__name__}, expected SegmentationPlugin",
                        file=sys.stderr,
                    )
                    continue
                if obj.name not in self._plugins:
                    self._plugins[obj.name] = obj
            self._discovered = True
            return list(self._plugins.keys())

    # -- queries -------------------------------------------------------------

    def get(self, name: str) -> SegmentationPlugin:
        """Return the named plugin or raise ``KeyError``."""
        self.discover()
        with self._lock:
            return self._plugins[name]

    def names(self) -> list[str]:
        """Return the sorted list of registered plugin names."""
        self.discover()
        with self._lock:
            return sorted(self._plugins.keys())

    def all(self) -> list[SegmentationPlugin]:
        """Return all registered plugins, sorted by name."""
        self.discover()
        with self._lock:
            return [self._plugins[n] for n in sorted(self._plugins.keys())]

    def __iter__(self) -> Iterator[SegmentationPlugin]:
        return iter(self.all())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        self.discover()
        with self._lock:
            return name in self._plugins

    def __len__(self) -> int:
        self.discover()
        with self._lock:
            return len(self._plugins)

    # -- testing helpers -----------------------------------------------------

    def clear(self) -> None:
        """Drop all registered plugins. Tests only."""
        with self._lock:
            self._plugins.clear()
            self._discovered = False


# Module-level singleton
REGISTRY = PluginRegistry()


def register_plugin(plugin: SegmentationPlugin, *, replace: bool = False) -> None:
    """Register ``plugin`` in the module-level :data:`REGISTRY`."""
    REGISTRY.register(plugin, replace=replace)


def get_plugin(name: str) -> SegmentationPlugin:
    """Return the named plugin from the module-level :data:`REGISTRY`."""
    return REGISTRY.get(name)


def list_plugins() -> list[SegmentationPlugin]:
    """Return all registered plugins, sorted by name."""
    return REGISTRY.all()
