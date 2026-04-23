"""Reusable FastAPI dependencies (Depends factories).

Routes import these instead of repeating session-lookup boilerplate. Tests
can override any factory via ``app.dependency_overrides[get_session_state] = ...``
to inject mocks.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Request

from .exceptions import (
    NoActivePlugin,
    PluginCapabilityMissing,
    PluginNotRegistered,
    UnknownSession,
)
from .plugins.base import SegmentationPlugin
from .plugins.registry import REGISTRY
from .schemas import SessionPayload
from .segmentation import ACTIVE_PLUGIN
from .session import SESSION_COOKIE_NAME, SESSION_MANAGER, SessionState


# ----- session lookup -----------------------------------------------------


def get_session_state(payload: SessionPayload) -> SessionState:
    """Resolve a SessionState from a request body's ``sessionId``."""
    try:
        return SESSION_MANAGER.get(payload.sessionId)
    except KeyError as exc:
        raise UnknownSession() from exc


def get_or_create_session(request: Request) -> SessionState:
    """Resolve (or mint) a SessionState from the request cookie. Used by GET /."""
    return SESSION_MANAGER.get_or_create(request.cookies.get(SESSION_COOKIE_NAME))


# ----- plugin lookup ------------------------------------------------------


def get_plugin_by_name(name: str) -> SegmentationPlugin:
    """Fetch a plugin from the registry, mapping KeyError → 404."""
    try:
        return REGISTRY.get(name)
    except KeyError as exc:
        raise PluginNotRegistered(detail=name) from exc


def get_active_plugin() -> SegmentationPlugin:
    """Resolve the currently active plugin or raise 400."""
    plugin = ACTIVE_PLUGIN.current()
    if plugin is None:
        raise NoActivePlugin()
    return plugin


def require_plugin_capability(*hooks: str):
    """Build a dependency that requires the active plugin to expose given hooks."""

    def _check(plugin: SegmentationPlugin = Depends(get_active_plugin)) -> SegmentationPlugin:
        for hook in hooks:
            if getattr(plugin, hook, None) is None:
                raise PluginCapabilityMissing(
                    detail=f"plugin {plugin.name!r} does not implement {hook}"
                )
        return plugin

    return _check
