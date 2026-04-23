"""Tests for SessionManager LRU + TTL eviction (issue #2)."""

from __future__ import annotations

import time

import pytest

from ocdkit.viewer.session import SessionManager


def test_lru_cap_evicts_least_recent():
    """Once the cap is exceeded the oldest unused session is dropped."""
    mgr = SessionManager(max_count=3, ttl_seconds=0)  # 0 = TTL disabled
    a = mgr.get_or_create(None)
    b = mgr.get_or_create(None)
    c = mgr.get_or_create(None)
    assert mgr.session_count() == 3

    # Touch a so b becomes least-recent.
    mgr.get(a.session_id)

    # Adding a 4th evicts b.
    mgr.get_or_create(None)
    assert mgr.session_count() == 3
    with pytest.raises(KeyError):
        mgr.get(b.session_id)
    # a and c survive.
    assert mgr.get(a.session_id) is a
    assert mgr.get(c.session_id) is c


def test_ttl_evicts_stale_sessions():
    """Sessions idle longer than TTL are evicted on the next access."""
    mgr = SessionManager(max_count=100, ttl_seconds=0.1)
    a = mgr.get_or_create(None)
    assert mgr.session_count() == 1

    time.sleep(0.15)
    # Any get_or_create() triggers an eviction sweep.
    mgr.get_or_create(None)
    # `a` should be gone; only the new session remains.
    assert mgr.session_count() == 1
    with pytest.raises(KeyError):
        mgr.get(a.session_id)


def test_get_touches_last_seen_to_prevent_ttl_eviction():
    """A recently-accessed session should not be evicted by TTL."""
    mgr = SessionManager(max_count=100, ttl_seconds=0.2)
    a = mgr.get_or_create(None)
    time.sleep(0.1)
    mgr.get(a.session_id)  # bump last_seen
    time.sleep(0.15)        # total elapsed = 0.25s, but a was touched at 0.1s
    # Session should still be alive (only 0.15s since touch).
    assert mgr.get(a.session_id) is a


def test_eviction_does_not_lose_active_session_in_get_or_create():
    """get_or_create with a known session_id should never evict the session it returns."""
    mgr = SessionManager(max_count=2, ttl_seconds=0)
    a = mgr.get_or_create(None)
    b = mgr.get_or_create(None)
    # Repeatedly request `a` while creating new sessions; `a` should survive.
    for _ in range(5):
        mgr.get_or_create(a.session_id)
        mgr.get_or_create(None)
    assert mgr.get(a.session_id) is a
