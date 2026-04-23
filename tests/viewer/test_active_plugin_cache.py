"""Tests for ActivePlugin's cached base64 encodings (issue #7)."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from ocdkit.viewer.segmentation import ACTIVE_PLUGIN


@pytest.fixture(autouse=True)
def _clean_active_plugin():
    ACTIVE_PLUGIN.reset_cache()
    yield
    ACTIVE_PLUGIN.reset_cache()


def test_get_encoded_mask_returns_none_when_empty():
    assert ACTIVE_PLUGIN.get_encoded_mask() is None
    assert ACTIVE_PLUGIN.get_encoded_ncolor() is None


def test_get_encoded_mask_caches_on_first_call(monkeypatch):
    mask = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint32)
    ACTIVE_PLUGIN.store_result(mask, {})

    encoded1 = ACTIVE_PLUGIN.get_encoded_mask()
    assert encoded1 is not None

    # Decode and verify pixel-perfect round-trip.
    raw = base64.b64decode(encoded1)
    decoded = np.frombuffer(raw, dtype=np.uint32).reshape(2, 3)
    np.testing.assert_array_equal(decoded, mask)

    # Second call must return the same string instance (proves caching).
    encoded2 = ACTIVE_PLUGIN.get_encoded_mask()
    assert encoded2 is encoded1


def test_store_result_invalidates_encoded_cache():
    mask_a = np.array([[0, 1]], dtype=np.uint32)
    mask_b = np.array([[5, 5]], dtype=np.uint32)

    ACTIVE_PLUGIN.store_result(mask_a, {})
    enc_a = ACTIVE_PLUGIN.get_encoded_mask()

    ACTIVE_PLUGIN.store_result(mask_b, {})
    enc_b = ACTIVE_PLUGIN.get_encoded_mask()
    assert enc_a != enc_b


def test_get_encoded_ncolor_caches_separately():
    mask = np.array([[1, 2]], dtype=np.uint32)
    ncolor = np.array([[1, 2]], dtype=np.uint32)
    ACTIVE_PLUGIN.store_result(mask, {}, ncolor=ncolor)

    n1 = ACTIVE_PLUGIN.get_encoded_ncolor()
    n2 = ACTIVE_PLUGIN.get_encoded_ncolor()
    assert n1 is n2
    assert n1 is not None


def test_reset_cache_clears_encoded_strings():
    mask = np.array([[0, 1]], dtype=np.uint32)
    ACTIVE_PLUGIN.store_result(mask, {})
    assert ACTIVE_PLUGIN.get_encoded_mask() is not None
    ACTIVE_PLUGIN.reset_cache()
    assert ACTIVE_PLUGIN.get_encoded_mask() is None
