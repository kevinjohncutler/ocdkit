"""Tests for ocdkit.slice — slice tuple builders."""

import pytest

from ocdkit.slice import get_slice_tuple


class TestGetSliceTuple:
    def test_scalar_axis(self):
        shape = (5, 6)
        slc = get_slice_tuple(1, 4, shape)
        assert slc == (slice(1, 4, None), slice(None))

        slc_axis = get_slice_tuple(0, 2, shape, axis=1)
        assert slc_axis == (slice(None), slice(0, 2, None))

    def test_iterable(self):
        shape = (10, 12, 14)
        slc = get_slice_tuple([1, 2, 3], [4, 6, 9], shape)
        assert slc == (slice(1, 4, None), slice(2, 6, None), slice(3, 9, None))

        slc_axis = get_slice_tuple([1, 2], [4, 6], shape, axis=[0, 2])
        assert slc_axis == (slice(1, 4, None), slice(None), slice(2, 6, None))

    def test_mismatch_errors(self):
        shape = (5, 6)
        with pytest.raises(ValueError):
            get_slice_tuple([1, 2], [3], shape)

        with pytest.raises(ValueError):
            get_slice_tuple([1, 2], [3, 4], shape, axis=[0])
