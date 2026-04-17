"""Tests for ocdkit.morphology — boundary detection and skeletonization."""

import numpy as np
import pytest
from skimage.segmentation import find_boundaries as skimage_find_boundaries

from ocdkit.morphology import find_boundaries, skeletonize, masks_to_outlines


# ---------------------------------------------------------------------------
# find_boundaries
# ---------------------------------------------------------------------------

class TestFindBoundaries:
    def test_solid_square_boundary(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:8, 2:8] = 1
        bd = find_boundaries(labels)
        # Boundary should mark the perimeter
        assert bd.dtype == np.uint8
        assert bd.sum() > 0
        # Interior pixel should not be boundary
        assert bd[5, 5] == 0
        # Edge pixel should be boundary
        assert bd[2, 2] == 1

    def test_no_foreground(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        bd = find_boundaries(labels)
        assert bd.sum() == 0

    def test_full_foreground(self):
        labels = np.ones((10, 10), dtype=np.int32)
        bd = find_boundaries(labels)
        # No internal boundaries; all pixels touch image edge but
        # find_boundaries inner-mode marks pixels with different neighbor labels
        # When everything is the same label and touches no zero, no boundary
        assert bd.sum() == 0

    def test_two_touching_labels(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:, :5] = 1
        labels[:, 5:] = 2
        bd = find_boundaries(labels)
        # Both sides of the interface should have boundary pixels
        assert bd[:, 4].sum() > 0  # left side of interface
        assert bd[:, 5].sum() > 0  # right side of interface

    def test_matches_skimage_inner(self):
        np.random.seed(0)
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[3:8, 3:8] = 1
        labels[10:15, 10:15] = 2
        labels[2:5, 12:18] = 3

        ours = find_boundaries(labels, connectivity=1)
        theirs = skimage_find_boundaries(labels, mode='inner', connectivity=1)
        np.testing.assert_array_equal(ours.astype(bool), theirs)

    def test_3d(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        bd = find_boundaries(labels)
        assert bd.shape == labels.shape
        assert bd.sum() > 0

    def test_connectivity_2(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[3:7, 3:7] = 1
        labels[7:10, 7:10] = 2  # diagonally adjacent
        bd_c1 = find_boundaries(labels, connectivity=1)
        bd_c2 = find_boundaries(labels, connectivity=2)
        # Connectivity 2 should detect more boundaries
        assert bd_c2.sum() >= bd_c1.sum()


# ---------------------------------------------------------------------------
# skeletonize
# ---------------------------------------------------------------------------

class TestSkeletonize:
    def test_rectangle(self):
        labels = np.zeros((20, 30), dtype=np.int32)
        labels[5:15, 5:25] = 1
        skel = skeletonize(labels)
        assert skel.shape == labels.shape
        assert skel.max() == 1
        # Skeleton should be a subset of the foreground
        assert ((skel > 0) & (labels > 0)).sum() == (skel > 0).sum()

    def test_preserves_label_id(self):
        labels = np.zeros((20, 30), dtype=np.int32)
        labels[5:15, 5:15] = 3
        labels[5:15, 16:25] = 7
        skel = skeletonize(labels)
        unique = set(np.unique(skel)) - {0}
        assert unique == {3, 7}

    def test_small_label_preserved(self):
        # Tiny label should not be lost during thinning (re-attachment logic)
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1
        labels[10, 10] = 2  # single pixel
        skel = skeletonize(labels)
        # Both labels must be present after skeletonization
        assert 1 in skel
        assert 2 in skel

    def test_with_precomputed_dt(self):
        labels = np.zeros((20, 30), dtype=np.int32)
        labels[5:15, 5:25] = 1
        import edt
        dt = edt.edt(labels)
        skel = skeletonize(labels, dt=dt, dt_thresh=1)
        assert skel.shape == labels.shape
        assert skel.max() > 0

    def test_method_lee(self):
        labels = np.zeros((20, 30), dtype=np.int32)
        labels[5:15, 5:25] = 1
        skel = skeletonize(labels, method='lee')
        assert skel.max() == 1


# ---------------------------------------------------------------------------
# masks_to_outlines
# ---------------------------------------------------------------------------

class TestMasksToOutlines:
    def test_2d_legacy_path(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:7, 2:7] = 1
        out = masks_to_outlines(labels)
        assert out.shape == labels.shape
        assert out.dtype == bool or out.dtype == np.bool_
        # Outline pixels only on the border of the square
        assert out[2, 2] and out[6, 6]
        assert not out[4, 4]  # interior

    def test_2d_omni_path(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:7, 2:7] = 1
        out = masks_to_outlines(labels, omni=True)
        assert out.shape == labels.shape
        assert out.sum() > 0

    def test_3d_stacked(self):
        labels_2d = np.zeros((10, 10), dtype=np.int32)
        labels_2d[3:7, 3:7] = 1
        labels_3d = np.stack([labels_2d, labels_2d])
        out = masks_to_outlines(labels_3d, omni=True)
        assert out.shape == labels_3d.shape
        assert out.sum() > 0

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            masks_to_outlines(np.zeros((5,), dtype=np.int32))


# ---------------------------------------------------------------------------
# hysteresis_threshold
# ---------------------------------------------------------------------------

class TestHysteresisThreshold:
    def test_2d(self):
        import torch
        from ocdkit.morphology import hysteresis_threshold
        img = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        img[0, 0, 2, 2] = 1.0
        img[0, 0, 2, 3] = 0.6
        img[0, 0, 2, 4] = 0.2
        mask = hysteresis_threshold(img, low=0.5, high=0.9)
        assert mask.shape == img.shape
        assert mask[0, 0, 2, 2]       # above high
        assert mask[0, 0, 2, 3]       # above low and connected
        assert not mask[0, 0, 2, 4]   # below low

    def test_numpy_input(self):
        from ocdkit.morphology import hysteresis_threshold
        img = np.zeros((1, 1, 5, 5), dtype=np.float32)
        img[0, 0, 2, 2] = 1.0
        mask = hysteresis_threshold(img, low=0.3, high=0.5)
        assert mask[0, 0, 2, 2]
        assert not mask[0, 0, 0, 0]
