"""Tests for ocdkit.spatial — kernel setup, neighbors, affinity, contours."""

import numpy as np
import pytest

from ocdkit.spatial import (
    kernel_setup,
    get_neighbors,
    get_neigh_inds,
    masks_to_affinity,
    boundary_to_masks,
    get_contour,
    get_link_matrix,
    nd_grid_hypercube_labels,
    make_label_matrix,
)


# ---------------------------------------------------------------------------
# kernel_setup
# ---------------------------------------------------------------------------

class TestKernelSetup:
    def test_2d_shape(self):
        k = kernel_setup(2)
        assert k.steps.shape == (9, 2)  # 3^2
        assert k.idx == 4  # center index

    def test_3d_shape(self):
        k = kernel_setup(3)
        assert k.steps.shape == (27, 3)
        assert k.idx == 13

    def test_2d_groups(self):
        k = kernel_setup(2)
        # Groups: center (1), cardinal (4), ordinal (4)
        assert len(k.inds) == 3
        assert len(k.inds[0]) == 1  # center
        assert len(k.inds[1]) == 4  # cardinal
        assert len(k.inds[2]) == 4  # ordinal

    def test_3d_groups(self):
        k = kernel_setup(3)
        # center, face (6), edge (12), vertex (8)
        assert len(k.inds) == 4
        assert len(k.inds[0]) == 1
        assert len(k.inds[1]) == 6
        assert len(k.inds[2]) == 12
        assert len(k.inds[3]) == 8

    def test_fact(self):
        k = kernel_setup(2)
        # fact is per-group: [0, 1, sqrt(2)]
        np.testing.assert_allclose(k.fact, [0, 1, np.sqrt(2)])

    def test_unpacking(self):
        # Should support tuple-like unpacking via Result
        steps, inds, idx, fact, sign = kernel_setup(2)
        assert steps.shape == (9, 2)
        assert idx == 4

    def test_center_step_is_zero(self):
        k = kernel_setup(2)
        assert np.all(k.steps[k.idx] == 0)


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    def test_shape(self):
        k = kernel_setup(2)
        coords = (np.array([5, 5, 5]), np.array([3, 4, 5]))
        neighbors = get_neighbors(coords, k.steps, 2, (10, 10))
        assert neighbors.shape == (2, 9, 3)  # (dim, nsteps, npix)

    def test_center_returns_self(self):
        k = kernel_setup(2)
        coords = (np.array([5]), np.array([5]))
        neighbors = get_neighbors(coords, k.steps, 2, (10, 10))
        # Center step should give back the original coordinate
        assert neighbors[0, k.idx, 0] == 5
        assert neighbors[1, k.idx, 0] == 5

    def test_boundary_clamps(self):
        k = kernel_setup(2)
        coords = (np.array([0]), np.array([0]))  # corner
        neighbors = get_neighbors(coords, k.steps, 2, (10, 10))
        # All neighbor coords must be in [0, 9]
        assert neighbors.min() >= 0
        assert neighbors.max() <= 9


# ---------------------------------------------------------------------------
# get_neigh_inds
# ---------------------------------------------------------------------------

class TestGetNeighInds:
    def test_basic(self):
        k = kernel_setup(2)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[5, 5] = 1
        coords = np.nonzero(labels)
        neighbors = get_neighbors(coords, k.steps, 2, labels.shape)
        indexes, neigh_inds, ind_matrix = get_neigh_inds(neighbors, coords, labels.shape)

        assert indexes.shape == (1,)
        assert neigh_inds.shape == (9, 1)
        assert ind_matrix.shape == labels.shape
        assert ind_matrix[5, 5] == 0  # foreground gets index 0
        assert ind_matrix[0, 0] == -1  # background


# ---------------------------------------------------------------------------
# masks_to_affinity
# ---------------------------------------------------------------------------

class TestMasksToAffinity:
    def test_solid_block(self):
        masks = np.zeros((10, 10), dtype=np.int32)
        masks[3:7, 3:7] = 1
        coords = np.nonzero(masks)
        k = kernel_setup(2)
        aff = masks_to_affinity(masks, coords, k.steps, k.inds, k.idx, k.fact, k.sign, 2)
        # (nsteps, npix)
        assert aff.shape[0] == 9
        assert aff.shape[1] == coords[0].size
        # Center self-connections should be False
        assert not aff[k.idx].any()

    def test_two_separate_labels(self):
        masks = np.zeros((10, 10), dtype=np.int32)
        masks[2:4, 2:4] = 1
        masks[6:8, 6:8] = 2
        coords = np.nonzero(masks)
        k = kernel_setup(2)
        aff = masks_to_affinity(masks, coords, k.steps, k.inds, k.idx, k.fact, k.sign, 2)
        assert aff.shape[0] == 9
        # Some affinities should be True (within each block)
        assert aff.any()


# ---------------------------------------------------------------------------
# boundary_to_masks
# ---------------------------------------------------------------------------

class TestBoundaryToMasks:
    def test_basic(self):
        # Build a simple 0-1-2 boundary map: inside=1, boundary=2, outside=0
        bmap = np.zeros((20, 20), dtype=np.int32)
        bmap[5:15, 5:15] = 2  # boundary
        bmap[7:13, 7:13] = 1  # inside
        masks, bounds, inner = boundary_to_masks(bmap)
        assert masks.shape == bmap.shape
        assert masks.max() > 0


# ---------------------------------------------------------------------------
# get_contour
# ---------------------------------------------------------------------------

class TestGetContour:
    def test_single_object(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1
        coords = np.nonzero(labels)
        k = kernel_setup(2)
        neighbors = get_neighbors(coords, k.steps, 2, labels.shape)
        aff = masks_to_affinity(labels, coords, k.steps, k.inds, k.idx, k.fact, k.sign, 2,
                                neighbors=neighbors)
        cm, contours, unique_L = get_contour(labels, aff, coords=coords, neighbors=neighbors)
        assert cm.shape == labels.shape
        assert len(contours) >= 1
        assert 1 in unique_L

    def test_two_objects(self):
        labels = np.zeros((20, 30), dtype=np.int32)
        labels[5:15, 5:13] = 1
        labels[5:15, 17:25] = 2
        coords = np.nonzero(labels)
        k = kernel_setup(2)
        neighbors = get_neighbors(coords, k.steps, 2, labels.shape)
        aff = masks_to_affinity(labels, coords, k.steps, k.inds, k.idx, k.fact, k.sign, 2,
                                neighbors=neighbors)
        cm, contours, unique_L = get_contour(labels, aff, coords=coords, neighbors=neighbors)
        assert len(contours) == 2
        assert set(unique_L) == {1, 2}


# ---------------------------------------------------------------------------
# get_link_matrix
# ---------------------------------------------------------------------------

class TestGetLinkMatrix:
    def test_empty_links(self):
        piece_masks = np.zeros((9, 5), dtype=np.int64)
        is_link = np.zeros((9, 5), dtype=np.bool_)
        result = get_link_matrix([], piece_masks, np.arange(9), 4, is_link)
        assert not result.any()

    def test_with_links(self):
        # Build a piece_masks where pixel 0 has center label 1 and neighbor label 2
        piece_masks = np.zeros((9, 1), dtype=np.int64)
        piece_masks[4, 0] = 1   # center
        piece_masks[0, 0] = 2   # one neighbor
        is_link = np.zeros((9, 1), dtype=np.bool_)
        result = get_link_matrix({(1, 2)}, piece_masks, np.arange(9), 4, is_link)
        assert result[0, 0]  # link should be marked


# ---------------------------------------------------------------------------
# nd_grid_hypercube_labels
# ---------------------------------------------------------------------------

class TestNDGridHypercubeLabels:
    def test_basic_centered(self):
        labels = nd_grid_hypercube_labels((6, 6), side=2, center=True)
        assert labels.shape == (6, 6)
        assert labels.min() == 1
        assert labels.max() == 9
        assert np.unique(labels).size == 9

    def test_center_false_margins(self):
        labels = nd_grid_hypercube_labels((7, 7), side=3, center=False)
        assert labels.shape == (7, 7)
        # Last row/col should be outside grid span -> zeros.
        assert np.all(labels[-1, :] == 0)
        assert np.all(labels[:, -1] == 0)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            nd_grid_hypercube_labels([[4, 4]], side=2)

    def test_invalid_side(self):
        with pytest.raises(ValueError):
            nd_grid_hypercube_labels((4, 4), side=0)

    def test_side_too_large(self):
        with pytest.raises(ValueError):
            nd_grid_hypercube_labels((2, 2), side=3)


# ---------------------------------------------------------------------------
# make_label_matrix
# ---------------------------------------------------------------------------

class TestMakeLabelMatrix:
    def test_quadrants(self):
        labels = make_label_matrix(2, 2)
        assert labels.shape == (4, 4)
        assert labels[0, 0] == 0
        assert labels[0, -1] == 2
        assert labels[-1, 0] == 1
        assert labels[-1, -1] == 3

    def test_invalid_N(self):
        with pytest.raises(ValueError):
            make_label_matrix(0, 2)
