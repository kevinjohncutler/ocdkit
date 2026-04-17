"""Tests for ocdkit.measure — bounding boxes, diameters, and medoids."""

import numpy as np
import pytest

from ocdkit.measure import (
    bbox_to_slice,
    make_square,
    crop_bbox,
    dist_to_diam,
    diameters,
    get_medoids,
    argmin_cdist,
    bartlett_nd,
    find_highest_density_box,
    extract_patches,
    create_pill_mask,
)


# ---------------------------------------------------------------------------
# bbox_to_slice
# ---------------------------------------------------------------------------

class TestBboxToSlice:
    def test_basic_2d(self):
        slc = bbox_to_slice([2, 3, 7, 8], (10, 10))
        assert slc[0] == slice(2, 7)
        assert slc[1] == slice(3, 8)

    def test_padding(self):
        slc = bbox_to_slice([2, 3, 7, 8], (10, 10), pad=1)
        assert slc[0] == slice(1, 8)
        assert slc[1] == slice(2, 9)

    def test_padding_clamped(self):
        slc = bbox_to_slice([0, 0, 5, 5], (10, 10), pad=10)
        assert slc[0] == slice(0, 10)
        assert slc[1] == slice(0, 10)

    def test_im_pad(self):
        slc = bbox_to_slice([0, 0, 10, 10], (10, 10), im_pad=2)
        assert slc[0] == slice(2, 8)
        assert slc[1] == slice(2, 8)

    def test_per_axis_pad(self):
        slc = bbox_to_slice([2, 3, 7, 8], (10, 10), pad=[1, 2])
        assert slc[0] == slice(1, 8)
        assert slc[1] == slice(1, 10)

    def test_3d(self):
        slc = bbox_to_slice([0, 0, 0, 5, 5, 5], (10, 10, 10))
        assert len(slc) == 3
        assert slc[0] == slice(0, 5)


# ---------------------------------------------------------------------------
# make_square
# ---------------------------------------------------------------------------

class TestMakeSquare:
    def test_already_square(self):
        bbox = make_square((0, 0, 10, 10), (20, 20))
        assert bbox[2] - bbox[0] == bbox[3] - bbox[1]

    def test_wide_to_square(self):
        # 20 wide x 4 tall → expand height to 20
        bbox = make_square((10, 0, 14, 20), (30, 30))
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        assert h == w

    def test_tall_to_square(self):
        bbox = make_square((0, 10, 20, 14), (30, 30))
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        assert h == w

    def test_clamped_to_image(self):
        # bbox near edge — square expansion can't go below 0
        miny, minx, maxy, maxx = make_square((0, 10, 4, 14), (20, 20))
        assert miny >= 0
        assert minx >= 0
        assert maxy <= 20
        assert maxx <= 20


# ---------------------------------------------------------------------------
# crop_bbox
# ---------------------------------------------------------------------------

class TestCropBbox:
    def test_single_region(self):
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[5:15, 5:15] = 1
        slices = crop_bbox(mask, pad=0, iterations=0)
        assert len(slices) == 1
        assert slices[0][0].start <= 5
        assert slices[0][0].stop >= 15

    def test_multiple_regions(self):
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[2:6, 2:6] = 1
        mask[20:24, 20:24] = 2
        slices = crop_bbox(mask, pad=0, iterations=0)
        assert len(slices) == 2

    def test_get_biggest(self):
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[2:5, 2:5] = 1     # 9 pixels
        mask[10:20, 10:20] = 2  # 100 pixels
        slices = crop_bbox(mask, pad=0, iterations=0, get_biggest=True)
        assert len(slices) == 1
        # Largest region's bounds
        s = slices[0]
        assert s[0].start <= 10 and s[0].stop >= 20

    def test_area_cutoff(self):
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[0, 0] = 1   # tiny
        mask[10:20, 10:20] = 2
        slices = crop_bbox(mask, pad=0, iterations=0, area_cutoff=10)
        assert len(slices) == 1

    def test_binary_merged(self):
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[2:5, 2:5] = 1
        mask[20:25, 20:25] = 2
        merged = crop_bbox(mask, pad=0, iterations=0, binary=True)
        # Should be a single tuple of slices, not a list
        assert isinstance(merged, tuple)
        assert merged[0].start <= 2 and merged[0].stop >= 25


# ---------------------------------------------------------------------------
# dist_to_diam
# ---------------------------------------------------------------------------

class TestDistToDiam:
    def test_2d(self):
        # For a 2D circle of radius R, mean dt = R/3, so diam = 2*(2+1)*R/3 = 2R
        dt_pos = np.full(100, 5.0)
        d = dist_to_diam(dt_pos, n=2)
        assert np.isclose(d, 2 * 3 * 5.0)

    def test_3d(self):
        dt_pos = np.full(100, 5.0)
        d = dist_to_diam(dt_pos, n=3)
        assert np.isclose(d, 2 * 4 * 5.0)


# ---------------------------------------------------------------------------
# diameters
# ---------------------------------------------------------------------------

class TestDiameters:
    def test_disk(self):
        # 20x20 disk should have diameter ~20
        y, x = np.ogrid[:30, :30]
        mask = ((y - 15)**2 + (x - 15)**2 < 100).astype(np.int32)
        d = diameters(mask)
        # Within 30% of expected (distance transform approximation)
        assert 10 < d < 25

    def test_empty_mask_returns_zero(self):
        mask = np.zeros((10, 10), dtype=np.int32)
        d = diameters(mask)
        assert d == 0

    def test_with_precomputed_dt(self):
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[10:20, 10:20] = 1
        import edt
        dt = edt.edt(mask)
        d = diameters(mask, dt=dt)
        assert d > 0

    def test_dist_threshold_filters(self):
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[10:20, 10:20] = 1
        d_no = diameters(mask, dist_threshold=0)
        d_high = diameters(mask, dist_threshold=10)
        # High threshold should filter out everything → 0
        assert d_no > 0
        assert d_high == 0

    def test_return_length(self):
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[10:20, 10:20] = 1
        result = diameters(mask, return_length=True)
        assert isinstance(result, tuple)
        diam, length = result
        assert diam > 0 and length > 0

    def test_pill_decomposition(self):
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[10:20, 5:25] = 1  # rectangle (rod)
        result = diameters(mask, pill=True)
        assert isinstance(result, tuple)
        R, L = result
        assert R > 0
        assert L > 0


# ---------------------------------------------------------------------------
# get_medoids
# ---------------------------------------------------------------------------

def _paint_disk(labels, center, radius, label_id):
    """Paint a filled disk into *labels* at *center* with given *radius*."""
    rr = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(rr, rr, indexing="ij")
    mask = (yy ** 2 + xx ** 2) <= radius ** 2
    cy, cx = center
    pts_y = yy[mask] + cy
    pts_x = xx[mask] + cx
    valid = (
        (pts_y >= 0)
        & (pts_y < labels.shape[0])
        & (pts_x >= 0)
        & (pts_x < labels.shape[1])
    )
    labels[pts_y[valid], pts_x[valid]] = label_id


class TestGetMedoids:
    def test_circle_centers(self):
        """Medoid of a disk should be near its geometric center."""
        labels = np.zeros((64, 64), dtype=np.int32)
        centers = [(16, 16), (16, 48), (48, 16), (48, 48)]
        radius = 5
        for i, center in enumerate(centers, start=1):
            _paint_disk(labels, center, radius, i)

        medoid_coords, medoid_labels = get_medoids(labels, do_skel=False)
        assert medoid_coords is not None
        assert medoid_labels is not None

        by_label = {int(lbl): coord for lbl, coord in zip(medoid_labels, medoid_coords)}
        for label_id, center in enumerate(centers, start=1):
            coord = by_label[label_id]
            dist = np.linalg.norm(coord - np.array(center))
            assert dist <= 2.0

    def test_return_dists(self):
        labels = np.zeros((32, 32), dtype=np.int32)
        _paint_disk(labels, (16, 16), 5, 1)
        medoid_coords, medoid_labels, inner_dists = get_medoids(
            labels, do_skel=False, return_dists=True
        )
        assert medoid_coords is not None
        assert inner_dists.shape == labels.shape

    def test_empty(self):
        labels = np.zeros((16, 16), dtype=np.int32)
        coords, mlabels = get_medoids(labels, do_skel=False)
        assert coords is None and mlabels is None


# ---------------------------------------------------------------------------
# bartlett_nd
# ---------------------------------------------------------------------------

class TestBartlettND:
    def test_1d_normalized(self):
        kernel = bartlett_nd(5)
        assert kernel.shape == (5,)
        assert np.isclose(kernel.sum(), 1.0)

    def test_2d_normalized(self):
        kernel = bartlett_nd((3, 5))
        assert kernel.shape == (3, 5)
        assert np.isclose(kernel.sum(), 1.0)

    def test_3d(self):
        kernel = bartlett_nd((3, 3, 3))
        assert kernel.shape == (3, 3, 3)
        assert np.isclose(kernel.sum(), 1.0)


# ---------------------------------------------------------------------------
# find_highest_density_box
# ---------------------------------------------------------------------------

class TestFindHighestDensityBox:
    def test_full_image(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:5, 2:5] = 1
        full = find_highest_density_box(labels, -1)
        assert full == (slice(0, 10), slice(0, 10))

    def test_finds_dense_cluster(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:5, 2:5] = 1
        labels[7:8, 7:8] = 1

        slc = find_highest_density_box(labels, 3)
        assert len(slc) == 2
        assert (slc[0].stop - slc[0].start) == 3
        assert (slc[1].stop - slc[1].start) == 3
        # box should include the densest cluster near the top-left
        assert slc[0].start <= 3 <= slc[0].stop
        assert slc[1].start <= 3 <= slc[1].stop


# ---------------------------------------------------------------------------
# extract_patches
# ---------------------------------------------------------------------------

class TestExtractPatches:
    def test_edges_and_fill(self):
        img = np.arange(25, dtype=np.int32).reshape(5, 5)
        points = [(0, 0), (4, 4)]
        patches, slices = extract_patches(img, points, box_size=3, fill_value=-1, point_order="yx")
        assert patches.shape == (2, 3, 3)
        # Top-left point: out-of-bounds upper/left filled with -1
        assert patches[0, 0, 0] == -1
        # Bottom-right point: out-of-bounds lower/right filled with -1
        assert patches[1, -1, -1] == -1
        assert len(slices) == 2

    def test_point_order_xy(self):
        img = np.arange(25, dtype=np.int32).reshape(5, 5)
        patches, _ = extract_patches(img, [(0, 0)], box_size=3, fill_value=-1, point_order="xy")
        assert patches.shape == (1, 3, 3)

    def test_invalid_point_order(self):
        img = np.zeros((5, 5), dtype=np.int32)
        with pytest.raises(ValueError):
            extract_patches(img, [(0, 0)], box_size=3, point_order="bad")


# ---------------------------------------------------------------------------
# create_pill_mask
# ---------------------------------------------------------------------------

class TestCreatePillMask:
    def test_shape_and_content(self):
        mask = create_pill_mask(R=2, L=4, f=1)
        assert mask.dtype == np.uint8
        assert mask.shape == (11, 15)
        assert mask.sum() > 0


# ---------------------------------------------------------------------------
# curve_filter
# ---------------------------------------------------------------------------

class TestCurveFilter:
    def test_shapes(self):
        from ocdkit.measure import curve_filter
        img = np.zeros((32, 32), dtype=np.float32)
        img[16, 16] = 1.0
        outputs = curve_filter(img, filterWidth=1.2)
        for out in outputs:
            assert out.shape == img.shape
            assert np.isfinite(out).all()
        # nonneg variants (first 4)
        for out in outputs[:4]:
            assert (out >= 0).all()


# ---------------------------------------------------------------------------
# label_overlap / intersection_over_union / true_positive
# ---------------------------------------------------------------------------

class TestLabelOverlap:
    def test_basic(self):
        from ocdkit.measure import label_overlap
        x = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
        y = np.array([0, 1, 2, 2, 2, 0], dtype=np.int32)
        overlap = label_overlap(x, y)
        assert overlap.shape == (3, 3)
        assert overlap[0, 0] == 1  # both background
        assert overlap[1, 1] == 1
        assert overlap[1, 2] == 1
        assert overlap[2, 2] == 2
        assert overlap[2, 0] == 1


class TestIntersectionOverUnion:
    def _make_masks(self):
        gt = np.zeros((32, 32), dtype=np.int32)
        gt[5:15, 5:15] = 1
        gt[18:28, 18:28] = 2
        pred = np.zeros((32, 32), dtype=np.int32)
        pred[7:17, 7:17] = 1  # shifted
        pred[18:28, 18:28] = 2  # perfect
        return gt, pred

    def test_perfect_match(self):
        from ocdkit.measure import intersection_over_union
        gt = np.zeros((16, 16), dtype=np.int32)
        gt[2:10, 2:10] = 1
        iou = intersection_over_union(gt, gt)
        assert iou[1, 1] == pytest.approx(1.0)

    def test_shifted(self):
        from ocdkit.measure import intersection_over_union
        gt, pred = self._make_masks()
        iou = intersection_over_union(gt, pred)
        assert iou[2, 2] == pytest.approx(1.0)  # perfect overlap
        assert 0 < iou[1, 1] < 1.0  # partial overlap


class TestTruePositive:
    def test_basic(self):
        from ocdkit.measure import intersection_over_union, true_positive
        gt = np.zeros((32, 32), dtype=np.int32)
        gt[2:12, 2:12] = 1
        gt[15:25, 15:25] = 2
        pred = gt.copy()
        iou = intersection_over_union(gt, pred)[1:, 1:]
        assert true_positive(iou, 0.5) == 2

    def test_no_match_at_high_threshold(self):
        from ocdkit.measure import intersection_over_union, true_positive
        gt = np.zeros((32, 32), dtype=np.int32)
        gt[2:12, 2:12] = 1
        pred = np.zeros_like(gt)
        pred[8:18, 8:18] = 1  # small overlap
        iou = intersection_over_union(gt, pred)[1:, 1:]
        assert true_positive(iou, 0.9) == 0
