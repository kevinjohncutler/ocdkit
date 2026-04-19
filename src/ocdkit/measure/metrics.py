"""Label overlap metrics (IoU, average precision)."""

from .imports import *
from numba import jit
from scipy.optimize import linear_sum_assignment


@jit(nopython=True)
def label_overlap(x, y):
    """Pixel overlap matrix between two label arrays.

    Returns an ``(x.max()+1, y.max()+1)`` matrix where entry ``(i, j)``
    is the number of pixels that have label *i* in *x* and label *j* in *y*.
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def intersection_over_union(masks_true, masks_pred):
    """IoU matrix for all mask pairs in two label images.

    Returns an ``(N_true+1, N_pred+1)`` float array where row/column 0
    corresponds to the background label.
    """
    overlap = label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def true_positive(iou, th):
    """Count true positives at IoU threshold *th* via linear sum assignment."""
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    return match_ok.sum()
