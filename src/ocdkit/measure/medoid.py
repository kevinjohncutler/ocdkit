"""Per-label medoid extraction."""

from .imports import *
import edt as _edt


def argmin_cdist(X, labels, distance_values):
    """Per-label medoid index via pairwise-distance argmin (torch).

    For each unique label group in *labels*, find the index in *X* that
    minimizes the distance-field-weighted sum of pairwise distances within
    the group. The weighting prefers points that are both centrally located
    in the group AND have a high distance-field value.

    Parameters
    ----------
    X : torch.Tensor
        Coordinates, shape ``(N, D)``. Must be sorted by *labels*.
    labels : torch.Tensor
        Per-row label index, shape ``(N,)``. Must be sorted.
    distance_values : torch.Tensor
        Per-row distance-field value, shape ``(N,)``.

    Returns
    -------
    argmin_indices : torch.Tensor
        Indices into ``X`` of the chosen medoids, one per unique label.
    adjusted_summed_distances_all : torch.Tensor
        The weighted-sum-of-distances score for every row of *X*.
    """
    unique_labels, label_counts = torch.unique_consecutive(labels, return_counts=True)
    label_starts = torch.cumsum(
        torch.cat([torch.tensor([0], device=labels.device), label_counts[:-1]]), dim=0
    )
    label_ends = torch.cumsum(label_counts, dim=0)

    argmin_indices = []
    adjusted_summed_distances_all = torch.full((len(X),), float('nan'), device=X.device)

    for i in range(len(unique_labels)):
        start_idx = label_starts[i]
        end_idx = label_ends[i]
        label_indices = torch.arange(start_idx, end_idx, device=X.device)

        X_label = X[label_indices]
        distance_values_label = distance_values[label_indices]

        if X_label.shape[0] > 1:
            distances = torch.cdist(X_label, X_label)
            summed_distances = torch.sum(distances, dim=1)

            # Weight: penalizes points with low distance-field values
            adjusted = summed_distances * (1 + 1 / distance_values_label)

            adjusted_summed_distances_all[label_indices] = adjusted
            argmin_index_in_label = torch.argmin(adjusted)
            argmin_indices.append(label_indices[argmin_index_in_label])
        else:
            argmin_indices.append(label_indices[0])
            adjusted_summed_distances_all[label_indices] = 0

    return torch.tensor(argmin_indices, device=X.device), adjusted_summed_distances_all


def get_medoids(labels, do_skel=True, return_dists=False):
    """Get medoid coordinates and labels from a label mask.

    Parameters
    ----------
    labels : ndarray
        Integer label mask.
    do_skel : bool
        If True, restrict candidate medoid points to the label skeleton
        (faster + topologically meaningful for elongated cells). If False,
        consider all foreground pixels and weight by the EDT distance field.
    return_dists : bool
        If True, also return a per-pixel "centeredness" score map.

    Returns
    -------
    medoids : ndarray, shape ``(n_labels, ndim)`` or ``None``
    mlabels : ndarray, shape ``(n_labels,)`` or ``None``
    inner_dists : ndarray (only if return_dists=True)
    """
    from ..array import skeletonize

    if do_skel:
        masks = skeletonize(labels)
        dists = np.ones_like(labels)
    else:
        masks = labels
        dists = _edt.edt(labels)

    coords = np.argwhere(masks > 0)
    slc = tuple(coords.T)
    labs = masks[slc]
    sort = np.argsort(labs)
    sort_coords = coords[sort]
    sort_labels = labs[sort]
    sort_dists = dists[slc][sort]

    inds_tensor, dists_tensor = argmin_cdist(
        torch.tensor(sort_coords).float(),
        torch.tensor(sort_labels).float(),
        torch.tensor(sort_dists).float(),
    )

    inds = inds_tensor.cpu().numpy()
    dists_arr = dists_tensor.cpu().numpy()

    if len(inds):
        inds = np.atleast_1d(inds)
        medoids = sort_coords[inds]
        mlabels = sort_labels[inds]
        if medoids.ndim == 1:
            medoids = medoids[None]

        if return_dists:
            inner_dists = np.zeros(masks.shape, dtype=dists_arr.dtype)
            inner_dists[tuple(sort_coords.T)] = dists_arr
            return medoids, mlabels, inner_dists
        return medoids, mlabels

    return None, None
