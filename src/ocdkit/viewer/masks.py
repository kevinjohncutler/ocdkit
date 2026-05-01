"""Generic mask utilities — plugin-independent operations.

N-coloring and label formatting via the ``ncolor`` package. Plugin-agnostic;
the viewer routes call these directly so plugins do not have to re-implement
them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_ncolor_mask(mask: np.ndarray, *, expand: bool = True) -> Optional[np.ndarray]:
    """Apply n-coloring to a label mask.

    Returns a uint32 array where each pixel's value is its color group id
    (1..N), or ``None`` if ``ncolor`` is unavailable / mask is empty.
    """
    try:
        import ncolor
    except ImportError:
        return None
    if mask.size == 0:
        return None
    mask_int = np.asarray(mask, dtype=np.int32)
    mask_for_label = mask_int
    try:
        import fastremap  # type: ignore
        unique = fastremap.unique(mask_int)
        if unique.size:
            unique = unique[unique > 0]
        if unique.size:
            mapping = {int(value): idx + 1 for idx, value in enumerate(unique)}
            mask_for_label = fastremap.remap(
                mask_int, mapping, preserve_missing_labels=True, in_place=False
            )
    except Exception:
        mask_for_label = mask_int
    try:
        labeled, _ngroups = ncolor.label(
            mask_for_label,
            max_depth=20,
            expand=expand,
            return_n=True,
            format_input=False,
        )
    except TypeError:
        try:
            labeled = ncolor.label(
                mask_for_label, max_depth=20, expand=expand, format_input=False
            )
        except TypeError:
            labeled = ncolor.label(mask_for_label, max_depth=20, format_input=False)
    return np.ascontiguousarray(labeled.astype(np.uint32, copy=False))


def format_labels(mask: np.ndarray, *, clean: bool = False, min_area: int = 1) -> np.ndarray:
    """Renumber a label mask to be consecutive (1..N).

    Returns int32 by convention. Raises ``RuntimeError`` if ``ncolor`` is
    unavailable.
    """
    try:
        import ncolor
    except ImportError as exc:
        raise RuntimeError("ncolor package is required for format_labels") from exc
    return ncolor.format_labels(
        mask, clean=clean, min_area=min_area, despur=False, verbose=False
    )
