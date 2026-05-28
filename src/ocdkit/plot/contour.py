"""Vector contour rendering for segmentation masks.

Two pipelines live here:

  ``vector_contours``           — legacy: find_boundaries → boundary_to_masks →
      affinity graph → ocdkit.array.spatial.get_contour → splprep/splev. Outline
      only. Kept for backwards compatibility.

  ``vector_contours_marching``  — preferred: single-pass marching-squares numba
      kernel + periodic Gaussian smooth. Supports fill / edge / per-cell
      colors (incl. ncolor-style dict-of-lists). Faster and cleaner.

  ``cells_to_polygons``         — data-only sibling of ``vector_contours_marching``:
      returns ``list[(verts_xy, fill_rgba)]`` for SVG / WebGL / non-matplotlib
      consumers. Same smoothing as the matplotlib path so visual output matches.

  ``cells_to_webgpu_mesh``      — triangulated mesh (CellMesh) ready for upload
      to WebGPU vertex/index buffers. Smoothing left for the vertex shader.
"""

from .imports import *
import matplotlib.path as mpath
from matplotlib.collections import PatchCollection, PathCollection, PolyCollection
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from numba import njit


# Default Gaussian smoothing length, in pixels of contour arclength.
# 2*sqrt(2) ≈ 2.83 sits at the natural scale where 2-pixel zigzags from the
# marching-squares staircase are averaged out, while cell-shape harmonics with
# wavelength larger than the cell radius are preserved.
DEFAULT_SMOOTH_SIGMA = 2.0 * np.sqrt(2.0)


def vector_contours(fig,ax,mask, crop=None, smooth_factor=5, color = 'r', linewidth=1,
                    y_offset=0, x_offset=0,
                    pad=2,
                    mode='constant',
                    zorder=1,
                    ):

    msk = np.pad(mask,pad,mode='edge')

    # msk = np.pad(mask,pad,mode=mode)

    if crop is not None:
        # Crop the mask to the specified region
        msk = msk[crop]

    msk = np.pad(msk,1,mode='constant', constant_values=0)

    # set up dimensions
    dim = msk.ndim
    shape = msk.shape

    from ..array.spatial import (
        kernel_setup, get_neighbors,
        boundary_to_masks, masks_to_affinity, get_contour,
    )
    from skimage.segmentation import find_boundaries

    steps,inds,idx,fact,sign = kernel_setup(dim)

    # remove spur points - this method is way easier than running core._despur() on the priginal affinity graph
    bd = find_boundaries(msk,mode='inner',connectivity=2)
    msk, bounds, _ = boundary_to_masks(bd,binary_mask=msk>0,connectivity=1,min_size=0)

    # generate affinity graph
    coords = np.nonzero(msk)
    neighbors = get_neighbors(tuple(coords),steps,dim,shape) # shape (d,3**d,npix)
    affinity_graph =  masks_to_affinity(msk, coords, steps, inds, idx, fact, sign, dim, neighbors)

    # find contours
    contour_map, contour_list, unique_L = get_contour(msk,
                                                    affinity_graph,
                                                    coords,
                                                    neighbors,
                                                    cardinal_only=True)

    # List to hold patches
    patches = []
    for contour in contour_list:
        if len(contour) > 1:
            pts = np.stack([c[contour] for c in coords]).T[:, ::-1]  # YX to XY
            pts+= np.array([x_offset,y_offset])  # Apply offsets
            tck, u = splprep(pts.T, u=None, s=len(pts)/smooth_factor, per=1)
            u_new = np.linspace(u.min(), u.max(), len(pts))
            x_new, y_new = splev(u_new, tck, der=0)


            # Define the points of the polygon
            # points = np.column_stack([y_new-pad+y_offset, x_new-pad+x_offset])
            # points = np.column_stack([ x_new-pad+x_offset,y_new-pad+y_offset])
            # points = np.column_stack([ x_new-2*pad+x_offset,y_new-2*pad+y_offset])
            # points = np.column_stack([x_new-pad,y_new-pad])
            if isinstance(pad,tuple):
                # If pad is a tuple, apply it to x and y separately
                points = np.column_stack([x_new-(pad[0][0]+1), y_new-(pad[1][0]+1)])
            else:
                points = np.column_stack([x_new-(pad+1),y_new-(pad+1)])




            # Create a Path from the points
            path = mpath.Path(points, closed=True)

            # Create a PathPatch from the Path
            patch = mpatches.PathPatch(path, fill=None, edgecolor=color,
                                    #    linewidth= fig.dpi/72,
                                        linewidth=linewidth,
                                        zorder=zorder,
                                       capstyle='round')

            # ax.add_patch(patch)

            # Add patch to list
            patches.append(patch)

    # Create a PatchCollection from the list of patches
    # Add the PatchCollection to the axis/axes
    if isinstance(ax,list):
        for a in ax:
            patch_collection = PatchCollection(patches, match_original=True, snap=False)
            a.add_collection(patch_collection)
    else:
        patch_collection = PatchCollection(patches, match_original=True, snap=False)
        ax.add_collection(patch_collection)


# ===========================================================================
# Marching-squares pipeline (preferred path).
# ===========================================================================
#
# Bypasses the entire legacy chain (find_boundaries -> boundary_to_masks ->
# get_neighbors -> masks_to_affinity -> parametrize_contours) with one numba
# kernel that walks the dual lattice (corners between pixels) directly,
# emitting per-label closed polylines at sub-pixel coordinates.
#
# Cell pattern at a lattice corner (TL, TR, BL, BR) where each bit is 1 iff
# that cell == the active label. Bits: TL=8, TR=4, BL=2, BR=1.
#
# Edges around a corner: N (between TL,TR), E (TR,BR), S (BL,BR), W (TL,BL).
# An edge is part of L's boundary iff its two cells have different "is L"
# status. For non-saddle patterns the corner has exactly 0 or 2 such edges,
# and the 2-edge case pairs uniquely.
#
# ``_NEXT_DIR[pattern, entry_edge] = exit_direction`` (and exit_direction also
# encodes the direction of motion on leaving). Direction codes: N=0, E=1,
# S=2, W=3. -1 means "should not occur" (entry edge not part of pattern).
#
# Saddles (patterns 6 and 9) split into two separate crossings: pattern 6
# (TR,BL diagonal) pairs N<->E (around TR) and S<->W (around BL); pattern 9
# (TL,BR diagonal) pairs N<->W (around TL) and S<->E (around BR).

_NEXT_DIR = np.array([
    [-1, -1, -1, -1],   # 0  (no L cells)
    [-1,  2,  1, -1],   # 1  BR        edges E,S
    [-1, -1,  3,  2],   # 2  BL        edges S,W
    [-1,  3, -1,  1],   # 3  BL,BR     edges E,W
    [ 1,  0, -1, -1],   # 4  TR        edges N,E
    [ 2, -1,  0, -1],   # 5  TR,BR     edges N,S
    [ 1,  0,  3,  2],   # 6  TR,BL  -- saddle: N<->E, S<->W
    [ 3, -1, -1,  0],   # 7  TR,BL,BR  edges N,W
    [ 3, -1, -1,  0],   # 8  TL        edges N,W
    [ 3,  2,  1,  0],   # 9  TL,BR  -- saddle: N<->W, S<->E
    [ 2, -1,  0, -1],   # 10 TL,BL     edges N,S
    [ 1,  0, -1, -1],   # 11 TL,BL,BR  edges N,E
    [-1,  3, -1,  1],   # 12 TL,TR     edges E,W
    [-1, -1,  3,  2],   # 13 TL,TR,BR  edges S,W   (BL=0 is the outsider)
    [-1,  2,  1, -1],   # 14 TL,TR,BL  edges E,S   (BR=0 is the outsider)
    [-1, -1, -1, -1],   # 15 (all L, interior)
], dtype=np.int8)


@njit(cache=True)
def _trace_all_labels(labels, next_dir):
    """Trace one closed boundary loop per label.

    Walks the topmost-leftmost pixel's TL corner per label, marching CCW
    along the dual lattice. Returns concatenated (x, y) corners and per-label
    offsets.
    """
    H, W = labels.shape

    max_label = 0
    for i in range(H):
        for j in range(W):
            v = labels[i, j]
            if v > max_label:
                max_label = v

    # Topmost-leftmost pixel for each label (single scan).
    starts = np.full((max_label + 2, 2), -1, dtype=np.int64)
    for i in range(H):
        for j in range(W):
            l = labels[i, j]
            if l > 0 and starts[l, 0] == -1:
                starts[l, 0] = i
                starts[l, 1] = j

    cap = max(64, H * W * 2)
    out_xy = np.empty((cap, 2), dtype=np.float64)
    offsets = np.zeros(max_label + 2, dtype=np.int64)

    cursor = 0

    for lab in range(1, max_label + 1):
        offsets[lab] = cursor
        if starts[lab, 0] < 0:
            continue

        sy = starts[lab, 0]
        sx = starts[lab, 1]
        y, x = sy, sx
        direction = np.int8(1)  # E (along top edge of starting pixel)

        max_steps = H * W * 4
        steps = 0

        while steps < max_steps:
            out_xy[cursor, 0] = x
            out_xy[cursor, 1] = y
            cursor += 1

            if direction == 0:
                y -= 1
            elif direction == 1:
                x += 1
            elif direction == 2:
                y += 1
            else:
                x -= 1

            if y == sy and x == sx:
                break

            tl = (y > 0 and x > 0) and (labels[y - 1, x - 1] == lab)
            tr = (y > 0 and x < W) and (labels[y - 1, x] == lab)
            bl = (y < H and x > 0) and (labels[y, x - 1] == lab)
            br = (y < H and x < W) and (labels[y, x] == lab)

            pattern = 0
            if tl:
                pattern |= 8
            if tr:
                pattern |= 4
            if bl:
                pattern |= 2
            if br:
                pattern |= 1

            entry = direction ^ 2  # opposite edge
            nd = next_dir[pattern, entry]
            if nd < 0:
                break
            direction = nd
            steps += 1

    offsets[max_label + 1] = cursor
    return out_xy[:cursor], offsets, max_label


def _gaussian_smooth_closed(P, sigma=2.0):
    if len(P) < 3 or sigma <= 0:
        return P
    x = gaussian_filter1d(P[:, 0], sigma=sigma, mode='wrap')
    y = gaussian_filter1d(P[:, 1], sigma=sigma, mode='wrap')
    return np.column_stack([x, y])


def _offset_closed_polygon(P, distance):
    """Shift each vertex of a closed polygon along the local inward normal.

    The Tier-2 walk traverses each label's boundary with the cell interior on
    the right of the direction of motion (in screen coords with y-down). For
    a forward tangent (tx, ty), the right-perpendicular pointing into the
    cell is (-ty, tx). At each vertex we average the inward normal of the
    incoming and outgoing edges, normalize, then shift by ``distance``.
    """
    if len(P) < 3 or distance == 0:
        return P
    fwd = np.roll(P, -1, axis=0) - P
    n_fwd = np.column_stack([-fwd[:, 1], fwd[:, 0]])
    n_fwd /= np.maximum(np.linalg.norm(n_fwd, axis=1, keepdims=True), 1e-9)
    n_prev = np.roll(n_fwd, 1, axis=0)
    n = n_fwd + n_prev
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return P + distance * n


def _is_color_spec(x):
    """Quick check: does matplotlib think ``x`` is a single color?"""
    from matplotlib.colors import to_rgba
    try:
        to_rgba(x)
        return True
    except (ValueError, TypeError):
        return False


def _resolve_colors(colors, max_label, default='r'):
    """Build a length-(max_label+1) RGBA lookup table from various input forms.

    Accepted shapes for ``colors``:
      - None or a single color spec ('r', '#ff0000', RGBA tuple, ...) → uniform
      - dict {label_id: color}                                        → per label
      - dict {color: [label_ids]}     (detected by list-like values)  → N-coloring
      - array-like of length max_label+1 (color specs or RGBA rows)   → indexed
      - callable lab -> color                                         → per call

    Index 0 is reserved for background.
    """
    from matplotlib.colors import to_rgba

    lut = np.empty((max_label + 1, 4), dtype=np.float64)
    lut[:] = to_rgba(default if colors is None or isinstance(colors, dict)
                     else colors if _is_color_spec(colors) else default)

    if colors is None:
        return lut

    if callable(colors):
        for lab in range(1, max_label + 1):
            lut[lab] = to_rgba(colors(lab))
        return lut

    if isinstance(colors, dict):
        first_val = next(iter(colors.values()))
        is_inverse = isinstance(first_val, (list, tuple, np.ndarray)) \
                     and not _is_color_spec(first_val)
        if is_inverse:
            for color, labels in colors.items():
                rgba = to_rgba(color)
                for lab in labels:
                    if 1 <= lab <= max_label:
                        lut[lab] = rgba
        else:
            for lab, color in colors.items():
                if 1 <= lab <= max_label:
                    lut[lab] = to_rgba(color)
        return lut

    if _is_color_spec(colors):
        return lut

    arr = np.asarray(colors, dtype=object) if not isinstance(colors, np.ndarray) \
          else colors
    if arr.ndim == 1 and arr.dtype != object and arr.shape[0] >= max_label + 1:
        for i in range(min(len(arr), max_label + 1)):
            lut[i] = to_rgba(arr[i])
    elif arr.ndim == 2 and arr.shape[1] in (3, 4):
        n = min(arr.shape[0], max_label + 1)
        if arr.shape[1] == 3:
            lut[:n, :3] = arr[:n]
            lut[:n, 3] = 1.0
        else:
            lut[:n] = arr[:n]
    else:
        for i in range(min(len(arr), max_label + 1)):
            lut[i] = to_rgba(arr[i])
    return lut


def _kiss_offset(fig, ax, linewidth_pt):
    """Inward inset (in image pixels) implementing matplotlib-equivalent
    "stroke-alignment: inner". Inset by linewidth/2 along the inward
    normal so the stroke's OUTER extent lands exactly on the contour
    (same edge as the filled polygon would have).
    """
    try:
        pos = ax.get_position()
        fig_w_in, _ = fig.get_size_inches()
        ax_w_in = pos.width * fig_w_in
        xlim = ax.get_xlim()
        ax_w_data = abs(xlim[1] - xlim[0])
        if ax_w_in <= 0 or ax_w_data <= 0 or xlim == (0.0, 1.0):
            raise ValueError
        data_per_inch = ax_w_data / ax_w_in
        lw_inch = linewidth_pt / 72.0
        return lw_inch * data_per_inch * 0.5
    except (ValueError, ZeroDivisionError):
        return 0.5


def cells_to_polygons(mask, *, colors=None, default_color='r',
                      smooth_sigma=DEFAULT_SMOOTH_SIGMA,
                      offset=None,
                      x_offset=0.0, y_offset=0.0):
    """Return per-cell smoothed closed polygons as Python data.

    Data-only sibling of :func:`vector_contours_marching` -- runs the same
    marching-squares + Gaussian smoothing pipeline but emits a list of
    ``(verts_xy, fill_rgba)`` tuples instead of building a PolyCollection.
    Use this when you need vector polygons for SVG / WebGL / custom canvas
    rendering without going through matplotlib.

    Parameters
    ----------
    mask : (H, W) int array
        Label matrix; 0 = background.
    colors : various
        Per-cell color spec. See ``_resolve_colors``. ``None`` falls back to
        ``default_color``.
    smooth_sigma : float
        Gaussian smoothing length in pixels of arclength. Set 0 for no smoothing.
    offset : float or None
        Inward inset along the local normal. ``None`` = 0.
    x_offset, y_offset : float
        Translation applied to all vertices (so the polygon lands in plot
        coordinates that aren't the raw image grid).

    Returns
    -------
    list of (ndarray (N, 2) float32, ndarray (4,) float64)
        Per-cell (closed) polygon vertices in (x, y) order, and its RGBA fill.
        Polygons are CLOSED (last vertex != first; caller is responsible for
        re-emitting the first vertex if their renderer needs an explicit close).
    """
    if mask.dtype != np.int32:
        mask_i = mask.astype(np.int32)
    else:
        mask_i = mask
    pts, offsets, max_label = _trace_all_labels(mask_i, _NEXT_DIR)
    color_lut = _resolve_colors(colors if colors is not None else default_color,
                                max_label)
    out = []
    for lab in range(1, max_label + 1):
        s, e = offsets[lab], offsets[lab + 1]
        if e - s < 3:
            continue
        contour = pts[s:e].copy()
        contour[:, 0] += x_offset - 0.5
        contour[:, 1] += y_offset - 0.5
        if smooth_sigma and smooth_sigma > 0:
            contour = _gaussian_smooth_closed(contour, sigma=smooth_sigma)
        if offset:
            contour = _offset_closed_polygon(contour, offset)
        out.append((contour.astype(np.float32, copy=False),
                    color_lut[lab].copy()))
    return out


def vector_contours_marching(fig, ax, mask, smooth_sigma=DEFAULT_SMOOTH_SIGMA,
                             offset=None, colors=None, color='r', linewidth=1,
                             fill=False, edge=True, alpha=None,
                             x_offset=0, y_offset=0, zorder=1):
    """Marching-squares vector contour renderer (preferred path).

    Single-pass numba kernel + periodic Gaussian smooth, rendered as one
    PolyCollection (fill mode) or PathCollection annulus (outline mode).
    Adjacent cells' outlines kiss exactly at the shared boundary (no gap,
    no overlap) regardless of linewidth.

    Parameters
    ----------
    offset : float or None
        Inward shift (in image pixels) of the contour along the local normal,
        applied after Gaussian smoothing.
        - ``None`` (default): auto-compute so adjacent cell strokes just touch
          at the shared boundary (kiss, no overlap). For ``fill=True`` the
          auto value is 0.
        - ``0.0``: contour sits on the geometric cell boundary.
        - ``0.5``: contour passes through the centers of the outermost pixels.
        - ``< 0``: push the contour outside the cell.
    smooth_sigma : float
        Periodic Gaussian smoothing applied to (x, y) along the contour.
        - sqrt(2) ≈ 1.41 :: minimum to suppress the pixel staircase
        - 2*sqrt(2) ≈ 2.83 :: removes 2-pixel zigzags (default)
        - 3*sqrt(2) ≈ 4.24 :: smoother / more stylized
    colors : None, color spec, dict, array-like, or callable
        Per-cell color control. See :func:`_resolve_colors`.
    fill : bool, default False
        If True, render filled polygons (PolyCollection). If False, only
        outlines (rendered as filled annuli for kiss-aligned strokes).
    edge : bool, default True
        Whether to stroke the outline when ``fill=True``.
    alpha : float or None
        Opacity applied uniformly to faces and edges.
    """
    if mask.dtype != np.int32:
        mask_i = mask.astype(np.int32)
    else:
        mask_i = mask

    pts, offsets, max_label = _trace_all_labels(mask_i, _NEXT_DIR)

    color_lut = _resolve_colors(colors if colors is not None else color,
                                max_label)

    if not fill:
        ring_width = 2.0 * _kiss_offset(fig, ax, linewidth)
    else:
        ring_width = 0.0

    outer_polys = []
    annulus_paths = []
    polyline_colors = []
    for lab in range(1, max_label + 1):
        s, e = offsets[lab], offsets[lab + 1]
        if e - s < 3:
            continue
        contour = pts[s:e].copy()
        contour[:, 0] += x_offset - 0.5
        contour[:, 1] += y_offset - 0.5
        if smooth_sigma > 0:
            contour = _gaussian_smooth_closed(contour, sigma=smooth_sigma)
        if fill and offset:
            contour = _offset_closed_polygon(contour, offset)
        outer_polys.append(np.vstack([contour, contour[:1]]))
        polyline_colors.append(color_lut[lab])

        if not fill:
            inner = _offset_closed_polygon(contour, ring_width)[::-1]
            n_o, n_i = len(contour), len(inner)
            verts = np.empty((n_o + n_i + 2, 2), dtype=np.float64)
            verts[:n_o] = contour
            verts[n_o] = contour[0]
            verts[n_o + 1:n_o + 1 + n_i] = inner
            verts[n_o + 1 + n_i] = inner[0]
            codes = np.full(len(verts), mpath.Path.LINETO, dtype=np.uint8)
            codes[0] = mpath.Path.MOVETO
            codes[n_o] = mpath.Path.CLOSEPOLY
            codes[n_o + 1] = mpath.Path.MOVETO
            codes[n_o + 1 + n_i] = mpath.Path.CLOSEPOLY
            annulus_paths.append(mpath.Path(verts, codes))

    if not outer_polys:
        return

    polyline_colors = np.asarray(polyline_colors)

    def _make_outline_pc():
        coll = PathCollection(annulus_paths, facecolors=polyline_colors,
                              edgecolors='none', zorder=zorder)
        if alpha is not None:
            coll.set_alpha(alpha)
        return coll

    def _make_fill_pc():
        coll = PolyCollection(outer_polys, facecolors=polyline_colors,
                              edgecolors=(polyline_colors if edge else 'none'),
                              linewidths=(linewidth if edge else 0.0),
                              zorder=zorder, capstyle='round',
                              joinstyle='round')
        if alpha is not None:
            coll.set_alpha(alpha)
        return coll

    _make = _make_fill_pc if fill else _make_outline_pc

    if isinstance(ax, list):
        for a in ax:
            a.add_collection(_make())
    else:
        ax.add_collection(_make())


# ---------------------------------------------------------------------------
# WebGPU mesh export.
#
# Same geometry as the matplotlib path above, but emitted as triangulated
# meshes ready to upload to WebGPU vertex/index buffers.
#
# Two meshes per call:
#   - fill mesh:     polygon fans (one fan per cell, from centroid)
#   - annulus mesh:  strip of triangles between outer and inner contours
#
# Both meshes share the same outer polygon vertices.
# ---------------------------------------------------------------------------


class CellMesh:
    """Triangulated geometry ready for upload to a WebGPU pipeline.

    Attributes
    ----------
    positions       : float32 (V, 2)
    normals         : float32 (V, 2)
    cell_ids        : uint32  (V,)
    is_inner        : uint32  (V,)
    fill_indices    : uint32 (T_f, 3)
    outline_indices : uint32 (T_o, 3)
    color_lut       : float32 (max_label+1, 4)
    """

    def __init__(self, positions, normals, cell_ids, is_inner,
                 fill_indices, outline_indices, color_lut):
        self.positions = positions
        self.normals = normals
        self.cell_ids = cell_ids
        self.is_inner = is_inner
        self.fill_indices = fill_indices
        self.outline_indices = outline_indices
        self.color_lut = color_lut

    def __repr__(self):
        return (f"CellMesh({len(self.positions)} verts; "
                f"{len(self.fill_indices)} fill tris, "
                f"{len(self.outline_indices)} outline tris; "
                f"{len(self.color_lut)} colors)")


def cells_to_webgpu_mesh(mask, colors=None, default_color='r',
                         x_offset=0.0, y_offset=0.0):
    """Triangulate a label matrix into a WebGPU-ready mesh.

    Emits the RAW marching-squares polygons (no smoothing, no offset). The
    shader can apply periodic smoothing as a vertex transform and synthesize
    the annulus inner ring as ``position + ring_width * normal``.

    Per-vertex normals are precomputed (averaged inward edge normals).
    """
    if mask.dtype != np.int32:
        mask_i = mask.astype(np.int32)
    else:
        mask_i = mask
    pts, offsets, max_label = _trace_all_labels(mask_i, _NEXT_DIR)
    color_lut = _resolve_colors(colors if colors is not None else default_color,
                                max_label).astype(np.float32, copy=False)

    pos_blocks = []
    nrm_blocks = []
    cid_blocks = []
    inner_blocks = []
    fill_idx = []
    out_idx = []
    vcount = 0

    for lab in range(1, max_label + 1):
        s, e = offsets[lab], offsets[lab + 1]
        if e - s < 3:
            continue
        outer = pts[s:e].astype(np.float32).copy()
        outer[:, 0] += np.float32(x_offset - 0.5)
        outer[:, 1] += np.float32(y_offset - 0.5)
        n = len(outer)

        fwd = np.roll(outer, -1, axis=0) - outer
        n_fwd = np.empty_like(outer)
        n_fwd[:, 0] = -fwd[:, 1]
        n_fwd[:, 1] = fwd[:, 0]
        n_fwd /= np.maximum(np.linalg.norm(n_fwd, axis=1, keepdims=True), 1e-9)
        normal = n_fwd + np.roll(n_fwd, 1, axis=0)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)

        centroid = outer.mean(axis=0).astype(np.float32)

        i_outer = vcount
        i_cent = vcount + n
        i_inner = vcount + n + 1

        pos_blocks.append(outer)
        nrm_blocks.append(normal)
        cid_blocks.append(np.full(n, lab, dtype=np.uint32))
        inner_blocks.append(np.zeros(n, dtype=np.uint32))

        pos_blocks.append(centroid[None, :])
        nrm_blocks.append(np.zeros((1, 2), dtype=np.float32))
        cid_blocks.append(np.array([lab], dtype=np.uint32))
        inner_blocks.append(np.zeros(1, dtype=np.uint32))

        pos_blocks.append(outer)
        nrm_blocks.append(normal)
        cid_blocks.append(np.full(n, lab, dtype=np.uint32))
        inner_blocks.append(np.ones(n, dtype=np.uint32))

        ring = np.arange(n, dtype=np.uint32)
        fill_tri = np.empty((n, 3), dtype=np.uint32)
        fill_tri[:, 0] = i_cent
        fill_tri[:, 1] = i_outer + ring
        fill_tri[:, 2] = i_outer + (ring + 1) % n
        fill_idx.append(fill_tri)

        annu_tri = np.empty((2 * n, 3), dtype=np.uint32)
        annu_tri[0::2, 0] = i_outer + ring
        annu_tri[0::2, 1] = i_outer + (ring + 1) % n
        annu_tri[0::2, 2] = i_inner + (ring + 1) % n
        annu_tri[1::2, 0] = i_outer + ring
        annu_tri[1::2, 1] = i_inner + (ring + 1) % n
        annu_tri[1::2, 2] = i_inner + ring
        out_idx.append(annu_tri)

        vcount += 2 * n + 1

    if not pos_blocks:
        return CellMesh(
            positions=np.empty((0, 2), dtype=np.float32),
            normals=np.empty((0, 2), dtype=np.float32),
            cell_ids=np.empty(0, dtype=np.uint32),
            is_inner=np.empty(0, dtype=np.uint32),
            fill_indices=np.empty((0, 3), dtype=np.uint32),
            outline_indices=np.empty((0, 3), dtype=np.uint32),
            color_lut=color_lut,
        )

    return CellMesh(
        positions=np.concatenate(pos_blocks, axis=0),
        normals=np.concatenate(nrm_blocks, axis=0),
        cell_ids=np.concatenate(cid_blocks, axis=0),
        is_inner=np.concatenate(inner_blocks, axis=0),
        fill_indices=np.concatenate(fill_idx, axis=0),
        outline_indices=np.concatenate(out_idx, axis=0),
        color_lut=color_lut,
    )


__all__ = [
    "vector_contours",
    "vector_contours_marching",
    "cells_to_polygons",
    "cells_to_webgpu_mesh",
    "CellMesh",
    "DEFAULT_SMOOTH_SIGMA",
]
