"""Tier-2 prototype: single-pass marching-squares boundary tracer for label
matrices.

Bypasses the entire current pipeline:
    find_boundaries -> boundary_to_masks -> get_neighbors ->
        masks_to_affinity -> parametrize_contours

and replaces it with one numba kernel that walks the dual lattice (corners
between pixels) directly, emitting per-label closed polylines at sub-pixel
coordinates.

Compared in this script:
    - Current pipeline:  ocdkit.plot.contour.vector_contours
    - Tier 1 prototype:  fast walker + Gaussian smooth + LineCollection
    - Tier 2 prototype:  marching-squares numba kernel + Gaussian + LineCollection

Run:
    python scripts/bench_vector_contours_tier2.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.patches as mpatches  # noqa: F401
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from matplotlib.collections import PathCollection, PolyCollection

# Default Gaussian smoothing length, in pixels of contour arclength.
# 2*sqrt(2) ≈ 2.83 sits at the natural scale where 2-pixel zigzags from the
# marching-squares staircase are averaged out, while cell-shape harmonics with
# wavelength larger than the cell radius are preserved. Raise toward 3*sqrt(2)
# for softer outlines, lower toward sqrt(2) for pixel-honest fidelity.
DEFAULT_SMOOTH_SIGMA = 2.0 * np.sqrt(2.0)
from numba import njit
from scipy.ndimage import gaussian_filter1d

from ocdkit.plot.contour import vector_contours

# Tier 1 prototype (already validated)
from bench_vector_contours import vector_contours_fast


# ---------------------------------------------------------------------------
# Marching-squares lookup table.
#
# Cell pattern at a lattice corner (TL, TR, BL, BR) where each bit is 1 iff
# that cell == the active label. Bits: TL=8, TR=4, BL=2, BR=1.
#
# Edges around a corner: N (between TL,TR), E (TR,BR), S (BL,BR), W (TL,BL).
# An edge is part of L's boundary iff its two cells have different "is L"
# status. For non-saddle patterns the corner has exactly 0 or 2 such edges,
# and the 2-edge case pairs uniquely.
#
# `_NEXT_DIR[pattern, entry_edge] = exit_direction` (and exit_direction also
# encodes the direction of motion on leaving). Direction codes: N=0, E=1,
# S=2, W=3. -1 means "should not occur" (entry edge not part of pattern).
#
# Saddles (patterns 6 and 9) split into two separate crossings: pattern 6
# (TR,BL diagonal) pairs N<->E (around TR) and S<->W (around BL); pattern 9
# (TL,BR diagonal) pairs N<->W (around TL) and S<->E (around BR).
# ---------------------------------------------------------------------------

_NEXT_DIR = np.array([
    [-1, -1, -1, -1],   # 0  -- (no L cells)
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

    # Worst-case allocation: each pixel contributes <= 4 corners along its
    # boundary. For typical cell masks this is far oversized.
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

        # Bound the walk: a single connected component's boundary length is
        # < 4 * pixel_count. Use H*W as a hard upper bound to prevent runaway.
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


# --- Stroke alignment note -------------------------------------------------
# matplotlib's LineCollection (and PolyCollection's edge stroke) draws lines
# CENTERED on the path: a stroke of width L extends L/2 on each side. SVG 2,
# Skia, and Direct2D have a "stroke-alignment" property (inner | center |
# outer); matplotlib does not. The equivalent here is to inset the path by
# L/2 along the inward normal — the resulting stroke's OUTER edge then lies
# exactly on the polygon's boundary, matching the filled polygon's extent.
#
# This is independent of any smoothing-induced shrinkage. With offset = L/2,
# the outline's outer extent matches the filled polygon edge-for-edge for
# free; adjacent cells' outlines have the same gap structure as adjacent
# filled cells. No need to measure or compensate for MCF shrinkage.


def _offset_closed_polygon(P, distance):
    """Shift each vertex of a closed polygon along the local inward normal.

    The Tier-2 walk traverses each label's boundary with the cell interior on
    the right of the direction of motion (in screen coords with y-down). For
    a forward tangent (tx, ty), the right-perpendicular pointing into the
    cell is (-ty, tx). At each vertex we average the inward normal of the
    incoming and outgoing edges, normalize, then shift by `distance`.

    distance > 0  :: inward (toward cell interior — contour sits inside the
                     outermost ring of pixels)
    distance < 0  :: outward
    distance == 0 :: contour stays on the geometric cell boundary

    distance == 0.5 puts the contour through the centers of the outermost
    ring of pixels (visually similar to the current splprep pipeline, but
    without the up-left B-spline closure artifact).
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
        return lut  # already filled with this single color above

    # array-like indexed by label
    arr = np.asarray(colors, dtype=object) if not isinstance(colors, np.ndarray) \
          else colors
    if arr.ndim == 1 and arr.dtype != object and arr.shape[0] >= max_label + 1:
        # already RGBA-shaped row vectors? unlikely 1D — treat as scalar specs
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


def _is_color_spec(x):
    """Quick check: does matplotlib think `x` is a single color?"""
    from matplotlib.colors import to_rgba
    try:
        to_rgba(x)
        return True
    except (ValueError, TypeError):
        return False


def _kiss_offset(fig, ax, linewidth_pt):
    """Inward inset (in image pixels) implementing matplotlib-equivalent
    "stroke-alignment: inner". Inset by linewidth/2 along the inward
    normal so the stroke's OUTER extent lands exactly on the contour
    (same edge as the filled polygon would have). Outline visual extent
    matches the filled-cell visual extent for free.
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


def vector_contours_marching(fig, ax, mask, smooth_sigma=DEFAULT_SMOOTH_SIGMA,
                             offset=None, colors=None, color='r', linewidth=1,
                             fill=False, edge=True, alpha=None,
                             x_offset=0, y_offset=0, zorder=1):
    """Tier-2 prototype: marching-squares numba kernel + Gaussian smooth +
    one LineCollection.

    No padding, no affinity graph, no boundary_to_masks.

    Parameters
    ----------
    offset : float or None
        Inward shift (in image pixels) of the contour along the local normal,
        applied after Gaussian smoothing.
        - None (default) : auto-compute so adjacent cell strokes just touch
                           at the shared boundary (kiss, no overlap). Uses
                           figure DPI and current axes layout to convert
                           ``linewidth`` from points to image pixels. For
                           ``fill=True`` the auto value is 0 (filled regions
                           tile correctly without insetting).
        - 0.0  : contour sits on the geometric cell boundary.
        - 0.5  : contour passes through the centers of the outermost pixels
                 (the convention the current splprep pipeline targets).
        - <0   : push the contour outside the cell.
    smooth_sigma : float
        Periodic Gaussian smoothing applied to (x, y) along the contour.
        Units are pixels of arclength (1 vertex = 1 pixel of arclength for
        marching-squares output). Default is 2*sqrt(2) ≈ 2.83.
        - sqrt(2) ≈ 1.41 :: minimum to suppress the pixel staircase
        - 2*sqrt(2) ≈ 2.83 :: removes 2-pixel zigzags, preserves cell shape (default)
        - 3*sqrt(2) ≈ 4.24 :: smoother / more stylized
    colors : None, color spec, dict, array-like, or callable
        Per-cell color control. See _resolve_colors. If None, falls back to
        `color` (kept for backwards-compatible single-color use).
    fill : bool, default False
        If True, render filled polygons (PolyCollection) — each cell is a
        filled region in its assigned color. If False, only outlines.
    edge : bool, default True
        Whether to draw the outline (only meaningful when fill=True; if both
        fill and edge are True, the cell is filled and stroked).
    alpha : float or None
        Opacity applied uniformly to faces and edges. None = use color alpha.
    """
    if mask.dtype != np.int32:
        mask_i = mask.astype(np.int32)
    else:
        mask_i = mask

    pts, offsets, max_label = _trace_all_labels(mask_i, _NEXT_DIR)

    # Build the per-label color LUT (None falls through to legacy `color`).
    color_lut = _resolve_colors(colors if colors is not None else color,
                                max_label)

    # For outlines: render as a filled annulus (ring) so the OUTER edge is a
    # hard polygon boundary at the same position as the filled cell. The
    # ring's width in image-pixel units = linewidth in points converted via
    # _kiss_offset (which gives lw/2; multiply by 2 for the full width).
    # No `offset` is needed for the outer ring — it sits on the polygon edge
    # (matching fill); the inner ring is offset inward by lw to form the ring.
    if not fill:
        ring_width = 2.0 * _kiss_offset(fig, ax, linewidth)
    else:
        ring_width = 0.0

    # Lattice corner (i, j) is the TL of pixel (i, j). With matplotlib's
    # default imshow (pixel (r, c) centered at (c, r), spanning (c-0.5,
    # r-0.5) to (c+0.5, r+0.5)), the TL corner of pixel (r, c) sits at
    # display coordinate (c - 0.5, r - 0.5). So we subtract 0.5 to put
    # the contour exactly on the geometric cell boundary.
    outer_polys = []           # smoothed polygons on the cell boundary
    annulus_paths = []          # used only for outline (fill=False) mode
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
        # Apply the (legacy) offset only when explicitly requested AND not
        # using the annulus-based outline path. The annulus's outer edge
        # always sits on the polygon boundary by construction.
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
        # Filled-annulus rendering: outer edge of the ring sits exactly on
        # the polygon boundary (matches the filled cell). No AA fade between
        # the visible stroke outer edge and the underlying fill.
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
# WebGPU mesh export
#
# Same geometry as the matplotlib path above, but emitted as triangulated
# meshes ready to upload to WebGPU vertex/index buffers.
#
# Two meshes per call:
#   - fill mesh:     polygon fans (one fan per cell, from centroid)
#                    → triangles tile each cell's interior
#   - annulus mesh:  strip of triangles between outer and inner contours
#                    → triangles tile each cell's "outline ring"
#
# Both meshes share the same outer polygon vertices (so the fill's outer
# boundary and the annulus's outer boundary are pixel-identical), exactly
# matching the matplotlib annulus + PolyCollection approach.
#
# Output format follows WebGPU buffer conventions:
#   positions: float32  (V, 2)
#   indices:   uint32   (T, 3)
#   cell_id:   uint32   (V,)         — per-vertex cell index for shader lookup
#   colors:    float32  (max_label+1, 4) — RGBA LUT, indexed by cell_id
# ---------------------------------------------------------------------------

class CellMesh:
    """Triangulated geometry ready for upload to a WebGPU pipeline.

    Vertex layout (one buffer, used by both fill and outline pipelines):
        position : float32x2  raw marching-squares vertex (no smoothing, no
                              offset applied — both happen in shader)
        normal   : float32x2  unit inward normal (averaged forward + previous
                              edge normals at each vertex)
        cell_id  : uint32     per-vertex cell index (look up colors[cell_id])
        is_inner : uint32     0 = outer ring, 1 = inner ring (annulus only).
                              For fill triangles, always 0.

    Vertex shader (sketch):
        let smoothed = smooth(position, neighbors, sigma);   // optional
        let pos      = smoothed + f32(is_inner) * ring_width * normal;
        return view_proj * vec4(pos, 0.0, 1.0);

    Fragment shader looks up color = color_lut[cell_id] (storage buffer).

    Attributes
    ----------
    positions  : float32 (V, 2)
    normals    : float32 (V, 2)
    cell_ids   : uint32  (V,)
    is_inner   : uint32  (V,)             // 0 for fill verts, 0/1 for ring
    fill_indices    : uint32 (T_f, 3)     // centroid-fan triangulation
    outline_indices : uint32 (T_o, 3)     // annulus strip triangulation
    color_lut  : float32 (max_label+1, 4) // RGBA
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
    shader can:
      - apply periodic smoothing as a vertex transform reading neighbor
        positions from a storage buffer
      - synthesize the annulus inner ring as ``position + ring_width * normal``
        for vertices flagged ``is_inner = 1``

    Per-vertex normals are precomputed (averaged inward edge normals) so the
    shader's annulus offset is one MAD per vertex.

    Returns
    -------
    CellMesh
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

        # Per-vertex inward unit normal (average of incoming and outgoing
        # edge normals). For axis-aligned marching-squares output this is
        # already unit-length; the explicit normalize handles diagonals.
        fwd = np.roll(outer, -1, axis=0) - outer
        n_fwd = np.empty_like(outer)
        n_fwd[:, 0] = -fwd[:, 1]
        n_fwd[:, 1] = fwd[:, 0]
        n_fwd /= np.maximum(np.linalg.norm(n_fwd, axis=1, keepdims=True), 1e-9)
        normal = n_fwd + np.roll(n_fwd, 1, axis=0)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)

        centroid = outer.mean(axis=0).astype(np.float32)

        # Layout for this cell: outer (n) + centroid (1) + inner (n) = 2n + 1
        # Indices:
        #   i_outer   = vcount
        #   i_cent    = vcount + n
        #   i_inner   = vcount + n + 1
        i_outer = vcount
        i_cent = vcount + n
        i_inner = vcount + n + 1

        # Outer vertices (used by fill fan AND annulus outer)
        pos_blocks.append(outer)
        nrm_blocks.append(normal)
        cid_blocks.append(np.full(n, lab, dtype=np.uint32))
        inner_blocks.append(np.zeros(n, dtype=np.uint32))

        # Centroid (fan apex; no normal needed but pad to 0)
        pos_blocks.append(centroid[None, :])
        nrm_blocks.append(np.zeros((1, 2), dtype=np.float32))
        cid_blocks.append(np.array([lab], dtype=np.uint32))
        inner_blocks.append(np.zeros(1, dtype=np.uint32))

        # Inner vertices (same position + normal as outer, but is_inner = 1
        # so the vertex shader translates them by ring_width * normal). This
        # avoids storing a separate offset polygon — shader does the math.
        pos_blocks.append(outer)
        nrm_blocks.append(normal)
        cid_blocks.append(np.full(n, lab, dtype=np.uint32))
        inner_blocks.append(np.ones(n, dtype=np.uint32))

        # Fill: centroid-fan triangulation
        ring = np.arange(n, dtype=np.uint32)
        fill_tri = np.empty((n, 3), dtype=np.uint32)
        fill_tri[:, 0] = i_cent
        fill_tri[:, 1] = i_outer + ring
        fill_tri[:, 2] = i_outer + (ring + 1) % n
        fill_idx.append(fill_tri)

        # Annulus: strip triangulation between outer (i_outer + i) and
        # inner (i_inner + i)
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


# ---------------------------------------------------------------------------
# Benchmark + figure
# ---------------------------------------------------------------------------

def _time_call(fn, repeat=3):
    fn()  # warm-up (incl numba compile)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts), ts


def main():
    mask = skimage.io.imread('/Volumes/DataDrive/ncolor/test_files/example.png')
    print(f"mask shape={mask.shape} n_labels={len(np.unique(mask)) - 1}")

    def run_current():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        vector_contours(fig, ax, mask, smooth_factor=5, color='r', linewidth=1.0)
        ax.set_axis_off()
        plt.close(fig)

    def run_tier1():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        vector_contours_fast(fig, ax, mask, smooth_sigma=2.0, color='r', linewidth=1.0)
        ax.set_axis_off()
        plt.close(fig)

    def run_tier2():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        vector_contours_marching(fig, ax, mask, smooth_sigma=2.0, color='r', linewidth=1.0)
        ax.set_axis_off()
        plt.close(fig)

    print("\nBenchmarking (best of 3) ...")
    cur_best, _ = _time_call(run_current, repeat=3)
    print(f"  current : {cur_best*1000:6.1f} ms")
    t1_best, _ = _time_call(run_tier1, repeat=3)
    print(f"  tier 1  : {t1_best*1000:6.1f} ms  ({cur_best/t1_best:.2f}x)")
    t2_best, _ = _time_call(run_tier2, repeat=3)
    print(f"  tier 2  : {t2_best*1000:6.1f} ms  ({cur_best/t2_best:.2f}x)")

    # 3-way figure with zoom row
    H, W = mask.shape
    crop_y = slice(H // 3, H // 3 + 80)
    crop_x = slice(W // 3, W // 3 + 80)

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    for a in axes[0]:
        a.imshow(mask, cmap='gray', interpolation='nearest')
        a.set_axis_off()
    vector_contours(fig, axes[0, 0], mask, smooth_factor=5, color='r', linewidth=1.0)
    vector_contours_fast(fig, axes[0, 1], mask, smooth_sigma=2.0, color='r', linewidth=1.0)
    vector_contours_marching(fig, axes[0, 2], mask, smooth_sigma=2.0, color='r', linewidth=1.0)
    axes[0, 0].set_title(f"Current vector_contours\n{cur_best*1000:.1f} ms", fontsize=11)
    axes[0, 1].set_title(f"Tier 1 (graph + Gaussian + LineCollection)\n"
                         f"{t1_best*1000:.1f} ms  ({cur_best/t1_best:.2f}x)", fontsize=11)
    axes[0, 2].set_title(f"Tier 2 (marching squares numba kernel)\n"
                         f"{t2_best*1000:.1f} ms  ({cur_best/t2_best:.2f}x)", fontsize=11)

    for a in axes[1]:
        a.imshow(mask[crop_y, crop_x], cmap='gray', interpolation='nearest',
                 extent=(crop_x.start, crop_x.stop, crop_y.stop, crop_y.start))
        a.set_xlim(crop_x.start, crop_x.stop)
        a.set_ylim(crop_y.stop, crop_y.start)
        a.set_axis_off()
        a.set_title("zoomed crop", fontsize=10)
    vector_contours(fig, axes[1, 0], mask, smooth_factor=5, color='r', linewidth=1.5)
    vector_contours_fast(fig, axes[1, 1], mask, smooth_sigma=2.0, color='r', linewidth=1.5)
    vector_contours_marching(fig, axes[1, 2], mask, smooth_sigma=2.0, color='r', linewidth=1.5)

    fig.tight_layout()
    out = Path('/Volumes/DataDrive/ocdkit/scripts/bench_vector_contours_tier2.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nsaved: {out}")


if __name__ == '__main__':
    main()
