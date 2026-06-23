"""Canonical interactive linked-axes image-cell emitter.

Emits one ``<svg class="ocd-linked-cell">`` pan/zoom cell consumed by the figure
shell's linked controller (``createLinkedGLLayer`` / ``createLinkedHDRLayer`` in
``ocdkit.io.figure``). Operates purely on plain data — a ``GridLayout`` ``TileInfo``
(display geometry + label) plus a ``link_ctx`` dict (full-res texture source, ROI,
per-cell overlays, per-excitation / mask / HDR metadata) — so it carries no
Scene/domain coupling. Extracted verbatim from the host application's
``key_slices_svg._emit_linked_cell`` to de-duplicate the two near-identical
linked-tile emitters (this one + image_grid's simpler ``link_axes`` path).

``link_ctx`` keys consumed (all optional unless noted):
  tiles (required, list of full-res TileInfo), raster_w/raster_h (required),
  roi_str (required), stream {base,sid,async}, tile_urls, comp (full composite),
  exc {base,sid,n,w,h,total,names,scales|async}, polys [(verts, rgba)],
  outline_scale, outline_use_id, mask_ncolor_url, stream_outline.
"""
from __future__ import annotations

from . import rgba_to_css, png_data_url


def _cell_data_url(arr, raster_format):
    """Encode a linked-cell texture as a data URL, honoring ``raster_format``
    (JXL by default — ~3× smaller than PNG; PNG fallback for Firefox /
    encode errors).

    Lossless (lossy ``distance=1`` is visibly degraded on these images, and the
    ncolor mask tile in particular MUST stay pixel-crisp). Payload is bounded
    instead by ``max_texture_px`` (render at the displayed resolution, not the
    full FOV) — effort=1 for fast encode."""
    if raster_format and str(raster_format).startswith('jxl'):
        try:
            from . import jxl_data_url
            return jxl_data_url(arr, effort=1)        # lossless
        except Exception:
            pass
    return png_data_url(arr)


def emit_linked_cell(svg, i, tile_info, link_ctx, yaxis_w,
                     all_outlined, seg_overlay_mode, outline_px,
                     raster_format='jxl', linked_outlines=True,
                     outline_image_px=1.0, tile_box=True,
                     box_color_css='#888', hdr=False, extra_inner=''):
    """Emit one image-grid tile as a linked-axes pan/zoom cell.

    Display geometry (position + size) comes from the cropped layout
    ``tile_info`` (so a crop keeps the figure compact); the TEXTURE is the
    full uncropped slice (``link_ctx``) and the cell's initial ``viewBox``
    is the crop ROI — pan/zoom out reveals the whole slice. A transparent
    hit-``<rect>`` sibling in outer coords gives a stable bbox under zoom.

    ``tile_box`` draws a fixed frame around each tile in OUTER coords, so it
    stays put while the image pans/zooms inside (this is the grid box, not
    the per-cell segmentation outline).

    Per-cell seg outlines (``linked_outlines`` for Masks / ``all_outlined``
    for every tile) are vector ``<polygon>``s referenced from a shared
    ``<defs>`` group. Their stroke-width is in viewBox = image-pixel units
    (1 image px stays 1 image px — thicker zoomed in, thinner out), via a
    CSS var (``--ocd-osw``) the controller can floor to a screen-px minimum.
    """
    x0, y0, x1, y1 = tile_info.x0, tile_info.y0, tile_info.x1, tile_info.y1
    Wd, Hd = int(x1 - x0), int(y1 - y0)         # display size
    X = yaxis_w + x0
    ft = link_ctx['tiles'][i]                   # matching full-res tile
    _stream = link_ctx.get('stream')            # stream from tile server (no bake)
    _urls = link_ctx.get('tile_urls')           # pre-encoded in parallel (if any)
    _async_tile = False
    if _stream is not None:
        # Stream the tile from the in-kernel tile server (PNG, lazily encoded
        # on first request) instead of baking it into the SVG. The figure
        # appears immediately; tiles fill in as the browser fetches them.
        # RAW float32 (the GL layer colormaps on the GPU). No PNG: encoding a
        # raw-intensity tile to PNG just clips to white, and the GL polls the
        # raw URL itself (retry on 204), so no separate PNG controller is needed.
        url = f"{_stream['base']}/tile/{_stream['sid']}/tile{i}/0?fmt=raw"
        _async_tile = bool(_stream.get('async'))
    elif _urls is not None:
        url = _urls[i]
    else:
        sub = link_ctx['comp'][int(ft.y0):int(ft.y1), int(ft.x0):int(ft.x1)]
        url = _cell_data_url(sub, raster_format)
    # ``data-tile-src`` (raw URL) for the GL layer to poll+colormap; NO
    # ``data-tile-async`` (that PNG-swap controller is redundant now and
    # 500s on raw float32). Non-stream path keeps the baked ``href``.
    _href_attr = (f'data-tile-src="{url}" '
                  if _async_tile else f'href="{url}" ')
    rw, rh = link_ctx['raster_w'], link_ctx['raster_h']
    # rgb_live: the RGB tile carries per-excitation linear-P3 layers so the
    # WebGPU sub-layer composites + toggles them live. The baked ``href`` above
    # stays as the no-WebGPU / SDR fallback. b64 has no '"' or '|', so they're
    # safe in a '|'-joined double-quoted attribute.
    exc_attr = ''
    _exc = link_ctx.get('exc')
    if _exc is not None and tile_info.label == 'RGB':
        # The per-excitation layers are served from the in-kernel tile server
        # (``data-exc-base``/``data-exc-sid``); the browser fetches
        # ``{base}/tile/{sid}/exc{k}/0?fmt=raw`` as raw RGBA bytes. Only small
        # metadata rides inline.
        exc_attr = (
            'data-exc="1" '
            f'data-exc-base="{_exc["base"]}" data-exc-sid="{_exc["sid"]}" '
            f'data-exc-n="{_exc["n"]}" '
            f'data-exc-w="{_exc["w"]}" data-exc-h="{_exc["h"]}" '
            f'data-exc-total="{_exc["total"]}" '
            f'data-exc-names="{",".join(_exc["names"])}" '
        )
        if _exc.get('async'):
            # Layers project on a background thread; the controller retries the
            # raw fetch (204 until ready) and reads ``scales`` from /info meta.
            exc_attr += 'data-exc-async="1" '
        else:
            exc_attr += (
                f'data-exc-scales="{",".join(format(s, ".6g") for s in _exc["scales"])}" ')
    inner = []
    is_mask_tile = (tile_info.label == 'Masks')
    # Deferred (streamed) outline → drawn on the GPU from /outline; tag the cell
    # ``data-outline=1`` and emit NO <use> (the <defs> aren't traced on return).
    _stream_outline = bool(link_ctx.get('stream_outline'))
    gpu_outline = _stream_outline and (is_mask_tile or all_outlined)
    if (linked_outlines and not _stream_outline and tile_info.has_content
            and (is_mask_tile or all_outlined)):
        filled = (seg_overlay_mode == 'filled' and is_mask_tile)
        if filled:
            # Filled Masks: per-cell colored fills (unique to this tile).
            sc = link_ctx['outline_scale']
            for verts, rgba in link_ctx['polys']:
                pts = ' '.join(f'{vx * sc:.2f},{vy * sc:.2f}' for vx, vy in verts)
                inner.append(f'<polygon points="{pts}" fill="{rgba_to_css(rgba)}"/>')
        else:
            # Stroked outlines: reference the shared <defs> group of smooth
            # Bézier <path>s (defined once) so dense all_outlined grids stay
            # small. Rendered NATIVELY now (the paths use non-scaling-stroke for
            # constant screen-px width under zoom) — no GPU outline layer, so we
            # do NOT set ``data-outline`` (which would hide this <use>).
            uid = link_ctx['outline_use_id']
            inner.append(f'<use href="#{uid}" xlink:href="#{uid}"/>')
    # Caller-supplied overlay pasted inside the rotation group (e.g. image_grid's
    # ``seg_polygons`` contour overlay). Pans/zooms with the image.
    if extra_inner:
        inner.append(extra_inner)
    svg.add(
        f'<svg class="ocd-linked-cell" x="{X:.2f}" y="{y0:.2f}" '
        + ('data-outline="1" ' if gpu_outline else '')
        + f'width="{Wd}" height="{Hd}" viewBox="{link_ctx["roi_str"]}" '
        f'preserveAspectRatio="xMidYMid slice" overflow="hidden" '
        # Pin the nested-svg viewport in inline style so a host stylesheet
        # rule like JupyterLab's ``.jp-RenderedHTMLCommon svg {height:auto;
        # max-width:100%}`` (lower priority than inline) cannot rewrite this
        # cell's height to ``auto`` — that inflates the viewport, breaking the
        # ``slice`` mapping + ``overflow:hidden`` clip so the streamed tile
        # <image> bleeds past the cell into the row/spectra below. ``px`` here
        # equals user units in the cell's coordinate context, so geometry is
        # unchanged in conformant hosts; it only defeats the override.
        f'style="width:{Wd}px;height:{Hd}px;max-width:none">'
        # Rotatable content group: the controller sets its transform on
        # multitouch/trackpad rotate, so SVG outlines (and the SDR-fallback
        # <image>) rotate in lockstep with the GPU-rendered image.
        f'<g class="ocd-cell-rot">'
        f'<image x="0" y="0" width="{rw}" height="{rh}" {_href_attr}'
        f'preserveAspectRatio="none" image-rendering="pixelated" '
        # ``data-hdr`` flags this tile's texture as gain-mapped HDR encoded
        # as OETF(hdr_linear) (1.0 = XDR peak). The SvgFigure controller's
        # WebGPU sub-layer re-interprets it (EOTF → ×headroom → OETF) on an
        # rgba16float/extended canvas for adaptive EDR; the WebGL layer
        # below renders the same 8-bit texture as the SDR fallback.
        + ('data-hdr="1" ' if hdr else '')
        + exc_attr
        # ``data-alt-href`` is the ncolor mask raster (pixel-exact) the clickable
        # "Masks" label toggles this tile to (rendered under the outlines).
        + (f'data-alt-href="{link_ctx["mask_ncolor_url"]}" '
           if is_mask_tile and link_ctx.get('mask_ncolor_url') else '')
        + 'pointer-events="none"/>'
        + ''.join(inner) +
        '</g>'
        '</svg>'
        f'<rect class="ocd-linked-cell-hit" x="{X:.2f}" y="{y0:.2f}" '
        f'width="{Wd}" height="{Hd}" fill="transparent" pointer-events="all"/>'
        # Tile box frame: outer coords (fixed) so it doesn't zoom with the
        # image; sits on top of the WebGL/SVG image and the hit rect. Inset by
        # half the stroke so the WHOLE frame sits inside the cell — otherwise the
        # centered stroke's outer half lands on/over the figure viewBox edge and
        # gets clipped (the "cut-off tile boundaries").
        + (f'<rect class="ocd-tile-box" x="{X + outline_px/2.0:.2f}" '
           f'y="{y0 + outline_px/2.0:.2f}" '
           f'width="{max(0.0, Wd - outline_px):.2f}" '
           f'height="{max(0.0, Hd - outline_px):.2f}" fill="none" '
           f'stroke="{box_color_css}" stroke-width="{outline_px}" '
           f'pointer-events="none"/>'
           if tile_box else '')
    )


# ---------------------------------------------------------------------------
# Generic linked-layer tile prep (scene-free). Extracted from the host application's
# ``key_slices_svg._prepare_linked_layer`` so image_grid's large/label tiles
# reuse the SAME machinery as the scene key-slice grid: a block-mean texture
# cap, screen-px vector seg outlines, and the ncolor mask raster. Operates on
# plain numpy arrays — no Scene coupling.
# ---------------------------------------------------------------------------

def block_mean_cap(img, max_texture_px):
    """Integer block-mean (area) downsample any tile whose longest side exceeds
    ``max_texture_px``; otherwise return ``img`` unchanged.

    Block-mean (NOT stride ``img[::f]``): stride keeps the top-left pixel of
    each f×f block, so a cell edge in the displayed image snaps to the kept
    pixel while the full-res vector outline traces the true edge — up to half a
    texture-pixel of drift. Block-mean maps orig σ → texture σ/f *continuously*,
    matching the outline's continuous scaling (fx = raster_w/seg_w), so the two
    register. Full-res 2000-px FOVs inlined as base64 also produce tens of MB of
    SVG (and overflow the XML parse buffer); the cap keeps the payload sane.
    """
    import numpy as np
    if img is None:
        return None
    if not (max_texture_px and max_texture_px > 0):
        return img
    h, w = img.shape[:2]
    f = int(np.ceil(max(h, w) / float(max_texture_px)))
    if f <= 1:
        return img
    hc, wc = (h // f) * f, (w // f) * f
    t = img[:hc, :wc]
    if t.ndim == 3:
        t = t.reshape(hc // f, f, wc // f, f, t.shape[2]).mean(axis=(1, 3))
    else:
        t = t.reshape(hc // f, f, wc // f, f).mean(axis=(1, 3))
    return t.astype(img.dtype, copy=False)


def polys_to_smooth_paths(polys, scale, *, simplify_px=1.5):
    """Closed Catmull-Rom (→ cubic Bézier) smooth-outline ``<path>`` strings,
    one per polygon, in texture coords (verts × ``scale``).

    Each boundary is decimated then drawn as a CLOSED cubic Bézier spline
    through the survivors — smooth curves instead of pixel-stair corners,
    ~10× fewer string ops, much smaller payload. The group draws them with
    ``vector-effect:non-scaling-stroke`` so the line stays a constant screen px
    under zoom (no GPU re-render layer needed).
    """
    import numpy as np
    target_pts = max(8, int(round(24 / max(simplify_px, 1.0) * 1.5)))

    def _smooth_closed(p, w):
        # wrap-around moving average → removes marching-squares pixel stairs
        # before the spline, so the Catmull-Rom curve hugs the true boundary.
        if w < 2 or len(p) < 2 * w:
            return p
        k = np.ones(w) / w
        pad = np.vstack([p[-w:], p, p[:w]])
        sx = np.convolve(pad[:, 0], k, mode='same')[w:-w]
        sy = np.convolve(pad[:, 1], k, mode='same')[w:-w]
        return np.column_stack([sx, sy])

    parts = []
    for verts, rgba in polys:
        v = np.asarray(verts, np.float64) * scale
        if len(v) > target_pts:
            stride = int(np.ceil(len(v) / target_pts))
            v = _smooth_closed(v, stride)              # de-stair at decimation scale
            v = v[::stride]
        n = len(v)
        if n < 3:
            continue
        # Catmull-Rom tangents → cubic Bézier control points around the closed
        # loop (neighbours via roll): c1 leaves Pi toward Pi+1, c2 enters Pi+1.
        pm1 = np.roll(v, 1, axis=0); pp1 = np.roll(v, -1, axis=0); pp2 = np.roll(v, -2, axis=0)
        c1 = v + (pp1 - pm1) / 6.0
        c2 = pp1 - (pp2 - v) / 6.0
        seg = np.concatenate([c1, c2, pp1], axis=1)     # (n,6): ctrl1, ctrl2, end
        d = (f'M{v[0, 0]:.1f},{v[0, 1]:.1f}C'
             + ' '.join(f'{r[0]:.1f},{r[1]:.1f} {r[2]:.1f},{r[3]:.1f} {r[4]:.1f},{r[5]:.1f}'
                        for r in seg) + 'Z')
        # vector-effect is per-element (NOT inherited from the <g>), so set it on
        # each path → the stroke stays a constant screen px regardless of zoom.
        parts.append(f'<path d="{d}" stroke="{rgba_to_css(rgba)}" '
                     f'vector-effect="non-scaling-stroke"/>')
    return parts


def _magma_cmap():
    # Match the host application's keyslice colormap source EXACTLY (cmap's magma →
    # matplotlib, not matplotlib's own magma — the 256-color tables differ
    # slightly) so the extracted mask/outline rasters stay byte-identical.
    from cmap import Colormap
    return Colormap('magma').to_matplotlib()


def build_outline_defs(seg_full, scale, *, seg_overlay_mode='outline', uid='',
                       outline_image_px=1.0, outline_simplify_px=0.5):
    """Build the shared ``<defs>`` smooth-outline group for a full seg array.

    Returns ``(outline_defs, outline_use_id, outline_rgba, polys)``. ``polys``
    are the cells_to_polygons output (texture-coord scaling applied later via
    ``scale`` inside :func:`polys_to_smooth_paths`); ``outline_defs`` is the
    ready-to-inject ``<defs>`` string and ``outline_use_id`` its group id.
    """
    import numpy as np
    from .contour import cells_to_polygons
    if seg_overlay_mode == 'filled':
        import ncolor
        ncolored = ncolor.label(seg_full, expand=True).astype(np.int32)
        n_max = max(1, int(ncolored.max()))
        cmap_mpl = _magma_cmap()
        t_lo, t_hi = 0.20, 0.90
        color_spec = {}
        for g in range(1, n_max + 1):
            t = t_lo + ((g - 0.5) / n_max) * (t_hi - t_lo)
            members = sorted({int(L) for L in np.unique(seg_full[ncolored == g]) if L > 0})
            if members:
                color_spec[tuple(cmap_mpl(t))] = members
        polys = cells_to_polygons(seg_full, colors=color_spec,
                                  x_offset=0.5, y_offset=0.5, smooth_sigma=0)
    else:
        polys = cells_to_polygons(seg_full, default_color=(0.75, 0.75, 0.75, 1.0),
                                  x_offset=0.5, y_offset=0.5, smooth_sigma=0)
    base = max(0.0, float(outline_image_px))
    use_id = f'ocd-outlines-{uid}'
    _simp_px = outline_simplify_px if (outline_simplify_px and outline_simplify_px > 0) else 1.5
    _path_parts = polys_to_smooth_paths(polys, scale, simplify_px=_simp_px)
    outline_defs = (f'<defs><g id="{use_id}" fill="none" '
                    f'vector-effect="non-scaling-stroke" '
                    f'style="stroke-width:var(--ocd-osw,{max(base, 1.0):.3f})">'
                    + ''.join(_path_parts) + '</g></defs>')
    _oc = polys[0][1] if polys else (0.75, 0.75, 0.75, 1.0)
    outline_rgba = (float(_oc[0]), float(_oc[1]), float(_oc[2]),
                    float(_oc[3]) if len(_oc) > 3 else 1.0)
    return outline_defs, use_id, outline_rgba, polys


def build_mask_ncolor_url(seg_full, raster_format):
    """Build the ncolor mask raster data URL (every cell filled by its ncolor
    group via the magma ramp) for the clickable "Masks" toggle. Returns ``None``
    on any failure."""
    import numpy as np
    try:
        import ncolor as _ncolor
        _ncl = _ncolor.label(seg_full, expand=True).astype(np.int32)
        _nmax = max(1, int(_ncl.max()))
        _cm = _magma_cmap()
        _lut = np.zeros((_nmax + 1, 3), dtype=np.float32)   # 0 = bg (black)
        for _g in range(1, _nmax + 1):
            _lut[_g] = _cm(0.20 + ((_g - 0.5) / _nmax) * 0.70)[:3]
        return _cell_data_url(_lut[_ncl], raster_format)
    except Exception:
        return None
