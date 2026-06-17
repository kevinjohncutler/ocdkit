"""Generic in-kernel tile server for zoomable, GPU-colormapped linked viewers.

A ``TileSource`` is an in-memory set of named layer *pyramids* (built once via
:mod:`ocdkit.plot.pyramid`) plus opaque named *attachments* (vector overlays,
per-object geometry, panel data — whatever a host wants to stream to its viewer)
and an ``info_extra`` dict merged into ``/info``. The server is a lazy daemon
thread (no separate process); it exposes the GENERIC routes — ``/info``,
``/tile``, ``/attach`` — and a ``register_extension`` hook so a host application
mounts its own routes (the viewer HTML, domain endpoints) on the same app.

Nothing here knows what the layers, labels, grid, or attachments *mean* — that's
the host's job — so the same engine composes into any application.
"""
from __future__ import annotations

import threading
import uuid

import numpy as np

from ..plot.pyramid import image_pyramid, pyramid_dims


# ─────────────────────────────── source ─────────────────────────────────

class TileSource:
    """In-memory, per-source set of layer pyramids + opaque attachments.

    ``layers`` maps a label to a full-res 2D array ((H, W) intensity that the
    client colormaps on the GPU, or (H, W, 3|4) RGB(A)). Level dims are
    deterministic from (H, W), so a viewer can lay out + pick levels before any
    data has filled in (data arrives asynchronously via :func:`fill`).
    """

    def __init__(self, width: int, height: int, n_levels: int = 5):
        self.width = int(width)
        self.height = int(height)
        self.n_levels = n_levels
        self._dims = pyramid_dims(self.height, self.width, n_levels)
        self._declared: list[str] = []
        # Layers stored WITHOUT a pyramid (a single full-res level): label masks
        # that swap in only at full res, big overlays, etc. — the host opts a
        # layer in via ``single_level=`` at register time or ``meta['pyramid']``.
        self._single: set[str] = set()
        self._pyr: dict[str, list] = {}
        # Per-layer display meta the client uses to colormap/normalize on the
        # GPU: {"mode": "intensity"|"rgb", "lo": float, "hi": float, ...}.
        self.meta: dict[str, dict] = {}
        # Optional 2D LAYOUT: list[list[str|None]] (visual rows x cols; None =
        # blank). When set, the viewer arranges cells by this grid; labels index
        # meta/the pyramids exactly like flat layers.
        self.grid: list | None = None
        # Opaque named attachments the host streams to its viewer: name ->
        # (blob bytes, extra response headers, media type). Served by /attach.
        # 204 until present, so the viewer can retry while they build.
        self.attachments: dict[str, tuple] = {}
        # Extra fields merged into /info (e.g. host panel-axis geometry).
        self.info_extra: dict = {}

    def declare(self, label: str, single_level: bool = False):
        if label not in self._declared:
            self._declared.append(label)
        if single_level:
            self._single.add(label)

    def add_layer(self, label: str, arr: np.ndarray, meta: dict | None = None):
        self.declare(label)
        m = meta or {}
        # A layer may opt OUT of the pyramid (build only the full-res level).
        # ``meta['pyramid'] is False`` triggers it too, for hosts that decide at
        # fill time — though ``single_level=`` at register time is preferred so
        # /info reports a single level from the start.
        if m.get("pyramid") is False:
            self._single.add(label)
        # label-like layers set meta['downsample']='nearest' so coarse levels
        # keep exact values instead of blending across edges.
        _ds = m.get("downsample", "mean")
        _nl = 1 if label in self._single else self.n_levels
        self._pyr[label] = image_pyramid(np.asarray(arr), _nl, mode=_ds)
        if meta is not None:
            self.meta[label] = meta

    def attach(self, name: str, blob: bytes, headers: dict | None = None,
               media: str = "application/octet-stream"):
        self.attachments[name] = (blob, dict(headers or {}), media)

    def ready(self, label: str) -> bool:
        return label in self._pyr

    def layers(self) -> list[str]:
        return list(self._declared)

    def level_dims(self, label: str):
        if label in self._single:
            return [[self.height, self.width]]    # single full-res level (no pyramid)
        return [list(d) for d in self._dims]      # same for every multi-level layer (FOV)

    def n_level(self, label: str) -> int:
        return 1 if label in self._single else len(self._dims)

    def level(self, label: str, level: int):
        """Return ``(lh, lw, ndarray)`` for a level, or None if not filled."""
        pyr = self._pyr.get(label)
        if not pyr:
            return None
        level = max(0, min(int(level), len(pyr) - 1))
        return pyr[level]

    def pick_level(self, label: str, target_w: int, target_h: int) -> int:
        """Index of the COARSEST pyramid level whose dims cover
        ``(target_w, target_h)`` — the finest level if none does.

        Levels are indexed coarse(0) → fine. A headless consumer rendering to a
        fixed-size raster should render *down* from the coarsest covering tier
        rather than grabbing the coarse default (level 0), which would be blocky,
        or always the finest, which wastes work."""
        dims = self.level_dims(label)              # [[h, w], ...] coarse → fine
        li = self.n_level(label) - 1
        for j, (lh, lw) in enumerate(dims):
            if lw >= target_w and lh >= target_h:
                li = j
                break
        return li


# ───────────────────────────── registry ─────────────────────────────────

_SOURCES: dict[str, TileSource] = {}
_LOCK = threading.Lock()
_LAZY: dict = {}          # (sid, label) -> producer() -> (arr, meta)
_LAZY_STARTED: dict = {}  # (sid, label) -> True once compute kicked off


def register(width: int, height: int, layers: dict[str, np.ndarray],
             n_levels: int = 5, single_level=None) -> str:
    """Register a source's full-res layers (handed in as ready arrays).

    ``single_level`` is an optional collection of labels stored WITHOUT a
    pyramid (one full-res level) — e.g. label masks that need no coarse levels.
    """
    src = TileSource(width, height, n_levels=n_levels)
    src._single |= set(single_level or ())      # mark before add_layer (sets level count)
    for label, arr in layers.items():
        if arr is not None:
            src.add_layer(label, arr)
    sid = uuid.uuid4().hex[:12]
    with _LOCK:
        _SOURCES[sid] = src
    return sid


def register_pending(width: int, height: int, labels, n_levels: int = 5,
                     grid=None, single_level=None) -> str:
    """Declare a source's layers (dims known) with NO data yet; returns sid.

    The viewer can lay out + request tiles immediately; data fills in
    asynchronously via :func:`fill`. ``grid`` is the optional 2D tile layout.
    ``single_level`` is an optional collection of labels stored WITHOUT a
    pyramid (one full-res level) — /info reports a single level for them, so the
    viewer never requests a coarse tile that doesn't exist.
    """
    src = TileSource(width, height, n_levels=n_levels)
    _single = set(single_level or ())
    for label in labels:
        src.declare(label, single_level=(label in _single))
    src.grid = grid
    sid = uuid.uuid4().hex[:12]
    with _LOCK:
        _SOURCES[sid] = src
    return sid


def fill(sid: str, label: str, arr: np.ndarray, meta: dict | None = None):
    """Attach a projected layer to a pending source (background thread)."""
    src = _SOURCES.get(sid)
    if src is not None and arr is not None:
        src.add_layer(label, arr, meta)


def register_lazy(sid: str, label: str, producer):
    """Register a deferred producer for ``label``; ``producer()`` returns
    ``(arr, meta)`` and runs on the FIRST request for that tile."""
    _LAZY[(sid, label)] = producer


# Content-addressed registration: identical array bytes map to ONE source, so
# callers never need their own per-object cache and can't serve a stale tile
# (different content → different hash → different sid).
_CONTENT_SIDS: dict[str, str] = {}


def register_array(arr: np.ndarray, *, meta: dict | None = None,
                   label: str = "scalar", single_level: bool = True,
                   n_levels: int = 1) -> tuple[str, str]:
    """Register a single ready array as a tile source, deduplicated by content.

    Returns ``(sid, label)``. Build the raw-tile URL as
    ``f"{ensure_server()}/tile/{sid}/{label}/99?fmt=raw"``. Repeated calls with
    identical ``arr`` reuse the same source (no re-fill, stable URL = browser-cache
    friendly); changed content registers a fresh source automatically.
    """
    import hashlib
    a = np.ascontiguousarray(arr)
    key = hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()
    with _LOCK:
        sid = _CONTENT_SIDS.get(key)
        if sid is not None and sid in _SOURCES:
            return sid, label
    h, w = a.shape[:2]
    sid = register_pending(w, h, [label], n_levels=n_levels,
                           single_level=[label] if single_level else None)
    fill(sid, label, a, meta)
    with _LOCK:
        _CONTENT_SIDS[key] = sid
    return sid, label


def attach(sid: str, name: str, blob: bytes, headers: dict | None = None,
           media: str = "application/octet-stream"):
    """Attach an opaque named blob to a source, served by ``/attach/{sid}/{name}``."""
    src = _SOURCES.get(sid)
    if src is not None:
        src.attach(name, blob, headers, media)


def get_source(sid: str) -> "TileSource | None":
    return _SOURCES.get(sid)


def drop(sid: str):
    with _LOCK:
        _SOURCES.pop(sid, None)


def _encode_level(lh: int, lw: int, arr: np.ndarray, fmt: str):
    """Encode a level array to bytes + media type for the wire.

    ``fmt='raw'`` → little-endian raw bytes (uint8 RGBA / float32) the client
    uploads straight to a GPU texture. ``fmt='jxl'`` / ``'png'`` → compressed.
    """
    a = np.ascontiguousarray(arr)
    if fmt == "raw":
        return a.tobytes(), "application/octet-stream"
    if fmt == "jxl":
        try:
            from ..plot.svg import _jxl_bytes  # type: ignore
        except Exception:
            _jxl_bytes = None
        if _jxl_bytes is not None:
            return _jxl_bytes(a), "image/jxl"
    from ..plot.svg import encode_png
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
    return encode_png(a), "image/png"


# ──────────────────────── host extension hooks ──────────────────────────

_EXTENSIONS: list = []     # fn(app) -> mounts host routes on the FastAPI app
_RESET_HOOKS: list = []    # fn() -> clears host-side per-source state on reset


def register_extension(fn):
    """Register ``fn(app)`` to mount host routes (viewer HTML, domain endpoints)
    on the server's FastAPI app. Call before :func:`ensure_server`."""
    _EXTENSIONS.append(fn)


def register_reset_hook(fn):
    """Register ``fn()`` invoked by :func:`reset_server` to clear host state."""
    _RESET_HOOKS.append(fn)


# ─────────────────────── lazy in-kernel server ──────────────────────────

_SERVER = None
_SERVER_LOCK = threading.Lock()


def make_app():
    """Build the FastAPI app with the generic routes + host extensions."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="ocdkit tileserve")
    # Figures display from a different origin than this localhost-only server and
    # fetch tile bytes cross-origin — allow it (no security boundary to protect).
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["*"])

    @app.get("/info/{sid}")
    async def info(sid: str):
        src = get_source(sid)
        if src is None:
            return JSONResponse({"error": "unknown source"}, status_code=404)
        out = {
            "width": src.width, "height": src.height,
            "layers": {lbl: src.level_dims(lbl) for lbl in src.layers()},
            "meta": {lbl: src.meta.get(lbl, {"mode": "rgb"}) for lbl in src.layers()},
            "grid": src.grid,
        }
        out.update(src.info_extra)
        return JSONResponse(out)

    @app.get("/layout/{sid}")
    async def layout(sid: str, w: float = 1000.0, label_pos: str = "top_middle"):
        """Width-driven figure geometry (cell rects, panel box, canvas/full
        height) for container width ``w`` — the single source of truth shared by
        the browser viewer and the headless compositor (see :mod:`.layout`)."""
        src = get_source(sid)
        if src is None:
            return JSONResponse({"error": "unknown source"}, status_code=404)
        from .layout import compute_layout
        ie = getattr(src, "info_extra", {}) or {}
        panel_axes = ie.get("panel_axes") or ie.get("spectra_axes")
        layers = {lbl: None for lbl in src.layers()}
        return JSONResponse(compute_layout(src.grid, layers, panel_axes, w,
                                           label_pos=label_pos))

    @app.get("/tile/{sid}/{label}/{level}")
    async def tile(sid: str, label: str, level: int, fmt: str = "raw", f32: int = 0,
                   crop: str = "", rgbf16: int = 0):
        src = get_source(sid)
        if src is None:
            return JSONResponse({"error": "unknown source"}, status_code=404)
        lv = src.level(label, level)
        if lv is None:
            # declared but not filled → 204 (retry); unknown layer → 404.
            if label in src.layers():
                _key = (sid, label)
                _prod = _LAZY.get(_key)
                if _prod is not None:
                    with _LOCK:
                        _go = not _LAZY_STARTED.get(_key)
                        if _go:
                            _LAZY_STARTED[_key] = True
                    if _go:
                        def _run(_k=_key, _p=_prod):
                            try:
                                _arr, _meta = _p()
                                fill(_k[0], _k[1], _arr, _meta)
                            except Exception as _e:
                                print("[tileserve lazy]", _k[1], _e)
                        threading.Thread(target=_run, daemon=True,
                                         name=f"lazy-{label}").start()
                return Response(status_code=204, headers={"Cache-Control": "no-store"})
            return JSONResponse({"error": f"no layer {label}"}, status_code=404)
        lh, lw, arr = lv
        m = src.meta.get(label, {"mode": "rgb"})
        # ?crop=x0,y0,x1,y1 (FOV-norm [0,1]) → serve ONLY that sub-rect of the level,
        # snapped to pixel edges. Lets a zoomed-in / snapped view fetch just the
        # region in frame at full res instead of the whole-FOV tile. The actual
        # served rect (after snapping + clamping) rides back in X-Crop so the client
        # can place it; absent/invalid crop = the full level (unchanged behaviour).
        croprect = "0,0,1,1"
        if crop:
            try:
                cx0, cy0, cx1, cy1 = (float(v) for v in crop.split(","))
                px0 = max(0, min(lw, int(round(cx0 * lw)))); px1 = max(0, min(lw, int(round(cx1 * lw))))
                py0 = max(0, min(lh, int(round(cy0 * lh)))); py1 = max(0, min(lh, int(round(cy1 * lh))))
                if px1 <= px0: px1 = min(lw, px0 + 1)
                if py1 <= py0: py1 = min(lh, py0 + 1)
                arr = arr[py0:py1, px0:px1]
                croprect = f"{px0/lw:.6f},{py0/lh:.6f},{px1/lw:.6f},{py1/lh:.6f}"
            except Exception:
                croprect = "0,0,1,1"
        # float16 wire for scalar INTENSITY tiles: halves the transfer (the GPU
        # samples R16F as float, so the shader is unchanged). Display error is
        # ~0.01% of the value range and lo/hi ride exact in the headers, so the
        # global-pool normalization is unaffected. raw fmt only; ?f32=1 opts out.
        if (fmt == "raw" and m.get("mode") == "intensity" and arr.ndim == 2
                and arr.dtype == np.float32 and not f32):
            # clip to the float16 range first — raw 16-bit intensity reaches 65535
            # but float16 maxes at 65504, so the top ~31 codes would overflow to
            # inf. Clamping costs ≤0.05% only at the absolute peak (lo/hi exact).
            arr = np.clip(arr, -65504.0, 65504.0).astype("<f2")
        # float16 RGBA wire for FLOAT RGB tiles (opt-in via ?rgbf16=1 — sent by the
        # WebGPU/HDR client). Packs 3->4 (opaque alpha) and casts to <f2 so the
        # client uploads the raw bytes straight to an rgba16float texture (the GPU
        # reads them as half). This removes the client-side per-pixel float->half
        # conversion loop, which was a ~30ms frame hitch each time a zoom crossed
        # an RGB pyramid level. Other clients omit the flag → float32 (unchanged).
        elif (fmt == "raw" and rgbf16 and arr.ndim == 3
                and arr.dtype == np.float32):
            if arr.shape[2] == 3:
                arr = np.concatenate(
                    [arr, np.ones((arr.shape[0], arr.shape[1], 1), arr.dtype)], axis=2)
            arr = np.clip(arr, -65504.0, 65504.0).astype("<f2")
        sh, sw = arr.shape[0], arr.shape[1]          # SERVED dims (cropped if cropping)
        body, media = _encode_level(sh, sw, arr, fmt)
        ch = 1 if arr.ndim == 2 else arr.shape[2]
        return Response(content=body, media_type=media, headers={
            "X-Level-Width": str(sw), "X-Level-Height": str(sh),
            "X-Crop": croprect,                       # FOV-norm rect this tile covers
            "X-Level": str(max(0, min(level, src.n_level(label) - 1))),
            "X-Channels": str(ch), "X-Dtype": str(arr.dtype),
            "X-Mode": str(m.get("mode", "rgb")),
            "X-Lo": repr(float(m.get("lo", 0.0))),
            "X-Hi": repr(float(m.get("hi", 1.0))),
            "X-Kind": str(m.get("kind", "reduction")),
            "X-Bitmax": repr(float(m.get("bit_max", 1.0))),
            "X-Downsample": str(m.get("downsample", "mean")),
            # a (sid, label, level, fmt) tile is IMMUTABLE once filled (sid is a
            # fresh uuid per run) → let the browser cache it, so re-zooming /
            # snapping back to a visited region is instant (no refetch). The 204
            # "not filled yet" path stays no-store so it keeps retrying.
            "Cache-Control": "private, max-age=86400, immutable",
        })

    @app.get("/attach/{sid}/{name}")
    async def attach_get(sid: str, name: str):
        src = get_source(sid)
        a = src.attachments.get(name) if src else None
        if a is None:
            return Response(status_code=204)        # not ready / none — retry
        blob, headers, media = a
        h = {"Cache-Control": "no-store"}
        h.update(headers or {})
        return Response(content=blob, media_type=media, headers=h)

    # ── generic viewer HTML (the zoomable colormap tile grid + LinkedPanel) ──
    from fastapi.responses import HTMLResponse
    from .viewer import grid_html, view_html, viewgl_html

    @app.get("/grid/{sid}", response_class=HTMLResponse)
    async def grid(sid: str, panel: str = "spectra", hdr_gain: str = "auto",
                   hdr_cmap: str = "", ref_re: str = ""):
        # panel: which LinkedPanel the bottom box hosts ("spectra" density lines
        # or "scatter" discrete object points). hdr_gain: "auto" tracks the display
        # headroom, a number forces the HDR multiplier. hdr_cmap: a colormap name
        # lifts intensity tiles through the HDR pipeline. ref_re: optional host
        # regex (token chars only) that lights up matching reference overlays.
        return grid_html(sid, panel=panel, hdr_cmap=hdr_cmap, hdr_gain=hdr_gain,
                         ref_token_re=ref_re)

    @app.get("/view/{sid}", response_class=HTMLResponse)
    async def view(sid: str, layer: str = ""):
        return view_html(sid, layer)

    @app.get("/viewgl/{sid}", response_class=HTMLResponse)
    async def viewgl(sid: str, layer: str = ""):
        return viewgl_html(sid, layer)

    for ext in _EXTENSIONS:
        try:
            ext(app)
        except Exception as e:                       # noqa: BLE001
            print("[tileserve] extension:", e)
    return app


def _free_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# Stable port candidates so the server keeps the SAME origin across kernel
# restarts — browsers partition the GPU shader/pipeline disk cache by origin, so
# a constant port lets a re-run reuse persisted compiled shaders (warm restart).
_STABLE_PORTS = (8137, 8138, 8139, 8140)


def _pick_port() -> int:
    import socket
    for port in _STABLE_PORTS:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            s.close()
            continue
    return _free_port()


def ensure_server() -> str:
    """Lazily start the server on a daemon thread; return its base URL.

    Started once and reused for the life of the kernel. Blocks until the port
    actually accepts connections (a cold first uvicorn import can take seconds,
    and clients fetch immediately), so the first fetch never hits a dead socket.
    """
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is not None:
            return _SERVER["url"]
        import os
        import sys
        import time
        import socket
        import uvicorn

        # The server runs in a daemon THREAD of this (usually a Jupyter kernel)
        # process, so it shares the GIL with the main thread. At the default 5 ms
        # GIL switch interval, a main thread busy in pure-Python starves the
        # server thread for up to a full interval per hand-off → tile-fetch TTFB
        # balloons ~20x (~0.6→13 ms), which is what makes an in-kernel grid's
        # low→high-res upgrade feel laggy vs a remote-routed kernel (measured
        # ~7-14x under a GIL-busy main thread). A shorter switch
        # interval lets the server thread interleave promptly (0.5 ms → ~1.3 ms
        # TTFB, 14x better). It is ~free in the common case: the server thread is
        # BLOCKED on its socket when idle, so it isn't runnable and triggers no
        # extra hand-offs; the small cost (~6% on a pure-Python loop) applies only
        # during a concurrent compute+request burst — exactly when prompt serving
        # matters. Tune/disable via OCDKIT_TILESERVE_SWITCHINTERVAL (0 = leave as-is).
        try:
            _swi = float(os.environ.get("OCDKIT_TILESERVE_SWITCHINTERVAL", "0.0005"))
            if _swi > 0 and sys.getswitchinterval() > _swi:
                sys.setswitchinterval(_swi)
        except Exception:
            pass

        # Bind host: 127.0.0.1 (default, local-only) or 0.0.0.0 to also accept
        # connections from OTHER machines — needed when the notebook is opened on a
        # different host than the kernel (the iframe then targets the kernel host).
        # Opt in via OCDKIT_TILESERVE_HOST=0.0.0.0 (no auth on this server, so only
        # expose it on a trusted network).
        host = os.environ.get("OCDKIT_TILESERVE_HOST", "127.0.0.1")
        port = _pick_port()
        config = uvicorn.Config(make_app(), host=host, port=port,
                                log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True,
                                  name="ocdkit-tileserve")
        thread.start()
        deadline = time.time() + 20.0
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.03)
        url = f"http://127.0.0.1:{port}"
        _SERVER = {"thread": thread, "server": server, "url": url}
        return url


def reset_server():
    """Tear down the running server so the NEXT call starts fresh with current
    code (the daemon thread does not hot-reload). Drops all sources + host
    state via the registered reset hooks."""
    global _SERVER
    with _SERVER_LOCK:
        srv, _SERVER = _SERVER, None
    if srv is not None:
        try:
            srv["server"].should_exit = True
        except Exception:
            pass
    with _LOCK:
        _SOURCES.clear()
        _LAZY.clear()
        _LAZY_STARTED.clear()
    for fn in _RESET_HOOKS:
        try:
            fn()
        except Exception as e:                       # noqa: BLE001
            print("[tileserve] reset hook:", e)
