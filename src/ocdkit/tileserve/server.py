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

    def declare(self, label: str):
        if label not in self._declared:
            self._declared.append(label)

    def add_layer(self, label: str, arr: np.ndarray, meta: dict | None = None):
        self.declare(label)
        # label-like layers set meta['downsample']='nearest' so coarse levels
        # keep exact values instead of blending across edges.
        _ds = (meta or {}).get("downsample", "mean")
        self._pyr[label] = image_pyramid(np.asarray(arr), self.n_levels, mode=_ds)
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
        return [list(d) for d in self._dims]      # same for every layer (FOV)

    def n_level(self, label: str) -> int:
        return len(self._dims)

    def level(self, label: str, level: int):
        """Return ``(lh, lw, ndarray)`` for a level, or None if not filled."""
        pyr = self._pyr.get(label)
        if not pyr:
            return None
        level = max(0, min(int(level), len(pyr) - 1))
        return pyr[level]


# ───────────────────────────── registry ─────────────────────────────────

_SOURCES: dict[str, TileSource] = {}
_LOCK = threading.Lock()
_LAZY: dict = {}          # (sid, label) -> producer() -> (arr, meta)
_LAZY_STARTED: dict = {}  # (sid, label) -> True once compute kicked off


def register(width: int, height: int, layers: dict[str, np.ndarray],
             n_levels: int = 5) -> str:
    """Register a source's full-res layers (handed in as ready arrays)."""
    src = TileSource(width, height, n_levels=n_levels)
    for label, arr in layers.items():
        if arr is not None:
            src.add_layer(label, arr)
    sid = uuid.uuid4().hex[:12]
    with _LOCK:
        _SOURCES[sid] = src
    return sid


def register_pending(width: int, height: int, labels, n_levels: int = 5,
                     grid=None) -> str:
    """Declare a source's layers (dims known) with NO data yet; returns sid.

    The viewer can lay out + request tiles immediately; data fills in
    asynchronously via :func:`fill`. ``grid`` is the optional 2D tile layout.
    """
    src = TileSource(width, height, n_levels=n_levels)
    for label in labels:
        src.declare(label)
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

    @app.get("/tile/{sid}/{label}/{level}")
    async def tile(sid: str, label: str, level: int, fmt: str = "raw"):
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
                return Response(status_code=204)
            return JSONResponse({"error": f"no layer {label}"}, status_code=404)
        lh, lw, arr = lv
        body, media = _encode_level(lh, lw, arr, fmt)
        ch = 1 if arr.ndim == 2 else arr.shape[2]
        m = src.meta.get(label, {"mode": "rgb"})
        return Response(content=body, media_type=media, headers={
            "X-Level-Width": str(lw), "X-Level-Height": str(lh),
            "X-Level": str(max(0, min(level, src.n_level(label) - 1))),
            "X-Channels": str(ch), "X-Dtype": str(arr.dtype),
            "X-Mode": str(m.get("mode", "rgb")),
            "X-Lo": repr(float(m.get("lo", 0.0))),
            "X-Hi": repr(float(m.get("hi", 1.0))),
            "X-Kind": str(m.get("kind", "reduction")),
            "X-Bitmax": repr(float(m.get("bit_max", 1.0))),
            "Cache-Control": "no-store",
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
        import time
        import socket
        import uvicorn

        port = _pick_port()
        config = uvicorn.Config(make_app(), host="127.0.0.1", port=port,
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
