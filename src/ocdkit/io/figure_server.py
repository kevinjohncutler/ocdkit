"""Pluggable hi-res tile-byte transport for SvgFigure zoom.

Why this exists
---------------
SvgFigure embeds raster tiles inline as ``<image href="data:image/...">``
data URIs. That's perfect for thumbnails (everything self-contained, no
network) but bad if you want full-resolution zoom — embedding a 14×4MB
JXL composite as base64 bloats the notebook output to ~75MB and chokes
the browser.

This module exposes a tiny pluggable registry that hands out URLs the
browser can fetch on demand. The shape of that URL is determined by the
*active* registry — a :class:`FigureSourceRegistry` instance — which
the calling application picks at boot time (or relies on auto-detection
to pick).

Available registries
--------------------
* :class:`LocalHttpRegistry` (default) — runs a stdlib HTTP server on
  ``127.0.0.1`` and hands out absolute ``http://127.0.0.1:PORT/...``
  URLs. Works whenever the browser can reach the kernel's localhost:
  a Jupyter kernel on the same machine as the browser, a script that
  also runs the browser, etc. Fails for remote kernels accessed over
  SSH tunnels (browser's localhost ≠ kernel's localhost).
* :class:`JupyterProxyRegistry` — subclass that rewrites URLs to
  ``/proxy/PORT/...`` (relative to the Jupyter origin) so they survive
  remote-kernel + tunneled JupyterLab setups. Requires the
  ``jupyter-server-proxy`` package installed in the kernel; the proxy
  forwards requests from Jupyter's origin to ``127.0.0.1:PORT``.
* **Your own registry** — implement the :class:`FigureSourceRegistry`
  protocol however your app serves bytes (a dashboard's existing
  HTTP server, a CDN, a Comm-based pipe, …) and install it with
  :func:`set_registry`. Every ``image_grid(...)`` call then routes
  through it.

Auto-detection
--------------
:func:`get_registry` lazily picks a sensible default:

1. If a Jupyter kernel is detected AND ``jupyter-server-proxy`` is
   importable → :class:`JupyterProxyRegistry`.
2. Otherwise → :class:`LocalHttpRegistry`.

Applications that want explicit control call
:func:`ocdkit.plot.setup(registry=...)` (or :func:`set_registry`
directly) before any ``image_grid`` calls.

When the chosen registry produces URLs the browser can't reach, the
SvgFigure JS overlay silently falls back to the embedded thumbnail —
nothing breaks, but the zoomed image is at thumbnail resolution.
"""

from __future__ import annotations

import http.server
import io
import os
import secrets
import socketserver
import threading
from pathlib import Path
from typing import Optional, Protocol
from urllib.parse import urlparse

import numpy as np


# ─── source types ────────────────────────────────────────────────────


class Source:
    """Base for tile-content sources. Subclasses implement
    :meth:`get_bytes` and may override :attr:`content_type`."""

    content_type: str = "application/octet-stream"

    def get_bytes(self) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


_EXT_TO_CTYPE = {
    "jxl": "image/jxl",
    "avif": "image/avif",
    "heic": "image/heic",
    "heif": "image/heif",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
_CTYPE_TO_EXT = {v: k for k, v in _EXT_TO_CTYPE.items()}


class PathSource(Source):
    """Stream a file from disk verbatim — no re-encode."""

    def __init__(self, path):
        self.path = str(path)
        ext = os.path.splitext(self.path)[1].lstrip(".").lower()
        self.content_type = _EXT_TO_CTYPE.get(ext, "application/octet-stream")

    def get_bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()


class NpySliceSource(Source):
    """Lazy in-memory render of one channel from a ``.npy`` spectral stack.

    On the first ``get_bytes()`` call: ``mmap`` the file, slice out the
    requested channel (no full-file load), hand the 2D array to a
    ``render_fn`` that returns encoded image bytes (typically a
    colormap + ``uhdr.encode_native``). Cached for subsequent calls.

    Use to back :func:`image_grid` cells with the spectral channels of
    a scene without preloading or pre-encoding all N channels.

    Parameters
    ----------
    path : str | Path
        Path to the ``.npy`` file.
    channel_index : int
        Which channel to slice. Convention is detected at slice time:
        if the array is 3-D with the channel axis last (``shape[-1] ==
        n_channels``), indexes as ``arr[..., channel_index]``; if
        channel-first (``shape[0] == n_channels``), as
        ``arr[channel_index]``. Either layout works.
    render_fn : Callable[[np.ndarray], bytes]
        Takes the 2D channel array, returns encoded image bytes
        (uhdr / jpeg / png). The function should be deterministic and
        share state across cells (e.g. one shared colormap closure)
        so the registry's dedup key collapses identical-render-fn calls.
    shape : tuple, optional
        ``(H, W)`` of the channel, in source pixels. Pre-computed at
        construction so ``_native_dims`` doesn't have to mmap the file
        just to peek. ``None`` (default) does a one-time header parse.
    content_type : str
        MIME type for the rendered bytes. Default ``"image/jpeg"``
        (uhdr is a JPEG).
    """

    def __init__(self, path, channel_index: int, render_fn,
                 shape: Optional[tuple] = None,
                 content_type: str = "image/jpeg"):
        self.path = str(path)
        self.channel_index = int(channel_index)
        self.render_fn = render_fn
        self.content_type = content_type
        self._cached: Optional[bytes] = None
        if shape is None:
            self.shape = _peek_npy_shape(self.path, self.channel_index)
        else:
            self.shape = (int(shape[0]), int(shape[1]))

    def get_bytes(self) -> bytes:
        if self._cached is not None:
            return self._cached
        arr = np.load(self.path, mmap_mode='r')
        if arr.ndim == 3 and arr.shape[-1] >= self.channel_index + 1 \
                and arr.shape[-1] < arr.shape[0]:
            # Channel-last layout (H, W, C). Common for spectral stacks.
            slc = arr[..., self.channel_index]
        elif arr.ndim == 3:
            # Channel-first layout (C, H, W).
            slc = arr[self.channel_index]
        elif arr.ndim == 2:
            # 2D array — only one "channel" available.
            slc = arr
        else:
            raise ValueError(
                f"NpySliceSource: unsupported array shape {arr.shape} "
                f"for channel slicing")
        # ascontiguous makes a copy out of the memmap so the file
        # handle can be released; render_fn gets a plain in-memory array.
        slc = np.ascontiguousarray(slc)
        self._cached = self.render_fn(slc)
        return self._cached


class CziSliceSource(Source):
    """Lazy in-memory render of one channel from a CZI file.

    For single-excitation CZIs (one channel per spectral slice).
    Decodes only the requested channel plane via :mod:`aicspylibczi`,
    hands the 2D array to ``render_fn``, caches the result.

    Same shape as :class:`NpySliceSource` — the two are interchangeable
    in :func:`image_grid`.

    Parameters
    ----------
    path : str | Path
        Path to the ``.czi`` file.
    channel_index : int
        Channel (= C dimension) to extract.
    render_fn : Callable[[np.ndarray], bytes]
        Takes the 2D channel array, returns encoded image bytes.
    shape : tuple, optional
        ``(H, W)`` for ``_native_dims`` without re-reading the CZI
        header. ``None`` does a one-time header parse.
    content_type : str
        Default ``"image/jpeg"``.
    """

    def __init__(self, path, channel_index: int, render_fn,
                 shape: Optional[tuple] = None,
                 content_type: str = "image/jpeg"):
        self.path = str(path)
        self.channel_index = int(channel_index)
        self.render_fn = render_fn
        self.content_type = content_type
        self._cached: Optional[bytes] = None
        if shape is None:
            self.shape = _peek_czi_shape(self.path)
        else:
            self.shape = (int(shape[0]), int(shape[1]))

    def get_bytes(self) -> bytes:
        if self._cached is not None:
            return self._cached
        import aicspylibczi  # type: ignore
        czi = aicspylibczi.CziFile(self.path)
        # Decode just the requested channel plane.
        slc, _ = czi.read_image(C=self.channel_index)
        # czi returns extra leading dims; squeeze to 2D.
        slc = np.squeeze(slc)
        if slc.ndim != 2:
            raise ValueError(
                f"CziSliceSource: expected 2D channel plane, "
                f"got shape {slc.shape}")
        self._cached = self.render_fn(np.ascontiguousarray(slc))
        return self._cached


def _peek_npy_shape(path: str, channel_index: int) -> tuple:
    """Return ``(H, W)`` for the channel at ``channel_index``.
    Layout-agnostic: handles channel-first ``(C, H, W)`` and
    channel-last ``(H, W, C)``. Uses ``np.load(mmap_mode='r')`` which
    parses the .npy header without loading the array data."""
    arr = np.load(path, mmap_mode='r')
    shape = arr.shape
    if len(shape) == 2:
        return (int(shape[0]), int(shape[1]))
    if len(shape) == 3:
        # Heuristic: the channel axis is the smaller one.
        if shape[0] < shape[-1]:
            return (int(shape[1]), int(shape[2]))   # (C, H, W)
        return (int(shape[0]), int(shape[1]))       # (H, W, C)
    raise ValueError(f"_peek_npy_shape: unsupported ndim {len(shape)}")


def _peek_czi_shape(path: str) -> tuple:
    """Header-only ``(H, W)`` peek for a CZI file via aicspylibczi."""
    import aicspylibczi  # type: ignore
    czi = aicspylibczi.CziFile(path)
    dims = czi.get_dims_shape()[0]   # dict like {'X': (0, W), 'Y': (0, H), ...}
    h = dims.get('Y', (0, 1))[1]
    w = dims.get('X', (0, 1))[1]
    return (int(h), int(w))


class BytesSource(Source):
    """Serve pre-encoded image bytes from memory verbatim — no re-encode.

    Use when the caller already has an encoded image in memory (e.g.,
    ``scene._rgb_uhdr`` ultra-HDR bytes) and wants to ship it to the
    browser without round-tripping through a numpy array or a disk file.

    Pair with :func:`~ocdkit.io.figure_server.resolve_source` (which
    picks this for scenes with a populated ``_rgb_uhdr`` attribute) or
    use directly for one-off bytes streaming.
    """

    def __init__(self, data, content_type: str = "image/jpeg"):
        # Coerce memoryview / bytearray etc. to immutable bytes so the
        # registry's stable key (id of the bytes object) stays stable
        # across the cache lifetime.
        self.data = bytes(data) if not isinstance(data, bytes) else data
        self.content_type = content_type

    def get_bytes(self) -> bytes:
        return self.data


def _stable_source_key(source) -> Optional[str]:
    """Identity key for ``LocalHttpRegistry.register`` deduplication.

    Returns a stable string when the same ``source`` should map to the
    same URL across repeated ``register()`` calls (so the browser's
    HTTP cache hits AND the registry doesn't leak), or ``None`` when no
    stable identity exists.

    * ``PathSource`` → ``path:{abspath}@{mtime}`` (file re-saved = new
      mtime = fresh token, browser refetches).
    * ``ArraySource`` → ``arr:{buf_ptr}:{shape}:{dtype}``. As long as
      the registry holds a strong reference to the source it keeps the
      underlying ndarray alive, so the buffer pointer is unique for
      the lifetime of that entry. Without this key every repeat
      ``imshow(scene.rgb)`` registers a fresh ArraySource — registry
      grows without bound (each entry holds ~10–50 MB of cached JXL
      bytes plus a ref to the (H, W, 3) float array), and the kernel
      RSS grows ~70 MB per call. With the key, same array → same
      token → entry replaced in place, no growth.
    """
    if isinstance(source, PathSource):
        try:
            mtime = os.path.getmtime(source.path)
        except OSError:
            return None
        return f"path:{os.path.abspath(source.path)}@{mtime}"
    if isinstance(source, ArraySource):
        a = getattr(source, "arr", None)
        if a is None:
            return None
        try:
            # ctypes.data is the buffer's memory address. Combined with
            # shape+dtype it's a stable key for the lifetime of this
            # ndarray.  id() alone isn't safe because it can be reused
            # after gc; the registry holding a strong ref to ``source``
            # (which holds ``arr``) prevents the address from being
            # reused for a different array.
            return f"arr:{a.ctypes.data}:{tuple(a.shape)}:{a.dtype}"
        except Exception:
            return None
    if isinstance(source, BytesSource):
        # id() of the immutable bytes object — the registry holds a
        # strong ref so the id can't be recycled while it's live.
        return f"bytes:{id(source.data)}:{len(source.data)}"
    if isinstance(source, (NpySliceSource, CziSliceSource)):
        # ``render_fn`` identity is the variable axis here — same path
        # + channel rendered with a different cmap should land on a
        # different URL. ``id(render_fn)`` is stable for the lifetime
        # of the registry entry.
        kind = "npy" if isinstance(source, NpySliceSource) else "czi"
        try:
            mtime = os.path.getmtime(source.path)
        except OSError:
            mtime = 0
        return (f"{kind}:{os.path.abspath(source.path)}@{mtime}:"
                f"c{source.channel_index}:r{id(source.render_fn)}")
    return None


class ArraySource(Source):
    """Encode an in-memory array on demand. Two output formats.

    ``fmt='jxl'`` (default):
      * Float input → linear-light Display-P3 → PQ uint16 → P3+PQ JXL.
        Safari renders HDR natively, Chrome behind the JXL flag,
        Firefox does not decode JXL at all.
      * uint8 input → sRGB→Display-P3 retag → SDR JXL.

    ``fmt='uhdr'`` (Ultra-HDR JPEG, ISO 21496-1):
      * Float input → SDR base + per-pixel gain map in one JPEG.
        Safari + Chrome composite the gain map for HDR; Firefox /
        Preview / any other viewer falls back to the SDR base
        cleanly (never a broken image).
      * uint8 input → tagged Display-P3 JPEG (no gain map).

    Pre-encodes in a background daemon thread on construction so the
    bytes are usually ready by the time the browser asks for them.
    ``get_bytes()`` blocks on the thread if it's still running.

    For zero re-encode, prefer constructing a :class:`PathSource` from
    a scene's ``rgb_path`` — it streams raw on-disk bytes with no
    decode/encode round-trip.
    """

    def __init__(self, arr, *, sdr_white_nits: float = 1600.0,
                 fmt: str = "uhdr", quality: int = 95):
        self.arr = np.asarray(arr)
        self.sdr_white_nits = float(sdr_white_nits)
        if fmt not in ("jxl", "uhdr"):
            raise ValueError(f"ArraySource fmt must be 'jxl' or 'uhdr', got {fmt!r}")
        self.fmt = fmt
        self.quality = int(quality)
        self.content_type = "image/jxl" if fmt == "jxl" else "image/jpeg"
        self._cached: Optional[bytes] = None
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._encode_into_cache, daemon=True,
            name="ocdkit.figure_server.ArraySource",
        )
        self._thread.start()

    def _encode_into_cache(self):
        try:
            self._cached = self._compute_bytes()
        except BaseException as exc:
            self._error = exc

    def _compute_bytes(self) -> bytes:
        import opencodecs  # type: ignore
        # Lazy import so importing this module doesn't drag in
        # ocdkit.plot at server-module load time.
        from ..plot.svg import _p3_linear_to_pq_uint16, _srgb_to_display_p3_uint8

        arr = self.arr

        if self.fmt == "uhdr":
            import opencodecs.uhdr as uhdr
            if np.issubdtype(arr.dtype, np.floating):
                hdr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
                # encode_native: custom Cython fused-kernel fast path
                # (~30-40 ms / 2k², 3-4x faster than uhdr.encode()'s
                # libuhdr-driven reference path). Same byte output.
                return uhdr.encode_native(
                    hdr, gamut="display-p3",
                    sdr_white_nits=self.sdr_white_nits,
                    quality=self.quality,
                )
            if arr.dtype == np.uint8:
                # SDR-only input → plain JPEG with Display-P3 ICC. uhdr
                # has no gain map to encode, so go through the simple
                # libjpeg-turbo path with a P3-retag.
                arr_p3 = _srgb_to_display_p3_uint8(arr)
                import imagecodecs  # type: ignore
                return imagecodecs.jpeg_encode(arr_p3, self.quality)
            raise TypeError(
                f"ArraySource: unsupported dtype {arr.dtype} for fmt='uhdr'")

        # ``effort=1, lossless=True`` is the libjxl fast path. Bench on
        # a 2k² uint8 RGB shows ~6 ms encode vs ~2.6 s at libjxl's
        # default effort — roughly 400× faster. Payload is ~8 % bigger
        # than effort=7 lossless but that's irrelevant: hi-res bytes
        # stream over HTTP from the local registry, they never land in
        # the notebook's saved .ipynb. The user-visible win is that the
        # inline auto-upgrade lands in milliseconds instead of seconds.
        if np.issubdtype(arr.dtype, np.floating):
            arr_pq = _p3_linear_to_pq_uint16(arr, sdr_white_nits=self.sdr_white_nits)
            color = opencodecs.ColorSpec(
                primaries=11, transfer=16,
                white_point=1, rendering_intent=1, gamma=0.0,
            )
            buf = io.BytesIO()
            opencodecs.jxl.write(
                buf, arr_pq, color=color, intensity_target=10000.0,
                effort=1, lossless=True,
            )
            return buf.getvalue()
        if arr.dtype == np.uint8:
            arr_p3 = _srgb_to_display_p3_uint8(arr)
            buf = io.BytesIO()
            opencodecs.jxl.write(buf, arr_p3, color="display-p3",
                                  effort=1, lossless=True)
            return buf.getvalue()
        raise TypeError(
            f"ArraySource: unsupported dtype {arr.dtype} "
            "(expected float linear-P3 for HDR or uint8 sRGB for SDR)"
        )

    def get_bytes(self) -> bytes:
        # Wait for the background encode to finish; if it never started
        # (e.g., this Source was constructed via subclass), fall through
        # to a synchronous encode.
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise self._error
        if self._cached is None:
            self._cached = self._compute_bytes()
        return self._cached


# ─── registry protocol + default + Jupyter-proxy variant ─────────────


class FigureSourceRegistry(Protocol):
    """Pluggable backend that maps :class:`Source` objects to URLs the
    browser can fetch from.

    Applications (dashboards, custom servers) implement this protocol
    and install their instance with :func:`set_registry`. The plotting
    layer (``ocdkit.plot.image_grid``) only ever calls ``register``.
    """

    def register(self, source: Source) -> str:
        """Register a source; return a URL the browser will GET."""
        ...


class LocalHttpRegistry:
    """Stdlib HTTP server on ``127.0.0.1`` (random port). Default
    registry; works when the browser can reach the kernel's localhost."""

    def __init__(self):
        self._sources: dict[str, Source] = {}
        # Reverse map: stable source-identity key → existing token, so
        # repeated ``register(source)`` calls for the same path return
        # the same URL. Without this, every ``image_grid()`` re-render
        # generates fresh tokens and the browser HTTP cache always
        # misses on hi-res. Key format: ``"path:{abspath}@{mtime}"``.
        self._token_by_key: dict[str, str] = {}
        self._lock = threading.Lock()
        self._port: Optional[int] = None
        self._server: Optional[socketserver.ThreadingTCPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _ensure_server(self) -> int:
        if self._port is not None:
            return self._port
        with self._lock:
            if self._port is not None:
                return self._port
            registry = self  # capture for handler

            class _Handler(http.server.BaseHTTPRequestHandler):
                # HTTP/1.1 enables keep-alive so the browser reuses one
                # TCP connection across all hi-res tile fetches in a
                # grid, saving the per-request handshake (~1 ms each on
                # localhost, more on first-time TLS / cold sockets).
                # ``Content-Length`` is already set on every response,
                # so message framing works without chunked encoding.
                protocol_version = "HTTP/1.1"

                def do_GET(self):  # noqa: N802 (Handler API)
                    parts = self.path.strip("/").split("/")
                    if len(parts) != 2 or parts[0] != "svgfig":
                        self.send_error(404, "not a figure_server URL")
                        return
                    token = parts[1].split(".", 1)[0]
                    with registry._lock:
                        source = registry._sources.get(token)
                    if source is None:
                        self.send_error(404, f"unknown token {token!r}")
                        return
                    try:
                        data = source.get_bytes()
                    except Exception as exc:  # pragma: no cover
                        self.send_error(500, f"{type(exc).__name__}: {exc}")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", source.content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(data)

                def log_message(self, *args, **kwargs):  # noqa: A003
                    return  # quiet the request log

            server = socketserver.ThreadingTCPServer(
                ("127.0.0.1", 0), _Handler,
            )
            port = server.server_address[1]
            thread = threading.Thread(
                target=server.serve_forever, daemon=True,
                name=f"ocdkit.figure_server:{port}",
            )
            thread.start()
            self._port = port
            self._server = server
            self._thread = thread
            return port

    def register(self, source: Source) -> str:
        port = self._ensure_server()
        key = _stable_source_key(source)
        with self._lock:
            if key is not None:
                existing = self._token_by_key.get(key)
                if existing is not None and existing in self._sources:
                    # Same path+mtime → reuse token so the browser
                    # gets the same URL and its HTTP cache hits.
                    self._sources[existing] = source  # refresh reference
                    ext = _CTYPE_TO_EXT.get(source.content_type, "bin")
                    return f"http://127.0.0.1:{port}/svgfig/{existing}.{ext}"
            token = secrets.token_urlsafe(16)
            self._sources[token] = source
            if key is not None:
                self._token_by_key[key] = token
        ext = _CTYPE_TO_EXT.get(source.content_type, "bin")
        return f"http://127.0.0.1:{port}/svgfig/{token}.{ext}"


class JupyterProxyRegistry(LocalHttpRegistry):
    """Like :class:`LocalHttpRegistry` but emits URLs relative to the
    Jupyter origin via ``jupyter-server-proxy``.

    The stdlib HTTP server still binds to ``127.0.0.1``; the proxy
    extension forwards requests from Jupyter's origin to it. URLs are
    of the form ``/proxy/<port>/svgfig/<token>.<ext>``.

    Requires ``jupyter-server-proxy`` installed in the kernel's
    environment AND served by the Jupyter server hosting the notebook.
    """

    def register(self, source: Source) -> str:
        absolute = super().register(source)
        p = urlparse(absolute)
        return f"/proxy/{p.port}{p.path}"


# ─── active registry + auto-detect ───────────────────────────────────


_registry: Optional[FigureSourceRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> FigureSourceRegistry:
    """Return the active registry, lazily creating one via
    :func:`_auto_registry` on first call."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = _auto_registry()
    return _registry


def set_registry(registry: Optional[FigureSourceRegistry]) -> None:
    """Install an explicit registry. Pass ``None`` to reset to
    auto-detect on the next :func:`get_registry` call."""
    global _registry
    with _registry_lock:
        _registry = registry


def _detect_jupyter() -> bool:
    """Best-effort: are we running inside a Jupyter / IPython kernel?"""
    if "JPY_PARENT_PID" in os.environ or "JUPYTER_SERVER_ROOT" in os.environ:
        return True
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is not None and "ZMQ" in type(ip).__name__:
            return True
    except ImportError:
        pass
    return False


def _auto_registry() -> FigureSourceRegistry:
    """Pick a sensible default registry for the runtime environment."""
    if _detect_jupyter():
        try:
            import jupyter_server_proxy  # noqa: F401
            return JupyterProxyRegistry()
        except ImportError:
            pass
    return LocalHttpRegistry()


# ─── convenience pass-throughs (used by image_grid) ─────────────────


def register(source: Source) -> str:
    """Shortcut for ``get_registry().register(source)``."""
    return get_registry().register(source)


def _peek_jxl_size(path) -> Optional[tuple]:
    """Cheap header-only read returning ``(ysize, xsize)`` for a JXL
    or Ultra-HDR JPEG file. Returns ``None`` if the file can't be
    parsed.

    For ``.jpg``/``.jpeg`` (Ultra-HDR), reads the JPEG SOF marker
    directly — no full decode. For ``.jxl``, uses libjxl's header
    parse via opencodecs.
    """
    path_str = str(path)
    if path_str.endswith((".jpg", ".jpeg")):
        try:
            import struct
            with open(path_str, "rb") as f:
                if f.read(2) != b"\xff\xd8":
                    return None
                while True:
                    b = f.read(2)
                    if len(b) != 2 or b[0] != 0xff:
                        return None
                    mk = b[1]
                    if 0xc0 <= mk <= 0xcf and mk not in (0xc4, 0xc8, 0xcc):
                        # SOFn (n != 4/8/12 = DHT/JPG/DAC). Frame header:
                        # length(2) + precision(1) + height(2) + width(2)
                        f.read(3)  # length + precision
                        height = struct.unpack(">H", f.read(2))[0]
                        width = struct.unpack(">H", f.read(2))[0]
                        return int(height), int(width)
                    ln_bytes = f.read(2)
                    if len(ln_bytes) != 2:
                        return None
                    ln = struct.unpack(">H", ln_bytes)[0]
                    f.seek(ln - 2, 1)
        except Exception:
            return None
    try:
        import opencodecs  # type: ignore
        with opencodecs.jxl.open(path_str, parse_color=False) as r:
            return int(r.ysize), int(r.xsize)
    except Exception:
        return None


def _pick_downsample(src_h: int, src_w: int, target_px: int = 0) -> int:
    """Pick the libjxl-native downsample ratio.

    libjxl 0.11 only fast-paths ratio 8 (DC-progressive decode); ds=2
    and ds=4 take the same time as a full decode + numpy slice. The
    choice is binary: ds=8 (fast) or ds=1 (full decode).

    We pick ds=8 whenever the source has enough pixels that ds=8
    produces a usable thumbnail (≥ 64 px on the longest side, i.e.
    source ≥ 512 px). ``target_px`` is ignored: if ds=8 gives fewer
    pixels than the target cell size, we embed the smaller raster at
    the cell's SVG bbox and let the browser do display scaling
    (``image-rendering: pixelated`` already gives nearest-neighbor on
    upscale). No CPU-side upscale needed.
    """
    longest = max(int(src_h), int(src_w))
    if longest // 8 >= 64:
        return 8
    return 1


def _centered_subsample(arr, n):
    """Centered stride subsample by integer factor ``n``. Bit-exact
    equivalent to libjxl's native ds=N path; what libuhdr lacks as a
    first-class API but achieves the same speed (one numpy view, no
    copy, no reduction). Multi-pixel HDR hot spots survive at high
    probability — a 3×3 spot in an 8×8 block has p ≈ 32% chance of
    landing on the stride sample, but a 5×5 spot has p ≈ 89%.

    The single-bright-pixel pathological case (where stride drops the
    peak and collapses ``max_content_boost``) is accepted as a trade
    for speed and a clean noise-free background: stride doesn't
    amplify the noise floor the way block-max does, and doesn't
    smear-then-dim peaks the way block-mean does.
    """
    if n <= 1:
        return arr
    half = n // 2
    return arr[half::n, half::n]


def _subsample_uhdr_bytes(uhdr_bytes, downsample):
    """Return ``(thumb_bytes, (h, w))`` — a smaller Ultra-HDR JPEG
    whose gain map preserves the source's per-pixel boost values
    bit-faithfully (no recomputation), at ``1/downsample`` spatial
    resolution suitable for inline embedding in an SVG / notebook.

    Pipeline:
        1. Extract base + gain map JPEGs + metadata from the source.
        2. DCT-domain IDCT-scaled decode of both JPEG layers at
           ``1/downsample`` via libjpeg-turbo
           (``opencodecs.codecs._jpeg.decode``). The exact ratio is
           snapped to libjpeg-turbo's supported M/8 set if not native.
        3. Re-encode the smaller base + gain at 4:4:0 chroma
           subsampling (the gain map's best-fidelity mode in
           libjpeg-turbo's quant tables — see
           ``scripts/_bench_gain_subsampling_corpus.py``).
        4. Re-assemble with the ORIGINAL gainmap metadata via
           ``encode_assembled``. ``max_content_boost``,
           ``sdr_white_nits``, and the per-pixel gain values are
           preserved end-to-end — only spatial resolution changes,
           plus a single generation of JPEG quantisation noise.

    Short-circuits to verbatim source bytes when ``downsample <= 1``.
    """
    import opencodecs.uhdr as uhdr
    from opencodecs.codecs import _jpeg as _oc_jpeg
    import imagecodecs

    layers = uhdr._cython_extract_layers(uhdr_bytes)
    src_h, src_w = layers["height"], layers["width"]

    n = max(1, int(downsample))
    if n <= 1:
        return uhdr_bytes, (src_h, src_w)

    # Snap ``1/n`` to the closest libjpeg-turbo-supported M/8 ratio.
    supported = _oc_jpeg.supported_scaling_factors()
    target_ratio = 1.0 / n
    num, denom = min(supported, key=lambda nd: abs(nd[0] / nd[1] - target_ratio))

    metadata = layers["gainmap_metadata"]
    new_base = _oc_jpeg.decode(layers["base_jpeg"], scale=(num, denom))
    new_gain = _oc_jpeg.decode(layers["gainmap_jpeg"], scale=(num, denom))

    # Base layer: SDR fallback, default 4:2:0 chroma is fine.
    new_base_jpeg = imagecodecs.jpeg_encode(new_base, level=95)
    # Gain map: per-pixel HDR boost. 4:4:0 beats 4:2:0 / 4:2:2 / 4:4:4
    # in libjpeg-turbo's quantiser tables for typical natural-image
    # gain content; the 4:4:0 win is robust across the corpus.
    new_gain_jpeg = imagecodecs.jpeg_encode(
        new_gain, level=100, subsampling='440')

    thumb_bytes = uhdr.encode_assembled(
        base_jpeg=new_base_jpeg, gainmap_jpeg=new_gain_jpeg,
        metadata=metadata)
    return thumb_bytes, (int(new_base.shape[0]), int(new_base.shape[1]))


def _encode_rgb_thumb_bytes(rgb_float, downsample):
    """Centered-stride downsample a linear-light Display-P3 float array
    by integer factor ``downsample`` and encode it as a fresh Ultra-HDR
    JPEG via ``encode_native``.

    Much faster than IDCT-decoding both layers of an existing UHDR
    JPEG and re-encoding (~15 ms vs ~60 ms per 2k² source on M-class)
    because:
      1. ``_rgb`` is already a decoded float array — no JPEG decode.
      2. ``encode_native`` on a small array is faster than encoding
         the full source size.

    Stride sampling preserves the per-pixel HDR peak with high
    probability for multi-pixel hot spots (e.g. 5×5 fluorescence
    dots in an 8×8 block survive ~89% of the time) without
    amplifying the dark-noise floor (block-max) or smearing peaks
    (block-mean). Sub-pixel peaks may be lost, but the resulting
    ``max_content_boost`` metadata is still computed from the
    stride-sampled peak — consistent with on-disk-file rendering.
    """
    import opencodecs.uhdr as uhdr

    arr = np.asarray(rgb_float)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(
            f"_encode_rgb_thumb_bytes: expected (H, W, 3|4) float; "
            f"got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    n = max(1, int(downsample))
    if n > 1:
        half = n // 2
        arr = arr[half::n, half::n]
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    thumb_bytes = uhdr.encode_native(
        arr, gamut="display-p3", sdr_white_nits=1600.0, quality=95)
    return thumb_bytes, (int(arr.shape[0]), int(arr.shape[1]))


def resolve_uhdr_thumb_bytes(item, downsample=4):
    """Return ``(thumb_bytes, (h, w))`` for ``item`` — a small Ultra-HDR
    JPEG suitable for inline embedding in an SVG / notebook data URL.

    ``downsample`` is an integer stride factor (1=full resolution, 2=
    half, 4=quarter, 8=eighth). Source-agnostic — no target-pixel
    math, no edge cases. Default 4 maps to a 500² thumb from a 2000²
    source (~400 KB inline payload per scene).

    Caches the result on the scene as
    ``_uhdr_thumb_bytes_ds{downsample}`` so subsequent grid renders
    at the same downsample skip the encode entirely. The cache
    invalidates when ``scene.make_rgb()`` regenerates ``_rgb``
    (keyed on ``id(_rgb)``).

    Source preference (fastest first):
        1. ``item._rgb`` (in-memory float HDR array). Stride-downsample
           in numpy + ``encode_native`` on the small array. ~7 ms at
           downsample=4.
        2. ``item._rgb_uhdr`` (in-memory UHDR bytes). Extract layers,
           DCT-domain decode at ``1/downsample``, re-encode. ~40 ms.
        3. ``item.rgb_path`` if it points at a ``.jpg`` / ``.jpeg``
           UHDR file on disk. Same as 2 but with disk I/O.
    """
    n = max(1, int(downsample))
    rgb_float = getattr(item, "_rgb", None)
    if rgb_float is not None and not (
            hasattr(rgb_float, "dtype")
            and np.issubdtype(rgb_float.dtype, np.floating)):
        rgb_float = None
    rgb_uhdr = getattr(item, "_rgb_uhdr", None)
    rgb_path = getattr(item, "rgb_path", None)
    if not (rgb_path and os.path.exists(str(rgb_path))):
        rgb_path = None

    cache_attr = f"_uhdr_thumb_bytes_ds{n}"
    fp_attr = cache_attr + "_fp"
    if rgb_uhdr is not None:
        fp_now = ("_rgb_uhdr", id(rgb_uhdr))
    elif rgb_float is not None:
        fp_now = ("_rgb", id(rgb_float))
    elif rgb_path:
        fp_now = ("rgb_path", os.path.getmtime(str(rgb_path)))
    else:
        return None
    cached = getattr(item, cache_attr, None)
    if cached is not None and getattr(item, fp_attr, None) == fp_now:
        return cached

    # Fast path: if the source UHDR has an embedded UHDR thumbnail
    # (opencodecs v0.1.12+ `encode_native(..., thumbnail_size=N)`),
    # extract it directly — ~5 us vs ~7 ms for re-encode-on-demand.
    # The embedded thumb preserves the gain map at 99% fidelity, so
    # HDR rendering matches the full-res file. Always tried first,
    # regardless of which other sources are also present.
    fast_source = rgb_uhdr
    if fast_source is None and rgb_path:
        path_str = str(rgb_path)
        if path_str.lower().endswith((".jpg", ".jpeg")):
            with open(path_str, "rb") as f:
                fast_source = f.read()
    if fast_source is not None:
        try:
            import opencodecs.uhdr as _uhdr
            embedded = _uhdr.read_thumbnail_bytes(fast_source)
            if embedded is not None and _uhdr.is_uhdr(embedded):
                info = _uhdr._cython_extract_layers(embedded)
                result = (embedded, (int(info["height"]), int(info["width"])))
                try:
                    setattr(item, cache_attr, result)
                    setattr(item, fp_attr, fp_now)
                except Exception:
                    pass
                return result
        except Exception:
            pass

    # Fallback: encode on demand. Prefer the in-memory float (fastest
    # encode path) when available; otherwise IDCT-subsample the source
    # UHDR bytes.
    if rgb_float is not None:
        result = _encode_rgb_thumb_bytes(rgb_float, n)
    elif rgb_uhdr is not None:
        result = _subsample_uhdr_bytes(rgb_uhdr, n)
    elif fast_source is not None:
        result = _subsample_uhdr_bytes(fast_source, n)
    else:
        return None

    try:
        setattr(item, cache_attr, result)
        setattr(item, fp_attr, fp_now)
    except Exception:
        pass
    return result


def _linear_p3_fingerprint(item) -> tuple:
    """Tuple-valued fingerprint used by ``resolve_linear_p3`` to detect
    when a previously-cached linear-P3 array has gone stale.

    Captures the mtime of the on-disk source (if present) plus the
    ``id()`` of ``_rgb``. Any regeneration of ``scene._rgb`` (new
    array id) or re-save of the on-disk JXL (new mtime) shifts the
    fingerprint and busts any downsample caches.
    """
    fp = []
    rgb_path = getattr(item, "rgb_path", None)
    if rgb_path:
        try:
            fp.append(("mtime", os.path.getmtime(str(rgb_path))))
        except OSError:
            pass
    v = getattr(item, "_rgb", None)
    if v is not None:
        fp.append(("_rgb", id(v)))
    return tuple(fp)


def resolve_linear_p3(item, *,
                       sdr_white_nits: float = 1600.0,
                       target_px: Optional[int] = None,
                       downsample: Optional[int] = None):
    """Best-effort linear-light Display-P3 source for HDR-preserving
    thumbnail downsampling.

    Parameters
    ----------
    target_px : int, optional
        Auto-pick a libjxl-native downsample ratio (8 / 4 / 2 / 1) so
        the longest source side stays ≥ ``target_px`` after decode.
        Combined with the caller's ``_resize_nearest`` final shrink,
        this drops the per-thumbnail decode from a full-res decode +
        Python downsample to libjxl's progressive DC path (free 8× on
        suitable images; matches ``decode(...)[::N,::N]`` on 2× / 4×).
    downsample : int ∈ {1, 2, 4, 8}, optional
        Explicit ratio; overrides ``target_px``. ``1`` (or ``None``)
        does a full decode.

    Resolution order:

    1. Cached downsampled array on the item (per-ratio:
       ``_rgb_linear_p3_ds2``, ``_rgb_linear_p3_ds4``,
       ``_rgb_linear_p3_ds8``).
    2. In-memory ``scene._rgb`` (scene-linear Display P3 float) — the
       fresh user-facing render; stride for ``downsample > 1``.
    3. ``item.rgb_path`` if it points at an HDR JXL on disk — decode
       (at the chosen ratio) and run the PQ⁻¹ OETF + inverse
       shadow-gamma to recover scene-linear P3.
    4. Returns ``None`` if nothing suitable is available.
    """
    if downsample is None:
        if target_px is not None:
            rgb_path = getattr(item, "rgb_path", None)
            if rgb_path and os.path.exists(str(rgb_path)):
                # Cache the header-peek size on the item: ~6 ms per
                # peek (NAS open + JXL header parse) × N scenes adds
                # up. Same scene, same file = same size, so once is
                # enough for the life of the Scene object.
                size = getattr(item, "_rgb_jxl_size", None)
                if size is None:
                    size = _peek_jxl_size(rgb_path)
                    if size is not None:
                        try:
                            item._rgb_jxl_size = size
                        except Exception:
                            pass
                downsample = (_pick_downsample(size[0], size[1], target_px)
                              if size else 1)
            else:
                downsample = 1
        else:
            downsample = 1
    if downsample not in (1, 2, 4, 8):
        raise ValueError(
            f"resolve_linear_p3: downsample must be 1/2/4/8, got {downsample}")

    # ``_mp`` suffix marks the max-pool downsample path (peak-preserving
    # for HDR). Bumping the cache key invalidates any pre-existing
    # strided-sample caches on scenes from earlier runs of this code.
    cache_key = ("_rgb_linear_p3_ds1_mp" if downsample == 1
                 else f"_rgb_linear_p3_ds{downsample}_mp")
    fp_key = cache_key + "_fp"
    # Source fingerprint: rgb_path mtime + id of ``_rgb``. If the user
    # regenerates ``scene._rgb`` (new id) or the on-disk JXL is re-saved
    # (new mtime), the fingerprint shifts and the cache gets rebuilt.
    fp_now = _linear_p3_fingerprint(item)
    cached = getattr(item, cache_key, None)
    if cached is not None and getattr(item, fp_key, None) == fp_now:
        return cached

    # Fast path: in-memory ``scene._rgb`` is already scene-linear
    # Display P3 (upstream ``make_rgb`` implementations consolidated
    # the dual SDR/HDR representation into this single buffer). Stride
    # for the requested downsample. This is also the FRESH render —
    # disk file may be older if the user re-ran ``make_rgb`` without
    # saving.
    #
    # Use the centred subsample ``[N//2::N, N//2::N]`` (not the top-left
    # ``[::N, ::N]``) so each thumbnail pixel sits at the geometric
    # centre of its NxN block. This matches the centred-slice we use on
    # the disk-decode fallback below; both paths must agree or the
    # popup/hover hi-res view drifts by (N-1)/2 viewBox units relative
    # to the thumbnail.
    in_mem = getattr(item, "_rgb", None)
    if in_mem is not None and hasattr(in_mem, "dtype") and \
            np.issubdtype(in_mem.dtype, np.floating):
        if downsample > 1:
            arr = _centered_subsample(np.asarray(in_mem), downsample)
        else:
            arr = in_mem
        try:
            setattr(item, cache_key, arr)
            setattr(item, fp_key, fp_now)
        except Exception:
            pass
        return arr

    # Bytes-as-canonical path: ``scene._rgb_uhdr`` is the in-memory
    # encoded form. Decode once → float HDR → stride for downsample.
    rgb_uhdr = getattr(item, "_rgb_uhdr", None)
    if rgb_uhdr is not None:
        try:
            import opencodecs.uhdr as uhdr
            info = uhdr.decode(rgb_uhdr)
            full = np.asarray(info["hdr_fp16"]).astype(np.float32, copy=False)
            if full.ndim == 3 and full.shape[-1] == 4:
                full = full[..., :3]
            if downsample > 1:
                arr = _centered_subsample(full, downsample)
            else:
                arr = full
            try:
                setattr(item, cache_key, arr)
                setattr(item, fp_key, fp_now)
            except Exception:
                pass
            return arr
        except Exception:
            pass  # fall through to disk path

    # Last resort: on-disk HDR file. JXL via opencodecs, Ultra-HDR JPEG
    # via opencodecs.uhdr (decodes to fp16 → recast to float32).
    rgb_path = getattr(item, "rgb_path", None)
    if rgb_path and os.path.exists(str(rgb_path)):
        try:
            import opencodecs  # type: ignore
            path_str = str(rgb_path)
            if path_str.endswith((".jpg", ".jpeg")):
                import opencodecs.uhdr as uhdr
                with open(path_str, "rb") as f:
                    info = uhdr.decode(f.read())
                full = np.asarray(info["hdr_fp16"]).astype(np.float32, copy=False)
                if full.ndim == 3 and full.shape[-1] == 4:
                    full = full[..., :3]
                if downsample > 1:
                    arr = _centered_subsample(full, downsample)
                else:
                    arr = full
            else:
                # libjxl's ``downsample=N`` path on modular-lossless streams
                # is bit-exact equivalent to ``arr[::N, ::N]`` — the
                # *top-left* pixel of each NxN block. Rendered at the
                # centre of an NxN-viewBox-unit cell, that's offset by
                # ~(N-1)/2 viewBox units from where the hires would place
                # the same content. Full decode + centred slice
                # (``[N//2::N, N//2::N]``) puts each thumb pixel at the
                # geometric centre of its block. Zero speed cost vs
                # libjxl ds=N on modular-lossless. TODO(opencodecs): add
                # a centred-subsample option to ``jxl.read`` so VarDCT
                # streams with progressive DC can keep the native fast path.
                if downsample > 1:
                    full = opencodecs.jxl.read(path_str)
                    arr = _centered_subsample(full, downsample)
                else:
                    arr = opencodecs.jxl.read(path_str)
        except Exception:
            return None
    else:
        return None

    if arr.dtype == np.uint16:
        # HDR PQ-encoded — invert PQ to linear-P3. Source files are
        # display-p3-pq (primaries=11) per the upstream encoder convention
        # of P3-only output; no color-space matrix needed.
        from ..plot.svg import _pq_uint16_to_p3_linear
        linear = _pq_uint16_to_p3_linear(arr, sdr_white_nits=sdr_white_nits)
        try:
            setattr(item, fp_key, fp_now)
        except Exception:
            pass
        try:
            setattr(item, cache_key, linear)
        except Exception:
            pass
        return linear
    # uint8 (or other) SDR — promote to linear-[0,1]. Not HDR but at
    # least consistent dtype for the float path.
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return None


def _slice_source_for(item) -> Optional[Source]:
    """If ``item`` is already a :class:`Source`, return it unchanged.
    Used by image_grid so callers can pass ``NpySliceSource`` /
    ``CziSliceSource`` / ``BytesSource`` / ``PathSource`` / etc.
    directly as grid items."""
    if isinstance(item, Source):
        return item
    return None


def resolve_source(item) -> Optional[Source]:
    """Best-effort source selection for a grid item.

    Preference order — pick the cheapest path that produces servable
    bytes:

    1. ``item.rgb_path`` if it points at an existing file — :class:`PathSource`
       streams the bytes verbatim (no encode, no decode). Browser caches
       by ``mtime``-stamped URL token, so re-renders hit cache.
    2. ``item._rgb_uhdr`` if present (pre-encoded Ultra-HDR JPEG bytes
       held in memory) — :class:`BytesSource` ships them verbatim. Same
       zero-encode cost as PathSource, but doesn't need a disk file —
       used when the scene was just regenerated and not yet saved.
    3. ``item._rgb`` if present (scene-linear Display P3 float) —
       :class:`ArraySource` encodes on demand (~40 ms for a 2k² scene).
    4. ``item.rgb`` (lazy property fallback) if present.
    5. ``item`` itself if it's an ndarray.

    Stale-data caveat: if the caller regenerated ``item._rgb`` in memory
    after the file was saved AND didn't update ``_rgb_uhdr``, the older
    disk file wins. Upstream ``make_rgb``-style helpers should populate
    ``_rgb_uhdr`` alongside the float to keep both fresh.

    Returns ``None`` when nothing usable was found (caller falls back
    to the inline thumbnail in the overlay).
    """
    # Caller-supplied Source (NpySliceSource, CziSliceSource, raw
    # BytesSource, etc.) wins outright — no further resolution.
    if isinstance(item, Source):
        return item
    rgb_path = getattr(item, "rgb_path", None)
    if rgb_path and os.path.exists(str(rgb_path)):
        return PathSource(rgb_path)
    rgb_uhdr = getattr(item, "_rgb_uhdr", None)
    if rgb_uhdr is not None:
        return BytesSource(rgb_uhdr, content_type="image/jpeg")
    in_mem = getattr(item, "_rgb", None)
    if in_mem is not None:
        return ArraySource(in_mem)
    rgb = getattr(item, "rgb", None)
    if rgb is not None:
        return ArraySource(np.asarray(rgb))
    if isinstance(item, np.ndarray):
        return ArraySource(item)
    return None


__all__ = [
    "Source",
    "PathSource",
    "ArraySource",
    "BytesSource",
    "NpySliceSource",
    "CziSliceSource",
    "FigureSourceRegistry",
    "LocalHttpRegistry",
    "JupyterProxyRegistry",
    "get_registry",
    "set_registry",
    "register",
    "resolve_source",
]
