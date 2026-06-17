"""Jupyter server extension: serve the local tileserve engine through Jupyter's
own origin — ``{base_url}ocdkit-tiles/{port}/{path}`` proxies to
``http://127.0.0.1:{port}/{path}``.

Why: the tile server is an in-kernel daemon bound to 127.0.0.1, so a notebook
opened on ANOTHER machine can't reach it directly (and an HTTPS Jupyter would
mixed-content-block a raw http iframe anyway). Routing through Jupyter inherits
its HTTPS + token/cookie auth and same origin — no extra open ports and no
``jupyter-server-proxy`` dependency (this handler is ~70 lines of tornado, which
Jupyter itself runs on).

Scope: GET-only (the tile API is read-only: /info, /tile, /attach, /grid …) and
restricted to the engine's stable ports — this is NOT a general-purpose proxy.

Enable once per environment (then restart the Jupyter server)::

    jupyter server extension enable ocdkit.tileserve.jupyter_ext

(A regular ``pip install ocdkit`` also ships the config via data files; the
explicit enable covers editable/dev installs, where data files don't land.)
"""
from __future__ import annotations

import tornado.httpclient
import tornado.web

from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.utils import url_path_join

# Mirrors server._STABLE_PORTS (kept literal so importing this module never pulls
# numpy/the engine into the Jupyter server process).
_ALLOWED_PORTS = {"8137", "8138", "8139", "8140"}

# Engine response headers the viewer reads — forwarded verbatim.
_FWD_HEADERS = ("Content-Type", "Cache-Control")
_FWD_PREFIX = "X-"   # X-Level-Width, X-Mode, X-Lo/Hi, X-Downsample, X-Map-W …


# A full-res tile is ~16 MB; buffering it upstream before replying SERIALIZES the
# two hops (engine→Jupyter, then Jupyter→browser) and tripled-plus the latency.
# Stream chunks through as they arrive instead, and widen the client pool so a
# layer refresh's burst of parallel tile fetches doesn't queue (default is 10).
tornado.httpclient.AsyncHTTPClient.configure(
    None, max_clients=64, max_body_size=1024 ** 3)


class TileProxyHandler(JupyterHandler):
    """Authenticated, STREAMING reverse proxy for the local tileserve engine."""

    @tornado.web.authenticated
    async def get(self, port: str, path: str):
        if port not in _ALLOWED_PORTS:
            raise tornado.web.HTTPError(403, f"port {port} is not a tileserve port")
        url = f"http://127.0.0.1:{port}/{path}"
        if self.request.query:
            url += "?" + self.request.query
        hdr_lines: list = []
        state = {"started": False}

        def _apply_headers():
            # first line = "HTTP/1.1 200 OK"; the rest are header lines
            code = 200
            if hdr_lines:
                try:
                    code = int(hdr_lines[0].split()[1])
                except Exception:
                    pass
            self.set_status(code)
            for ln in hdr_lines[1:]:
                if ":" not in ln:
                    continue
                name, _, v = ln.partition(":")
                name = name.strip(); v = v.strip()
                if name in _FWD_HEADERS or name.startswith(_FWD_PREFIX):
                    self.set_header(name, v)
            state["started"] = True

        def _on_chunk(chunk):
            if not state["started"]:
                _apply_headers()
            self.write(chunk)
            # flush in ~1MB batches: per-64KB flushes cost more than they buy;
            # the first flush still goes out immediately (started just flipped).
            state["pending"] = state.get("pending", 0) + len(chunk)
            if state["pending"] >= (1 << 20):
                state["pending"] = 0
                self.flush()

        req = tornado.httpclient.HTTPRequest(
            url, method="GET", header_callback=lambda ln: hdr_lines.append(ln),
            streaming_callback=_on_chunk, decompress_response=False,
            connect_timeout=5, request_timeout=120)
        client = tornado.httpclient.AsyncHTTPClient()
        try:
            await client.fetch(req, raise_error=False)
        except Exception:                               # engine not running / unreachable
            if state["started"]:
                return                                   # mid-stream — just stop
            # The upstream tileserve is gone — typically a SAVED notebook output
            # reopened after its kernel (and tile server) restarted: its stale
            # tiles poll this proxy. A 502 here makes jupyter_server log a WARNING +
            # ERROR PER request → a terminal flood. ``ensure_server`` blocks until
            # the server accepts before any figure renders, so a refused tile fetch
            # always means GONE (never a startup blip) — so reply 204 (quiet: an
            # info-level access line, suppressed at the usual WARNING log level; no
            # exception warning) + a marker header. The figure tile controller sees
            # X-Tileserve-Gone and stops polling + shows a re-run notice; older
            # saved controllers treat 204 as "not ready", poll quietly, self-limit.
            self.set_status(204)
            self.set_header("X-Tileserve-Gone", "1")
            await self.finish()
            return
        if not state["started"]:
            _apply_headers()                             # empty body (204 / errors)
        await self.finish()


def _jupyter_server_extension_points():
    return [{"module": "ocdkit.tileserve.jupyter_ext"}]


def _load_jupyter_server_extension(serverapp):
    web_app = serverapp.web_app
    route = url_path_join(web_app.settings["base_url"], r"ocdkit-tiles/(\d+)/(.*)")
    web_app.add_handlers(".*$", [(route, TileProxyHandler)])
    serverapp.log.info("ocdkit.tileserve.jupyter_ext: /ocdkit-tiles/{port}/ -> 127.0.0.1:{port}")


# legacy alias some jupyter versions look for
load_jupyter_server_extension = _load_jupyter_server_extension
