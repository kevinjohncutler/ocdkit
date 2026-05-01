"""FastAPI application factory and server / desktop launchers for ocdkit.viewer.

The factory itself is a slim assembler: it wires up middleware, exception
handlers, the static-file mount, and includes the route modules under
``ocdkit.viewer.routers``. All actual routing/validation logic lives in those
modules (which pull dependencies from :mod:`ocdkit.viewer.dependencies` and
schemas from :mod:`ocdkit.viewer.schemas`).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from fastapi import Request  # noqa: F401  (resolved by FastAPI annotation introspection)

from .assets import WEB_DIR
from . import exceptions as _exc_module
from .middleware import BodySizeLimitMiddleware
from .plugins.base import SegmentationPlugin
from .plugins.registry import REGISTRY
from .segmentation import ACTIVE_PLUGIN

logger = logging.getLogger("ocdkit.viewer")

# Body-size limit on POST endpoints (bytes). Overridable via env var.
MAX_BODY_BYTES = int(os.environ.get("OCDKIT_VIEWER_MAX_BODY_BYTES", str(64 * 1024 * 1024)))

# Default fallback for window/page/docs title when no override is set.
DEFAULT_TITLE = "ocdkit.viewer"


def viewer_title() -> str:
    """Read the configured viewer title.

    Resolution order:
      1. ``OCDKIT_VIEWER_TITLE`` environment variable (set by ``--title`` CLI
         flag or by an embedding launcher like ``omnirefactor-gui``).
      2. :data:`DEFAULT_TITLE`.
    """
    return os.environ.get("OCDKIT_VIEWER_TITLE") or DEFAULT_TITLE

SCRIPT_START = time.perf_counter()
_DEV_CERT_DIR = Path(tempfile.gettempdir()) / "ocdkit_viewer_dev_ssl"


# -----------------------------------------------------------------------------
# Cert / port helpers (used by run_server / run_desktop)
# -----------------------------------------------------------------------------


def _ensure_dev_certificate() -> tuple[str, str]:
    _DEV_CERT_DIR.mkdir(exist_ok=True)
    cert_path = _DEV_CERT_DIR / "localhost.pem"
    key_path = _DEV_CERT_DIR / "localhost.key"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError(
            "openssl not found; install it or provide --ssl-cert/--ssl-key"
        )
    subprocess.run(
        [
            openssl, "req", "-x509", "-nodes",
            "-newkey", "rsa:2048",
            "-keyout", str(key_path),
            "-out", str(cert_path),
            "-days", "7",
            "-subj", "/CN=localhost",
        ],
        check=True,
    )
    return str(cert_path), str(key_path)


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=0.5)):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server at {host}:{port} did not become ready within {timeout}s")


# -----------------------------------------------------------------------------
# Static-file mount with cache headers
# -----------------------------------------------------------------------------


class _CachedStaticFiles:
    """Lazy wrapper that adds long-lived Cache-Control to static responses.

    The viewer's index template emits ``?v=<mtime>`` query params on every
    JS/CSS reference (see :func:`assets.render_index`), so we can safely tell
    browsers to cache the file body itself for a year. Editing a file changes
    its mtime, which changes the query param, which forces a fresh fetch.
    """

    def __init__(self, directory: Path) -> None:
        from fastapi.staticfiles import StaticFiles
        self._app = StaticFiles(directory=directory)

    async def __call__(self, scope, receive, send):
        async def _send_with_cache(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Don't override an existing Cache-Control if some other layer set one.
                if not any(k == b"cache-control" for k, _ in headers):
                    headers.append(
                        (b"cache-control", b"public, max-age=31536000, immutable")
                    )
                message = {**message, "headers": headers}
            await send(message)

        await self._app(scope, receive, _send_with_cache)


# -----------------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------------


def create_app() -> "Any":
    """Create and configure the FastAPI application."""
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from .routers import index, log, mask, plugin, segment, session_routes, system, trust

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Always register the bundled threshold (skimage) plugin — discover()
        # only handles entry-point plugins, and uvicorn's reload worker runs
        # lifespan but not the supervisor's _autoload_plugins().
        from .plugins.threshold import plugin as _threshold_plugin
        if _threshold_plugin.name not in REGISTRY:
            REGISTRY.register(_threshold_plugin)
        REGISTRY.discover()
        active = ACTIVE_PLUGIN.current()
        if active and active.warmup is not None:
            try:
                active.warmup("")
            except Exception:
                logger.exception("plugin warmup failed during lifespan startup")
        yield

    app = FastAPI(title=viewer_title(), lifespan=lifespan)

    # Body-size cap BEFORE CORS so oversized bodies get 413 before CORS work.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    # CORS: ``allow_credentials=True`` with ``allow_origins=["*"]`` is invalid
    # per the CORS spec — browsers reject it. We don't use cross-origin cookies,
    # so disable credentials and keep the wildcard origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Standard error envelope across all routes + Pydantic validation errors.
    _exc_module.install(app)

    # Static files with long-lived cache (assets URLs are mtime-busted).
    if WEB_DIR.exists():
        app.mount("/static", _CachedStaticFiles(WEB_DIR), name="static")

    # Routers — order matters only insofar as paths must be unique.
    app.include_router(index.router)
    app.include_router(system.router, tags=["system"])
    app.include_router(log.router, tags=["log"])
    app.include_router(session_routes.router, tags=["session"])
    app.include_router(segment.router, tags=["segment"])
    app.include_router(plugin.router, tags=["plugin"])
    app.include_router(mask.router, tags=["mask"])
    app.include_router(trust.router)

    return app


# -----------------------------------------------------------------------------
# Server / desktop launchers
# -----------------------------------------------------------------------------


def start_trust_setup_sidecar(host: str, port: int) -> None:
    """Start a tiny HTTP server that only serves /trust/* routes.

    Runs in a daemon thread so the main server owns the lifecycle. Users hit
    this over plain HTTP (no TLS warning) to download the root CA before
    their OS trusts our HTTPS cert. Safe to call from any app embedding an
    HTTPS server (hiprpy, omnipose, bare FastAPI, etc.).
    """
    import uvicorn as _uvicorn
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    from .routers import trust as _trust

    setup_app = FastAPI(
        title="ocdkit.viewer — trust install",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @setup_app.get("/")
    def _root():  # type: ignore[no-untyped-def]
        return RedirectResponse("/trust/install")

    setup_app.include_router(_trust.router)

    def _serve() -> None:
        cfg = _uvicorn.Config(
            setup_app, host=host, port=port, log_level="warning",
            access_log=False,
        )
        server = _uvicorn.Server(cfg)
        try:
            server.run()
        except Exception as exc:
            print(f"[viewer] trust sidecar stopped: {exc}", flush=True)

    threading.Thread(target=_serve, daemon=True, name="ocdkit-trust-setup").start()


def start_https_demuxer(
    host: str,
    port: int,
    *,
    https_backend_port: int,
    http_backend_port: int,
) -> None:
    """Start a TCP demultiplexer that listens on a single public port.

    Sniffs the first byte of each incoming connection:
      - ``0x16`` (TLS Client Hello) → routes to the HTTPS backend
      - anything else (HTTP method like 'G', 'P', 'H', ...) → HTTP backend

    Runs in a daemon thread with its own asyncio event loop. Both backends
    must be uvicorn instances bound to ``127.0.0.1:<backend_port>``.
    """
    import asyncio as _asyncio

    async def _proxy(src, dst):
        try:
            while True:
                buf = await src.read(8192)
                if not buf:
                    break
                dst.write(buf)
                await dst.drain()
        except Exception:
            pass
        finally:
            try:
                dst.close()
            except Exception:
                pass

    async def _handle(reader, writer):
        try:
            first = await _asyncio.wait_for(reader.read(1), timeout=10.0)
            if not first:
                writer.close()
                return
            backend_port = https_backend_port if first == b"\x16" else http_backend_port
            try:
                br, bw = await _asyncio.open_connection("127.0.0.1", backend_port)
            except Exception:
                writer.close()
                return
            bw.write(first)
            await bw.drain()
            await _asyncio.gather(
                _proxy(reader, bw),
                _proxy(br, writer),
                return_exceptions=True,
            )
        except _asyncio.TimeoutError:
            try:
                writer.close()
            except Exception:
                pass
        except Exception:
            try:
                writer.close()
            except Exception:
                pass

    def _runner() -> None:
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
        try:
            server = loop.run_until_complete(
                _asyncio.start_server(_handle, host, port)
            )
            try:
                loop.run_until_complete(server.serve_forever())
            finally:
                server.close()
                loop.run_until_complete(server.wait_closed())
        except Exception as exc:
            print(f"[viewer] demuxer stopped: {exc}", flush=True)
        finally:
            loop.close()

    threading.Thread(target=_runner, daemon=True, name="ocdkit-demuxer").start()


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
    reload: bool = False,
    https_dev: bool = False,
    https_auto: str | list[str] | bool = False,
    tls_config: dict | None = None,
    plugins: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Launch the FastAPI viewer over HTTP or HTTPS.

    For HTTPS via your private step-ca (see :mod:`ocdkit.tls`), pass
    ``https_auto=True`` (or a hostname / list of hostnames) together with
    ``tls_config={"ca_url": ..., "provisioner": ..., "provisioner_password_file": ...}``.
    Apps embedding the viewer (hiprpy, omnipose, etc.) should ship their own
    ``tls_config`` rather than relying on a shared global file.
    """
    if title:
        os.environ["OCDKIT_VIEWER_TITLE"] = title
    _autoload_plugins(plugins)

    if https_dev and (ssl_cert or ssl_key):
        print("[viewer] ignoring --https-dev because custom SSL provided", flush=True)
    if https_dev and not (ssl_cert and ssl_key):
        try:
            ssl_cert, ssl_key = _ensure_dev_certificate()
            print(f"[viewer] using dev TLS cert at {ssl_cert}", flush=True)
        except Exception as exc:
            print(f"[viewer] dev cert provision failed: {exc}", flush=True)
            ssl_cert = ssl_key = None

    if https_auto and not (ssl_cert and ssl_key):
        from .. import tls as _tls
        hostnames = None if https_auto is True else https_auto
        # Default: use LocalCA (pure-Python, no external infra). Pass tls_config=
        # to override (e.g. shared step-ca via ca_url+provisioner+...).
        kwargs = dict(tls_config) if tls_config else {}
        try:
            ssl_cert, ssl_key = _tls.ensure_cert(hostnames, **kwargs)
            cert_path_obj = Path(ssl_cert)
            cert_name = cert_path_obj.stem  # e.g. "mac-studio"
            print(f"[viewer] auto-provisioned TLS cert for "
                  f"{hostnames or cert_name}: {ssl_cert}", flush=True)
        except _tls.TLSConfigError as exc:
            print(f"[viewer] --https-auto failed: {exc}", flush=True)
            raise SystemExit(1)

        # Tell the install banner what origins to probe so it can hide itself
        # automatically once the CA is trusted (works even in incognito).
        try:
            from . import assets as _assets
            sans = _tls._resolve_hostnames(hostnames, kwargs.get("hostmap_path"))
            _assets.set_trust_probe_origins(
                [f"https://{n}:{port}" for n in sans]
            )
        except Exception as exc:
            print(f"[viewer] could not set trust probe origins: {exc}", flush=True)

    try:
        import uvicorn
    except ImportError:
        print(
            "fastapi + uvicorn required. Install with: pip install 'ocdkit[viewer]'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # When source lives on a NAS / SMB / NFS share, kernel inotify events
    # don't fire on remote writes. Force the watchfiles backend to poll so
    # auto-reload works regardless of where the code lives. Cheap (one stat
    # per watched dir per cycle) and only enabled when --reload is on.
    if reload:
        os.environ.setdefault("WATCHFILES_FORCE_POLLING", "true")

    # Use current() (which auto-selects a default), not name() — name() returns
    # the raw stored value which is None at startup before discovery has set
    # anything, leading to misleading "(none)" log on a fully-functional setup.
    _active = ACTIVE_PLUGIN.current()
    print(f"[viewer] active plugin: {_active.name if _active else '(none)'}", flush=True)
    print(f"[viewer] registered: {REGISTRY.names() or '(none)'}", flush=True)

    # When TLS is on, listen on a single public port that demuxes HTTPS vs HTTP:
    #   https:// hits the viewer; http:// hits the trust install page.
    # Goal: end users type one URL; if browser warning blocks them, switching
    # to plain http:// on the same port lands them on the install page.
    if ssl_cert and ssl_key:
        https_internal = _pick_free_port()
        http_internal = _pick_free_port()
        # Demuxer + trust sidecar live in the supervisor process (this one).
        # When uvicorn reloads, it spawns/kills the worker process — our
        # daemon threads here keep forwarding to whatever port the worker
        # binds (they reconnect each new TCP request).
        start_trust_setup_sidecar("127.0.0.1", http_internal)
        start_https_demuxer(
            host, port,
            https_backend_port=https_internal,
            http_backend_port=http_internal,
        )
        print(f"[viewer] serving at https://{host}:{port} (and http:// for trust install)",
              flush=True)
        uvicorn_kwargs: dict[str, Any] = dict(
            host="127.0.0.1",
            port=https_internal,
            ssl_certfile=ssl_cert,
            ssl_keyfile=ssl_key,
            log_level="info",
        )
        if reload:
            reload_dirs = _collect_reload_dirs()
            uvicorn.run(
                "ocdkit.viewer.app:create_app",
                factory=True,
                reload=True,
                reload_dirs=reload_dirs,
                **uvicorn_kwargs,
            )
        else:
            uvicorn.run(create_app(), **uvicorn_kwargs)
        return

    print(f"[viewer] serving at http://{host}:{port}", flush=True)
    if reload:
        reload_dirs = _collect_reload_dirs()
        uvicorn.run(
            "ocdkit.viewer.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=reload_dirs,
            log_level="info",
        )
        return

    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        log_level="info",
    )


def run_desktop(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
    reload: bool = False,
    plugins: list[str] | None = None,
    snapshot_path: str | None = None,
    snapshot_timeout: float = 4.0,
    eval_js: str | None = None,
    title: str | None = None,
    icon: str | None = None,
    app_name: str | None = None,
    app_identity: Any = None,
) -> None:
    """Launch the viewer in a pywebview desktop window with an embedded uvicorn.

    Parameters
    ----------
    app_name:
        Host application identity for desktop integration — controls the
        macOS bundle name, Windows ``AppUserModelID``, Linux
        ``StartupWMClass``, AND the ``platformdirs`` key the viewer uses
        for any persistent state/cache.  When an embedding launcher (e.g.
        omnirefactor, hiprpy) calls this function it should pass its own
        app name so each host stays namespaced.

        Ignored if ``app_identity`` is given.  Defaults to ``"ocdkit-viewer"``
        when neither is specified.

    app_identity:
        Fully-specified :class:`ocdkit.desktop.pinning.AppIdentity` for
        callers that need to customise bundle IDs, icon paths, etc.  Takes
        precedence over ``app_name``.

    icon:
        Path to a PNG used as the window/dock icon. Resolution order:
        explicit ``icon`` → ``OCDKIT_VIEWER_ICON`` env var → ``app_identity``'s
        icon → auto-generated fallback. Applied only when ``app_identity``
        is not given.
    """
    if title:
        os.environ["OCDKIT_VIEWER_TITLE"] = title
    if icon:
        os.environ["OCDKIT_VIEWER_ICON"] = str(icon)

    # Reload spawns a uvicorn reloader subprocess that survives the parent's
    # ``os._exit(0)``. Incompatible with --snapshot / --eval-js where the
    # caller (CI / tests) waits for the parent to exit. Force-disable.
    if reload and (snapshot_path or eval_js):
        reload = False

    try:
        import webview  # noqa: F401
    except ImportError:
        print(
            "pywebview required for desktop mode. Install with: pip install 'ocdkit[desktop]'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Apply Windows dark mode hints *before* importing webview — the flag is
    # locked in during the first WebView2 initialisation. No-op elsewhere.
    from ocdkit.desktop.pinning import (
        AppIdentity, apply_early_dark_mode, relaunch_via_bundle,
        setup_platform, set_window_icon,
    )
    apply_early_dark_mode()

    import webview  # type: ignore[import]

    _autoload_plugins(plugins)

    # Resolve the source PNG for the dock/window icon. Resolution order:
    #   1. ``OCDKIT_VIEWER_ICON`` env var (set by ``run_desktop(icon=...)`` or
    #      by an embedding launcher like ``omnirefactor-gui``).
    #   2. AppIdentity default (None) → ocdkit.desktop.pinning auto-generates
    #      a fallback blue circle into the app's local dir.
    icon_env = os.environ.get("OCDKIT_VIEWER_ICON") or None
    if icon_env and not Path(icon_env).is_file():
        logger.warning("OCDKIT_VIEWER_ICON points at missing file: %s", icon_env)
        icon_env = None

    if app_identity is not None:
        VIEWER_APP = app_identity
    else:
        name = app_name or "ocdkit-viewer"
        VIEWER_APP = AppIdentity(
            name=name,
            gui_entry_point="ocdkit-viewer-gui",
            windows_app_id=f"{name}.Viewer.Launcher",
            linux_app_id=name.lower().replace(" ", "-"),
            macos_bundle_id=f"com.{name.lower().replace(' ', '').replace('-', '')}.viewer",
            description=f"{name} image viewer",
            categories="Science;Graphics",
            icon_png=icon_env,
        )
    setup_platform(VIEWER_APP)

    # macOS: on terminal launches, re-exec through the bundle so LaunchServices
    # associates the PID and the Dock uses Icon Services (Liquid Glass tile,
    # shadow, auto-synthesised sizes). No-op on other platforms and on
    # bundle-launched processes. See ocdkit.desktop.pinning.relaunch_via_bundle
    # for the underlying rationale (NSISIconImageRep vs NSBitmapImageRep).
    #
    # Skip in automation mode: ``os.execvp`` detaches the process so stdout is
    # lost and the caller (CI / snapshot tests) can't capture results.
    if not (snapshot_path or eval_js):
        relaunch_via_bundle(VIEWER_APP)

    serve_host = host or "127.0.0.1"
    serve_port = port if port and port > 0 else _pick_free_port(serve_host)
    scheme = "https" if (ssl_cert and ssl_key) else "http"

    server, server_thread, server_proc = None, None, None
    try:
        if reload:
            args = [
                sys.executable, "-m", "uvicorn",
                "ocdkit.viewer.app:create_app",
                "--factory",
                "--host", serve_host,
                "--port", str(serve_port),
                "--reload",
                "--reload-dir", str(Path(__file__).resolve().parent),
            ]
            if WEB_DIR.exists():
                args.extend(["--reload-dir", str(WEB_DIR)])
            if ssl_cert and ssl_key:
                args.extend(["--ssl-certfile", ssl_cert, "--ssl-keyfile", ssl_key])
            server_proc = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
        else:
            import uvicorn
            config = uvicorn.Config(
                create_app(),
                host=serve_host,
                port=serve_port,
                ssl_certfile=ssl_cert,
                ssl_keyfile=ssl_key,
                log_level="info",
            )
            server = uvicorn.Server(config)
            server_thread = threading.Thread(
                target=server.run, name="ViewerUvicorn", daemon=True
            )
            server_thread.start()
            while not server.started:
                if not server_thread.is_alive():
                    raise RuntimeError("uvicorn server thread exited prematurely")
                time.sleep(0.05)
        _wait_for_port(serve_host, serve_port, timeout=10.0)
    except Exception:
        if server_proc:
            server_proc.terminate()
        raise

    # ?ui=desktop tells the index renderer to swap body to translucent so the
    # OS-native vibrancy / blur shows through.
    window_url = f"{scheme}://{serve_host}:{serve_port}/?ui=desktop"
    print(f"[viewer] desktop UI loading {window_url}", flush=True)

    snapshot_target = Path(snapshot_path).expanduser() if snapshot_path else None
    automation_needed = bool(snapshot_target or eval_js)
    loaded_event = threading.Event()

    # Solid window background that adapts to the OS dark/light appearance.
    # We don't use ``transparent=True``/``vibrancy=True``: the macOS Cocoa
    # backend's vibrancy setup is fragile (NSVisualEffectView added as a
    # webview subview rather than below it) and produces an unstable look
    # across OS versions. A solid dark/light background gives the same look
    # as Finder/file-explorer chrome: dark in dark mode, white in light.
    from ocdkit.desktop.pinning import is_dark_mode
    bg = "#111111" if is_dark_mode() else "#ffffff"
    window = webview.create_window(
        viewer_title(),
        url=window_url,
        width=1024,
        height=768,
        resizable=True,
        hidden=automation_needed,
        background_color=bg,
    )

    def _automation_worker():
        if not loaded_event.wait(timeout=max(snapshot_timeout, 10.0)):
            print("[viewer] automation timeout waiting for window load", flush=True)
            try:
                webview.destroy_window()
            except Exception:
                pass
            os._exit(2)
        if eval_js:
            try:
                result = window.evaluate_js(eval_js)
                print(f"[viewer] eval-js result: {result!r}", flush=True)
            except Exception as exc:
                print(f"[viewer] eval-js error: {exc}", file=sys.stderr)
        if snapshot_target:
            if snapshot_target.parent and not snapshot_target.parent.exists():
                snapshot_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                capture = window.evaluate_js(
                    "(function(){var c=document.getElementById('canvas');"
                    "if(!c||!c.width||!c.height) return null;"
                    "return c.toDataURL('image/png');})();"
                )
                if isinstance(capture, str) and capture.startswith("data:"):
                    _, _, b64 = capture.partition(",")
                    snapshot_target.write_bytes(base64.b64decode(b64))
                    print(f"[viewer] snapshot saved to {snapshot_target}", flush=True)
            except Exception as exc:
                print(f"[viewer] snapshot error: {exc}", file=sys.stderr)
        # Force exit (issue #10): webview.start() doesn't reliably return after
        # destroy_window() on macOS; flush + os._exit ensures the subprocess
        # terminates so callers (CI, tests) don't hang.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            webview.destroy_window()
        except Exception:
            pass
        time.sleep(0.1)
        os._exit(0)

    def on_window_loaded() -> None:
        loaded_event.set()

    window.events.loaded += on_window_loaded

    # Server-health watchdog: if uvicorn dies behind us, close the window.
    def _watch_server_health() -> None:
        while True:
            time.sleep(1.0)
            if server_proc is not None and server_proc.poll() is not None:
                print(
                    f"[viewer] uvicorn subprocess exited with code "
                    f"{server_proc.returncode}; closing desktop window",
                    file=sys.stderr,
                )
                break
            if server_thread is not None and not server_thread.is_alive():
                print("[viewer] uvicorn thread died; closing desktop window",
                      file=sys.stderr)
                break
        try:
            webview.destroy_window()
        except Exception:
            pass

    threading.Thread(
        target=_watch_server_health, name="ViewerServerWatchdog", daemon=True
    ).start()

    def on_start() -> None:
        set_window_icon(VIEWER_APP, window_title=viewer_title())
        if automation_needed:
            threading.Thread(
                target=_automation_worker, name="ViewerAutomation", daemon=True
            ).start()

    # Note: ``webview.start(icon=...)`` only does anything on Linux GTK/Qt;
    # the Cocoa backend ignores it. macOS dock-icon setting is handled inside
    # ``on_start`` via ``set_window_icon`` → ``NSApplication.setApplicationIconImage_``.
    try:
        webview.start(on_start)
    finally:
        if server is not None:
            server.should_exit = True
        if server_proc is not None:
            try:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait(timeout=2)
            except Exception as exc:
                print(f"[viewer] server_proc shutdown error: {exc}", file=sys.stderr)
        if server_thread is not None:
            server_thread.join(timeout=5)
            if server_thread.is_alive():
                print("[viewer] uvicorn thread did not exit cleanly", file=sys.stderr)


def _collect_reload_dirs() -> list[str]:
    """Directories uvicorn should watch for hot-reload.

    Always: the viewer package + the bundled web assets. Then: the source
    directory of every registered plugin's run() module, so editing an
    external plugin (e.g. omnipose's ocdkit_plugin.py) triggers a reload too.
    """
    dirs = [str(Path(__file__).resolve().parent)]
    if WEB_DIR.exists():
        dirs.append(str(WEB_DIR))
    seen = set(dirs)
    for plugin in REGISTRY.all():
        run_fn = getattr(plugin, "run", None)
        mod_name = getattr(run_fn, "__module__", None) if run_fn else None
        module = sys.modules.get(mod_name) if mod_name else None
        mod_file = getattr(module, "__file__", None) if module else None
        if not mod_file:
            continue
        try:
            pkg_dir = str(Path(mod_file).resolve().parent)
        except Exception:
            continue
        if pkg_dir not in seen:
            dirs.append(pkg_dir)
            seen.add(pkg_dir)
    return dirs


def _autoload_plugins(extra: list[str] | None = None) -> None:
    """Discover entry-point plugins and load extras specified via --plugin.

    Always ensures the bundled ``threshold`` reference plugin is registered.
    """
    REGISTRY.discover()
    from .plugins.threshold import plugin as threshold_plugin
    if threshold_plugin.name not in REGISTRY:
        REGISTRY.register(threshold_plugin)
    for spec in extra or []:
        module_path, _, attr = spec.partition(":")
        if not attr:
            print(f"[viewer] --plugin {spec!r} must be 'module:attr'", file=sys.stderr)
            continue
        try:
            import importlib
            mod = importlib.import_module(module_path)
            obj = getattr(mod, attr)
        except Exception as exc:
            print(f"[viewer] failed to load --plugin {spec!r}: {exc}", file=sys.stderr)
            continue
        if not isinstance(obj, SegmentationPlugin):
            print(
                f"[viewer] --plugin {spec!r} resolved to {type(obj).__name__}, "
                "expected SegmentationPlugin",
                file=sys.stderr,
            )
            continue
        if obj.name in REGISTRY:
            REGISTRY.unregister(obj.name)
        REGISTRY.register(obj)


# Backwards-compatible alias for older tests.
_autoload_builtin_plugins = _autoload_plugins
