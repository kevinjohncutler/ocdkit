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

    from .routers import index, log, mask, plugin, segment, session_routes, system

    @asynccontextmanager
    async def lifespan(_: FastAPI):
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

    return app


# -----------------------------------------------------------------------------
# Server / desktop launchers
# -----------------------------------------------------------------------------


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
    reload: bool = False,
    https_dev: bool = False,
    plugins: list[str] | None = None,
    title: str | None = None,
) -> None:
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

    try:
        import uvicorn
    except ImportError:
        print(
            "fastapi + uvicorn required. Install with: pip install 'ocdkit[viewer]'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    scheme = "https" if (ssl_cert and ssl_key) else "http"
    print(f"[viewer] serving at {scheme}://{host}:{port}", flush=True)
    print(f"[viewer] active plugin: {ACTIVE_PLUGIN.name() or '(none)'}", flush=True)
    print(f"[viewer] registered: {REGISTRY.names() or '(none)'}", flush=True)

    if reload:
        reload_dirs = [str(Path(__file__).resolve().parent)]
        if WEB_DIR.exists():
            reload_dirs.append(str(WEB_DIR))
        uvicorn.run(
            "ocdkit.viewer.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=reload_dirs,
            ssl_certfile=ssl_cert,
            ssl_keyfile=ssl_key,
            log_level="info",
        )
        return

    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
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
        AppIdentity, apply_early_dark_mode, setup_platform, set_window_icon,
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

    # macOS gets the native NSVisualEffectView frosted-glass effect via
    # ``vibrancy=True``. ``transparent=True`` (alpha-capable window) is
    # required for the body's translucent background to actually show through
    # to the OS material rather than over a white background. On Windows /
    # Linux pywebview ignores ``vibrancy``; ``transparent`` is supported.
    window = webview.create_window(
        viewer_title(),
        url=window_url,
        width=1024,
        height=768,
        resizable=True,
        hidden=automation_needed,
        transparent=True,
        vibrancy=True,
        background_color="#111111",
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
