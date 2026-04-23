"""CLI for ``python -m ocdkit.viewer``.

Subcommands:

* ``serve``    — run the FastAPI web server (HTTP/HTTPS).
* ``desktop``  — launch a pywebview desktop window with embedded uvicorn.
* ``plugins``  — list registered plugins.
* ``describe`` — print one plugin's manifest as JSON.

Entry-point plugins (declared via ``[project.entry-points."ocdkit.plugins"]``)
are auto-discovered. Use ``--plugin module:attr`` to load extras.
"""

from __future__ import annotations

import argparse
import sys


def _add_common_plugin_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE:ATTR",
        help="Extra plugin to load (e.g. omnipose.ocdkit_plugin:plugin). "
        "Repeatable. Entry-point plugins are auto-discovered separately.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Window/tab/docs title (default: 'ocdkit.viewer'). "
        "Also overridable via OCDKIT_VIEWER_TITLE env var.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ocdkit.viewer",
        description="Generic image-viewer + segmentation-editor toolkit.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # serve
    serve = sub.add_parser("serve", help="Run the viewer HTTP/HTTPS server.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--ssl-cert", default=None)
    serve.add_argument("--ssl-key", default=None)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument(
        "--https-dev",
        action="store_true",
        help="Provision a temporary self-signed localhost cert (requires openssl).",
    )
    _add_common_plugin_arg(serve)

    # desktop
    desktop = sub.add_parser("desktop", help="Launch a pywebview desktop window.")
    desktop.add_argument("--host", default="127.0.0.1")
    desktop.add_argument("--port", type=int, default=0, help="0 = auto-pick a free port")
    desktop.add_argument("--ssl-cert", default=None)
    desktop.add_argument("--ssl-key", default=None)
    desktop.add_argument("--reload", action="store_true")
    desktop.add_argument(
        "--snapshot",
        metavar="PNG_PATH",
        help="Capture the viewer canvas to the given PNG file and exit.",
    )
    desktop.add_argument("--snapshot-timeout", type=float, default=4.0)
    desktop.add_argument(
        "--eval-js",
        dest="eval_js",
        default=None,
        help="JavaScript snippet to evaluate after the viewer loads (testing).",
    )
    _add_common_plugin_arg(desktop)

    # plugins / describe
    sub.add_parser("plugins", help="List registered plugins and exit.")
    desc = sub.add_parser("describe", help="Print a plugin's manifest JSON and exit.")
    desc.add_argument("name")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command in (None, "serve"):
        from .app import run_server
        run_server(
            host=getattr(args, "host", "0.0.0.0"),
            port=getattr(args, "port", 8765),
            ssl_cert=getattr(args, "ssl_cert", None),
            ssl_key=getattr(args, "ssl_key", None),
            reload=getattr(args, "reload", False),
            https_dev=getattr(args, "https_dev", False),
            plugins=getattr(args, "plugin", None) or [],
            title=getattr(args, "title", None),
        )
        return

    if args.command == "desktop":
        from .app import run_desktop
        port = getattr(args, "port", 0)
        run_desktop(
            host=getattr(args, "host", "127.0.0.1"),
            port=port if port and port > 0 else None,
            ssl_cert=getattr(args, "ssl_cert", None),
            ssl_key=getattr(args, "ssl_key", None),
            reload=getattr(args, "reload", False),
            plugins=getattr(args, "plugin", None) or [],
            snapshot_path=getattr(args, "snapshot", None),
            snapshot_timeout=getattr(args, "snapshot_timeout", 4.0),
            eval_js=getattr(args, "eval_js", None),
            title=getattr(args, "title", None),
        )
        return

    if args.command == "plugins":
        from .app import _autoload_plugins
        from .plugins.registry import REGISTRY
        _autoload_plugins()
        if not REGISTRY.names():
            print("(no plugins registered)")
            return
        for p in REGISTRY.all():
            print(f"{p.name}  {p.version}  — {p.description}")
        return

    if args.command == "describe":
        import json
        from .app import _autoload_plugins
        from .plugins.registry import REGISTRY
        _autoload_plugins()
        try:
            plugin = REGISTRY.get(args.name)
        except KeyError:
            print(f"plugin {args.name!r} not registered", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(plugin.manifest(), indent=2))
        return
