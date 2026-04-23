"""``--paths`` helper: print an app's user directories and exit.

Provides three consumer APIs:

1. **:class:`PathsAction`** — drop-in argparse action, wired in a single
   line alongside regular options.  Mirrors argparse's built-in
   ``action='version'`` idiom — triggers print-and-exit when the flag
   appears on the command line.

2. **:func:`print_paths`** — plain function for programmatic use.

3. **``ocdkit paths``** — top-level subcommand that uses this module.

Downstream example (``omnirefactor``, ``hiprpy``, etc.)::

    import argparse
    from ocdkit.cli.paths import PathsAction

    parser = argparse.ArgumentParser(prog="omnirefactor")
    parser.add_argument("--paths", action=PathsAction, app="omnirefactor")
    # ... other args ...
    args = parser.parse_args()

Now ``omnirefactor --paths`` prints omnirefactor's XDG/platform-appropriate
directories and exits, identical in spirit to ``jupyter --paths``.

The action accepts optional ``extra`` for app-specific entries::

    from ocdkit.utils.paths import user_data
    parser.add_argument(
        "--paths", action=PathsAction, app="omnirefactor",
        extra={"models": str(user_data("omnirefactor", "models", create=False))},
    )
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Mapping

from ..utils.paths import user_cache, user_config, user_data, user_log, user_state

_RESOLVERS = {
    "config": user_config,
    "data": user_data,
    "cache": user_cache,
    "state": user_state,
    "log": user_log,
}


def print_paths(
    app: str,
    *,
    format: str = "text",
    extra: Mapping[str, str] | None = None,
    create: bool = False,
    file=None,
) -> None:
    """Print *app*'s user directories to *file* (default stdout).

    Parameters
    ----------
    app:
        Application name (used as the platformdirs key).
    format:
        ``"text"`` (default) or ``"json"``.
    extra:
        Optional mapping of additional ``{label: path}`` entries to append
        after the standard config/data/cache/state/log block.
    create:
        If True, ``mkdir -p`` the standard directories.
    """
    out = file or sys.stdout
    dirs: dict[str, str] = {
        kind: str(resolver(app, create=create)) for kind, resolver in _RESOLVERS.items()
    }
    if extra:
        dirs.update({str(k): str(v) for k, v in extra.items()})

    if format == "json":
        json.dump({"app": app, "dirs": dirs}, out, indent=2)
        out.write("\n")
    else:
        out.write(f"{app} directories:\n")
        width = max(len(k) for k in dirs)
        for kind, path in dirs.items():
            out.write(f"  {kind:<{width}}  {path}\n")


class PathsAction(argparse.Action):
    """argparse action — print the app's user directories and exit.

    Use like the built-in ``action='version'``::

        parser.add_argument("--paths", action=PathsAction, app="myapp")

    Optional keyword arguments:

    ``app`` (required)
        Application name.
    ``extra`` : Mapping[str, str]
        Extra ``{label: path}`` entries to show after the standard block.
    ``format`` : ``"text"`` | ``"json"``
        Output format. Default ``"text"``.
    """

    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        *,
        app: str,
        extra: Mapping[str, str] | None = None,
        format: str = "text",
        help: str | None = None,
        **kwargs,
    ):
        if help is None:
            help = f"Print {app} user directories and exit."
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
            **kwargs,
        )
        self._app = app
        self._extra = dict(extra) if extra else None
        self._format = format

    def __call__(self, parser, namespace, values, option_string=None):
        print_paths(self._app, format=self._format, extra=self._extra)
        parser.exit()


# ---------------------------------------------------------------------------
# ``ocdkit paths`` subcommand — thin wrapper around print_paths
# ---------------------------------------------------------------------------

def add_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``ocdkit paths`` subcommand."""
    p = subparsers.add_parser(
        "paths",
        help="Print user config/data/cache/state/log directories for an app.",
        description="Print user config/data/cache/state/log directories for an app.",
    )
    p.add_argument(
        "app", nargs="?", default="ocdkit",
        help="Target application name (default: ocdkit).",
    )
    p.add_argument(
        "--create", action="store_true",
        help="Create the directories if they don't exist (default: query only).",
    )
    p.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output format (default: text).",
    )
    p.set_defaults(func=_run)
    return p


def _run(args: argparse.Namespace) -> int:
    print_paths(args.app, format=args.format, create=args.create)
    return 0
