"""``ocdkit paths`` — print platform-appropriate user directories.

Defaults to showing ocdkit's own dirs; pass an app name to inspect another
package (e.g. ``ocdkit paths omnipose``).

Example::

    $ ocdkit paths
    ocdkit directories:
      config:  /Users/you/Library/Application Support/ocdkit
      data:    /Users/you/Library/Application Support/ocdkit
      cache:   /Users/you/Library/Caches/ocdkit
      state:   /Users/you/Library/Application Support/ocdkit
      log:     /Users/you/Library/Logs/ocdkit
"""

from __future__ import annotations

import argparse
import json
import sys

from ..utils.paths import user_cache, user_config, user_data, user_log, user_state

_RESOLVERS = {
    "config": user_config,
    "data": user_data,
    "cache": user_cache,
    "state": user_state,
    "log": user_log,
}


def add_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``paths`` subcommand."""
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
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> int:
    dirs = {
        kind: str(resolver(args.app, create=args.create))
        for kind, resolver in _RESOLVERS.items()
    }
    if args.format == "json":
        json.dump({"app": args.app, "dirs": dirs}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"{args.app} directories:")
        width = max(len(k) for k in dirs)
        for kind, path in dirs.items():
            print(f"  {kind:<{width}}  {path}")
    return 0
