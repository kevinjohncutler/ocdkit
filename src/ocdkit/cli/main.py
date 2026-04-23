"""Top-level ``ocdkit`` command dispatcher.

Subcommands register themselves by exposing an ``add_parser(subparsers)``
callable that wires a ``func(args)`` handler onto the returned parser.
New subcommands are added by importing their module below.
"""

from __future__ import annotations

import argparse
import sys

from . import paths as _paths_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocdkit",
        description="ocdkit — Obsessive Coder's Dependency Toolkit.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # Subcommand registration — one line per new subcommand.
    _paths_cmd.add_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
