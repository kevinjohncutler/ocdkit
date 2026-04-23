"""``ocdkit migrate`` — move a legacy ``~/.<app>/`` dotfolder to the
platform-appropriate user-data directory.

Thin CLI wrapper over :func:`ocdkit.utils.paths.migrate_legacy_dotfolder`.
Designed for users who prefer an explicit one-shot over the automatic
import-time migration that downstream packages typically wire up.

Example::

    $ ocdkit migrate omnipose
    moved ~/.omnipose → /Users/you/Library/Application Support/omnipose

    $ ocdkit migrate omnipose --legacy cellpose     # custom source dotfolder
    $ ocdkit migrate omnipose --dry-run             # just print what would happen
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.paths import migrate_legacy_dotfolder, user_data


def add_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``migrate`` subcommand."""
    p = subparsers.add_parser(
        "migrate",
        help="Move a legacy ~/.<app>/ folder to the platform user-data dir.",
        description="Move a legacy ~/.<app>/ folder to the platform user-data dir.",
    )
    p.add_argument("app", help="Target application name (platformdirs key).")
    p.add_argument(
        "--legacy", default=None,
        help="Legacy dotfolder basename without the dot "
             "(default: same as app).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without moving anything.",
    )
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> int:
    legacy = args.legacy or args.app
    src = Path.home() / f".{legacy}"
    dst = user_data(args.app, create=False)
    marker = dst / ".migrated"

    if args.dry_run:
        if marker.exists():
            print(f"already migrated ({marker})")
            return 0
        if not src.exists():
            print(f"no legacy folder at {src} — nothing to do")
            return 0
        child_count = sum(1 for _ in src.iterdir())
        existing = [p for p in dst.iterdir() if p.name != ".migrated"] if dst.exists() else []
        if existing:
            print(f"would SKIP: destination {dst} already has {len(existing)} entries")
        else:
            print(f"would move {child_count} entries:  {src}  →  {dst}")
        return 0

    result = migrate_legacy_dotfolder(args.app, legacy=legacy)
    if result is None:
        print(f"no legacy folder at {src} — nothing to do")
        return 0
    if src.exists():
        # Helper left src in place because dest already had content.
        print(f"skipped: {dst} already has content; "
              f"{src} preserved for manual review")
    else:
        print(f"moved {src} → {dst}")
    return 0
