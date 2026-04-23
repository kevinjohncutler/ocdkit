"""Entry point for ``python -m ocdkit``."""

from .cli.main import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
