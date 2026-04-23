"""Entry point for ``python -m ocdkit.cli`` (and ``python -m ocdkit``)."""

from .main import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
