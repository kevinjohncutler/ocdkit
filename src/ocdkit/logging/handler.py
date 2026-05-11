"""
ocdkit.logging
==============

Rich-based colored logging with per-module color control.

Provides a single ``get_logger`` entry point that lazily installs a
project-wide :class:`~rich.logging.RichHandler` on the root logger.
Colors are assigned per logger name (or any prefix), giving each
submodule its own color in terminal output.

Usage
-----
::

    from ocdkit.logging import get_logger

    log = get_logger(__name__, color="#5c9edc")
    log.info("loaded %d items", n)

Works out of the box on macOS, Linux, Windows Terminal, VS Code,
Jupyter, and SSH sessions.  Rich auto-detects capabilities.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Lazy Rich imports — Rich is only loaded when the handler is first needed.
# This keeps import time near zero for libraries that configure but don't
# immediately log.
# ---------------------------------------------------------------------------
_rich_handler: Optional[logging.Handler] = None
_console = None

# Module-name → hex color.  Populated by ``set_color`` / ``get_logger``.
_PALETTE: Dict[str, str] = {}
_DEFAULT_COLOR = "#cccccc"

# Override Rich's default logging styles. Defaults are too dark on dark
# terminals (info = navy "blue", time = dim cyan that disappears).
_LEVEL_THEME: Dict[str, str] = {
    "log.time": "grey62",
    "log.path": "grey42",
    "logging.level.debug": "grey50",
    "logging.level.info": "bright_cyan",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold reverse red",
}

# ---------------------------------------------------------------------------
# Color registration
# ---------------------------------------------------------------------------

def set_color(name: str, hex_color: str) -> None:
    """Register *hex_color* for logger *name* (or any prefix thereof).

    This can be called before or after ``get_logger``; colors are resolved
    at format time, so late registration works fine.
    """
    _PALETTE[name] = hex_color
    # If the handler is already installed, update the console theme live.
    if _console is not None:
        _console.push_theme(_build_theme())


def set_colors(mapping: Dict[str, str]) -> None:
    """Bulk-register colors from a ``{name: hex}`` dict."""
    _PALETTE.update(mapping)
    if _console is not None:
        _console.push_theme(_build_theme())


# ---------------------------------------------------------------------------
# Handler setup (idempotent)
# ---------------------------------------------------------------------------

def _build_theme():
    from rich.theme import Theme
    # Module palette overrides level theme if there's a (very unlikely) name clash.
    return Theme({**_LEVEL_THEME, **_PALETTE}, inherit=True)


def _install_jupyter_css() -> None:
    """Zero out the default <pre> margin on Rich's Jupyter output.

    In JupyterLab, Rich emits one ``<pre style="line-height:normal;...">``
    per log record. That style does not include ``margin``, so the browser
    default (~1em top + 1em bottom) shows up as a large gap between every
    consecutive log line. One CSS rule fixes it for the rest of the
    session.
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return
        from IPython.display import display, HTML
    except ImportError:
        return

    display(HTML(
        "<style>"
        ".jp-RenderedHTML > pre[style*='line-height:normal'],"
        ".rendered_html > pre[style*='line-height:normal']"
        "{ margin: 0; }"
        "</style>"
    ))


def _ensure_handler(level: str | int = "INFO") -> logging.Handler:
    """Install the Rich handler on the root logger exactly once."""
    global _rich_handler, _console

    if _rich_handler is not None:
        return _rich_handler

    from rich.console import Console
    from rich.logging import RichHandler

    _console = Console(theme=_build_theme(), force_terminal=True)
    _install_jupyter_css()

    class _ColorByModule(logging.Filter):
        """Wrap each record's message in the module's theme color."""

        def filter(self, record: logging.LogRecord) -> bool:
            name = record.name
            # Walk from most-specific to least-specific prefix to find a color.
            style = _DEFAULT_COLOR
            best_len = 0
            for key in _PALETTE:
                if name == key or name.startswith(key + ".") or name.startswith(key + ":"):
                    if len(key) > best_len:
                        style = key
                        best_len = len(key)

            txt = record.getMessage()
            record.msg = f"[{style}]{txt}[/{style}]"
            record.args = ()  # prevent double %-substitution
            return True

    _rich_handler = RichHandler(
        console=_console,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True,
        show_path=False,                  # right-edge path column truncated badly in jupyter
        omit_repeated_times=False,        # always show time — the blank time-column was the worst spacing offender
        log_time_format="[%H:%M:%S]",     # short HH:MM:SS instead of locale date+time
    )
    _rich_handler.addFilter(_ColorByModule())

    root = logging.getLogger()
    root.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))

    # Remove any existing handlers to avoid duplicate output, but only
    # if we are the first ocdkit setup.  Other libraries' handlers are
    # left alone if they were added *after* us.
    root.handlers.clear()
    root.addHandler(_rich_handler)

    return _rich_handler


# ---------------------------------------------------------------------------
# Silence noisy third-party loggers
# ---------------------------------------------------------------------------

def silence(*names: str, level: int = logging.WARNING) -> None:
    """Set third-party loggers to *level* so they don't clutter output.

    Call once at startup::

        from ocdkit.logging import silence
        silence("xmlschema", "bfio", "OpenGL", "qdarktheme", "mip")
    """
    for name in names:
        logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_logger(
    name: str | None = None,
    color: str | None = None,
) -> logging.Logger:
    """Return a :class:`logging.Logger` with the shared Rich handler.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__``.
    color : str, optional
        Hex color (e.g. ``"#5c9edc"``) to use for this logger's output.
        Equivalent to calling ``set_color(name, color)`` first.
    """
    if color and name:
        set_color(name, color)

    level = os.environ.get("OCDKIT_LOG_LEVEL",
                           os.environ.get("LOG_LEVEL", "INFO"))
    _ensure_handler(level=level)

    return logging.getLogger(name or "ocdkit")


# ---------------------------------------------------------------------------
# tqdm redirect
# ---------------------------------------------------------------------------

class TqdmToLogger(io.StringIO):
    """File-like wrapper that sends tqdm progress output to a logger.

    Usage::

        from ocdkit.logging import get_logger, TqdmToLogger
        from tqdm import tqdm

        log = get_logger(__name__)
        tqdm_out = TqdmToLogger(log)
        for item in tqdm(items, file=tqdm_out):
            ...
    """

    def __init__(self, logger: logging.Logger, level: int | None = None):
        super().__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self._buf = ""

    def write(self, buf: str) -> int:
        self._buf = buf.strip("\r\n\t ")
        return len(buf)

    def flush(self) -> None:
        if self._buf:
            self.logger.log(self.level, self._buf)
            self._buf = ""
