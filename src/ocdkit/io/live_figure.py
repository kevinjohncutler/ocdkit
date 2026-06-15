"""Generic lazy handle for an interactive ("live") figure that can also be
exported statically.

A :class:`LiveFigure` carries two materializers and some metadata; it knows
nothing about what it draws (key-slices, grids, scatter, …) — the producer
supplies that knowledge as closures:

* ``render(*, as_svg, out)`` — the *static* materializer. Called with
  ``as_svg=True, out=None`` to get an :class:`ocdkit.io.SvgFigure` for export
  (this is what :func:`ocdkit.io.pptx.figs_to_deck` consumes), or with
  ``out=<path>`` to write a file (``.pptx`` / ``.svg`` / ``.png``).
* ``display()`` — the *interactive* materializer, returning an object with a
  ``_repr_html_`` (e.g. the live tile-server grid). Optional; when omitted the
  notebook repr falls back to a static SVG preview.

It is lazy: constructing one is free, so many can be queued cheaply
(``[scene.plot_key_slices(backend='live', hold=True) for scene in scenes]``)
and rendered only at export time. ``to_svg()`` memoizes, so display + export
don't double-render.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class LiveFigure:
    def __init__(
        self,
        *,
        render: Callable[..., Any],
        display: Optional[Callable[[], Any]] = None,
        title: Optional[str] = None,
        **meta: Any,
    ) -> None:
        self._render = render
        self._display = display
        self.title = title
        self.meta = meta
        self._svg = None  # memoized SvgFigure

    # ── export protocol (consumed by figs_to_deck via duck-typing) ──────────
    def to_svg(self):
        """Render (once) to an :class:`ocdkit.io.SvgFigure` and cache it."""
        if self._svg is None:
            self._svg = self._render(as_svg=True, out=None)
        return self._svg

    def render(self, *, as_svg: bool = False, out: Optional[str] = None):
        """Escape hatch to the underlying static materializer."""
        return self._render(as_svg=as_svg, out=out)

    def save(self, path: str):
        """Write the figure to ``path`` — ``.pptx`` / ``.svg`` / ``.png``."""
        p = str(path)
        as_svg = p.lower().endswith((".svg", ".pptx"))
        return self._render(as_svg=as_svg, out=p)

    # ── notebook display ────────────────────────────────────────────────────
    def _repr_html_(self):
        if self._display is not None:
            obj = self._display()
            rh = getattr(obj, "_repr_html_", None)
            if rh is not None:
                return rh()
            return repr(obj)
        # no live display supplied → static SVG preview
        svg = self.to_svg()
        rh = getattr(svg, "_repr_html_", None)
        return rh() if rh is not None else f"<LiveFigure {self.title or ''}>"

    def __repr__(self) -> str:
        return f"LiveFigure(title={self.title!r})"
