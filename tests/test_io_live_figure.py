"""Tests for ocdkit.io.LiveFigure — the generic lazy live-figure handle."""
from ocdkit.io import LiveFigure


class TestLiveFigure:
    def test_construction_is_lazy(self):
        calls = []

        def render(*, as_svg=True, out=None):
            calls.append((as_svg, out))
            return "SVG"

        lf = LiveFigure(render=render, title="t")
        assert calls == []          # constructing the handle does NOT render
        assert lf.title == "t"

    def test_to_svg_renders_once_and_memoizes(self):
        calls = []

        def render(*, as_svg=True, out=None):
            calls.append((as_svg, out))
            return object()

        lf = LiveFigure(render=render)
        a = lf.to_svg()
        b = lf.to_svg()
        assert a is b                       # memoized, same object
        assert calls == [(True, None)]      # rendered exactly once, as_svg=True/out=None

    def test_save_routes_as_svg_by_extension(self):
        seen = {}

        def render(*, as_svg=True, out=None):
            seen["as_svg"], seen["out"] = as_svg, out
            return out

        lf = LiveFigure(render=render)
        lf.save("/tmp/x.pptx")
        assert seen == {"as_svg": True, "out": "/tmp/x.pptx"}
        lf.save("/tmp/x.svg")
        assert seen["as_svg"] is True
        lf.save("/tmp/x.png")
        assert seen["as_svg"] is False

    def test_repr_html_prefers_live_display(self):
        class Live:
            def _repr_html_(self):
                return "<div>live</div>"

        rendered = []
        lf = LiveFigure(render=lambda **k: rendered.append(1), display=lambda: Live())
        assert lf._repr_html_() == "<div>live</div>"
        assert rendered == []               # live display path does not render the static figure

    def test_repr_html_falls_back_to_static_svg(self):
        class Svg:
            def _repr_html_(self):
                return "<svg>static</svg>"

        lf = LiveFigure(render=lambda *, as_svg=True, out=None: Svg())  # no display supplied
        assert "static" in lf._repr_html_()

    def test_meta_and_repr(self):
        lf = LiveFigure(render=lambda **k: None, title="A", foo=1, bar=2)
        assert lf.meta == {"foo": 1, "bar": 2}
        assert "A" in repr(lf)
