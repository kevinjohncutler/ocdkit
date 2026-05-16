"""Standard palettes + style defaults for SVG plot construction.

One module, one place to look up "what's the standard tick fontsize"
or "give me a categorical palette."  Anything that previously appeared
as a magic constant inside a plot function should pull from here.

Conventions:
  * Sizes are in SVG user units (≈ px at 96 dpi).
  * Colors are hex strings unless noted.
  * Palettes are tuples (immutable) so callers can index without
    worrying about mutating shared state.

If a plot needs a value that differs from the default, override at
the call site — don't mutate the constants in this module.  For
project-wide style shifts (e.g. a dark theme), add a Theme preset
that returns a different ``AXIS_DEFAULTS`` dict rather than mutating
the existing one.
"""
from __future__ import annotations


# ─── default axis style ───────────────────────────────────────────────

AXIS_DEFAULTS = {
    "tick_length": 5.0,            # px — tick mark length perpendicular to spine
    "tick_label_size": 9.0,        # px — tick label font size
    "tick_label_padding": 3.0,     # px — gap between tick end and tick label
    "axis_label_size": 11.0,       # px — axis title font size
    "axis_label_padding": 10.0,    # px — gap between tick label and axis label
    "spine_width": 1.5,            # px — spine and tick stroke width
    "spine_color": "#000",         # default before apply_color_scheme
    "text_color": "#000",          # default before apply_color_scheme
}

# Plot/figure-level defaults.  Pulled out so layout-pass code (compute
# total canvas size, margins) doesn't have to hand-roll them.
FIGURE_DEFAULTS = {
    "background": "white",         # set via Figure.set_facecolor for themes
    "title_size": 14.0,            # px — figure suptitle font size
    "data_stroke_width": 2.0,      # px — default line plot stroke width
    "outline_width": 1.0,          # px — for image borders / spine outlines
}


# ─── categorical palettes ─────────────────────────────────────────────

# The 6-color palette used by hiprpy's key-slices SVG (top axis band
# coloring + bottom axis exc labels).  Hand-tuned for contrast on
# both dark and light backgrounds; not strictly perceptually uniform.
PALETTE_SIX = (
    "#7400EF",   # violet
    "#00FFC6",   # cyan-green
    "#3EFF00",   # green
    "#D6FF00",   # yellow-green
    "#FFB800",   # orange
    "#FF0000",   # red
)

# Tab10 — matplotlib's default categorical palette.  Use when you
# want results to look like a matplotlib plot.
PALETTE_TAB10 = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

# Okabe-Ito — 8-color colorblind-safe palette.  First choice when the
# audience might include CVD viewers (default ~8% of men, ~0.5% of women).
PALETTE_OKABE_ITO = (
    "#000000",   # black
    "#E69F00",   # orange
    "#56B4E9",   # sky blue
    "#009E73",   # bluish green
    "#F0E442",   # yellow
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#CC79A7",   # reddish purple
)


def categorical(n: int, palette: tuple[str, ...] = PALETTE_TAB10) -> tuple[str, ...]:
    """Return the first ``n`` colors from *palette*, cycling if needed."""
    if n <= 0:
        return ()
    out = []
    for i in range(n):
        out.append(palette[i % len(palette)])
    return tuple(out)


# ─── grayscale / neutral ──────────────────────────────────────────────

GRAYS = {
    "fg_strong": "#222",
    "fg_default": "#666",
    "fg_dim":    "#888",
    "fg_faint":  "#aaa",
    "bg_light":  "#f5f5f5",
    "bg_dark":   "#1a1a1a",
}


# ─── theme presets ────────────────────────────────────────────────────

# Themes are dicts of overrides for the apply_color_scheme call.
# Use as: ``figs_to_deck(figs, **THEME_DARK_DECK)`` or apply directly
# to a Figure: ``fig.apply_color_scheme(font=THEME_DARK['font'], ...)``.

THEME_LIGHT = {
    "font": GRAYS["fg_strong"],
    "axes": GRAYS["fg_default"],
}

THEME_DARK = {
    "font": "#ffeb3b",   # high-contrast yellow
    "axes": "#03a9f4",   # high-contrast cyan
}

THEME_PRINT = {
    "font": "#222",
    "axes": "#777",
}

# Convenience for the ``figs_to_deck`` kwargs that also set background.
THEME_DARK_DECK = {
    **THEME_DARK,
    "background_rgb": (20, 20, 20),
}

THEME_LIGHT_DECK = {
    **THEME_LIGHT,
    "background_rgb": (255, 255, 255),
}
