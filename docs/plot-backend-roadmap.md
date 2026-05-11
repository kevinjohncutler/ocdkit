# Plot backend roadmap

**Status:** Design discussion / pending decision.
**Owner:** kevin@kanvasbio.com
**Driver:** matplotlib accounts for ~80% of warm-call time in `hiprpy.plot.plot_spectra` (~115 ms of ~140 ms) and ~1.5 s of cold-import overhead. We also want native hover/tooltip behavior like the classification-debugger GUI, which matplotlib can't deliver inline.

## Context

`ocdkit.plot` and `hiprpy.plot` already own ~10 K LOC of plotting code:

| Package | LOC |
|---|---|
| `ocdkit.plot` (figure, grid, label, color, defaults, display, contour, export) | 1,935 |
| `hiprpy.plot` non-WGPU (datashade, cell, text, barcode, line, background, …) | 4,641 |
| `hiprpy.plot.wgpu` (lines, scatter, aggregators — already custom GPU primitives) | 4,783 |

What we still depend on matplotlib for is narrow: 2D Cartesian axes, ticks, labels, legends, image display, save-to-PNG/PDF/SVG, inline-in-Jupyter. **Not** 3D, animation, multi-backend abstraction (Qt/GTK/MacOSX), most chart types, complex tick locators/formatters, or any of the other ~180 K LOC of matplotlib's surface.

The driver question: replace matplotlib with what?

## Library survey (2026-05-07)

Shallow-cloned each candidate, counted Python LOC excluding tests, sphinx, sample data, and auto-generated validators:

| Library | Repo | Relevant Py LOC | Native code | Notes |
|---|---|---|---|---|
| matplotlib | — | 186 K | C extensions | What we're replacing |
| bokeh | 120 MB | 64 K | 150 K TypeScript | UI lives in TS frontend; Py is scene graph + serialization |
| plotly | 66 MB | 13 K relevant (+ ~600 K autogen) | JS bundle as data | Most LOC is auto-generated trait validators |
| datoviz | 137 MB | 8 K Py wrapper | 140 K C++ Vulkan engine | Heaviest binary footprint, fastest renderer |
| vispy | 11 MB | 32 K | None — pure Py + OpenGL | OpenGL stack, not WGPU |
| pygfx | 84 MB | 20 K | None — uses `wgpu-py` | Same backend ocdkit/hiprpy already use |

## Tradeoff summary

**Bokeh** — toolbar/logo *can* be removed (`figure(toolbar_location=None)`, `fig.toolbar.logo = None`), but the interaction model lives in a separate TypeScript frontend. To customize hover/zoom UX we'd be writing or forking TypeScript, not Python. The 150 K-LOC TS frontend is essentially the part we'd want to control, and it's the part we can't easily touch.

**Plotly** — same shape. Most LOC is auto-generated validators; the actual rendering and interaction is in the JS bundle. `displayModeBar=False` removes the toolbar but customizing hover beyond what their config exposes means reaching into JS.

**Datoviz** — Vulkan engine is fast and well-architected, but adds a 137 MB binary dependency and a different graphics stack from our existing wgpu-py code. Two graphics backends to maintain instead of one.

**Vispy** — pure Python and has scene + visuals modules that overlap heavily with what we want. But it's OpenGL via PyOpenGL, not WebGPU. We'd be running a second graphics stack alongside our wgpu-py code.

**pygfx** — uses *exactly* the `wgpu-py` we already depend on. Its `gfx.Lines`, `gfx.Mesh`, `gfx.Text`, `gfx.OrthographicCamera` primitives compose directly with our `DensityLineRenderer`. The most architecturally compatible third-party option.

**Roll our own** — given that we already have ~10 K LOC of plotting infrastructure, including a working WGPU line/scatter rasterizer, the increment to "complete 2D plotting library covering our actual needs" is roughly 3–5 K LOC. The scope is well-defined: axes, ticks, labels, legend, image display, hover, export.

## Recommendation

**Two-step.**

### Step 1: pygfx prototype (1 day)

Spike a `plot_spectra_pygfx` that uses `pygfx.Lines` + `pygfx.Text` + an `OrthographicCamera` to reproduce the current spectra layout. This is cheap because pygfx and our existing WGPU code share the same `wgpu-py` device — they can literally run in the same process without backend conflict.

What to evaluate:
- Visual quality vs. our current WGPU-rendered density lines (especially anti-aliasing)
- Text rendering quality (pygfx uses FreeType-rendered SDFs)
- Hover/picking — does pygfx's built-in picking suffice for our tooltip needs?
- Export — pygfx renders to a wgpu canvas; PNG export is straightforward, PDF/SVG would need our own rasterize-and-vectorize path

If pygfx covers ≥80% of our needs at ≥80% of the visual quality, **adopt pygfx**. We get hover, zoom, pan, picking essentially for free, plus the `pygfx` ecosystem (geometries, materials, post-processing).

### Step 2 (only if pygfx is insufficient): roll our own

Estimated scope, building on existing `hiprpy.plot.wgpu.lines`:

| Module | LOC est. | Notes |
|---|---|---|
| `ocdkit.plot.figure_v2` | 400 | New `Figure` class owning the wgpu canvas + axes layout; coexists with current matplotlib `figure()` so migration is piecewise |
| `ocdkit.plot.axis` | 800 | `LinearAxis`, `LogAxis` — tick locator + formatter + render-time placement. The matplotlib equivalent is ~3 K LOC; we don't need most of it |
| `hiprpy.plot.wgpu.text` | 600 | FreeType-rendered glyph atlas + WGSL textured-quad shader. The one piece of genuinely new rasterization work |
| `ocdkit.plot.legend` | 200 | Boxed layout: text + marker swatches |
| `ocdkit.plot.hover` | 300 | Mouse → data-coord lookup → DOM/Jupyter tooltip overlay. Generalize the classification-debugger pattern |
| `ocdkit.plot.export` | 400 | PNG (numpy→PIL), SVG (string templates), maybe PDF (skip if PIL→PNG covers science exports) |
| Migration: rewrite `plot_spectra`, `plot_image_grid`, `key_slice_grid`, `label_axes` against new primitives | ~600 | Mostly drop-in replacements at the call sites |
| **Total** | **~3.3 K** | New code, all in our control |

Plus: deletion of matplotlib-specific code paths in `hiprpy.plot` (~1 K LOC saved) and removal of matplotlib runtime dep.

### Why not pygfx + own axes?

A hybrid path is also viable: use pygfx for the rasterization layer (lines/text/transform/camera) and write our own axis/tick/legend/hover layer on top. That'd be the best of both — pygfx handles the rendering substrate, we own the plotting semantics. Estimated ~1.5 K LOC of new code on top of pygfx.

## Open questions

1. **Hover semantics** — does the classification-debugger tooltip pattern generalize cleanly, or do different plots need different hover content models?
2. **Export fidelity** — do we need true vector PDF, or is high-DPI PNG enough? Vector requires re-rasterizing axes/text on the CPU side, which is non-trivial.
3. **Notebook + standalone parity** — pygfx renders to a wgpu canvas that displays inline in Jupyter via `wgpu_jupyter`. Does that path work in VS Code's Jupyter extension and in standalone scripts (`savefig`-equivalent)?
4. **Remoting** — if we ever want a server-side render pipeline (for cloud GUIs), pygfx's WGPU backend can render headlessly; matplotlib `Agg` does the same. Bokeh/plotly's JS-frontend assumption is harder to remote.

## Concrete next action

Spike `plot_spectra_pygfx` using `scope.mixed_spectra[-1]` from `notebooks/hiprpy_demo_notebook.ipynb` as the test case. Compare visual output side-by-side with `plot_spectra_wgpu` and `plot_spectra_cpu`. Decide pygfx-vs-roll-own from that single comparison.

If the answer ends up being "pygfx + own axis layer", the new module structure would live in `ocdkit.plot.gfx_*` (parallel to the current matplotlib-based modules) so the migration is a per-call-site flip rather than a big-bang rewrite.
