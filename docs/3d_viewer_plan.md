# 3D segmentation + visualization for the omnipose / ocdkit viewer

Scope + phased implementation plan. Test case: spacetime cells
`/Volumes/DataDrive/3D_spacetime/linked/a_baylii/dnaA_xy1_crop.tif`
(133 time-frames × 302 × 302; masks have 40 labels = spacetime tubes;
`_links.txt` = division lineage, e.g. `1,7 / 1,8` = cell 1 → daughters 7,8).

## Key findings (from code survey)

- **omnipose core is already dimension-generic.** Flows, affinity graph, distance,
  and Euler-integration points compute as `(dim, *spatial)` for `dim=2` or `3`;
  `eval.py` already has a `do_3D` path. `kernel_setup(dim)` yields 8 steps (2D) or
  26 steps (3D).
- **The two presentation layers are 2D-locked:**
  - Segmenter `src/omnipose/gui/_segmenter.py` squeezes 3D input to 2D
    (`arr.mean(axis=-1)`, ~L156), guards affinity on `mask.ndim==2` (~L432),
    emits `[y,x]` points (~L862).
  - Frontend `ocdkit/src/ocdkit/viewer/web/app.js` (~11.5k LOC) assumes a single
    `H×W` plane: 3×3 affine viewport (~L5754), 2D textures, 8-step affinity
    (~L6065), `[y,x]` points (~L8727).
- **One genuine gap:** `ocdkit/plot/color.py:rgb_flow` maps `dy + dx·i` to a complex
  plane → no 3D analog (can't HSV a 3-vector that way).
- **Two distinct graph objects** for spacetime: the spatial **affinity graph**
  (26-neighbor voxel connectivity, already in core) and the temporal **lineage /
  trajectory** (label→label division edges, currently only in the `_links.txt`
  sidecar — not emitted by the viewer at all).

## Architecture decision (settled)

- **2D / 2.5D slice view:** keep the existing hand-rolled **WebGL2** pipeline in
  `app.js`, driven by a slice index. When depth==1 it must be byte-identical to
  today (no regression).
- **True 3D view:** a NEW, isolated **raw-WebGPU** layer (own canvas), **no
  three.js.** Rationale: the colormaps repo started on three.js (`js/3d-view.js`)
  then deliberately replaced it with a hand-rolled raw-WebGPU renderer
  (`js/webgpu-view.js`, header: *"Replaces Three.js — native P3 HDR support
  (rgba16float + display-p3)"*) for HDR control; ocdkit's own
  `plot/web/colormap_image.js` already follows the same "WebGPU(HDR) → WebGL2(SDR)"
  hand-rolled idiom. Reuse:
  - **colormaps `js/webgpu-view.js`** — LINE / POINT / THICK_LINE / SURFACE
    pipelines + `mat4x4 mvp` camera + display-p3 HDR canvas → affinity edges,
    points, trajectory polylines, bbox/axes, label isosurfaces.
  - **hostpkg `prototyping/sim3d` ray-march** (AABB slab + MIP/additive/mean +
    `Data3DTexture`) → the volume. Its TSL needs rewriting to plain WGSL, but the
    algorithm ports verbatim (already proven, see below).
- **WebGPU-only for the 3D view is acceptable** (current Chrome/Safari/Edge have
  WebGPU); a WebGL2 ray-march fallback can be added later if needed. The 2.5D tier
  covers non-WebGPU browsers.

## Testing strategy (verified)

- **WGSL / pipelines → wgpu-native in Python, headless, no browser.** Proven on
  this MBP (Apple M5 Max, Metal, `wgpu` 0.31.0): the hostpkg ray-march algorithm
  ported to raw WGSL renders MIP/mean/emission-absorption from two orthographic
  directions and matches `np.max` / `np.mean` / `1-prod(1-v)` exactly to fp16
  (~5e-4). Harness: `outputs/repro/wgpu_raymarch_headless/proof.py`. Extend it for
  every new WGSL stage (label colormap, blend, streamlines) BEFORE it ships.
- **JS host logic** (camera matrices, buffer packing, slice indexing, payload
  decode) → Node CPU-harness pattern (load the real module, call internals on
  synthetic inputs, diff vs NumPy).
- **True browser integration** (canvas context-type locking, live EDR headroom):
  real Google Chrome via Playwright `channel="chrome" --headless=new
  --enable-unsafe-webgpu`, or Deno (native `navigator.gpu`), or the pywebview
  launcher for visual HDR checks. Playwright's *bundled* Chromium has no WebGPU.

---

## Phase 0 — Backend goes dimension-generic  (~2–4 days)

Files: `omnipose/src/omnipose/gui/_segmenter.py`, `omnipose/src/omnipose/gui/ocdkit_plugin.py`,
`ocdkit/src/ocdkit/plot/color.py`.

1. **`ocdkit_plugin.py`:** add a `WidgetSpec` toggle `do_3D` (a.k.a. "Volume / 3D");
   `_coerce_settings` forwards it. The host viewer surfaces it.
2. **`segment()`:** when `do_3D`, treat input as `(Z,H,W)` volume — do NOT
   `mean(axis=-1)`. Thread `dim=3` into `model.eval` (uses the existing `do_3D`
   path). Distinguish `(H,W,C)` color vs `(Z,H,W)` volume via the explicit flag,
   not a shape heuristic.
3. **Cache 3D-shaped products:** `dP (3,Z,H,W)`, `dist (Z,H,W)`,
   `affinity (26,Z,H,W)`, `p (3,Z,H,W)`, `mask (Z,H,W)`.
4. **`get_affinity_graph_payload()`:** drop the `mask.ndim != 2` guard; emit
   `steps (26,3)` + a `dim`/`depth` field. **Do NOT ship the whole 26×Z×H×W array**
   (~300 MB for the test stack) — provide a **per-slice** endpoint
   (`affinity(z)`) the frontend requests on demand, plus gzip. (Slice view needs
   only one z at a time; true-3D affinity is region-on-demand — see P3.)
5. **`get_points_payload()`:** emit `[z,y,x]` interleaved (3/point) + `dim` field.
6. **Flow visualization:**
   - *Cheap, ships in P0:* per-slice in-plane RGB — run existing `rgb_flow` on
     `dP[1:, z]` (the `(dy,dx)` components) for each z → a depth-stack of PNGs
     (or a tiled atlas). Emit as a flow volume.
   - *For true-3D later:* emit the raw `(3,Z,H,W)` flow as a typed-array payload
     (downsample for size). Add `rgb_flow_3d(dP)` to `color.py` — a directional
     colormap mapping a unit 3-vector → RGB (e.g. abs-components or a spherical
     map) for the volumetric flow-color option.
7. **NEW `get_trajectory_payload()`:** parse the `_links.txt` sidecar when present,
   else compute. Emit `{ centroids: per-label per-frame [t,y,x],
   edges: [[parent,daughter],...] }`. This is the "trajectories" overlay.
8. **Payload sizing:** narrow mask dtype when `max_label` fits (uint8/uint16),
   gzip volume payloads. Mask `(Z,H,W)` uint32 = ~48 MB raw for the test stack.

Tests: plain pytest on the spacetime tif — assert payload shapes/dtypes. No GPU.

## Phase 1 — 2.5D slice scrolling  (~3–5 days)  ← biggest value/effort ratio

Files: `app.js` (volume data model + slice upload), new `js/volume-nav.js`,
`html/sidebar.html` / controls (slice slider), css. Existing WebGL2 path only.

1. **Volume data model:** hold image/mask/overlay **volumes** `(Z,H,W)` + `currentZ`
   + axis label ("t" for spacetime). Decode the new volume payloads. depth==1 →
   exactly current behavior.
2. **Slice navigation:** scrollwheel (over canvas), slider, arrow keys, a
   `z: 12 / 133` readout. On change → re-upload the active slice into the existing
   2D textures (base, mask RG, outline, flow, distance) + redraw. One H×W upload =
   cheap.
3. **Per-slice overlays through the existing renderers:**
   - Mask: index volume at z → existing Uint32 H×W path; recompute outline per slice
     (lazy cache).
   - Affinity: request `affinity(z)`; draw in-plane steps (`dz==0`) as GL_LINES;
     optionally mark through-plane steps (`dz≠0`) as dots.
   - Points: filter `[z,y,x]` to `|pz - z| < 0.5` → existing GL_POINTS.
   - Flow / distance: index the per-slice PNG stack.
   - Trajectories: project centroid tracks to 2D; draw the polyline up to current t
     with a marker at frame t (reuse the lines/points overlay).
4. **No-regression guard:** explicit test that depth==1 output matches current.

Delivers "scroll through the stack and see flow / affinity / points / trajectories"
on the proven-stable WebGL2 path, with zero WebGPU dependency.

## Phase 2 — True 3D volume view  (~1–2 weeks)

New raw-WebGPU layer (no three.js), under `viewer/web/js/volume3d/`:

- `renderer.js` — adapter/device init (feature-detect WebGPU; hide the 3D toggle if
  absent), HDR `rgba16float` display-p3 canvas (copy `colormap_image.js` /
  colormaps `webgpu-view.js`), render loop.
- `camera.js` — arcball/orbit + perspective `mat4` (port from colormaps
  `webgpu-view.js`; confirm it has mouse-drag orbit, add if not).
- `raymarch.wgsl.js` — the proven ported ray-march; MIP / additive / mean selected
  by uniform; **Z-scale uniform** for time anisotropy.

1. **View-mode toggle** (2.5D ⇄ 3D) mounting/unmounting the 3D canvas over the same
   viewport region.
2. **Volumes:** upload raw intensity (`r16float`/`r8`) + a **label volume**
   (`r8` for ≤255 ids, else `r32uint`) as a second 3D texture; sample both, colormap
   labels via the existing palette LUT, blend (opacity slider). MIP/additive/mean
   buttons + density/threshold sliders (mirror hostpkg uniforms).
3. **Picking:** raycast → first-hit label for hover/highlight (mirror the 2D
   `labelAt` pattern).

Every WGSL change validated by extending `proof.py` (wgpu-native) before shipping.

## Phase 3 — 3D overlays, incremental  (~1–2 weeks)

Reuse colormaps `webgpu-view.js` LINE / THICK_LINE / POINT / SURFACE pipelines.

- **Trajectories / lineage (do first — highest impact for spacetime):** 3D polylines
  of per-label centroid tracks along the time axis + division branch points +
  endpoint markers → the lineage shows as branching tubes. THICK_LINE + POINT.
- **Points:** 3D scatter of cell sinks `[z,y,x]` → POINT pipeline.
- **Flow:** start with subsampled 3D quiver (LINE); upgrade to streamlines
  (integrate the raw 3-vector field → polylines) or a directional-color volume
  (`rgb_flow_3d`).
- **Affinity (defer / decimate — the one real perf risk):** 26 steps × ~12M voxels
  is huge. Options: render on-demand around the hovered cell, a coarse decimated
  field, or keep affinity slice-only in true-3D. **Log any decimation** (no silent
  caps).
- Axes / bbox + time-scale UI.

---

## Cross-cutting risks

- **Memory** (133×302×302 ≈ 12M voxels): raw r16 ≈ 24 MB, label r8 ≈ 12 MB, flow
  3×f32 ≈ 145 MB (downsample for 3D), **affinity 26× ≈ 300 MB** (keep slice-only /
  region-on-demand — never ship whole).
- **Anisotropy:** time axis ≠ space; Z-scale uniform + UI control.
- **WebGPU availability:** 2.5D works everywhere (WebGL2); 3D needs WebGPU
  (feature-detected). Optional WebGL2 ray-march fallback later.
- **No-regression** on the 2D path is a hard requirement.

## Effort + suggested first PR

| Phase | Scope | Estimate |
|---|---|---|
| P0 | backend dim-generic | 2–4 days |
| P1 | 2.5D slice view (all overlays) | 3–5 days |
| P2 | true-3D volume (raw WebGPU) | 1–2 weeks |
| P3 | 3D overlays (lineage, points, flow, affinity) | 1–2 weeks |

**First PR = P0 + P1 (~1 week):** full 2.5D viewing of the spacetime stack with all
overlays, end-to-end testable on `dnaA_xy1_crop.tif`, no WebGPU dependency, no
regression risk. P2/P3 build the rotatable volume on top.
