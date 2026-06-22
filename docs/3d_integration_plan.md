# Integrating the 3D volume viewer into the MAIN ocdkit viewer

The 3D viewer currently lives as a self-contained page (`viewer/web/volume.html` +
`js/volume3d*.js` + `raymarch.wgsl`/`overlay.wgsl`). This plans folding it into the
main viewer app (the `/` page driven by `app.js`).

## Prerequisites — status
- **three.js-free: CONFIRMED.** No imports / importmap / vendored three.js; the 3D
  view is pure raw-WebGPU with a quaternion arcball camera (free rotation, pan,
  dolly). Design spec met; plan keeps it that way.
- 3D component (`volume3d-gpu.js` + `raymarch.wgsl` + `volume3d-overlays*`),
  2.5D (`volume3d-view.js`), and decode (`volume3d.js`) are self-contained, consume
  the `POST /api/volume` bundle, and are headless-tested (wgpu-native + Node +
  Playwright). Camera incl. pan is done.
- Backend bundle via the `build_volume_bundle` plugin capability + `POST /api/volume`
  (intensity uint8, label volume, flow, distance, affinity, trajectories, recon
  points; lazy/opt-in heavy parts).

## Main viewer structure (mapped)
- `#viewer` (html/viewer.html:3) holds `<canvas id="canvas">` (2D WebGL2) + brush
  preview; `assets.py:_get_layout_markup()` assembles fragments into `#app`.
- File load: `file-navigation.js:requestImageChange()` → `POST /api/open_image` →
  `app.js:reinitializeForNewImage(config)`. CONFIG carries width/height/
  imageDataUrl/imagePath/directory. **No volume/3D notion exists yet.**
- **No view-mode system** (only tool modes). Need a lightweight 2D⇄2.5D⇄3D switch.
- Per-slice reuse points: `app.js:uploadBaseTextureFromCanvas()` (1396),
  `reinitializeForNewImage()` (10051), `maskValues` buffer (458), `draw()` (6848),
  `resizeCanvas()` (7220).
- Session: `session.py` SessionState + JS CONFIG/state; viewer state persisted to
  localStorage (`saveViewerState`).

## Recommended architecture
- **Embed as a view mode, keep raw-WebGPU.** Add a sibling `<canvas id="volumeViewer">`
  inside `#viewer` (separate element → separate GPU context, avoids the getContext
  lock), plus a 2D / 2.5D / 3D toggle. Mount the existing `VolumeGPU` on it. The
  2D mode keeps app.js untouched (no regression).
- Feature-detect WebGPU; if absent, hide 3D and keep 2D/2.5D (already graceful).

## Decisions (forks) — see questions
1. **2.5D slice view**: (a) reuse app.js per-slice — feed each z-slice + its mask to
   the existing 2D WebGL2 renderer so you get the full painting/ncolor/affinity
   toolset per slice (high value, more integration into the 11k-line app.js); vs
   (b) embed the standalone canvas2d `volume3d-view.js` (simple, isolated, fewer
   tools).
2. **MVP scope**: view-only first (load volume + masks → 2.5D + 3D), with in-app 3D
   *segmentation* as a later phase; vs include 3D segmentation now.
3. **Mask source**: existing masks (sidecar/precomputed) + `do_recon` points; vs run
   3D segmentation in-app (needs the affinity bug fix below).

## Phases
- **A — Server volume open** (~2–3 d): detect a 3D/multi-page tiff (or `.npz`) in
  `open_image`; set `config.isVolume` + volume metadata; serve the bundle (reuse
  `build_volume_bundle`). File navigator already lists files.
- **B — Embed 3D in the app** (~3–5 d): sibling `#volumeViewer` canvas + view-mode
  toggle; mount `VolumeGPU`; port the volume.html control panel (mode / density /
  zScale / image+label layers / shading / overlays / reset) into a main-viewer
  panel section. Wire resize.
- **C — 2.5D slice nav** (per decision #1): reuse app.js per-slice (slice slider +
  per-slice overlays + painting) ~3–5 d, OR embed `volume3d-view.js` ~1–2 d.
- **D — Polish** (~2–3 d): persist camera/slice/render settings in viewer state;
  volume-aware file navigator (next/prev volume); status/HUD.
- **E — Later**: cross-slice painting, saving 3D masks.

## Native 3D segmentation — FIXED (not divergence)
3D `affinity_seg` crashed for two reasons, both in `omnipose/core/masks.py` (not
`divergence`, whose batched-torch contract is correct and shared with `loss.py`):
1. `_get_affinity_torch` is batched-design (`(B,D,*spatial)` → `(S,B,*DIMS)`, with a
   downstream `.squeeze()` that drops B), but `masks.py` called it with **unbatched**
   inputs. Fix: add `[None]` (B=1) at the call site.
2. `flow_error` (flow-QC) called `masks_to_flows_batch` without `dim`, defaulting to
   `dim=2` on 3D data. Fix: pass `dim=maski.ndim`.
Verified end-to-end on synthetic + the real spacetime crop (2D + 3D). These edits
live in the WIP `masks.py` (uncommitted), to be committed with the core refactor.

## Risks
- app.js is 11k lines — embedding must keep the 2D path byte-identical; 3D is an
  additive sibling canvas + a view flag.
- WebGPU only for the 3D mode (feature-detected); 2D/2.5D cover the rest.
- 3D `affinity_seg` divergence bug blocks in-app 3D segmentation until fixed.
- One GPU context per canvas (3D canvas is its own element — already the pattern).
