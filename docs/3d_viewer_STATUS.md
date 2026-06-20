# 3D viewer build — STATUS

Branch `feat/3d-viewer` in both repos (omnipose backend, ocdkit frontend).
Autonomous build against the plan in `3d_viewer_plan.md`. No push.

## Legend
[x] done + headless-tested · [~] in progress · [ ] todo

## P0 — backend dim-generic engine
- [x] Empirically mapped omnipose 3D core API (masks_to_flows→Result(.mu (3,Z,H,W), .dists), masks_to_affinity/spatial_affinity→(27,Z,H,W), kernel_setup(3)→26 non-centre steps). Script: `ocdkit/outputs/repro/3d_backend/explore_3d_core.py`
- [x] `omnipose/gui/_volume3d.py` payload engine: kernel_steps, flow_and_dist, affinity_volume, flow_rgb_slices (in-plane 2.5D), rgb_flow_3d (directional), dist_rgb_slices, points_from_p, parse_links, trajectories (centroid tracks + lineage), encode/decode (gzip+b64, label-dtype narrowing), build_bundle, bundle_from_files
- [x] `Segmenter.build_volume_bundle(...)` delegating method (routes flow solve through the GPU device)
- [x] pytest `omnipose/tests/test_volume3d.py` — 15 pass incl. real spacetime crop (mask roundtrip exact, flow (3,Z,H,W), affinity (26,Z,H,W), 36 lineage edges, every parent→2 daughters)
- [ ] (deferred to P1 wiring) `do_3D` widget + segment() volume model-eval path — the GT-mask bundle path covers the test case; model-3D is a later refinement

## P1 — 2.5D slice frontend (existing WebGL2)
- [ ] route layer: serve volume bundle + per-slice intensity/mask/affinity
- [ ] app.js volume data model + slice nav (scroll/slider/keys) + per-slice texture upload
- [ ] per-slice overlays (mask/outline, affinity in-plane, points near z, flow/dist slices, trajectory projection)
- [ ] Node CPU-harness tests for payload decode + slice indexing
- [ ] no-regression: depth==1 identical to today

## P2 — true-3D volume (raw WebGPU, no three.js)
- [x] (prereq) headless WGSL ray-march proven via wgpu-native: `ocdkit/outputs/repro/wgpu_raymarch_headless/proof.py`
- [ ] renderer.js (device init, HDR rgba16float display-p3 canvas, render loop)
- [ ] camera.js (arcball/perspective mat4, port from colormaps webgpu-view.js)
- [ ] raymarch.wgsl.js (MIP/additive/mean + Z-scale) + intensity & label 3D textures + blend
- [ ] view-mode toggle, picking; wgpu-native shader tests

## P3 — 3D overlays
- [ ] trajectories/lineage 3D polylines (THICK_LINE+POINT) — do first
- [ ] points 3D scatter; flow quiver/streamlines; affinity (region-on-demand)

## Verification channels
- backend: `python -m pytest omnipose/tests/test_volume3d.py`
- WGSL: `python ocdkit/outputs/repro/wgpu_raymarch_headless/proof.py`
- JS logic: Node CPU harness (P1+)
- browser integration (eventual, needs you): real Chrome / Deno / pywebview
