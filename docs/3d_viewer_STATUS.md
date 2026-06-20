# 3D viewer build — STATUS

Branch `feat/3d-viewer` in both repos (omnipose backend, ocdkit frontend).
Autonomous build against the plan in `3d_viewer_plan.md`. No push.

## ✅ COMPLETE — P0–P3 implemented, all headless suites green
Final tally: omnipose pytest 15 · ocdkit pytest 11 (route 4, page 2, raymarch 4,
overlay 1) · Node 23 (mat4 5, volume3d 13, overlays3d 5) · wgpu-native proof PASS.
Commits — omnipose: 881d7b2 (single clean backend commit; see note). ocdkit:
3bd3ef9, a85201c, 5fafa99, abbf156, fba299a, ce5db13, 34d4ffb, 3274413, d77a547.
Not pushed.

NOTE (omnipose history): the first backend commit accidentally swept in a large
pre-staged refactor that was already in the index (test renames, networks/*, etc.
— NOT 3D-viewer work). Fixed by resetting feat/3d-viewer to main and re-committing
ONLY the 4 viewer files as 881d7b2. Your refactor is fully preserved, just back to
unstaged working-tree state (`git add` to re-stage). ocdkit commits were unaffected.

ONE thing needs a real-WebGPU browser (you) for final visual confirmation: the
live WebGPU device render of the 3D volume + overlays. Its shader, camera math,
and uniform/texture byte-layouts are each individually headless-verified
(wgpu-native + Node); only the in-browser device path can't run in headless
Chromium (no adapter). Open the viewer in Chrome/Safari, or use Deno, to confirm.
To view: serve the viewer and open `/static/volume.html?masks=<path>&raw=<path>`
(e.g. the spacetime stack), toggle 2.5D⇄3D.

## Legend
[x] done + headless-tested · [~] in progress · [ ] todo

## P0 — backend dim-generic engine
- [x] Empirically mapped omnipose 3D core API (masks_to_flows→Result(.mu (3,Z,H,W), .dists), masks_to_affinity/spatial_affinity→(27,Z,H,W), kernel_setup(3)→26 non-centre steps). Script: `ocdkit/outputs/repro/3d_backend/explore_3d_core.py`
- [x] `omnipose/gui/_volume3d.py` payload engine: kernel_steps, flow_and_dist, affinity_volume, flow_rgb_slices (in-plane 2.5D), rgb_flow_3d (directional), dist_rgb_slices, points_from_p, parse_links, trajectories (centroid tracks + lineage), encode/decode (gzip+b64, label-dtype narrowing), build_bundle, bundle_from_files
- [x] `Segmenter.build_volume_bundle(...)` delegating method (routes flow solve through the GPU device)
- [x] pytest `omnipose/tests/test_volume3d.py` — 15 pass incl. real spacetime crop (mask roundtrip exact, flow (3,Z,H,W), affinity (26,Z,H,W), 36 lineage edges, every parent→2 daughters)
- [ ] (deferred to P1 wiring) `do_3D` widget + segment() volume model-eval path — the GT-mask bundle path covers the test case; model-3D is a later refinement

## P1 — 2.5D slice frontend (existing WebGL2)
- [x] client logic `viewer/web/js/volume3d.js`: decodeArray (gzip/b64/typed incl float16), volume/rgb slice views, in/through-plane affinity split, affinity slice segments (deduped), points-near-slice, trajectory projection + lineage segments
- [x] Node CPU-harness `tests/js/volume3d.test.mjs` — 13 pass incl. exact Python->JS cross-language decode (uint8/float16/uint32). Runtimes: node v26 + deno at /opt/homebrew/bin
- [x] plugin capability `build_volume_bundle` (base.py contract + manifest flag; omnipose ocdkit_plugin.py + Segmenter.build_volume_bundle_from_files) + ocdkit route `POST /api/volume` (routers/volume.py, registered). Tests `tests/test_volume_route.py` — 4 pass incl. end-to-end through the real omnipose plugin on the spacetime stack (133x302x302, 40 labels, 36 lineage edges). Installed httpx + python-multipart for TestClient.
- [x] DECISION: built a self-contained volume-viewer PAGE (viewer/web/volume.html + js/volume3d-view.js) using volume3d.js, NOT editing the 11k-line app.js. Zero regression risk; one mount point for 2.5D + P2 WebGPU-3D. Existing app.js untouched, so "depth==1 identical" holds trivially.
- [x] 2.5D slice page: fetch POST /api/volume (or injected __TEST_BUNDLE__), canvas2d slice render (image/flow/distance/mask) + slice slider/wheel/arrow-keys; async gzip decode via DecompressionStream; refactored volume3d.js to expose bytesToTyped/b64ToBytes
- [x] per-slice overlays (affinity in-plane segments, points-near-z, trajectory projection + lineage dashed)
- [x] Playwright smoke test `tests/test_volume_page.py` — PASS in headless Chromium: renders non-blank, slices navigate, all 4 layers render, affinity+trajectory overlays draw, no JS errors

## P2 — true-3D volume (raw WebGPU, no three.js)
- [x] (prereq) headless WGSL ray-march proven via wgpu-native: `ocdkit/outputs/repro/wgpu_raymarch_headless/proof.py`
- [x] P2a: canonical `viewer/web/js/raymarch.wgsl` (perspective/ortho via invViewProj; MIP/additive/mean; intensity texture_3d<f32> + label texture_3d<u32>; in-shader label colour matching the 2.5D view; density/labelOpacity/showLabels uniforms). Validated by `tests/test_raymarch_wgsl.py` (wgpu-native, loads the EXACT shipped file) — 3 pass: MIP==np.max, mean==np.mean, label colour+blend exact.
- [x] P2b: pure camera math `viewer/web/js/mat4.js` (column-major, WebGPU [0,1]-depth perspective, lookAt, invert, orbit) — Node-tested `tests/js/mat4.test.mjs` (5 pass: invert∘mat=I, orbit geometry, invViewProj→centre-ray=forward, project/unproject round-trip). Browser host `viewer/web/js/volume3d-gpu.js` (raw WebGPU, no three.js: feature-detect→null, rgba16float display-p3 canvas, orbit camera, r32float intensity + uint label 3D textures with byte layout matching the verified wgpu-native harness, render loop, mode/density/labelOpacity/zScale + drag-orbit/wheel). Perspective integration verified: `tests/test_raymarch_wgsl.py` bridges the shipped mat4.js (via Node emit_camera.mjs) into the shader → centred cube projects to centred pixels (4 wgpu-native tests pass). Live WebGPU device render needs a real browser (user check).
- [x] P2c: wired into volume.html — second canvas #stage3d (separate context to avoid getContext locking), 2.5D<->3D toggle, MIP/additive/mean buttons + density/zScale/labelOpacity sliders + showLabels (call VolumeGPU methods), shares the decoded bundle (vv.d). Graceful fallback: VolumeGPU.create returns null without a usable adapter -> "WebGPU unavailable", reverts to 2.5D. Playwright test `tests/test_volume_page.py::test_3d_toggle_degrades_or_renders` PASS (headless Chromium has navigator.gpu but no adapter -> verified clean fallback, 2.5D still renders, no JS errors).
- NOTE: live WebGPU device render is the one piece not headless-verifiable here (bundled Chromium has no adapter). Its shader (P2a) + camera (P2b) + uniform/texture byte-layout (matching the verified wgpu-native harness) are all tested; needs a real-WebGPU browser (Chrome/Safari) or Deno for final visual confirmation.

## P3 — 3D overlays (raw-WebGPU line primitives)
- [x] P3a: pure builders `viewer/web/js/volume3d-overlays.js` — trajPolylines3D, lineageSegs3D, pointCrosses3D (points as 3D crosses), flowQuiver3D (subsampled, dir-coloured), affinitySegs3D (deduped + decimated with logged cap). All emit line segments in voxel coords + per-vertex colour. Node-tested `tests/js/overlays3d.test.mjs` (5 pass: counts, coords, colour determinism, dedup, cap).
- [x] P3b: `viewer/web/js/overlay.wgsl` line-list shader (voxel->world via box uniforms -> viewProj, per-vertex colour). wgpu-native test `tests/test_overlay_wgsl.py` (1 pass: known segment -> coloured pixels at expected row/cols).
- [x] P3c: `viewer/web/js/volume3d-overlays-gpu.js` OverlayLayer (builds GPU buffers from builders, draws into VolumeGPU's render pass sharing the camera). Integrated into volume3d-gpu.js (render computes camera once, draws overlays on top) + decodeBundle now decodes flow.raw. volume.html: 3D overlay checkboxes (trajectories+lineage / points / flow / affinity) -> vgpu.setOverlay. Playwright page tests still pass (no JS errors, graceful degradation).

## Verification channels
- backend: `python -m pytest omnipose/tests/test_volume3d.py`
- WGSL: `python ocdkit/outputs/repro/wgpu_raymarch_headless/proof.py`
- JS logic: Node CPU harness (P1+)
- browser integration (eventual, needs you): real Chrome / Deno / pywebview
