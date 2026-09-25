# 3D renderer A/B benchmark: luxar-inspired ideas vs the shipped renderer

Run 2026-09-25 on an Apple M5 Max laptop (Metal), real Google Chrome WebGPU
(headless, Playwright `channel="chrome"`), GPU time from timestamp queries.
Output 2048x1280. Each variant: 4 interleaved rounds x 20 frames, randomized order.
Baseline is the SHIPPED `raymarch_compute.wgsl`, loaded verbatim; every variant is a
checked text patch of it (`variants.js`), so a failed patch errors instead of
silently measuring the baseline.

Datasets (`prep.py`): dnaA_xy1 133x302x302 (40 labels), 5I 141x348x325 (316 labels,
uint16), ftsN_xy1 241x603x234 (34M voxels), skimage cells3d nuclei 60x256x256
(dense tissue). Views: home (viewer default), zoom (volume fills frame), top.
Configs: image only, image + opaque labels (viewer default), labels only,
image + 50% labels. Modes: EA (emission-absorption), MIP, mean.

## Verdicts

| Idea | Verdict | Evidence |
|---|---|---|
| Binary float16 transport | Adopt | 785/1096/1851 ms -> 28/40/89 ms request-to-texture; bit-identical texture |
| gzip on top of it | Reject (localhost) | server gzip dominates: 356/489/1185 ms |
| Override constants + label-first + 16^3 label bricks + MIP-only image bricks (combo16m) | Adopt | 0.54x (EA) / 0.55x (MIP) GPU time geo mean, 0 of 84 scenarios slower |
| Override constants alone | Adopt (part of above) | 0.87x, 0 of 120 slower, pixel-identical |
| Label-first (skip image march under opaque labels) | Adopt (part of above) | 0.78x with opaque labels, 0 of 72 slower, pixel-identical |
| Brick skipping alone | Reject in isolation | slower in 24-33 of 120 (all EA / image-only); only good combined |
| Brick skip for EA image | Reject | real backgrounds are never 0, check never fires, +20-37% |
| "Exact" re-entry before brick exit (skipx) | Reject | more changed pixels (up to 732), not fewer |
| EA alpha 1-exp(-tau) | Adopt (accuracy) | alpha error vs analytic 0.093 -> <=0.005 at oblique views; +4.5% on EA frames |
| Clip image at label depth, composite in front | Reject | washes labels to pastel (noisy background over them); no speed gain over label-first |
| Per-label LUT (label ids + storage buffer) | Feature call | render 1.01x (within noise); recolor 0.1 ms vs 1.3-3.9 ms re-upload; 2x label memory above 255 labels |
| Per-frame GPU ID buffer | Feature call | 1.00x, free; only needed for hover highlight |
| GPU pick instead of server pick | Not for latency | HTTP pick 0.4-1.2 ms median vs GPU 0.3-0.4 ms |
| Server pick correctness | Fix | fixed-step march disagrees with drawn DDA on up to 1.3% of hit rays; port the DDA |

## Fidelity notes
- Brick skipping recomputes tMax from the ray origin after a jump, while the
  shipped DDA accumulates tMax += tDelta; rays grazing a voxel edge can enter the
  other neighbor. Median 0 changed pixels; worst 39 of 2.6M (zoomed MIP).
- Brick build (JS, per load): 14-44 ms at 16^3; label bricks need a local rebuild
  after edits.

## Files
- `prep.py` data prep, `server.py` bench server (runs shipped server code paths),
  `variants.js` shader patches, `bench.js` suites (render|pick|update|tx),
  `run.py` Chrome driver, `analyze.py` tables, `ea_accuracy.py` wgpu-native
  accuracy check. Raw results: `results/*.jsonl`, sample frames `results/png/`.
- Rerun: `run.py DATA "suite=render&k=20&rounds=4"` (render_combo / render_mip_only
  passes via `variants=` and `tag=`). Pass-1 `override` numbers in render.jsonl are
  INVALID for datasets after dnaA (stale bind group, harness bug, fixed); use
  render_combo.jsonl.
