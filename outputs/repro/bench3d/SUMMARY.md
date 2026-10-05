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

## Port into the viewer (render_final.jsonl)

The shipped `raymarch_compute.wgsl` now implements combo16m + the EA fix as clean
code (the benchmark's "Shipped" baseline is pinned to the pre-port commit via
`server.py /baseline/`). Measured against that baseline, same 120 scenarios:

| Result | Value |
|---|---|
| GPU time vs pre-port (geo mean) | 0.557x (EA 0.567, MIP 0.553, mean 0.551) |
| Faster / same / slower | 100 / 13 / 7 |
| Slower cases | 6 image-only EA, +3-4% (the 1-exp(-tau) accuracy fix); 1 image-only MIP +1.3% |
| Pixels changed outside EA | median 0, worst 39 of 2.6M |

Lesson from the port: the first clean rewrite nested the MIP brick check under
its own `if (MODE == 1)`, which ran up to 35% slower for image-only MIP than the
benchmarked flat form (same output). Hoisting the ray setup out of the two
marches did not matter; the nesting did (`render_port_ab`, `render_bisect*`).
The shader keeps the flat form with a comment saying why.

Real viewer, 5I (141x348x325, 316 labels), `outputs/repro/volume_port_e2e`:
3D mounts in 0.16 s; volume fetch 29 ms (binary float16) vs 203 ms (old JSON
route + decode) in the same session; EA/MIP/mean, hidden labels and a 3D erase
(label bricks rebuilt in place) all render with no errors.

## Correction: transport speedup in the real app

The transport suite encoded the stacks' float64 source, but the viewer session
stores every volume as uint8 (`_apply_image`), so the old JSON route never
shipped float64 in practice. The 20-28x figure is for float64 sources; in the
real app the measured gain is ~7x (5I: 203 ms -> 29 ms, `volume_port_e2e`).

## Per-mode cost (modes.html, 2026-10-05)

`bench_modes.js` times the CURRENT shader per projection mode, image only, in real
Chrome on the M5 Max, 2048x1280, GPU time median of 3 rounds x 20 frames
(`results/modes_pass1.jsonl`; rows there use the mode's old name, "Scatter").
Window "full" = the data's whole range, "clipped" = low end at the volume median.
AMIP at block 0.25, depth 25. Zoomed view (the volume fills the frame), ms:

| Dataset | Window | EA density 0 | EA density 1 | MIP | Mean | MIDA | AMIP |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 | full | 7.77 | 1.07 | 5.69 | 5.79 | 8.13 | 9.46 |
| dnaA_xy1 | clipped | 7.76 | 1.07 | 5.55 | 5.79 | 8.13 | 7.38 |
| 5I | full | 8.27 | 0.96 | 7.01 | 6.18 | 9.42 | 9.02 |
| 5I | clipped | 8.82 | 1.04 | 7.33 | 6.42 | 9.90 | 4.79 |
| ftsN_xy1 | full | 9.90 | 1.00 | 5.19 | 7.22 | 10.37 | 9.94 |
| ftsN_xy1 | clipped | 9.77 | 0.98 | 5.13 | 7.12 | 10.27 | 5.09 |

Cost = voxels each ray visits x work per voxel. Every mode runs the same DDA.
- Early exit dominates: EA with density stops at 99.5% opacity (6 to 10x faster
  than density 0, the viewer default); AMIP stops once nothing behind can beat the
  brightest so far (depth 2: 2 to 3 ms; depth 1e6, i.e. MIP: slower than MIP).
- Brick skipping: MIP skips bricks that cannot raise the running max; AMIP only
  skips bricks below the window's low end, so a clipped window halves its cost.
  EA, mean and MIDA never skip image bricks.
- Per voxel: mean (add) < MIP (max) < EA (exp) < AMIP (pow + exp) ~ MIDA.
- Rejected AMIP shader variants (exact, `vars=fastT,lim`): carrying transmittance
  with a branch on empty voxels, +12%; comparing optical depth with -log(best)
  instead of an exp per voxel, +6%. Branch-free code wins on Apple GPUs.
