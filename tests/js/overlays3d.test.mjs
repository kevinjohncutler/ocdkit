/* Node tests for volume3d-overlays.js (pure 3D overlay geometry builders).
 * Run: /opt/homebrew/bin/node tests/js/overlays3d.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const O = require(path.join(here, "../../src/ocdkit/viewer/web/js/volume3d-overlays.js"));

let n = 0;
const test = (name, fn) => { fn(); n++; console.log("  ok -", name); };
const arr = (ta, i, k) => Array.from(ta.slice(i, i + k));

test("trajPolylines3D: segments + voxel coords + per-track colour", () => {
  const tracks = [{ label: 7, frames: [0, 1, 2], centroids: [[0, 0], [1, 1], [2, 2]] }];
  const r = O.trajPolylines3D(tracks);
  assert.equal(r.count, 4);                       // 2 segments -> 4 vertices
  assert.deepEqual(arr(r.positions, 0, 6), [0, 0, 0, 1, 1, 1]);  // (x,y,z) pairs: t=0->t=1
  const lc = O._labelColor(7);
  assert.deepEqual(arr(r.colors, 0, 3).map((x) => +x.toFixed(5)), lc.map((x) => +x.toFixed(5)));
});

test("lineageSegs3D: parent tail -> daughter head", () => {
  const tracks = [
    { label: 1, frames: [0, 1], centroids: [[0, 0], [1, 1]] },
    { label: 2, frames: [2, 3], centroids: [[2, 2], [3, 3]] },
  ];
  const r = O.lineageSegs3D(tracks, [[1, 2]]);
  assert.equal(r.count, 2);
  assert.deepEqual(arr(r.positions, 0, 6), [1, 1, 1, 2, 2, 2]); // (x,y,z): tail(frame1) -> head(frame2)
});

test("pointCrosses3D: 3 segments per point centred correctly", () => {
  const pts = new Float32Array([2, 3, 4]);          // z=2,y=3,x=4
  const r = O.pointCrosses3D(pts, 1, 1.0);
  assert.equal(r.count, 6);                          // 3 segments
  // first segment is the x-cross: (x-1,y,z)->(x+1,y,z)
  assert.deepEqual(arr(r.positions, 0, 6), [3, 3, 2, 5, 3, 2]);
});

test("flowQuiver3D: one segment per non-zero subsampled voxel", () => {
  const D = 2, H = 2, W = 2;
  const data = new Float32Array(3 * D * H * W);     // [3,D,H,W] = [dz,dy,dx]
  // set flow at voxel (z0,y0,x0): dx=1 -> index in dx plane = 0
  data[2 * D * H * W + 0] = 1.0;                     // dx channel, voxel 0
  const r = O.flowQuiver3D({ data, shape: [3, D, H, W] }, 1, 2);
  assert.equal(r.count, 2);                          // exactly one segment
  assert.deepEqual(arr(r.positions, 0, 3), [0.5, 0.5, 0.5]);
  assert.deepEqual(arr(r.positions, 3, 3), [2.5, 0.5, 0.5]); // +dx*scale
});

test("affinitySegs3D: dedup + decimation cap is logged", () => {
  const S = 4, D = 2, H = 2, W = 2;
  // steps incl +x (kept) and -x (dropped by dedup)
  const steps = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [1, 0, 0]];
  const sp = new Uint8Array(S * D * H * W);
  const idx = (s, z, y, x) => ((s * D + z) * H + y) * W + x;
  sp[idx(0, 0, 0, 0)] = 1;                           // +x edge at voxel 0
  sp[idx(1, 0, 0, 1)] = 1;                           // -x edge (must be deduped away)
  const r = O.affinitySegs3D(sp, [S, D, H, W], steps, 1000);
  assert.equal(r.count, 2);                          // one segment (the +x), -x dropped
  assert.deepEqual(arr(r.positions, 0, 6), [0.5, 0.5, 0.5, 1.5, 0.5, 0.5]);
  // cap: set many edges, maxSegs small -> capped flag + fewer drawn
  const sp2 = new Uint8Array(S * D * H * W).fill(0);
  for (let z = 0; z < D; z++) for (let y = 0; y < H; y++) for (let x = 0; x < W - 1; x++) sp2[idx(0, z, y, x)] = 1;
  const r2 = O.affinitySegs3D(sp2, [S, D, H, W], steps, 1);
  assert.equal(r2.capped, true);
  assert.equal(r2.drawn, 1);
  assert.ok(r2.total > 1);
});

test("axesTriad3D: 3 colour-coded axes from origin + arrowheads", () => {
  const r = O.axesTriad3D([10, 20, 30], 0.5);
  // 3 axes x (1 shaft + 4 head) = 15 segments = 30 vertices
  assert.equal(r.count, 30);
  // first shaft starts at origin, goes +X (len = max(dims)*0.5 = 15)
  assert.deepEqual(arr(r.positions, 0, 6), [0, 0, 0, 15, 0, 0]);
  // X shaft colour is red-dominant
  const c = arr(r.colors, 0, 3);
  assert.ok(c[0] > c[1] && c[0] > c[2], "X axis should be red");
});

console.log(`\n${n} passed`);
