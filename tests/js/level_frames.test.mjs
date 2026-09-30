/* Node test for frame leveling (VolumeGPU._displayF16 with setLevelFrames on):
 * each z slice is divided by its median background, in the data's own units.
 * Run: node tests/js/level_frames.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const { VolumeGPU } = require(path.join(here, "../../src/ocdkit/viewer/web/js/volume3d-gpu.js"));

const toF16 = (x) => { const u = new Uint16Array(new Float16Array([x]).buffer); return u[0]; };
const fromF16 = (h) => new Float16Array(new Uint16Array([h]).buffer)[0];
let n = 0;
const test = (name, fn) => { fn(); n++; console.log("ok -", name); };

// 3 slices of 20 x 20, backgrounds 0.1 / 0.2 / 0.4 (data units, valueRange [0, 1]),
// each with a 4 x 4 "cell" twice as bright as its own background
function stack() {
  const NX = 20, NY = 20, NZ = 3, bg = [0.1, 0.2, 0.4], a = new Uint16Array(NX * NY * NZ);
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    const cell = x >= 8 && x < 12 && y >= 8 && y < 12;
    a[(z * NY + y) * NX + x] = toF16(Math.min(1, bg[z] * (cell ? 2 : 1)));
  }
  return { NX, NY, NZ, a };
}
const at = (f16, NX, NY, x, y, z) => fromF16(f16[(z * NY + y) * NX + x]);

test("each slice's background moves to the median of the slice medians; ratios are kept", () => {
  const { NX, NY, NZ, a } = stack();
  const g = { NX, NY, NZ, valueRange: [0, 1], _volF16Orig: a, _level: true, _invert: false };
  const out = VolumeGPU.prototype._displayF16.call(g);
  for (let z = 0; z < NZ; z++) {
    const bg = at(out, NX, NY, 0, 0, z), cell = at(out, NX, NY, 10, 10, z);
    assert.ok(Math.abs(bg - 0.2) < 0.01, `slice ${z} background ${bg}`);
    assert.ok(Math.abs(cell / bg - 2) < 0.06, `slice ${z} cell/background ${cell / bg}`);
  }
});

test("off: the volume is unchanged; invert applies after leveling", () => {
  const { NX, NY, NZ, a } = stack();
  const off = VolumeGPU.prototype._displayF16.call({ NX, NY, NZ, valueRange: [0, 1], _volF16Orig: a, _level: false, _invert: false });
  assert.equal(off, a);
  const inv = VolumeGPU.prototype._displayF16.call({ NX, NY, NZ, valueRange: [0, 1], _volF16Orig: a, _level: true, _invert: true });
  for (let z = 0; z < NZ; z++) assert.ok(Math.abs(at(inv, NX, NY, 0, 0, z) - 0.8) < 0.01);
});

test("in the data's own units when the texture is normalized over [lo, hi]", () => {
  // data 100 + 100 t: backgrounds 110 / 120 / 140 in data units; the median background is 120
  const { NX, NY, NZ, a } = stack();
  const out = VolumeGPU.prototype._displayF16.call({ NX, NY, NZ, valueRange: [100, 200], _volF16Orig: a, _level: true, _invert: false });
  const data = (t) => 100 + 100 * t;
  for (let z = 0; z < NZ; z++) assert.ok(Math.abs(data(at(out, NX, NY, 0, 0, z)) - 120) < 1.5, `slice ${z}`);
});
test("fade z ends: the outermost slices blend toward the background, the middle is untouched", () => {
  // 40 slices of 8 x 8: background 0.1 everywhere, a bright voxel column of 0.9 in every slice
  const NX = 8, NY = 8, NZ = 40, a = new Uint16Array(NX * NY * NZ);
  for (let z = 0; z < NZ; z++) for (let i = 0; i < NX * NY; i++) a[z * NX * NY + i] = toF16(i === 27 ? 0.9 : 0.1);
  const g = { NX, NY, NZ, valueRange: [0, 1], _volF16Orig: a, _level: false, _fadeZ: true, _invert: false };
  const out = VolumeGPU.prototype._displayF16.call(g);
  const col = (z) => fromF16(out[z * NX * NY + 27]);
  assert.ok(Math.abs(col(20) - 0.9) < 0.01, "the middle is untouched");
  assert.ok(Math.abs(col(0) - 0.1) < 0.05 && Math.abs(col(NZ - 1) - 0.1) < 0.05, `the ends reach the background (${col(0)}, ${col(NZ - 1)})`);
  assert.ok(col(0) < col(2) && col(2) < col(4) && col(4) < col(6), "a smooth ramp inward");
  assert.ok(Math.abs(fromF16(out[0]) - 0.1) < 0.01, "the background stays the background");
  const off = VolumeGPU.prototype._displayF16.call({ ...g, _fadeZ: false, _volF16Faded: {} });
  assert.equal(off, a);
});
console.log(`${n} passed`);
