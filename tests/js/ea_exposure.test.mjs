/* Node test for VolumeGPU._updateExposure: EA's exposure comes from the data at
 * its full range and the brightest ray along ANY axis, so it never saturates on
 * its own and does not change when the window or gamma change (no "dance").
 * Run: node tests/js/ea_exposure.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const { VolumeGPU } = require(path.join(here, "../../src/ocdkit/viewer/web/js/volume3d-gpu.js"));

const HALF = { 0.5: 0x3800, 1.0: 0x3c00 };
const lut = new Float32Array(256).fill(1);            // gray-ish: brightest channel 1 everywhere
const fake = (extra) => Object.assign({
  NX: 8, NY: 6, NZ: 10, gamma: 1, _win: [0, 1], _transparent: false, _lutMaxc: lut, _lutPeakAll: 1,
  _volF16: new Uint16Array(8 * 6 * 10).fill(HALF[0.5]),
}, extra || {});
const expOf = (f) => { VolumeGPU.prototype._updateExposure.call(f); return f._eaExposure; };
const K = -Math.log(0.05);
const close = (a, b, tol = 1e-4) => assert.ok(Math.abs(a - b) < tol * Math.max(1, Math.abs(b)), `${a} != ${b}`);
let n = 0;
const test = (name, fn) => { fn(); n++; console.log("ok -", name); };

test("uniform: the longest axis (z, 10 voxels of 0.5) sets it", () => close(expOf(fake()), K / (10 * 0.5)));
test("the window does not change it", () => close(expOf(fake({ _win: [0.2, 3] })), K / (10 * 0.5)));
test("gamma does not change it", () => close(expOf(fake({ gamma: 0.4 })), K / (10 * 0.5)));
test("a bright line along x counts", () => {
  const f = fake({ NX: 40, NY: 6, NZ: 4, _volF16: new Uint16Array(40 * 6 * 4).fill(0) });
  for (let x = 0; x < 40; x += 1) f._volF16[x] = HALF[1.0];      // y = 0, z = 0
  close(expOf(f), K / 40);                                         // 40 voxels of 1 along x
});
console.log(`${n} passed`);
