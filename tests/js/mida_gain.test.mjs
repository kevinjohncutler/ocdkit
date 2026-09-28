/* Node test for VolumeGPU._updateMidaGain: MIDA's data-driven gain.
 * On a uniform volume MIDA's value is the value itself, so the gain must be
 * 0.95 / value, clamped to [1, 4] (never darkens, never runs away).
 * Run: node tests/js/mida_gain.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const { VolumeGPU } = require(path.join(here, "../../src/ocdkit/viewer/web/js/volume3d-gpu.js"));

const HALF = { 0.5: 0x3800, 0.25: 0x3400, 1.0: 0x3c00, 0.1: 0x2e66 };
const fake = (value, extra) => Object.assign({
  NX: 8, NY: 6, NZ: 10, gamma: 1, density: 0.5, _win: [0, 1], _transparent: false,
  _volF16: new Uint16Array(8 * 6 * 10).fill(HALF[value]),
}, extra || {});
const gainOf = (f) => { VolumeGPU.prototype._updateMidaGain.call(f); return f._midaGain; };
const close = (a, b, tol = 2e-3) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
let n = 0;
const test = (name, fn) => { fn(); n++; console.log("ok -", name); };

test("uniform 0.5 -> gain 1.9", () => close(gainOf(fake(0.5)), 0.95 / 0.5));
test("uniform 0.25 -> gain 3.8", () => close(gainOf(fake(0.25)), 0.95 / 0.25));
test("already bright never darkens", () => close(gainOf(fake(1.0)), 1.0));
test("very dim is capped at 4", () => close(gainOf(fake(0.1)), 4.0));
test("opacity 0 shows nothing: gain stays 1", () => close(gainOf(fake(0.5, { density: 0 })), 1.0));
test("the window counts: [0, 0.5] makes 0.25 read as 0.5", () => close(gainOf(fake(0.25, { _win: [0, 2] })), 0.95 / 0.5));
console.log(`${n} passed`);
