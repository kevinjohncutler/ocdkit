/* Node test for VolumeGPU._updateMidaGain: MIDA's data-driven gain puts the
 * brightest ray (any axis, either direction) exactly at the top of the colormap.
 * On a uniform volume MIDA's value is the value itself, so the gain must be
 * 1 / value, clamped to [1, 4] (never darkens, never runs away).
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

test("uniform 0.5 -> gain 2 (brightest ray exactly at the top)", () => close(gainOf(fake(0.5)), 2.0));
test("already bright never darkens", () => close(gainOf(fake(1.0)), 1.0));
test("very dim is capped at 4", () => close(gainOf(fake(0.1)), 4.0));
test("opacity 0 shows nothing: gain stays 1", () => close(gainOf(fake(0.5, { density: 0 })), 1.0));
test("the window counts: [0, 0.5] makes 0.25 read as 0.5", () => close(gainOf(fake(0.25, { _win: [0, 2] })), 2.0));
test("every direction counts: a bright line along x sets the gain", () => {
  const f = fake(0.25);
  for (let x = 0; x < f.NX; x += 1) f._volF16[x] = HALF[1.0];   // y = 0, z = 0: only an x ray runs along it
  close(gainOf(f), 1.0);                                          // a z-only estimate would give > 1 and clip it
});
console.log(`${n} passed`);
