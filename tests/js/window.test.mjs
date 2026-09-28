/* Node test for VolumeGPU.setWindow: the 2D histogram bounds (positions on its
 * 0..255 axis, which spans the data's min..max) map onto the normalized 3D
 * texture at full precision.
 * Run: /opt/homebrew/bin/node tests/js/window.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const { VolumeGPU } = require(path.join(here, "../../src/ocdkit/viewer/web/js/volume3d-gpu.js"));

let n = 0;
const test = (name, fn) => { fn(); n++; console.log("ok -", name); };
const fake = (valueRange) => ({ valueRange, renders: 0, exposures: 0, _requestRender() { this.renders++; },
                                _scheduleExposure() { this.exposures++; },
                                _applyWindow: VolumeGPU.prototype._applyWindow });
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("full range is the identity window", () => {
  const g = fake([0, 255]); VolumeGPU.prototype.setWindow.call(g, 0, 255);
  close(g._win[0], 0); close(g._win[1], 1); assert.equal(g.renders, 1);
  assert.equal(g.exposures, 0);   // the window maps values like a LUT: it never triggers a rescale
});

test("the 0..255 axis spans the data's min..max whatever its units", () => {
  const g = fake([0.013, 812.5]);                   // a float volume: texture t = (v - min) / (max - min)
  VolumeGPU.prototype.setWindow.call(g, 51, 102);
  close(g._win[0], 0.2); close(g._win[1], 1 / 0.2);
});

test("windows are not rounded to 256 steps", () => {
  const g = fake(null); VolumeGPU.prototype.setWindow.call(g, 0, 127.6);   // between two 8-bit steps
  close(g._win[1], 255 / 127.6);
});

console.log(`${n} passed`);
