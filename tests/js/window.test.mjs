/* Node test for VolumeGPU.setWindow: the 2D histogram bounds (data units, the
 * 0..255 of the viewer's 8-bit volume) map onto the normalized 3D texture.
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
const fake = (valueRange) => ({ valueRange, renders: 0, _requestRender() { this.renders++; },
                                _applyWindow: VolumeGPU.prototype._applyWindow });
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("full range is the identity window", () => {
  const g = fake([0, 255]); VolumeGPU.prototype.setWindow.call(g, 0, 255);
  close(g._win[0], 0); close(g._win[1], 1); assert.equal(g.renders, 1);
});

test("bounds map through the volume's value range", () => {
  const g = fake([10, 210]);                        // texture t = (v - 10) / 200
  VolumeGPU.prototype.setWindow.call(g, 60, 110);
  close(g._win[0], 0.25); close(g._win[1], 1 / 0.25);
});

test("no value range falls back to 0..255", () => {
  const g = fake(null); VolumeGPU.prototype.setWindow.call(g, 51, 102);
  close(g._win[0], 0.2); close(g._win[1], 1 / 0.2);
});

console.log(`${n} passed`);
