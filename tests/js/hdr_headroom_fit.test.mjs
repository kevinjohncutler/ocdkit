/* Node tests for hdrLutForHeadroom (plot/web/hdr_colormap.js): the HDR colormap
 * LUT's brightest channel lands exactly at headroom x gain, so gain 1 touches the
 * display limit and gain > 1 exceeds it, with every entry's hue preserved.
 * Run: /opt/homebrew/bin/node tests/js/hdr_headroom_fit.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const H = require(path.join(here, "../../src/ocdkit/plot/web/hdr_colormap.js"));

let n = 0;
const test = (name, fn) => { fn(); n++; console.log("ok -", name); };
const close = (a, b, tol = 1e-4) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} != ${b}`);

test("the plain lift stops short of the headroom (why the fit exists)", () => {
  const raw = H.generateImageCmapLutHdr("viridis", { auto: true, peakNits: 16 * H.HDR_SDR_WHITE_NITS });
  assert.ok(H.lutPeak(raw) < 0.9 * 16);
});

for (const cmap of ["gray", "viridis", "magma", "inferno"]) {
  test(`${cmap}: peak = headroom x gain for gains 0.5, 1, 2`, () => {
    for (const headroom of [4, 16]) for (const gain of [0.5, 1, 2]) {
      close(H.lutPeak(H.hdrLutForHeadroom(cmap, headroom, gain)), headroom * gain);
    }
  });
}

test("gain only scales brightness: channel ratios (hue) are unchanged", () => {
  const a = H.hdrLutForHeadroom("magma", 16, 1), b = H.hdrLutForHeadroom("magma", 16, 2);
  for (let i = 0; i < a.length; i += 4) for (let c = 0; c < 3; c++) close(b[i + c], 2 * a[i + c], 1e-4);
});

console.log(`${n} passed`);
