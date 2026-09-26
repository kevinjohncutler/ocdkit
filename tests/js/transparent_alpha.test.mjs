/* Node tests for transparentAlpha (plot/web/hdr_colormap.js): the transparent
 * low end's alpha ramp follows the colormap's lightness, dark end at 0.
 * Run: /opt/homebrew/bin/node tests/js/transparent_alpha.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const H = require(path.join(here, "../../src/ocdkit/plot/web/hdr_colormap.js"));

let n = 0;
const test = (name, fn) => { fn(); n++; console.log("ok -", name); };

for (const cmap of ["gray", "viridis", "magma", "inferno"]) {
  test(`${cmap}: dark end transparent, bright end opaque, values in [0, 1]`, () => {
    const a = H.transparentAlpha(cmap);
    assert.equal(a.length, H.IMAGE_CMAP_LUT_SIZE);
    assert.ok(a[0] < 1e-6, `first alpha ${a[0]}`);
    assert.ok(Math.max(...a) > 0.999);
    for (const v of a) assert.ok(v >= 0 && v <= 1);
  });
}

test("gray: alpha rises monotonically with lightness", () => {
  const a = H.transparentAlpha("gray");
  for (let i = 1; i < a.length; i++) assert.ok(a[i] >= a[i - 1] - 1e-7);
});

test("gamma shapes the ramp (0.5 default opens the low end faster than 1)", () => {
  const h = H.transparentAlpha("gray"), l = H.transparentAlpha("gray", 1.0);
  assert.ok(h[64] > l[64]);
});

console.log(`${n} passed`);
