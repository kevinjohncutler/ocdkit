/* Node tests for the empty-space-skipping brick builders in volume3d-gpu.js
 * (brickMax / brickAny feed raymarch_compute.wgsl bindings 5 and 6).
 * Run: /opt/homebrew/bin/node tests/js/bricks.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const G = require(path.join(here, "../../src/ocdkit/viewer/web/js/volume3d-gpu.js"));

let n = 0;
const test = (name, fn) => { fn(); n++; console.log("ok -", name); };

// naive reference: visit every brick, scan its voxels
function naive(vals, NX, NY, NZ, B, reduce, init) {
  const [bx, by, bz] = G.brickDims(NX, NY, NZ, B), out = [];
  for (let k = 0; k < bz; k++) for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
    let acc = init;
    for (let z = k * B; z < Math.min(NZ, k * B + B); z++)
      for (let y = j * B; y < Math.min(NY, j * B + B); y++)
        for (let x = i * B; x < Math.min(NX, i * B + B); x++) acc = reduce(acc, vals[(z * NY + y) * NX + x]);
    out.push(acc);
  }
  return out;
}

let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
const [NX, NY, NZ, B] = [37, 21, 18, 16];            // not multiples of B: partial edge bricks

test("brickDims rounds up", () => {
  assert.deepEqual(G.brickDims(NX, NY, NZ, B), [3, 2, 2]);
  assert.equal(G.BRICK, 16);
});

test("brickMax (float32) equals per-brick max", () => {
  const v = new Float32Array(NX * NY * NZ).map(() => rnd());
  assert.deepEqual(Array.from(G.brickMax(v, NX, NY, NZ, B, false)),
                   naive(v, NX, NY, NZ, B, Math.max, 0).map((x) => Math.fround(x)));
});

test("brickMax (half-float bits) decodes before taking the max", () => {
  const f = new Float32Array(NX * NY * NZ).map(() => rnd());
  const h = new Uint16Array(new Float16Array(f).buffer);
  const exact = Float32Array.from(new Float16Array(h.buffer));
  assert.deepEqual(Array.from(G.brickMax(h, NX, NY, NZ, B, true)), naive(exact, NX, NY, NZ, B, Math.max, 0));
});

test("brickAny marks exactly the bricks holding a label", () => {
  const lab = new Uint16Array(NX * NY * NZ);
  lab[(17 * NY + 20) * NX + 36] = 300;                  // lone voxel in the far corner brick
  lab[(0 * NY + 0) * NX + 0] = 1;
  const got = Array.from(G.brickAny(lab, NX, NY, NZ, B));
  assert.deepEqual(got, naive(lab, NX, NY, NZ, B, (a, v) => (a || v ? 1 : 0), 0));
  assert.equal(got.reduce((a, b) => a + b), 2);
});

console.log(`${n} passed`);
