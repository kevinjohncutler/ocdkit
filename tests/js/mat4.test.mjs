/* Node tests for mat4.js — the camera math feeding raymarch.wgsl's invViewProj.
 * Run: /opt/homebrew/bin/node tests/js/mat4.test.mjs
 */
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const M = require(path.join(here, "../../src/ocdkit/viewer/web/js/mat4.js"));

let n = 0;
const test = (name, fn) => { fn(); n++; console.log("  ok -", name); };
const close = (a, b, t = 1e-5) => Math.abs(a - b) <= t;
const vclose = (a, b, t = 1e-5) => a.every((x, i) => close(x, b[i], t));

test("identity * m == m", () => {
  const m = M.perspective(1.0, 1.5, 0.1, 100);
  assert.ok(vclose(M.multiply(M.identity(), m), m));
});

test("invert(m) * m == identity", () => {
  const vp = M.orbitCamera({ target: [1, 2, 3], radius: 7, yaw: 0.5, pitch: 0.3,
                             fovy: Math.PI / 3, aspect: 1.3, near: 0.1, far: 100 }).viewProj;
  const I = M.multiply(M.invert(vp), vp);
  assert.ok(vclose(I, M.identity(), 1e-4), "not identity: " + I.map((x) => x.toFixed(3)));
});

test("orbitEye geometry", () => {
  assert.ok(vclose(M.orbitEye([0, 0, 0], 10, 0, 0), [0, 0, 10]));         // yaw0 -> +Z
  assert.ok(vclose(M.orbitEye([0, 0, 0], 10, Math.PI / 2, 0), [10, 0, 0], 1e-4)); // yaw90 -> +X
  assert.ok(vclose(M.orbitEye([0, 0, 0], 10, 0, Math.PI / 2), [0, 10, 0], 1e-4)); // pitch90 -> +Y
});

test("invViewProj reconstructs the centre ray as the view forward", () => {
  const cam = M.orbitCamera({ target: [0, 0, 0], radius: 10, yaw: 0, pitch: 0,
                             fovy: Math.PI / 3, aspect: 1, near: 0.1, far: 100 });
  // eye at +Z looking at origin -> forward = (0,0,-1)
  assert.ok(vclose(cam.eye, [0, 0, 10]));
  const near = M.transformVec4(cam.invViewProj, [0, 0, 0, 1]); // centre, near plane
  const far = M.transformVec4(cam.invViewProj, [0, 0, 1, 1]);  // centre, far plane
  const np = [near[0] / near[3], near[1] / near[3], near[2] / near[3]];
  const fp = [far[0] / far[3], far[1] / far[3], far[2] / far[3]];
  const dir = [fp[0] - np[0], fp[1] - np[1], fp[2] - np[2]];
  const L = Math.hypot(...dir);
  const rd = dir.map((x) => x / L);
  assert.ok(vclose(rd, [0, 0, -1], 1e-4), "centre ray dir " + rd);
  // near point sits ~0.1 in front of the eye along -Z
  assert.ok(close(np[2], 9.9, 1e-3), "near z " + np[2]);
});

test("project then unproject round-trips a world point", () => {
  const cam = M.orbitCamera({ target: [0, 0, 0], radius: 8, yaw: 0.7, pitch: 0.4,
                             fovy: Math.PI / 3, aspect: 1.2, near: 0.1, far: 100 });
  const world = [0.3, -0.5, 0.2, 1];
  const clip = M.transformVec4(cam.viewProj, world);
  const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1];
  const back = M.transformVec4(cam.invViewProj, ndc);
  const w = [back[0] / back[3], back[1] / back[3], back[2] / back[3]];
  assert.ok(vclose(w, [0.3, -0.5, 0.2], 1e-4), "round-trip " + w);
});

console.log(`\n${n} passed`);
