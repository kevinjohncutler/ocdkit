/* Headless Node tests for viewer/web/js/volume3d.js.
 * Run: /opt/homebrew/bin/node tests/js/volume3d.test.mjs
 * Cross-language section reads /tmp/xlang_v3.json if present (emitted by the
 * test runner via omnipose.gui._volume3d.encode_array).
 */
import assert from "node:assert/strict";
import zlib from "node:zlib";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const V = require(path.join(here, "../../src/ocdkit/viewer/web/js/volume3d.js"));

const inflate = (bytes) => new Uint8Array(zlib.gunzipSync(Buffer.from(bytes)));
let n = 0;
const test = (name, fn) => { fn(); n++; console.log("  ok -", name); };

// --- decodeArray roundtrip (JS-encoded) ------------------------------------
function encodeJs(typed, dtype, shape, gzip) {
  let bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  if (gzip) bytes = new Uint8Array(zlib.gzipSync(Buffer.from(bytes)));
  return { dtype, shape, gzip, b64: Buffer.from(bytes).toString("base64") };
}

test("decodeArray uint8 gzip + plain", () => {
  const a = Uint8Array.from([0, 1, 2, 3, 250, 255]);
  for (const gz of [true, false]) {
    const { data, shape } = V.decodeArray(encodeJs(a, "uint8", [2, 3], gz), { inflate });
    assert.deepEqual([...data], [...a]);
    assert.deepEqual(shape, [2, 3]);
  }
});

test("decodeArray uint32 + float32", () => {
  const u = Uint32Array.from([0, 70000, 4294967295]);
  assert.deepEqual([...V.decodeArray(encodeJs(u, "uint32", [3], true), { inflate }).data], [...u]);
  const f = Float32Array.from([0.5, -1.25, 3.0]);
  assert.deepEqual([...V.decodeArray(encodeJs(f, "float32", [3], false), {}).data], [...f]);
});

test("decodeArray throws on gzip without inflate", () => {
  const a = Uint8Array.from([1, 2]);
  assert.throws(() => V.decodeArray(encodeJs(a, "uint8", [2], true), {}), /inflate/);
});

test("halfToFloat known values", () => {
  assert.equal(V._halfToFloat(0x3c00), 1.0);
  assert.equal(V._halfToFloat(0x4000), 2.0);
  assert.equal(V._halfToFloat(0xc000), -2.0);
  assert.equal(V._halfToFloat(0x0000), 0.0);
  assert.equal(V._halfToFloat(0x3800), 0.5);
});

// --- slice views -----------------------------------------------------------
test("volumeSlice + rgbVolumeSlice", () => {
  const D = 3, H = 2, W = 2;
  const vol = Uint8Array.from({ length: D * H * W }, (_, i) => i);
  assert.deepEqual([...V.volumeSlice(vol, [D, H, W], 1)], [4, 5, 6, 7]);
  assert.throws(() => V.volumeSlice(vol, [D, H, W], 3), RangeError);
  const rgb = Uint8Array.from({ length: D * H * W * 3 }, (_, i) => i);
  assert.equal(V.rgbVolumeSlice(rgb, [D, H, W, 3], 2).length, H * W * 3);
  assert.equal(V.rgbVolumeSlice(rgb, [D, H, W, 3], 0)[0], 0);
});

// --- affinity --------------------------------------------------------------
const STEPS3 = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1]];
test("in/through plane step indices", () => {
  assert.deepEqual(V.inPlaneStepIndices(STEPS3), [0, 1, 3]);
  assert.deepEqual(V.throughPlaneStepIndices(STEPS3), [2]);
});

test("affinitySliceSegments emits in-plane edge once", () => {
  const S = STEPS3.length, D = 2, H = 2, W = 2;
  const sp = new Uint8Array(S * D * H * W);
  const idx = (s, z, y, x) => ((s * D + z) * H + y) * W + x;
  sp[idx(0, 0, 0, 0)] = 1;            // +x edge at (0,0) on slice 0
  const segs = V.affinitySliceSegments(sp, [S, D, H, W], STEPS3, 0);
  assert.deepEqual([...segs], [0.5, 0.5, 1.5, 0.5]);
  // -x step (dx<0) must be deduped away even if set
  sp[idx(3, 0, 1, 1)] = 1;
  assert.equal(V.affinitySliceSegments(sp, [S, D, H, W], STEPS3, 0).length, 4);
  // nothing on slice 1
  assert.equal(V.affinitySliceSegments(sp, [S, D, H, W], STEPS3, 1).length, 0);
});

// --- points + trajectories -------------------------------------------------
test("pointsNearSlice filters by z", () => {
  const pts = Float32Array.from([0, 5, 6, 2, 7, 8, 0.4, 1, 2]);
  const out = V.pointsNearSlice(pts, 3, 0, 0.5);
  assert.deepEqual([...out], [6.5, 5.5, 2.5, 1.5]);  // z=0 and z=0.4 kept (x+.5,y+.5)
});

test("projectTracks up to frame", () => {
  const tracks = [{ label: 1, frames: [0, 1, 2], centroids: [[0, 0], [1, 1], [2, 2]] }];
  const pr = V.projectTracks(tracks, 1);
  assert.deepEqual([...pr[0].points], [0.5, 0.5, 1.5, 1.5]);
  assert.deepEqual(pr[0].head, [1.5, 1.5]);
});

test("lineageSegments after daughter appears", () => {
  const tracks = [
    { label: 1, frames: [0, 1], centroids: [[0, 0], [1, 1]] },
    { label: 2, frames: [2, 3], centroids: [[2, 2], [3, 3]] },
  ];
  assert.equal(V.lineageSegments(tracks, [[1, 2]], 1).length, 0);     // daughter not yet present
  assert.deepEqual([...V.lineageSegments(tracks, [[1, 2]], 2)], [1.5, 1.5, 2.5, 2.5]);
});

// --- cross-language (Python encode -> JS decode) ---------------------------
const XL = "/tmp/xlang_v3.json";
if (fs.existsSync(XL)) {
  const x = JSON.parse(fs.readFileSync(XL, "utf8"));
  test("xlang uint8 mask exact", () => {
    const { data } = V.decodeArray(x.mask, { inflate });
    const exp = Array.from({ length: 3 * 4 * 5 }, (_, i) => i % 5);
    assert.deepEqual([...data], exp);
  });
  test("xlang float16 flow exact", () => {
    const { data } = V.decodeArray(x.flow, { inflate });
    assert.deepEqual([...data], [1.0, -2.0, 0.5, 0.25, 0.0, -1.0, 2.0, 0.125]);
  });
  test("xlang uint32 exact", () => {
    const { data } = V.decodeArray(x.big, { inflate });
    assert.deepEqual([...data], Array.from({ length: 10 }, (_, i) => i * 100000));
  });
} else {
  console.log("  (skip cross-language: " + XL + " not present)");
}

console.log(`\n${n} passed`);
