/* volume3d.js — client-side decode + slice/overlay logic for the 3D viewer.
 *
 * Pure, environment-agnostic helpers (no DOM, no GL) so they can be unit-tested
 * headlessly in Node. Consumes the bundle produced by omnipose.gui._volume3d:
 *   - array fields are {dtype, shape, gzip, b64}
 *   - steps: [[dz,dy,dx], ...]            (26 non-centre neighbour offsets)
 *   - trajectories: {tracks:[{label,frames,centroids:[[y,x]]}], edges:[[p,d]]}
 *
 * Coordinate conventions (image space): x = column, y = row, z = depth/time.
 *   - flow raw mu : (3, D, H, W) = [d_depth, dy, dx]   (float16)
 *   - flow/dist rgb slices : (D, H, W, 3) uint8
 *   - affinity spatial : (S, D, H, W) uint8
 *   - points : (N, 3) float32 [z, y, x]
 *
 * gzip: pass a synchronous inflate fn via opts.inflate (Node: zlib.gunzipSync;
 * browser: serve binary endpoints with HTTP Content-Encoding so JS never gunzips,
 * or wrap DecompressionStream). decodeArray throws if a gzipped field arrives
 * with no inflate provided, rather than silently corrupting.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.Volume3D = api;
})(this, function () {
  "use strict";

  const DTYPE_CTORS = {
    uint8: Uint8Array, int8: Int8Array,
    uint16: Uint16Array, int16: Int16Array,
    uint32: Uint32Array, int32: Int32Array,
    float32: Float32Array, float64: Float64Array,
  };

  const HAS_F16 = typeof globalThis.Float16Array !== "undefined";

  function b64ToBytes(b64) {
    if (typeof Buffer !== "undefined") return new Uint8Array(Buffer.from(b64, "base64"));
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // half-precision -> float32 (fallback when Float16Array is unavailable)
  function halfToFloat(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function product(shape) { return shape.reduce((a, b) => a * b, 1); }

  /** Decode a {dtype,shape,gzip,b64} field into {data:TypedArray, shape}. */
  function decodeArray(field, opts) {
    opts = opts || {};
    let bytes = b64ToBytes(field.b64);
    if (field.gzip) {
      if (!opts.inflate) throw new Error("decodeArray: gzipped field needs opts.inflate");
      bytes = opts.inflate(bytes);
    }
    const shape = field.shape.slice();
    const n = product(shape);
    if (field.dtype === "float16") {
      const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      if (HAS_F16) {
        const f16 = new globalThis.Float16Array(buf, 0, n);
        return { data: Float32Array.from(f16), shape };
      }
      const u16 = new Uint16Array(buf, 0, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = halfToFloat(u16[i]);
      return { data: out, shape };
    }
    const Ctor = DTYPE_CTORS[field.dtype];
    if (!Ctor) throw new Error("decodeArray: unsupported dtype " + field.dtype);
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return { data: new Ctor(buf, 0, n), shape };
  }

  // --- slice views --------------------------------------------------------

  /** Scalar volume (D,H,W) -> subarray view of slice z (length H*W). */
  function volumeSlice(data, shape, z) {
    const [D, H, W] = shape;
    if (z < 0 || z >= D) throw new RangeError("z out of range");
    return data.subarray(z * H * W, (z + 1) * H * W);
  }

  /** RGB volume (D,H,W,3) -> subarray view of slice z (length H*W*3). */
  function rgbVolumeSlice(data, shape, z) {
    const [D, H, W, C] = shape;
    if (z < 0 || z >= D) throw new RangeError("z out of range");
    const stride = H * W * C;
    return data.subarray(z * stride, (z + 1) * stride);
  }

  // --- affinity (in-plane edges for the 2.5D view) ------------------------

  /** Indices of steps lying in the current plane (dz === 0). */
  function inPlaneStepIndices(steps) {
    const out = [];
    for (let i = 0; i < steps.length; i++) if (steps[i][0] === 0) out.push(i);
    return out;
  }

  /** Indices of steps that cross planes (dz !== 0) — drawn as markers. */
  function throughPlaneStepIndices(steps) {
    const out = [];
    for (let i = 0; i < steps.length; i++) if (steps[i][0] !== 0) out.push(i);
    return out;
  }

  /**
   * Line segments for in-plane affinity edges at slice z.
   * spatial: (S,D,H,W) uint8; steps: [[dz,dy,dx]]. Returns Float32Array of
   * [x0,y0,x1,y1, ...] in pixel-centre coords, for GL_LINES. Each undirected
   * edge is emitted once (only positive dy, or dy==0 & positive dx).
   */
  function affinitySliceSegments(spatial, shape, steps, z) {
    const [S, D, H, W] = shape;
    const planeStride = H * W;
    const stepStride = D * planeStride;
    const segs = [];
    const inPlane = inPlaneStepIndices(steps);
    for (const s of inPlane) {
      const dy = steps[s][1], dx = steps[s][2];
      if (dy < 0 || (dy === 0 && dx <= 0)) continue;   // dedup undirected edges
      const base = s * stepStride + z * planeStride;
      for (let y = 0; y < H; y++) {
        const ny = y + dy;
        if (ny < 0 || ny >= H) continue;
        for (let x = 0; x < W; x++) {
          if (spatial[base + y * W + x] === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= W) continue;
          segs.push(x + 0.5, y + 0.5, nx + 0.5, ny + 0.5);
        }
      }
    }
    return new Float32Array(segs);
  }

  // --- points -------------------------------------------------------------

  /**
   * Points (N,3 float32 [z,y,x]) near slice z -> Float32Array [x,y,...] in
   * pixel-centre coords, for the existing 2D point renderer.
   */
  function pointsNearSlice(points, count, z, tol) {
    tol = tol == null ? 0.5 : tol;
    const out = [];
    for (let i = 0; i < count; i++) {
      const pz = points[i * 3], py = points[i * 3 + 1], px = points[i * 3 + 2];
      if (Math.abs(pz - z) <= tol) out.push(px + 0.5, py + 0.5);
    }
    return new Float32Array(out);
  }

  // --- trajectories -------------------------------------------------------

  /**
   * Project centroid tracks to 2D polylines up to (and including) frame `upto`.
   * Returns [{label, points:Float32Array[x,y,...], head:[x,y]|null}].
   * centroids are [y,x]; head = centroid at the current frame if present.
   */
  function projectTracks(tracks, upto) {
    const out = [];
    for (const t of tracks) {
      const pts = [];
      let head = null;
      for (let i = 0; i < t.frames.length; i++) {
        if (t.frames[i] > upto) break;
        const [y, x] = t.centroids[i];
        pts.push(x + 0.5, y + 0.5);
        if (t.frames[i] === upto) head = [x + 0.5, y + 0.5];
      }
      if (pts.length) out.push({ label: t.label, points: new Float32Array(pts), head });
    }
    return out;
  }

  /**
   * Lineage edges as parent-tail -> daughter-head segments at/around frame
   * `upto`. Returns Float32Array [x0,y0,x1,y1,...] for GL_LINES. An edge is
   * drawn once the daughter has appeared (its first frame <= upto).
   */
  function lineageSegments(tracks, edges, upto) {
    const byLabel = new Map();
    for (const t of tracks) byLabel.set(t.label, t);
    const segs = [];
    for (const [p, d] of edges) {
      const pt = byLabel.get(p), dt = byLabel.get(d);
      if (!pt || !dt) continue;
      if (dt.frames[0] > upto) continue;
      const [py, px] = pt.centroids[pt.centroids.length - 1];   // parent tail
      const [dy, dx] = dt.centroids[0];                          // daughter head
      segs.push(px + 0.5, py + 0.5, dx + 0.5, dy + 0.5);
    }
    return new Float32Array(segs);
  }

  return {
    decodeArray, volumeSlice, rgbVolumeSlice,
    inPlaneStepIndices, throughPlaneStepIndices, affinitySliceSegments,
    pointsNearSlice, projectTracks, lineageSegments,
    _halfToFloat: halfToFloat,
  };
});
