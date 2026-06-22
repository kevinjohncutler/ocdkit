/* volume3d-overlays.js — pure builders that turn bundle overlays into LINE
 * geometry (voxel coords + per-vertex colour) for the 3D view. Everything is
 * line segments — points become small 3D crosses — so a single line pipeline
 * (overlay.wgsl) draws all of them. DOM-free + dependency-free -> Node-testable.
 *
 * Output: { positions: Float32Array (x,y,z per vertex), colors: Float32Array
 * (r,g,b per vertex), count: vertex count }. Vertices come in pairs (line-list).
 * Coords are VOXEL space (x=col, y=row, z=depth/time); the shader maps voxel->
 * world via the same box as the volume, so overlays register with the volume.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VolumeOverlays = api;
})(this, function () {
  "use strict";

  function hsv2rgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    const m = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
    return [m[0], m[1], m[2]];
  }
  function labelColor(v) { return v ? hsv2rgb((v * 0.61803398875) % 1, 0.65, 1.0) : [1, 1, 1]; }

  function _pack(pos, col) {
    return { positions: new Float32Array(pos), colors: new Float32Array(col), count: pos.length / 3 };
  }
  function _seg(pos, col, a, b, c) { // push segment a->b with colour c
    pos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    col.push(c[0], c[1], c[2], c[0], c[1], c[2]);
  }

  /** Centroid tracks -> polyline segments (one per consecutive frame pair). */
  function trajPolylines3D(tracks) {
    const pos = [], col = [];
    for (const t of tracks) {
      const c = labelColor(t.label);
      for (let i = 1; i < t.frames.length; i++) {
        const [y0, x0] = t.centroids[i - 1], [y1, x1] = t.centroids[i];
        _seg(pos, col, [x0, y0, t.frames[i - 1]], [x1, y1, t.frames[i]], c);
      }
    }
    return _pack(pos, col);
  }

  /** Division lineage -> parent-tail -> daughter-head segments. */
  function lineageSegs3D(tracks, edges) {
    const byLabel = new Map(tracks.map((t) => [t.label, t]));
    const pos = [], col = [];
    const white = [1, 1, 1];
    for (const [p, d] of edges || []) {
      const pt = byLabel.get(p), dt = byLabel.get(d);
      if (!pt || !dt) continue;
      const [py, px] = pt.centroids[pt.centroids.length - 1];
      const [dy, dx] = dt.centroids[0];
      _seg(pos, col, [px, py, pt.frames[pt.frames.length - 1]], [dx, dy, dt.frames[0]], white);
    }
    return _pack(pos, col);
  }

  /** Points (N,3 float32 [z,y,x]) -> 3 axis-aligned cross segments each. */
  function pointCrosses3D(points, count, size) {
    size = size == null ? 1.5 : size;
    const c = [1.0, 0.3, 0.8];
    const pos = [], col = [];
    for (let i = 0; i < count; i++) {
      const z = points[i * 3], y = points[i * 3 + 1], x = points[i * 3 + 2];
      _seg(pos, col, [x - size, y, z], [x + size, y, z], c);
      _seg(pos, col, [x, y - size, z], [x, y + size, z], c);
      _seg(pos, col, [x, y, z - size], [x, y, z + size], c);
    }
    return _pack(pos, col);
  }

  /** Subsampled flow field -> quiver segments, coloured by direction. */
  function flowQuiver3D(rawFlow, step, scale) {
    step = step || 6; scale = scale == null ? 4 : scale;
    const [C, D, H, W] = rawFlow.shape, data = rawFlow.data;
    const plane = H * W, vol = D * plane;
    const pos = [], col = [];
    for (let z = 0; z < D; z += step)
      for (let y = 0; y < H; y += step)
        for (let x = 0; x < W; x += step) {
          const idx = z * plane + y * W + x;
          const dz = data[idx], dy = data[vol + idx], dx = data[2 * vol + idx];
          const mag = Math.hypot(dx, dy, dz);
          if (mag < 1e-4) continue;
          const c = [Math.abs(dx) / (mag || 1), Math.abs(dy) / (mag || 1), Math.abs(dz) / (mag || 1)];
          _seg(pos, col, [x + 0.5, y + 0.5, z + 0.5],
               [x + 0.5 + dx * scale, y + 0.5 + dy * scale, z + 0.5 + dz * scale], c);
        }
    return _pack(pos, col);
  }

  function _stepKept(s) { // keep one direction of each undirected pair
    return s[0] > 0 || (s[0] === 0 && (s[1] > 0 || (s[1] === 0 && s[2] > 0)));
  }

  /** Spatial affinity (S,D,H,W) -> 3D edge segments, decimated to maxSegs. */
  function affinitySegs3D(spatial, shape, steps, maxSegs) {
    maxSegs = maxSegs || 200000;
    const [S, D, H, W] = shape, data = spatial;
    const plane = H * W, stepStride = D * plane;
    const c = [0.31, 0.86, 1.0];
    const pos = [], col = [];
    let total = 0, drawn = 0, capped = false;
    for (let s = 0; s < S; s++) {
      if (!_stepKept(steps[s])) continue;
      const dz = steps[s][0], dy = steps[s][1], dx = steps[s][2];
      const base = s * stepStride;
      for (let z = 0; z < D; z++) {
        const nz = z + dz; if (nz < 0 || nz >= D) continue;
        for (let y = 0; y < H; y++) {
          const ny = y + dy; if (ny < 0 || ny >= H) continue;
          for (let x = 0; x < W; x++) {
            if (data[base + z * plane + y * W + x] === 0) continue;
            const nx = x + dx; if (nx < 0 || nx >= W) continue;
            total++;
            if (drawn >= maxSegs) { capped = true; continue; }
            _seg(pos, col, [x + 0.5, y + 0.5, z + 0.5], [nx + 0.5, ny + 0.5, nz + 0.5], c);
            drawn++;
          }
        }
      }
    }
    if (capped && typeof console !== "undefined") {
      console.warn(`affinitySegs3D: drew ${drawn}/${total} edges (capped at ${maxSegs}); raise maxSegs or use slice view`);
    }
    const out = _pack(pos, col); out.total = total; out.drawn = drawn; out.capped = capped;
    return out;
  }

  function _vcross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function _vnorm(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

  /** XYZ arrow triad at the volume origin corner (voxel coords). X=red, Y=green,
   *  Z=blue. Each axis length = frac * max(dims), with a 4-segment arrowhead. */
  function axesTriad3D(dims, frac) {
    frac = frac == null ? 0.5 : frac;
    const L = Math.max(dims[0], dims[1], dims[2]) * frac;
    const O = [0, 0, 0];
    const AX = [
      { dir: [1, 0, 0], col: [1.0, 0.25, 0.25] },   // X red
      { dir: [0, 1, 0], col: [0.25, 1.0, 0.25] },   // Y green
      { dir: [0, 0, 1], col: [0.35, 0.55, 1.0] },   // Z blue
    ];
    const pos = [], col = [];
    for (const { dir, col: c } of AX) {
      const tip = [O[0] + dir[0] * L, O[1] + dir[1] * L, O[2] + dir[2] * L];
      _seg(pos, col, O, tip, c);                                  // shaft
      const ref = Math.abs(dir[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      const p1 = _vnorm(_vcross(dir, ref)), p2 = _vcross(dir, p1);
      const base = [tip[0] - dir[0] * L * 0.18, tip[1] - dir[1] * L * 0.18, tip[2] - dir[2] * L * 0.18];
      const w = L * 0.08;
      for (const [s, t] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        _seg(pos, col, tip,
          [base[0] + (p1[0] * s + p2[0] * t) * w, base[1] + (p1[1] * s + p2[1] * t) * w, base[2] + (p1[2] * s + p2[2] * t) * w], c);
      }
    }
    return _pack(pos, col);
  }

  return { trajPolylines3D, lineageSegs3D, pointCrosses3D, flowQuiver3D, affinitySegs3D, axesTriad3D, _labelColor: labelColor };
});
