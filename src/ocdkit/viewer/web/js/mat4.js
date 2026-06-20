/* mat4.js — minimal column-major 4x4 matrix + camera helpers for the WebGPU
 * volume viewer. Column-major to match WGSL mat4x4 / the raymarch.wgsl
 * invViewProj uniform. Perspective uses [0,1] depth (WebGPU NDC, not GL's
 * [-1,1]). Pure + dependency-free so it's Node-testable. UMD.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.Mat4 = api;
})(this, function () {
  "use strict";
  // matrices are length-16 arrays, column-major: m[col*4 + row]

  function identity() { const m = new Array(16).fill(0); m[0] = m[5] = m[10] = m[15] = 1; return m; }

  function multiply(a, b) { // returns a*b
    const o = new Array(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
    return o;
  }

  function transformVec4(m, v) { // column-major mat * vec4
    const o = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += m[k * 4 + r] * v[k];
      o[r] = s;
    }
    return o;
  }

  // WebGPU [0,1] depth perspective (glMatrix perspectiveZO), column-major.
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const o = new Array(16).fill(0);
    o[0] = f / aspect; o[5] = f;
    const nf = 1 / (near - far);
    o[10] = far * nf; o[11] = -1; o[14] = far * near * nf;
    return o;
  }

  function _sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function _cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function _norm(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
  function _dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

  // Right-handed lookAt, column-major (glMatrix-compatible).
  function lookAt(eye, center, up) {
    const z = _norm(_sub(eye, center));      // forward (points away from center)
    const x = _norm(_cross(up, z));          // right
    const y = _cross(z, x);                  // true up
    return [
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -_dot(x, eye), -_dot(y, eye), -_dot(z, eye), 1,
    ];
  }

  function invert(m) {
    const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
    const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
    const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
    const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
    const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    det = 1.0 / det;
    return [
      (a11 * b11 - a12 * b10 + a13 * b09) * det,
      (a02 * b10 - a01 * b11 - a03 * b09) * det,
      (a31 * b05 - a32 * b04 + a33 * b03) * det,
      (a22 * b04 - a21 * b05 - a23 * b03) * det,
      (a12 * b08 - a10 * b11 - a13 * b07) * det,
      (a00 * b11 - a02 * b08 + a03 * b07) * det,
      (a32 * b02 - a30 * b05 - a33 * b01) * det,
      (a20 * b05 - a22 * b02 + a23 * b01) * det,
      (a10 * b10 - a11 * b08 + a13 * b06) * det,
      (a01 * b08 - a00 * b10 - a03 * b06) * det,
      (a30 * b04 - a31 * b02 + a33 * b00) * det,
      (a21 * b02 - a20 * b04 - a23 * b00) * det,
      (a11 * b07 - a10 * b09 - a12 * b06) * det,
      (a00 * b09 - a01 * b07 + a02 * b06) * det,
      (a31 * b01 - a30 * b03 - a32 * b00) * det,
      (a20 * b03 - a21 * b01 + a22 * b00) * det,
    ];
  }

  /** Eye position for an orbit camera around `target` (yaw,pitch in radians). */
  function orbitEye(target, radius, yaw, pitch) {
    const cp = Math.cos(pitch);
    return [
      target[0] + radius * cp * Math.sin(yaw),
      target[1] + radius * Math.sin(pitch),
      target[2] + radius * cp * Math.cos(yaw),
    ];
  }

  /** Build {eye, viewProj, invViewProj} for an orbit camera. */
  function orbitCamera(opts) {
    const target = opts.target || [0, 0, 0];
    const up = opts.up || [0, 1, 0];
    const eye = orbitEye(target, opts.radius, opts.yaw, opts.pitch);
    const view = lookAt(eye, target, up);
    const proj = perspective(opts.fovy, opts.aspect, opts.near, opts.far);
    const viewProj = multiply(proj, view);
    return { eye, viewProj, invViewProj: invert(viewProj) };
  }

  return { identity, multiply, transformVec4, perspective, lookAt, invert, orbitEye, orbitCamera };
});
