/*
 * ColormapImage — one renderer interface, two backends, auto-dispatched.
 *
 * Like a numpy/dask/torch dispatch: callers write ONE code path and never branch
 * on HDR. `createColormapRenderer(canvas)` returns a WebGPU backend (HDR — the
 * lifted colormap on an extended-range display-p3 canvas) when WebGPU is
 * available, else a WebGL2 backend (SDR — the same colormap clamped). Both
 * implement the identical interface:
 *
 *   await ColormapImage.createColormapRenderer(canvas, { headroom })
 *     .setImage(scalarFloat32, w, h)   // single-channel image, row 0 = top
 *     .setColormap(name)               // 'viridis', 'magma', …
 *     .setRange(vmin, vmax)            // contrast window, in image-value units
 *     .setGamma(g)                     // display gamma (1 = linear)
 *     .setTransform(mat3col9)          // image-px → clip (host pan/zoom matrix)
 *     .requestRedraw(); .destroy();
 *   // r.backend === 'webgpu' | 'webgl2' ;  r.hdr === true | false
 *
 * The WebGPU backend is HdrColormap.HdrColormapRenderer; the WebGL2 backend lives
 * here. Both colour from the same lift (HdrColormap), so colours match — the
 * WebGL2 one just can't exceed SDR white.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.ColormapImage = mod;
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const VERT = `#version 300 es
  uniform mat3 u_matrix;
  uniform vec2 u_imgSize;
  out vec2 v_uv;
  const vec2 C[4] = vec2[4](vec2(0., 0.), vec2(1., 0.), vec2(0., 1.), vec2(1., 1.));
  void main() {
    vec2 c = C[gl_VertexID];
    vec3 clip = u_matrix * vec3(c * u_imgSize, 1.0);
    gl_Position = vec4(clip.xy, 0.0, 1.0);
    v_uv = c;
  }`;
  const FRAG = `#version 300 es
  precision highp float;
  uniform highp sampler2D u_img;     // R32F intensity (texelFetch, no filtering)
  uniform sampler2D u_lut;           // 256x1 RGBA8, display-p3 gamma-encoded
  uniform vec2 u_imgSize;
  uniform float u_vmin, u_vmax, u_gamma;
  in vec2 v_uv;
  out vec4 o;
  void main() {
    ivec2 px = clamp(ivec2(v_uv * u_imgSize), ivec2(0), ivec2(u_imgSize) - ivec2(1));
    float val = texelFetch(u_img, px, 0).r;
    float t = clamp((val - u_vmin) / max(u_vmax - u_vmin, 1e-9), 0.0, 1.0);
    t = pow(t, u_gamma);
    o = vec4(texture(u_lut, vec2(t, 0.5)).rgb, 1.0);
  }`;

  function _compile(gl, type, src) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  // WebGL2 SDR backend — mirrors HdrColormapRenderer's interface exactly.
  class WebGLColormapRenderer {
    constructor(canvas, opts, HC) {
      this.canvas = canvas; this.HC = HC; this.backend = 'webgl2'; this.hdr = false;
      const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true });
      if (!gl) throw new Error('WebGL2 unavailable');
      this.gl = gl;
      try { gl.drawingBufferColorSpace = 'display-p3'; } catch (e) { /* wide-gamut optional */ }
      const prog = gl.createProgram();
      gl.attachShader(prog, _compile(gl, gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, _compile(gl, gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
      this.prog = prog;
      this.u = {};
      ['u_matrix', 'u_imgSize', 'u_vmin', 'u_vmax', 'u_gamma', 'u_img', 'u_lut'].forEach((n) => { this.u[n] = gl.getUniformLocation(prog, n); });
      this.vao = gl.createVertexArray();
      this.imgTex = gl.createTexture();
      this.lutTex = gl.createTexture();
      this._w = 0; this._h = 0; this._cw = 0; this._ch = 0;
      this._vmin = 0; this._vmax = 1; this._gamma = 1.0; this._matrix = null;
      this._raf = 0; this._cmap = 'viridis';
      this._setLut('viridis');
      // Re-render when the canvas resizes (e.g. revealed from display:none, or
      // a panel reflow) — otherwise it stays at a stale/zero backing size.
      if (typeof ResizeObserver !== 'undefined') { this._ro = new ResizeObserver(function () { this.requestRedraw(); }.bind(this)); this._ro.observe(canvas); }
    }

    setImage(data, w, h) {
      const gl = this.gl;
      const f = (data instanceof Float32Array) ? data : Float32Array.from(data);
      this._w = w; this._h = h;
      gl.bindTexture(gl.TEXTURE_2D, this.imgTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, f);
      this.requestRedraw();
    }

    _setLut(name) {
      const gl = this.gl;
      const u8 = this.HC ? this.HC.sdrLutU8(name) : new Uint8Array(256 * 4).fill(255);
      gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, u8);
    }

    setColormap(name) { this._cmap = name; this._setLut(name); this.requestRedraw(); }
    setRange(vmin, vmax) { this._vmin = vmin; this._vmax = vmax; this.requestRedraw(); }
    setGamma(g) { this._gamma = g || 1.0; this.requestRedraw(); }
    setHdr() { /* WebGL2 is SDR-only; no-op for interface parity */ }
    setGain() { /* SDR can't exceed white; no-op for interface parity */ }
    setTransform(mat3col9) { this._matrix = mat3col9 || null; this.requestRedraw(); }
    _fillMatrix() { const W = this._w || 1, H = this._h || 1; return [2 / W, 0, 0, 0, -2 / H, 0, -1, 1, 1]; }

    requestRedraw() { if (!this._raf && typeof requestAnimationFrame !== 'undefined') this._raf = requestAnimationFrame(() => this._render()); }

    _render() {
      this._raf = 0;
      if (!this._w) return;
      const gl = this.gl;
      const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
      const cw = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
      const ch = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
      if (cw !== this._cw || ch !== this._ch) { this._cw = cw; this._ch = ch; this.canvas.width = cw; this.canvas.height = ch; }
      gl.viewport(0, 0, cw, ch);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      gl.uniformMatrix3fv(this.u.u_matrix, false, this._matrix || this._fillMatrix());
      gl.uniform2f(this.u.u_imgSize, this._w, this._h);
      gl.uniform1f(this.u.u_vmin, this._vmin);
      gl.uniform1f(this.u.u_vmax, this._vmax);
      gl.uniform1f(this.u.u_gamma, this._gamma);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.imgTex); gl.uniform1i(this.u.u_img, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.lutTex); gl.uniform1i(this.u.u_lut, 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
    }

    destroy() { const gl = this.gl; if (this.imgTex) gl.deleteTexture(this.imgTex); if (this.lutTex) gl.deleteTexture(this.lutTex); if (this.prog) gl.deleteProgram(this.prog); }
  }

  // Dispatcher: WebGPU (HDR) when available, else WebGL2 (SDR). Probes the GPU
  // device FIRST (doesn't touch the canvas) so the WebGL2 fallback can still
  // claim the canvas context if WebGPU isn't usable.
  async function createColormapRenderer(canvas, opts) {
    opts = opts || {};
    const HC = (typeof window !== 'undefined' && window.HdrColormap) || (typeof self !== 'undefined' && self.HdrColormap) || null;
    if (!opts.forceWebgl && HC && typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        await HC.getDevice();                       // probe; doesn't touch the canvas
        return await HC.HdrColormapRenderer.create(canvas, opts);   // WebGPU / HDR
      } catch (e) { /* fall through to WebGL2 */ }
    }
    return new WebGLColormapRenderer(canvas, opts, HC);             // WebGL2 / SDR
  }

  return { createColormapRenderer, WebGLColormapRenderer };
}));
