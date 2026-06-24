/*
 * LabelGLRenderer — shared WebGL2 label/segmentation renderer.
 *
 * Extracted from the ocdkit viewer (viewer/web/app.js) so the full editor
 * AND the lightweight image_grid/imshow tiles render labels through ONE
 * engine. Pure rendering, no editor state.
 *
 * Mirrors the viewer's conventions, with one extension: the mask texture
 * carries TWO ids per pixel so ncolor (where many cells share a color)
 * still hover-highlights a single cell, exactly like the viewer's separate
 * nColorInstanceMask.
 *   - mask matrix uploaded as an RGBA8 texture:
 *       RG = COLOR/group id   (palette lookup; id = R + G*256)
 *       BA = INSTANCE id      (hover/picking; id = B + A*256)
 *     With no ncolor relabel the two ids are identical (color == instance).
 *   - 1-D RGBA8 palette LUT (paletteColor), procedural sinebow fallback
 *   - R8 outline texture (boundary mask), computed from INSTANCE ids so
 *     adjacent same-color cells still get a divider; u_maskStyle selects
 *     fill / fill+outline / outline-only
 * Adds u_highlightLabel (compared against the INSTANCE id) for hover.
 *
 * Usage (single tile filling its canvas):
 *   const r = new LabelGLRenderer(gl);
 *   r.setLabels(colorIds, w, h, instanceIds);  // computes + uploads outline
 *   r.setPalette(rgbaUint8 | null);            // null → sinebow
 *   r.setBase(imageBitmap | null);             // optional underlay
 *   r.setUniforms({maskOpacity:0.6, maskStyle:0, highlightLabel:0});
 *   r.draw(orthoTile(w, h, viewport));         // mat3, image px → clip
 *   const id = r.labelAt(px, py);              // instance id, for hover
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.LabelGL = mod;
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const VERT = `#version 300 es
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec2 a_texCoord;
uniform mat3 u_matrix;
out vec2 v_texCoord;
void main() {
  vec3 pos = u_matrix * vec3(a_position, 1.0);
  gl_Position = vec4(pos.xy, 0.0, 1.0);
  v_texCoord = a_texCoord;
}`;

  // Mask path lifted verbatim from app.js PIPELINE_FRAGMENT_SHADER, trimmed
  // of the editor-only flow/distance/points overlays, plus u_highlightLabel.
  const FRAG = `#version 300 es
precision highp float;
in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_baseSampler;
uniform sampler2D u_maskSampler;
uniform sampler2D u_paletteSampler;

uniform float u_maskOpacity;
uniform float u_maskVisible;
uniform float u_outlinesVisible;
uniform float u_maskStyle;     // 0 fill+outline, 1 fill, 2 outline-only
uniform float u_imageVisible;
uniform float u_colorOffset;
uniform float u_paletteSize;
uniform float u_usePalette;
uniform float u_highlightLabel;
uniform vec3  u_highlightColor;    // outline-tile hover tint (translucent), default red
uniform float u_highlightAlpha;    // outline-tile hover tint opacity (translucent)
uniform float u_highlightBoost;    // fill-tile hover: cell-color brightness multiplier
uniform float u_outlineHdrBoost;   // >1 → boundary pixels emit HDR-bright
uniform vec3  u_outlineColor;      // uniform contour color (e.g. red)
uniform float u_useOutlineColor;   // >0.5 → override per-cell color on edges
uniform float u_baseHeadroom;      // >1 → lift the GPU base via EOTF×headroom×OETF (HDR glow)
uniform float u_baseLinear;        // >0.5 → base texture is RAW linear-light f16 (skip EOTF)

float _eotf(float c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
float _oetf(float c) { float x = max(c, 0.0); return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
vec3 _baseHdr(vec3 c) { return vec3(_oetf(_eotf(c.r) * u_baseHeadroom), _oetf(_eotf(c.g) * u_baseHeadroom), _oetf(_eotf(c.b) * u_baseHeadroom)); }

vec3 sinebow(float t) {
  float angle = 6.28318530718 * fract(t);
  float r = sin(angle) * 0.5 + 0.5;
  float g = sin(angle + 2.09439510239) * 0.5 + 0.5;
  float b = sin(angle + 4.18879020479) * 0.5 + 0.5;
  return vec3(r, g, b);
}
vec3 hashColor(float label) {
  float golden = 0.61803398875;
  return sinebow(fract(label * golden + u_colorOffset));
}
vec3 paletteColor(float label) {
  float size = max(u_paletteSize, 1.0);
  float idx = mod(label, size);
  return texture(u_paletteSampler, vec2((idx + 0.5) / size, 0.5)).rgb;
}
// Instance id at a uv (BA channels of the packed mask) — for screen-space outlines.
float _instAt(vec2 uv) {
  vec4 p = texture(u_maskSampler, uv);
  return floor(p.b * 255.0 + 0.5) + floor(p.a * 255.0 + 0.5) * 256.0;
}
void main() {
  vec2 baseCoord = vec2(v_texCoord.x, 1.0 - v_texCoord.y);
  // When a GPU base image is set we composite over it and emit opaque. When
  // not (the common image_grid path — the HDR base, if any, is a separate
  // SVG <image> BEHIND this canvas so it keeps its gain-map HDR), we emit a
  // TRANSPARENT overlay: straight color + coverage alpha, so the image
  // shows through where there's no label. premultipliedAlpha:false, so rgb
  // stays un-premultiplied.
  bool hasBase = (u_imageVisible > 0.5);
  vec3 color = vec3(0.0);
  float outA = hasBase ? 1.0 : 0.0;
  if (hasBase) {
    color = texture(u_baseSampler, baseCoord).rgb;
    if (u_baseLinear > 0.5) {
      // RAW linear-light f16 base (setBaseRaw): boost (default 1 = no-op) then
      // OETF for display. Same math as the inline static-HDR controller's raw path.
      color = vec3(_oetf(color.r * u_baseHeadroom), _oetf(color.g * u_baseHeadroom), _oetf(color.b * u_baseHeadroom));
    } else if (u_baseHeadroom > 1.0001) {
      // sRGB-OETF PNG base: lift to HDR so it glows (u_baseHeadroom 1 = no-op).
      color = _baseHdr(color);
    }
  }
  if (u_maskVisible > 0.5 && u_maskOpacity > 0.0) {
    vec4 packed = texture(u_maskSampler, v_texCoord);
    float label = floor(packed.r * 255.0 + 0.5) + floor(packed.g * 255.0 + 0.5) * 256.0;  // color/group id
    float inst  = floor(packed.b * 255.0 + 0.5) + floor(packed.a * 255.0 + 0.5) * 256.0;  // instance id (hover)
    if (label > 0.5) {
      float alpha = clamp(u_maskOpacity, 0.0, 1.0);
      float outline = 0.0;
      if (u_outlinesVisible > 0.5) {
        // Boundary detected in the shader by comparing this pixel's instance id
        // against its 4 neighbours, offset by  max(1 IMAGE pixel, 1 DISPLAY pixel)
        // in uv. So the contour width is:
        //   • zoomed IN  → 1 image-grid pixel (tracks the image's own pixels), and
        //   • zoomed OUT → clamped to ≥1 display pixel, so a minified matrix never
        //     gaps/speckles (unlike point-sampling a baked 1-matrix-px R8 mask).
        // Requires the canvas to render at DISPLAY resolution (dFdx = 1 display px);
        // the inline label controller sizes it so. Instance ids are exact integer
        // floats, so != is safe; also removes the O(w*h) CPU outline scan + R8 upload.
        vec2 scrPx = vec2(abs(dFdx(v_texCoord.x)), abs(dFdy(v_texCoord.y)));
        vec2 imgPx = 1.0 / vec2(textureSize(u_maskSampler, 0));
        vec2 d = max(scrPx, imgPx);
        // 8-connectivity (orthogonal + DIAGONAL neighbours). 4-neighbour detection
        // leaves a 1px gap at every corner where the boundary turns — the corner
        // pixel's 4 orthogonal neighbours are all same-instance and only a diagonal
        // differs — so the perimeter breaks at each turn. The 4 diagonal samples
        // fill those corners. (Straight edges are already caught orthogonally, so
        // this doesn't thicken them.)
        if (_instAt(v_texCoord + vec2(d.x, 0.0)) != inst || _instAt(v_texCoord - vec2(d.x, 0.0)) != inst
         || _instAt(v_texCoord + vec2(0.0, d.y)) != inst || _instAt(v_texCoord - vec2(0.0, d.y)) != inst
         || _instAt(v_texCoord + d) != inst || _instAt(v_texCoord - d) != inst
         || _instAt(v_texCoord + vec2(d.x, -d.y)) != inst || _instAt(v_texCoord + vec2(-d.x, d.y)) != inst) outline = 1.0;
      }
      if (u_maskStyle > 1.5) {
        alpha = alpha * outline;
      } else if (u_maskStyle < 0.5 && u_outlinesVisible > 0.5) {
        alpha = mix(alpha * 0.5, alpha, outline);
      }
      vec3 maskColor = (u_usePalette > 0.5) ? paletteColor(label) : hashColor(label);
      // Uniform contour color: an outline-only overlay reads best as a
      // single bright color (e.g. red) rather than a per-cell sinebow that
      // washes out over an image. Override edge pixels before any boost, and
      // make them OPAQUE — otherwise the base behind (a separate HDR <image>,
      // often brighter than SDR white) bleeds through the contour's alpha and
      // washes the red toward the base color (looked "dark"/desaturated over
      // an HDR maxp). Opaque = crisp, consistent red regardless of the base.
      if (outline > 0.5 && u_useOutlineColor > 0.5) {
        maskColor = u_outlineColor;
        alpha = 1.0;
      }
      // HDR boundary: lift outline pixels with the SAME formula as the
      // hover-highlight (color times boost plus 0.12) so the boundary
      // matches the hover-fill brightness. Emits bright on an extended-
      // range canvas; clamps to a brighter SDR edge otherwise.
      if (outline > 0.5 && u_outlineHdrBoost > 1.0) {
        maskColor = maskColor * u_outlineHdrBoost + 0.12;
      }
      if (u_highlightLabel > 0.5 && abs(inst - u_highlightLabel) < 0.5) {
        if (u_useOutlineColor > 0.5) {
          // Outline overlay: a uniform TRANSLUCENT red tint over the base —
          // a consistent cue that reads over any image.
          maskColor = u_highlightColor;
          alpha = max(alpha, u_highlightAlpha);
        } else {
          // Highlighted fill: the WHOLE cell — interior AND its perimeter pixels —
          // gets the SAME brightened cell color at the same alpha, so a hovered cell
          // reads as ONE uniform fill with its edges matching the inside. Recompute
          // the cell color FRESH (NOT reuse maskColor, which on an outline pixel
          // already carries the outline HDR boost + hue-shifting +0.12, which made
          // the rim brighter and off-hue versus the interior).
          vec3 _hc = (u_usePalette > 0.5) ? paletteColor(label) : hashColor(label);
          maskColor = _hc * u_highlightBoost + 0.12;
          alpha = max(alpha, 0.9);
        }
      }
      color = hasBase ? mix(color, maskColor, alpha) : maskColor;
      outA = max(outA, alpha);
    }
  }
  outColor = vec4(color, outA);
}`;

  function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.warn('LabelGL shader compile failed:', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }
  function link(gl, vsrc, fsrc) {
    const vs = compile(gl, gl.VERTEX_SHADER, vsrc);
    const fs = compile(gl, gl.FRAGMENT_SHADER, fsrc);
    if (!vs || !fs) return null;
    const p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
    gl.deleteShader(vs); gl.deleteShader(fs);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.warn('LabelGL link failed:', gl.getProgramInfoLog(p));
      gl.deleteProgram(p); return null;
    }
    return p;
  }

  // mat3 (column-major for uniformMatrix3fv) mapping image-pixel coords
  // [0,w]x[0,h] (row 0 = top) into the canvas clip box described by
  // `view` = {tx,ty,scale} in normalized [0,1] tile space (pan/zoom).
  // Default view (full tile, no pan) → fills the canvas, y-flipped so the
  // first image row is at the top.
  function orthoTile(w, h, view) {
    const s = (view && view.scale) || 1.0;
    const tx = (view && view.tx) || 0.0;     // pan in clip-x
    const ty = (view && view.ty) || 0.0;
    // x: [0,w] -> [-1,1]*s + tx ;  y: [0,h] -> [1,-1]*s + ty (flip)
    return new Float32Array([
      (2 / w) * s, 0, 0,
      0, (-2 / h) * s, 0,
      -s + tx, s + ty, 1,
    ]);
  }

  class LabelGLRenderer {
    constructor(gl) {
      this.gl = gl;
      this.w = 0; this.h = 0;
      this.labels = null;            // Int32Array CPU copy for labelAt()
      this.paletteSize = 256;
      this.usePalette = 0;
      this.uniformsState = {
        maskOpacity: 0.6, maskVisible: 1, outlinesVisible: 1, maskStyle: 0,
        imageVisible: 0, colorOffset: 0, highlightLabel: 0,
        highlightColor: [1, 0, 0], highlightAlpha: 0.5, highlightBoost: 1.8,
        outlineHdrBoost: 1.0, outlineColor: [1, 0, 0], useOutlineColor: 0,
        baseHeadroom: 1.0, baseLinear: 0,
      };
      const prog = link(gl, VERT, FRAG);
      if (!prog) throw new Error('LabelGLRenderer: program creation failed');
      this.prog = prog;
      this.u = {};
      [
        'matrix', 'baseSampler', 'maskSampler', 'paletteSampler',
        'maskOpacity', 'maskVisible', 'outlinesVisible', 'maskStyle', 'imageVisible',
        'colorOffset', 'paletteSize', 'usePalette', 'highlightLabel',
        'highlightColor', 'highlightAlpha', 'highlightBoost',
        'outlineHdrBoost', 'outlineColor', 'useOutlineColor', 'baseHeadroom', 'baseLinear',
      ].forEach((n) => { this.u[n] = gl.getUniformLocation(prog, 'u_' + n); });

      // unit quad in image-pixel coords + texcoords (matches app.js)
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      const pos = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, pos);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);  // a_position (re-scaled in setLabels)
      const tc = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, tc);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      this.posBuffer = pos;

      this.maskTex = this._mkTex(gl.NEAREST);
      this.paletteTex = this._mkTex(gl.NEAREST);
      // NEAREST, not LINEAR: the base (e.g. a max projection under the labels)
      // must show real pixels on zoom — matching the RGB raster tile's
      // image-rendering:pixelated — instead of bilinearly smoothing.
      this.baseTex = this._mkTex(gl.NEAREST);
      this._emptyBase();
    }

    _mkTex(filter) {
      const gl = this.gl;
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return t;
    }
    _emptyBase() {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array([0, 0, 0, 255]));
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // colorLabels: per-pixel COLOR/group id (palette lookup). instanceLabels:
    // per-pixel INSTANCE id used for hover/picking + outline boundaries —
    // defaults to colorLabels when there's no separate ncolor relabel (then
    // color == instance). Both typed arrays are length w*h.
    setLabels(colorLabels, w, h, instanceLabels) {
      const gl = this.gl;
      this.w = w; this.h = h;
      const inst = instanceLabels || colorLabels;
      // this.labels = INSTANCE ids: labelAt() + the outline boundary scan
      // both want per-cell granularity, not the coarse ncolor group id.
      this.labels = inst.constructor === Int32Array ? inst : Int32Array.from(inst);
      // RGBA-pack: RG = color id (low+high*256), BA = instance id.
      const rgba = new Uint8Array(w * h * 4);
      for (let i = 0; i < w * h; i += 1) {
        const c = colorLabels[i] | 0, v = this.labels[i] | 0;
        rgba[i * 4] = c & 0xff;
        rgba[i * 4 + 1] = (c >> 8) & 0xff;
        rgba[i * 4 + 2] = v & 0xff;
        rgba[i * 4 + 3] = (v >> 8) & 0xff;
      }
      gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      gl.bindTexture(gl.TEXTURE_2D, null);
      // Outlines are detected per DISPLAY pixel in the fragment shader from the
      // mask's instance ids (see _instAt) — no CPU boundary scan / R8 texture.
    }

    // rgba: Uint8Array length n*4 (palette[0]=bg). null → procedural sinebow.
    setPalette(rgba) {
      const gl = this.gl;
      if (!rgba) { this.usePalette = 0; return; }
      this.usePalette = 1;
      this.paletteSize = rgba.length / 4;
      gl.bindTexture(gl.TEXTURE_2D, this.paletteTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.paletteSize, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    setBase(src) {
      const gl = this.gl;
      if (!src) { this.uniformsState.imageVisible = 0; return; }
      this.uniformsState.imageVisible = 1;
      this.uniformsState.baseLinear = 0;   // sRGB-OETF image source
      gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // RAW linear-light f16 base (no image codec): ``data`` = top-down RGBA float16
    // (Uint16Array, the _RawF16Source payload), w×h. Uploaded as RGBA16F so the
    // base carries full HDR precision and skips the PNG encode/decode entirely —
    // the same texture the inline static-HDR controller uses. UNPACK_FLIP_Y is
    // ignored for an ArrayBufferView, so flip rows on the CPU to match setBase's
    // orientation (the shader's baseCoord = (x, 1-y) assumes a flipped upload).
    setBaseRaw(data, w, h) {
      const gl = this.gl;
      if (!data) { this.uniformsState.imageVisible = 0; return; }
      this.uniformsState.imageVisible = 1;
      this.uniformsState.baseLinear = 1;   // linear-light → shader skips EOTF
      const rowU16 = w * 4;
      const flipped = new Uint16Array(data.length);
      for (let y = 0; y < h; y++) {
        flipped.set(data.subarray(y * rowU16, (y + 1) * rowU16), (h - 1 - y) * rowU16);
      }
      gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, flipped);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    setUniforms(o) { Object.assign(this.uniformsState, o); }

    labelAt(px, py) {
      const x = Math.round(px), y = Math.round(py);
      if (!this.labels || x < 0 || y < 0 || x >= this.w || y >= this.h) return 0;
      return this.labels[y * this.w + x] | 0;
    }

    draw(matrix) {
      const gl = this.gl, U = this.u, st = this.uniformsState;
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      // scale unit-quad positions to image dims via the matrix's /w,/h terms,
      // so a_position is in [0,1]; rebind matrix to map [0,1]->clip.
      gl.uniformMatrix3fv(U.matrix, false, matrix);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.baseTex); gl.uniform1i(U.baseSampler, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.maskTex); gl.uniform1i(U.maskSampler, 1);
      // (TEXTURE2 / u_outlineSampler retired — outlines are shader-computed from the mask)
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.paletteTex); gl.uniform1i(U.paletteSampler, 3);
      gl.uniform1f(U.maskOpacity, st.maskOpacity);
      gl.uniform1f(U.maskVisible, st.maskVisible);
      gl.uniform1f(U.outlinesVisible, st.outlinesVisible);
      gl.uniform1f(U.maskStyle, st.maskStyle);
      gl.uniform1f(U.imageVisible, st.imageVisible);
      gl.uniform1f(U.colorOffset, st.colorOffset);
      gl.uniform1f(U.paletteSize, this.paletteSize);
      gl.uniform1f(U.usePalette, this.usePalette);
      gl.uniform1f(U.highlightLabel, st.highlightLabel);
      const hc = st.highlightColor || [1, 0, 0];
      gl.uniform3f(U.highlightColor, hc[0], hc[1], hc[2]);
      gl.uniform1f(U.highlightAlpha, st.highlightAlpha == null ? 0.5 : st.highlightAlpha);
      gl.uniform1f(U.highlightBoost, st.highlightBoost == null ? 1.8 : st.highlightBoost);
      gl.uniform1f(U.outlineHdrBoost, st.outlineHdrBoost);
      gl.uniform1f(U.baseHeadroom, st.baseHeadroom == null ? 1.0 : st.baseHeadroom);
      gl.uniform1f(U.baseLinear, st.baseLinear == null ? 0 : st.baseLinear);
      const oc = st.outlineColor || [1, 0, 0];
      gl.uniform3f(U.outlineColor, oc[0], oc[1], oc[2]);
      gl.uniform1f(U.useOutlineColor, st.useOutlineColor);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
      gl.useProgram(null);
    }
  }

  // matrix mapping a_position in [0,1] (unit quad) → clip, y-flipped, with
  // optional pan/zoom view {scale,tx,ty}. (a_position is unit-quad now.)
  function ortho(view) {
    const s = (view && view.scale) || 1.0;
    const tx = (view && view.tx) || 0.0;
    const ty = (view && view.ty) || 0.0;
    return new Float32Array([
      2 * s, 0, 0,
      0, -2 * s, 0,
      -s + tx, s + ty, 1,
    ]);
  }

  // ── shared helpers (used by both the inline controller and the popup
  //    viewer in io/figure.py, so label tiles render through one path) ──

  function _b64bytes(s) {
    const bin = atob(s), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) u[i] = bin.charCodeAt(i);
    return u;
  }

  // Decode a ``<canvas data-label-tile>``'s attributes into a plain config.
  // The big label/instance matrices come EITHER inline (``data-labels`` base64,
  // small tiles) OR streamed (``data-labels-src`` URL → raw uint16 LE bytes,
  // large tiles — keeps multi-MB matrices out of the SVG document). When
  // streamed, ``labels``/``instances`` are null here; call ``fetchMatrices``
  // (returns a Promise) before building the renderer.
  function decodeAttrs(el) {
    const w = +el.getAttribute('data-w'), h = +el.getAttribute('data-h');
    // data-labels = COLOR/group ids (palette lookup). data-instances =
    // INSTANCE ids for hover/picking; absent when color == instance (no
    // ncolor relabel), in which case the color ids double as instances.
    const lblB64 = el.getAttribute('data-labels');
    const labels = lblB64 ? new Uint16Array(_b64bytes(lblB64).buffer) : null;
    const instAttr = el.getAttribute('data-instances');
    const instances = instAttr
      ? new Uint16Array(_b64bytes(instAttr).buffer) : labels;
    const palAttr = el.getAttribute('data-palette');
    const palette = (palAttr && palAttr !== 'sinebow') ? _b64bytes(palAttr) : null;
    const styleName = el.getAttribute('data-style') || 'both';
    // Optional uniform contour color (#rrggbb). Absent / 'none' → per-cell
    // palette color on edges (current behaviour).
    const ocAttr = el.getAttribute('data-outline-color');
    const oc = (ocAttr && ocAttr !== 'none') ? _hexColor(ocAttr) : null;
    return {
      w, h, labels, instances, palette,
      labelsSrc: el.getAttribute('data-labels-src') || null,
      instancesSrc: el.getAttribute('data-instances-src') || null,
      baseSrc: el.getAttribute('data-base') || null,
      uniforms: {
        maskOpacity: parseFloat(el.getAttribute('data-opacity') || '0.6'),
        maskStyle: styleName === 'outline' ? 2 : (styleName === 'fill' ? 1 : 0),
        outlinesVisible: el.getAttribute('data-outlines') !== '0' ? 1 : 0,
        outlineHdrBoost: parseFloat(el.getAttribute('data-outline-hdr') || '1'),
        outlineColor: oc || [1, 0, 0],
        useOutlineColor: oc ? 1 : 0,
        highlightLabel: 0,
      },
    };
  }

  // Ensure cfg.labels / cfg.instances are populated. If they were streamed
  // (cfg.labelsSrc set, labels null), fetch the raw uint16 LE bytes (resolving
  // the URL through the remote-tile proxy) and mutate cfg in place; otherwise
  // resolve immediately. Returns a Promise<cfg>. Idempotent — a second call
  // (e.g. the popup viewer after the inline controller already fetched) is a
  // no-op because cfg.labels is now set.
  function fetchMatrices(cfg) {
    if (cfg.labels || !cfg.labelsSrc) return Promise.resolve(cfg);
    function resolve(u) {
      try { return (self.__ocdResolveTileUrl ? self.__ocdResolveTileUrl(u) : u); }
      catch (e) { return u; }
    }
    function grab(u, tries) {
      return fetch(resolve(u)).then(function (r) {
        if (r.status === 204) return Promise.reject('204');   // attached off-thread; not ready
        if (!r.ok) return Promise.reject(r.status);
        return r.arrayBuffer();
      }).then(function (b) { return new Uint16Array(b); })
        .catch(function (e) {
          // Retry on 204 AND transient TRANSPORT failures: under heavy kernel
          // compute the GIL-starved in-kernel server / jupyter proxy cuts the
          // response mid-stream (ERR_INCOMPLETE_CHUNKED_ENCODING → TypeError),
          // which would otherwise leave the seg matrix empty (no labels/outlines).
          // Only a real 4xx is fatal. Generous budget + backoff to outlast Run-All.
          var fatal = (typeof e === 'number' && e >= 400 && e < 500);
          if (!fatal && (tries || 0) < 600) {
            return new Promise(function (res) { setTimeout(res, Math.min(250 + (tries || 0) * 30, 2000)); })
              .then(function () { return grab(u, (tries || 0) + 1); });
          }
          throw e;
        });
    }
    var jobs = [grab(cfg.labelsSrc)];
    jobs.push(cfg.instancesSrc ? grab(cfg.instancesSrc) : Promise.resolve(null));
    return Promise.all(jobs).then(function (res) {
      cfg.labels = res[0];
      cfg.instances = res[1] || res[0];
      return cfg;
    });
  }

  // '#rrggbb' → [r,g,b] floats in [0,1]; null on parse failure.
  function _hexColor(s) {
    const m = /^#?([0-9a-fA-F]{6})$/.exec(s.trim());
    if (!m) return null;
    const v = parseInt(m[1], 16);
    return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
  }

  // Apply a decodeAttrs config to ANY label renderer (WebGL2 or WebGPU — both
  // expose setLabels/setPalette/setUniforms/setBase). Loads the optional base
  // image + streamed matrices async, calling ``onRender`` once they arrive so
  // the caller can repaint. Backend-agnostic so buildRenderer (WebGL2) and
  // createLabelRenderer (the dispatcher) share one config path.
  function _applyConfig(r, cfg, onRender) {
    if (cfg.labels) {
      r.setLabels(cfg.labels, cfg.w, cfg.h, cfg.instances);
    } else if (cfg.labelsSrc) {
      // Streamed matrix (large tiles ship the labels as a tileserve attachment,
      // not inline base64). The figure-shell controller is FROZEN at import, so
      // a running kernel can hold a stale copy that never fetches; do it HERE
      // (label_gl.js is re-read each render) so the matrix loads + the outline
      // draws regardless of controller version — no kernel restart. The canvas
      // is transparent until the bytes arrive, then onRender repaints.
      fetchMatrices(cfg).then(function () {
        r.setLabels(cfg.labels, cfg.w, cfg.h, cfg.instances);
        if (onRender) onRender();
      }).catch(function (e) { console.warn('LabelGL matrix fetch failed:', e); });
    }
    r.setPalette(cfg.palette);
    r.setUniforms(cfg.uniforms);
    if (cfg.baseSrc) {
      const bimg = new Image();
      // The base is usually a figure-server URL (cross-origin to the page);
      // without this the texImage2D upload throws a security error (tainted
      // texture) that the catch below swallows → base silently never shows.
      bimg.crossOrigin = 'anonymous';
      bimg.onload = function () {
        try { r.setBase(bimg); if (onRender) onRender(); }
        catch (e) { console.warn('LabelGL base upload failed:', e); }
      };
      bimg.src = cfg.baseSrc;
    }
    return r;
  }

  // Build a fully-configured WebGL2 renderer on ``gl`` from a config (decodeAttrs
  // output). Synchronous; the explicit-WebGL2 path for callers that already hold
  // a gl context. For backend dispatch use createLabelRenderer.
  function buildRenderer(gl, cfg, onRender) {
    return _applyConfig(new LabelGLRenderer(gl), cfg, onRender);
  }

  // Dispatcher: WebGPU (HDR — true >1.0 outline/highlight glow via
  // toneMapping:'extended') when available, else WebGL2 (SDR). Mirrors
  // colormap_image.js createColormapRenderer: probe the shared GPU device FIRST
  // (LabelGPURenderer.create awaits HdrColormap.getDevice, which doesn't touch
  // the canvas) so the WebGL2 fallback can still claim the canvas context if
  // WebGPU is unusable. Returns Promise<renderer>; the renderer exposes the SAME
  // interface as LabelGLRenderer, so callers `await` then use draw()/labelAt()
  // unchanged. ``opts.forceWebgl`` pins WebGL2 (e.g. SDR tiles dodging the
  // WebGPU cold-init flash, exactly as the colormap path does for hdr=false).
  function createLabelRenderer(canvas, cfg, onRender, opts) {
    opts = opts || {};
    const LG = (typeof window !== 'undefined' && window.LabelGPU) || (typeof self !== 'undefined' && self.LabelGPU) || null;
    function webgl2() {
      const gl = canvas.getContext('webgl2');
      if (!gl) throw new Error('createLabelRenderer: no webgl2');
      return _applyConfig(new LabelGLRenderer(gl), cfg, onRender);
    }
    if (!opts.forceWebgl && LG && LG.LabelGPURenderer && typeof navigator !== 'undefined' && navigator.gpu) {
      return LG.LabelGPURenderer.create(canvas, opts)
        .then(function (r) { return _applyConfig(r, cfg, onRender); })
        .catch(function () { return webgl2(); });   // WebGPU unusable → WebGL2 (SDR)
    }
    try { return Promise.resolve(webgl2()); }
    catch (e) { return Promise.reject(e); }
  }

  // mat3 (column-major) for LabelGLRenderer.draw that fits + pans + zooms
  // the label image into a popup canvas, matching io/figure.py's WebGL
  // image-viewer vertex transform EXACTLY (so pan/zoom feels identical).
  // ``state`` = {s, tx, ty} (zoom + CSS-px pan); cssW/cssH = canvas CSS size.
  // Reduces to ``ortho()`` for the trivial fit (verified).
  function mat3ForFit(state, imgW, imgH, cssW, cssH) {
    const s = (state && state.s) || 1, tx = (state && state.tx) || 0,
          ty = (state && state.ty) || 0;
    const fit = Math.min(cssW / imgW, cssH / imgH);
    const originX = (cssW * 0.5 - imgW * fit * 0.5) * s + tx;
    const originY = (cssH * 0.5 - imgH * fit * 0.5) * s + ty;
    const sizeX = imgW * fit * s, sizeY = imgH * fit * s;
    const sx = 2 * sizeX / cssW, txc = 2 * originX / cssW - 1;
    const sy = -2 * sizeY / cssH, tyc = 1 - 2 * originY / cssH;
    return new Float32Array([sx, 0, 0, 0, sy, 0, txc, tyc, 1]);
  }

  // Inverse mapping: a popup-canvas CSS-local point → image pixel coords,
  // for hover (labelAt). Returns null if outside the image.
  function imagePointFromCss(state, imgW, imgH, cssW, cssH, localX, localY) {
    const s = (state && state.s) || 1, tx = (state && state.tx) || 0,
          ty = (state && state.ty) || 0;
    const fit = Math.min(cssW / imgW, cssH / imgH);
    const originX = (cssW * 0.5 - imgW * fit * 0.5) * s + tx;
    const originY = (cssH * 0.5 - imgH * fit * 0.5) * s + ty;
    const dispW = imgW * fit * s, dispH = imgH * fit * s;
    const u = (localX - originX) / dispW, v = (localY - originY) / dispH;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return { px: u * imgW, py: v * imgH };
  }

  return {
    LabelGLRenderer, ortho, orthoTile, VERT, FRAG,
    decodeAttrs, fetchMatrices, buildRenderer, createLabelRenderer, mat3ForFit, imagePointFromCss,
  };
}));
