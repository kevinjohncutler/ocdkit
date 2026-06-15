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
uniform sampler2D u_outlineSampler;
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
  }
  if (u_maskVisible > 0.5 && u_maskOpacity > 0.0) {
    vec4 packed = texture(u_maskSampler, v_texCoord);
    float label = floor(packed.r * 255.0 + 0.5) + floor(packed.g * 255.0 + 0.5) * 256.0;  // color/group id
    float inst  = floor(packed.b * 255.0 + 0.5) + floor(packed.a * 255.0 + 0.5) * 256.0;  // instance id (hover)
    if (label > 0.5) {
      float alpha = clamp(u_maskOpacity, 0.0, 1.0);
      float outline = 0.0;
      if (u_outlinesVisible > 0.5) {
        outline = texture(u_outlineSampler, v_texCoord).r > 0.5 ? 1.0 : 0.0;
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
          // Fill segmentation: emphasize the cell's OWN (ncolor) color —
          // brightened (HDR-bright on an extended-range canvas), nearly
          // opaque so it pops while still reading as that cell's color.
          maskColor = maskColor * u_highlightBoost + 0.12;
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
      this.outlineState = null;      // Uint8Array boundary flags
      this.paletteSize = 256;
      this.usePalette = 0;
      this.uniformsState = {
        maskOpacity: 0.6, maskVisible: 1, outlinesVisible: 1, maskStyle: 0,
        imageVisible: 0, colorOffset: 0, highlightLabel: 0,
        highlightColor: [1, 0, 0], highlightAlpha: 0.5, highlightBoost: 1.8,
        outlineHdrBoost: 1.0, outlineColor: [1, 0, 0], useOutlineColor: 0,
      };
      const prog = link(gl, VERT, FRAG);
      if (!prog) throw new Error('LabelGLRenderer: program creation failed');
      this.prog = prog;
      this.u = {};
      [
        'matrix', 'baseSampler', 'maskSampler', 'outlineSampler', 'paletteSampler',
        'maskOpacity', 'maskVisible', 'outlinesVisible', 'maskStyle', 'imageVisible',
        'colorOffset', 'paletteSize', 'usePalette', 'highlightLabel',
        'highlightColor', 'highlightAlpha', 'highlightBoost',
        'outlineHdrBoost', 'outlineColor', 'useOutlineColor',
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
      this.outlineTex = this._mkTex(gl.NEAREST);
      this.paletteTex = this._mkTex(gl.NEAREST);
      this.baseTex = this._mkTex(gl.LINEAR);
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
      this._computeOutline();
      this._uploadOutlineFull();
    }

    // boundary = any 4-neighbour has a different instance id (mirror
    // app.js outlineState — uses INSTANCE ids, see this.labels above)
    _computeOutline() {
      const { w, h, labels } = this;
      const o = new Uint8Array(w * h);
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          const i = y * w + x;
          const v = labels[i];
          if (v === 0) continue;
          let edge = (x === 0) || (y === 0) || (x === w - 1) || (y === h - 1);
          if (!edge) {
            edge = labels[i - 1] !== v || labels[i + 1] !== v ||
                   labels[i - w] !== v || labels[i + w] !== v;
          }
          if (edge) o[i] = 1;
        }
      }
      this.outlineState = o;
    }
    _uploadOutlineFull() {
      const gl = this.gl;
      const buf = new Uint8Array(this.w * this.h);
      for (let i = 0; i < buf.length; i += 1) buf[i] = this.outlineState[i] ? 255 : 0;
      gl.bindTexture(gl.TEXTURE_2D, this.outlineTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, this.w, this.h, 0, gl.RED, gl.UNSIGNED_BYTE, buf);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
    // dirty-region upload (mirror app.js uploadOutlineRegion) — for edited masks
    uploadOutlineRegion(rect) {
      const gl = this.gl, { w } = this;
      const area = rect.width * rect.height;
      if (area <= 0) return;
      const buf = new Uint8Array(area);
      let off = 0;
      for (let r = 0; r < rect.height; r += 1) {
        const base = (rect.y + r) * w + rect.x;
        for (let c = 0; c < rect.width; c += 1) buf[off++] = this.outlineState[base + c] ? 255 : 0;
      }
      gl.bindTexture(gl.TEXTURE_2D, this.outlineTex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, rect.x, rect.y, rect.width, rect.height,
        gl.RED, gl.UNSIGNED_BYTE, buf);
      gl.bindTexture(gl.TEXTURE_2D, null);
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
      gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
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
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.outlineTex); gl.uniform1i(U.outlineSampler, 2);
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
  function decodeAttrs(el) {
    const w = +el.getAttribute('data-w'), h = +el.getAttribute('data-h');
    // data-labels = COLOR/group ids (palette lookup). data-instances =
    // INSTANCE ids for hover/picking; absent when color == instance (no
    // ncolor relabel), in which case the color ids double as instances.
    const labels = new Uint16Array(_b64bytes(el.getAttribute('data-labels')).buffer);
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

  // '#rrggbb' → [r,g,b] floats in [0,1]; null on parse failure.
  function _hexColor(s) {
    const m = /^#?([0-9a-fA-F]{6})$/.exec(s.trim());
    if (!m) return null;
    const v = parseInt(m[1], 16);
    return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
  }

  // Build a fully-configured renderer on ``gl`` from a config (decodeAttrs
  // output). Loads the optional base image async, calling ``onRender`` once
  // it arrives so the caller can repaint.
  function buildRenderer(gl, cfg, onRender) {
    const r = new LabelGLRenderer(gl);
    r.setLabels(cfg.labels, cfg.w, cfg.h, cfg.instances);
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
    decodeAttrs, buildRenderer, mat3ForFit, imagePointFromCss,
  };
}));
