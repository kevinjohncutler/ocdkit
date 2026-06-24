/*
 * LabelGPURenderer — WebGPU label/segmentation renderer.
 *
 * The WebGPU sibling of LabelGLRenderer (label_gl.js). Same interface
 * (setLabels / setPalette / setBase / setBaseRaw / setUniforms / labelAt /
 * draw) and the SAME mask conventions (RG = colour/group id, BA = instance id;
 * screen-space 8-neighbour outline; outline HDR boost; hover highlight), so it
 * is a drop-in behind the `createLabelRenderer` dispatcher in label_gl.js.
 *
 * Why it exists: WebGL2 has no `toneMapping:'extended'` equivalent, so the
 * outline/highlight HDR boost (`maskColor * boost + 0.12`, >1.0) CLAMPS to SDR
 * white on an HDR display instead of glowing. This renders to an
 * `rgba16float` / display-p3 / `toneMapping:'extended'` WebGPU canvas — the same
 * config HdrColormapRenderer uses — so >1.0 emits true HDR. It shares
 * HdrColormap's ONE pre-warmed GPUDevice (no second cold init).
 *
 * The fragment math is byte-for-byte the GLSL FRAG of label_gl.js, validated
 * pixel-identical headless (wgpu-native) in outputs/repro/label_gpu/parity.py
 * (max abs diff 8e-4, HDR >1.0 confirmed). One compositing difference: a WebGPU
 * canvas has no straight-alpha mode, so the fragment emits PREMULTIPLIED
 * (`color*outA, outA`) under `alphaMode:'premultiplied'`, which composites the
 * transparent label overlay identically to WebGL's premultipliedAlpha:false.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.LabelGPU = mod;
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // WGSL — port of label_gl.js VERT+FRAG. dpdx/dpdy are hoisted to the top of
  // fs (uniform control flow); every mask/palette/base read is textureSampleLevel
  // (explicit LOD — implicit-derivative samples are illegal in the non-uniform
  // branches). Vertex uses @builtin(vertex_index) corners (unit quad [0,1]) ×
  // u.matrix, matching LabelGL's [0,1]-quad → clip convention, so the SAME mat3
  // (orthoTile / ortho / mat3ForFit) feeds either backend.
  const WGSL = `
struct U {
  matrix: mat3x3<f32>,
  highlightColor: vec3<f32>,
  outlineColor: vec3<f32>,
  maskOpacity: f32, maskVisible: f32, outlinesVisible: f32, maskStyle: f32,
  imageVisible: f32, colorOffset: f32, paletteSize: f32, usePalette: f32,
  highlightLabel: f32, highlightAlpha: f32, highlightBoost: f32, outlineHdrBoost: f32,
  useOutlineColor: f32, baseHeadroom: f32, baseLinear: f32, pad0: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var baseTex: texture_2d<f32>;
@group(0) @binding(3) var maskTex: texture_2d<f32>;
@group(0) @binding(4) var palTex: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var corners = array<vec2<f32>, 4>(vec2<f32>(0, 0), vec2<f32>(1, 0), vec2<f32>(0, 1), vec2<f32>(1, 1));
  let c = corners[i];
  let p = u.matrix * vec3<f32>(c, 1.0);
  var o: VOut;
  o.pos = vec4<f32>(p.xy, 0.0, 1.0);
  o.uv = c;
  return o;
}

fn eotf(c: f32) -> f32 { return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045); }
fn oetf(c: f32) -> f32 { let x = max(c, 0.0); return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, 12.92 * x, x <= 0.0031308); }
fn baseHdr(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(oetf(eotf(c.r) * u.baseHeadroom), oetf(eotf(c.g) * u.baseHeadroom), oetf(eotf(c.b) * u.baseHeadroom)); }
fn sinebow(t: f32) -> vec3<f32> {
  let a = 6.28318530718 * fract(t);
  return vec3<f32>(sin(a) * 0.5 + 0.5, sin(a + 2.09439510239) * 0.5 + 0.5, sin(a + 4.18879020479) * 0.5 + 0.5);
}
fn hashColor(label: f32) -> vec3<f32> { return sinebow(fract(label * 0.61803398875 + u.colorOffset)); }
fn paletteColor(label: f32) -> vec3<f32> {
  let size = max(u.paletteSize, 1.0);
  let idx = label - size * floor(label / size);
  return textureSampleLevel(palTex, samp, vec2<f32>((idx + 0.5) / size, 0.5), 0.0).rgb;
}
fn instAt(uv: vec2<f32>) -> f32 {
  let p = textureSampleLevel(maskTex, samp, uv, 0.0);
  return floor(p.b * 255.0 + 0.5) + floor(p.a * 255.0 + 0.5) * 256.0;
}

@fragment
fn fs(@location(0) v_uv: vec2<f32>) -> @location(0) vec4<f32> {
  let scrPx = vec2<f32>(abs(dpdx(v_uv.x)), abs(dpdy(v_uv.y)));
  let baseCoord = vec2<f32>(v_uv.x, 1.0 - v_uv.y);
  let hasBase = u.imageVisible > 0.5;
  var color = vec3<f32>(0.0);
  var outA = select(0.0, 1.0, hasBase);
  if (hasBase) {
    color = textureSampleLevel(baseTex, samp, baseCoord, 0.0).rgb;
    if (u.baseLinear > 0.5) {
      color = vec3<f32>(oetf(color.r * u.baseHeadroom), oetf(color.g * u.baseHeadroom), oetf(color.b * u.baseHeadroom));
    } else if (u.baseHeadroom > 1.0001) {
      color = baseHdr(color);
    }
  }
  if (u.maskVisible > 0.5 && u.maskOpacity > 0.0) {
    let packed = textureSampleLevel(maskTex, samp, v_uv, 0.0);
    let label = floor(packed.r * 255.0 + 0.5) + floor(packed.g * 255.0 + 0.5) * 256.0;
    let inst  = floor(packed.b * 255.0 + 0.5) + floor(packed.a * 255.0 + 0.5) * 256.0;
    if (label > 0.5) {
      var alpha = clamp(u.maskOpacity, 0.0, 1.0);
      var outline = 0.0;
      if (u.outlinesVisible > 0.5) {
        let imgPx = 1.0 / vec2<f32>(textureDimensions(maskTex, 0));
        let d = max(scrPx, imgPx);
        if (instAt(v_uv + vec2<f32>(d.x, 0.0)) != inst || instAt(v_uv - vec2<f32>(d.x, 0.0)) != inst
         || instAt(v_uv + vec2<f32>(0.0, d.y)) != inst || instAt(v_uv - vec2<f32>(0.0, d.y)) != inst
         || instAt(v_uv + d) != inst || instAt(v_uv - d) != inst
         || instAt(v_uv + vec2<f32>(d.x, -d.y)) != inst || instAt(v_uv + vec2<f32>(-d.x, d.y)) != inst) { outline = 1.0; }
      }
      if (u.maskStyle > 1.5) { alpha = alpha * outline; }
      else if (u.maskStyle < 0.5 && u.outlinesVisible > 0.5) { alpha = mix(alpha * 0.5, alpha, outline); }
      var maskColor = select(hashColor(label), paletteColor(label), u.usePalette > 0.5);
      if (outline > 0.5 && u.useOutlineColor > 0.5) { maskColor = u.outlineColor; alpha = 1.0; }
      if (outline > 0.5 && u.outlineHdrBoost > 1.0) { maskColor = maskColor * u.outlineHdrBoost + 0.12; }
      if (u.highlightLabel > 0.5 && abs(inst - u.highlightLabel) < 0.5) {
        if (u.useOutlineColor > 0.5) {
          maskColor = u.highlightColor;
          alpha = max(alpha, u.highlightAlpha);
        } else {
          let hc = select(hashColor(label), paletteColor(label), u.usePalette > 0.5);
          maskColor = hc * u.highlightBoost + 0.12;
          alpha = max(alpha, u.maskOpacity);   // match the outline's translucency (was 0.9)
        }
      }
      color = select(maskColor, mix(color, maskColor, alpha), hasBase);
      outA = max(outA, alpha);
    }
  }
  // PREMULTIPLIED for alphaMode:'premultiplied' (a WebGPU canvas has no
  // straight-alpha mode). hasBase → outA==1 → unchanged; transparent overlay →
  // color*alpha, which composites identically to WebGL's premultipliedAlpha:false.
  return vec4<f32>(color * outA, outA);
}`;

  function getDevice() {
    const HC = (typeof window !== 'undefined' && window.HdrColormap) || (typeof self !== 'undefined' && self.HdrColormap) || null;
    if (!HC || !HC.getDevice) throw new Error('LabelGPU: HdrColormap.getDevice unavailable');
    return HC.getDevice();   // ONE shared, pre-warmed device across all ocdkit WebGPU
  }

  class LabelGPURenderer {
    // opts: { device? }. Async because it needs a GPUDevice + canvas config.
    static async create(canvas, opts) {
      const r = new LabelGPURenderer();
      await r._init(canvas, opts || {});
      return r;
    }

    async _init(canvas, opts) {
      this.canvas = canvas;
      this.device = opts.device || await getDevice();
      this.ctx = canvas.getContext('webgpu');
      this.backend = 'webgpu'; this.hdr = true;
      this.w = 0; this.h = 0;
      this.labels = null;               // Int32Array instance ids for labelAt()
      this.paletteSize = 256;
      this.usePalette = 0;
      this._cw = 0; this._ch = 0; this._raf = 0;
      this.uniformsState = {
        maskOpacity: 0.6, maskVisible: 1, outlinesVisible: 1, maskStyle: 0,
        imageVisible: 0, colorOffset: 0, highlightLabel: 0,
        highlightColor: [1, 0, 0], highlightAlpha: 0.5, highlightBoost: 1.8,
        outlineHdrBoost: 1.0, outlineColor: [1, 0, 0], useOutlineColor: 0,
        baseHeadroom: 1.0, baseLinear: 0,
      };
      this._configure();

      const mod = this.device.createShaderModule({ code: WGSL });
      this.pipeline = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: mod, entryPoint: 'vs' },
        fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-strip' },
      });
      // unit-quad via vertex_index → no vertex buffer needed.
      this.uBuf = this.device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.samp = this.device.createSampler({
        magFilter: 'nearest', minFilter: 'nearest',
        addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      });
      this.maskTex = this._mkTex(1, 1, 'rgba8unorm');
      this.palTex = this._mkTex(1, 1, 'rgba8unorm');
      this.baseTex = this._mkTex(1, 1, 'rgba8unorm');
      this._rebuildBind();

      if (typeof ResizeObserver !== 'undefined') {
        this._ro = new ResizeObserver(() => this.requestRedraw());
        this._ro.observe(this.canvas);
      }
    }

    _configure() {
      const base = { device: this.device, format: 'rgba16float', colorSpace: 'display-p3', alphaMode: 'premultiplied' };
      try { this.ctx.configure(Object.assign({}, base, { toneMapping: { mode: 'extended' } })); }
      catch (e) { this.ctx.configure(base); }
    }

    _mkTex(w, h, format) {
      return this.device.createTexture({
        size: [w, h], format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    _rebuildBind() {
      this._bind = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uBuf } },
          { binding: 1, resource: this.samp },
          { binding: 2, resource: this.baseTex.createView() },
          { binding: 3, resource: this.maskTex.createView() },
          { binding: 4, resource: this.palTex.createView() },
        ],
      });
    }

    setLabels(colorLabels, w, h, instanceLabels) {
      this.w = w; this.h = h;
      const inst = instanceLabels || colorLabels;
      this.labels = inst.constructor === Int32Array ? inst : Int32Array.from(inst);
      const rgba = new Uint8Array(w * h * 4);
      for (let i = 0; i < w * h; i += 1) {
        const c = colorLabels[i] | 0, v = this.labels[i] | 0;
        rgba[i * 4] = c & 0xff; rgba[i * 4 + 1] = (c >> 8) & 0xff;
        rgba[i * 4 + 2] = v & 0xff; rgba[i * 4 + 3] = (v >> 8) & 0xff;
      }
      if (this.maskTex.width !== w || this.maskTex.height !== h) {
        this.maskTex.destroy(); this.maskTex = this._mkTex(w, h, 'rgba8unorm'); this._rebuildBind();
      }
      this.device.queue.writeTexture({ texture: this.maskTex }, rgba, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h]);
      this.requestRedraw();
    }

    setPalette(rgba) {
      if (!rgba) { this.usePalette = 0; this.requestRedraw(); return; }
      this.usePalette = 1;
      const n = rgba.length / 4;
      this.paletteSize = n;
      if (this.palTex.width !== n) { this.palTex.destroy(); this.palTex = this._mkTex(n, 1, 'rgba8unorm'); this._rebuildBind(); }
      const u8 = rgba instanceof Uint8Array ? rgba : new Uint8Array(rgba);
      this.device.queue.writeTexture({ texture: this.palTex }, u8, { bytesPerRow: n * 4, rowsPerImage: 1 }, [n, 1]);
      this.requestRedraw();
    }

    setBase(src) {
      if (!src) { this.uniformsState.imageVisible = 0; this.requestRedraw(); return; }
      const finish = (bmp) => {
        const w = bmp.width, h = bmp.height;
        if (this.baseTex.width !== w || this.baseTex.height !== h || this.baseTex.format !== 'rgba8unorm') {
          this.baseTex.destroy(); this.baseTex = this._mkTex(w, h, 'rgba8unorm'); this._rebuildBind();
        }
        // flipY → match the shader's baseCoord = (x, 1-y) (same as WebGL UNPACK_FLIP_Y).
        this.device.queue.copyExternalImageToTexture({ source: bmp, flipY: true }, { texture: this.baseTex }, [w, h]);
        this.uniformsState.imageVisible = 1; this.uniformsState.baseLinear = 0;
        this.requestRedraw();
      };
      if (typeof createImageBitmap === 'function' && !(src instanceof ImageBitmap)) {
        createImageBitmap(src).then(finish).catch((e) => console.warn('LabelGPU base bitmap failed:', e));
      } else { finish(src); }
    }

    // RAW linear-light f16 base: ``data`` = top-down RGBA float16 (Uint16Array),
    // w×h. Uploaded as rgba16float (HDR precision, no codec). CPU-flip rows to
    // match setBase's flipped orientation (the shader assumes a flipped upload).
    setBaseRaw(data, w, h) {
      if (!data) { this.uniformsState.imageVisible = 0; this.requestRedraw(); return; }
      const rowU16 = w * 4;
      const flipped = new Uint16Array(data.length);
      for (let y = 0; y < h; y += 1) flipped.set(data.subarray(y * rowU16, (y + 1) * rowU16), (h - 1 - y) * rowU16);
      if (this.baseTex.width !== w || this.baseTex.height !== h || this.baseTex.format !== 'rgba16float') {
        this.baseTex.destroy(); this.baseTex = this._mkTex(w, h, 'rgba16float'); this._rebuildBind();
      }
      this.device.queue.writeTexture({ texture: this.baseTex }, flipped, { bytesPerRow: w * 8, rowsPerImage: h }, [w, h]);
      this.uniformsState.imageVisible = 1; this.uniformsState.baseLinear = 1;
      this.requestRedraw();
    }

    setUniforms(o) { Object.assign(this.uniformsState, o); this.requestRedraw(); }

    labelAt(px, py) {
      const x = Math.round(px), y = Math.round(py);
      if (!this.labels || x < 0 || y < 0 || x >= this.w || y >= this.h) return 0;
      return this.labels[y * this.w + x] | 0;
    }

    // matrix: column-major mat3 (image-[0,1]-quad → clip), same as LabelGL.draw.
    // Stored so a ResizeObserver/dpr redraw can repaint without a re-supplied matrix.
    draw(matrix) {
      if (matrix) this._matrix = matrix;
      this._renderNow();
    }
    requestRedraw() { if (!this._raf && typeof requestAnimationFrame !== 'undefined') this._raf = requestAnimationFrame(() => { this._raf = 0; this._renderNow(); }); }

    _renderNow() {
      const M = this._matrix;
      if (!M || !this.w) return;
      const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
      // Size from the ACTUAL on-screen rect, NOT clientWidth: a label tile lives in
      // an SVG <foreignObject>, so clientWidth is the unscaled SVG-user-unit size
      // while getBoundingClientRect() is the real post-viewBox-scale pixel size.
      // Backing at clientWidth*dpr under-renders the canvas (e.g. 256 for a 512-device-px
      // display) → the screen-space outline subsamples + breaks. The popup canvas
      // isn't SVG-scaled, so rect.width == clientWidth there (no change).
      const rect = this.canvas.getBoundingClientRect();
      const cw = Math.max(1, Math.round((rect.width || this.canvas.clientWidth) * dpr));
      const ch = Math.max(1, Math.round((rect.height || this.canvas.clientHeight) * dpr));
      if (cw !== this._cw || ch !== this._ch) { this._cw = cw; this._ch = ch; this.canvas.width = cw; this.canvas.height = ch; this._configure(); }

      const st = this.uniformsState;
      const u = new Float32Array(36);                 // 144 bytes, std140 layout
      u[0] = M[0]; u[1] = M[1]; u[2] = M[2];          // matrix col0 (padded vec4)
      u[4] = M[3]; u[5] = M[4]; u[6] = M[5];          // col1
      u[8] = M[6]; u[9] = M[7]; u[10] = M[8];         // col2
      const hc = st.highlightColor || [1, 0, 0], oc = st.outlineColor || [1, 0, 0];
      u[12] = hc[0]; u[13] = hc[1]; u[14] = hc[2];    // highlightColor @ float 12
      u[16] = oc[0]; u[17] = oc[1]; u[18] = oc[2];    // outlineColor   @ float 16
      // scalars @ float 19 (byte 76): vec3 size 12 packs the next f32 at 76, not 80.
      u[19] = st.maskOpacity; u[20] = st.maskVisible; u[21] = st.outlinesVisible; u[22] = st.maskStyle;
      u[23] = st.imageVisible; u[24] = st.colorOffset; u[25] = this.paletteSize; u[26] = this.usePalette;
      u[27] = st.highlightLabel; u[28] = st.highlightAlpha == null ? 0.5 : st.highlightAlpha;
      u[29] = st.highlightBoost == null ? 1.8 : st.highlightBoost; u[30] = st.outlineHdrBoost;
      u[31] = st.useOutlineColor; u[32] = st.baseHeadroom == null ? 1.0 : st.baseHeadroom;
      u[33] = st.baseLinear == null ? 0 : st.baseLinear;
      this.device.queue.writeBuffer(this.uBuf, 0, u);

      const enc = this.device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      });
      pass.setPipeline(this.pipeline); pass.setBindGroup(0, this._bind); pass.draw(4); pass.end();
      this.device.queue.submit([enc.finish()]);
    }

    destroy() {
      if (this._ro) { try { this._ro.disconnect(); } catch (e) { /* ignore */ } }
      [this.maskTex, this.palTex, this.baseTex].forEach((t) => { try { t.destroy(); } catch (e) { /* ignore */ } });
    }
  }

  return { LabelGPURenderer, getDevice, WGSL };
}));
