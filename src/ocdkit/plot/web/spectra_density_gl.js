/* spectra_density_gl.js — live client-side spectra-density renderer (WebGPU).
 *
 * Renders the "datashaded" spectra panel in the browser at the canvas's true
 * device resolution (getBoundingClientRect × devicePixelRatio), so it stays
 * crisp at any on-screen size with no server rasterize + PNG/JXL round-trip.
 *
 * Pipeline — ALL on the GPU in a single submit (no readback, no CPU colorize,
 * no mapAsync await → paints in the same frame):
 *   1. col + row compute passes rasterize the lines into an atomic count buffer
 *      using the exact the host V11 half-open dedup + perpendicular cross-product
 *      SDF + miter clip (lines.py RASTER_TEMPLATE) — each line deposits at most
 *      once per pixel, so vertices never produce "hotspots". (no-shared-mem,
 *      one thread per (line,major); full scan — the reach test skips far segs.)
 *   2. histogram compute: counts → per-count bins (count = density/scale).
 *   3. cdf compute (1 thread): exact eq-hist CDF over nonzero counts
 *      (cdf[c] = #{0<dens≤c}/#{dens>0}, matches numpy _eq_hist).
 *   4. colormap render pass → the WebGPU canvas: count → cdf[count] → LUT,
 *      alpha = count>0 (premultiplied; empty pixels stay transparent so the
 *      vector shading/refs behind/over it show through).
 *
 * Hover highlight (separate 2D overlay canvas): SpectraGL.highlight strokes the
 * nearest line — port of the GUI's findNearestLineInPane + drawHighlightOnPane.
 *
 * Requires WebGPU (Chromium today). render() resolves false if unavailable.
 *
 * Data contract — `<canvas data-spectra-density="1">` carries (base64 LE):
 *   data-ylines Float32 (numLines×numPoints), data-xpix Float32 (numPoints),
 *   data-num-lines, data-num-points, data-plot-w, data-intervals JSON,
 *   data-line-width, data-ylo, data-yhi, data-lut Uint8 (256*4 RGBA).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SpectraGL = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var COVERAGE_SCALE = 256.0;

  function b64bytes(s) {
    var bin = atob(s), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }

  function decodeAttrs(cv) {
    return {
      numLines: parseInt(cv.getAttribute('data-num-lines'), 10),
      numPoints: parseInt(cv.getAttribute('data-num-points'), 10),
      yLines: new Float32Array(b64bytes(cv.getAttribute('data-ylines')).buffer),
      xPix: new Float32Array(b64bytes(cv.getAttribute('data-xpix')).buffer),
      plotW: parseFloat(cv.getAttribute('data-plot-w')),
      intervals: JSON.parse(cv.getAttribute('data-intervals') || '[]'),
      lineWidth: parseFloat(cv.getAttribute('data-line-width')) || 1.0,
      yLo: parseFloat(cv.getAttribute('data-ylo')),
      yHi: parseFloat(cv.getAttribute('data-yhi')),
      lut: b64bytes(cv.getAttribute('data-lut')),
      cellIds: cv.getAttribute('data-cellids')
        ? new Int32Array(b64bytes(cv.getAttribute('data-cellids')).buffer) : null,
      cellLabels: JSON.parse(cv.getAttribute('data-celllabels') || '[]'),
    };
  }

  // ── WGSL: raster (col/row) ───────────────────────────────────────────────
  function rasterWGSL(isCol) {
    var MAJOR_DIM = isCol ? 'U.width' : 'U.height';
    var MINOR_DIM = isCol ? 'U.height' : 'U.width';
    var MAJ1 = isCol ? 'x1' : 'y1', MAJ2 = isCol ? 'x2' : 'y2';
    var MIN1 = isCol ? 'y1' : 'x1', MIN2 = isCol ? 'y2' : 'x2';
    var DEN_COND = isCol ? 'abs(d_major) >= 1.0'
                         : 'abs(d_minor) < 1.0 && abs(d_major) >= 1.0';
    var GI = isCol ? 'u32(mi) * U.width + major_idx'
                   : 'major_idx * U.width + u32(mi)';
    var head = `
struct U_ { width:u32, height:u32, num_lines:u32, num_points:u32,
            half_width:f32, scale:f32, disp_stride:u32, _p1:f32 }
@group(0) @binding(0) var<uniform> U: U_;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> ys: array<f32>;
@group(0) @binding(3) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> seg_valid: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let MAJOR = ${MAJOR_DIM};
  let gi = gid.y * U.disp_stride + gid.x;   // 2D dispatch → linear index (each gid dim is capped at 65535)
  if (gi >= U.num_lines * MAJOR) { return; }
  let line = gi / MAJOR;
  let major_idx = gi % MAJOR;
  let major_pos = f32(major_idx) + 0.5;
  let MINOR_DIM = ${MINOR_DIM};
  let half_width = U.half_width;
  let scale = U.scale;
  let reach = half_width + 1.5;
  let base = line * U.num_points;
  let num_segs = U.num_points - 1u;
  ${isCol ? '' : 'var last_den_major: i32 = -999;'}
  for (var seg = 0u; seg < num_segs; seg++) {
    if (seg_valid[seg] == 0u) { continue; }
    let x1 = xs[base + seg]; let x2 = xs[base + seg + 1u];
    let y1 = ys[base + seg]; let y2 = ys[base + seg + 1u];
    let dx = x2 - x1; let dy = y2 - y1;
    let maj1 = ${MAJ1}; let maj2 = ${MAJ2};
    let min1 = ${MIN1}; let min2 = ${MIN2};
    let d_major = maj2 - maj1; let d_minor = min2 - min1;
    let maj_lo = min(maj1, maj2); let maj_hi = max(maj1, maj2);
    if (major_pos < maj_lo - reach || major_pos > maj_hi + reach) {
      ${isCol ? '' : 'last_den_major = -999;'}
      continue;
    }
    let seg_len_sq = dx * dx + dy * dy;
    let seg_len = sqrt(seg_len_sq);
    if (seg_len < 0.01) { continue; }
    let den_eligible = ${DEN_COND};
`;
    var col = `
    if (den_eligible) {
      var in_range = major_pos >= maj_lo && major_pos < maj_hi;
      if (seg == num_segs - 1u) { in_range = major_pos >= maj_lo && major_pos <= maj_hi; }
      if (in_range) {
        let sec_den = seg_len / max(abs(d_major), 0.01);
        let fill_hw = half_width * sec_den;
        let den_tc = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
        let minor_at = min1 + den_tc * d_minor;
        let mi_lo = i32(floor(minor_at - fill_hw + 0.5));
        let n_fill_seg = max(1, i32(ceil(2.0 * fill_hw)));
        let hw_sq = half_width * half_width;
        var ndx_v: f32 = 0.0; var ndy_v: f32 = 0.0; var nlen_sq_v: f32 = 0.0;
        if (seg < num_segs - 1u) { if (seg_valid[seg + 1u] != 0u) {
          ndx_v = xs[base + seg + 2u] - x2; ndy_v = ys[base + seg + 2u] - y2;
          nlen_sq_v = ndx_v * ndx_v + ndy_v * ndy_v; } }
        var pdx_v: f32 = 0.0; var pdy_v: f32 = 0.0; var plen_sq_v: f32 = 0.0;
        if (seg > 0u) { if (seg_valid[seg - 1u] != 0u) {
          pdx_v = x1 - xs[base + seg - 1u]; pdy_v = y1 - ys[base + seg - 1u];
          plen_sq_v = pdx_v * pdx_v + pdy_v * pdy_v; } }
        for (var k = 0; k < n_fill_seg; k++) {
          let mi = mi_lo + k;
          if (mi < 0 || mi >= i32(MINOR_DIM)) { continue; }
          let minor_f = f32(mi) + 0.5;
          let dax_c = major_pos - x1; let day_c = minor_f - y1;
          let cross_c = dax_c * dy - day_c * dx;
          if (cross_c * cross_c <= hw_sq * seg_len_sq) {
            let dot_seg_c = dax_c * dx + day_c * dy;
            var clip_c = false;
            if (dot_seg_c > seg_len_sq) {
              if (nlen_sq_v > 0.01) {
                let dpx_n = major_pos - x2; let dpy_n = minor_f - y2;
                let cross_n = dpx_n * ndy_v - dpy_n * ndx_v;
                if (cross_n * cross_n > hw_sq * nlen_sq_v) { clip_c = true; }
                if (dpx_n * dpx_n + dpy_n * dpy_n > hw_sq * 9.0) { clip_c = true; }
              } else { clip_c = true; }
            }
            if (dot_seg_c < 0.0) {
              if (plen_sq_v > 0.01) {
                let cross_p = dax_c * pdy_v - day_c * pdx_v;
                if (cross_p * cross_p > hw_sq * plen_sq_v) { clip_c = true; }
                if (dax_c * dax_c + day_c * day_c > hw_sq * 9.0) { clip_c = true; }
              } else { clip_c = true; }
            }
            if (!clip_c) { atomicAdd(&density[${GI}], u32(scale)); }
          }
        }
      }
    }
  }
}`;
    var row = `
    if (den_eligible) {
      let maj_lo_i = i32(floor(maj_lo)); let maj_hi_i = i32(floor(maj_hi));
      if (i32(major_idx) >= maj_lo_i && i32(major_idx) <= maj_hi_i) {
        var skip_den = i32(major_idx) == last_den_major;
        if (!skip_den && seg < num_segs - 1u) {
          let next_dx = xs[base + seg + 2u] - xs[base + seg + 1u];
          if (abs(next_dx) >= 1.0 && i32(major_idx) == i32(floor(maj2))) { skip_den = true; }
        }
        if (!skip_den) {
          let sec_den = seg_len / max(abs(d_major), 0.01);
          let fill_hw = half_width * sec_den;
          let den_tc = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
          let minor_at = min1 + den_tc * d_minor;
          let mi_lo_d = i32(floor(minor_at - fill_hw + 0.5));
          let n_fill_seg = max(1, i32(ceil(2.0 * fill_hw)));
          let hw_sq = half_width * half_width;
          var ndx_v: f32 = 0.0; var ndy_v: f32 = 0.0; var nlen_sq_v: f32 = 0.0;
          if (seg < num_segs - 1u) { if (seg_valid[seg + 1u] != 0u) {
            ndx_v = xs[base + seg + 2u] - x2; ndy_v = ys[base + seg + 2u] - y2;
            nlen_sq_v = ndx_v * ndx_v + ndy_v * ndy_v; } }
          var pdx_v: f32 = 0.0; var pdy_v: f32 = 0.0; var plen_sq_v: f32 = 0.0;
          if (seg > 0u) { if (seg_valid[seg - 1u] != 0u) {
            pdx_v = x1 - xs[base + seg - 1u]; pdy_v = y1 - ys[base + seg - 1u];
            plen_sq_v = pdx_v * pdx_v + pdy_v * pdy_v; } }
          for (var k = 0; k < n_fill_seg; k++) {
            let mi = mi_lo_d + k;
            if (mi < 0 || mi >= i32(MINOR_DIM)) { continue; }
            let minor_f = f32(mi) + 0.5;
            let dax_r = minor_f - x1; let day_r = major_pos - y1;
            let cross_r = dax_r * dy - day_r * dx;
            if (cross_r * cross_r <= hw_sq * seg_len_sq) {
              let dot_seg_r = dax_r * dx + day_r * dy;
              var clip_r = false;
              if (dot_seg_r > seg_len_sq) {
                if (nlen_sq_v > 0.01) {
                  let dpx_n = minor_f - x2; let dpy_n = major_pos - y2;
                  let cross_n = dpx_n * ndy_v - dpy_n * ndx_v;
                  if (cross_n * cross_n > hw_sq * nlen_sq_v) { clip_r = true; }
                  if (dpx_n * dpx_n + dpy_n * dpy_n > hw_sq * 9.0) { clip_r = true; }
                } else { clip_r = true; }
              }
              if (dot_seg_r < 0.0) {
                if (plen_sq_v > 0.01) {
                  let cross_p = dax_r * pdy_v - day_r * pdx_v;
                  if (cross_p * cross_p > hw_sq * plen_sq_v) { clip_r = true; }
                  if (dax_r * dax_r + day_r * day_r > hw_sq * 9.0) { clip_r = true; }
                } else { clip_r = true; }
              }
              if (!clip_r) { atomicAdd(&density[${GI}], u32(scale)); }
            }
          }
        }
        last_den_major = i32(floor(maj2));
      } else { last_den_major = -999; }
    }
  }
}`;
    return head + (isCol ? col : row);
  }

  // count → per-count histogram bin (count = density/scale; integer ≤ numLines)
  var HIST_WGSL = `
struct P_ { width:u32, height:u32, num_lines:u32, scale:u32 }
@group(0) @binding(0) var<uniform> P: P_;
@group(0) @binding(1) var<storage, read> density: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P.width * P.height;
  if (gid.x >= n) { return; }
  let c = density[gid.x] / P.scale;
  if (c > 0u) { atomicAdd(&hist[min(c, P.num_lines)], 1u); }
}`;

  // exact eq-hist CDF over nonzero counts (one thread; num_lines+1 bins)
  var CDF_WGSL = `
struct P_ { width:u32, height:u32, num_lines:u32, scale:u32 }
@group(0) @binding(0) var<uniform> P: P_;
@group(0) @binding(1) var<storage, read> hist: array<u32>;
@group(0) @binding(2) var<storage, read_write> cdf: array<f32>;
@compute @workgroup_size(1)
fn main() {
  var total = 0u;
  for (var c = 1u; c <= P.num_lines; c++) { total += hist[c]; }
  let denom = f32(max(total, 1u));
  var acc = 0u; cdf[0] = 0.0;
  for (var c = 1u; c <= P.num_lines; c++) { acc += hist[c]; cdf[c] = f32(acc) / denom; }
}`;

  // colormap render pass → canvas: count → cdf[count] → LUT, alpha=count>0
  var COLORMAP_WGSL = `
struct P_ { width:u32, height:u32, num_lines:u32, scale:u32 }
@group(0) @binding(0) var<uniform> P: P_;
@group(0) @binding(1) var<storage, read> density: array<u32>;
@group(0) @binding(2) var<storage, read> cdf: array<f32>;
@group(0) @binding(3) var<storage, read> lut: array<vec4<f32>>;
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let x = u32(pos.x); let y = u32(pos.y);
  if (x >= P.width || y >= P.height) { return vec4<f32>(0.0); }
  let c = density[y * P.width + x] / P.scale;
  if (c == 0u) { return vec4<f32>(0.0); }
  let eq = cdf[min(c, P.num_lines)];
  let li = u32(clamp(eq, 0.0, 1.0) * 255.0);
  return vec4<f32>(lut[li].rgb, 1.0);   // premultiplied (alpha = 1 → opaque)
}`;

  var _gpu = null;
  async function initGPU() {
    if (_gpu) return _gpu;
    if (!(navigator.gpu)) return null;
    var adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    var device = await adapter.requestDevice();
    var fmt = navigator.gpu.getPreferredCanvasFormat();
    function comp(code) {
      var mod = device.createShaderModule({ code });
      if (mod.getCompilationInfo) mod.getCompilationInfo().then(function (i) {
        (i.messages || []).forEach(function (m) { if (m.type === 'error') console.warn('SpectraGL WGSL @' + m.lineNum + ': ' + m.message); });
      });
      return device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    }
    // Only the col/row raster pipelines are used (display is a 2D canvas via
    // putImageData — see render()). The histogram/cdf/colormap WGSL is kept in
    // the file for the GPU-colormap path but NOT compiled here, since each
    // pipeline compile adds to the first-render latency.
    // HDR display pipeline: a float density texture (lifted linear-P3 LUT, >1)
    // → OETF on an rgba16float/display-p3/extended canvas, so the density glows
    // into the headroom like the colormap tiles. Only used when cfg.hdr.
    var DISP = `
fn oetf(v:vec3f)->vec3f{let a=max(v,vec3f(0.0));return select(12.92*a,1.055*pow(a,vec3f(1.0/2.4))-0.055,a>vec3f(0.0031308));}
@group(0)@binding(0) var t:texture_2d<f32>;
@group(0)@binding(1) var s:sampler;
struct VO{ @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VO{
  var p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));
  var o:VO; o.pos=vec4f(p[i],0.,1.); o.uv=vec2f(p[i].x*0.5+0.5, 1.0-(p[i].y*0.5+0.5)); return o; }
@fragment fn fs(in:VO)->@location(0) vec4f{ let c=textureSample(t,s,in.uv); return vec4f(oetf(c.rgb), c.a); }`;
    var dispPipe = null, dispSmp = null;
    try {
      dispPipe = device.createRenderPipeline({ layout: 'auto',
        vertex: { module: device.createShaderModule({ code: DISP }), entryPoint: 'vs' },
        fragment: { module: device.createShaderModule({ code: DISP }), entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list' } });
      dispSmp = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
    } catch (e) { console.warn('SpectraGL HDR pipeline:', e && e.message || e); }
    _gpu = { device: device, fmt: fmt, dispPipe: dispPipe, dispSmp: dispSmp,
             colPipe: comp(rasterWGSL(true)), rowPipe: comp(rasterWGSL(false)) };
    return _gpu;
  }

  // Float colorize — same eq-hist CDF as colorize(), but emits Float32 RGBA from
  // a lifted linear-P3 LUT (values >1), for the HDR display pipeline.
  function colorizeF(counts, n, lut) {
    var maxC = 0, nz = 0, i;
    for (i = 0; i < n; i++) { var c = counts[i]; if (c > 0) { nz++; if (c > maxC) maxC = c; } }
    var out = new Float32Array(n * 4);
    if (nz === 0) return out;
    var nb = (maxC | 0) + 1, hist = new Float64Array(nb);
    for (i = 0; i < n; i++) { var ci = counts[i] | 0; if (ci > 0) hist[ci]++; }
    var acc = 0, cdf = new Float64Array(nb);
    for (i = 1; i < nb; i++) { acc += hist[i]; cdf[i] = acc / nz; }
    for (i = 0; i < n; i++) {
      var d = counts[i] | 0; if (d <= 0) continue;
      var idx = Math.min(255, Math.max(0, (cdf[d] * 255.0) | 0)) * 4, o = i * 4;
      out[o] = lut[idx]; out[o + 1] = lut[idx + 1]; out[o + 2] = lut[idx + 2]; out[o + 3] = 1;
    }
    return out;
  }

  function buildXY(cfg, W, H) {
    var N = cfg.numLines, P = cfg.numPoints, sx = W / cfg.plotW;
    var xf = new Float32Array(N * P), yf = new Float32Array(N * P);
    var ySpan = (cfg.yHi - cfg.yLo) || 1;
    for (var line = 0; line < N; line++) {
      var b = line * P;
      for (var i = 0; i < P; i++) {
        xf[b + i] = cfg.xPix[i] * sx;
        yf[b + i] = (cfg.yHi - cfg.yLines[b + i]) / ySpan * H;   // yHi → row 0 (top)
      }
    }
    var sv = new Uint32Array(Math.max(P - 1, 0));
    for (var k = 0; k < cfg.intervals.length; k++) {
      var lo = Math.max(0, cfg.intervals[k][0] | 0);
      var hi = Math.min((cfg.intervals[k][1] | 0) - 1, P - 1);
      for (var j = lo; j < hi; j++) sv[j] = 1;
    }
    return { xf: xf, yf: yf, sv: sv };
  }

  // CPU eq-hist CDF + LUT colorize (exact; matches numpy _eq_hist + cmap; alpha=count>0)
  function colorize(counts, n, lut) {
    var maxC = 0, nz = 0, i;
    for (i = 0; i < n; i++) { var c = counts[i]; if (c > 0) { nz++; if (c > maxC) maxC = c; } }
    var out = new Uint8ClampedArray(n * 4);
    if (nz === 0) return out;
    var nb = (maxC | 0) + 1, hist = new Float64Array(nb);
    for (i = 0; i < n; i++) { var ci = counts[i] | 0; if (ci > 0) hist[ci]++; }
    var acc = 0, cdf = new Float64Array(nb);
    for (i = 1; i < nb; i++) { acc += hist[i]; cdf[i] = acc / nz; }
    for (i = 0; i < n; i++) {
      var d = counts[i] | 0; if (d <= 0) continue;
      var idx = Math.min(255, Math.max(0, (cdf[d] * 255.0) | 0)) * 4, o = i * 4;
      out[o] = lut[idx]; out[o + 1] = lut[idx + 1]; out[o + 2] = lut[idx + 2]; out[o + 3] = 255;
    }
    return out;
  }

  // Render cfg into the canvas at on-screen device resolution. Async.
  // GPU rasterizes (V11 dedup) → density readback → CPU eq-hist + LUT →
  // putImageData to a 2D canvas. (We display via a 2D canvas, not a WebGPU
  // canvas: WebGPU-canvas presentation doesn't reliably composite inside a
  // Jupyter output's <foreignObject>; the 2D pixel buffer always does.)
  async function render(cv, cfg, cssW, cssH) {
    var g = await initGPU();
    if (!g) return false;
    var device = g.device;
    var dpr = self.devicePixelRatio || 1;
    cssW = cssW || cv.clientWidth || cfg.plotW;
    cssH = cssH || cv.clientHeight || 256;
    var W = Math.max(1, Math.round(cssW * dpr));
    var H = Math.max(1, Math.round(cssH * dpr));
    if (cv.width !== W) cv.width = W;
    if (cv.height !== H) cv.height = H;

    var d = buildXY(cfg, W, H);
    var USG = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    function sbuf(arr) { var b = device.createBuffer({ size: Math.max(4, arr.byteLength), usage: USG }); device.queue.writeBuffer(b, 0, arr); return b; }
    var xBuf = sbuf(d.xf), yBuf = sbuf(d.yf), svBuf = sbuf(d.sv);
    var densBuf = device.createBuffer({ size: W * H * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    var uA = new ArrayBuffer(32), uV = new DataView(uA);
    uV.setUint32(0, W, true); uV.setUint32(4, H, true);
    uV.setUint32(8, cfg.numLines, true); uV.setUint32(12, cfg.numPoints, true);
    uV.setFloat32(16, cfg.lineWidth * (W / cfg.plotW) / 2.0, true);
    uV.setFloat32(20, COVERAGE_SCALE, true);
    // 2D compute dispatch: WebGPU caps workgroups per dimension at 65535, so for
    // a large numLines×major (many cells × a wide retina canvas) we spread across
    // gid.y. ``disp_stride`` = invocations per gid.y row (= WGX*64); the shader
    // rebuilds the linear index as gid.y*disp_stride + gid.x. WGX is fixed across
    // the col + row passes (they share one uniform), sized to the larger pass.
    var _WGN = 64;
    var _maxNwg = Math.ceil(cfg.numLines * Math.max(W, H) / _WGN);
    var WGX = Math.min(65535, Math.max(1, _maxNwg));
    uV.setUint32(24, WGX * _WGN, true);          // disp_stride (offset 24 = old _p0)
    var uBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uBuf, 0, new Uint8Array(uA));

    function bg(pipe) { return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uBuf } }, { binding: 1, resource: { buffer: xBuf } },
      { binding: 2, resource: { buffer: yBuf } }, { binding: 3, resource: { buffer: densBuf } },
      { binding: 4, resource: { buffer: svBuf } }] }); }
    var enc = device.createCommandEncoder();
    enc.clearBuffer(densBuf, 0, W * H * 4);
    var WG = 64;
    var _disp = function (major) { var nwg = Math.ceil(cfg.numLines * major / WG);
      return [WGX, Math.max(1, Math.ceil(nwg / WGX))]; };
    var p = enc.beginComputePass();
    var dC = _disp(W); p.setPipeline(g.colPipe); p.setBindGroup(0, bg(g.colPipe)); p.dispatchWorkgroups(dC[0], dC[1], 1);
    var dR = _disp(H); p.setPipeline(g.rowPipe); p.setBindGroup(0, bg(g.rowPipe)); p.dispatchWorkgroups(dR[0], dR[1], 1);
    p.end();
    var readBuf = device.createBuffer({ size: W * H * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(densBuf, 0, readBuf, 0, W * H * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    var raw = new Uint32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    [xBuf, yBuf, svBuf, densBuf, uBuf, readBuf].forEach(function (b) { b.destroy(); });
    var counts = new Float32Array(W * H);
    for (var i = 0; i < counts.length; i++) counts[i] = raw[i] / COVERAGE_SCALE;
    if (cfg.hdr) {
      // HDR: float-colorize through the lifted LUT (>1) → rgba16float texture →
      // OETF on an extended display-p3 canvas (the density glows). The whole
      // canvas is WebGPU here (never 2D), so no context conflict.
      if (!g.dispPipe || typeof Float16Array === 'undefined' || !cfg.lut) return false;
      try {
        var fc = colorizeF(counts, W * H, cfg.lut);
        var half = new Float16Array(fc.length);
        for (var k = 0; k < fc.length; k++) half[k] = fc[k];
        var ctxg = cv.getContext('webgpu'); if (!ctxg) return false;
        try { ctxg.configure({ device: device, format: 'rgba16float', colorSpace: 'display-p3', alphaMode: 'premultiplied', toneMapping: { mode: 'extended' } }); }
        catch (e) { ctxg.configure({ device: device, format: 'rgba16float', colorSpace: 'display-p3', alphaMode: 'premultiplied' }); }
        var tex = device.createTexture({ size: [W, H], format: 'rgba16float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        var rb = W * 8, pbr = Math.ceil(rb / 256) * 256, sb = new Uint8Array(half.buffer), srcU8 = sb;
        if (pbr !== rb) { srcU8 = new Uint8Array(pbr * H); for (var y = 0; y < H; y++) srcU8.set(sb.subarray(y * rb, (y + 1) * rb), y * pbr); }
        device.queue.writeTexture({ texture: tex }, srcU8, { bytesPerRow: pbr, rowsPerImage: H }, { width: W, height: H });
        var bg2 = device.createBindGroup({ layout: g.dispPipe.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: g.dispSmp }] });
        var enc2 = device.createCommandEncoder();
        var rp = enc2.beginRenderPass({ colorAttachments: [{ view: ctxg.getCurrentTexture().createView(),
          loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
        rp.setPipeline(g.dispPipe); rp.setBindGroup(0, bg2); rp.draw(3); rp.end();
        device.queue.submit([enc2.finish()]);
        tex.destroy();
        cv.__sgState = { cfg: cfg, W: W, H: H };
        return true;
      } catch (e) { console.warn('SpectraGL HDR render:', e && e.message || e); return false; }
    }
    var img = new ImageData(colorize(counts, W * H, cfg.lut), W, H);
    cv.getContext('2d').putImageData(img, 0, 0);
    cv.__sgState = { cfg: cfg, W: W, H: H };
    return true;
  }

  // ── hover highlight (drawn on a separate 2D overlay canvas) ──────────────
  function nearestLine(st, mx, my) {
    var cfg = st.cfg, W = st.W, H = st.H, P = cfg.numPoints;
    var sx = W / cfg.plotW, ySpan = (cfg.yHi - cfg.yLo) || 1;
    function xc(i) { return cfg.xPix[i] * sx; }
    function yc(line, i) { return (cfg.yHi - cfg.yLines[line * P + i]) / ySpan * H; }
    var best = -1, bestD = Infinity;
    for (var iv = 0; iv < cfg.intervals.length; iv++) {
      var s0 = cfg.intervals[iv][0] | 0, s1 = cfg.intervals[iv][1] | 0;
      if (s1 - s0 < 2) continue;
      if (mx < xc(s0) - 5 || mx > xc(s1 - 1) + 5) continue;
      var lo = s0, hi = s1 - 2;
      while (lo < hi) { var mid = (lo + hi + 1) >> 1; if (xc(mid) <= mx) lo = mid; else hi = mid - 1; }
      var xA = xc(lo), xB = xc(lo + 1);
      var t = Math.min(1, Math.max(0, (mx - xA) / Math.max(xB - xA, 1e-9)));
      for (var line = 0; line < cfg.numLines; line++) {
        var yI = yc(line, lo) + t * (yc(line, lo + 1) - yc(line, lo));
        var dd = Math.abs(yI - my);
        if (dd < bestD) { bestD = dd; best = line; }
      }
    }
    // gate: only when the cursor is actually near a line (within ~half_width+3 px)
    var thr = (cfg.lineWidth * (W / cfg.plotW)) * 0.5 + 3.0;
    return bestD <= thr ? best : -1;
  }

  // overlayCv: the 2D overlay canvas; densityCv carries __sgState.
  function highlight(densityCv, overlayCv, mx, my, color) {
    var st = densityCv.__sgState; if (!st) return -1;
    var W = st.W, H = st.H;
    if (overlayCv.width !== W) overlayCv.width = W;
    if (overlayCv.height !== H) overlayCv.height = H;
    var ctx = overlayCv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    var line = nearestLine(st, mx, my);
    if (line < 0) return -1;
    var cfg = st.cfg, P = cfg.numPoints, sx = W / cfg.plotW, ySpan = (cfg.yHi - cfg.yLo) || 1;
    ctx.strokeStyle = color || 'rgba(255,64,64,0.95)';
    ctx.lineWidth = Math.max(3, (self.devicePixelRatio || 1) * 2.5);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    for (var iv = 0; iv < cfg.intervals.length; iv++) {
      var s0 = cfg.intervals[iv][0] | 0, s1 = cfg.intervals[iv][1] | 0;
      if (s1 - s0 < 2) continue;
      ctx.beginPath();
      ctx.moveTo(cfg.xPix[s0] * sx, (cfg.yHi - cfg.yLines[line * P + s0]) / ySpan * H);
      for (var j = s0 + 1; j < s1; j++)
        ctx.lineTo(cfg.xPix[j] * sx, (cfg.yHi - cfg.yLines[line * P + j]) / ySpan * H);
      ctx.stroke();
    }
    return line;
  }

  function clearHighlight(overlayCv) {
    if (overlayCv && overlayCv.width) overlayCv.getContext('2d').clearRect(0, 0, overlayCv.width, overlayCv.height);
  }

  // reverse link: stroke the line belonging to a known id (hover-a-cell -> light
  // its spectrum). Mirrors highlight() but selects by id, not nearest-to-cursor.
  // Part of the LinkedPanel interface shared with ScatterGL.highlightById.
  function highlightById(densityCv, overlayCv, id, color) {
    var st = densityCv.__sgState; if (!st) return -1;
    var cfg = st.cfg; if (!cfg.cellIds) return -1;
    if (!st._id2line) { st._id2line = {}; for (var i = 0; i < cfg.cellIds.length; i++) st._id2line[cfg.cellIds[i]] = i; }
    var line = st._id2line[id]; if (line == null) line = -1;
    var W = st.W, H = st.H;
    if (overlayCv.width !== W) overlayCv.width = W;
    if (overlayCv.height !== H) overlayCv.height = H;
    var ctx = overlayCv.getContext('2d'); ctx.clearRect(0, 0, W, H);
    if (line < 0) return -1;
    var P = cfg.numPoints, sx = W / cfg.plotW, ySpan = (cfg.yHi - cfg.yLo) || 1;
    ctx.strokeStyle = color || 'rgba(255,64,64,0.95)';
    ctx.lineWidth = Math.max(3, (self.devicePixelRatio || 1) * 2.5);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    for (var iv = 0; iv < cfg.intervals.length; iv++) {
      var s0 = cfg.intervals[iv][0] | 0, s1 = cfg.intervals[iv][1] | 0;
      if (s1 - s0 < 2) continue;
      ctx.beginPath();
      ctx.moveTo(cfg.xPix[s0] * sx, (cfg.yHi - cfg.yLines[line * P + s0]) / ySpan * H);
      for (var j = s0 + 1; j < s1; j++)
        ctx.lineTo(cfg.xPix[j] * sx, (cfg.yHi - cfg.yLines[line * P + j]) / ySpan * H);
      ctx.stroke();
    }
    return line;
  }

  return { decodeAttrs: decodeAttrs, render: render, highlight: highlight,
           highlightById: highlightById, clearHighlight: clearHighlight,
           nearestLine: nearestLine, COVERAGE_SCALE: COVERAGE_SCALE };
}));
