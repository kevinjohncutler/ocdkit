/* spectra_density_gl.js — live client-side spectra-density renderer (WebGPU).
 *
 * Renders the "datashaded" spectra panel in the browser at the canvas's true
 * device resolution (getBoundingClientRect × devicePixelRatio), so it stays
 * crisp at any on-screen size with no server rasterize + PNG/JXL round-trip.
 *
 * Pipeline — ALL on the GPU in a single submit (no readback, no CPU colorize,
 * no mapAsync await → paints in the same frame):
 *   1. col + row compute passes rasterize the lines into an atomic count buffer
 *      using the exact V11 half-open dedup + perpendicular cross-product
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

  // ── HDR overlay-line lift ────────────────────────────────────────────────
  // When HDR is on, overlay lines (hover highlight; reference spectra next)
  // drawn on a plain sRGB canvas read as DARK streaks against the glow, which
  // blooms past SDR white. We instead draw them on a WebGPU rgba16float/
  // display-p3 extended canvas with the colour lifted into the headroom by this
  // boost (≈ the lifted-LUT glow peak of ~2.3) so they glow like the data.
  var HL_HDR_BOOST = 2.5;
  // linear sRGB -> linear Display-P3 (P3_FROM_XYZ @ XYZ_FROM_SRGB)
  var _M_SRGB_P3 = [0.822462, 0.177538, 0.0,
                    0.033194, 0.966806, 0.0,
                    0.017083, 0.072397, 0.910520];
  function _srgbToLin(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function _parseRGBA(s) {
    var m = /rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+))?\s*\)/.exec(s || '');
    if (!m) return [1, 0.25, 0.25, 0.95];
    return [+m[1] / 255, +m[2] / 255, +m[3] / 255, m[4] != null ? +m[4] : 1];
  }
  // CSS rgba() string -> [r,g,b,a] linear Display-P3, lifted by `boost` (1 = SDR).
  function _colorToP3(str, boost) {
    var c = _parseRGBA(str), r = _srgbToLin(c[0]), g = _srgbToLin(c[1]), b = _srgbToLin(c[2]), M = _M_SRGB_P3;
    return [(M[0] * r + M[1] * g + M[2] * b) * boost,
            (M[3] * r + M[4] * g + M[5] * b) * boost,
            (M[6] * r + M[7] * g + M[8] * b) * boost, c[3]];
  }
  // Expand polylines (device px) into clip-space triangle-list verts
  // (x, y, vp, ta, L) for the capsule-SDF shader: vp = signed perpendicular
  // distance, ta = along-axis distance from the segment start, L = segment length.
  // Each segment's quad is grown by hw (= half-width + AA feather) in BOTH the
  // normal and axis directions so the round cap/feather is fully contained.
  function _segsToQuads(segs, W, H, halfW) {
    var AA = 1.0, hw = halfW + AA, v = [];
    function p(x, y, vp, ta, L) { v.push(x / W * 2 - 1, 1 - y / H * 2, vp, ta, L); }
    for (var s = 0; s < segs.length; s++) {
      var pl = segs[s];
      for (var i = 0; i + 1 < pl.length; i++) {
        var x0 = pl[i][0], y0 = pl[i][1], x1 = pl[i + 1][0], y1 = pl[i + 1][1];
        var dx = x1 - x0, dy = y1 - y0, len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1e-3) continue;
        var ux = dx / len, uy = dy / len, nx = -uy, ny = ux;
        var sx0 = x0 - ux * hw, sy0 = y0 - uy * hw, sx1 = x1 + ux * hw, sy1 = y1 + uy * hw;
        var ax = sx0 + nx * hw, ay = sy0 + ny * hw, bx = sx0 - nx * hw, by = sy0 - ny * hw;
        var cx = sx1 + nx * hw, cy = sy1 + ny * hw, dx2 = sx1 - nx * hw, dy2 = sy1 - ny * hw;
        p(ax, ay, hw, -hw, len); p(bx, by, -hw, -hw, len); p(cx, cy, hw, len + hw, len);
        p(bx, by, -hw, -hw, len); p(dx2, dy2, -hw, len + hw, len); p(cx, cy, hw, len + hw, len);
      }
    }
    return new Float32Array(v);
  }

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
      // AA mode (default 'solid'): RGB straight from the colormap LUT by density,
      // alpha modulated by the AA coverage (line keeps its hue, edges feather).
      // 'alpha' = lines.py parity (opaque band, floor-coloured halo); 'crisp' =
      // opaque, extent≥0.5 threshold (round joins, no feather).
      aa: cv.getAttribute('data-aa') || 'solid',
      cpuColorize: cv.getAttribute('data-cpu-colorize') === '1',   // force CPU colorize (validation/debug)
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
@group(0) @binding(5) var<storage, read_write> extent: array<atomic<u32>>;
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
      // === EXTENT (AA edge, lines.py parity): max over segments of the
      // anti-aliased coverage clamp(hw+0.5 - dist_to_segment). atomicMax (not
      // add) → overlaps/joints take the max, never sum, so no hotspots. ===
      // NOTE: the overlay hover line (HLINE / _segsToQuads, search "capsule SDF")
      // computes the SAME clamp(hw+0.5 - dist_to_segment) coverage independently,
      // in a plain stroke pipeline. Two parallel impls — keep them in sync.
      {
        let sec_theta = seg_len / max(abs(d_major), 0.01);
        let ext_r = i32(ceil((half_width + 0.5) * sec_theta + 0.5));
        let t_center = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
        let minor_center = min1 + t_center * d_minor;
        for (var dmi = -ext_r; dmi <= ext_r; dmi++) {
          let emi = i32(minor_center) + dmi;
          if (emi < 0 || emi >= i32(MINOR_DIM)) { continue; }
          let eminor_f = f32(emi) + 0.5;
          let eax = major_pos - x1; let eay = eminor_f - y1;
          let etp = clamp((eax * dx + eay * dy) / seg_len_sq, 0.0, 1.0);
          let eqx = eax - etp * dx; let eqy = eay - etp * dy;
          let edsq = eqx * eqx + eqy * eqy;
          let r_out = half_width + 0.5; let r_in = max(half_width - 0.5, 0.0);
          if (edsq < r_out * r_out) {                    // skip outside; skip sqrt for the cov=1 interior
            var ext_cov = 1.0;
            if (edsq > r_in * r_in) { ext_cov = clamp(r_out - sqrt(edsq), 0.0, 1.0); }
            let ext_int = u32(ext_cov * scale);
            if (ext_int > 0u) { atomicMax(&extent[u32(emi) * U.width + major_idx], ext_int); }
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
      // === EXTENT (AA edge, lines.py parity) — see col pass. row: minor = x ===
      {
        let sec_theta = seg_len / max(abs(d_major), 0.01);
        let ext_r = i32(ceil((half_width + 0.5) * sec_theta + 0.5));
        let t_center = clamp((major_pos - maj1) / d_major, 0.0, 1.0);
        let minor_center = min1 + t_center * d_minor;
        for (var dmi = -ext_r; dmi <= ext_r; dmi++) {
          let emi = i32(minor_center) + dmi;
          if (emi < 0 || emi >= i32(MINOR_DIM)) { continue; }
          let eminor_f = f32(emi) + 0.5;
          let eax = eminor_f - x1; let eay = major_pos - y1;
          let etp = clamp((eax * dx + eay * dy) / seg_len_sq, 0.0, 1.0);
          let eqx = eax - etp * dx; let eqy = eay - etp * dy;
          let edsq = eqx * eqx + eqy * eqy;
          let r_out = half_width + 0.5; let r_in = max(half_width - 0.5, 0.0);
          if (edsq < r_out * r_out) {                    // skip outside; skip sqrt for the cov=1 interior
            var ext_cov = 1.0;
            if (edsq > r_in * r_in) { ext_cov = clamp(r_out - sqrt(edsq), 0.0, 1.0); }
            let ext_int = u32(ext_cov * scale);
            if (ext_int > 0u) { atomicMax(&extent[major_idx * U.width + u32(emi)], ext_int); }
          }
        }
      }
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
  var acc = 0u; var minC = 0u;
  for (var c = 1u; c <= P.num_lines; c++) {
    acc += hist[c]; cdf[c] = f32(acc) / denom;
    if (minC == 0u && hist[c] > 0u) { minC = c; }
  }
  // cdf[0] repurposed as the eq-hist FLOOR (cdf at the minimum nonzero count), so
  // the colorize rescales (cdf[c]-floor)/(1-floor) → min count maps to 0 (lines.py).
  cdf[0] = select(0.0, cdf[minC], minC > 0u);
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

  // GPU colorize (compute) — the CPU colorize() ported to WGSL. Reads density +
  // extent + cdf (+ floor at cdf[0]) + LUT, writes packed RGBA8 to out_rgba so it
  // reads back as a 4-byte ImageData buffer (no CPU per-pixel loop). mode: 0=alpha
  // (lines.py), 1=solid, 2=crisp.
  var COLORIZE_WGSL = `
struct CP_ { width:u32, height:u32, num_lines:u32, scale:u32,
             mode:u32, core_floor:f32, edge_floor:f32, _p:u32 }
@group(0) @binding(0) var<uniform> CP: CP_;
@group(0) @binding(1) var<storage, read> density: array<u32>;
@group(0) @binding(2) var<storage, read> extent: array<u32>;
@group(0) @binding(3) var<storage, read> cdf: array<f32>;
@group(0) @binding(4) var<storage, read> lut: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_rgba: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= CP.width * CP.height) { return; }
  let d = density[i] / CP.scale;
  let e0 = f32(extent[i]) / f32(CP.scale);
  if (e0 <= 0.0 && d == 0u) { out_rgba[i] = 0u; return; }
  // RGB = LUT[ vOf(max(d,1)) ]: eq-hist rescaled by the floor, lifted by core_floor.
  let cdf_floor = cdf[0];
  let raw = cdf[min(max(d, 1u), CP.num_lines)];
  let e = (raw - cdf_floor) / max(1.0 - cdf_floor, 0.001);
  var v = CP.core_floor + max(0.0, e) * (1.0 - CP.core_floor);
  v = max(v, CP.edge_floor);
  let li = min(255u, u32(clamp(v, 0.0, 1.0) * 255.0));
  let col = lut[li];
  var alpha = 0.0;
  if (CP.mode == 1u) {                         // solid: alpha = AA coverage
    alpha = min(1.0, e0);
  } else if (CP.mode == 2u) {                  // crisp: opaque ≥0.5, else drop
    if (d > 0u || e0 >= 0.5) { alpha = 1.0; } else { out_rgba[i] = 0u; return; }
  } else {                                     // alpha (lines.py): opaque core, feather halo
    let core = (d > 0u) || (e0 >= 0.75);
    alpha = select(min(1.0, e0), 1.0, core);
  }
  let r8 = u32(round(clamp(col.r, 0.0, 1.0) * 255.0));
  let g8 = u32(round(clamp(col.g, 0.0, 1.0) * 255.0));
  let b8 = u32(round(clamp(col.b, 0.0, 1.0) * 255.0));
  let a8 = u32(round(clamp(alpha * col.a, 0.0, 1.0) * 255.0));
  out_rgba[i] = r8 | (g8 << 8u) | (b8 << 16u) | (a8 << 24u);   // little-endian → R,G,B,A bytes
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
@fragment fn fs(in:VO)->@location(0) vec4f{ let c=textureSample(t,s,in.uv); return vec4f(oetf(c.rgb)*c.a, c.a); }`;  // premultiplied (canvas alphaMode='premultiplied') so the coverage alpha feathers edges
    var dispPipe = null, dispSmp = null;
    try {
      dispPipe = device.createRenderPipeline({ layout: 'auto',
        vertex: { module: device.createShaderModule({ code: DISP }), entryPoint: 'vs' },
        fragment: { module: device.createShaderModule({ code: DISP }), entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list' } });
      dispSmp = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
    } catch (e) { console.warn('SpectraGL HDR pipeline:', e && e.message || e); }
    // HDR overlay-line pipeline: solid triangle-list quads, colour from a uniform
    // (lifted linear-P3), premultiplied src-over so overlapping joints don't
    // double-darken. Drives the hover highlight (and reference lines) on the
    // extended display-p3 overlay canvas so they glow with the data.
    // Analytic coverage AA via a capsule SDF — round caps + round joins, matching
    // the original 2D stroke's lineCap/lineJoin='round'. Each vertex carries
    // d = (vp, ta, L): vp = signed perpendicular distance from the centreline, ta
    // = distance along the segment axis from its start, L = segment length. `dt`
    // = how far past either end the fragment is, so `dist` is the true distance to
    // the segment (a capsule of radius = half-width), feathered over the outer
    // ~1px. u.p.x = half-width (px).
    //   This is the SAME analytic-coverage AA idea the density renderer uses (its
    // `extent` buffer from RASTER_TEMPLATE, feathered in colorize/colorizeF) — a
    // deliberately separate, parallel implementation: the density AA is a byproduct
    // of a compute rasterizer feeding eq-hist, this is a plain stroke pipeline, so
    // they don't share code. Keep the two feather rules conceptually in sync.
    var HLINE = `
struct U{ col:vec4f, p:vec4f };
@group(0)@binding(0) var<uniform> u:U;
struct VO{ @builtin(position) pos:vec4f, @location(0) d:vec3f };
@vertex fn vs(@location(0) xy:vec2f, @location(1) d:vec3f)->VO{ var o:VO; o.pos=vec4f(xy,0.,1.); o.d=d; return o; }
@fragment fn fs(in:VO)->@location(0) vec4f{
  let dt=max(max(-in.d.y, in.d.y - in.d.z), 0.0);      // past-the-end distance along the axis
  let dist=sqrt(dt*dt + in.d.x*in.d.x);                // distance to the segment (capsule)
  let cov=clamp(u.p.x + 0.5 - dist, 0.0, 1.0);
  let a=u.col.a*cov; return vec4f(u.col.rgb*a, a); }`;   // premultiplied
    var hlPipe = null;
    try {
      hlPipe = device.createRenderPipeline({ layout: 'auto',
        vertex: { module: device.createShaderModule({ code: HLINE }), entryPoint: 'vs',
          buffers: [{ arrayStride: 20, attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            { shaderLocation: 1, offset: 8, format: 'float32x3' }] }] },
        fragment: { module: device.createShaderModule({ code: HLINE }), entryPoint: 'fs',
          targets: [{ format: 'rgba16float', blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] },
        primitive: { topology: 'triangle-list' } });
    } catch (e) { console.warn('SpectraGL HL pipeline:', e && e.message || e); }
    _gpu = { device: device, fmt: fmt, dispPipe: dispPipe, dispSmp: dispSmp, hlPipe: hlPipe,
             colPipe: comp(rasterWGSL(true)), rowPipe: comp(rasterWGSL(false)),
             // GPU colorize path: eq-hist + LUT on the GPU → RGBA buffer (no CPU
             // per-pixel colorize, half the readback). Compiled lazily on first use.
             histPipe: null, cdfPipe: null, czPipe: null };
    try {
      _gpu.histPipe = comp(HIST_WGSL); _gpu.cdfPipe = comp(CDF_WGSL); _gpu.czPipe = comp(COLORIZE_WGSL);
    } catch (e) { console.warn('SpectraGL GPU-colorize pipelines:', e && e.message || e); }
    return _gpu;
  }

  // Float colorize — the EXACT same density→colour mapping as colorize() (eq-hist
  // floor rescale + core_floor visibility lift + edge_floor), only emitting float
  // linear-P3 instead of uint8 sRGB. Keeping the two in lockstep is what makes the
  // figure's HDR toggle reversible: rendered through the *non-lifted* spec_lut it
  // reproduces the plain SDR colormap (same black point, luma ≈ colorize); through
  // the *lifted* spec_lut_hdr the identical index mapping simply glows into the
  // headroom. Dropping the eq-hist floor (raw cdf[cc]) lands the lowest density
  // high in the LUT = "lifted/washed"; dropping core_floor crushes the lowest
  // density to LUT[0] = "too dark" — both break parity with SDR.
  //   core_floor is computed on the OETF-ENCODED (displayed) LUT luminance so it
  // lands at the same fractional index as colorize()'s uint8-sRGB basis (the LUT
  // here is linear-P3; the display pass applies the sRGB OETF). AA: carry the
  // ext-based coverage alpha (per mode) so HDR edges feather like the SDR path.
  function colorizeF(counts, ext, n, lut, mode) {
    mode = mode || 'alpha';
    var maxC = 0, nz = 0, i;
    for (i = 0; i < n; i++) { var c = counts[i]; if (c > 0) { nz++; if (c > maxC) maxC = c; } }
    var out = new Float32Array(n * 4);
    if (nz === 0) return out;
    function oetf1(a) { a = Math.max(0, a); return a <= 0.0031308 ? 12.92 * a : 1.055 * Math.pow(a, 1 / 2.4) - 0.055; }
    var lumMax = 1e-6, L;          // core_floor on DISPLAYED luminance → parity with colorize()'s uint8 basis
    for (i = 0; i < 256; i++) { L = 0.299 * oetf1(lut[i * 4]) + 0.587 * oetf1(lut[i * 4 + 1]) + 0.114 * oetf1(lut[i * 4 + 2]); if (L > lumMax) lumMax = L; }
    var coreFloorIdx = 0;
    for (i = 0; i < 256; i++) { L = 0.299 * oetf1(lut[i * 4]) + 0.587 * oetf1(lut[i * 4 + 1]) + 0.114 * oetf1(lut[i * 4 + 2]); if (L >= 0.2 * lumMax) { coreFloorIdx = i; break; } }
    var coreFloor = coreFloorIdx / 255, edgeFloor = 1 / 255;
    var nb = (maxC | 0) + 1, hist = new Float64Array(nb);
    for (i = 0; i < n; i++) { var ci = counts[i] | 0; if (ci > 0) hist[ci]++; }
    var acc = 0, cdf = new Float64Array(nb);
    for (i = 1; i < nb; i++) { acc += hist[i]; cdf[i] = acc / nz; }
    var minC = 1; while (minC < nb && hist[minC] === 0) minC++;   // lowest nonzero count
    var cdfFloor = (minC < nb) ? cdf[minC] : 0;                    // min density → 0 (lines.py eq-hist floor)
    for (i = 0; i < n; i++) {
      var e0 = ext ? ext[i] : (counts[i] > 0 ? 1 : 0);
      if (e0 <= 0) continue;
      var d = counts[i] | 0, alpha;
      if (mode === 'solid') {                 // alpha = AA coverage everywhere (smoothest)
        alpha = Math.min(1, e0);
      } else if (mode === 'crisp') {          // opaque, distance-thresholded (no feather)
        if (!(d > 0 || e0 >= 0.5)) continue;
        alpha = 1;
      } else {                                // 'alpha': opaque core, feathered outer edge
        alpha = ((d > 0) || (e0 >= 0.75)) ? 1 : Math.min(1, e0);
      }
      // colour value in [0,1] = eq-hist CDF rescaled by the floor, lifted by
      // core_floor, clamped to edge_floor — IDENTICAL to colorize().vOf().
      var cc = Math.min(Math.max(d, 1), nb - 1);
      var ev = (cdf[cc] - cdfFloor) / Math.max(1 - cdfFloor, 0.001);
      ev = coreFloor + Math.max(0, ev) * (1 - coreFloor);
      ev = Math.max(ev, edgeFloor);
      var idx = Math.min(255, Math.max(0, (ev * 255.0) | 0)) * 4, o = i * 4;
      out[o] = lut[idx]; out[o + 1] = lut[idx + 1]; out[o + 2] = lut[idx + 2];
      out[o + 3] = alpha * lut[idx + 3];      // straight coverage alpha (× LUT's own alpha)
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

  // CPU eq-hist CDF + LUT colorize with EXTENT anti-aliasing — parity with
  // lines.py _colorize_density_extent. Color from the density count (eq-hist);
  // a pixel is visible iff extent>0; core = count>0 || ext>=0.75 → opaque, count
  // color; edge → core-floor color, alpha = the extent coverage (the AA feather
  // that makes the line full instead of skinny). ``ext`` in [0,1] (extent/scale).
  function colorize(counts, ext, n, lut, mode) {
    mode = mode || 'alpha';
    var maxC = 0, nz = 0, i;
    for (i = 0; i < n; i++) { var c = counts[i]; if (c > 0) { nz++; if (c > maxC) maxC = c; } }
    var out = new Uint8ClampedArray(n * 4);
    // core floor: lowest LUT index with luminance ≥ 0.2·max (DensityLineRenderer
    // core_floor_lightness=0.2); edge floor: index 1. Keeps low/edge pixels visible.
    var lumMax = 1e-6, L;
    for (i = 0; i < 256; i++) { L = 0.299 * lut[i * 4] + 0.587 * lut[i * 4 + 1] + 0.114 * lut[i * 4 + 2]; if (L > lumMax) lumMax = L; }
    var coreFloorIdx = 0;
    for (i = 0; i < 256; i++) { L = 0.299 * lut[i * 4] + 0.587 * lut[i * 4 + 1] + 0.114 * lut[i * 4 + 2]; if (L >= 0.2 * lumMax) { coreFloorIdx = i; break; } }
    var coreFloor = coreFloorIdx / 255, edgeFloor = 1 / 255, nb = 0, cdf = null, cdfFloor = 0;
    if (nz > 0) {
      nb = (maxC | 0) + 1; var hist = new Float64Array(nb);
      for (i = 0; i < n; i++) { var ci = counts[i] | 0; if (ci > 0) hist[ci]++; }
      var acc = 0; cdf = new Float64Array(nb);
      for (i = 1; i < nb; i++) { acc += hist[i]; cdf[i] = acc / nz; }
      var minC = 1; while (minC < nb && hist[minC] === 0) minC++;     // lowest nonzero count
      cdfFloor = (minC < nb) ? cdf[minC] : 0;                          // lines.py: min count → 0 → floor
    }
    // density count → colour value in [0,1]: eq-hist CDF, rescaled so the minimum
    // count maps to 0 (so the colormap spans the actual density range, like
    // lines.py — a single line lands on the floor, a dense crossing on the top),
    // then lifted by core_floor so low/single density stays visible.
    function vOf(cc) {
      if (!cdf) return coreFloor;
      cc = Math.min(Math.max(cc, 1), nb - 1);
      var e = (cdf[cc] - cdfFloor) / Math.max(1 - cdfFloor, 0.001);
      e = coreFloor + Math.max(0, e) * (1 - coreFloor);
      return Math.max(e, edgeFloor);
    }
    for (i = 0; i < n; i++) {
      var e0 = ext ? ext[i] : (counts[i] > 0 ? 1 : 0);
      if (e0 <= 0) continue;
      var d = counts[i] | 0, v, alpha;
      // RGB always = LUT[ vOf(density) ]; modes differ only in how alpha (the AA)
      // is applied, never in hue/intensity.
      if (mode === 'solid') {                 // alpha = AA coverage everywhere (smoothest)
        v = vOf(Math.max(d, 1)); alpha = Math.min(1, e0);
      } else if (mode === 'crisp') {          // opaque, distance-thresholded (no feather)
        if (!(d > 0 || e0 >= 0.5)) continue;
        v = vOf(Math.max(d, 1)); alpha = 1;
      } else {                                // 'alpha' (lines.py): opaque core, feathered outer edge
        var core = (d > 0) || (e0 >= 0.75);
        v = vOf(Math.max(d, 1));
        alpha = core ? 1 : Math.min(1, e0);
      }
      var li = Math.min(255, Math.max(0, (v * 255) | 0)), o = i * 4, idx = li * 4;
      // AA *multiplies* the LUT's own alpha (don't override it) — so a colormap with
      // its own alpha ramp is respected; for opaque LUTs this == the AA alpha.
      out[o] = lut[idx]; out[o + 1] = lut[idx + 1]; out[o + 2] = lut[idx + 2];
      out[o + 3] = Math.round(alpha * lut[idx + 3]);
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
    var extBuf = device.createBuffer({ size: W * H * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
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
      { binding: 4, resource: { buffer: svBuf } }, { binding: 5, resource: { buffer: extBuf } }] }); }
    var enc = device.createCommandEncoder();
    enc.clearBuffer(densBuf, 0, W * H * 4);
    enc.clearBuffer(extBuf, 0, W * H * 4);
    var WG = 64;
    var _disp = function (major) { var nwg = Math.ceil(cfg.numLines * major / WG);
      return [WGX, Math.max(1, Math.ceil(nwg / WGX))]; };
    var p = enc.beginComputePass();
    var dC = _disp(W); p.setPipeline(g.colPipe); p.setBindGroup(0, bg(g.colPipe)); p.dispatchWorkgroups(dC[0], dC[1], 1);
    var dR = _disp(H); p.setPipeline(g.rowPipe); p.setBindGroup(0, bg(g.rowPipe)); p.dispatchWorkgroups(dR[0], dR[1], 1);
    p.end();
    var NPX = W * H;

    // ── GPU colorize path (default): eq-hist (hist+cdf) + LUT + AA alpha all on the
    //    GPU → packed RGBA buffer. No CPU per-pixel colorize, and the readback is
    //    4 bytes/px (RGBA) instead of 8 (density+extent). Falls back to CPU below. ──
    if (g.czPipe && g.histPipe && g.cdfPipe && cfg.lut && !cfg.hdr && !cfg.cpuColorize) {
      var nbin = cfg.numLines + 1;
      var histBuf = device.createBuffer({ size: nbin * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      var cdfBuf = device.createBuffer({ size: nbin * 4, usage: GPUBufferUsage.STORAGE });
      var lutF = new Float32Array(1024); for (var lf = 0; lf < 1024; lf++) lutF[lf] = cfg.lut[lf] / 255;
      var lutBuf = device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(lutBuf, 0, lutF);
      var outBuf = device.createBuffer({ size: NPX * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      var pA = new ArrayBuffer(16), pV2 = new DataView(pA);
      pV2.setUint32(0, W, true); pV2.setUint32(4, H, true); pV2.setUint32(8, cfg.numLines, true); pV2.setUint32(12, COVERAGE_SCALE, true);
      var pBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(pBuf, 0, new Uint8Array(pA));
      var lumMax = 1e-6, Lk;                                  // core_floor: LUT idx with luminance ≥ 0.2·max
      for (var k = 0; k < 256; k++) { Lk = 0.299 * cfg.lut[k * 4] + 0.587 * cfg.lut[k * 4 + 1] + 0.114 * cfg.lut[k * 4 + 2]; if (Lk > lumMax) lumMax = Lk; }
      var cfi = 0; for (var k2 = 0; k2 < 256; k2++) { Lk = 0.299 * cfg.lut[k2 * 4] + 0.587 * cfg.lut[k2 * 4 + 1] + 0.114 * cfg.lut[k2 * 4 + 2]; if (Lk >= 0.2 * lumMax) { cfi = k2; break; } }
      var modeI = cfg.aa === 'solid' ? 1 : (cfg.aa === 'crisp' ? 2 : 0);
      var cA = new ArrayBuffer(32), cV = new DataView(cA);
      cV.setUint32(0, W, true); cV.setUint32(4, H, true); cV.setUint32(8, cfg.numLines, true); cV.setUint32(12, COVERAGE_SCALE, true);
      cV.setUint32(16, modeI, true); cV.setFloat32(20, cfi / 255, true); cV.setFloat32(24, 1 / 255, true);
      var cBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(cBuf, 0, new Uint8Array(cA));
      var mkbg = function (pipe, ents) { return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: ents }); };
      enc.clearBuffer(histBuf, 0, nbin * 4);
      var hp = enc.beginComputePass();
      hp.setPipeline(g.histPipe); hp.setBindGroup(0, mkbg(g.histPipe, [
        { binding: 0, resource: { buffer: pBuf } }, { binding: 1, resource: { buffer: densBuf } }, { binding: 2, resource: { buffer: histBuf } }]));
      hp.dispatchWorkgroups(Math.ceil(NPX / 64)); hp.end();
      var cp2 = enc.beginComputePass();
      cp2.setPipeline(g.cdfPipe); cp2.setBindGroup(0, mkbg(g.cdfPipe, [
        { binding: 0, resource: { buffer: pBuf } }, { binding: 1, resource: { buffer: histBuf } }, { binding: 2, resource: { buffer: cdfBuf } }]));
      cp2.dispatchWorkgroups(1); cp2.end();
      var zp = enc.beginComputePass();
      zp.setPipeline(g.czPipe); zp.setBindGroup(0, mkbg(g.czPipe, [
        { binding: 0, resource: { buffer: cBuf } }, { binding: 1, resource: { buffer: densBuf } }, { binding: 2, resource: { buffer: extBuf } },
        { binding: 3, resource: { buffer: cdfBuf } }, { binding: 4, resource: { buffer: lutBuf } }, { binding: 5, resource: { buffer: outBuf } }]));
      zp.dispatchWorkgroups(Math.ceil(NPX / 64)); zp.end();
      var readRgba = device.createBuffer({ size: NPX * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.copyBufferToBuffer(outBuf, 0, readRgba, 0, NPX * 4);
      device.queue.submit([enc.finish()]);
      await readRgba.mapAsync(GPUMapMode.READ);
      var rgbaOut = new Uint8ClampedArray(readRgba.getMappedRange().slice(0));
      readRgba.unmap();
      [xBuf, yBuf, svBuf, densBuf, extBuf, uBuf, histBuf, cdfBuf, lutBuf, outBuf, pBuf, cBuf, readRgba].forEach(function (b) { b.destroy(); });
      cv.getContext('2d').putImageData(new ImageData(rgbaOut, W, H), 0, 0);
      cv.__sgState = { cfg: cfg, W: W, H: H };
      return true;
    }

    // ── CPU colorize / HDR fallback: read density+extent back, colorize on CPU ──
    var readBoth = device.createBuffer({ size: NPX * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(densBuf, 0, readBoth, 0, NPX * 4);
    enc.copyBufferToBuffer(extBuf, 0, readBoth, NPX * 4, NPX * 4);
    device.queue.submit([enc.finish()]);
    await readBoth.mapAsync(GPUMapMode.READ);
    var both = new Uint32Array(readBoth.getMappedRange().slice(0));
    readBoth.unmap();
    [xBuf, yBuf, svBuf, densBuf, extBuf, uBuf, readBoth].forEach(function (b) { b.destroy(); });
    var counts = new Float32Array(NPX), ext = new Float32Array(NPX), invS = 1 / COVERAGE_SCALE;
    for (var i = 0; i < NPX; i++) { counts[i] = both[i] * invS; ext[i] = both[NPX + i] * invS; }
    if (cfg.hdr) {
      // HDR: float-colorize through the lifted LUT (>1) → rgba16float texture →
      // OETF on an extended display-p3 canvas (the density glows). The whole
      // canvas is WebGPU here (never 2D), so no context conflict.
      if (!g.dispPipe || typeof Float16Array === 'undefined' || !cfg.lut) return false;
      try {
        var fc = colorizeF(counts, ext, W * H, cfg.lut, cfg.aa);
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
    var img = new ImageData(colorize(counts, ext, W * H, cfg.lut, cfg.aa), W, H);
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

  // build the polyline(s) for `line` in device px (one per interval).
  function _lineSegs(cfg, line, W, H) {
    var P = cfg.numPoints, sx = W / cfg.plotW, ySpan = (cfg.yHi - cfg.yLo) || 1, segs = [];
    for (var iv = 0; iv < cfg.intervals.length; iv++) {
      var s0 = cfg.intervals[iv][0] | 0, s1 = cfg.intervals[iv][1] | 0;
      if (s1 - s0 < 2) continue;
      var pl = [];
      for (var j = s0; j < s1; j++) pl.push([cfg.xPix[j] * sx, (cfg.yHi - cfg.yLines[line * P + j]) / ySpan * H]);
      segs.push(pl);
    }
    return segs;
  }

  // Draw overlay polylines on the WebGPU rgba16float/display-p3 extended overlay
  // canvas (HDR mode), colour lifted into the headroom so the line glows like the
  // data instead of reading as a dark streak. Configures the canvas as WebGPU on
  // first use; once GPU, the canvas stays GPU (its context type is fixed).
  function _drawHLGpu(overlayCv, segs, W, H, colorP3, halfW) {
    var g = _gpu; if (!g || !g.hlPipe) return false;
    var dev = g.device;
    if (overlayCv.width !== W) overlayCv.width = W;
    if (overlayCv.height !== H) overlayCv.height = H;
    var ctx = overlayCv.__hlGpuCtx;
    if (!ctx) {
      ctx = overlayCv.getContext('webgpu'); if (!ctx) return false;
      try { ctx.configure({ device: dev, format: 'rgba16float', colorSpace: 'display-p3', alphaMode: 'premultiplied', toneMapping: { mode: 'extended' } }); }
      catch (e) { try { ctx.configure({ device: dev, format: 'rgba16float', colorSpace: 'display-p3', alphaMode: 'premultiplied' }); } catch (e2) { return false; } }
      overlayCv.__hlGpuCtx = ctx; overlayCv.__hlMode = 'gpu';
    }
    var verts = _segsToQuads(segs, W, H, halfW);
    var enc = dev.createCommandEncoder();
    var rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    if (verts.length) {
      var vbuf = dev.createBuffer({ size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(vbuf, 0, verts);
      var ubuf = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(ubuf, 0, new Float32Array([colorP3[0], colorP3[1], colorP3[2], colorP3[3], halfW, 0, 0, 0]));
      var bg = dev.createBindGroup({ layout: g.hlPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }] });
      rp.setPipeline(g.hlPipe); rp.setBindGroup(0, bg); rp.setVertexBuffer(0, vbuf); rp.draw(verts.length / 5);
      rp.end(); dev.queue.submit([enc.finish()]); vbuf.destroy(); ubuf.destroy();
    } else { rp.end(); dev.queue.submit([enc.finish()]); }
    return true;
  }

  // Stroke `line` onto the overlay. HDR figure (cfg.hdr) → WebGPU extended canvas
  // with the colour lifted (skipped when the HDR toggle is OFF, i.e. the SDR LUT
  // is active, so it matches the non-lifted data). Non-HDR → plain 2D stroke.
  function _strokeOverlay(overlayCv, cfg, line, W, H, color) {
    var lw = Math.max(3, (self.devicePixelRatio || 1) * 2.5), segs = _lineSegs(cfg, line, W, H);
    if (cfg.hdr) {
      if (!_gpu || !_gpu.hlPipe) return;   // GPU not ready: skip (don't taint the canvas with a 2D context)
      var boost = (cfg.lut && cfg.lut === cfg.lutHdr) ? HL_HDR_BOOST : 1.0;   // no lift when toggled to SDR
      _drawHLGpu(overlayCv, segs, W, H, _colorToP3(color || 'rgba(255,64,64,0.95)', boost), lw / 2);
      return;
    }
    if (overlayCv.width !== W) overlayCv.width = W;
    if (overlayCv.height !== H) overlayCv.height = H;
    var ctx = overlayCv.getContext('2d'); overlayCv.__hlMode = '2d'; ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = color || 'rgba(255,64,64,0.95)';
    ctx.lineWidth = lw; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    for (var s = 0; s < segs.length; s++) {
      var pl = segs[s]; ctx.beginPath(); ctx.moveTo(pl[0][0], pl[0][1]);
      for (var k = 1; k < pl.length; k++) ctx.lineTo(pl[k][0], pl[k][1]);
      ctx.stroke();
    }
  }

  // overlayCv: the overlay canvas (2D in SDR, WebGPU-extended in HDR); densityCv carries __sgState.
  function highlight(densityCv, overlayCv, mx, my, color) {
    var st = densityCv.__sgState; if (!st) return -1;
    var line = nearestLine(st, mx, my);
    if (line < 0) { clearHighlight(overlayCv); return -1; }
    _strokeOverlay(overlayCv, st.cfg, line, st.W, st.H, color);
    return line;
  }

  function clearHighlight(overlayCv) {
    if (!overlayCv) return;
    if (overlayCv.__hlMode === 'gpu' && overlayCv.__hlGpuCtx && _gpu) {   // clear the WebGPU overlay via an empty render pass
      var dev = _gpu.device, enc = dev.createCommandEncoder();
      enc.beginRenderPass({ colorAttachments: [{ view: overlayCv.__hlGpuCtx.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] }).end();
      dev.queue.submit([enc.finish()]); return;
    }
    // Only touch a 2D context if the overlay is ALREADY in 2D mode — calling
    // getContext('2d') on a fresh (or HDR/WebGPU) overlay would permanently lock
    // it to 2D and break the GPU highlight. Unset mode = nothing drawn = no-op.
    if (overlayCv.__hlMode === '2d' && overlayCv.width) overlayCv.getContext('2d').clearRect(0, 0, overlayCv.width, overlayCv.height);
  }

  // reverse link: stroke the line belonging to a known id (hover-a-cell -> light
  // its spectrum). Mirrors highlight() but selects by id, not nearest-to-cursor.
  // Part of the LinkedPanel interface shared with ScatterGL.highlightById.
  function highlightById(densityCv, overlayCv, id, color) {
    var st = densityCv.__sgState; if (!st) return -1;
    var cfg = st.cfg; if (!cfg.cellIds) return -1;
    if (!st._id2line) { st._id2line = {}; for (var i = 0; i < cfg.cellIds.length; i++) st._id2line[cfg.cellIds[i]] = i; }
    var line = st._id2line[id]; if (line == null) line = -1;
    if (line < 0) { clearHighlight(overlayCv); return -1; }
    _strokeOverlay(overlayCv, cfg, line, st.W, st.H, color);
    return line;
  }

  return { decodeAttrs: decodeAttrs, render: render, highlight: highlight,
           highlightById: highlightById, clearHighlight: clearHighlight,
           nearestLine: nearestLine, COVERAGE_SCALE: COVERAGE_SCALE };
}));
