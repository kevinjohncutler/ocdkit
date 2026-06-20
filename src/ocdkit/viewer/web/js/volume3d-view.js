/* volume3d-view.js — self-contained 2.5D slice viewer (canvas2d).
 *
 * Renders one slice of a volume bundle (from omnipose _volume3d / POST /api/volume)
 * with toggleable overlays, plus slice navigation. Pure rendering + DOM glue;
 * all slice/overlay GEOMETRY comes from the Node-tested volume3d.js. Browser-only
 * (DecompressionStream for gzip, canvas2d) — deliberately NOT touching the main
 * app.js, so the existing 2D viewer is unaffected.
 *
 * The P2 WebGPU volume layer mounts on the same page as a sibling canvas.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") { window.VolumeView = api.VolumeView; window.decodeBundle = api.decodeBundle; }
})(this, function () {
  "use strict";
  const V = (typeof require !== "undefined") ? require("./volume3d.js")
                                             : (typeof window !== "undefined" ? window.Volume3D : globalThis.Volume3D);

  async function inflateGzip(bytes) {
    const ds = new DecompressionStream("gzip");
    const buf = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
    return new Uint8Array(buf);
  }

  async function decodeField(field) {
    if (!field) return null;
    let bytes = V.b64ToBytes(field.b64);
    if (field.gzip) bytes = await inflateGzip(bytes);
    return V.bytesToTyped(bytes, field.dtype, field.shape);
  }

  /** Decode a full bundle JSON (browser, async) -> usable typed structures. */
  async function decodeBundle(b) {
    const out = { meta: b.meta, steps: b.steps || [], trajectories: b.trajectories || null };
    out.mask = b.mask && !b.mask.deferred ? await decodeField(b.mask) : null;
    out.image = b.image && !b.image.deferred ? await decodeField(b.image) : null;
    if (b.flow) {
      out.flowRgb = await decodeField(b.flow.rgbSlices);
    }
    if (b.distance) out.distRgb = await decodeField(b.distance.rgbSlices);
    if (b.affinity && b.affinity.spatial && !b.affinity.spatial.deferred) {
      out.affinity = { steps: b.affinity.steps, spatial: await decodeField(b.affinity.spatial) };
    }
    if (b.points && b.points.encoded) {
      out.points = { data: (await decodeField({ dtype: "float32", shape: [b.points.count, 3], gzip: b.points.gzip, b64: b.points.encoded })).data, count: b.points.count };
    }
    return out;
  }

  function hsv2rgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    const m = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
    return [m[0] * 255 | 0, m[1] * 255 | 0, m[2] * 255 | 0];
  }
  function labelColor(v) {
    if (!v) return [0, 0, 0];
    const h = (v * 0.61803398875) % 1;
    return hsv2rgb(h, 0.65, 1.0);
  }

  class VolumeView {
    constructor(canvas, decoded) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.d = decoded;
      const m = decoded.meta;
      this.D = m.depth; this.H = m.height; this.W = m.width;
      this.z = (this.D / 2) | 0;
      this.layer = decoded.image ? "image" : (decoded.flowRgb ? "flow" : "mask");
      this.overlays = { affinity: false, points: !!decoded.points, trajectories: !!decoded.trajectories };
      this._off = (typeof document !== "undefined") ? document.createElement("canvas") : null;
      if (this._off) { this._off.width = this.W; this._off.height = this.H; this._offctx = this._off.getContext("2d"); }
      // global intensity range for grayscale image layer
      if (decoded.image) {
        let lo = Infinity, hi = -Infinity; const a = decoded.image.data;
        for (let i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
        this._imgLo = lo; this._imgHi = hi > lo ? hi : lo + 1;
      }
    }

    static async load(canvas, bundleJson) { return new VolumeView(canvas, await decodeBundle(bundleJson)); }

    setSlice(z) { this.z = Math.max(0, Math.min(this.D - 1, z | 0)); this.render(); }
    stepSlice(dz) { this.setSlice(this.z + dz); }
    setLayer(name) { this.layer = name; this.render(); }
    setOverlay(name, on) { this.overlays[name] = !!on; this.render(); }

    _baseImageData() {
      const { W, H, z } = this; const img = new ImageData(W, H); const px = img.data;
      const d = this.d;
      if (this.layer === "image" && d.image) {
        const sl = V.volumeSlice(d.image.data, d.image.shape, z);
        const lo = this._imgLo, sc = 255 / (this._imgHi - lo);
        for (let i = 0; i < sl.length; i++) { const g = Math.max(0, Math.min(255, (sl[i] - lo) * sc)) | 0; px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = g; px[i * 4 + 3] = 255; }
      } else if (this.layer === "flow" && d.flowRgb) {
        const sl = V.rgbVolumeSlice(d.flowRgb.data, d.flowRgb.shape, z);
        for (let i = 0, j = 0; i < W * H; i++) { px[i * 4] = sl[j++]; px[i * 4 + 1] = sl[j++]; px[i * 4 + 2] = sl[j++]; px[i * 4 + 3] = 255; }
      } else if (this.layer === "distance" && d.distRgb) {
        const sl = V.rgbVolumeSlice(d.distRgb.data, d.distRgb.shape, z);
        for (let i = 0, j = 0; i < W * H; i++) { px[i * 4] = sl[j++]; px[i * 4 + 1] = sl[j++]; px[i * 4 + 2] = sl[j++]; px[i * 4 + 3] = 255; }
      } else if (this.layer === "mask" && d.mask) {
        const sl = V.volumeSlice(d.mask.data, d.mask.shape, z);
        for (let i = 0; i < sl.length; i++) { const c = labelColor(sl[i]); px[i * 4] = c[0]; px[i * 4 + 1] = c[1]; px[i * 4 + 2] = c[2]; px[i * 4 + 3] = 255; }
      } else {
        for (let i = 0; i < W * H; i++) px[i * 4 + 3] = 255; // black
      }
      return img;
    }

    render() {
      const { ctx, canvas, W, H, z } = this;
      if (!ctx) return;
      this._offctx.putImageData(this._baseImageData(), 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(this._off, 0, 0, canvas.width, canvas.height);
      const sx = canvas.width / W, sy = canvas.height / H;

      if (this.overlays.affinity && this.d.affinity) {
        const segs = V.affinitySliceSegments(this.d.affinity.spatial.data, this.d.affinity.spatial.shape, this.d.affinity.steps, z);
        ctx.strokeStyle = "rgba(80,220,255,0.55)"; ctx.lineWidth = 1; ctx.beginPath();
        for (let i = 0; i < segs.length; i += 4) { ctx.moveTo(segs[i] * sx, segs[i + 1] * sy); ctx.lineTo(segs[i + 2] * sx, segs[i + 3] * sy); }
        ctx.stroke();
      }
      if (this.overlays.trajectories && this.d.trajectories) {
        const tracks = this.d.trajectories.tracks;
        const pr = V.projectTracks(tracks, z);
        for (const t of pr) {
          const c = labelColor(t.label);
          ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},0.9)`; ctx.lineWidth = 1.5; ctx.beginPath();
          for (let i = 0; i < t.points.length; i += 2) { const X = t.points[i] * sx, Y = t.points[i + 1] * sy; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
          ctx.stroke();
          if (t.head) { ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.beginPath(); ctx.arc(t.head[0] * sx, t.head[1] * sy, 3, 0, 6.2832); ctx.fill(); }
        }
        const lin = V.lineageSegments(tracks, this.d.trajectories.edges, z);
        ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath();
        for (let i = 0; i < lin.length; i += 4) { ctx.moveTo(lin[i] * sx, lin[i + 1] * sy); ctx.lineTo(lin[i + 2] * sx, lin[i + 3] * sy); }
        ctx.stroke(); ctx.setLineDash([]);
      }
      if (this.overlays.points && this.d.points) {
        const pts = V.pointsNearSlice(this.d.points.data, this.d.points.count, z, 0.5);
        ctx.fillStyle = "rgba(255,80,200,0.9)";
        for (let i = 0; i < pts.length; i += 2) { ctx.beginPath(); ctx.arc(pts[i] * sx, pts[i + 1] * sy, 2, 0, 6.2832); ctx.fill(); }
      }
    }

    /** Wire DOM controls (slider/select/checkboxes/wheel/keys) to this view. */
    bindControls(els) {
      const self = this;
      if (els.slider) { els.slider.min = 0; els.slider.max = this.D - 1; els.slider.value = this.z; els.slider.addEventListener("input", () => self.setSlice(+els.slider.value)); }
      if (els.label) this._sliceLabel = els.label;
      if (els.layer) els.layer.addEventListener("change", () => self.setLayer(els.layer.value));
      for (const k of ["affinity", "points", "trajectories"]) {
        if (els[k]) { els[k].checked = self.overlays[k]; els[k].addEventListener("change", () => self.setOverlay(k, els[k].checked)); }
      }
      this.canvas.addEventListener("wheel", (e) => { e.preventDefault(); self.stepSlice(e.deltaY > 0 ? 1 : -1); if (els.slider) els.slider.value = self.z; }, { passive: false });
      window.addEventListener("keydown", (e) => {
        if (e.key === "ArrowUp" || e.key === "ArrowRight") { self.stepSlice(1); }
        else if (e.key === "ArrowDown" || e.key === "ArrowLeft") { self.stepSlice(-1); }
        else return;
        if (els.slider) els.slider.value = self.z;
      });
      const origRender = this.render.bind(this);
      this.render = function () { origRender(); if (self._sliceLabel) self._sliceLabel.textContent = `${self.d.meta.axes[0]}: ${self.z + 1} / ${self.D}`; };
    }
  }

  return { VolumeView, decodeBundle };
});
