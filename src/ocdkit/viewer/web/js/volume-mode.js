/* volume-mode.js — volume support in the main ocdkit viewer. No-op unless the
 * loaded image is a volume (CONFIG.isVolume).
 *
 * Views, switched from the "View" pane (left panel):
 *   - 2D slices (2.5D): the normal app.js canvas showing one slice; a slider
 *     along the bottom of the field of view + arrow keys scrub. An axis toggle
 *     (Z/Y/X) picks the slicing plane (switching axis reinitialises the 2D view
 *     for the new dimensions). Loaded/edited masks render and edit per slice,
 *     and edits persist back to the volume (and into the 3D render).
 *   - 3D volume: raw-WebGPU VolumeGPU render (needs WebGPU). H = home camera.
 */
(function () {
  "use strict";

  function init() {
    const cfg = (typeof window !== "undefined" && window.__VIEWER_CONFIG__) || {};
    if (!cfg.isVolume) return;                         // 2D image → nothing to do

    const panel = document.getElementById("viewModePanel");
    const sliceBar = document.getElementById("sliceBar");
    const slider = document.getElementById("sliceSlider");
    const sliceLabel = document.getElementById("sliceLabel");
    const vcanvas = document.getElementById("volumeViewer");
    const canvas2d = document.getElementById("canvas");
    const brush = document.getElementById("brushPreview");
    if (!panel || !sliceBar || !slider || !vcanvas || !canvas2d) return;
    const btn2d = panel.querySelector('[data-view="2d"]');
    const btn3d = panel.querySelector('[data-view="3d"]');
    const projRow = document.getElementById("projModeRow");
    const axisRow = document.getElementById("sliceAxisRow");
    const loadMasksBtn = document.getElementById("loadMasksButton");

    panel.hidden = false;
    sliceBar.hidden = false;

    // sinebow palette (vibrant; matches app.js's default colormap) — group g →
    // sinebow((g-1)/32). The 3D shader uses the identical formula on the same
    // group values, so 2D slices and the 3D volume render with the same colors.
    // Re-asserted after every mask set because app.js's async initialize()/
    // state-restore can otherwise reset the palette.
    const ncPalette = (function () {
      function sinebow(t) {
        const a = 2 * Math.PI * (t - Math.floor(t));
        return [
          Math.round((Math.sin(a) * 0.5 + 0.5) * 255),
          Math.round((Math.sin(a + 2 * Math.PI / 3) * 0.5 + 0.5) * 255),
          Math.round((Math.sin(a + 4 * Math.PI / 3) * 0.5 + 0.5) * 255),
        ];
      }
      // golden-ratio spread into sinebow so groups are well-separated for any count
      const PHI = 0.61803398875, pal = [];
      for (let i = 0; i < 32; i++) pal.push(sinebow(((i + 1) * PHI) % 1));
      return pal;
    })();
    function applyNColorPalette() {
      if (typeof window.__viewerSetNColorPalette === "function") {
        window.__viewerSetNColorPalette(ncPalette);
      }
    }
    applyNColorPalette();

    let mode = "2d";
    let vgpu = null;
    let loading = null;
    let curProj = 1;        // projection: 1=MIP, 2=mean, 0=additive
    let hasMask = !!cfg.hasVolumeMask;
    let mask3dStale = false; // 2D edits not yet reflected in the 3D bundle

    // axis: 0=Z, 1=Y, 2=X. volumeShape = [D, H, W].
    const AXES = ["Z", "Y", "X"];
    const vshape = Array.isArray(cfg.volumeShape) ? cfg.volumeShape
                 : [cfg.volumeDepth || 1, cfg.height || 0, cfg.width || 0];
    let curAxis = 0;
    function depthOf(a) { return vshape[a] || 1; }
    function sliceDims(a) {                              // {height, width} of a slice along axis a
      const rest = vshape.filter((_, i) => i !== a);
      return { height: rest[0], width: rest[1] };
    }

    let slice = (typeof cfg.currentSlice === "number") ? cfg.currentSlice : (depthOf(0) >> 1);
    slider.min = "0";
    slider.max = String(Math.max(0, depthOf(curAxis) - 1));
    slider.value = String(slice);
    function paintLabel() {
      sliceLabel.textContent = AXES[curAxis] + " " + (slice + 1) + " / " + depthOf(curAxis);
    }
    paintLabel();

    // overlay the volumetric mask for slice z (current axis) onto the 2D view:
    // ncolor groups (display) + identity labels (instance), both for this slice
    function _bufToU32(resp, buf) {
      const dt = resp.headers.get("X-Mask-Dtype") || "uint8";
      const raw = dt === "uint16" ? new Uint16Array(buf)
                : dt === "uint32" ? new Uint32Array(buf)
                : new Uint8Array(buf);
      const u = new Uint32Array(raw.length); u.set(raw); return u;
    }
    async function updateMaskSlice(z) {
      if (typeof window.__viewerSetMaskSlice !== "function") return;
      if (!hasMask) { window.__viewerSetMaskSlice(null, null); return; }
      try {
        const base = "/api/mask_slice/" + encodeURIComponent(cfg.sessionId) +
                     "?z=" + z + "&axis=" + curAxis + "&t=" + Date.now();
        const [gR, iR] = await Promise.all([fetch(base), fetch(base + "&kind=instance")]);
        if (!gR.ok || !iR.ok) return;
        const [gB, iB] = await Promise.all([gR.arrayBuffer(), iR.arrayBuffer()]);
        window.__viewerSetMaskSlice(_bufToU32(gR, gB), _bufToU32(iR, iB));
        applyNColorPalette();   // re-assert golden-HSV (beats any app.js reset)
      } catch (e) { /* leave current mask on transient error */ }
    }

    // reload the mask whenever the 2D image (re)loads (initial + axis switch)
    window.__onViewerImageReady = function () { if (hasMask) updateMaskSlice(slice); };

    // persist edits to the slice we're leaving back into the volume mask
    async function persistIfEdited() {
      if (!window.__viewerMaskEdited) return;
      window.__viewerMaskEdited = false;
      const mv = (typeof window.__viewerGetMask === "function") ? window.__viewerGetMask() : null;
      if (!mv) return;
      const u32 = new Uint32Array(mv);
      try {
        await fetch("/api/mask_slice/" + encodeURIComponent(cfg.sessionId) +
                    "?z=" + slice + "&axis=" + curAxis, {
          method: "POST",
          headers: { "content-type": "application/octet-stream", "X-Mask-Dtype": "uint32" },
          body: u32.buffer,
        });
        hasMask = true;
        mask3dStale = true;
      } catch (e) {
        window.__viewerMaskEdited = true;
      }
    }

    // scrub within the current axis (dimensions unchanged → cheap image swap)
    async function showSlice(z) {
      await persistIfEdited();
      z = Math.max(0, Math.min(depthOf(curAxis) - 1, z | 0));
      slice = z;
      slider.value = String(z);
      paintLabel();
      const url = "/api/volume_slice/" + encodeURIComponent(cfg.sessionId) +
                  "?z=" + z + "&axis=" + curAxis + "&t=" + Date.now();
      if (typeof window.__viewerSetSliceImage === "function") window.__viewerSetSliceImage(url);
      updateMaskSlice(z);
    }
    slider.addEventListener("input", () => showSlice(parseInt(slider.value, 10)));
    if (hasMask) updateMaskSlice(slice);                // initial mask overlay

    // switch slicing axis: dimensions change, so reinit the 2D view
    async function setAxis(a) {
      a = a | 0;
      if (a === curAxis) return;
      await persistIfEdited();
      curAxis = a;
      const dep = depthOf(a);
      slice = dep >> 1;
      slider.max = String(Math.max(0, dep - 1));
      slider.value = String(slice);
      paintLabel();
      if (axisRow) axisRow.querySelectorAll("[data-axis]").forEach((x) =>
        x.classList.toggle("is-active", parseInt(x.getAttribute("data-axis"), 10) === a));
      const dim = sliceDims(a);
      const url = "/api/volume_slice/" + encodeURIComponent(cfg.sessionId) +
                  "?z=" + slice + "&axis=" + a + "&t=" + Date.now();
      if (typeof window.__viewer_reinitialize === "function") {
        window.__viewer_reinitialize({
          width: dim.width, height: dim.height, imageUrl: url,
          imageName: (cfg.imageName || "volume") + " [" + AXES[a] + "]",
          isVolume: true, hasVolumeMask: hasMask, volumeDepth: dep,
          volumeShape: vshape, currentSlice: slice, sessionId: cfg.sessionId,
          savedViewerState: null, directoryEntries: [], directoryIndex: null,
          hasPrev: false, hasNext: false,
        });
        // mask reloads via __onViewerImageReady once the new image is decoded
      }
    }
    if (axisRow) {
      axisRow.querySelectorAll("[data-axis]").forEach((b) =>
        b.addEventListener("click", () => setAxis(parseInt(b.getAttribute("data-axis"), 10))));
    }

    // keys: H = home (reset 3D camera) in 3D; arrows scrub in 2D.
    window.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target;
      if (t && (t.isContentEditable ||
          (t.closest && t.closest('input, textarea, select, [contenteditable="true"]')))) {
        return;
      }
      if (mode === "3d") {
        if ((e.key === "h" || e.key === "H") && vgpu) { e.preventDefault(); vgpu.resetView(); }
        return;
      }
      let d = 0;
      if (e.key === "ArrowRight" || e.key === "ArrowUp") d = 1;
      else if (e.key === "ArrowLeft" || e.key === "ArrowDown") d = -1;
      if (!d) return;
      e.preventDefault();
      showSlice(slice + d);
    });

    // ── 3D volume view (WebGPU) ─────────────────────────────────────────────
    const hasWebGPU = !!(navigator.gpu && window.VolumeGPU && window.decodeBundle);
    if (!hasWebGPU) { btn3d.disabled = true; btn3d.title = "WebGPU not available in this browser"; }

    async function ensureVolume() {
      if (vgpu) return vgpu;
      if (loading) return loading;
      loading = (async () => {
        const r = await fetch("/api/volume_bundle/" + encodeURIComponent(cfg.sessionId));
        if (!r.ok) throw new Error("volume_bundle " + r.status);
        const decoded = await window.decodeBundle(await r.json());
        vgpu = await window.VolumeGPU.create(vcanvas, decoded, {
          shaderUrl: "/static/js/raymarch.wgsl",
          overlayShaderUrl: "/static/js/overlay.wgsl",
          mode: curProj,
        });
        vgpu.setOverlay("axes", false);
        window.__volumeGPU = vgpu;
        return vgpu;
      })();
      try {
        return await loading;
      } catch (e) {
        console.error("[volume-mode] failed to load 3D view", e);
        btn3d.disabled = true;
        btn3d.title = "3D view failed to load";
        loading = null;
        return null;
      }
    }

    async function setMode(next) {
      await persistIfEdited();
      mode = next;
      const is3d = next === "3d";
      btn2d.classList.toggle("is-active", !is3d);
      btn3d.classList.toggle("is-active", is3d);
      canvas2d.style.visibility = is3d ? "hidden" : "";
      if (brush) brush.style.visibility = is3d ? "hidden" : "";
      vcanvas.hidden = !is3d;
      sliceBar.hidden = is3d;
      if (axisRow) axisRow.hidden = is3d;               // axis picker is a 2D control
      if (projRow) projRow.hidden = !is3d;
      if (is3d) {
        if (mask3dStale && vgpu) {
          try { vgpu.destroy(); } catch (e) {}
          vgpu = null; window.__volumeGPU = null; loading = null;
        }
        mask3dStale = false;
        const g = await ensureVolume();
        if (g) g.render();
        else { mode = "2d"; setMode("2d"); }
      }
    }

    btn2d.addEventListener("click", () => setMode("2d"));
    btn3d.addEventListener("click", () => setMode("3d"));
    window.addEventListener("resize", () => { if (mode === "3d" && vgpu) vgpu.render(); });

    function setProj(p) {
      curProj = p | 0;
      if (projRow) projRow.querySelectorAll("[data-proj]").forEach((x) =>
        x.classList.toggle("is-active", parseInt(x.getAttribute("data-proj"), 10) === curProj));
      if (vgpu) vgpu.setMode(curProj);
    }
    if (projRow) {
      projRow.querySelectorAll("[data-proj]").forEach((b) =>
        b.addEventListener("click", () => setProj(parseInt(b.getAttribute("data-proj"), 10))));
    }

    async function loadMasks() {
      const r = await fetch("/api/select_mask_file", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ sessionId: cfg.sessionId }),
      });
      if (!r.ok) return false;
      hasMask = true;
      updateMaskSlice(slice);
      if (vgpu) { try { vgpu.destroy(); } catch (e) {} vgpu = null; window.__volumeGPU = null; }
      loading = null;
      if (mode === "3d") { const g = await ensureVolume(); if (g) g.render(); }
      return true;
    }
    if (loadMasksBtn) {
      loadMasksBtn.addEventListener("click", async () => {
        loadMasksBtn.disabled = true;
        try { await loadMasks(); } catch (e) { console.error("[volume-mode] load masks failed", e); }
        loadMasksBtn.disabled = false;
      });
    }

    // test/automation hook
    window.__volumeMode = {
      setMode, getMode: () => mode, gpu: () => vgpu,
      showSlice, getSlice: () => slice, setProj, getProj: () => curProj,
      setAxis, getAxis: () => curAxis,
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
