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

    // Labels are the source of truth; each label is colored by sinebow(fract(
    // group·φ)) of its volume ncolor group, so 2D and 3D match (the shader uses
    // the same formula on the group) and adjacent cells differ. golden-ratio
    // spread → well-separated for any group count.
    const PHI = 0.61803398875;
    function sinebow(t) {
      const a = 2 * Math.PI * (t - Math.floor(t));
      return [Math.round((Math.sin(a) * 0.5 + 0.5) * 255),
              Math.round((Math.sin(a + 2 * Math.PI / 3) * 0.5 + 0.5) * 255),
              Math.round((Math.sin(a + 4 * Math.PI / 3) * 0.5 + 0.5) * 255)];
    }
    let labelGroups = [0];   // label → ncolor group (index by label)
    // Per-GROUP palette: maskValues holds group IDs, so the ncolor panel shows
    // only the ~N group colors. group g → palette[g-1] = sinebow(fract(g·φ)),
    // matching the 3D shader.
    function applyNColorPalette() {
      if (typeof window.__viewerSetNColorPalette !== "function") return;
      let maxG = 1;
      for (let i = 1; i < labelGroups.length; i++) if (labelGroups[i] > maxG) maxG = labelGroups[i];
      const pal = [];
      for (let g = 1; g <= maxG; g++) pal.push(sinebow((g * PHI) % 1));
      window.__viewerSetNColorPalette(pal);
    }
    async function fetchNColorMap() {
      try {
        const r = await fetch("/api/ncolor_map/" + encodeURIComponent(cfg.sessionId) + "?t=" + Date.now());
        if (!r.ok) return;
        labelGroups = (await r.json()).groups || [0];
        applyNColorPalette();
      } catch (e) { /* keep prior palette */ }
    }

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
      if (!hasMask) { window.__viewerSetMaskSlice(null); return; }
      if (labelGroups.length <= 1) await fetchNColorMap();   // palette ready → no first-render flash
      try {
        const r = await fetch("/api/mask_slice/" + encodeURIComponent(cfg.sessionId) +
                              "?z=" + z + "&axis=" + curAxis + "&kind=group&t=" + Date.now());
        if (!r.ok) return;
        applyNColorPalette();   // set the per-group palette BEFORE the mask render
        window.__viewerSetMaskSlice(_bufToU32(r, await r.arrayBuffer()));   // group IDs
      } catch (e) { /* leave current mask on transient error */ }
    }

    // reload the mask whenever the 2D image (re)loads (initial + axis switch)
    window.__onViewerImageReady = function () { if (hasMask) updateMaskSlice(slice); };

    // Edits persist via __onViewerStroke (which writes LABELS to the volume).
    // maskValues holds group IDs, so we must NOT post it back as labels.
    async function persistIfEdited() { window.__viewerMaskEdited = false; }

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
    if (hasMask) { fetchNColorMap(); updateMaskSlice(slice); }   // initial palette + overlay

    // ── 2D / 3D brush ────────────────────────────────────────────────────────
    // 2D brush = paint the current slice only. 3D brush = extrude the stroke into
    // a true ball across neighbouring slices (server paint_sphere, radius R).
    let brushDim = 2;
    // ncolor mode: a stroke paints the SELECTED COLOUR (group) and merges into
    // adjacent cells of the same colour — it is never a separate new cell that
    // gets recoloured. maskValues already hold group IDs, so the value app.js
    // just painted (`after[0]`) IS the chosen group; 0 = erase.
    window.__onViewerStroke = function (indices, after) {
      if (!indices || !indices.length) return;
      const dim = sliceDims(curAxis);
      const fp = new Uint8Array((dim.width | 0) * (dim.height | 0));
      for (let i = 0; i < indices.length; i++) fp[indices[i]] = 1;
      const group = (after && after.length) ? (after[0] | 0) : 0;   // chosen colour; 0 = erase
      const radius = (brushDim === 3 && window.__viewerBrushRadius) ? window.__viewerBrushRadius() : 0;
      fetch("/api/paint_sphere/" + encodeURIComponent(cfg.sessionId) +
            "?z=" + slice + "&axis=" + curAxis + "&radius=" + radius + "&group=" + group, {
        method: "POST", headers: { "content-type": "application/octet-stream" }, body: fp.buffer,
      }).then(async (r) => {
        if (!r.ok) return;
        try { const j = await r.json(); srvCanUndo = !!j.canUndo; srvCanRedo = !!j.canRedo; } catch (e) {}
        hasMask = true; mask3dStale = true; window.__viewerMaskEdited = false;
        await fetchNColorMap(); await updateMaskSlice(slice);   // reflect the edit (groups + palette)
        if (window.__viewerUpdateHistory) window.__viewerUpdateHistory();
      }).catch(() => {});
    };

    // ----- server-owned undo / redo (the mask volume is the source of truth) -----
    let srvCanUndo = false, srvCanRedo = false;
    window.__viewerVolumeActive = () => !!(cfg.isVolume && hasMask);
    window.__viewerVolumeCanUndo = () => srvCanUndo;
    window.__viewerVolumeCanRedo = () => srvCanRedo;
    async function serverHistory(op) {
      try {
        const r = await fetch("/api/" + op + "/" + encodeURIComponent(cfg.sessionId), { method: "POST" });
        if (!r.ok) return;
        const j = await r.json();
        srvCanUndo = !!j.canUndo; srvCanRedo = !!j.canRedo;
        if (j.changed) {
          await fetchNColorMap();
          await updateMaskSlice(slice);   // refresh the 2D buffer
          await refresh3DLabels();        // AND the 3D render in place (was missing → 3D undo looked dead)
        }
        if (window.__viewerUpdateHistory) window.__viewerUpdateHistory();
      } catch (e) { /* leave display as-is on transient error */ }
    }
    window.__viewerVolumeUndo = () => serverHistory("undo");
    window.__viewerVolumeRedo = () => serverHistory("redo");

    // ----- 3D colour picker + 3D fill (whole-cell merge / delete) -----
    let pickedLabel = 0;     // the cell identity captured by the picker → merge target for fill
    window.__viewerVolumePick = function (wx, wy) {
      if (!cfg.isVolume || !hasMask) return false;
      const y = Math.floor(wy), x = Math.floor(wx);
      fetch("/api/label_at/" + encodeURIComponent(cfg.sessionId) +
            "?z=" + slice + "&axis=" + curAxis + "&y=" + y + "&x=" + x)
        .then((r) => r.json()).then((j) => {
          pickedLabel = j.label | 0;                       // remember the cell to merge into
          if (j.group > 0 && window.__viewerSetCurrentColor) window.__viewerSetCurrentColor(j.group);
        }).catch(() => {});
      return true;
    };
    window.__viewerVolumeFill = function (wx, wy) {
      if (!cfg.isVolume || !hasMask) return false;
      const y = Math.floor(wy), x = Math.floor(wx);
      // erase OR the zero-marker (currentLabel 0) → delete; a picked cell →
      // identity-merge; otherwise fill with the current colour.
      const group = (window.__viewerCurrentLabel ? window.__viewerCurrentLabel() : 0) | 0;
      const erasing = !!(window.__viewerEraseActive && window.__viewerEraseActive()) || group <= 0;
      const q = erasing ? "&erase=1"
              : (pickedLabel ? "&target=" + pickedLabel : "&group=" + group);
      fetch("/api/fill_cell/" + encodeURIComponent(cfg.sessionId) +
            "?z=" + slice + "&axis=" + curAxis + "&y=" + y + "&x=" + x + q, { method: "POST" })
        .then(async (r) => {
          if (!r.ok) return;
          const j = await r.json();
          srvCanUndo = !!j.canUndo; srvCanRedo = !!j.canRedo;
          mask3dStale = true;
          await fetchNColorMap(); await updateMaskSlice(slice);
          if (window.__viewerUpdateHistory) window.__viewerUpdateHistory();
        }).catch(() => {});
      return true;
    };

    // Re-upload the 3D label texture in place from the server (no destroy/recreate,
    // so no flash to a blank 2D frame mid-edit).
    async function refresh3DLabels() {
      if (mode !== "3d" || !vgpu || typeof vgpu.updateLabels !== "function") { mask3dStale = true; return; }
      try {
        const r = await fetch("/api/ncolor_volume/" + encodeURIComponent(cfg.sessionId));
        if (!r.ok) { mask3dStale = true; return; }
        vgpu.updateLabels(new Uint8Array(await r.arrayBuffer()));
        mask3dStale = false;            // 3D texture now reflects the server volume
      } catch (e) { mask3dStale = true; }
    }

    // ----- 3D-view picker / fill: click the cell under the cursor on the render --
    // The 3D canvas computes a world pick-ray; the server marches the label volume
    // (same math as the shader) to find the hit cell, then we pick or fill it.
    window.__viewerVolume3DPick = async function (ray, toolMode) {
      if (!cfg.isVolume || !hasMask || !ray) return;
      try {
        const pr = await fetch("/api/pick_ray/" + encodeURIComponent(cfg.sessionId),
          { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(ray) });
        if (!pr.ok) return;
        const hit = await pr.json();
        const label = hit.label | 0;
        if (toolMode === "picker") {
          // empty space → pick the ZERO marker (currentLabel 0) so the next stroke/
          // fill deletes; a cell → pick its colour.
          pickedLabel = label;
          if (window.__viewerSetCurrentColor) window.__viewerSetCurrentColor(label ? (hit.group | 0) : 0);
          return;
        }
        if (!label) return;                               // fill/erase on empty → nothing to fill
        // fill: erase OR the zero-marker (currentLabel 0) → delete; a picked cell →
        // identity-merge; otherwise the current colour. Fills the CONNECTED COMPONENT
        // under the cursor (one ray call → server picks voxel + fills contiguous region).
        const group = (window.__viewerCurrentLabel ? window.__viewerCurrentLabel() : 0) | 0;
        const erasing = !!(window.__viewerEraseActive && window.__viewerEraseActive()) || group <= 0;
        const q = erasing ? "&erase=1"
                : (pickedLabel ? "&target=" + pickedLabel : "&group=" + group);
        const fr = await fetch("/api/fill_ray/" + encodeURIComponent(cfg.sessionId) + "?" + q.slice(1),
                               { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(ray) });
        if (!fr.ok) return;
        const j = await fr.json();
        srvCanUndo = !!j.canUndo; srvCanRedo = !!j.canRedo;
        await fetchNColorMap();
        if (mode === "2d") { await updateMaskSlice(slice); mask3dStale = true; }
        else await refresh3DLabels();                     // in-place texture update, no flash
        if (window.__viewerUpdateHistory) window.__viewerUpdateHistory();
      } catch (e) { /* transient */ }
    };

    function setBrushDim(d) {
      brushDim = d | 0;
      if (brushDimRow) brushDimRow.querySelectorAll("[data-brush]").forEach((x) =>
        x.classList.toggle("is-active", parseInt(x.getAttribute("data-brush"), 10) === brushDim));
    }
    const brushDimRow = document.getElementById("brushDimRow");
    if (brushDimRow) {
      brushDimRow.querySelectorAll("[data-brush]").forEach((b) =>
        b.addEventListener("click", () => setBrushDim(parseInt(b.getAttribute("data-brush"), 10))));
    }

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
      setStyleButtonsFor3D(is3d);
      if (is3d) {
        // NOTE: do NOT change the global maskDisplayMode here — it drives the 2D
        // outline rendering and is persisted. The 3D render only cares about
        // show-vs-hide labels (applyLabelVisibilityToGpu), independent of the
        // 2D outline/fill style. (Forcing 'solid' here wiped 2D outlines.)
        if (mask3dStale && vgpu) {
          try { vgpu.destroy(); } catch (e) {}
          vgpu = null; window.__volumeGPU = null; loading = null;
        }
        mask3dStale = false;
        const g = await ensureVolume();
        if (g) { g.render(); applyLabelVisibilityToGpu(); }
        else { mode = "2d"; setMode("2d"); }
      }
    }

    // Label style in 3D: only "solid" (filled labels) and "hidden" (null) apply —
    // outline-based modes don't translate to the volume render, so grey them out.
    function setStyleButtonsFor3D(is3d) {
      document.querySelectorAll('#maskStyleToggle [data-mask-style]').forEach((b) => {
        const m = b.getAttribute("data-mask-style");
        const off = is3d && (m === "outlined" || m === "outline");
        b.disabled = off;
        b.classList.toggle("is-disabled", off);
        b.style.opacity = off ? "0.3" : "";
        b.style.pointerEvents = off ? "none" : "";
      });
    }
    function applyLabelVisibilityToGpu() {
      if (!vgpu || typeof vgpu.setShowLabels !== "function") return;
      const m = window.__viewerMaskDisplayMode ? window.__viewerMaskDisplayMode() : "solid";
      vgpu.setShowLabels(m === "hidden" ? 0 : 1);     // null selector turns labels off in 3D
    }
    // app.js calls this whenever the label-style slider changes.
    window.__viewerVolumeOnMaskStyle = function () { applyLabelVisibilityToGpu(); };

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
      setBrushDim, getBrushDim: () => brushDim,
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
