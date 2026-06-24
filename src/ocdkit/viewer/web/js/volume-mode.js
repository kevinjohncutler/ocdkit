/* volume-mode.js — volume support in the main ocdkit viewer. No-op unless the
 * loaded image is a volume (CONFIG.isVolume).
 *
 * Two views, switched from the "View" pane (left panel):
 *   - 2D slices (2.5D): the normal app.js canvas showing one z-slice; a slider
 *     along the bottom of the field of view + arrow keys scrub through z.
 *   - 3D volume: raw-WebGPU VolumeGPU render (needs WebGPU).
 */
(function () {
  "use strict";

  function init() {
    const cfg = (typeof window !== "undefined" && window.__VIEWER_CONFIG__) || {};
    if (!cfg.isVolume) return;                         // 2D image → nothing to do
    const depth = cfg.volumeDepth || 1;

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

    panel.hidden = false;
    sliceBar.hidden = false;

    let mode = "2d";
    let vgpu = null;
    let loading = null;

    // ── 2.5D slice scrubbing ────────────────────────────────────────────────
    let slice = (typeof cfg.currentSlice === "number") ? cfg.currentSlice : (depth >> 1);
    slider.min = "0";
    slider.max = String(Math.max(0, depth - 1));
    slider.value = String(slice);
    function paintLabel() { sliceLabel.textContent = "z " + (slice + 1) + " / " + depth; }
    paintLabel();

    function showSlice(z) {
      z = Math.max(0, Math.min(depth - 1, z | 0));
      slice = z;
      slider.value = String(z);
      paintLabel();
      // server tracks volume_slice; cache-bust so the <img> reloads each step
      const url = "/api/volume_slice/" + encodeURIComponent(cfg.sessionId) +
                  "?z=" + z + "&t=" + Date.now();
      if (typeof window.__viewerSetSliceImage === "function") {
        window.__viewerSetSliceImage(url);
      }
    }
    slider.addEventListener("input", () => showSlice(parseInt(slider.value, 10)));

    // arrow keys scrub z (2D mode only; ignore while typing / with modifiers)
    window.addEventListener("keydown", (e) => {
      if (mode !== "2d") return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target;
      if (t && (t.isContentEditable ||
          (t.closest && t.closest('input, textarea, select, [contenteditable="true"]')))) {
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
        });
        window.__volumeGPU = vgpu;                      // for the 3D control panel (later)
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
      mode = next;
      const is3d = next === "3d";
      btn2d.classList.toggle("is-active", !is3d);
      btn3d.classList.toggle("is-active", is3d);
      canvas2d.style.visibility = is3d ? "hidden" : "";
      if (brush) brush.style.visibility = is3d ? "hidden" : "";
      vcanvas.hidden = !is3d;
      sliceBar.hidden = is3d;                           // scrubber is a 2D control
      if (is3d) {
        const g = await ensureVolume();
        if (g) g.render();                              // size to the visible canvas + draw
        else { mode = "2d"; setMode("2d"); }            // load failed → fall back
      }
    }

    btn2d.addEventListener("click", () => setMode("2d"));
    btn3d.addEventListener("click", () => setMode("3d"));
    window.addEventListener("resize", () => { if (mode === "3d" && vgpu) vgpu.render(); });

    // test/automation hook
    window.__volumeMode = {
      setMode, getMode: () => mode, gpu: () => vgpu,
      showSlice, getSlice: () => slice,
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
