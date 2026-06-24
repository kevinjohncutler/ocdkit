/* volume-mode.js — embeds the raw-WebGPU 3D volume view as a switchable mode in
 * the main ocdkit viewer. Activates only when the loaded image is a volume
 * (CONFIG.isVolume) and WebGPU is available; otherwise it's a no-op so 2D images
 * are completely unaffected.
 *
 * The 3D view mounts VolumeGPU (raymarch.wgsl) on #volumeViewer, fed by the
 * intensity-only bundle from GET /api/volume_bundle/{session}. Masks/overlays
 * arrive later once 3D segmentation runs (Phase D).
 */
(function () {
  "use strict";

  function init() {
    const cfg = (typeof window !== "undefined" && window.__VIEWER_CONFIG__) || {};
    if (!cfg.isVolume) return;                       // 2D image → nothing to do

    const bar = document.getElementById("viewModeBar");
    const vcanvas = document.getElementById("volumeViewer");
    const canvas2d = document.getElementById("canvas");
    const brush = document.getElementById("brushPreview");
    if (!bar || !vcanvas || !canvas2d) return;
    const btn2d = bar.querySelector('[data-view="2d"]');
    const btn3d = bar.querySelector('[data-view="3d"]');

    bar.hidden = false;                              // a volume is loaded → show the switch
    const hasWebGPU = !!(navigator.gpu && window.VolumeGPU && window.decodeBundle);
    if (!hasWebGPU) {
      btn3d.disabled = true;
      btn3d.title = "WebGPU not available in this browser";
      return;                                        // 2.5D/2D still work; 3D unavailable
    }

    let vgpu = null;
    let loading = null;          // in-flight create promise (dedupe)
    let mode = "2d";

    async function ensureVolume() {
      if (vgpu) return vgpu;
      if (loading) return loading;
      loading = (async () => {
        const resp = await fetch("/api/volume_bundle/" + encodeURIComponent(cfg.sessionId));
        if (!resp.ok) throw new Error("volume_bundle " + resp.status);
        const decoded = await window.decodeBundle(await resp.json());
        vgpu = await window.VolumeGPU.create(vcanvas, decoded, {
          shaderUrl: "/static/js/raymarch.wgsl",
          overlayShaderUrl: "/static/js/overlay.wgsl",
        });
        window.__volumeGPU = vgpu;                   // expose for the 3D control panel (later)
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
      if (is3d) {
        const g = await ensureVolume();
        if (g) g.render();                           // size to the now-visible canvas + draw
        else { mode = "2d"; setMode("2d"); }         // load failed → fall back
      }
    }

    btn2d.addEventListener("click", () => setMode("2d"));
    btn3d.addEventListener("click", () => setMode("3d"));
    window.addEventListener("resize", () => { if (mode === "3d" && vgpu) vgpu.render(); });

    // test/automation hook
    window.__volumeMode = { setMode, getMode: () => mode, gpu: () => vgpu };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
