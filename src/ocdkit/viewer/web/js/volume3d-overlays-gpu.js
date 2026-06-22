/* volume3d-overlays-gpu.js — draws 3D overlays (trajectories, lineage, points,
 * flow quiver, affinity) over the volume on the SAME WebGPU canvas/context as
 * VolumeGPU, sharing its camera. Geometry comes from the Node-tested
 * volume3d-overlays.js builders; the line shader is the wgpu-native-tested
 * overlay.wgsl. Browser-only glue.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.OverlayLayer = api.OverlayLayer;
})(this, function () {
  "use strict";
  const VO = (typeof require !== "undefined") ? require("./volume3d-overlays.js")
                                              : (typeof window !== "undefined" ? window.VolumeOverlays : null);

  class OverlayLayer {
    static async create(device, format, decoded, opts = {}) {
      if (!VO) return null;
      const wgsl = await (await fetch(opts.shaderUrl || "js/overlay.wgsl")).text();
      const mod = device.createShaderModule({ code: wgsl });
      const bgl = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
      });
      const pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex: {
          module: mod, entryPoint: "vs",
          buffers: [
            { arrayStride: 12, attributes: [{ shaderLocation: 0, format: "float32x3", offset: 0 }] },
            { arrayStride: 12, attributes: [{ shaderLocation: 1, format: "float32x3", offset: 0 }] },
          ],
        },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "line-list" },
      });
      const self = new OverlayLayer();
      self.device = device;
      self.pipeline = pipeline;
      self.uniform = device.createBuffer({ size: 28 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      self.bindGroup = device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: self.uniform } }] });
      self.enabled = { axes: true, trajectories: true, points: false, flow: false, affinity: false };
      self._build(decoded);
      return self;
    }

    _gpu(geo) {
      if (!geo || !geo.count) return null;
      const p = this.device.createBuffer({ size: geo.positions.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(p, 0, geo.positions);
      const c = this.device.createBuffer({ size: geo.colors.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(c, 0, geo.colors);
      return { p, c, count: geo.count };
    }

    _build(d) {
      this.layers = {};
      if (d.meta) this.layers.axes = this._gpu(VO.axesTriad3D([d.meta.width, d.meta.height, d.meta.depth], 0.5));
      if (d.trajectories) {
        this.layers.traj = this._gpu(VO.trajPolylines3D(d.trajectories.tracks));
        this.layers.lineage = this._gpu(VO.lineageSegs3D(d.trajectories.tracks, d.trajectories.edges || []));
      }
      if (d.points) this.layers.points = this._gpu(VO.pointCrosses3D(d.points.data, d.points.count, 1.5));
      if (d.flowRaw) this.layers.flow = this._gpu(VO.flowQuiver3D(d.flowRaw, 6, 4));
      if (d.affinity) this.layers.affinity = this._gpu(VO.affinitySegs3D(d.affinity.spatial.data, d.affinity.spatial.shape, d.affinity.steps, 150000));
    }

    setEnabled(name, on) { this.enabled[name] = !!on; }
    setFlow(flowRaw) { if (flowRaw && VO) this.layers.flow = this._gpu(VO.flowQuiver3D(flowRaw, 6, 4)); }
    hasAny() { return this.layers && Object.values(this.layers).some(Boolean); }

    /** Record overlay draws into an existing render pass (no depth, drawn on top). */
    draw(pass, viewProj, boxMin, boxMax, dims) {
      const u = new Float32Array(28);
      u.set(viewProj, 0);
      u.set([boxMin[0], boxMin[1], boxMin[2], 0], 16);
      u.set([boxMax[0], boxMax[1], boxMax[2], 0], 20);
      u.set([dims[0], dims[1], dims[2], 0], 24);
      this.device.queue.writeBuffer(this.uniform, 0, u);
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      const draw = (L) => { if (!L) return; pass.setVertexBuffer(0, L.p); pass.setVertexBuffer(1, L.c); pass.draw(L.count); };
      if (this.enabled.axes) draw(this.layers.axes);
      if (this.enabled.affinity) draw(this.layers.affinity);
      if (this.enabled.flow) draw(this.layers.flow);
      if (this.enabled.trajectories) { draw(this.layers.traj); draw(this.layers.lineage); }
      if (this.enabled.points) draw(this.layers.points);
    }
  }

  return { OverlayLayer };
});
