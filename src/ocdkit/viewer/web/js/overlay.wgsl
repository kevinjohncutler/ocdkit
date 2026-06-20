// Line-list overlay shader for the 3D viewer (trajectories, lineage, points-as-
// crosses, flow quiver, affinity edges). Per-vertex position is in VOXEL coords;
// the vertex stage maps voxel -> world via the SAME box as the volume, then
// applies viewProj, so overlays register exactly with the ray-marched volume.
// Loaded verbatim by the browser host AND tests/test_overlay_wgsl.py.

struct U {
  viewProj : mat4x4<f32>,
  boxMin   : vec4<f32>,
  boxMax   : vec4<f32>,
  dims     : vec4<f32>,   // NX, NY, NZ, _
};
@group(0) @binding(0) var<uniform> u : U;

struct VIn { @location(0) pos : vec3<f32>, @location(1) col : vec3<f32> };
struct VOut { @builtin(position) clip : vec4<f32>, @location(0) col : vec3<f32> };

@vertex
fn vs(in : VIn) -> VOut {
  let n = in.pos / u.dims.xyz;                       // voxel -> [0,1]
  let world = u.boxMin.xyz + n * (u.boxMax.xyz - u.boxMin.xyz);
  var o : VOut;
  o.clip = u.viewProj * vec4<f32>(world, 1.0);
  o.col = in.col;
  return o;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.col, 1.0);
}
