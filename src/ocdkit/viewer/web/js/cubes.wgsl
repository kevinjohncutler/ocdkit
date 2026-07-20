// Object-order voxel-cube renderer — MIP prototype (A/B against raymarch.wgsl).
//
// Instead of image-order ray-casting (one heavy per-pixel fragment LOOP that
// marches the 3D texture), this rasterizes each OCCUPIED voxel as a unit cube and
// MAX-blends its colormapped value. MIP is order-independent, so there is no depth
// sort and no depth test — every cube just contributes max(). This stresses the
// rasteriser + fixed-function blend (ROP) path rather than the compute/texture
// loop path, which is the whole point of the experiment (see the coil-whine
// investigation: whine tracks WHICH pipeline is stressed, not how much).
//
// Nearest-neighbour by construction: a voxel IS a cube, no interpolation.

struct U {
  viewProj : mat4x4<f32>,
  boxMin   : vec4<f32>,   // world-space AABB min (xyz)
  span     : vec4<f32>,   // world-space AABB size (xyz); gamma in .w
  dims     : vec4<f32>,   // NX, NY, NZ, _
};
@group(0) @binding(0) var<uniform> u : U;
// Same 256x1 intensity LUT the raymarch uses (grayscale = identity, HDR entries >1).
@group(0) @binding(1) var lutTex : texture_2d<f32>;

struct VOut { @builtin(position) pos : vec4<f32>, @location(0) val : f32 };

@vertex
fn vs(@location(0) corner : vec3<f32>,     // unit cube corner in [-0.5, 0.5]
      @location(1) inst : vec4<f32>)       // instance = voxel (i, j, k, value)
      -> VOut {
  let res = u.dims.xyz;
  let cell = u.span.xyz / res;                              // world size of one voxel
  let center = u.boxMin.xyz + (inst.xyz + vec3<f32>(0.5)) * cell;
  let world = center + corner * cell;                       // the voxel's cube in world space
  var o : VOut;
  o.pos = u.viewProj * vec4<f32>(world, 1.0);
  o.val = inst.w;
  return o;
}

// Colormap the scalar value via the 256-entry LUT (matches raymarch lutColor).
fn lutColor(v : f32) -> vec3<f32> {
  let f = clamp(v, 0.0, 1.0) * 255.0;
  let i0 = i32(floor(f));
  let i1 = min(i0 + 1, 255);
  let fr = f - f32(i0);
  let c0 = textureLoad(lutTex, vec2<i32>(i0, 0), 0).rgb;
  let c1 = textureLoad(lutTex, vec2<i32>(i1, 0), 0).rgb;
  return mix(c0, c1, fr);
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let g = pow(clamp(in.val, 0.0, 1.0), u.span.w);           // gamma (matches raymarch MIP)
  // MAX blend over all cubes -> per-pixel maximum intensity projection. Value in
  // .a so the max also tracks the scalar (grayscale exact; premultiplied for the
  // display-p3/extended canvas, so HDR LUT entries >1 still carry through).
  return vec4<f32>(lutColor(g), g);
}
