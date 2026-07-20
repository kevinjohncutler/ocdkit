// Trivial blit: copy the compute ray-march's storage texture (bound as a sampled
// texture) 1:1 onto the canvas. A fullscreen triangle; each fragment reads the
// same pixel via textureLoad (nearest, no filtering). Handles any canvas format
// (rgba16float HDR or 8-bit sRGB) since the fragment just writes the value.

@group(0) @binding(0) var srcTex : texture_2d<f32>;

struct VOut { @builtin(position) pos : vec4<f32> };

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  return o;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  return textureLoad(srcTex, vec2<i32>(i32(in.pos.x), i32(in.pos.y)), 0);
}
