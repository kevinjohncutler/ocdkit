"""Phase A — headless WGSL parity for the LabelGPU port.

Renders the candidate WGSL label fragment shader via wgpu-native (real Metal
device, no browser) at 1:1 (1 screen px = 1 mask texel) and asserts it matches a
NumPy reference of LabelGL's GLSL math — dual-id decode, palette, screen-space
8-neighbour outline, outline HDR boost, hover highlight, transparent compositing.
Target is rgba16float so HDR (>1.0) outline pixels are captured exactly.
"""
import numpy as np
import wgpu
import wgpu.utils

W = H = 8

# ── WGSL: port of label_gl.js VERT+FRAG. dpdx/dpdy hoisted to uniform control
#    flow (top of fs); all mask/palette samples use textureSampleLevel (explicit
#    LOD — implicit-derivative samples are illegal in the non-uniform branches).
WGSL = """
struct U {
  matrix: mat3x3<f32>,
  highlightColor: vec3<f32>,
  outlineColor: vec3<f32>,
  maskOpacity: f32, maskVisible: f32, outlinesVisible: f32, maskStyle: f32,
  imageVisible: f32, colorOffset: f32, paletteSize: f32, usePalette: f32,
  highlightLabel: f32, highlightAlpha: f32, highlightBoost: f32, outlineHdrBoost: f32,
  useOutlineColor: f32, baseHeadroom: f32, baseLinear: f32, pad0: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var baseTex: texture_2d<f32>;
@group(0) @binding(3) var maskTex: texture_2d<f32>;
@group(0) @binding(4) var palTex: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@location(0) a_pos: vec2<f32>, @location(1) a_uv: vec2<f32>) -> VOut {
  let p = u.matrix * vec3<f32>(a_pos, 1.0);
  var o: VOut;
  o.pos = vec4<f32>(p.xy, 0.0, 1.0);
  o.uv = a_uv;
  return o;
}

fn eotf(c: f32) -> f32 { return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045); }
fn oetf(c: f32) -> f32 { let x = max(c, 0.0); return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, 12.92 * x, x <= 0.0031308); }
fn baseHdr(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(oetf(eotf(c.r) * u.baseHeadroom), oetf(eotf(c.g) * u.baseHeadroom), oetf(eotf(c.b) * u.baseHeadroom)); }
fn sinebow(t: f32) -> vec3<f32> {
  let a = 6.28318530718 * fract(t);
  return vec3<f32>(sin(a) * 0.5 + 0.5, sin(a + 2.09439510239) * 0.5 + 0.5, sin(a + 4.18879020479) * 0.5 + 0.5);
}
fn hashColor(label: f32) -> vec3<f32> { return sinebow(fract(label * 0.61803398875 + u.colorOffset)); }
fn paletteColor(label: f32) -> vec3<f32> {
  let size = max(u.paletteSize, 1.0);
  let idx = label - size * floor(label / size);
  return textureSampleLevel(palTex, samp, vec2<f32>((idx + 0.5) / size, 0.5), 0.0).rgb;
}
fn instAt(uv: vec2<f32>) -> f32 {
  let p = textureSampleLevel(maskTex, samp, uv, 0.0);
  return floor(p.b * 255.0 + 0.5) + floor(p.a * 255.0 + 0.5) * 256.0;
}

@fragment
fn fs(@location(0) v_uv: vec2<f32>) -> @location(0) vec4<f32> {
  let scrPx = vec2<f32>(abs(dpdx(v_uv.x)), abs(dpdy(v_uv.y)));   // uniform control flow
  let baseCoord = vec2<f32>(v_uv.x, 1.0 - v_uv.y);
  let hasBase = u.imageVisible > 0.5;
  var color = vec3<f32>(0.0);
  var outA = select(0.0, 1.0, hasBase);
  if (hasBase) {
    color = textureSampleLevel(baseTex, samp, baseCoord, 0.0).rgb;
    if (u.baseLinear > 0.5) {
      color = vec3<f32>(oetf(color.r * u.baseHeadroom), oetf(color.g * u.baseHeadroom), oetf(color.b * u.baseHeadroom));
    } else if (u.baseHeadroom > 1.0001) {
      color = baseHdr(color);
    }
  }
  if (u.maskVisible > 0.5 && u.maskOpacity > 0.0) {
    let packed = textureSampleLevel(maskTex, samp, v_uv, 0.0);
    let label = floor(packed.r * 255.0 + 0.5) + floor(packed.g * 255.0 + 0.5) * 256.0;
    let inst  = floor(packed.b * 255.0 + 0.5) + floor(packed.a * 255.0 + 0.5) * 256.0;
    if (label > 0.5) {
      var alpha = clamp(u.maskOpacity, 0.0, 1.0);
      var outline = 0.0;
      if (u.outlinesVisible > 0.5) {
        let imgPx = 1.0 / vec2<f32>(textureDimensions(maskTex, 0));
        let d = max(scrPx, imgPx);
        if (instAt(v_uv + vec2<f32>(d.x, 0.0)) != inst || instAt(v_uv - vec2<f32>(d.x, 0.0)) != inst
         || instAt(v_uv + vec2<f32>(0.0, d.y)) != inst || instAt(v_uv - vec2<f32>(0.0, d.y)) != inst
         || instAt(v_uv + d) != inst || instAt(v_uv - d) != inst
         || instAt(v_uv + vec2<f32>(d.x, -d.y)) != inst || instAt(v_uv + vec2<f32>(-d.x, d.y)) != inst) { outline = 1.0; }
      }
      if (u.maskStyle > 1.5) { alpha = alpha * outline; }
      else if (u.maskStyle < 0.5 && u.outlinesVisible > 0.5) { alpha = mix(alpha * 0.5, alpha, outline); }
      var maskColor = select(hashColor(label), paletteColor(label), u.usePalette > 0.5);
      if (outline > 0.5 && u.useOutlineColor > 0.5) { maskColor = u.outlineColor; alpha = 1.0; }
      if (outline > 0.5 && u.outlineHdrBoost > 1.0) { maskColor = maskColor * u.outlineHdrBoost + 0.12; }
      if (u.highlightLabel > 0.5 && abs(inst - u.highlightLabel) < 0.5) {
        if (u.useOutlineColor > 0.5) {
          maskColor = u.highlightColor;
          alpha = max(alpha, u.highlightAlpha);
        } else {
          let hc = select(hashColor(label), paletteColor(label), u.usePalette > 0.5);
          maskColor = hc * u.highlightBoost + 0.12;
          alpha = max(alpha, 0.9);
        }
      }
      color = select(maskColor, mix(color, maskColor, alpha), hasBase);
      outA = max(outA, alpha);
    }
  }
  return vec4<f32>(color, outA);
}
"""

# ── test scene: 8x8, three instance regions + bg, a 4-colour palette ──
color_ids = np.zeros((H, W), np.int32)
inst_ids = np.zeros((H, W), np.int32)
color_ids[1:4, 1:4] = 1; inst_ids[1:4, 1:4] = 1
color_ids[1:4, 5:7] = 1; inst_ids[1:4, 5:7] = 2     # SAME colour, different instance (divider!)
color_ids[5:7, 2:6] = 2; inst_ids[5:7, 2:6] = 3
palette = np.array([[0, 0, 0, 0], [220, 40, 40, 255], [40, 200, 80, 255], [60, 120, 240, 255]], np.uint8)

mask = np.zeros((H, W, 4), np.uint8)
mask[:, :, 0] = color_ids & 0xFF; mask[:, :, 1] = (color_ids >> 8) & 0xFF
mask[:, :, 2] = inst_ids & 0xFF;  mask[:, :, 3] = (inst_ids >> 8) & 0xFF

U = dict(maskOpacity=0.6, maskVisible=1, outlinesVisible=1, maskStyle=0,
         imageVisible=0, colorOffset=0, paletteSize=len(palette), usePalette=1,
         highlightLabel=3, highlightAlpha=0.5, highlightBoost=1.8, outlineHdrBoost=2.0,
         useOutlineColor=0, baseHeadroom=1.0, baseLinear=0)

# ── NumPy reference of the SAME math (1:1 → outline compares adjacent texels) ──
def ref():
    out = np.zeros((H, W, 4), np.float32)
    def palc(lbl):
        return palette[int(lbl) % len(palette)][:3].astype(np.float32) / 255.0
    # outline: any of 8 clamped neighbours has a different instance id
    outl = np.zeros((H, W), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            ys = np.clip(np.arange(H) + dy, 0, H - 1); xs = np.clip(np.arange(W) + dx, 0, W - 1)
            outl |= (inst_ids[ys][:, xs] != inst_ids)
    for y in range(H):
        for x in range(W):
            label = color_ids[y, x]; inst = inst_ids[y, x]
            color = np.zeros(3, np.float32); outA = 0.0
            if U['maskVisible'] > 0.5 and U['maskOpacity'] > 0.0 and label > 0.5:
                alpha = min(max(U['maskOpacity'], 0.0), 1.0)
                outline = 1.0 if (U['outlinesVisible'] > 0.5 and outl[y, x]) else 0.0
                if U['maskStyle'] > 1.5: alpha = alpha * outline
                elif U['maskStyle'] < 0.5 and U['outlinesVisible'] > 0.5:
                    alpha = (alpha * 0.5) * (1 - outline) + alpha * outline
                mc = palc(label)
                if outline > 0.5 and U['useOutlineColor'] > 0.5: mc = np.array([1, 0, 0], np.float32); alpha = 1.0
                if outline > 0.5 and U['outlineHdrBoost'] > 1.0: mc = mc * U['outlineHdrBoost'] + 0.12
                if U['highlightLabel'] > 0.5 and abs(inst - U['highlightLabel']) < 0.5:
                    hc = palc(label); mc = hc * U['highlightBoost'] + 0.12; alpha = max(alpha, 0.9)
                color = mc; outA = max(outA, alpha)
            out[y, x] = [color[0], color[1], color[2], outA]
    return out

# ── render the WGSL via wgpu-native ──
dev = wgpu.utils.get_default_device()

def mk_tex(arr):  # rgba8unorm from (h,w,4) uint8
    h, w = arr.shape[:2]
    t = dev.create_texture(size=(w, h, 1), format=wgpu.TextureFormat.rgba8unorm,
                           usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dev.queue.write_texture({"texture": t}, np.ascontiguousarray(arr).tobytes(),
                            {"bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
    return t.create_view()

maskV = mk_tex(mask)
palV = mk_tex(palette.reshape(1, -1, 4))
baseV = mk_tex(np.zeros((1, 1, 4), np.uint8))
samp = dev.create_sampler(mag_filter="nearest", min_filter="nearest",
                          address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")

# uniform buffer: matrix(48) + highlightColor(vec3@48) + outlineColor(vec3@64) + 16 f32 @80 → 144
buf = np.zeros(144 // 4, np.float32)
ortho = [2, 0, 0, 0, -2, 0, -1, 1, 1]   # column-major, y-flip → fb row r == mask row r
buf[0:3] = ortho[0:3]; buf[4:7] = ortho[3:6]; buf[8:11] = ortho[6:9]   # mat3 as 3 padded vec4
buf[12:15] = [1, 0, 0]      # highlightColor @ offset 48 (float index 12)
buf[16:19] = [1, 0, 0]      # outlineColor   @ offset 64 (float index 16)
sc = [U['maskOpacity'], U['maskVisible'], U['outlinesVisible'], U['maskStyle'],
      U['imageVisible'], U['colorOffset'], U['paletteSize'], U['usePalette'],
      U['highlightLabel'], U['highlightAlpha'], U['highlightBoost'], U['outlineHdrBoost'],
      U['useOutlineColor'], U['baseHeadroom'], U['baseLinear'], 0.0]
# vec3 size is 12, so the first scalar packs at byte 76 (float 19), NOT 80.
buf[19:35] = sc             # scalars @ offset 76 (float index 19)
ubuf = dev.create_buffer_with_data(data=buf.tobytes(), usage=wgpu.BufferUsage.UNIFORM)

# unit-quad: a_pos==a_uv in [0,1] (triangle-strip)
quad = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1], np.float32)
vbuf = dev.create_buffer_with_data(data=quad.tobytes(), usage=wgpu.BufferUsage.VERTEX)

shader = dev.create_shader_module(code=WGSL)
target = dev.create_texture(size=(W, H, 1), format=wgpu.TextureFormat.rgba16float,
                            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
pipe = dev.create_render_pipeline(
    layout="auto",
    vertex={"module": shader, "entry_point": "vs",
            "buffers": [{"array_stride": 16, "attributes": [
                {"format": "float32x2", "offset": 0, "shader_location": 0},
                {"format": "float32x2", "offset": 8, "shader_location": 1}]}]},
    fragment={"module": shader, "entry_point": "fs",
              "targets": [{"format": wgpu.TextureFormat.rgba16float}]},
    primitive={"topology": "triangle-strip"})
bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
    {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": buf.nbytes}},
    {"binding": 1, "resource": samp},
    {"binding": 2, "resource": baseV},
    {"binding": 3, "resource": maskV},
    {"binding": 4, "resource": palV}])

enc = dev.create_command_encoder()
rp = enc.begin_render_pass(color_attachments=[{
    "view": target.create_view(), "load_op": "clear", "store_op": "store",
    "clear_value": (0, 0, 0, 0)}])
rp.set_pipeline(pipe); rp.set_bind_group(0, bg); rp.set_vertex_buffer(0, vbuf); rp.draw(4)
rp.end()
# readback (bytes_per_row padded to 256)
bpr = ((W * 8 + 255) // 256) * 256
rb = dev.create_buffer(size=bpr * H, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
enc.copy_texture_to_buffer({"texture": target}, {"buffer": rb, "bytes_per_row": bpr, "rows_per_image": H}, (W, H, 1))
dev.queue.submit([enc.finish()])
rb.map_sync(mode=wgpu.MapMode.READ)
raw = np.frombuffer(rb.read_mapped(), np.float16).reshape(H, bpr // 2)[:, :W * 4].reshape(H, W, 4).astype(np.float32)
rb.unmap()

R = ref()
diff = np.abs(raw - R)
print(f"max abs diff: {diff.max():.4f}  | mean: {diff.mean():.5f}")
print(f"HDR present (a channel >1.0): gpu={raw.max():.3f} ref={R.max():.3f}")
print("PARITY:", "PASS" if diff.max() < 0.01 else "FAIL")
