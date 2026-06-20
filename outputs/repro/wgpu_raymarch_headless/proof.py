#!/usr/bin/env python
"""Headless proof: WebGPU volume ray-march tested via wgpu-native (no browser).

Runs the SAME ray-march algorithm as the hostpkg sim3d volume renderer
(AABB slab intersection -> fixed-step march -> MIP / mean / emission-absorption
compositing) on a real GPU through wgpu-native, renders to an offscreen
rgba16float texture, reads pixels back, and asserts the projected image equals
a NumPy ground-truth projection -- from two orthographic viewing directions, so
the camera/ray math (not just one axis) is verified.

This is the CI-able test path for a WebGPU-first 3D ocdkit viewer: the WGSL
below ports verbatim to the browser; only the host API differs.

Run:  /Users/kcutler/.pyenv/shims/python proof.py
"""
import math
import numpy as np
import wgpu
import wgpu.utils

WGSL = """
struct U {
  dims: vec4<f32>,    // NX, NY, NZ, mode(0=add,1=mip,2=mean)
  origin: vec4<f32>,  // image-plane corner in world
  right: vec4<f32>,   // spans full plane width  (world units)
  up: vec4<f32>,      // spans full plane height (world units)
  fwd: vec4<f32>,     // ray direction (unit)
  bmin: vec4<f32>,
  bmax: vec4<f32>,
  img: vec4<f32>,     // imgW, imgH, nsteps, _
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var vol: texture_3d<f32>;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
  var o: VOut;
  let xy = p[vi];
  o.pos = vec4<f32>(xy, 0.0, 1.0);
  // row 0 = top -> uv.y small ; col 0 = left -> uv.x small (matches numpy [row,col])
  o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let dims = vec3<i32>(i32(u.dims.x), i32(u.dims.y), i32(u.dims.z));
  let mode = i32(u.dims.w);
  let ro = u.origin.xyz + u.right.xyz * in.uv.x + u.up.xyz * in.uv.y;
  let rd = normalize(u.fwd.xyz);
  let inv = vec3<f32>(1.0) / rd;                 // axis-aligned dirs -> +/-inf, slab math is robust
  let t1 = (u.bmin.xyz - ro) * inv;
  let t2 = (u.bmax.xyz - ro) * inv;
  let tmn = min(t1, t2);
  let tmx = max(t1, t2);
  var tnear = max(max(tmn.x, tmn.y), tmn.z);
  tnear = max(tnear, 0.0);
  let tfar = min(min(tmx.x, tmx.y), tmx.z);
  if (tnear > tfar) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }

  let nsteps = i32(u.img.z);
  let dt = (tfar - tnear) / f32(nsteps);
  var t = tnear + dt * 0.5;
  var mip = 0.0;
  var ssum = 0.0;
  var cnt = 0.0;
  var acc = vec4<f32>(0.0);
  let den = 1.0;
  for (var i = 0; i < nsteps; i = i + 1) {
    let p = ro + rd * t;
    var vc = vec3<i32>(floor(p));
    vc = clamp(vc, vec3<i32>(0), dims - vec3<i32>(1));
    let s = textureLoad(vol, vc, 0).r;
    mip = max(mip, s);
    ssum = ssum + s;
    cnt = cnt + 1.0;
    let a = s * den;                              // emission-absorption "over" compositing
    let om = 1.0 - acc.w;
    acc = vec4<f32>(acc.rgb + vec3<f32>(s * den * om), acc.w + a * om);
    t = t + dt;
  }
  var outv = 0.0;
  if (mode == 1) { outv = mip; }
  else if (mode == 2) { outv = ssum / max(cnt, 1.0); }
  else { outv = acc.r; }
  return vec4<f32>(outv, outv, outv, 1.0);
}
"""


def build():
    dev = wgpu.utils.get_default_device()
    sm = dev.create_shader_module(code=WGSL)
    pipe = dev.create_render_pipeline(
        layout="auto",
        vertex={"module": sm, "entry_point": "vs"},
        fragment={"module": sm, "entry_point": "fs",
                  "targets": [{"format": wgpu.TextureFormat.rgba16float}]},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    )
    return dev, pipe


def upload_volume(dev, vol_zyx):
    """vol_zyx: (NZ, NY, NX) float32 -> texture sized (W=NX, H=NY, D=NZ)."""
    NZ, NY, NX = vol_zyx.shape
    # texel (x,y,z) must read vol[z,y,x]; texture data is laid out x-fastest, then y, then z.
    data = np.ascontiguousarray(np.transpose(vol_zyx, (0, 1, 2)).astype(np.float32))
    tex = dev.create_texture(
        size=(NX, NY, NZ), dimension=wgpu.TextureDimension.d3,
        format=wgpu.TextureFormat.r32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    dev.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        data.tobytes(),
        {"offset": 0, "bytes_per_row": NX * 4, "rows_per_image": NY},
        (NX, NY, NZ),
    )
    return tex.create_view()


def render(dev, pipe, texview, dims, mode, origin, right, up, fwd, bmin, bmax,
           imgW, imgH, nsteps):
    u = np.zeros(32, np.float32)
    u[0:4] = (*dims, mode)
    u[4:7] = origin
    u[8:11] = right
    u[12:15] = up
    u[16:19] = fwd
    u[20:23] = bmin
    u[24:27] = bmax
    u[28:31] = (imgW, imgH, nsteps)
    ubuf = dev.create_buffer_with_data(
        data=u, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    target = dev.create_texture(
        size=(imgW, imgH, 1), dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    tview = target.create_view()

    bg = dev.create_bind_group(
        layout=pipe.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": u.nbytes}},
                 {"binding": 1, "resource": texview}])

    enc = dev.create_command_encoder()
    rp = enc.begin_render_pass(color_attachments=[{
        "view": tview, "resolve_target": None, "clear_value": (0, 0, 0, 1),
        "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}])
    rp.set_pipeline(pipe)
    rp.set_bind_group(0, bg)
    rp.draw(3, 1, 0, 0)
    rp.end()

    row = imgW * 8                                  # rgba16 = 8 bytes/texel
    padded = math.ceil(row / 256) * 256
    rbuf = dev.create_buffer(size=padded * imgH,
                             usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc.copy_texture_to_buffer(
        {"texture": target, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": rbuf, "offset": 0, "bytes_per_row": padded, "rows_per_image": imgH},
        (imgW, imgH, 1))
    dev.queue.submit([enc.finish()])

    rbuf.map_sync(mode=wgpu.MapMode.READ)
    raw = np.frombuffer(rbuf.read_mapped(), dtype=np.float16).copy()
    rbuf.unmap()
    img = raw.reshape(imgH, padded // 2)[:, :imgW * 4].reshape(imgH, imgW, 4)
    return img[..., 0].astype(np.float32)


def main():
    dev, pipe = build()
    info = dev.adapter.info
    print(f"device: {info.get('device','?')}  backend: {info.get('backend_type','?')}  "
          f"(headless, no browser)\n")

    rng = np.random.default_rng(0)
    NZ, NY, NX = 8, 10, 12
    vol = (rng.random((NZ, NY, NX)) * 0.9).astype(np.float32)   # in [0,0.9]
    texview = upload_volume(dev, vol)
    dims = (NX, NY, NZ)
    BIG = 1000.0
    results = []

    def check(name, gpu, ref, tol=3e-3):
        err = float(np.max(np.abs(gpu - ref)))
        ok = err < tol
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} maxabs={err:.2e}  shape={gpu.shape}")

    # --- View +Z: rays along +Z, image plane spans (X,Y); nsteps=NZ -> 1 sample/voxel ---
    gpu = render(dev, pipe, texview, dims, 1,
                 origin=(0, 0, -BIG), right=(NX, 0, 0), up=(0, NY, 0), fwd=(0, 0, 1),
                 bmin=(0, 0, 0), bmax=dims, imgW=NX, imgH=NY, nsteps=NZ)
    check("MIP, view +Z  == max(vol, axis=0)", gpu, vol.max(axis=0))

    gpu = render(dev, pipe, texview, dims, 2,
                 origin=(0, 0, -BIG), right=(NX, 0, 0), up=(0, NY, 0), fwd=(0, 0, 1),
                 bmin=(0, 0, 0), bmax=dims, imgW=NX, imgH=NY, nsteps=NZ)
    check("MEAN, view +Z == mean(vol, axis=0)", gpu, vol.mean(axis=0))

    gpu = render(dev, pipe, texview, dims, 0,
                 origin=(0, 0, -BIG), right=(NX, 0, 0), up=(0, NY, 0), fwd=(0, 0, 1),
                 bmin=(0, 0, 0), bmax=dims, imgW=NX, imgH=NY, nsteps=NZ)
    check("ADD, view +Z  == 1-prod(1-vol, axis=0)", gpu, 1.0 - np.prod(1.0 - vol, axis=0))

    # --- View +Y: rays along +Y (different projection axis), plane spans (X,Z); nsteps=NY ---
    gpu = render(dev, pipe, texview, dims, 1,
                 origin=(0, -BIG, 0), right=(NX, 0, 0), up=(0, 0, NZ), fwd=(0, 1, 0),
                 bmin=(0, 0, 0), bmax=dims, imgW=NX, imgH=NZ, nsteps=NY)
    check("MIP, view +Y  == max(vol, axis=1)", gpu, vol.max(axis=1))

    print()
    print("ALL PASS" if all(results) else "SOME FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
