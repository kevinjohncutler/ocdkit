"""Item 3 accuracy check: does emission-absorption alpha = 1-exp(-tau) remove the
dependence on how voxel boundaries chop the ray, vs the shipped clamp(tau)?

Renders a CONSTANT-value cube with the shipped raymarch_compute.wgsl (and the
1-exp variant, same patch anchor as variants.js) through wgpu-native, with an
orthographic camera at several tilts. For a constant field the exact answer is
analytic: alpha = 1 - exp(-s * density * chord), chord = ray/box intersection
length. Reports the error of each shader against that.
"""
import math
from pathlib import Path

import numpy as np
import wgpu
import wgpu.utils

SRC = (Path(__file__).resolve().parents[3] / "src/ocdkit/viewer/web/js/raymarch_compute.wgsl").read_text()
A = "let a = clamp(sg * density * segLen, 0.0, 1.0);"
assert SRC.count(A) == 1
EXP = SRC.replace(A, "let a = 1.0 - exp(-sg * density * segLen);")
N, W, H, BIG = 48, 256, 256, 1000.0
dev = wgpu.utils.get_default_device()


def pipe(code):
    return dev.create_compute_pipeline(layout="auto",
                                       compute={"module": dev.create_shader_module(code=code), "entry_point": "cs"})


def tex(fmt, size, data, bpr, dim="3d"):
    t = dev.create_texture(size=size, format=fmt, dimension=dim,
                           usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dev.queue.write_texture({"texture": t}, data, {"bytes_per_row": bpr, "rows_per_image": size[1]}, size)
    return t


def cam(yaw, pitch, half):
    d = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    right = np.cross([0, 1, 0], d); right /= np.linalg.norm(right)
    up = np.cross(d, right)
    inv = np.zeros(16, np.float32)                        # column-major NDC -> world
    inv[0:3], inv[4:7], inv[8:11] = right * half, up * half, d * 2 * BIG
    inv[12:15], inv[15] = -d * BIG, 1.0
    return inv, d, right, up


def render(p, inv, s, density):
    vol = tex("r16float", (N, N, N), np.full(N ** 3, s, np.float16).tobytes(), N * 2)
    lab = tex("r8uint", (N, N, N), np.zeros(N ** 3, np.uint8).tobytes(), N)
    ramp = np.linspace(0, 1, 256, dtype=np.float32)
    lut = tex("rgba16float", (256, 1, 1), np.stack([ramp, ramp, ramp, np.ones(256, np.float32)], 1)
              .astype(np.float16).tobytes(), 256 * 8, dim="2d")
    out = dev.create_texture(size=(W, H, 1), format="rgba16float",
                             usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC)
    u = np.zeros(44, np.float32)
    u[0:16] = inv
    u[20:24] = [-N / 2, -N / 2, -N / 2, 0]; u[24:28] = [N / 2, N / 2, N / 2, 0]
    u[28:32] = [N, N, N, 0]                                   # mode 0 = emission-absorption
    u[32:36] = [2 * N, density, 1.0, 0.0]                     # labels off
    u[36:40] = [1.0, 1.0, 1.0, 1.0]
    u[40:44] = [0.4, 0.0, 24.0, 1.0]
    ub = dev.create_buffer_with_data(data=u.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    bg = dev.create_bind_group(layout=p.get_bind_group_layout(0), entries=[
        {"binding": 0, "resource": {"buffer": ub}}, {"binding": 1, "resource": vol.create_view()},
        {"binding": 2, "resource": lab.create_view()}, {"binding": 3, "resource": lut.create_view()},
        {"binding": 4, "resource": out.create_view()}])
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass(); cp.set_pipeline(p); cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(W // 8, H // 8, 1); cp.end()
    rb = dev.create_buffer(size=W * H * 8, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc.copy_texture_to_buffer({"texture": out}, {"buffer": rb, "bytes_per_row": W * 8}, (W, H, 1))
    dev.queue.submit([enc.finish()])
    rb.map_sync(mode=wgpu.MapMode.READ)
    a = np.frombuffer(rb.read_mapped(), np.float16).reshape(H, W, 4).astype(np.float64)
    rb.unmap()
    return a[..., 3]


def chords(d, right, up, half):
    ys, xs = np.mgrid[0:H, 0:W]
    ndx, ndy = (xs + 0.5) / W * 2 - 1, 1 - (ys + 0.5) / H * 2
    ro = (ndx[..., None] * right + ndy[..., None] * up) * half - d * BIG
    with np.errstate(divide="ignore"):
        inv = 1.0 / np.where(d == 0, 1e-12, d)
    t1, t2 = (-N / 2 - ro) * inv, (N / 2 - ro) * inv
    tn = np.minimum(t1, t2).max(-1); tf = np.maximum(t1, t2).min(-1)
    return np.clip(tf - tn, 0, None)


if __name__ == "__main__":
    P = {"shipped clamp(tau)": pipe(SRC), "1-exp(-tau)": pipe(EXP)}
    s = 0.5
    print("| Density | View (yaw, pitch deg) | Shader | Mean abs error | Max abs error | RMS error |")
    print("|---|---|---|---|---|---|")
    for density in [0.05, 0.5]:
        for yaw, pitch in [(0, 0), (30, 20), (45, 35)]:
            half = N * 0.95
            inv, d, right, up = cam(math.radians(yaw), math.radians(pitch), half)
            ch = chords(d, right, up, half)
            exact = 1 - np.exp(-s * density * ch)
            inside = ch > 1.0                                    # skip grazing edge pixels
            for name, p in P.items():
                err = render(p, inv, s, density)[inside] - exact[inside]
                print(f"| {density} | {yaw}, {pitch} | {name} | {np.abs(err).mean():.4f} | "
                      f"{np.abs(err).max():.4f} | {math.sqrt((err ** 2).mean()):.4f} |")
