"""Headless validation of the shipped overlay.wgsl line shader via wgpu-native.

Renders a known voxel-space segment with an axis-aligned ortho viewProj and
asserts coloured pixels appear along the expected screen row / column range —
verifying the voxel->world->clip transform + per-vertex colour the 3D overlays
(trajectories, lineage, points, flow, affinity) rely on.
"""
import math
import os

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")
import wgpu.utils  # noqa: E402

WGSL = "/Volumes/DataDrive/ocdkit/src/ocdkit/viewer/web/js/overlay.wgsl"
pytestmark = pytest.mark.skipif(not os.path.exists(WGSL), reason="overlay.wgsl missing")


def _ortho_view_proj(W, H):
    """world(=voxel, box 0..dims) -> NDC: x->[-1,1], y->[-1,1], z->0.5. Column-major."""
    return np.array([
        2 / W, 0, 0, 0,
        0, 2 / H, 0, 0,
        0, 0, 0, 0,
        -1, -1, 0.5, 1,
    ], np.float32)


def test_overlay_segment_renders_coloured_pixels():
    dev = wgpu.utils.get_default_device()
    with open(WGSL) as fh:
        sm = dev.create_shader_module(code=fh.read())
    pipe = dev.create_render_pipeline(
        layout="auto",
        vertex={"module": sm, "entry_point": "vs", "buffers": [
            {"array_stride": 12, "attributes": [{"shader_location": 0, "format": "float32x3", "offset": 0}]},
            {"array_stride": 12, "attributes": [{"shader_location": 1, "format": "float32x3", "offset": 0}]},
        ]},
        fragment={"module": sm, "entry_point": "fs",
                  "targets": [{"format": wgpu.TextureFormat.rgba16float}]},
        primitive={"topology": wgpu.PrimitiveTopology.line_list},
    )

    NX, NY, NZ = 16, 12, 4
    W, H = NX, NY
    # horizontal red segment at y=5 from x=2 to x=10 (voxel coords)
    positions = np.array([2, 5, 0, 10, 5, 0], np.float32)
    colors = np.array([1, 0, 0, 1, 0, 0], np.float32)
    pbuf = dev.create_buffer_with_data(data=positions, usage=wgpu.BufferUsage.VERTEX)
    cbuf = dev.create_buffer_with_data(data=colors, usage=wgpu.BufferUsage.VERTEX)

    u = np.zeros(28, np.float32)
    u[0:16] = _ortho_view_proj(W, H)
    u[16:20] = (0, 0, 0, 0)            # boxMin
    u[20:24] = (NX, NY, NZ, 0)         # boxMax
    u[24:28] = (NX, NY, NZ, 0)         # dims
    ubuf = dev.create_buffer_with_data(data=u, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    target = dev.create_texture(
        size=(W, H, 1), dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0),
                               entries=[{"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": u.nbytes}}])
    enc = dev.create_command_encoder()
    rp = enc.begin_render_pass(color_attachments=[{
        "view": target.create_view(), "clear_value": (0, 0, 0, 1),
        "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}])
    rp.set_pipeline(pipe)
    rp.set_bind_group(0, bg)
    rp.set_vertex_buffer(0, pbuf)
    rp.set_vertex_buffer(1, cbuf)
    rp.draw(2)
    rp.end()

    row = W * 8
    padded = math.ceil(row / 256) * 256
    rbuf = dev.create_buffer(size=padded * H, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc.copy_texture_to_buffer(
        {"texture": target, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": rbuf, "offset": 0, "bytes_per_row": padded, "rows_per_image": H}, (W, H, 1))
    dev.queue.submit([enc.finish()])
    rbuf.map_sync(mode=wgpu.MapMode.READ)
    img = np.frombuffer(rbuf.read_mapped(), np.float16).copy().reshape(H, padded // 2)[:, :W * 4].reshape(H, W, 4).astype(np.float32)
    rbuf.unmap()

    red = (img[..., 0] > 0.5) & (img[..., 1] < 0.3) & (img[..., 2] < 0.3)
    assert red.sum() >= 6, f"too few red pixels: {red.sum()}"
    ys, xs = np.nonzero(red)
    # y=5 -> ndc.y=2*5/12-1=-0.1667 -> framebuffer row ~7
    assert abs(np.median(ys) - 7) <= 1, f"row {np.median(ys)}"
    assert xs.min() <= 3 and xs.max() >= 9, f"x range {xs.min()}..{xs.max()}"
