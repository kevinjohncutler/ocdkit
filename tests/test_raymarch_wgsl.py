"""Headless validation of the shipped raymarch.wgsl via wgpu-native (no browser).

Loads the EXACT shader the browser ships, drives it with an axis-aligned
orthographic camera (so MIP/mean are exactly checkable vs numpy), and validates
ray generation, intensity sampling, MIP/mean compositing, and label colouring +
blend. This is the CI gate for the WebGPU 3D layer's shader.
"""
import math
import os

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")
import wgpu.utils  # noqa: E402

WGSL_PATH = "/Volumes/DataDrive/ocdkit/src/ocdkit/viewer/web/js/raymarch.wgsl"
pytestmark = pytest.mark.skipif(not os.path.exists(WGSL_PATH), reason="raymarch.wgsl missing")


def _label_color(lab):
    """numpy mirror of raymarch.wgsl labelColor / volume3d-view.js labelColor."""
    if lab == 0:
        return np.zeros(3, np.float32)
    h = (lab * 0.61803398875) % 1.0
    s, v = 0.65, 1.0
    i = math.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    return np.array(table[i % 6], np.float32)


def _ortho_inv_vp(NX, NY, NZ, BIG=1000.0):
    """Column-major invViewProj mapping NDC -> world voxel coords, rays along +Z."""
    return np.array([
        NX / 2, 0, 0, 0,
        0, NY / 2, 0, 0,
        0, 0, 2 * BIG, 0,
        NX / 2, NY / 2, -BIG, 1,
    ], np.float32)


class Harness:
    def __init__(self):
        self.dev = wgpu.utils.get_default_device()
        with open(WGSL_PATH) as fh:
            code = fh.read()
        sm = self.dev.create_shader_module(code=code)
        self.pipe = self.dev.create_render_pipeline(
            layout="auto",
            vertex={"module": sm, "entry_point": "vs"},
            fragment={"module": sm, "entry_point": "fs",
                      "targets": [{"format": wgpu.TextureFormat.rgba16float}]},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

    def _tex3d(self, data, fmt, bytes_per_elem):
        NZ, NY, NX = data.shape
        tex = self.dev.create_texture(
            size=(NX, NY, NZ), dimension=wgpu.TextureDimension.d3, format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.dev.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(data).tobytes(),
            {"offset": 0, "bytes_per_row": NX * bytes_per_elem, "rows_per_image": NY},
            (NX, NY, NZ))
        return tex.create_view()

    def render(self, vol_zyx, lab_zyx, mode, *, imgW, imgH, nsteps,
               density=1.0, label_opacity=0.0, show_labels=0.0, show_image=1.0, iscale=1.0,
               inv_vp=None, box_min=None, box_max=None):
        NZ, NY, NX = vol_zyx.shape
        volv = self._tex3d(vol_zyx.astype(np.float32), wgpu.TextureFormat.r32float, 4)
        labv = self._tex3d(lab_zyx.astype(np.uint8), wgpu.TextureFormat.r8uint, 1)
        if inv_vp is None:
            inv_vp = _ortho_inv_vp(NX, NY, NZ)       # axis-aligned ortho (exact)
            box_min = (0, 0, 0); box_max = (NX, NY, NZ)

        u = np.zeros(40, np.float32)
        u[0:16] = np.asarray(inv_vp, np.float32)
        u[16:20] = (0, 0, -1000, 1)              # camPos (unused by ray reconstruction)
        u[20:24] = (*box_min, 0)                  # boxMin
        u[24:28] = (*box_max, 0)                  # boxMax
        u[28:32] = (NX, NY, NZ, mode)             # dims + mode
        u[32:36] = (nsteps, density, label_opacity, show_labels)
        u[36:40] = (iscale, show_image, 0, 0)
        ubuf = self.dev.create_buffer_with_data(
            data=u, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        target = self.dev.create_texture(
            size=(imgW, imgH, 1), dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        bg = self.dev.create_bind_group(
            layout=self.pipe.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": u.nbytes}},
                     {"binding": 1, "resource": volv},
                     {"binding": 2, "resource": labv}])
        enc = self.dev.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{
            "view": target.create_view(), "clear_value": (0, 0, 0, 0),
            "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}])
        rp.set_pipeline(self.pipe); rp.set_bind_group(0, bg); rp.draw(3); rp.end()

        row = imgW * 8
        padded = math.ceil(row / 256) * 256
        rbuf = self.dev.create_buffer(size=padded * imgH,
                                      usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc.copy_texture_to_buffer(
            {"texture": target, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": rbuf, "offset": 0, "bytes_per_row": padded, "rows_per_image": imgH},
            (imgW, imgH, 1))
        self.dev.queue.submit([enc.finish()])
        rbuf.map_sync(mode=wgpu.MapMode.READ)
        raw = np.frombuffer(rbuf.read_mapped(), np.float16).copy()
        rbuf.unmap()
        return raw.reshape(imgH, padded // 2)[:, :imgW * 4].reshape(imgH, imgW, 4).astype(np.float32)


@pytest.fixture(scope="module")
def hz():
    return Harness()


def test_mip_matches_numpy(hz):
    rng = np.random.default_rng(0)
    NZ, NY, NX = 8, 10, 12
    vol = (rng.random((NZ, NY, NX)) * 0.9).astype(np.float32)
    lab = np.zeros((NZ, NY, NX), np.uint8)
    out = hz.render(vol, lab, 1, imgW=NX, imgH=NY, nsteps=NZ)
    mip = np.flipud(out[..., 0])  # camera flips Y
    assert np.max(np.abs(mip - vol.max(axis=0))) < 3e-3


def test_inverted_box_y_renders_clean_vertical_flip(hz):
    """The viewer flips the box Y (min.y>max.y) so 3D matches the 2.5D image
    orientation. Verify the shader's slab AABB handles an inverted Y range and
    that it produces exactly a vertical flip (no corruption)."""
    rng = np.random.default_rng(2)
    NZ, NY, NX = 6, 8, 10
    vol = (rng.random((NZ, NY, NX)) * 0.9).astype(np.float32)
    lab = np.zeros((NZ, NY, NX), np.uint8)
    normal = hz.render(vol, lab, 1, imgW=NX, imgH=NY, nsteps=NZ)[..., 0]
    inv = hz.render(vol, lab, 1, imgW=NX, imgH=NY, nsteps=NZ,
                    inv_vp=_ortho_inv_vp(NX, NY, NZ),
                    box_min=(0, NY, 0), box_max=(NX, 0, NZ))[..., 0]
    assert np.max(np.abs(inv - np.flipud(normal))) < 3e-3


def test_mean_matches_numpy(hz):
    rng = np.random.default_rng(1)
    NZ, NY, NX = 8, 10, 12
    vol = (rng.random((NZ, NY, NX)) * 0.9).astype(np.float32)
    lab = np.zeros((NZ, NY, NX), np.uint8)
    out = hz.render(vol, lab, 2, imgW=NX, imgH=NY, nsteps=NZ)
    mean = np.flipud(out[..., 0])
    assert np.max(np.abs(mean - vol.mean(axis=0))) < 3e-3


def test_label_coloring_and_blend(hz):
    NZ, NY, NX = 6, 12, 12
    vol = np.zeros((NZ, NY, NX), np.float32)
    lab = np.zeros((NZ, NY, NX), np.uint8)
    L = 7
    vol[:, 3:7, 4:8] = 1.0     # a bright block (footprint y∈[3,7) x∈[4,8))
    lab[:, 3:7, 4:8] = L
    out = hz.render(vol, lab, 1, imgW=NX, imgH=NY, nsteps=NZ,
                    show_labels=1.0, label_opacity=1.0)
    out = np.flipud(out)       # undo camera Y flip -> [y, x, rgba]
    want = _label_color(L)
    # inside footprint: rgb == label colour, alpha == 1
    inside = out[3:7, 4:8]
    assert np.max(np.abs(inside[..., :3] - want)) < 3e-3
    assert np.min(inside[..., 3]) > 0.99
    # outside: transparent
    assert out[0, 0, 3] < 1e-2


def test_image_and_label_layer_toggles(hz):
    """image-only shows grayscale (no label colour); label-only shows pure label
    colour (no grayscale); proving the two layers are independent + composited."""
    NZ, NY, NX = 6, 12, 12
    vol = np.zeros((NZ, NY, NX), np.float32)
    lab = np.zeros((NZ, NY, NX), np.uint8)
    L = 7
    vol[:, 3:7, 4:8] = 1.0
    lab[:, 3:7, 4:8] = L
    want = _label_color(L)

    # image only: footprint is grayscale white, NOT the label colour
    img = np.flipud(hz.render(vol, lab, 1, imgW=NX, imgH=NY, nsteps=NZ,
                              show_image=1.0, show_labels=0.0))
    inside = img[3:7, 4:8]
    assert np.allclose(inside[..., :3], 1.0, atol=3e-3)          # white/gray
    assert np.max(np.abs(inside[..., :3].mean(0).mean(0) - want)) > 0.1  # not label colour

    # label only: footprint is the label colour, no grayscale; rest transparent
    lab_only = np.flipud(hz.render(vol, lab, 1, imgW=NX, imgH=NY, nsteps=NZ,
                                   show_image=0.0, show_labels=1.0, label_opacity=1.0))
    assert np.max(np.abs(lab_only[3:7, 4:8, :3] - want)) < 3e-3
    assert lab_only[0, 0, 3] < 1e-2                               # bg transparent


import json
import shutil
import subprocess

_NODE = shutil.which("node") or "/opt/homebrew/bin/node"
_EMIT = "/Volumes/DataDrive/ocdkit/tests/js/emit_camera.mjs"


@pytest.mark.skipif(not os.path.exists(_NODE), reason="node not found")
@pytest.mark.skipif(not os.path.exists(_EMIT), reason="emit_camera.mjs missing")
def test_perspective_camera_from_mat4js_composes_with_shader(hz):
    """Bridge the SHIPPED mat4.js orbit camera into the shader: a centred cube
    must project to non-empty, roughly-centred pixels (validates the perspective
    invViewProj path + WebGPU [0,1] depth convention end-to-end)."""
    out = subprocess.run([_NODE, _EMIT, "0", "0"], capture_output=True, text=True, check=True)
    cam = json.loads(out.stdout)
    NX, NY, NZ = cam["dims"]
    vol = np.zeros((NZ, NY, NX), np.float32)
    c = NZ // 2
    vol[c - 2:c + 2, NY // 2 - 2:NY // 2 + 2, NX // 2 - 2:NX // 2 + 2] = 1.0
    lab = np.zeros((NZ, NY, NX), np.uint8)
    W = H = 64
    img = hz.render(vol, lab, 1, imgW=W, imgH=H, nsteps=192,
                    inv_vp=cam["invViewProj"], box_min=cam["box"]["min"], box_max=cam["box"]["max"])
    alpha = img[..., 3]
    assert alpha.sum() > 0, "perspective render produced nothing"
    ys, xs = np.nonzero(alpha > 0.1)
    assert ys.size > 0
    cy, cx = ys.mean(), xs.mean()
    # camera looks at the box centre -> the cube lands near image centre
    assert abs(cy - H / 2) < H / 4, f"centroid y={cy}"
    assert abs(cx - W / 2) < W / 4, f"centroid x={cx}"

