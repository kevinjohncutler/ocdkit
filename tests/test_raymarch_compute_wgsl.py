"""Headless validation of the shipped raymarch_compute.wgsl (the default 3D
render path) via wgpu-native.

The compute march adds pipeline-override render states, a label-first march and
empty-space skipping over a brick grid. These tests pin it against:
  * numpy, exactly, for MIP and mean on an axis-aligned orthographic view;
  * the unoptimized fragment twin raymarch.wgsl at oblique views, in every mode
    and layer combination (the only allowed differences are rays grazing a voxel
    edge, where the brick jump and the DDA can round to different neighbors);
  * the analytic answer for emission-absorption through a constant cube.
"""
import math
import os

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")
import wgpu.utils  # noqa: E402

JS = os.path.join(os.path.dirname(__file__), "..", "src", "ocdkit", "viewer", "web", "js")
COMPUTE = os.path.join(JS, "raymarch_compute.wgsl")
FRAGMENT = os.path.join(JS, "raymarch.wgsl")
BRICK = 16
BIG = 1000.0


@pytest.fixture(scope="module")
def dev():
    return wgpu.utils.get_default_device()


def _read(p):
    with open(p) as fh:
        return fh.read()


def _tex(dev, fmt, size, data, bpr, dim="3d"):
    t = dev.create_texture(size=size, format=fmt, dimension=dim,
                           usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dev.queue.write_texture({"texture": t}, data, {"bytes_per_row": bpr, "rows_per_image": size[1]}, size)
    return t


def _bricks(vol16, lab):
    """numpy mirror of volume3d-gpu.js brickMax / brickAny."""
    NZ, NY, NX = vol16.shape
    pz, py, px = (-NZ) % BRICK, (-NY) % BRICK, (-NX) % BRICK
    v = np.pad(vol16.astype(np.float32), ((0, pz), (0, py), (0, px)))
    m = np.pad(lab > 0, ((0, pz), (0, py), (0, px)))
    shp = (v.shape[0] // BRICK, BRICK, v.shape[1] // BRICK, BRICK, v.shape[2] // BRICK, BRICK)
    return (v.reshape(shp).max(axis=(1, 3, 5)).astype(np.float16),
            m.reshape(shp).any(axis=(1, 3, 5)).astype(np.uint8))


def _ortho(yaw, pitch, half):
    """Column-major NDC -> world matrix for an orthographic view along (yaw, pitch)."""
    d = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    right = np.cross([0, 1, 0], d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    inv = np.zeros(16, np.float32)
    inv[0:3], inv[4:7], inv[8:11] = right * half, up * half, d * 2 * BIG
    inv[12:15], inv[15] = -d * BIG, 1.0
    return inv, d, right, up


def _uniform(inv, dims, mode, density=1.0, opacity=1.0, show_img=1, show_lab=1, window=(0.0, 1.0),
             exposure=1.0, faces_mix=0.0):
    NX, NY, NZ = dims
    u = np.zeros(48, np.float32)
    u[0:16] = inv
    u[20:24] = [-NX / 2, -NY / 2, -NZ / 2, 0]
    u[24:28] = [NX / 2, NY / 2, NZ / 2, 0]
    u[28:32] = [NX, NY, NZ, mode]
    u[32:36] = [2 * max(dims), density, opacity, show_lab]
    u[36:40] = [1.0, show_img, 1.0, 1.0]
    u[40:44] = [0.4, 0.0, 24.0, 1.0]
    u[44:48] = [window[0], 1.0 / (window[1] - window[0]), exposure, faces_mix]
    return u


class Scene:
    def __init__(self, dev, vol16, lab, lut_alpha=None, lut_rgb=None):
        self.dev = dev
        NZ, NY, NX = vol16.shape
        self.dims = (NX, NY, NZ)
        self.vol = _tex(dev, "r16float", (NX, NY, NZ), vol16.tobytes(), NX * 2)
        self.lab = _tex(dev, "r8uint", (NX, NY, NZ), lab.astype(np.uint8).tobytes(), NX)
        bi, bl = _bricks(vol16, lab)
        bz, by, bx = bi.shape
        self.bimg = _tex(dev, "r16float", (bx, by, bz), bi.tobytes(), bx * 2)
        self.blab = _tex(dev, "r8uint", (bx, by, bz), bl.tobytes(), bx)
        ramp = np.linspace(0, 1, 256, dtype=np.float32)
        alpha = np.ones(256, np.float32) if lut_alpha is None else np.asarray(lut_alpha, np.float32)
        rgb = np.stack([ramp, ramp, ramp], 1) if lut_rgb is None else np.asarray(lut_rgb, np.float32)
        self.lut = _tex(dev, "rgba16float", (256, 1, 1),
                        np.concatenate([rgb, alpha[:, None]], 1).astype(np.float16).tobytes(),
                        256 * 8, dim="2d")
        self.cmod = dev.create_shader_module(code=_read(COMPUTE))
        fmod = dev.create_shader_module(code=_read(FRAGMENT))
        self.frag = dev.create_render_pipeline(
            layout="auto", vertex={"module": fmod, "entry_point": "vs"},
            fragment={"module": fmod, "entry_point": "fs", "targets": [{"format": "rgba16float"}]},
            primitive={"topology": "triangle-list"})

    def _readback(self, enc, tex, W, H):
        rb = self.dev.create_buffer(size=W * H * 8, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc.copy_texture_to_buffer({"texture": tex}, {"buffer": rb, "bytes_per_row": W * 8}, (W, H, 1))
        self.dev.queue.submit([enc.finish()])
        rb.map_sync(mode=wgpu.MapMode.READ)
        out = np.frombuffer(rb.read_mapped(), np.float16).reshape(H, W, 4).astype(np.float32)
        rb.unmap()
        return out

    def compute_pipeline(self, mode, show_img, show_lab, shade=1, transp=0, classify=0):
        return self.dev.create_compute_pipeline(layout="auto", compute={
            "module": self.cmod, "entry_point": "cs",
            "constants": {"MODE": mode, "SHOW_IMG": show_img, "SHOW_LAB": show_lab, "SHADE_LAB": shade,
                          "BRICK": float(BRICK), "TRANSP": transp,
                          "CLASSIFY": classify}})

    def compute(self, u, mode, show_img, show_lab, W, H, transp=0, faces=0, classify=0):
        u = u.copy(); u[47] = faces           # voxel shading t (0 = path length, 1 = voxel faces)
        p = self.compute_pipeline(mode, show_img, show_lab, transp=transp, classify=classify)
        out = self.dev.create_texture(size=(W, H, 1), format="rgba16float",
                                      usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC)
        ub = self.dev.create_buffer_with_data(data=u.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        res = [ub, self.vol, self.lab, self.lut, out, self.bimg, self.blab]
        bg = self.dev.create_bind_group(layout=p.get_bind_group_layout(0), entries=[
            {"binding": i, "resource": {"buffer": r} if i == 0 else r.create_view()} for i, r in enumerate(res)])
        enc = self.dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(p); cp.set_bind_group(0, bg); cp.dispatch_workgroups(-(-W // 8), -(-H // 8), 1); cp.end()
        return self._readback(enc, out, W, H)

    def fragment(self, u, W, H):
        out = self.dev.create_texture(size=(W, H, 1), format="rgba16float",
                                      usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        ub = self.dev.create_buffer_with_data(data=u.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        res = [ub, self.vol, self.lab, self.lut]
        bg = self.dev.create_bind_group(layout=self.frag.get_bind_group_layout(0), entries=[
            {"binding": i, "resource": {"buffer": r} if i == 0 else r.create_view()} for i, r in enumerate(res)])
        enc = self.dev.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{"view": out.create_view(), "load_op": "clear",
                                                        "store_op": "store", "clear_value": (0, 0, 0, 0)}])
        rp.set_pipeline(self.frag); rp.set_bind_group(0, bg); rp.draw(3); rp.end()
        return self._readback(enc, out, W, H)


def _sparse_scene(seed=0, shape=(40, 48, 56)):
    """Sparse blobs over a noisy background (like real fluorescence) plus a few
    label cells, so both the image and the label bricks really skip."""
    rng = np.random.default_rng(seed)
    NZ, NY, NX = shape
    z, y, x = np.mgrid[0:NZ, 0:NY, 0:NX]
    vol = rng.random(shape) * 0.15
    lab = np.zeros(shape, np.uint8)
    for k in range(6):
        c = rng.uniform([6, 6, 6], [NZ - 6, NY - 6, NX - 6])
        r2 = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2
        vol = np.maximum(vol, rng.uniform(0.5, 1.0) * np.exp(-r2 / 12.0))
        if k < 4:
            lab[r2 < 9] = k + 1
    return vol.astype(np.float16), lab


def test_every_render_state_compiles(dev):
    s = Scene(dev, *_sparse_scene())
    for mode in (0, 1, 2, 3):
        for img in (0, 1):
            for lab in (0, 1):
                for sh in (0, 1):
                    s.compute_pipeline(mode, img, lab, sh)


def test_mip_and_mean_exact_axis_aligned(dev):
    """Looking straight down z with one pixel per voxel column, MIP is the column
    max and mean the column mean, even with image-brick skipping active."""
    vol, lab = _sparse_scene()
    NZ, NY, NX = vol.shape
    # a square ortho view needs a square cross-section, and texture readback
    # rows must be a multiple of 256 bytes (32 rgba16float pixels)
    n = 32
    vol, lab = vol[:, :n, :n], lab[:, :n, :n]
    s = Scene(dev, np.ascontiguousarray(vol), np.ascontiguousarray(lab))
    inv, *_ = _ortho(0.0, 0.0, n / 2)
    ref_img = vol.astype(np.float32)[:, ::-1, :]                 # row iy <-> voxel y = n-1-iy
    for mode, ref in ((1, ref_img.max(axis=0)), (2, ref_img.mean(axis=0))):
        out = s.compute(_uniform(inv, (n, n, NZ), mode, show_lab=0), mode, 1, 0, n, n)
        np.testing.assert_allclose(out[..., 3], ref, atol=2e-3)


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("layers", ["img", "imglab", "lab", "imglab50"])
def test_compute_matches_fragment_reference(dev, mode, layers):
    if layers == "lab" and mode != 1:
        pytest.skip("mode is irrelevant without the image")
    vol, lab = _sparse_scene(seed=3)
    s = Scene(dev, vol, lab)
    NZ, NY, NX = vol.shape
    show_img, show_lab = (0 if layers == "lab" else 1), (0 if layers == "img" else 1)
    opacity = 0.5 if layers == "imglab50" else 1.0
    W = H = 160
    for yaw, pitch in ((0.4, 0.3), (1.1, -0.5), (2.5, 0.9)):
        inv, *_ = _ortho(yaw, pitch, max(NX, NY, NZ) * 0.8)
        u = _uniform(inv, (NX, NY, NZ), mode, opacity=opacity, show_img=show_img, show_lab=show_lab)
        a, b = s.compute(u, mode, show_img, show_lab, W, H), s.fragment(u, W, H)
        diff = np.abs(a - b).max(axis=-1)
        frac = float((diff > 1 / 255).mean())
        assert frac < 0.005, f"yaw={yaw} pitch={pitch}: {frac:.4%} of pixels differ"


def _mida_reference(cols, density, substeps=200):
    """MIDA along axis 0 (front first), one unit of path per voxel, on intensity,
    computed independently of the shader's closed form: the paper's fade-then-add
    I = beta I + (1 - beta A) a s, A = beta A + (1 - beta A) a in `substeps` tiny
    steps per voxel (its continuous limit), with the running max f approaching a
    brighter s as s - (s - f) e^(-h / 0.8) and beta = (1 - f_new) / (1 - f_old).
    The displayed value is the accumulated I (then colormapped, alpha I)."""
    h = 1.0 / substeps
    I = np.zeros(cols.shape[1:]); A = np.zeros(cols.shape[1:]); f = np.zeros(cols.shape[1:])
    for s in cols.astype(np.float64):
        a = 1 - np.exp(-density * s * h)
        for _ in range(substeps):
            f_new = np.where(s > f, s - (s - f) * np.exp(-h / 0.8), f)
            beta = (1 - f_new) / np.maximum(1 - f, 1e-9)
            keep = beta * A
            I = beta * I + (1 - keep) * a * s
            A = keep + (1 - keep) * a
            f = f_new
    return np.clip(I, 0, 1)


@pytest.mark.parametrize("density", [0.3, 2.0])
def test_mida_matches_reference(dev, density):
    """Straight down z with one pixel per voxel column, MIDA equals the published
    recurrence computed in NumPy (early termination only once nothing can change)."""
    vol, lab = _sparse_scene(seed=5)
    n = 32
    vol = np.ascontiguousarray(vol[:, :n, :n])
    NZ = vol.shape[0]
    s = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    inv, *_ = _ortho(0.0, 0.0, n / 2)
    out = s.compute(_uniform(inv, (n, n, NZ), 3, density=density, show_lab=0), 3, 1, 0, n, n)
    v = _mida_reference(vol.astype(np.float32)[:, ::-1, :], density)
    np.testing.assert_allclose(out[..., 3], v, atol=3e-3)          # gray colormap: color = alpha = v
    np.testing.assert_allclose(out[..., 0], v, atol=3e-3)


def _two_color_lut():
    """teal at the low end to yellow at the top, as encoded colors (like viridis)"""
    t = np.linspace(0, 1, 256)[:, None]
    return (1 - t) * np.array([0.1, 0.6, 0.55]) + t * np.array([1.0, 0.9, 0.1])


def test_mida_colors_stay_on_the_colormap(dev):
    """Every MIDA pixel is a true colormap color: rgb = LUT(alpha), as in MIP.
    (Blending colormapped samples instead produced hues between entries, e.g. an
    olive from teal + yellow, that are nowhere on the colormap.)"""
    vol, _ = _sparse_scene(seed=7)
    n = 32
    vol = np.ascontiguousarray(vol[:, :n, :n])
    lut = _two_color_lut()
    s = Scene(dev, vol, np.zeros(vol.shape, np.uint8), lut_rgb=lut)
    for yaw, pitch in ((0.0, 0.0), (0.5, 0.3)):
        inv, *_ = _ortho(yaw, pitch, n * 0.6)
        out = s.compute(_uniform(inv, (n, n, vol.shape[0]), 3, density=1.0, show_lab=0), 3, 1, 0, n, n)
        v = out[..., 3]
        f = np.clip(v, 0, 1) * 255
        i0 = np.floor(f).astype(int); i1 = np.minimum(i0 + 1, 255); fr = (f - i0)[..., None]
        expect = lut[i0] * (1 - fr) + lut[i1] * fr
        hit = v > 1e-3                           # rays that miss the volume output nothing
        assert hit.mean() > 0.5
        np.testing.assert_allclose(out[..., :3][hit], expect[hit], atol=4e-3)


def test_mida_keeps_hdr_colors(dev):
    """An HDR colormap (encoded values above 1) reaches the output unchanged: a
    thick uniform region at the data's top accumulates to nearly 1 and shows
    the lifted color there, above 1, as in MIP."""
    n, NZ = 32, 40
    vol = np.full((NZ, n, n), 1.0, np.float16)
    lut = _two_color_lut() * 1.6                 # lifted: the top entry is (1.6, 1.44, 0.16)
    s = Scene(dev, vol, np.zeros(vol.shape, np.uint8), lut_rgb=lut)
    inv, *_ = _ortho(0.0, 0.0, n / 2)
    out = s.compute(_uniform(inv, (n, n, NZ), 3, density=1.0, show_lab=0), 3, 1, 0, n, n)
    v = out[..., 3]
    assert v.min() > 0.99
    f = v * 255; i0 = np.floor(f).astype(int); fr = (f - i0)[..., None]
    expect = lut[i0] * (1 - fr) + lut[np.minimum(i0 + 1, 255)] * fr
    np.testing.assert_allclose(out[..., :3], expect, rtol=4e-3)
    assert out[..., 0].min() > 1.5


@pytest.mark.parametrize("view", [(0.35, 0.25), (0.8, 0.5)])
def test_mida_no_lines_at_voxel_edges(dev, view):
    """Random voxels, tilted, 8 pixels per voxel: no lines along voxel edges. A
    ray that clips a voxel's corner must fade only in proportion to its path
    through it (a full fade on any touch drew lines along voxel edges). Counted
    as one-pixel extremes forming vertical runs, since crisp cube corners alone
    give isolated one-pixel extremes."""
    n, W = 24, 192
    vol = np.random.default_rng(1).uniform(0.15, 1.0, (n, n, n)).astype(np.float16)
    s = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    inv, *_ = _ortho(view[0], view[1], n * 0.45)
    img = s.compute(_uniform(inv, (n, n, n), 3, density=0.5, show_lab=0), 3, 1, 0, W, W)[..., 3]
    c, l, r = img[:, 1:-1], img[:, :-2], img[:, 2:]
    spk = (c < np.minimum(l, r) - 0.03) | (c > np.maximum(l, r) + 0.03)
    lines = (spk[1:-1] & spk[:-2] & spk[2:]).mean()   # one-pixel extremes in vertical runs of 3+
    assert lines < 0.002                              # the full-fade rule: 0.26-0.28%; crisp cube corners alone ~0.1%


@pytest.mark.parametrize("mode", [0, 3])
def test_voxel_faces_make_a_voxel_flat(dev, mode):
    """One bright voxel seen corner-on: with voxel faces every ray through it
    collects the same amount, so it is a flat hexagon (EA's path-length shading
    peaks in the center with creases toward the corners)."""
    one = np.zeros((5, 5, 5), np.float16); one[2, 2, 2] = 1.0
    s = Scene(dev, one, np.zeros(one.shape, np.uint8))
    inv, *_ = _ortho(math.radians(45), math.asin(1 / math.sqrt(3)), 1.3)
    u = _uniform(inv, (5, 5, 5), mode, density=0.5 if mode == 3 else 0.0, show_lab=0)
    img = s.compute(u, mode, 1, 0, 256, 256, faces=1)[..., 3]
    inside = img[img > 0.05]
    assert inside.size > 5000 and inside.max() - np.percentile(inside, 2) < 0.01


def test_voxel_faces_step_between_voxels(dev):
    """Two neighboring voxels of different value, seen at an angle: with voxel
    faces MIDA shows flat levels with crisp steps (one voxel, the other, and a
    flat band where rays cross both), where exact path lengths ramp through
    every value in between."""
    two = np.zeros((3, 3, 4), np.float16); two[1, 1, 1] = 0.3; two[1, 1, 2] = 0.9
    s = Scene(dev, two, np.zeros(two.shape, np.uint8))
    inv, *_ = _ortho(0.6, 0.35, 1.6)
    u = _uniform(inv, (4, 3, 3), 3, density=0.5, show_lab=0)
    def levels(img):           # occupied value bins across the image's own range
        v = img[img > 0.01]
        h, _ = np.histogram(v, bins=24, range=(0, v.max() * 1.001))
        return int((h > 20).sum())
    faces = s.compute(u, 3, 1, 0, 256, 256, faces=1)[..., 3]
    exact = s.compute(u, 3, 1, 0, 256, 256)[..., 3]
    assert levels(exact) >= 12                 # a ramp through the in-between values (23 bins)
    assert levels(faces) <= 3                  # flat levels: one voxel, the other, the band crossing both


@pytest.mark.parametrize("mode", [0, 3])
def test_voxel_shading_blends_path_length_and_faces(dev, mode):
    """t = 0 is exact path length, t = 1 voxel faces, t = 0.5 in between; and for
    t < 1 the rim of a voxel still fades to nothing (no lines at its edges)."""
    one = np.full((5, 5, 5), 0.15, np.float16); one[2, 2, 2] = 1.0
    s = Scene(dev, one, np.zeros(one.shape, np.uint8))
    inv, *_ = _ortho(math.radians(45), math.asin(1 / math.sqrt(3)), 1.3)
    u = _uniform(inv, (5, 5, 5), mode, density=0.5 if mode == 3 else 0.0, show_lab=0, exposure=0.3)
    img = {t: s.compute(u, mode, 1, 0, 256, 256, faces=t)[..., 3] for t in (0.0, 0.5, 1.0)}
    np.testing.assert_array_equal(s.compute(u, mode, 1, 0, 256, 256)[..., 3], img[0.0])   # default = path length
    c = slice(120, 136)
    mid = img[0.5][c, c].mean()
    assert min(img[0.0][c, c].mean(), img[1.0][c, c].mean()) - 1e-3 <= mid <= max(img[0.0][c, c].mean(), img[1.0][c, c].mean()) + 1e-3
    step = {t: float(np.abs(np.diff(img[t][128])).max()) for t in (0.5, 1.0)}
    assert step[0.5] < 0.25 * step[1.0]          # the rim rises steeply but continuously for t < 1 (a hard step at t = 1)


def test_voxel_faces_keep_mida_even_and_leave_mip_alone(dev):
    """A uniform block seen straight on (every ray crosses the same thickness)
    renders evenly with path lengths and with voxel faces; MIP ignores the
    voxel shading setting."""
    flat = np.full((8, 8, 8), 0.6, np.float16)
    s = Scene(dev, flat, np.zeros(flat.shape, np.uint8))
    inv, *_ = _ortho(0.0, 0.0, 3.0)
    u = _uniform(inv, (8, 8, 8), 3, density=0.5, show_lab=0)
    for faces in (0, 1):
        img = s.compute(u, 3, 1, 0, 128, 128, faces=faces)[..., 3]
        v = img[img > 0.05]
        assert v.max() - v.min() < 2e-3
    vol, lab = _sparse_scene(seed=2)
    s2 = Scene(dev, vol, lab)
    NZ, NY, NX = vol.shape
    inv, *_ = _ortho(0.5, 0.3, max(NX, NY, NZ) * 0.8)
    u = _uniform(inv, (NX, NY, NZ), 1, show_lab=0)
    np.testing.assert_array_equal(s2.compute(u, 1, 1, 0, 96, 96, faces=1), s2.compute(u, 1, 1, 0, 96, 96))


def test_mida_hides_boundaries_between_equal_voxels(dev):
    """A bright block right behind a dim one, seen at an angle: where both are on
    the ray, boundaries between the bright block's equal voxels must stay
    invisible, as in EA (the old per-voxel fade drew a grid there: 21% of pixels
    were creases, now ~5%)."""
    n = 12
    vol = np.full((n, n, n), 0.1, np.float16)
    vol[3:9, 3:9, 2:6] = 0.45
    vol[3:9, 3:9, 6:10] = 0.9
    only = vol.copy(); only[3:9, 3:9, 2:6] = 0.1
    inv, *_ = _ortho(1.35, 0.3, 6.0)
    u = _uniform(inv, (n, n, n), 3, density=0.5, show_lab=0)
    img = Scene(dev, vol, np.zeros(vol.shape, np.uint8)).compute(u, 3, 1, 0, 384, 384)[..., 3]
    alone = Scene(dev, only, np.zeros(only.shape, np.uint8)).compute(u, 3, 1, 0, 384, 384)[..., 3]
    both = (np.abs(img - alone) > 0.02) & (alone > 0.3)
    lap = np.abs(img[1:-1, 2:] + img[1:-1, :-2] + img[2:, 1:-1] + img[:-2, 1:-1] - 4 * img[1:-1, 1:-1])
    assert both.sum() > 10000
    assert (lap[both[1:-1, 1:-1]] > 0.004).mean() < 0.08


def test_mida_shows_a_bright_voxel_behind_dim_ones(dev):
    """The point of MIDA: a bright voxel behind a dense dim slab still shows (as in
    MIP), where emission-absorption at the same density hides it."""
    n, NZ = 32, 24
    vol = np.zeros((NZ, n, n), np.float16)
    vol[2:12] = 0.3                        # dense dim slab in front
    vol[18, 8:24, 8:24] = 1.0              # bright square behind it
    s = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    inv, *_ = _ortho(0.0, 0.0, n / 2)
    mida = s.compute(_uniform(inv, (n, n, NZ), 3, density=2.0, show_lab=0), 3, 1, 0, n, n)[..., 0]
    ea = s.compute(_uniform(inv, (n, n, NZ), 0, density=2.0, show_lab=0, exposure=1.0), 0, 1, 0, n, n)[..., 0]
    inside, outside = mida[12:20, 12:20].mean(), mida[2:6, 2:6].mean()
    assert inside > outside + 0.1                # 0.45 vs 0.30 (the slab alone)
    ea_in, ea_out = ea[12:20, 12:20].mean(), ea[2:6, 2:6].mean()
    assert ea_in - ea_out < 0.1 * (inside - outside)   # EA: the slab hides the square


@pytest.mark.parametrize("mode", [0, 3])
def test_window_acts_like_a_lut_on_ea_and_mida(dev, mode):
    """EA and MIDA project the data at its full range; the display window and
    gamma then act on the result, as in MIP. So a window of [0, 0.5] gives
    exactly min(2 x the full-range value, 1): a full-range window never clips,
    and narrowing it pops what projects above its top to the colormap's top.
    (Windowing each voxel first let MIDA's averaging keep saturated structures
    below the top however narrow the window.)"""
    vol, _ = _sparse_scene(seed=9)
    n = 32
    vol = np.ascontiguousarray(vol[:, :n, :n])
    s = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    inv, *_ = _ortho(0.4, 0.3, n * 0.6)
    dims = (n, n, vol.shape[0])
    def render(window, gamma=1.0):
        u = _uniform(inv, dims, mode, density=0.5, show_lab=0, window=window, exposure=0.1)
        u[39] = gamma
        return s.compute(u, mode, 1, 0, n, n)[..., 3]
    full = render((0.0, 1.0))
    assert full.max() < 0.999                              # a full-range window: nothing at the top
    np.testing.assert_allclose(render((0.0, 0.5)), np.minimum(full * 2, 1), atol=4e-3)
    top = 0.6 * float(full.max())                          # a window whose top is below the brightest pixel
    narrow = render((0.0, top))
    np.testing.assert_allclose(narrow, np.minimum(full / top, 1), atol=4e-3)
    assert (narrow > 0.999).mean() > 0.005                 # ...pops those pixels to the top
    np.testing.assert_allclose(render((0.0, 1.0), gamma=2.0), full ** 2, atol=4e-3)


def test_display_window_matches_numpy(dev):
    """The 2D histogram window reaches 3D: MIP is the windowed column max, and
    an emission-absorption floor above the data makes it fully transparent."""
    vol, lab = _sparse_scene()
    vol, lab = np.ascontiguousarray(vol[:, :32, :32]), np.ascontiguousarray(lab[:, :32, :32])
    NZ, n = vol.shape[0], 32
    s = Scene(dev, vol, lab)
    inv, *_ = _ortho(0.0, 0.0, n / 2)
    lo, hi = 0.2, 0.6
    colmax = vol.astype(np.float32)[:, ::-1, :].max(axis=0)
    out = s.compute(_uniform(inv, (n, n, NZ), 1, show_lab=0, window=(lo, hi)), 1, 1, 0, n, n)
    np.testing.assert_allclose(out[..., 3], np.clip((colmax - lo) / (hi - lo), 0, 1), atol=3e-3)
    floor = float(vol.max()) + 0.01                      # everything below the window
    out = s.compute(_uniform(inv, (n, n, NZ), 0, density=5.0, show_lab=0, window=(floor, floor + 0.1)),
                    0, 1, 0, n, n)
    assert out[..., 3].max() == 0.0


def test_transparent_low_end(dev):
    """Colormap alpha (the transparent-low-end option) reaches the render: MIP
    alpha is value x alpha(value) with premultiplied color, and a colormap that
    is fully transparent makes emission-absorption draw nothing at all."""
    vol, lab = _sparse_scene()
    vol, lab = np.ascontiguousarray(vol[:, :32, :32]), np.ascontiguousarray(lab[:, :32, :32])
    NZ, n = vol.shape[0], 32
    ramp = np.linspace(0, 1, 256, dtype=np.float32)
    s = Scene(dev, vol, lab, lut_alpha=ramp)             # alpha = lightness (gray ramp)
    inv, *_ = _ortho(0.0, 0.0, n / 2)
    colmax = vol.astype(np.float32)[:, ::-1, :].max(axis=0)
    off = s.compute(_uniform(inv, (n, n, NZ), 1, show_lab=0), 1, 1, 0, n, n)          # option off: alpha ignored
    np.testing.assert_allclose(off[..., 3], colmax, atol=3e-3)
    out = s.compute(_uniform(inv, (n, n, NZ), 1, show_lab=0), 1, 1, 0, n, n, transp=1)
    np.testing.assert_allclose(out[..., 3], colmax * colmax, atol=4e-3)
    np.testing.assert_allclose(out[..., 0], colmax * colmax, atol=4e-3)   # gray color x alpha
    clear = Scene(dev, vol, lab, lut_alpha=np.zeros(256))
    out = clear.compute(_uniform(inv, (n, n, NZ), 0, density=5.0, show_lab=0), 0, 1, 0, n, n, transp=1)
    assert out[..., 3].max() == 0.0 and out[..., :3].max() == 0.0


@pytest.mark.parametrize("density", [0.0, 0.3])
def test_emission_absorption_glow_matches_analytic(dev, density):
    """EA color through a constant cube (gray colormap, so color(s) = s): the glow
    is c*s*(1 - e^-tau)/(density*s) per unit (pure c*s*chord at density 0, where
    nothing absorbs), shaded by the soft exposure 1 - e^(-k*acc)."""
    N, W, s_val, k = 48, 128, 0.5, 0.08
    sc = Scene(dev, np.full((N, N, N), s_val, np.float16), np.zeros((N, N, N), np.uint8))
    half = N * 0.95
    inv, d, right, up = _ortho(0.6, 0.4, half)
    out = sc.compute(_uniform(inv, (N, N, N), 0, density=density, show_lab=0, exposure=k), 0, 1, 0, W, W)
    ys, xs = np.mgrid[0:W, 0:W]
    ndx, ndy = (xs + 0.5) / W * 2 - 1, 1 - (ys + 0.5) / W * 2
    ro = (ndx[..., None] * right + ndy[..., None] * up) * half - d * BIG
    with np.errstate(divide="ignore"):
        iv = 1.0 / np.where(d == 0, 1e-12, d)
    t1, t2 = (-N / 2 - ro) * iv, (N / 2 - ro) * iv
    chord = np.clip(np.maximum(t1, t2).min(-1) - np.minimum(t1, t2).max(-1), 0, None)
    if density == 0:
        acc = s_val * s_val * chord
    else:
        acc = s_val * (1 - np.exp(-density * s_val * chord)) / density
    exact = 1 - np.exp(-k * acc)
    inside = chord > 1.0
    tol = 0.006 if density == 0 else 0.01     # (with absorption: early termination at 99.5%)
    assert np.abs(out[..., 0][inside] - exact[inside]).max() < tol


def test_emission_absorption_keeps_hue(dev):
    """A summed saturated color keeps its hue under the exposure curve: with a
    yellow colormap (1, 1, 0.1) the output's blue/red ratio stays 0.1 however
    bright the glow gets, rather than drifting to white."""
    N, W = 48, 64
    sc = Scene(dev, np.full((N, N, N), 1.0, np.float16), np.zeros((N, N, N), np.uint8))
    yellow = np.tile(np.array([1.0, 1.0, 0.1, 1.0], np.float16), (256, 1))
    sc.lut = _tex(dev, "rgba16float", (256, 1, 1), yellow.tobytes(), 256 * 8, dim="2d")
    inv, *_ = _ortho(0.5, 0.3, N * 0.95)
    out = sc.compute(_uniform(inv, (N, N, N), 0, density=0.0, show_lab=0, exposure=5.0), 0, 1, 0, W, W)
    lit = out[..., 0] > 0.5
    assert lit.sum() > 100
    np.testing.assert_allclose(out[..., 2][lit] / out[..., 0][lit], 0.1, atol=0.01)


# ── window per voxel (CLASSIFY) ─────────────────────────────────────────────

def _classify_scene(dev):
    """A bright background (0.5) holding a 6^3 cube of 0.9, like inverted phase contrast."""
    n = 20
    vol = np.full((n, n, n), 0.5, np.float16)
    vol[7:13, 7:13, 7:13] = 0.9
    return vol, Scene(dev, vol, np.zeros(vol.shape, np.uint8))


@pytest.mark.parametrize("mode", [0, 3])
@pytest.mark.parametrize("faces", [0.0, 1.0])
def test_classify_background_below_window_is_empty(dev, mode, faces):
    """Below the window's low end a voxel is empty space: at an angle, pixels that
    only cross background are exactly black, while without CLASSIFY the summed
    background shows."""
    vol, sc = _classify_scene(dev)
    inv, *_ = _ortho(0.6, 0.4, 16.0)
    u = _uniform(inv, (20, 20, 20), mode, density=0.5, show_lab=0, window=(0.6, 1.0), exposure=0.05)
    on = sc.compute(u, mode, 1, 0, 128, 128, faces=faces, classify=1)[..., 3]
    ref = sc.compute(_uniform(inv, (20, 20, 20), 1, show_lab=0, window=(0.6, 1.0)), 1, 1, 0, 128, 128)[..., 3]
    inside = ref > 0.05                                  # rays that cross the cube (MIP sees it)
    assert inside.mean() > 0.05
    assert on[~inside].max() == 0.0
    assert on[inside].mean() > 0.1


@pytest.mark.parametrize("mode", [0, 3])
def test_classify_solid_voxel_shows_its_windowed_value(dev, mode):
    """At full density the voxel in front is opaque and shows its windowed value,
    the same as MIP does for the cube."""
    vol, sc = _classify_scene(dev)
    inv, *_ = _ortho(0.6, 0.4, 16.0)
    u = _uniform(inv, (20, 20, 20), mode, density=1.0, show_lab=0, window=(0.6, 1.0))
    on = sc.compute(u, mode, 1, 0, 128, 128, faces=1.0, classify=1)[..., 3]
    mip = sc.compute(_uniform(inv, (20, 20, 20), 1, show_lab=0, window=(0.6, 1.0)), 1, 1, 0, 128, 128)[..., 3]
    core = mip > 0.7                                     # full cube coverage (not its rim pixels)
    expect = (0.9 - 0.6) / 0.4
    assert core.mean() > 0.05
    assert abs(float(np.median(on[core])) - expect) < 0.02


@pytest.mark.parametrize("mode", [0, 3])
def test_classify_brick_skipping_changes_nothing(dev, mode):
    """Bricks at or below the low end are skipped; the output matches marching
    every voxel (brick maxima forced to 1 so nothing is skipped)."""
    rng = np.random.default_rng(3)
    vol = (rng.random((40, 36, 44)) * 0.5).astype(np.float16)     # below the window everywhere...
    vol[20:30, 4:14, 25:40] = (0.7 + 0.3 * rng.random((10, 10, 15))).astype(np.float16)   # ...but here
    sc = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    full = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    bz, by, bx = [-(-d // BRICK) for d in vol.shape]
    full.bimg = _tex(dev, "r16float", (bx, by, bz), np.ones((bz, by, bx), np.float16).tobytes(), bx * 2)
    inv, *_ = _ortho(0.8, 0.35, 30.0)
    u = _uniform(inv, (44, 36, 40), mode, density=0.4, show_lab=0, window=(0.55, 1.0))
    a = sc.compute(u, mode, 1, 0, 128, 128, faces=0.5, classify=1)[..., 3]
    b = full.compute(u, mode, 1, 0, 128, 128, faces=0.5, classify=1)[..., 3]
    assert a.max() > 0.2
    assert np.abs(a - b).max() < 2e-3


# ── surface (MODE 4) ────────────────────────────────────────────────────────

def _surface_cube(dev, n=20, lo=7, hi=13, value=0.9, background=0.5):
    vol = np.full((n, n, n), background, np.float16)
    vol[lo:hi, lo:hi, lo:hi] = value
    return Scene(dev, vol, np.zeros(vol.shape, np.uint8))


@pytest.mark.parametrize("density", [0.0, 0.3, 1.0])
def test_surface_uniform_block_is_one_flat_surface(dev, density):
    """A uniform 6^3 block lights up once, at its outer surface: every pixel it
    covers shows its windowed value exactly (no lines where voxels share a face),
    whatever the density, and the background below the window is empty."""
    sc = _surface_cube(dev)
    inv, *_ = _ortho(0.6, 0.4, 16.0)
    u = _uniform(inv, (20, 20, 20), 4, density=density, show_lab=0, window=(0.6, 1.0))
    out = sc.compute(u, 4, 1, 0, 128, 128, faces=0.0)[..., 3]
    mip = sc.compute(_uniform(inv, (20, 20, 20), 1, show_lab=0, window=(0.6, 1.0)), 1, 1, 0, 128, 128)[..., 3]
    core, empty = mip > 0.7, mip == 0.0
    expect = (0.9 - 0.6) / 0.4
    assert core.mean() > 0.05 and empty.mean() > 0.3
    assert np.abs(out[core] - expect).max() < 2e-3
    assert out[empty].max() == 0.0


def test_surface_cosine_lighting_shades_each_face(dev):
    """With lighting 1 a cube shows three flat shades, its value times |ray . normal|
    for the three faces toward the camera."""
    sc = _surface_cube(dev, value=1.0, background=0.0)
    inv, d, *_ = _ortho(0.6, 0.4, 16.0)
    u = _uniform(inv, (20, 20, 20), 4, density=1.0, show_lab=0, window=(0.0, 1.0))
    out = sc.compute(u, 4, 1, 0, 256, 256, faces=1.0)[..., 3]
    vals = out[out > 0.05]
    cos = np.sort(np.abs(d))
    for c in cos:
        share = np.mean(np.abs(vals - c) < 3e-3)
        assert share > 0.1, (c, share)                      # each face is a flat patch of its cosine
    assert np.mean(np.min(np.abs(vals[:, None] - cos[None]), 1) < 3e-3) > 0.97


def test_surface_brick_skipping_changes_nothing(dev):
    rng = np.random.default_rng(5)
    vol = (rng.random((40, 36, 44)) * 0.5).astype(np.float16)
    vol[20:30, 4:14, 25:40] = (0.7 + 0.3 * rng.random((10, 10, 15))).astype(np.float16)
    sc = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    full = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    bz, by, bx = [-(-d // BRICK) for d in vol.shape]
    full.bimg = _tex(dev, "r16float", (bx, by, bz), np.ones((bz, by, bx), np.float16).tobytes(), bx * 2)
    inv, *_ = _ortho(0.8, 0.35, 30.0)
    u = _uniform(inv, (44, 36, 40), 4, density=0.3, show_lab=0, window=(0.55, 1.0))
    a = sc.compute(u, 4, 1, 0, 128, 128, faces=1.0)[..., 3]
    b = full.compute(u, 4, 1, 0, 128, 128, faces=1.0)[..., 3]
    assert a.max() > 0.2
    assert np.abs(a - b).max() < 2e-3


def _haze_scene(dev, depth, n=40, haze=0.66, cube=0.9):
    """A 6^3 cube near the back of the box, behind `depth` voxels of uniform haze
    (windowed 0.2 with the window 0.6..1.0), empty elsewhere; viewed along z."""
    vol = np.zeros((n, n, n), np.float16)
    vol[n - 8 - depth:n - 8, 10:30, 10:30] = haze
    vol[n - 8:n - 2, 17:23, 17:23] = cube
    return Scene(dev, vol, np.zeros(vol.shape, np.uint8))


@pytest.mark.parametrize("density", [0.0, 0.3])
def test_surface_through_haze_does_not_depend_on_its_depth(dev, density):
    """Dim voxels in the way emit and block only by their peak: a cube behind 3 or
    25 voxels of uniform haze shows the same value, h + (c - h) e^(-absorb h)
    (lighting 0), instead of fogging or darkening with depth."""
    inv, *_ = _ortho(0.0, 0.0, 22.0)                    # straight along z, the haze in front
    outs = []
    for depth in (3, 25):
        u = _uniform(inv, (40, 40, 40), 4, density=density, show_lab=0, window=(0.6, 1.0))
        outs.append(_haze_scene(dev, depth).compute(u, 4, 1, 0, 128, 128, faces=0.0)[..., 3])
    h, c = (0.66 - 0.6) / 0.4, (0.9 - 0.6) / 0.4
    absorb = 2.0 ** (6 * density) - 1.0
    expect = h + (c - h) * np.exp(-absorb * h)
    mid = (slice(58, 70), slice(58, 70))                 # pixels over the cube's middle
    for out in outs:
        assert np.abs(out[mid] - expect).max() < 3e-3, (out[mid].min(), out[mid].max(), expect)


def test_surface_is_mip_without_lighting_or_density(dev):
    rng = np.random.default_rng(9)
    vol = rng.random((24, 28, 32)).astype(np.float16)
    sc = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    inv, *_ = _ortho(0.7, 0.3, 24.0)
    surf = sc.compute(_uniform(inv, (32, 28, 24), 4, density=0.0, show_lab=0, window=(0.3, 0.9)), 4, 1, 0, 128, 128, faces=0.0)
    mip = sc.compute(_uniform(inv, (32, 28, 24), 1, show_lab=0, window=(0.3, 0.9)), 1, 1, 0, 128, 128)
    assert np.abs(surf[..., 3] - mip[..., 3]).max() < 2e-3
