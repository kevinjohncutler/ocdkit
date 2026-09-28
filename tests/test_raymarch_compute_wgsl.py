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
             exposure=1.0, peak=1.0):
    NX, NY, NZ = dims
    u = np.zeros(48, np.float32)
    u[0:16] = inv
    u[20:24] = [-NX / 2, -NY / 2, -NZ / 2, 0]
    u[24:28] = [NX / 2, NY / 2, NZ / 2, 0]
    u[28:32] = [NX, NY, NZ, mode]
    u[32:36] = [2 * max(dims), density, opacity, show_lab]
    u[36:40] = [1.0, show_img, 1.0, 1.0]
    u[40:44] = [0.4, 0.0, 24.0, 1.0]
    u[44:48] = [window[0], 1.0 / (window[1] - window[0]), exposure, peak]
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

    def compute_pipeline(self, mode, show_img, show_lab, shade=1, transp=0, smooth=0):
        return self.dev.create_compute_pipeline(layout="auto", compute={
            "module": self.cmod, "entry_point": "cs",
            "constants": {"MODE": mode, "SHOW_IMG": show_img, "SHOW_LAB": show_lab, "SHADE_LAB": shade,
                          "BRICK": float(BRICK), "TRANSP": transp, "SMOOTH": smooth}})

    def compute(self, u, mode, show_img, show_lab, W, H, transp=0, smooth=0):
        p = self.compute_pipeline(mode, show_img, show_lab, transp=transp, smooth=smooth)
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


def _mida_reference(cols, density):
    """MIDA along axis 0 (front first), one unit of path per voxel, on intensity:
    I = beta I + (1 - beta A) a s, A = beta A + (1 - beta A) a, beta = 1 - max(s - f, 0);
    the displayed value is v = I / A (then colormapped once, alpha v, like MIP)."""
    I = np.zeros(cols.shape[1:]); A = np.zeros(cols.shape[1:]); f = np.zeros(cols.shape[1:])
    for s in cols.astype(np.float64):
        a = 1 - np.exp(-density * s)
        beta = 1 - np.maximum(s - f, 0)
        keep = beta * A
        I = beta * I + (1 - keep) * a * s
        A = keep + (1 - keep) * a
        f = np.maximum(f, s)
    return np.where(A > 1e-6, np.clip(I / np.maximum(A, 1e-6), 0, 1), 0)


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
    """An HDR colormap (encoded values above 1) reaches the output unchanged: the
    top of the colormap shows its full lifted color, as in MIP."""
    n, NZ = 32, 40
    vol = np.full((NZ, n, n), 1.0, np.float16)
    lut = _two_color_lut() * 1.6                 # lifted: the top entry is (1.6, 1.44, 0.16)
    s = Scene(dev, vol, np.zeros(vol.shape, np.uint8), lut_rgb=lut)
    inv, *_ = _ortho(0.0, 0.0, n / 2)
    out = s.compute(_uniform(inv, (n, n, NZ), 3, density=1.0, show_lab=0), 3, 1, 0, n, n)
    np.testing.assert_allclose(out[..., 3], 1.0, atol=2e-3)
    np.testing.assert_allclose(out[..., :3], np.broadcast_to(lut[255], out[..., :3].shape), rtol=4e-3)


@pytest.mark.parametrize("view", [(0.35, 0.25), (0.8, 0.5)])
def test_mida_no_lines_at_voxel_edges(dev, view):
    """Random voxels, tilted, 8 pixels per voxel: no one-pixel lines. A ray that
    clips a voxel's corner must fade only in proportion to its path through it
    (a full fade on any touch drew lines along voxel edges: ~0.8% of pixels)."""
    n, W = 24, 192
    vol = np.random.default_rng(1).uniform(0.15, 1.0, (n, n, n)).astype(np.float16)
    s = Scene(dev, vol, np.zeros(vol.shape, np.uint8))
    inv, *_ = _ortho(view[0], view[1], n * 0.45)
    img = s.compute(_uniform(inv, (n, n, n), 3, density=0.5, show_lab=0), 3, 1, 0, W, W)[..., 3]
    c, l, r = img[:, 1:-1], img[:, :-2], img[:, 2:]
    spikes = ((c < np.minimum(l, r) - 0.03) | (c > np.maximum(l, r) + 0.03)).mean()
    assert spikes < 0.001


@pytest.mark.parametrize("mode", [0, 3])
def test_rounded_voxels_keep_even_regions_even(dev, mode):
    """Rounded voxel edges blend only near faces with weights that sum to 1, so a
    block of one value renders exactly as with sharp voxels (no grid)."""
    flat = np.full((8, 8, 8), 0.6, np.float16)
    s = Scene(dev, flat, np.zeros(flat.shape, np.uint8))
    for view in ((0.0, 0.0), (0.7, 0.5)):
        inv, *_ = _ortho(view[0], view[1], 5.0)
        u = _uniform(inv, (8, 8, 8), mode, density=0.5, show_lab=0, exposure=0.15)
        sharp = s.compute(u, mode, 1, 0, 128, 128)[..., 3]
        smooth = s.compute(u, mode, 1, 0, 128, 128, smooth=1)[..., 3]
        np.testing.assert_allclose(smooth, sharp, atol=3e-3)


def test_rounded_voxels_soften_the_creases(dev):
    """One bright voxel seen corner-on: with sharp cubes EA shows creases (slope
    breaks) along the projected edges; rounded edges soften them without adding
    rings from the sampling. Measured by the peak (99th percentile) of the
    discrete Laplacian, which lines like creases or rings dominate."""
    one = np.zeros((5, 5, 5), np.float16); one[2, 2, 2] = 1.0
    s = Scene(dev, one, np.zeros(one.shape, np.uint8))
    inv, *_ = _ortho(math.radians(45), math.asin(1 / math.sqrt(3)), 1.3)
    u = _uniform(inv, (5, 5, 5), 0, density=0.0, show_lab=0)
    def crease(img):                                   # peak |second difference| inside the voxel
        lap = np.abs(img[1:-1, 2:] + img[1:-1, :-2] + img[2:, 1:-1] + img[:-2, 1:-1] - 4 * img[1:-1, 1:-1])
        return float(np.percentile(lap[img[1:-1, 1:-1] > 0.3], 99))
    sharp = s.compute(u, 0, 1, 0, 256, 256)[..., 3]
    smooth = s.compute(u, 0, 1, 0, 256, 256, smooth=1)[..., 3]
    assert crease(smooth) < 0.85 * crease(sharp)       # 0.0068 vs 0.0088 (5 samples per voxel: 0.0125)
    # an isolated voxel shares a thin rim of its light with its empty neighbors: ~9% less at its center
    assert abs(smooth[125:131, 125:131].mean() - sharp[125:131, 125:131].mean()) < 0.1


def test_rounded_voxels_leave_mip_alone(dev):
    vol, lab = _sparse_scene(seed=2)
    s = Scene(dev, vol, lab)
    NZ, NY, NX = vol.shape
    inv, *_ = _ortho(0.5, 0.3, max(NX, NY, NZ) * 0.8)
    u = _uniform(inv, (NX, NY, NZ), 1, show_lab=0)
    np.testing.assert_array_equal(s.compute(u, 1, 1, 0, 96, 96, smooth=1), s.compute(u, 1, 1, 0, 96, 96))


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
    assert inside > 0.6 and inside > 2 * outside
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
