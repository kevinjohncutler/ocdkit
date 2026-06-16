"""Tests for ocdkit.wgpu — GPU plotting primitives (density lines, scatter).

Split into two tiers:

- **CPU-logic** tests run anywhere the ``wgpu`` package imports (no GPU
  adapter needed): API surface, interval tiling, the alpha-ramped colormap
  LUT and its ``lut_fn`` override seam, and the ``render_spectral_rgb``
  signature decoupling (takes precomputed CIE ``xyz_weights``, not
  wavelengths — keeps the module spectroscopy-agnostic).
- **GPU** tests are gated on a real wgpu adapter and skipped otherwise (e.g.
  headless CI runners with no Vulkan/Metal). They exercise the actual
  rasterizers and guard the buffer-size clamp that keeps allocations within
  the adapter's limits (the fix for Vulkan's 256 MB maxBufferSize).
"""
import numpy as np
import pytest

# The whole module needs the wgpu package importable (ocdkit.wgpu imports it
# at module load). Skip cleanly where it isn't installed.
wgpu = pytest.importorskip("wgpu")


def _have_adapter():
    try:
        return wgpu.gpu.request_adapter_sync(power_preference="high-performance") is not None
    except Exception:
        return False


gpu = pytest.mark.skipif(not _have_adapter(), reason="no wgpu adapter available")


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_public_api_importable():
    import ocdkit.wgpu as ow
    for name in (
        "DensityLineRenderer", "rasterize_spectra", "render_spectral_rgb",
        "render_scatter_gpu", "render_scatter_by", "render_scatter_value",
        "infer_channel_intervals", "get_device", "prewarm",
    ):
        assert hasattr(ow, name), f"missing public symbol: {name}"


def test_no_hostpkg_coupling_imports():
    """The moved module must not re-introduce domain coupling — guards the
    public-repo decoupling (no hostpkg colormap / wavelength imports)."""
    import ocdkit.wgpu.lines as lines
    src = __import__("inspect").getsource(lines)
    assert "from ..colormap" not in src
    assert "wavelengths_to_xyz" not in src


# ---------------------------------------------------------------------------
# infer_channel_intervals — pure logic
# ---------------------------------------------------------------------------

def test_infer_channel_intervals():
    from ocdkit.wgpu import infer_channel_intervals
    assert infer_channel_intervals(0, [3, 3]) == []
    assert infer_channel_intervals(6, [3, 3]) == [(0, 3), (3, 6)]
    # under-tiling widths get a trailing remainder interval
    assert infer_channel_intervals(7, [3, 3]) == [(0, 3), (3, 6), (6, 7)]
    # no widths -> a single full-span interval
    assert infer_channel_intervals(5, []) == [(0, 5)]
    # widths overrun num_points -> clamped, no empty intervals
    assert infer_channel_intervals(4, [3, 3]) == [(0, 3), (3, 4)]


# ---------------------------------------------------------------------------
# colormap LUT + the lut_fn decoupling seam
# ---------------------------------------------------------------------------

def test_transparent_lut_shape_and_alpha_ramp():
    from ocdkit.wgpu.lines import _transparent_lut
    lut = _transparent_lut("magma")
    assert lut.shape == (256, 4)
    # alpha 0 where data==0, ramps up, never exceeds 1
    assert lut[0, 3] == pytest.approx(0.0, abs=1e-6)
    assert lut[:, 3].max() <= 1.0 + 1e-6
    assert lut[-1, 3] > lut[0, 3]


def test_get_lut_rgba_uint8_and_override():
    from ocdkit.wgpu.lines import _get_lut_rgba
    lut = _get_lut_rgba("viridis")
    assert lut.shape == (256, 4)
    assert lut.dtype == np.uint8
    # lut_fn override is honored — the decoupling seam for domain callers
    sentinel = np.zeros((256, 4), dtype=float)
    sentinel[:, 0] = 1.0  # all-red, full opacity ignored
    out = _get_lut_rgba("viridis", lut_fn=lambda name: sentinel)
    assert (out[:, 0] == 255).all()
    assert (out[:, 1] == 0).all()


def test_render_spectral_rgb_signature_is_decoupled():
    """render_spectral_rgb must take precomputed CIE xyz_weights, NOT
    wavelengths — the spectroscopy math stays in the caller."""
    import inspect
    from ocdkit.wgpu import render_spectral_rgb
    params = list(inspect.signature(render_spectral_rgb).parameters)
    assert "xyz_weights" in params
    assert "wavelengths" not in params


# ---------------------------------------------------------------------------
# GPU rasterizers (gated on a real adapter)
# ---------------------------------------------------------------------------

@gpu
def test_rasterize_spectra_shape_and_deposit():
    from ocdkit.wgpu import rasterize_spectra
    N, P = 24, 16
    # flat lines across the middle band -> non-zero density there
    y = np.tile(np.linspace(0.4, 0.6, P), (N, 1)).astype(np.float32)
    dens = rasterize_spectra(
        y, plot_width=80, plot_height=40,
        x_range=(0.0, 1.0), y_range=(0.0, 1.0),
        line_width=2.0, x_coords=np.linspace(0.0, 1.0, P),
    )
    assert dens.shape == (40, 80)
    assert dens.dtype == np.float32
    assert dens.sum() > 0


@gpu
def test_density_renderer_buffer_clamp_within_limits():
    """The x/y value buffers must fit the adapter's limits — guards the
    fix for adapters (e.g. Vulkan) that cap maxBufferSize below 400 MB."""
    from ocdkit.wgpu.lines import DensityLineRenderer
    r = DensityLineRenderer(64, 64)
    lim = r.device.limits
    cap = min(int(lim["max-buffer-size"]),
              int(lim["max-storage-buffer-binding-size"]))
    assert r.x_buffer.size <= cap
    assert r.y_buffer.size <= cap
    assert r.x_buffer.size == r.y_buffer.size


@gpu
def test_render_scatter_gpu_rgba():
    from ocdkit.wgpu import render_scatter_gpu
    rng = np.random.default_rng(0)
    x = rng.random(500).astype(np.float32)
    y = rng.random(500).astype(np.float32)
    img = render_scatter_gpu(
        x, y, plot_width=64, plot_height=48,
        x_range=(0.0, 1.0), y_range=(0.0, 1.0), cmap_name="viridis",
    )
    assert img.shape == (48, 64, 4)
    assert img.dtype == np.uint8
    assert img[..., :3].sum() > 0
