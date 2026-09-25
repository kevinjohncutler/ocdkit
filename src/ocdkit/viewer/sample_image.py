"""Default-sample-image helpers and uint8 normalization utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from imageio import v2 as imageio

ArrayLike = np.ndarray

INSTANCE_COLOR_TABLE = np.array(
    [
        (0, 0, 0, 0),
        (255, 0, 0, 120),
        (0, 180, 255, 120),
        (0, 200, 0, 120),
        (255, 109, 0, 120),
        (255, 0, 200, 120),
        (150, 255, 0, 120),
        (255, 216, 0, 120),
        (137, 0, 255, 120),
        (0, 255, 164, 120),
    ],
    dtype=np.uint8,
)

DEFAULT_BRUSH_RADIUS = 1

# Search order for the preload sample (env var → user dir → none).
# Plugins can extend the search by setting OCDKIT_VIEWER_SAMPLE_IMAGE before
# launching the viewer.
DEFAULT_HOME_SAMPLE = Path.home() / ".ocdkit" / "sample.tif"


def get_preload_image_path() -> Optional[Path]:
    """Return the preferred sample image path if one exists.

    ``OCDKIT_VIEWER_SAMPLE_IMAGE=3d`` selects the built-in 3D sample
    (:func:`sample_volume_path`); any other value is a file path.
    """
    env = os.environ.get("OCDKIT_VIEWER_SAMPLE_IMAGE")
    if env and env.strip().lower() == "3d":
        return sample_volume_path()
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path
    if DEFAULT_HOME_SAMPLE.is_file():
        return DEFAULT_HOME_SAMPLE
    return None


def _load_raw() -> ArrayLike:
    path = get_preload_image_path()
    if path is not None:
        try:
            return imageio.imread(path)
        except FileNotFoundError:
            pass
    return _generate_fallback_image()


def _generate_fallback_image() -> ArrayLike:
    grid_x, grid_y = np.meshgrid(
        np.linspace(-1.0, 1.0, 256, dtype=np.float32),
        np.linspace(-1.0, 1.0, 256, dtype=np.float32),
        indexing="xy",
    )
    radius = np.sqrt(grid_x**2 + grid_y**2)
    rings = np.sin(8 * radius) * np.exp(-radius**2)
    blobs = np.exp(-((grid_x + 0.4) ** 2 + (grid_y - 0.3) ** 2) * 6.5)
    gradient = np.clip((rings + blobs + 1.0) * 0.5, 0.0, 1.0)
    return (gradient * 255.0).astype(np.uint8)


# ── built-in 3D sample ─────────────────────────────────────────────────────────
# A small synthetic widefield fluorescence stack (rod-shaped cells plus
# sub-resolution beads) with a matching label volume, so the 3D view has
# something to show without any data on disk. Sampled isotropically so the
# optical blur looks right in 3D rather than squashed along z.

SAMPLE_VOLUME_VERSION = 1
_VOXEL_UM = 0.1                                   # isotropic voxel size (microns)


def widefield_psf(shape=(64, 64, 64), voxel_um=_VOXEL_UM, wavelength_um=0.52,
                  na=1.1, n_medium=1.33, pad=4) -> ArrayLike:
    """Scalar-diffraction widefield PSF, ``(Z, Y, X)``, normalized to sum 1.

    Each z plane is the squared magnitude of the inverse Fourier transform of a
    circular pupil (cutoff ``na / wavelength``) times the defocus phase
    ``exp(2 pi i kz z)``, giving the characteristic hourglass with side lobes.
    Computed on a ``pad``-times wider lateral grid and cropped, because the
    defocused light cone outgrows the crop and would otherwise wrap around.
    """
    nz, cy, cx = shape
    ny, nx = pad * cy, pad * cx
    ky = np.fft.fftfreq(ny, d=voxel_um)
    kx = np.fft.fftfreq(nx, d=voxel_um)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    pupil = k2 <= (na / wavelength_um) ** 2
    kz = np.sqrt(np.maximum((n_medium / wavelength_um) ** 2 - k2, 0.0))
    z = (np.arange(nz) - nz // 2) * voxel_um
    field = np.fft.ifft2(pupil[None] * np.exp(2j * np.pi * kz[None] * z[:, None, None]), axes=(1, 2))
    psf = np.fft.fftshift(np.abs(field) ** 2, axes=(1, 2))
    y0, x0 = (ny - cy) // 2, (nx - cx) // 2
    psf = psf[:, y0:y0 + cy, x0:x0 + cx]
    return (psf / psf.sum()).astype(np.float32)


def _capsule(zyx_um, a, b, radius):
    """Boolean mask of points within ``radius`` of segment ``a``-``b`` (microns)."""
    ab = b - a
    t = np.clip(((zyx_um - a) @ ab) / (ab @ ab), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(zyx_um - closest, axis=-1) <= radius


def generate_sample_volume(shape=(64, 128, 128), n_cells=6, n_beads=4,
                           seed=0) -> tuple[ArrayLike, ArrayLike]:
    """Synthetic 3D widefield stack and its labels, both ``(Z, Y, X)``.

    Returns ``(image uint8, labels uint8)``: rod-shaped cells (labels 1..n,
    uniformly filled, like a cytoplasmic marker) and unlabeled point beads
    (which render as the bare PSF), blurred by :func:`widefield_psf`, over a
    dim background with shot noise. Deterministic for a given ``seed``.
    """
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    grid = np.stack(np.meshgrid(*(np.arange(s) * _VOXEL_UM for s in shape), indexing="ij"), axis=-1)
    extent = np.array(shape) * _VOXEL_UM
    labels = np.zeros(shape, np.uint8)
    lab = 0
    for _ in range(200):                               # rejection-sample non-touching rods
        if lab == n_cells:
            break
        length, radius = rng.uniform(1.6, 3.0), 0.4
        center = rng.uniform(extent * [0.35, 0.2, 0.2], extent * [0.65, 0.8, 0.8])
        theta, phi = rng.uniform(-0.6, 0.6), rng.uniform(0, np.pi)   # tilt out of the xy plane, azimuth
        axis = np.array([np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)])
        a, b = center - axis * length / 2, center + axis * length / 2
        m = _capsule(grid, a, b, radius)
        grown = _capsule(grid, a, b, radius + 0.3)     # keep a gap between cells
        if not m.any() or (labels[grown] > 0).any():
            continue
        lab += 1
        labels[m] = lab
    psf = widefield_psf(shape=(nz, 96, 96))
    blurred = fftconvolve((labels > 0).astype(np.float32), psf, mode="same")
    beads = np.zeros(shape, np.float32)
    for _ in range(n_beads):
        z, y, x = (int(rng.uniform(0.25, 0.75) * s) for s in shape)
        beads[z, y, x] = 0.8 * blurred.max() / psf.max()          # bead peak ~ 0.8x the brightest cell
    blurred = (blurred + fftconvolve(beads, psf, mode="same")).clip(0, None)
    photons = 400.0 * (blurred / blurred.max()) + 20.0             # signal + background
    noisy = rng.poisson(photons).astype(np.float32)
    image = _normalize_uint8(noisy)
    return image, labels


def sample_volume_path() -> Path:
    """Path of the built-in 3D sample TIFF, written with its ``_masks`` sidecar
    on first use (under ``~/.ocdkit/samples``), so the viewer loads it through
    the ordinary open-a-volume path."""
    import tifffile

    d = Path.home() / ".ocdkit" / "samples"
    img = d / f"sample3d_v{SAMPLE_VOLUME_VERSION}.tif"
    msk = d / f"sample3d_v{SAMPLE_VOLUME_VERSION}_masks.tif"
    if not (img.is_file() and msk.is_file()):
        d.mkdir(parents=True, exist_ok=True)
        image, labels = generate_sample_volume()
        tifffile.imwrite(msk, labels)
        tifffile.imwrite(img, image)
    return img


def _ensure_spatial_last(array: ArrayLike) -> ArrayLike:
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        return np.moveaxis(array, 0, -1)
    return array


def _normalize_uint8(array: ArrayLike) -> ArrayLike:
    array = np.asarray(array)
    array = array.astype(np.float32)
    array -= array.min()
    maxv = array.max()
    if maxv > 0:
        array /= maxv
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def load_image_uint8(as_rgb: bool = False) -> ArrayLike:
    """Load the default sample image and normalize it to uint8."""
    data = _load_raw()
    data = _ensure_spatial_last(data)
    data = _normalize_uint8(data)
    if as_rgb:
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=-1)
        elif data.ndim == 3 and data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
    return data


def apply_gamma(image_uint8: ArrayLike, gamma: float) -> ArrayLike:
    """Gamma correction on a uint8 image, preserving dtype."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    arr = np.asarray(image_uint8).astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0) ** gamma
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def get_instance_color_table() -> ArrayLike:
    """Return the RGBA lookup table for instance mask visualization."""
    return INSTANCE_COLOR_TABLE.copy()
