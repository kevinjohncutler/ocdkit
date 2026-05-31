"""HDR-lifted colormaps for the SVG/CSS-img image_grid pipeline.

Wraps any matplotlib / ``cmap`` colormap into a callable that maps a 2D
grayscale array to ``(H, W, 3) float32`` linear-light Display-P3 RGB.
Output flows straight into :func:`image_grid` — any float-dtype tile
triggers the existing ``jxl-hdr-pq`` encoder path, so values >1.0 land
on the HDR compositor with no additional wiring.

Lift algorithm — port of ``_liftToHdr`` from the colormap-explorer JS
prototype. For each cmap stop:

1. sRGB → linear sRGB → XYZ in absolute cd/m² (``Y=1.0 ≡ 203 nits``,
   BT.2408 SDR reference white).
2. XYZ → JzAzBz (perceptually uniform, PQ-based — handles HDR natively).
3. Uniform Jz scale so ``max(Jz)`` lands at ``hdr_jz`` (default 0.230 vs
   SDR's ≈0.155) — pulls the brightest stop into HDR brightness.
4. Vectorized binary-search of the largest *uniform* Cz scale that keeps
   every lifted stop inside HDR-P3 gamut at ``peak_mult × SDR-white``
   (default 7.88 ≈ 1600 nits, Apple Pro Display XDR peak), with a 5%
   safety margin.
5. JzAzBz → XYZ → linear Display-P3. Values may exceed 1.0 — that's the
   HDR headroom the PQ encoder downstream consumes.

Single Jz multiplier, single Cz multiplier, no per-hue distortion —
preserves the colormap's perceptual shape. Passing ``peak_mult=1.0``
yields a P3-widened SDR version (same gamut walk, no HDR).
"""

from __future__ import annotations

import numpy as np

# ─── reference levels ──────────────────────────────────────────────
SDR_WHITE_NITS = 203.0          # ITU-R BT.2408 — fixed JzAzBz reference
HDR_PEAK_NITS_DEFAULT = 1600.0  # Pro Display XDR; matches ocdkit svg.py
HDR_JZ_DEFAULT = 0.30           # lifts viridis yellow to ≈600 nits — visibly
                                # HDR but well short of the display peak.
                                # Push to 0.40+ for "wow" mode; 0.155 for SDR.

# ─── JzAzBz constants (Safdar et al. 2017) ─────────────────────────
# PQ here uses the MODIFIED m2 = 1.7 × 2523/32 (NOT the standard ST-2084
# 78.84375) — Safdar's tweak for perceptual uniformity.
_PQ_M1 = 0.1593017578125
_PQ_M2 = 134.034375
_PQ_C1 = 0.8359375
_PQ_C2 = 18.8515625
_PQ_C3 = 18.6875

_JZ_B = 1.15
_JZ_G = 0.66
_JZ_D = -0.56
_JZ_D0 = 1.6295499532821566e-11

# Modified Hunt-Pointer-Estevez (XYZ' → LMS)
_XYZ_TO_LMS = np.array([
    [ 0.41478972,  0.579999, 0.0146480],
    [-0.20151000,  1.120649, 0.0531008],
    [-0.01660080,  0.264800, 0.6684799],
])
_LMS_TO_XYZ = np.array([
    [ 1.9242264358, -1.0047923126, 0.0376514040],
    [ 0.3503167621,  0.7264811939, -0.0653844229],
    [-0.0909828110, -0.3127282905,  1.5227665613],
])

# LMS' (after PQ) → I, az, bz
_LMS_TO_IAB = np.array([
    [0.5,        0.5,        0.0],
    [3.524000,  -4.066708,   0.542708],
    [0.199076,   1.096799,  -1.295875],
])
_IAB_TO_LMS = np.linalg.inv(_LMS_TO_IAB)

# Linear Display-P3 ↔ XYZ (D65)
_P3_FROM_XYZ = np.array([
    [ 2.4934969119, -0.9313836179, -0.4027107845],
    [-0.8294889696,  1.7626640603,  0.0236246858],
    [ 0.0358458302, -0.0761723893,  0.9568845240],
])
# Linear sRGB → XYZ (D65)
_XYZ_FROM_SRGB = np.array([
    [0.4123907993, 0.3575843394, 0.1804807884],
    [0.2126390059, 0.7151686788, 0.0721923154],
    [0.0193308187, 0.1191947798, 0.9505321522],
])


# ─── PQ transfer ───────────────────────────────────────────────────
def _pq_forward(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    xp = np.power(x / 10000.0, _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * xp) / (1.0 + _PQ_C3 * xp), _PQ_M2)


def _pq_inverse(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    xp = np.power(x, 1.0 / _PQ_M2)
    num = np.maximum(xp - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * xp
    # den can go ≤0 for values above the PQ peak — clamp to 0 there
    safe = den > 0
    den_safe = np.where(safe, den, 1.0)
    out = 10000.0 * np.power(num / den_safe, 1.0 / _PQ_M1)
    return np.where(safe, out, 0.0)


# ─── sRGB transfer ─────────────────────────────────────────────────
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


# ─── JzAzBz ↔ XYZ ──────────────────────────────────────────────────
def xyz_to_jzazbz(XYZ: np.ndarray) -> np.ndarray:
    """XYZ (D65, absolute cd/m²) → JzAzBz. Input ``(..., 3)``."""
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    Xp = _JZ_B * X - (_JZ_B - 1) * Z
    Yp = _JZ_G * Y - (_JZ_G - 1) * X
    XYZm = np.stack([Xp, Yp, Z], axis=-1)
    LMS = XYZm @ _XYZ_TO_LMS.T
    LMSp = _pq_forward(LMS)
    IAB = LMSp @ _LMS_TO_IAB.T
    Iz = IAB[..., 0]
    Jz = (1.0 + _JZ_D) * Iz / (1.0 + _JZ_D * Iz) - _JZ_D0
    return np.stack([Jz, IAB[..., 1], IAB[..., 2]], axis=-1)


def jzazbz_to_xyz(Jab: np.ndarray) -> np.ndarray:
    """JzAzBz → XYZ (D65, absolute cd/m²). Input ``(..., 3)``."""
    Jz, az, bz = Jab[..., 0], Jab[..., 1], Jab[..., 2]
    Iz = (Jz + _JZ_D0) / (1.0 + _JZ_D - _JZ_D * (Jz + _JZ_D0))
    IAB = np.stack([Iz, az, bz], axis=-1)
    LMSp = IAB @ _IAB_TO_LMS.T
    LMS = _pq_inverse(LMSp)
    XYZm = LMS @ _LMS_TO_XYZ.T
    Xp, Yp, Z = XYZm[..., 0], XYZm[..., 1], XYZm[..., 2]
    X = (Xp + (_JZ_B - 1.0) * Z) / _JZ_B
    Y = (Yp + (_JZ_G - 1.0) * X) / _JZ_G
    return np.stack([X, Y, Z], axis=-1)


# ─── Gamut search ──────────────────────────────────────────────────
def _max_chroma_p3(Jz: np.ndarray, hz_deg: np.ndarray,
                   peak_mult: float, iters: int = 24) -> np.ndarray:
    """Vectorized binary-search of max Cz keeping ``(Jz, hz_deg)`` inside
    Display-P3 gamut. ``peak_mult`` is the gamut peak measured in units
    of SDR-white (e.g. 7.88 → 1600 nits)."""
    Jz = np.asarray(Jz, dtype=np.float64)
    hz_rad = np.radians(np.asarray(hz_deg, dtype=np.float64))
    cos_h, sin_h = np.cos(hz_rad), np.sin(hz_rad)
    lo = np.zeros_like(Jz)
    hi = np.full_like(Jz, 0.5)
    eps = 1e-4
    max_val = peak_mult + eps
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Jab = np.stack([Jz, mid * cos_h, mid * sin_h], axis=-1)
        rgb_p3 = jzazbz_to_xyz(Jab) @ _P3_FROM_XYZ.T / SDR_WHITE_NITS
        in_gamut = ((rgb_p3 >= -eps) & (rgb_p3 <= max_val)).all(axis=-1)
        lo = np.where(in_gamut, mid, lo)
        hi = np.where(in_gamut, hi, mid)
    return lo


# ─── Public API ────────────────────────────────────────────────────
def make_hdr_cmap_lut(
    cmap_input,
    *,
    n_stops: int = 256,
    hdr_jz: float = HDR_JZ_DEFAULT,
    hdr_peak_nits: float = HDR_PEAK_NITS_DEFAULT,
) -> np.ndarray:
    """Sample ``cmap_input`` at ``n_stops`` and lift to HDR Display-P3.

    Returns ``(n_stops, 3) float32`` linear-light Display-P3 normalized
    so ``1.0 ≡ hdr_peak_nits`` (matching the convention of
    :func:`ocdkit.plot.svg._p3_linear_to_pq_uint16`'s ``sdr_white_nits``
    argument). The downstream PQ-JXL encoder consumes the array as-is.

    The chroma cap binary-searches HDR-P3 gamut up to ``hdr_peak_nits``;
    the Jz lift pulls the brightest stop to ``hdr_jz`` (≈0.30 ⇒ ~600
    nits; 0.155 ⇒ SDR white; 0.40+ ⇒ aggressive HDR).

    ``cmap_input`` accepts a colormap name (e.g. ``'viridis'``), a
    :class:`cmap.Colormap`, a matplotlib :class:`Colormap`, or any
    callable mapping ``[0,1] → (...,4)`` RGBA.
    """
    cm = _coerce_cmap(cmap_input)
    t = np.linspace(0.0, 1.0, n_stops)
    rgba = np.asarray(cm(t))               # (N, 4) gamma-encoded sRGB
    if rgba.ndim == 1:                     # cmap quirk for single-pt
        rgba = rgba[None, :]
    srgb = rgba[..., :3]
    lin_srgb = _srgb_to_linear(srgb)
    XYZ = lin_srgb @ _XYZ_FROM_SRGB.T * SDR_WHITE_NITS
    Jab = xyz_to_jzazbz(XYZ)
    Jz = Jab[:, 0]
    az = Jab[:, 1]
    bz = Jab[:, 2]
    Cz = np.hypot(az, bz)
    hz_deg = (np.degrees(np.arctan2(bz, az)) + 360.0) % 360.0

    sdr_jz_max = float(np.max(Jz))
    jz_scale = (hdr_jz / sdr_jz_max) if sdr_jz_max > 1e-3 else 1.0
    new_jz = Jz * jz_scale

    peak_mult = float(hdr_peak_nits) / SDR_WHITE_NITS
    hdr_max_cz = _max_chroma_p3(new_jz, hz_deg, peak_mult)
    valid = Cz > 1e-3
    if valid.any():
        scales = (hdr_max_cz[valid] * 0.95) / Cz[valid]
        safe_scale = float(np.min(scales))
        if not np.isfinite(safe_scale) or safe_scale < 0.1:
            safe_scale = 1.0
    else:
        safe_scale = 1.0

    new_cz = Cz * safe_scale
    hz_rad = np.radians(hz_deg)
    Jab_new = np.stack(
        [new_jz, new_cz * np.cos(hz_rad), new_cz * np.sin(hz_rad)], axis=-1,
    )
    # Normalize so linear-1.0 ≡ hdr_peak_nits, matching the encoder.
    lin_p3 = jzazbz_to_xyz(Jab_new) @ _P3_FROM_XYZ.T / float(hdr_peak_nits)
    return np.maximum(lin_p3, 0.0).astype(np.float32)


class HdrCmapArray(np.ndarray):
    """``(H, W, 3) float32`` linear-light Display-P3 tile that also carries
    a pre-computed uint8 P3-curve SDR base for deterministic Ultra-HDR
    encoding.

    The Ultra-HDR pipeline in :mod:`ocdkit.plot.svg` will check
    ``arr._sdr_base_p3_u8`` and use it as the JPEG base layer instead
    of letting libuhdr auto-tonemap the HDR float (which visibly
    desaturates / hue-shifts cmap content).
    """

    def __new__(cls, hdr_arr, *, sdr_base_p3_u8=None):
        obj = np.asarray(hdr_arr).view(cls)
        obj._sdr_base_p3_u8 = sdr_base_p3_u8
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._sdr_base_p3_u8 = getattr(obj, '_sdr_base_p3_u8', None)


def apply_hdr_cmap(
    arr: np.ndarray,
    cmap_input,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    n_stops: int = 256,
    hdr_jz: float = HDR_JZ_DEFAULT,
    hdr_peak_nits: float = HDR_PEAK_NITS_DEFAULT,
    sdr_base: bool = True,
) -> np.ndarray:
    """Map a 2D grayscale array through an HDR-lifted colormap.

    Returns ``(H, W, 3) float32`` linear-light Display-P3 normalized to
    ``hdr_peak_nits`` — drop into :func:`ocdkit.plot.image_grid` and the
    existing HDR-PQ JXL encoder picks it up (default canvas
    ``sdr_white_nits=1600`` matches ``hdr_peak_nits=1600``).

    With ``sdr_base=True`` (default), the return value is a
    :class:`HdrCmapArray` carrying a pre-computed uint8 P3-curve SDR
    base derived from the *original* (non-lifted) colormap. The
    Ultra-HDR encoder in :mod:`ocdkit.plot.svg` reads that base via
    ``arr._sdr_base_p3_u8`` so SDR viewers (or HDR displays in SDR
    mode) see a bit-faithful rendition of the native cmap — not
    libuhdr's auto-tone-map of the lifted HDR float, which has
    per-channel rolloff that visibly desaturates bright stops.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"apply_hdr_cmap expects 2D, got shape {a.shape}")
    a = a.astype(np.float32, copy=False)

    if vmin is None:
        vmin = float(np.nanmin(a)) if a.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(a)) if a.size else 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-9

    t = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)

    lut = make_hdr_cmap_lut(
        cmap_input, n_stops=n_stops, hdr_jz=hdr_jz, hdr_peak_nits=hdr_peak_nits,
    )
    idx_f = t * (n_stops - 1)
    idx_lo = np.floor(idx_f).astype(np.int32)
    idx_hi = np.minimum(idx_lo + 1, n_stops - 1)
    frac = (idx_f - idx_lo)[..., None].astype(np.float32)
    hdr = (lut[idx_lo] * (1.0 - frac) + lut[idx_hi] * frac).astype(
        np.float32, copy=False,
    )

    if not sdr_base:
        return hdr

    sdr_u8 = _native_cmap_sdr_p3_u8(t, cmap_input, n_stops=n_stops)
    return HdrCmapArray(hdr, sdr_base_p3_u8=sdr_u8)


def _native_cmap_sdr_p3_u8(t_norm: np.ndarray, cmap_input,
                            *, n_stops: int = 256) -> np.ndarray:
    """Build the deterministic UHDR SDR-base layer: original cmap →
    P3-primaries uint8 with the sRGB OETF. ``t_norm`` is the same
    normalized [0, 1] index used to LUT-index the HDR tile.

    Samples ``cm(t_norm)`` directly on the full per-pixel index instead
    of nearest-neighbor lookups into an ``n_stops`` LUT — for smooth
    inputs (1024-wide gradient, 256-stop LUT) NN-LUT collapses runs of
    pixels into the same bin and the SDR base develops visible banding
    that diverges from the standalone jxl-p3 SDR tile (which always
    samples ``cm(t)`` directly). ``n_stops`` is retained for parity
    with the HDR lift's signature but is no longer used here.
    """
    del n_stops  # noqa: F841 -- direct sampling is parameter-free
    from .svg import _srgb_to_display_p3_uint8

    cm = _coerce_cmap(cmap_input)
    rgba = np.asarray(cm(t_norm))
    sdr_srgb_u8 = np.clip(rgba[..., :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return _srgb_to_display_p3_uint8(sdr_srgb_u8)


def _coerce_cmap(cmap_input):
    """Accept str / cmap.Colormap / mpl Colormap / callable → callable."""
    if isinstance(cmap_input, str):
        from cmap import Colormap
        return Colormap(cmap_input)
    if callable(cmap_input):
        return cmap_input
    from cmap import Colormap
    return Colormap(cmap_input)
