"""CIE 1931 chromaticity diagram with Display-P3-accurate fill.

- Transparent background, gray axes / labels (dark-mode rcParams convention).
- Region inside Display-P3 gamut is colored with the actual P3 RGB triplet
  for each chromaticity (max-normalized per pixel). On a P3 display, colors
  *between* the sRGB and P3 triangles render with full P3 saturation; sRGB
  displays narrow them to their gamut.
- Everything outside the P3 triangle is left transparent — those
  chromaticities cannot be displayed on a P3 monitor.

Outputs:
- figures/chromaticity_p3.svg — JXL raster tagged display-p3; best in Safari.
- figures/chromaticity_p3.png — flat PNG with an embedded Display P3 ICC
  profile (iCCP chunk). PowerPoint / Keynote / Preview honor the profile
  and render the wide gamut on a P3 display.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ocdkit.plot.svg import SVG, jxl_data_url  # noqa: E402


# Linear Display-P3 ← XYZ (D65) — same matrix used in ocdkit.plot.hdr_cmap
_P3_FROM_XYZ = np.array([
    [ 2.4934969119, -0.9313836179, -0.4027107845],
    [-0.8294889696,  1.7626640603,  0.0236246858],
    [ 0.0358458302, -0.0761723893,  0.9568845240],
])

# Gamut primaries (xy chromaticities, D65)
SRGB_PRIMARIES = np.array([(0.6400, 0.3300), (0.3000, 0.6000), (0.1500, 0.0600)])
P3_PRIMARIES   = np.array([(0.6800, 0.3200), (0.2650, 0.6900), (0.1500, 0.0600)])
D65            = (0.3127, 0.3290)

# CIE 1931 2° observer color-matching functions (10 nm samples).
_CMF = np.array([
    [380, 0.001368, 0.000039, 0.006450],
    [390, 0.004243, 0.000120, 0.020050],
    [400, 0.014310, 0.000396, 0.067850],
    [410, 0.043510, 0.001210, 0.207400],
    [420, 0.134380, 0.004000, 0.645600],
    [430, 0.283900, 0.011600, 1.385600],
    [440, 0.348280, 0.023000, 1.747060],
    [450, 0.336200, 0.038000, 1.772110],
    [460, 0.290800, 0.060000, 1.669200],
    [470, 0.195360, 0.090980, 1.287640],
    [480, 0.095640, 0.139020, 0.812950],
    [490, 0.032010, 0.208020, 0.465180],
    [500, 0.004900, 0.323000, 0.272000],
    [510, 0.009300, 0.503000, 0.158200],
    [520, 0.063270, 0.710000, 0.078250],
    [530, 0.165500, 0.862000, 0.042160],
    [540, 0.290400, 0.954000, 0.020300],
    [550, 0.433450, 0.994950, 0.008750],
    [560, 0.594500, 0.995000, 0.003900],
    [570, 0.762100, 0.952000, 0.002100],
    [580, 0.916300, 0.870000, 0.001650],
    [590, 1.026300, 0.757000, 0.001100],
    [600, 1.062200, 0.631000, 0.000800],
    [610, 1.002600, 0.503000, 0.000340],
    [620, 0.854450, 0.381000, 0.000190],
    [630, 0.642400, 0.265000, 0.000050],
    [640, 0.447900, 0.175000, 0.000020],
    [650, 0.283500, 0.107000, 0.000000],
    [660, 0.164900, 0.061000, 0.000000],
    [670, 0.087400, 0.032000, 0.000000],
    [680, 0.046770, 0.017000, 0.000000],
    [690, 0.022700, 0.008210, 0.000000],
    [700, 0.011360, 0.004102, 0.000000],
    [710, 0.005790, 0.002091, 0.000000],
    [720, 0.002899, 0.001047, 0.000000],
    [730, 0.001440, 0.000520, 0.000000],
])
def _densify_locus(cmf: np.ndarray, step_nm: float = 1.0):
    """Upsample the 10 nm CMF to a fine spectral-locus polyline.

    A cubic spline on each tristimulus channel keeps the locus visually
    smooth at the sharp 380-440 nm bend and along the 460-540 nm shoulder
    where 10 nm sampling looks polygonal. xy is computed AFTER the spline
    so the locus stays consistent with the densified CMF.
    """
    from scipy.interpolate import CubicSpline

    nm_dense = np.arange(cmf[0, 0], cmf[-1, 0] + step_nm * 0.5, step_nm)
    xyz_dense = np.column_stack([
        CubicSpline(cmf[:, 0], cmf[:, k])(nm_dense)
        for k in (1, 2, 3)
    ])
    xyz_dense = np.maximum(xyz_dense, 0.0)
    s = xyz_dense.sum(axis=1)
    return nm_dense, xyz_dense[:, 0] / s, xyz_dense[:, 1] / s


LOCUS_NM, LOCUS_X, LOCUS_Y = _densify_locus(_CMF, step_nm=1.0)


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    """Apply the sRGB / Display-P3 transfer curve (linear → encoded)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(np.maximum(x, 1e-12), 1.0 / 2.4) - 0.055,
    )


def make_chromaticity_raster(res: int, x_max: float, y_max: float) -> np.ndarray:
    """(res, res, 4) uint8 RGBA — top-left origin, y_max at top.

    For each pixel (chromaticity x, y) we set Y=1, compute XYZ → linear-P3,
    and (if all components are ≥ 0) normalize so the max channel = 1 — the
    most saturated P3 color at that chromaticity. sRGB OETF for encoding,
    then quantize to uint8. Pixels outside the P3 triangle get alpha = 0.
    """
    xs = np.linspace(0.0, x_max, res)
    ys = np.linspace(y_max, 0.0, res)
    XX, YY = np.meshgrid(xs, ys)
    safe_y = np.where(YY > 1e-4, YY, 1.0)
    X = XX / safe_y
    Y = np.ones_like(XX)
    Z = (1.0 - XX - YY) / safe_y
    XYZ = np.stack([X, Y, Z], axis=-1)
    lin = XYZ @ _P3_FROM_XYZ.T
    # Soft alpha at the gamut boundary: 1 deep inside, ramps to 0 just past
    # the boundary where the most-negative channel exceeds ``feather``.
    feather = 0.004
    neg = -np.minimum(lin.min(axis=-1), 0.0)
    alpha = np.clip(1.0 - neg / feather, 0.0, 1.0)
    alpha[YY <= 1e-3] = 0.0
    # Per-pixel max normalize the (clipped-positive) linear-P3 values.
    lin_pos = np.clip(lin, 0.0, None)
    mx = lin_pos.max(axis=-1, keepdims=True)
    mx_safe = np.where(mx > 1e-9, mx, 1.0)
    rgb = srgb_oetf(lin_pos / mx_safe)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return (rgba * 255.0).round().clip(0, 255).astype(np.uint8)


X_MAX, Y_MAX = 0.8, 0.9


def build_svg(raster: np.ndarray) -> str:
    PLOT_W, PLOT_H = 640, 720
    M_L, M_R, M_T, M_B = 70, 40, 50, 60
    W = PLOT_W + M_L + M_R
    H = PLOT_H + M_T + M_B
    svg = SVG(W, H)

    def xpx(x: float) -> float:
        return M_L + (x / X_MAX) * PLOT_W

    def ypx(y: float) -> float:
        return M_T + PLOT_H - (y / Y_MAX) * PLOT_H

    # --- 1. Display-P3-tagged raster of the colored gamut --------------
    raster_url = jxl_data_url(raster, color="display-p3")
    svg.add(
        f'<image x="{xpx(0):.2f}" y="{ypx(Y_MAX):.2f}" '
        f'width="{PLOT_W:.2f}" height="{PLOT_H:.2f}" '
        f'href="{raster_url}" preserveAspectRatio="none" '
        f'image-rendering="auto"/>'
    )

    # --- 2. spectral locus + line of purples ---------------------------
    locus_pts = list(zip(LOCUS_X, LOCUS_Y))
    d_locus = "M " + " L ".join(f"{xpx(x):.2f},{ypx(y):.2f}" for x, y in locus_pts)
    svg.path(d_locus, stroke="gray", stroke_width=1.0, fill="none")
    # line of purples (last sample → first sample), dashed
    x0, y0 = locus_pts[0]
    x1, y1 = locus_pts[-1]
    svg.line(xpx(x1), ypx(y1), xpx(x0), ypx(y0),
             stroke="gray", stroke_width=1.0, dasharray="4,3")

    # --- 3. wavelength tick marks + labels -----------------------------
    # Tick at every entry; label only the well-separated ones (the locus
    # bunches near 460-480 nm and 510-520 nm, so we'd otherwise overlap).
    tick_nm  = [400, 460, 470, 480, 490, 500, 510, 520,
                540, 560, 580, 600, 620, 700]
    label_nm = {400, 460, 480, 500, 520, 540, 560, 580, 600, 620, 700}
    for nm in tick_nm:
        idx = int(np.argmin(np.abs(LOCUS_NM - nm)))
        lx, ly = LOCUS_X[idx], LOCUS_Y[idx]
        dx, dy = lx - D65[0], ly - D65[1]
        n = float(np.hypot(dx, dy)) or 1.0
        ux, uy = dx / n, dy / n
        x0p, y0p = xpx(lx), ypx(ly)
        svg.line(x0p, y0p,
                 x0p + ux * 6.0, y0p - uy * 6.0,
                 stroke="gray", stroke_width=1)
        if nm in label_nm:
            svg.text(x0p + ux * 24.0, y0p - uy * 24.0,
                     str(nm), size=10, fill="gray",
                     anchor="middle", baseline="middle")

    # --- 4. gamut triangles --------------------------------------------
    tri = lambda P: [(xpx(x), ypx(y)) for x, y in P]
    # sRGB: solid stroke
    svg.polygon(tri(SRGB_PRIMARIES), fill="none",
                stroke="gray", stroke_width=1.6)
    # P3: dashed stroke (so the two outlines never alias to the same line)
    p3_d = ("M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in tri(P3_PRIMARIES))
            + " Z")
    svg.path(p3_d, stroke="gray", stroke_width=1.6, fill="none",
             dasharray="6,3")

    # --- 5. D65 white point --------------------------------------------
    svg.add(
        f'<circle cx="{xpx(D65[0]):.2f}" cy="{ypx(D65[1]):.2f}" '
        f'r="3" fill="gray" stroke="none"/>'
    )
    svg.text(xpx(D65[0]) + 6, ypx(D65[1]) - 6, "D65",
             size=10, fill="gray", anchor="start")

    # --- 6. axes -------------------------------------------------------
    svg.line(xpx(0), ypx(0), xpx(X_MAX), ypx(0),
             stroke="gray", stroke_width=1)
    svg.line(xpx(0), ypx(0), xpx(0), ypx(Y_MAX),
             stroke="gray", stroke_width=1)
    for tx in np.arange(0.0, X_MAX + 1e-9, 0.1):
        px = xpx(tx)
        svg.line(px, ypx(0), px, ypx(0) + 5,
                 stroke="gray", stroke_width=1)
        svg.text(px, ypx(0) + 18, f"{tx:.1f}",
                 size=10, fill="gray", anchor="middle")
    for ty in np.arange(0.0, Y_MAX + 1e-9, 0.1):
        py = ypx(ty)
        svg.line(xpx(0) - 5, py, xpx(0), py,
                 stroke="gray", stroke_width=1)
        svg.text(xpx(0) - 8, py + 4, f"{ty:.1f}",
                 size=10, fill="gray", anchor="end")
    svg.text((xpx(0) + xpx(X_MAX)) / 2, ypx(0) + 38, "x",
             size=14, fill="gray", anchor="middle",
             baseline="middle")
    svg.text(xpx(0) - 42, (ypx(0) + ypx(Y_MAX)) / 2, "y",
             size=14, fill="gray", anchor="middle",
             baseline="middle")

    # --- 7. legend (gamut key) -----------------------------------------
    lg_x = xpx(0.50)
    lg_y = ypx(0.85)
    svg.line(lg_x, lg_y, lg_x + 32, lg_y,
             stroke="gray", stroke_width=1.6)
    svg.text(lg_x + 38, lg_y + 4, "sRGB",
             size=11, fill="gray", anchor="start")
    svg.line(lg_x, lg_y + 18, lg_x + 32, lg_y + 18,
             stroke="gray", stroke_width=1.6,
             dasharray="6,3")
    svg.text(lg_x + 38, lg_y + 22, "Display P3",
             size=11, fill="gray", anchor="start")

    # title
    svg.text(W / 2, 24,
             "CIE 1931 chromaticity — Display-P3 fill",
             size=13, fill="gray",
             anchor="middle", baseline="middle")

    return svg.finalize()


def build_png_rgba(raster: np.ndarray, fig_in: float = 8.0,
                    dpi: int = 300) -> np.ndarray:
    """Render the figure to an (H, W, 4) uint8 RGBA array via matplotlib Agg.

    The chromaticity raster is laid into the axes via ``imshow`` as raw
    uint8 — matplotlib + Agg do not color-transform pre-encoded RGBA, so
    the Display-P3-encoded bytes pass through to the final composite. The
    gray overlays (axes, ticks, labels, gamut triangles, locus) are
    neutral, so their byte values are identical under sRGB and P3
    interpretation — the composite remains a valid P3 buffer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor":  "none",
        "axes.facecolor":    "none",
        "savefig.facecolor": "none",
        "axes.edgecolor":    "gray",
        "axes.labelcolor":   "gray",
        "xtick.color":       "gray",
        "ytick.color":       "gray",
        "text.color":        "gray",
        "axes.titlecolor":   "gray",
    })

    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)

    # 1. Display-P3-encoded raster — interpolation='bilinear' is fine
    #    because blending neighboring P3-encoded samples on a smooth
    #    chromaticity gradient just gives the in-between P3 value.
    ax.imshow(raster, extent=(0.0, X_MAX, 0.0, Y_MAX), origin="upper",
              interpolation="bilinear", zorder=1)

    # 2. spectral locus
    ax.plot(LOCUS_X, LOCUS_Y, color="gray", lw=1.0, zorder=2)
    # line of purples (last locus point → first), dashed
    ax.plot([LOCUS_X[-1], LOCUS_X[0]], [LOCUS_Y[-1], LOCUS_Y[0]],
            color="gray", lw=1.0, dashes=(4, 3), zorder=2)

    # 3. wavelength ticks + labels
    tick_nm  = [400, 460, 470, 480, 490, 500, 510, 520,
                540, 560, 580, 600, 620, 700]
    label_nm = {400, 460, 480, 500, 520, 540, 560, 580, 600, 620, 700}
    tick_pt = 5     # tick length, screen points
    label_pt = 14   # label offset, screen points
    for nm in tick_nm:
        idx = int(np.argmin(np.abs(LOCUS_NM - nm)))
        lx, ly = float(LOCUS_X[idx]), float(LOCUS_Y[idx])
        dx, dy = lx - D65[0], ly - D65[1]
        n = float(np.hypot(dx, dy)) or 1.0
        ux, uy = dx / n, dy / n
        # tick: use annotate with offset transform so length is in points
        ax.annotate("", xy=(lx, ly), xycoords="data",
                    xytext=(ux * tick_pt, uy * tick_pt),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=1.0))
        if nm in label_nm:
            ax.annotate(str(nm), xy=(lx, ly),
                        xytext=(ux * label_pt, uy * label_pt),
                        textcoords="offset points",
                        color="gray", fontsize=8,
                        ha="center", va="center")

    # 4. gamut triangles
    def plot_tri(P, **kw):
        Pc = np.vstack([P, P[:1]])
        ax.plot(Pc[:, 0], Pc[:, 1], **kw)

    plot_tri(SRGB_PRIMARIES, color="gray", lw=1.6, label="sRGB", zorder=3)
    plot_tri(P3_PRIMARIES,   color="gray", lw=1.6,
             dashes=(6, 3), label="Display P3", zorder=3)

    # 5. white point
    ax.plot(D65[0], D65[1], "o", color="gray", markersize=4, zorder=4)
    ax.annotate("D65", xy=D65, xytext=(6, 6),
                textcoords="offset points", color="gray", fontsize=8)

    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(0.0, Y_MAX)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("CIE 1931 chromaticity — Display-P3 fill")
    leg = ax.legend(loc="upper right", frameon=False)
    for txt in leg.get_texts():
        txt.set_color("gray")

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def encode_png_display_p3(rgba: np.ndarray) -> bytes:
    """Encode RGBA → PNG bytes with an embedded Display P3 ICC profile.

    The ``iCCP`` chunk is what tells PowerPoint / Keynote / Preview etc.
    that the pixel values are Display-P3-encoded — without it they fall
    back to sRGB and the wide-gamut extension between sRGB and P3 gets
    narrowed.
    """
    from opencodecs._png_codec import PngCodec
    icc_path = Path("/System/Library/ColorSync/Profiles/Display P3.icc")
    icc_bytes = icc_path.read_bytes()
    return PngCodec().encode(
        np.ascontiguousarray(rgba),
        iccprofile=icc_bytes,
        iccprofile_name="Display P3",
    )


def main() -> None:
    figures = REPO / "figures"
    figures.mkdir(exist_ok=True)

    raster = make_chromaticity_raster(res=2048, x_max=X_MAX, y_max=Y_MAX)

    svg_out = figures / "chromaticity_p3.svg"
    svg_out.write_text(build_svg(raster))
    print(f"wrote {svg_out}")

    png_out = figures / "chromaticity_p3.png"
    rgba = build_png_rgba(raster)
    png_out.write_bytes(encode_png_display_p3(rgba))
    print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
