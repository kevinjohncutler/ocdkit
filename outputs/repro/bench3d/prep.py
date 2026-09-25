"""Prepare benchmark volumes for the 3D renderer A/B harness.

Writes, per dataset, the EXACT arrays the shipped viewer uploads to the GPU:
  vol_f16.bin   intensity normalized (a-lo)/(hi-lo) as float32, then float16
                (same math as volume3d-gpu.js _uploadTextures + _toF16)
  grp_u8.bin    ncolor group volume (what the viewer's label texture holds)
  lab.bin       raw label ids, narrowed to u8/u16/u32 (for the per-label LUT variant)
  grp_lut.bin   uint32 label -> group map (so the LUT variant can reproduce colors)
  meta.json     shape (NX, NY, NZ), dtypes, max label

Usage: BENCH3D_SPACETIME_DIR=<stacks dir> prep.py OUT_DIR
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

# directory holding <name>.tif + <name>_masks.tif spacetime stacks
SPACETIME = Path(os.environ.get("BENCH3D_SPACETIME_DIR", "."))


def narrow(a):
    m = int(a.max())
    return a.astype(np.uint8 if m <= 0xFF else np.uint16 if m <= 0xFFFF else np.uint32)


def load(name):
    if name == "cells3d":
        from scipy import ndimage
        from skimage import data, filters
        img = data.cells3d()[:, 1].astype(np.float32)          # nuclei channel (dense tissue)
        sm = ndimage.gaussian_filter(img, 1.0)
        lab, _ = ndimage.label(sm > filters.threshold_otsu(sm))
        return img, lab
    img = tifffile.imread(SPACETIME / f"{name}.tif")
    lab = tifffile.imread(SPACETIME / f"{name}_masks.tif")
    z = min(img.shape[0], lab.shape[0])                           # dnaA: 135 vs 133 frames
    return img[:z], lab[:z]


def main(out):
    out = Path(out)
    import ncolor
    for name in ["dnaA_xy1", "5I", "ftsN_xy1", "cells3d"]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        if (d / "meta.json").exists():
            print("skip", name)
            continue
        img, lab = load(name)
        NZ, NY, NX = img.shape
        a = img.astype(np.float64)
        lo, hi = float(a.min()), float(a.max())
        f = ((a - lo) * (1.0 / (hi - lo))).astype(np.float32)     # JS: (a[i]-lo)*sc into Float32Array
        f.astype(np.float16).tofile(d / "vol_f16.bin")
        lab = narrow(lab)
        g = np.asarray(ncolor.label(lab))
        lut = np.zeros(int(lab.max()) + 1, np.uint32)
        nz = lab > 0
        lut[lab[nz]] = g[nz]
        grp = lut[lab].astype(np.uint8)
        grp.tofile(d / "grp_u8.bin")
        lab.tofile(d / "lab.bin")
        lut.tofile(d / "grp_lut.bin")
        # the float64 source, for the transport benchmark (what the server holds today)
        np.save(d / "src_img.npy", img)
        meta = dict(NX=NX, NY=NY, NZ=NZ, lab_dtype=str(lab.dtype), max_label=int(lab.max()),
                    n_groups=int(grp.max()), src_dtype=str(img.dtype),
                    label_fraction=float(nz.mean()),
                    intensity_median=float(np.median(f)), intensity_p99=float(np.percentile(f, 99)))
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(name, meta)


if __name__ == "__main__":
    main(sys.argv[1])
