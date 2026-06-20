#!/usr/bin/env python
"""Empirically map omnipose's 3D core API + shapes on the spacetime masks.

Run: /Users/kcutler/.pyenv/shims/python explore_3d_core.py
"""
import inspect
import numpy as np
import tifffile

MASKS = "/Volumes/DataDrive/3D_spacetime/linked/a_baylii/dnaA_xy1_crop_masks.tif"
RAW = "/Volumes/DataDrive/3D_spacetime/linked/a_baylii/dnaA_xy1_crop.tif"
LINKS = "/Volumes/DataDrive/3D_spacetime/linked/a_baylii/dnaA_xy1_crop_links.txt"


def sig(fn):
    try:
        return str(inspect.signature(fn))
    except Exception as e:
        return f"<no sig: {e}>"


def main():
    import omnipose
    from omnipose import core, utils

    print("=== signatures ===")
    for name in ("masks_to_flows", "masks_to_flows_torch", "masks_to_affinity",
                 "spatial_affinity", "compute_masks", "affinity_to_masks"):
        fn = getattr(core, name, None)
        print(f"core.{name}{sig(fn) if fn else ' MISSING'}")
    for name in ("kernel_setup", "get_neighbors", "get_neigh_inds"):
        fn = getattr(utils, name, None)
        print(f"utils.{name}{sig(fn) if fn else ' MISSING'}")

    print("\n=== kernel_setup(2) vs (3) ===")
    for d in (2, 3):
        steps, inds, idx, fact, sign = utils.kernel_setup(d)
        steps = np.asarray(steps)
        print(f"dim={d}: steps.shape={steps.shape} center_idx={idx} "
              f"n_noncenter={steps.shape[0]-1}")

    print("\n=== load masks crop ===")
    m_full = tifffile.imread(MASKS)
    print("masks full:", m_full.shape, m_full.dtype, "labels:", np.unique(m_full).size)
    # small crop for speed: a few frames, spatial subset around content
    crop = m_full[:12, 120:200, 120:200].astype(np.int32)
    crop = np.ascontiguousarray(crop)
    print("crop:", crop.shape, "labels in crop:", np.unique(crop).size)

    print("\n=== masks_to_flows(dim=3) on crop ===")
    try:
        out = core.masks_to_flows(crop, dim=3, use_gpu=False)
        if isinstance(out, tuple):
            print("returned tuple of", len(out))
            for i, o in enumerate(out):
                shp = getattr(o, "shape", None)
                print(f"  [{i}] type={type(o).__name__} shape={shp} "
                      f"dtype={getattr(o,'dtype',None)}")
        else:
            print("returned single:", type(out).__name__, getattr(out, "shape", None))
    except Exception as e:
        import traceback; traceback.print_exc()

    print("\n=== masks_to_affinity(dim=3) on crop ===")
    try:
        steps, inds, idx, fact, sign = utils.kernel_setup(3)
        coords = np.nonzero(crop)
        aff = core.masks_to_affinity(crop, coords, steps, inds, idx, fact, sign, 3)
        print("affinity:", type(aff).__name__, getattr(aff, "shape", None),
              getattr(aff, "dtype", None))
        sp = core.spatial_affinity(aff, coords, crop.shape)
        print("spatial_affinity:", getattr(sp, "shape", None), getattr(sp, "dtype", None))
    except Exception as e:
        import traceback; traceback.print_exc()

    print("\n=== links.txt ===")
    edges = []
    with open(LINKS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a, b = line.replace(" ", "").split(",")
            edges.append((int(a), int(b)))
    print("n edges:", len(edges), "sample:", edges[:6])
    # which labels are parents (appear in col0), children (col1)
    parents = sorted({a for a, _ in edges})
    children = sorted({b for _, b in edges})
    print("parents:", parents[:10], "... n=", len(parents))
    print("children:", children[:10], "... n=", len(children))

    print("\n=== per-frame label presence (first 12 frames) ===")
    for z in range(min(12, m_full.shape[0])):
        labs = np.unique(m_full[z])
        labs = labs[labs > 0]
        print(f"  z={z}: {labs.tolist()}")


if __name__ == "__main__":
    main()
