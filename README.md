# ocdkit

**Obsessive Coder's Dependency Toolkit** — Python utilities for array manipulation, GPU dispatch, image I/O, spatial operations, morphology, and plotting.

## Install

```bash
pip install ocdkit             # core (numpy, scipy, scikit-image, tifffile, matplotlib)
pip install ocdkit[torch]      # + PyTorch GPU support
pip install ocdkit[plot]       # + ncolor, cmap, opt_einsum
pip install ocdkit[spatial]    # + numba, fastremap (contour extraction, skeletonization)
pip install ocdkit[all]        # everything
```

## Modules

| Module | What's in it |
|---|---|
| `ocdkit.array` | `rescale`, `safe_divide`, `is_integer`, `get_module`, `unique_nonzero` |
| `ocdkit.gpu` | `resolve_device`, `empty_cache`, `torch_GPU`, `torch_CPU` |
| `ocdkit.io` | `imread`, `imwrite`, `getname`, `check_dir` |
| `ocdkit.spatial` | `kernel_setup`, `get_neighbors`, `get_neigh_inds`, `masks_to_affinity`, `get_contour`, `boundary_to_masks` |
| `ocdkit.morphology` | `find_boundaries`, `skeletonize` |
| `ocdkit.measure` | `crop_bbox`, `bbox_to_slice`, `make_square`, `diameters` |
| `ocdkit.plot` | `figure`, `image_grid`, `split_list`, `colorize`, `rgb_flow`, `vector_contours`, `apply_ncolor`, `color_swatches`, `recolor_label`, `add_label_background` |

## Quick start

```python
from ocdkit.array import rescale
from ocdkit.gpu import resolve_device
from ocdkit.plot import figure, image_grid

device = resolve_device()  # auto-detect CUDA / MPS / CPU
```

## License

BSD-3-Clause
