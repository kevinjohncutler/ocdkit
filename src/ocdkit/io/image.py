"""Image reading and writing (TIFF, NPY, NPZ, CZI, PNG, JPEG, WebP, JXL, BMP)."""

import logging
import os

from .imports import *
import tifffile
import imagecodecs

io_logger = logging.getLogger(__name__)


def imread(filename):
    """Read an image from *filename*.

    Dispatches by extension:
    - ``.tif`` / ``.tiff`` -> tifffile
    - ``.npy`` -> numpy
    - ``.npz`` -> numpy (key ``arr_0``)
    - ``.czi`` -> aicspylibczi
    - everything else -> imagecodecs
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext in ('.tif', '.tiff'):
        return tifffile.imread(filename)
    elif ext == '.npy':
        return np.load(filename)
    elif ext == '.npz':
        return np.load(filename)['arr_0']
    elif ext == '.czi':
        # aicspylibczi is ~5x faster than bioio for raw pixel reads (146 MB
        # CZI: 19 ms vs 108 ms). bioio's value is its uniform multi-format
        # API and xarray output; for ndarray reads we can skip that overhead.
        from aicspylibczi import CziFile
        return CziFile(filename).read_image()[0]
    else:
        try:
            with open(filename, 'rb') as f:
                data = f.read()
            return imagecodecs.imread(data)
        except Exception as e:
            io_logger.critical('ERROR: could not read file, %s' % e)
            return None


def imwrite(filename, arr, **kwargs):
    """Save an image to *filename* using imagecodecs for encoding.

    Supported extensions:
    - ``.tif`` / ``.tiff`` -> tifffile
    - ``.npy`` -> numpy
    - ``.png`` -> imagecodecs (accepts ``level``)
    - ``.jpg`` / ``.jpeg`` / ``.jp2`` -> imagecodecs (accepts ``level``)
    - ``.webp`` -> imagecodecs (accepts ``level`` or ``quality``)
    - ``.jxl`` -> imagecodecs (accepts ``effort``, ``distance``)
    - ``.bmp`` -> imagecodecs (lossless)

    Other extensions fall back to PNG encoding.
    """
    ext = os.path.splitext(filename)[-1].lower()

    if ext in ('.tif', '.tiff'):
        tifffile.imwrite(filename, arr, **kwargs)
        return
    elif ext == '.npy':
        np.save(filename, arr, **kwargs)
        return

    encoded = None
    if ext == '.png':
        level = kwargs.pop('level', 9)
        encoded = imagecodecs.png_encode(arr, level=level, **kwargs)
    elif ext in ('.jpg', '.jpeg', '.jp2'):
        level = kwargs.pop('level', 95)
        encoded = imagecodecs.jpeg_encode(arr, level=level, **kwargs)
    elif ext == '.webp':
        level = kwargs.pop('level', None)
        quality = kwargs.pop('quality', None)
        if quality is not None and level is None:
            level = quality
        if level is not None:
            encoded = imagecodecs.webp_encode(arr, level=level, **kwargs)
        else:
            encoded = imagecodecs.webp_encode(arr, **kwargs)
    elif ext == '.jxl':
        effort = kwargs.pop('effort', 1)
        distance = kwargs.pop('distance', 1.0)
        encoded = imagecodecs.jpegxl_encode(
            arr, effort=effort, distance=distance, **kwargs
        )
    elif ext == '.bmp':
        encoded = imagecodecs.bmp_encode(arr, **kwargs)
    else:
        encoded = imagecodecs.png_encode(arr, **kwargs)

    with open(filename, 'wb') as f:
        f.write(encoded)


def encode_rgb_image(img, format='png', effort_level=0,
                     quality_level=90, lossless=True):
    """Encode an RGB[A] array to a base-64 string for HTML embedding.

    Supports PNG, JPEG, WebP, and JPEG-XL via *imagecodecs*.
    Float images are assumed [0, 1] and scaled to uint8.
    """
    import base64

    if img.dtype in (np.float32, np.float64):
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    img = np.ascontiguousarray(img)

    fmt = format.lower()
    alias_map = {
        "jpeg": "jpg", "jpe": "jpg", "jfif": "jpg",
        "jpegxl": "jxl", "jxl": "jxl",
        "png": "png", "webp": "webp",
    }
    fmt = alias_map.get(fmt, fmt)

    if fmt == 'jpg':
        if img.ndim == 3 and img.shape[-1] == 4:
            raise ValueError("JPEG does not support alpha channel.")
        buf = imagecodecs.jpeg_encode(img, level=quality_level,
                                      lossless=(effort_level == 0))
    elif fmt == 'png':
        buf = imagecodecs.png_encode(img, level=effort_level)
    elif fmt == 'webp':
        buf = imagecodecs.webp_encode(img, level=quality_level,
                                      lossless=lossless)
    elif fmt == "jxl":
        buf = imagecodecs.jpegxl_encode(img, effort=effort_level,
                                        level=quality_level,
                                        lossless=lossless)
    else:
        raise ValueError(f"Unsupported format {fmt!r}; use 'png', 'jpg', 'webp', or 'jxl'.")

    return base64.b64encode(buf).decode("utf-8")
