"""
Image I/O and path utilities.

Read/write images in common microscopy formats (TIFF, NPY, NPZ, CZI,
PNG, JPEG, WebP, JPEG XL, BMP) without OpenCV. Cross-platform path
normalization for shared NAS volumes.
"""

import logging
import os
import platform
import re
from pathlib import Path

import shutil
import tempfile
from urllib.request import urlopen

from .imports import *
import tifffile
import imagecodecs
from natsort import natsorted
from tqdm import tqdm

io_logger = logging.getLogger(__name__)


def check_dir(path):
    """Create *path* (and parents) if it does not already exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def getname(path, prefix='', suffix='', padding=0):
    """Return the file stem of *path*, optionally stripping *prefix*/*suffix*.

    Parameters
    ----------
    path : str or Path
        File path.
    prefix : str
        Prefix to strip from the stem.
    suffix : str
        Suffix to strip from the stem.
    padding : int
        Zero-pad the result to this width.
    """
    return os.path.splitext(Path(path).name)[0].replace(prefix, '').replace(suffix, '').zfill(padding)


def imread(filename):
    """Read an image from *filename*.

    Dispatches by extension:
    - ``.tif`` / ``.tiff`` -> tifffile
    - ``.npy`` -> numpy
    - ``.npz`` -> numpy (key ``arr_0``)
    - ``.czi`` -> bioio
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
        from bioio import BioImage
        return BioImage(filename).data
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


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def adjust_file_path(file_path):
    """Normalize a file path for the current OS.

    Handles shared NAS volumes that are mounted at different paths on
    macOS (``/Volumes/...``), Linux (``/home/<user>/...``), and Windows
    (``C:\\Users\\<user>\\...``).

    Parameters
    ----------
    file_path : str
        The original file path.

    Returns
    -------
    str
        The adjusted file path.
    """
    system = platform.system()
    if system == 'Darwin':
        return re.sub(r'^/home/\w+', '/Volumes', file_path)
    elif system == 'Linux':
        home_dir = os.path.expanduser('~')
        return re.sub(r'^/Volumes', home_dir, file_path)
    elif system == 'Windows':
        home_dir = os.path.expanduser('~')
        replace_with_home = lambda _match: home_dir
        adjusted = re.sub(r'^/home/[^/]+', replace_with_home, file_path)
        adjusted = re.sub(r'^/Volumes', replace_with_home, adjusted)
        return os.path.normpath(adjusted)
    else:
        return file_path


def findbetween(s, string1='[', string2=']'):
    """Return the text between *string1* and *string2* in *s*, or ``''``."""
    matches = re.findall(re.escape(string1) + "(.*)" + re.escape(string2), s)
    return matches[0] if matches else ''


def find_files(directory, suffix, exclude_suffixes=None):
    """Walk *directory* and return paths whose stems end with *suffix*.

    Parameters
    ----------
    directory : str or Path
        Root directory to search.
    suffix : str
        Stem suffix to match (e.g. ``"_mask"``).
    exclude_suffixes : list of str, optional
        Stem suffixes to exclude.

    Returns
    -------
    list of str
        Matching file paths.
    """
    if exclude_suffixes is None:
        exclude_suffixes = []
    matching = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            name, ext = os.path.splitext(basename)
            if name.endswith(suffix) and not any(name.endswith(ex) for ex in exclude_suffixes):
                matching.append(os.path.join(root, basename))
    return matching


def get_image_files(folder, mask_filter='_masks', img_filter='',
                    look_one_level_down=False,
                    extensions=('png', 'jpg', 'jpeg', 'tif', 'tiff'),
                    pattern=None):
    """Find image files in *folder*, excluding masks and other derivatives.

    Parameters
    ----------
    folder : str or Path
        Root directory to search.
    mask_filter : str
        Stem suffix used for mask files (default ``'_masks'``).
    img_filter : str
        Only include files whose stem ends with this string.
    look_one_level_down : bool
        Also search immediate subdirectories.
    extensions : sequence of str
        File extensions to include (without dot).
    pattern : str, optional
        Regex pattern the stem must match (anchored to end).

    Returns
    -------
    list of str
        Naturally sorted list of matching image file paths.

    Raises
    ------
    ValueError
        If no images are found.
    """
    import glob

    mask_filters = ['_cp_masks', '_cp_output', '_flows', mask_filter]

    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*", '')))
    folders.append(str(folder))

    image_names = []
    for d in folders:
        for ext in extensions:
            image_names.extend(glob.glob(d + ('/*%s.' + ext) % img_filter))

    image_names = natsorted(image_names)
    filtered = []
    for im in image_names:
        stem = os.path.splitext(im)[0]
        good = all(
            (len(stem) > len(mf) and stem[-len(mf):] != mf) or len(stem) < len(mf)
            for mf in mask_filters
        )
        if img_filter and stem[-len(img_filter):] != img_filter:
            good = False
        if pattern is not None:
            good = good and bool(re.search(pattern + r'$', stem))
        if good:
            filtered.append(im)

    if not filtered:
        raise ValueError('no images found in folder')

    return filtered


def download_url_to_file(url, dst, progress=True):
    """Download object at the given URL to a local path.

    Uses a temporary file and atomic rename so partial downloads never
    leave a corrupt *dst*.
    """
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    u = urlopen(url)
    meta = u.info()
    content_length = (
        meta.getheaders("Content-Length")
        if hasattr(meta, "getheaders")
        else meta.get_all("Content-Length")
    )
    file_size = int(content_length[0]) if content_length else None
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
