"""File discovery and download utilities."""

import os
import re
import shutil
import tempfile
from urllib.request import urlopen

from .imports import *
from natsort import natsorted
from tqdm import tqdm


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
