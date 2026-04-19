"""Path manipulation and string utilities."""

import os
import platform
import re
from pathlib import Path


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


def adjust_file_path(file_path):
    """Normalize a file path for the current OS.

    Handles shared NAS volumes that are mounted at different paths on
    macOS (``/Volumes/...``), Linux (``/home/<user>/...``), and Windows
    (``C:\\\\Users\\\\<user>\\\\...``).

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
