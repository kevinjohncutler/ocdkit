"""Video and animation export utilities (ffmpeg-based)."""

import os
import subprocess

import numpy as np
from scipy import ndimage

from ..array import to_8_bit, to_16_bit


def export_gif(frames, basename, basedir, scale=1, fps=15, loop=0, bounce=True):
    """Export frame sequence to GIF using ffmpeg with palette generation.

    Parameters
    ----------
    frames : ndarray
        Frames as ``(T, Y, X, C)`` or ``(T, Y, X)`` for grayscale.
    basename : str
        Base name for output file.
    basedir : str
        Directory to save into.
    scale : float
        Scale factor (applied via ``scipy.ndimage.zoom``; ffmpeg scaling is
        unreliable here).
    fps : int
        Frames per second.
    loop : int
        Number of loops (``0`` = infinite).
    bounce : bool
        If True, append reversed frames for ping-pong effect.
    """
    if scale != 1:
        frames = ndimage.zoom(frames, [1, scale, scale, 1], order=0)
    try:
        if frames.ndim == 4:
            frame_width, frame_height, nchan = frames.shape[-3:]
            pixel_format = 'rgb24' if nchan == 3 else 'rgba'
        else:
            frame_width, frame_height = frames.shape[-2:]
            pixel_format = 'gray'

        file = os.path.join(basedir, basename + '_{}_fps_scale_{}.gif'.format(fps, scale))

        p = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-f', 'rawvideo', '-vcodec', 'rawvideo',
             '-s', '{}x{}'.format(frame_height, frame_width),
             '-pix_fmt', pixel_format,
             '-r', str(fps), '-i', '-', '-an',
             '-filter_complex',
             '[0:v]palettegen=stats_mode=full[pal],[0:v][pal]paletteuse=dither=none',
             '-vcodec', 'gif', '-loop', str(loop),
             file],
            stdin=subprocess.PIPE,
        )

        frames_8_bit = to_8_bit(frames)
        if bounce:
            frames_8_bit = np.concatenate((frames_8_bit, frames_8_bit[::-1]), axis=0)
        for frame in frames_8_bit:
            p.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        p.stdin.close()
        p.wait()


def export_movie(frames, basename, basedir, scale=1, fps=15):
    """Export frame sequence to MP4 video using ffmpeg.

    Parameters
    ----------
    frames : ndarray
        Frames as ``(T, Y, X, C)`` — must have 3 or 4 channels.
    basename : str
        Base name for output file.
    basedir : str
        Directory to save into.
    scale : float
        Output scale factor.
    fps : int
        Frames per second.
    """
    frame_width, frame_height, nchan = frames.shape[-3:]
    pixel_format = 'rgb48le' if nchan == 3 else 'rgba64le'

    file = os.path.join(basedir, basename + '_{}_fps.mp4'.format(fps))

    p = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-vcodec', 'rawvideo',
         '-s', '{}x{}'.format(frame_height, frame_width),
         '-pix_fmt', pixel_format,
         '-r', str(fps), '-i', '-',
         '-f', 'lavfi', '-i', 'anullsrc',
         '-vf', 'scale=iw*{}:ih*{}:flags=neighbor'.format(scale, scale),
         '-shortest', '-c:v', 'mpeg4', '-q:v', '0',
         file],
        stdin=subprocess.PIPE,
    )

    for frame in to_16_bit(frames):
        p.stdin.write(frame.tobytes())

    p.stdin.close()
    p.wait()
