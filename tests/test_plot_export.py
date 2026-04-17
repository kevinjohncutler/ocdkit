"""Tests for ocdkit.plot.export — ffmpeg-backed gif/movie export."""

import numpy as np

from ocdkit.plot import export as export_mod


def test_export_gif_and_movie_smoke(tmp_path, monkeypatch):
    calls = []

    class DummyStdin:
        def __init__(self):
            self.buf = bytearray()

        def write(self, data):
            self.buf.extend(data)

        def close(self):
            return None

    class DummyProc:
        def __init__(self, args, **_kwargs):
            calls.append(args)
            self.stdin = DummyStdin()

        def wait(self):
            return 0

    monkeypatch.setattr(export_mod.subprocess, "Popen", DummyProc)

    frames_rgb = (np.random.rand(2, 4, 4, 3) * 255).astype(np.uint8)
    frames_gray = (np.random.rand(2, 4, 4) * 255).astype(np.uint8)

    export_mod.export_gif(frames_rgb, "rgb", str(tmp_path), fps=5, scale=1, bounce=True)
    export_mod.export_gif(frames_gray, "gray", str(tmp_path), fps=5, scale=1, bounce=False)
    export_mod.export_movie(frames_rgb, "movie", str(tmp_path), fps=5)

    assert len(calls) >= 3
