"""PyWebView desktop snapshot tests.

Drives ``python -m ocdkit.viewer desktop --snapshot ... --eval-js ...`` as a
subprocess. The desktop launcher already supports headless automation via
those flags; this file just wraps them in pytest fixtures and assertions.

Why a separate test file: pywebview uses the platform's WebKit (Cocoa on
macOS, WebKitGTK on Linux), which is NOT what Playwright tests. Some bugs
only surface in WebKit (texture sampling, cookie handling, drag/drop quirks).

These tests are slow (5-15s each because they spawn a real desktop window)
and require pywebview installed. Most CI environments will skip them.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import os
from pathlib import Path

import pytest

pytest.importorskip("webview", reason="pywebview required for desktop tests")

# CI / headless servers often have no display; pywebview can still launch on
# macOS but on Linux without Xvfb it will fail. Skip cleanly.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("no DISPLAY available for pywebview tests", allow_module_level=True)


_CMD = [sys.executable, "-m", "ocdkit.viewer", "desktop"]


def _run(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        _CMD + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_desktop_snapshot_writes_png(tmp_path):
    """The viewer launches in headless desktop mode, captures the canvas, exits."""
    out = tmp_path / "snap.png"
    proc = _run([
        "--port", "0",
        "--snapshot", str(out),
        "--snapshot-timeout", "8",
    ], timeout=30)

    # Print stdout/stderr for debugging if assertions fail.
    if proc.returncode != 0 or not out.exists():
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)

    assert out.exists(), f"snapshot PNG not written; stderr: {proc.stderr[:500]}"
    assert out.stat().st_size > 1024, "snapshot suspiciously small"

    # Verify it's a real PNG (magic bytes).
    with out.open("rb") as fh:
        header = fh.read(8)
    assert header.startswith(b"\x89PNG\r\n\x1a\n"), "not a valid PNG file"


def test_desktop_eval_js_returns_value(tmp_path):
    """`--eval-js` should run user code in the loaded page and print the result."""
    out = tmp_path / "snap.png"
    proc = _run([
        "--port", "0",
        "--snapshot", str(out),
        "--snapshot-timeout", "8",
        "--eval-js", "(function(){return window.__VIEWER_CONFIG__ && window.__VIEWER_CONFIG__.sessionId ? 'has-session' : 'no-session';})()",
    ], timeout=30)

    if "has-session" not in (proc.stdout + proc.stderr):
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
    assert "has-session" in (proc.stdout + proc.stderr)
