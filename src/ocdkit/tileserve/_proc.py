"""Out-of-process tile server: child entry + kernel-side RPC client (Phase 1).

The child runs the SAME FastAPI app (``server.make_app``) in its OWN process, so
HTTP serving never shares the kernel's GIL — eliminating the tile-fetch stall that
the in-kernel daemon thread suffers when the kernel computes. The kernel pushes
source data over a length-prefixed pickle control socket (AF_UNIX); the child
applies each command to its own in-process server stores via ``server._DISPATCH``.
Transfer is BY VALUE (pickle copy) in this phase; shared memory is a later phase.

Flag-gated: only used when ``OCDKIT_TILESERVE_OOP`` is set (see
``server.ensure_server``). The default in-process path never imports this module.
"""
from __future__ import annotations

import os
import socket
import struct
import pickle
import sys
import threading
import time

_HDR = struct.Struct(">Q")   # 8-byte big-endian length prefix


def _recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _send(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HDR.pack(len(data)) + data)


def _recv(sock):
    h = _recvn(sock, _HDR.size)
    if h is None:
        return None
    (n,) = _HDR.unpack(h)
    body = _recvn(sock, n)
    if body is None:
        return None
    return pickle.loads(body)


# ── child process ────────────────────────────────────────────────────────────
def serve_child(control_path: str, port: int, ext_modules):
    """Child entry: import host route extensions, run the FastAPI app on ``port``,
    then serve the control socket — applying each pushed command to the local
    in-process server stores."""
    import importlib
    for m in ext_modules:
        if m:
            try:
                importlib.import_module(m)   # module-level register_extension re-mounts routes
            except Exception as e:           # noqa: BLE001
                print(f"[tileserve-child] ext import {m!r} failed: {e}", flush=True)

    from . import server as S
    import uvicorn

    config = uvicorn.Config(S.make_app(), host="127.0.0.1", port=int(port),
                            log_level="warning")
    srv = uvicorn.Server(config)
    threading.Thread(target=srv.run, daemon=True,
                     name="ocdkit-tileserve-child").start()
    deadline = time.time() + 20.0
    while time.time() < deadline:                    # wait until HTTP accepts
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
                break
        except OSError:
            time.sleep(0.03)

    if os.path.exists(control_path):
        os.unlink(control_path)
    lsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    lsock.bind(control_path)
    lsock.listen(1)
    print(f"[tileserve-child] ready port={port}", flush=True)
    conn, _ = lsock.accept()
    while True:
        msg = _recv(conn)
        if msg is None:
            break
        op, args, kwargs = msg
        try:
            fn = S._DISPATCH.get(op)
            if fn is None:
                raise KeyError(f"unknown command {op!r}")
            _send(conn, ("ok", fn(*args, **kwargs)))
        except Exception as e:                       # noqa: BLE001
            import traceback
            _send(conn, ("err", f"{e!r}\n{traceback.format_exc()}"))


# ── kernel-side client ───────────────────────────────────────────────────────
class ProcClient:
    """Kernel-side proxy to the child. ``call(op, args, kwargs)`` runs the named
    command in the child and returns its result (raising on the child's error)."""

    def __init__(self, control_path, proc, url):
        self.url = url
        self._proc = proc
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(control_path)

    def call(self, op, args=(), kwargs=None):
        with self._lock:
            _send(self._sock, (op, tuple(args), dict(kwargs or {})))
            reply = _recv(self._sock)
        if reply is None:
            raise RuntimeError("tileserve child closed the control connection")
        status, payload = reply
        if status == "err":
            raise RuntimeError(f"tileserve child: {payload}")
        return payload

    def alive(self):
        return self._proc.poll() is None

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            pass


def spawn(port: int, ext_modules, control_dir=None) -> "ProcClient":
    """Launch the child process bound to ``port`` and return a connected client.
    ``ext_modules`` = module names whose import re-registers host route extensions."""
    import subprocess
    import tempfile
    control_path = os.path.join(control_dir or tempfile.gettempdir(),
                                f"ocdtile-{os.getpid()}-{port}.sock")
    if os.path.exists(control_path):
        os.unlink(control_path)
    env = dict(os.environ)
    env.pop("OCDKIT_TILESERVE_OOP", None)            # child runs in-process; never recurse
    proc = subprocess.Popen(
        [sys.executable, "-m", "ocdkit.tileserve._proc",
         control_path, str(port), ",".join(ext_modules)],
        env=env)
    deadline = time.time() + 40.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"tileserve child exited early (code {proc.returncode})")
        if os.path.exists(control_path):
            try:
                return ProcClient(control_path, proc, f"http://127.0.0.1:{port}")
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                pass
        time.sleep(0.05)
    proc.terminate()
    raise RuntimeError("tileserve child failed to start within 40s")


if __name__ == "__main__":
    _exts = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []
    serve_child(sys.argv[1], int(sys.argv[2]), _exts)
