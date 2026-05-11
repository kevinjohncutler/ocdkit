"""SAN auto-detection for "this machine".

Resolves the names that should appear in a leaf cert's Subject Alternative
Names: the mDNS hostname, the OS hostname, and the primary LAN IP. A shared
JSON hostmap can override the autodetect for known fleets.
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from typing import Iterable


def _primary_lan_ip() -> str | None:
    """Best-effort: the local IP that would be used to reach the LAN.

    Uses connect-without-sending: opens a UDP socket to a non-routable address,
    asks the kernel which local IP it picked, closes. No packets sent.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


def _mdns_hostname() -> str | None:
    """The machine's actual mDNS / Bonjour hostname (no .local suffix).

    On macOS, ``socket.gethostname()`` returns the BSD hostname which DHCP can
    overwrite — *not* the Bonjour name browsers use to find the machine. The
    real mDNS name lives at ``scutil --get LocalHostName``. Linux + Windows
    just use ``socket.gethostname()``.
    """
    import platform
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(
                ["scutil", "--get", "LocalHostName"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    h = socket.gethostname()
    return h.split(".")[0] if h else None


def _autodetect_names() -> list[str]:
    """Reasonable default SANs for "this machine".

    Includes the mDNS hostname + ``.local`` suffix, the OS hostname (in case
    DHCP set it differently), and the primary LAN IP. Dedupes preserving order.
    """
    names: list[str] = []
    mdns = _mdns_hostname()
    if mdns:
        names.append(mdns)
        names.append(f"{mdns}.local")
    raw = socket.gethostname()
    if raw and raw not in names and f"{raw}.local" not in names:
        names.append(raw)
    ip = _primary_lan_ip()
    if ip:
        names.append(ip)
    seen: set[str] = set()
    return [n for n in names if n and not (n in seen or seen.add(n))]


def _load_hostmap_entry(path: Path | str | None, hostname: str) -> list[str] | None:
    """Look up SANs for ``hostname`` in a shared JSON hostmap file."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    short = hostname.split(".")[0]
    entry = data.get(hostname) or data.get(short)
    if entry is None or not isinstance(entry, list):
        return None
    return [str(x) for x in entry]


def _resolve_hostnames(
    hostnames: str | Iterable[str] | None,
    hostmap_path: str | Path | None,
) -> list[str]:
    if hostnames is None:
        hosts = _load_hostmap_entry(hostmap_path, socket.gethostname()) \
                or _autodetect_names()
    elif isinstance(hostnames, str):
        hosts = [hostnames]
    else:
        hosts = list(hostnames)
    if not hosts:
        raise ValueError("hostnames must be non-empty")
    return hosts
