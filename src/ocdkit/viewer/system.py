"""System info: RAM, CPU, GPU detection.

GPU detection delegates to the active plugin's ``get_use_gpu`` hook when
available; otherwise falls back to a generic torch probe so the API still
returns useful info even with no plugin.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any, Optional

from .plugins.base import SegmentationPlugin


def _mps_chip_name() -> Optional[str]:
    """Return the Apple Silicon chip marketing name, e.g. 'M3 Max'.

    Falls back to None if not on macOS / sysctl unavailable. Cheap one-shot:
    a single sysctl invocation, ~5 ms on a warm shell.
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2.0,
        )
    except Exception:
        return None
    raw = (out.stdout or "").strip()
    # Examples: "Apple M1", "Apple M3 Max", "Apple M2 Ultra"
    m = re.match(r"Apple\s+(M\d[A-Za-z0-9 ]*)", raw)
    if m:
        return m.group(1).strip()
    return raw or None


def _shorten_gpu_name(raw: str, backend: Optional[str]) -> str:
    """Strip vendor/generic noise from torch's verbose GPU names.

    Examples:
      "NVIDIA GeForce RTX 4090"        -> "RTX 4090"
      "NVIDIA H100 80GB HBM3"          -> "H100 80GB"
      "Tesla V100-SXM2-32GB"           -> "Tesla V100 32GB"
      "AMD Radeon Pro W6800"           -> "Radeon Pro W6800"
      "AMD Instinct MI250X"            -> "Instinct MI250X"
      "Radeon RX 7900 XTX"             -> "RX 7900 XTX"
      "Intel(R) Arc(TM) A770 Graphics" -> "Arc A770"
      "Apple MPS"                       -> "M3 Max"  (resolved via sysctl)
    """
    if backend == "mps":
        # torch reports a generic "Apple MPS" — query sysctl for the chip.
        chip = _mps_chip_name()
        return chip or "Apple Silicon"
    name = raw.strip()
    # Drop trademark markers (R) (TM)
    name = re.sub(r"\s*\((?:R|r|TM|tm)\)\s*", " ", name)
    # Drop vendor prefixes
    name = re.sub(
        r"^(NVIDIA|AMD|Intel|Advanced Micro Devices,?\s*Inc\.?)\s+", "", name, flags=re.I,
    )
    # NVIDIA: "GeForce" prefix is just the consumer-line marker
    name = re.sub(r"^GeForce\s+", "", name, flags=re.I)
    # Intel: trailing "Graphics" / "Graphics Controller" / "Display"
    name = re.sub(r"\s+(Graphics(\s+Controller)?|Display)\s*$", "", name, flags=re.I)
    # Tesla/A100/H100 SKU variants: "V100-SXM2-32GB" -> keep base + memory
    m = re.match(r"^(Tesla\s+)?([A-Z]\d{2,4}[A-Z0-9]*)-(?:[A-Z0-9]+)-(\d+GB)$", name)
    if m:
        prefix = (m.group(1) or "").strip()
        return f"{prefix} {m.group(2)} {m.group(3)}".strip()
    return name.strip() or raw


def _raw_cpu_brand() -> Optional[str]:
    """Return the OS-reported CPU brand string (vendor + model + noise)."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2.0,
            )
            v = (out.stdout or "").strip()
            if v:
                return v
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        _, _, val = line.partition(":")
                        v = val.strip()
                        if v:
                            return v
        except Exception:
            pass
    # Windows / fallback: platform.processor() returns the brand on Windows,
    # the chip family on macOS (already covered above), often empty on Linux.
    try:
        import platform as _platform
        v = (_platform.processor() or "").strip()
        return v or None
    except Exception:
        return None


def _shorten_cpu_name(raw: str) -> str:
    """Strip vendor + marketing noise from a CPU brand string.

    Examples:
      'AMD Ryzen Threadripper PRO 3995WX 64-Cores'   -> 'Threadripper PRO 3995WX'
      'Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz'     -> 'Core i9-9900K'
      'AMD Ryzen 9 7950X 16-Core Processor'          -> 'Ryzen 9 7950X'
      'Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz'     -> 'Xeon Gold 6248'
      'Apple M1 Ultra'                                -> 'M1 Ultra'
    """
    name = (raw or "").strip()
    if not name:
        return ""
    # Drop trademark markers
    name = re.sub(r"\s*\((?:R|r|TM|tm)\)\s*", " ", name)
    # Drop vendor prefixes
    name = re.sub(r"^(AMD|Intel|Apple)\s+", "", name, flags=re.I)
    # Threadripper line is officially "AMD Ryzen Threadripper" — strip the
    # redundant "Ryzen" so it reads "Threadripper PRO 3995WX".
    name = re.sub(r"^Ryzen\s+(?=Threadripper)", "", name, flags=re.I)
    # Trailing clock spec ("CPU @ 3.60GHz", "@ 2.5 GHz")
    name = re.sub(r"\s*(CPU\s+)?@\s*[\d.]+\s*[GMK]Hz\s*$", "", name, flags=re.I)
    # Trailing core-count noise ("64-Cores", "16-Core Processor", "Processor")
    name = re.sub(r"\s+\d+-Cores?(\s+Processor)?\s*$", "", name, flags=re.I)
    name = re.sub(r"\s+Processor\s*$", "", name, flags=re.I)
    # Collapse whitespace
    return re.sub(r"\s+", " ", name).strip()


def _physical_cpu_cores() -> Optional[int]:
    """Best-effort physical core count (excluding SMT/hyperthreads)."""
    try:
        import psutil  # type: ignore
        v = psutil.cpu_count(logical=False)
        if v and v > 0:
            return int(v)
    except Exception:
        pass
    return None


def _cpu_max_freq_mhz() -> Optional[float]:
    """Best-effort max clock in MHz (psutil.cpu_freq().max)."""
    try:
        import psutil  # type: ignore
        f = psutil.cpu_freq()
        if f and f.max and f.max > 0:
            return float(f.max)
    except Exception:
        pass
    return None


def _torch_gpu_memory_bytes() -> Optional[int]:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return None


def _generic_torch_gpu_probe() -> tuple[bool, Optional[str], Optional[str]]:
    """Return (available, backend, name) for the best available torch device."""
    try:
        import torch  # type: ignore
    except Exception:
        return False, None, None
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA GPU"
        return True, "cuda", name
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True, "mps", "Apple MPS"
    return False, None, None


def _read_meminfo() -> tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available), int(vm.total - vm.available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            data = handle.read().splitlines()
        meminfo: dict[str, int] = {}
        for line in data:
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            value = parts[1].strip().split()[0]
            meminfo[key] = int(value) * 1024
        total = meminfo.get("MemTotal")
        available = meminfo.get("MemAvailable")
        if total is not None and available is not None:
            return total, available, total - available
    except Exception:
        pass
    return None, None, None


def get_system_info(plugin: Optional[SegmentationPlugin] = None) -> dict[str, Any]:
    """Return a dict describing RAM/CPU/GPU and the plugin GPU setting."""
    total, available, used = _read_meminfo()
    cpu_cores_logical = os.cpu_count() or 1
    cpu_cores_physical = _physical_cpu_cores() or cpu_cores_logical
    cpu_brand_raw = _raw_cpu_brand()
    cpu_label = _shorten_cpu_name(cpu_brand_raw) if cpu_brand_raw else None
    cpu_max_mhz = _cpu_max_freq_mhz()

    gpu_available, gpu_backend, gpu_name = _generic_torch_gpu_probe()
    gpu_label = None
    if gpu_available and gpu_name:
        gpu_label = _shorten_gpu_name(gpu_name, gpu_backend)
    gpu_memory_bytes = _torch_gpu_memory_bytes() if gpu_available else None

    use_gpu = False
    if plugin is not None and plugin.get_use_gpu is not None:
        try:
            use_gpu = bool(plugin.get_use_gpu())
        except Exception:
            use_gpu = False

    return {
        "ram_total": total,
        "ram_available": available,
        "ram_used": used,
        "cpu_cores": cpu_cores_logical,            # back-compat — logical
        "cpu_cores_logical": cpu_cores_logical,
        "cpu_cores_physical": cpu_cores_physical,
        "cpu_brand_raw": cpu_brand_raw,
        "cpu_label": cpu_label,
        "cpu_max_mhz": cpu_max_mhz,
        "gpu_available": gpu_available,
        "gpu_torch_ok": gpu_available,  # alias for clarity in the UI
        "gpu_name": gpu_name,
        "gpu_backend": gpu_backend,
        "gpu_label": gpu_label,
        "gpu_memory_bytes": gpu_memory_bytes,
        "use_gpu": use_gpu,
        "plugin": plugin.name if plugin else None,
    }
