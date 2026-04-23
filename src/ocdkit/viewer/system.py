"""System info: RAM, CPU, GPU detection.

GPU detection delegates to the active plugin's ``get_use_gpu`` hook when
available; otherwise falls back to a generic torch probe so the API still
returns useful info even with no plugin.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .plugins.base import SegmentationPlugin


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
    cpu_cores = os.cpu_count() or 1

    gpu_available, gpu_backend, gpu_name = _generic_torch_gpu_probe()
    gpu_label = None
    if gpu_available and gpu_name:
        if gpu_backend and gpu_backend.upper() not in gpu_name.upper():
            gpu_label = f"{gpu_backend.upper()}: {gpu_name}"
        else:
            gpu_label = str(gpu_name)

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
        "cpu_cores": cpu_cores,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_backend": gpu_backend,
        "gpu_label": gpu_label,
        "use_gpu": use_gpu,
        "plugin": plugin.name if plugin else None,
    }
