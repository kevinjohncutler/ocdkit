"""External CA mode — wraps the ``step`` CLI for shared step-ca PKI.

Useful for orgs that run a central PKI; ocdkit shells out to ``step ca
certificate`` to request short-lived certs from the configured authority.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .paths import config_path

_STEP_SEARCH_PATHS = (
    "/opt/homebrew/bin/step",
    "/usr/local/bin/step",
    "/usr/bin/step",
    "/home/linuxbrew/.linuxbrew/bin/step",
)


class TLSConfigError(RuntimeError):
    """Raised when ocdkit TLS cannot be used (missing config, missing step CLI, etc.)."""


def _find_step(step_binary: str | None = None) -> str:
    if step_binary:
        return step_binary
    found = shutil.which("step")
    if found:
        return found
    for candidate in _STEP_SEARCH_PATHS:
        if Path(candidate).is_file():
            return candidate
    raise TLSConfigError(
        "step CLI not found on PATH. External-CA mode requires it. Install:\n"
        "  macOS:    brew install step\n"
        "  Ubuntu:   sudo apt install step-cli\n"
        "  Windows:  scoop install step\n"
        "  Releases: https://github.com/smallstep/cli/releases"
    )


def load_config_file(path: Path | str | None = None) -> dict:
    """Read a JSON external-CA config file (CLI convenience)."""
    cfg = Path(path) if path else config_path()
    if not cfg.exists():
        raise TLSConfigError(
            f"ocdkit external-CA config not found at {cfg}. "
            'Provide ca_url/provisioner/provisioner_password_file directly to '
            'ensure_cert() or create the JSON file.'
        )
    return json.loads(cfg.read_text())


def _step_cert_valid(cert_path: Path, threshold_hours: float, step_bin: str) -> bool:
    if not cert_path.exists():
        return False
    try:
        out = subprocess.check_output(
            [step_bin, "certificate", "inspect", str(cert_path), "--format", "json"],
            text=True, stderr=subprocess.DEVNULL,
        )
        info = json.loads(out)
        end = info.get("validity", {}).get("end") or info.get("not_after")
        if not end:
            return False
        not_after = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return (not_after - datetime.now(timezone.utc)).total_seconds() / 3600 > threshold_hours
    except Exception:
        return False


def _ensure_cert_external(
    hostnames: list[str],
    *,
    ca_url: str,
    provisioner: str,
    provisioner_password_file: str | Path,
    out_dir: Path,
    step_binary: str | None,
    renewal_hours: float,
) -> tuple[str, str]:
    step_bin = _find_step(step_binary)
    primary = hostnames[0]
    cert_path = out_dir / f"{primary}.pem"
    key_path = out_dir / f"{primary}.key"
    if _step_cert_valid(cert_path, renewal_hours, step_bin):
        return str(cert_path), str(key_path)
    cmd = [
        step_bin, "ca", "certificate",
        "--ca-url", ca_url,
        "--provisioner", provisioner,
        "--provisioner-password-file", str(provisioner_password_file),
        "--force",
    ]
    for san in hostnames:
        cmd.extend(["--san", san])
    cmd.extend([primary, str(cert_path), str(key_path)])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return str(cert_path), str(key_path)
