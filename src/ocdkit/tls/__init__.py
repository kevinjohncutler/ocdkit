"""TLS cert provisioning for local Python apps.

Two modes:

1. **LocalCA (default, pure Python)** — generates a self-signed root CA on
   first use and signs short-lived leaf certs for the requested hostnames.
   Each machine runs its own CA; clients install the CA via the trust
   install page (``/trust/install``) once per device, then HTTPS to that
   server is trusted forever (10-year CA validity).

2. **External CA (opt-in)** — wraps the ``step`` CLI to request certs from
   a shared step-ca instance. Useful for orgs running their own central PKI.

Default usage (no args needed)::

    from ocdkit import tls
    cert, key = tls.ensure_cert()  # auto-detect hostnames, local CA

External CA::

    cert, key = tls.ensure_cert(
        ca_url="https://ca.lab.local:9000",
        provisioner="admin",
        provisioner_password_file="/etc/ocdkit/provisioner.pwd",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .external_ca import (
    TLSConfigError,
    _ensure_cert_external,
    load_config_file,
)
from .hostnames import _resolve_hostnames
from .local_ca import (
    _DEFAULT_CA_VALIDITY_YEARS,
    _DEFAULT_LEAF_VALIDITY_DAYS,
    _RENEWAL_THRESHOLD_HOURS,
    LocalCA,
    root_ca_der_bytes,
    root_ca_pem_bytes,
    root_ca_pem_path,
)
from .paths import ca_dir, config_dir, config_path
from .trust import (
    build_macos_install_command,
    build_mobileconfig,
    build_windows_reg,
    linux_install_oneliner,
    macos_install_oneliner,
    root_ca_fingerprint_sha256,
    root_ca_subject,
)

__all__ = [
    "TLSConfigError",
    "LocalCA",
    "ensure_cert",
    "load_config_file",
    "config_dir",
    "config_path",
    "ca_dir",
    "root_ca_pem_path",
    "root_ca_pem_bytes",
    "root_ca_der_bytes",
    "root_ca_fingerprint_sha256",
    "root_ca_subject",
    "build_windows_reg",
    "build_mobileconfig",
    "build_macos_install_command",
    "macos_install_oneliner",
    "linux_install_oneliner",
]


def ensure_cert(
    hostnames: str | Iterable[str] | None = None,
    *,
    # External CA opt-in (advanced — use a shared step-ca / Let's Encrypt)
    ca_url: str | None = None,
    provisioner: str | None = None,
    provisioner_password_file: str | Path | None = None,
    step_binary: str | None = None,
    # LocalCA tuning (default mode)
    ca_dir: Path | str | None = None,
    ca_name: str = "Local Dev CA",
    ca_validity_years: int = _DEFAULT_CA_VALIDITY_YEARS,
    leaf_validity_days: int = _DEFAULT_LEAF_VALIDITY_DAYS,
    # Common
    cert_dir: str | Path | None = None,
    hostmap_path: str | Path | None = None,
    renewal_hours: float = _RENEWAL_THRESHOLD_HOURS,
) -> tuple[str, str]:
    """Return ``(cert_path, key_path)`` for a valid cert covering ``hostnames``.

    Two modes:

    1. **LocalCA (default)** — pure-Python self-signed CA stored under
       :func:`ca_dir`. Each device that wants to trust this server needs to
       install the root CA cert (via the trust install page). 10-year CA
       validity by default.
    2. **External CA** — set ``ca_url`` + ``provisioner`` +
       ``provisioner_password_file`` to request certs from a shared step-ca
       instance via the ``step`` CLI. For orgs with central PKI.

    SAN resolution (in priority order):

      1. Explicit ``hostnames`` argument
      2. ``hostmap_path`` JSON file with this machine's entry
      3. Auto-detect: short hostname + ``<hostname>.local`` + primary LAN IP
    """
    hosts = _resolve_hostnames(hostnames, hostmap_path)
    out_dir = Path(cert_dir) if cert_dir else config_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if ca_url:
        if not (provisioner and provisioner_password_file):
            raise TLSConfigError(
                "External-CA mode requires ca_url, provisioner, and "
                "provisioner_password_file together."
            )
        return _ensure_cert_external(
            hosts,
            ca_url=ca_url,
            provisioner=provisioner,
            provisioner_password_file=provisioner_password_file,
            out_dir=out_dir,
            step_binary=step_binary,
            renewal_hours=renewal_hours,
        )

    ca = LocalCA(dir=ca_dir, name=ca_name, validity_years=ca_validity_years)
    return ca.issue_cert(
        hosts,
        out_dir=out_dir,
        validity_days=leaf_validity_days,
        renewal_hours=renewal_hours,
    )
