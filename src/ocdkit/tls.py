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

import base64
import hashlib
import ipaddress
import json
import plistlib
import shutil
import socket
import struct
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

try:
    from platformdirs import user_config_dir
except ImportError:
    import os
    def user_config_dir(appname: str, appauthor: str = "") -> str:
        return os.path.expanduser(f"~/.config/{appname}")

_RENEWAL_THRESHOLD_HOURS = 8
_DEFAULT_CA_VALIDITY_YEARS = 10
_DEFAULT_LEAF_VALIDITY_DAYS = 30
_STEP_SEARCH_PATHS = (
    "/opt/homebrew/bin/step",
    "/usr/local/bin/step",
    "/usr/bin/step",
    "/home/linuxbrew/.linuxbrew/bin/step",
)


class TLSConfigError(RuntimeError):
    """Raised when ocdkit TLS cannot be used (missing config, missing step CLI, etc.)."""


# ── Paths ──────────────────────────────────────────────────────────────


def config_dir() -> Path:
    """Directory holding ocdkit TLS config, CA, and leaf cert/key files."""
    return Path(user_config_dir("ocdkit")) / "tls"


def config_path() -> Path:
    return config_dir() / "config.json"


def ca_dir() -> Path:
    return config_dir() / "ca"


def root_ca_pem_path() -> Path:
    """Path to the local root CA certificate. Auto-creates the CA if missing."""
    LocalCA()  # idempotent; ensures root cert + key exist on disk
    return ca_dir() / "root.pem"


def root_ca_pem_bytes() -> bytes:
    """PEM-encoded root CA cert bytes (auto-creates CA if missing)."""
    return root_ca_pem_path().read_bytes()


def root_ca_der_bytes() -> bytes:
    """DER-encoded root CA cert bytes (for Windows .cer download)."""
    pem = root_ca_pem_bytes()
    body = b"".join(
        line for line in pem.splitlines()
        if line and not line.startswith(b"-----")
    )
    return base64.b64decode(body)


# ── Hostname auto-detection ────────────────────────────────────────────


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


# ── LocalCA — pure-Python self-signed CA ───────────────────────────────


class LocalCA:
    """Self-signed local CA stored under :func:`ca_dir`.

    Idempotent constructor: creates the root cert + key on first use, reuses
    them on subsequent instantiations. Each ``ocdkit`` install on a machine
    has its own CA — sharing across machines requires using the trust install
    flow on each client device.

    :param dir: override the on-disk location (default :func:`ca_dir`).
    :param name: human-readable label used as the CA's Common Name.
    :param validity_years: lifetime of the root CA cert (default 10 years).
    """

    def __init__(
        self,
        *,
        dir: Path | str | None = None,
        name: str = "Local Dev CA",
        validity_years: int = _DEFAULT_CA_VALIDITY_YEARS,
    ) -> None:
        self.dir = Path(dir) if dir else ca_dir()
        self.name = name
        self.validity_years = validity_years
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            self.dir.chmod(0o700)
        except PermissionError:
            pass
        self._ensure_root()

    @property
    def cert_path(self) -> Path:
        return self.dir / "root.pem"

    @property
    def key_path(self) -> Path:
        return self.dir / "root.key"

    def _ensure_root(self) -> None:
        if self.cert_path.exists() and self.key_path.exists():
            return
        key = ec.generate_private_key(ec.SECP256R1())
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ocdkit"),
            x509.NameAttribute(NameOID.COMMON_NAME, self.name),
        ])
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=365 * self.validity_years))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False, content_commitment=False,
                    key_encipherment=False, data_encipherment=False,
                    key_agreement=False, key_cert_sign=True,
                    crl_sign=True, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        self.cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        self.key_path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        try:
            self.cert_path.chmod(0o644)
            self.key_path.chmod(0o600)
        except PermissionError:
            pass

    def _load(self) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
        cert = x509.load_pem_x509_certificate(self.cert_path.read_bytes())
        key = serialization.load_pem_private_key(
            self.key_path.read_bytes(), password=None
        )
        return cert, key  # type: ignore[return-value]

    def issue_cert(
        self,
        hostnames: list[str],
        *,
        out_dir: Path | str | None = None,
        validity_days: int = _DEFAULT_LEAF_VALIDITY_DAYS,
        renewal_hours: float = _RENEWAL_THRESHOLD_HOURS,
    ) -> tuple[str, str]:
        """Issue (or reuse) a leaf cert covering ``hostnames``.

        Returns ``(cert_path, key_path)`` as strings. The cert file is a chain
        (leaf followed by root) so servers present the full chain.
        """
        out = Path(out_dir) if out_dir else config_dir()
        out.mkdir(parents=True, exist_ok=True)
        primary = hostnames[0]
        cert_path = out / f"{primary}.pem"
        key_path = out / f"{primary}.key"

        if _local_cert_valid(cert_path, renewal_hours, hostnames):
            return str(cert_path), str(key_path)

        ca_cert, ca_key = self._load()
        leaf_key = ec.generate_private_key(ec.SECP256R1())
        sans: list[x509.GeneralName] = []
        for h in hostnames:
            try:
                sans.append(x509.IPAddress(ipaddress.ip_address(h)))
            except ValueError:
                sans.append(x509.DNSName(h))

        now = datetime.now(timezone.utc)
        builder = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, primary),
            ]))
            .issuer_name(ca_cert.subject)
            .public_key(leaf_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=validity_days))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, content_commitment=False,
                    key_encipherment=True, data_encipherment=False,
                    key_agreement=False, key_cert_sign=False,
                    crl_sign=False, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    ExtendedKeyUsageOID.SERVER_AUTH,
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                ]),
                critical=False,
            )
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()),
                critical=False,
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()),
                critical=False,
            )
        )
        leaf_cert = builder.sign(ca_key, hashes.SHA256())

        chain_pem = (
            leaf_cert.public_bytes(serialization.Encoding.PEM) +
            ca_cert.public_bytes(serialization.Encoding.PEM)
        )
        cert_path.write_bytes(chain_pem)
        key_path.write_bytes(
            leaf_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        try:
            cert_path.chmod(0o644)
            key_path.chmod(0o600)
        except PermissionError:
            pass
        return str(cert_path), str(key_path)


def _local_cert_valid(
    cert_path: Path,
    threshold_hours: float,
    required_hostnames: list[str] | None = None,
) -> bool:
    """Check whether an existing leaf cert is still usable.

    Renews if expired-soon OR if the SAN list has changed (e.g. user added a
    new hostname to the args since last issuance).
    """
    if not cert_path.exists():
        return False
    try:
        certs = _read_pem_certs(cert_path.read_bytes())
        if not certs:
            return False
        leaf = certs[0]
        remaining_h = (
            leaf.not_valid_after_utc - datetime.now(timezone.utc)
        ).total_seconds() / 3600
        if remaining_h <= threshold_hours:
            return False
        if required_hostnames:
            try:
                san_ext = leaf.extensions.get_extension_for_class(
                    x509.SubjectAlternativeName
                ).value
            except x509.ExtensionNotFound:
                return False
            current: set[str] = set()
            for entry in san_ext:
                if isinstance(entry, x509.DNSName):
                    current.add(entry.value)
                elif isinstance(entry, x509.IPAddress):
                    current.add(str(entry.value))
            if not set(required_hostnames).issubset(current):
                return False
        return True
    except Exception:
        return False


def _read_pem_certs(data: bytes) -> list[x509.Certificate]:
    """Parse one or more PEM certs from a single byte blob."""
    certs: list[x509.Certificate] = []
    chunk: list[bytes] = []
    in_cert = False
    for line in data.splitlines():
        if b"BEGIN CERTIFICATE" in line:
            in_cert = True
            chunk = [line]
        elif b"END CERTIFICATE" in line:
            chunk.append(line)
            try:
                certs.append(x509.load_pem_x509_certificate(b"\n".join(chunk)))
            except Exception:
                pass
            in_cert = False
            chunk = []
        elif in_cert:
            chunk.append(line)
    return certs


# ── External CA (opt-in: shared step-ca via step CLI) ──────────────────


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


# ── Public API ─────────────────────────────────────────────────────────


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


# ── Trust install: build downloadable installers for end-user clients ──


def build_windows_reg(*, scope: str = "user") -> bytes:
    """Build a Windows ``.reg`` file that installs the root CA as a Trusted Root.

    Double-clicking the file on Windows triggers a single "Are you sure you want
    to add this to the registry?" confirmation — no Certificate Import Wizard
    and no risk of the user picking the wrong store. After they click Yes, the
    cert lands directly in Trusted Root Certification Authorities.

    :param scope: ``"user"`` writes to ``HKEY_CURRENT_USER`` (no admin, trusts
        only for the logged-in user). ``"machine"`` writes to
        ``HKEY_LOCAL_MACHINE`` (UAC prompt, trusts for all users on the box).
    """
    if scope == "user":
        hive = "HKEY_CURRENT_USER"
    elif scope == "machine":
        hive = "HKEY_LOCAL_MACHINE"
    else:
        raise ValueError(f"scope must be 'user' or 'machine', got {scope!r}")

    der = root_ca_der_bytes()
    sha1_hash = hashlib.sha1(der).digest()
    thumbprint = sha1_hash.hex().upper()

    # Each property TLV: PropID(LE u32) + Flags(LE u32) + Size(LE u32) + Data
    blob = (
        struct.pack("<III", 0x03, 1, len(sha1_hash)) + sha1_hash +
        struct.pack("<III", 0x20, 1, len(der)) + der
    )
    hex_str = ",".join(f"{b:02x}" for b in blob)
    key_path = (
        f"{hive}\\SOFTWARE\\Microsoft\\SystemCertificates\\Root\\Certificates\\{thumbprint}"
    )
    body = (
        "Windows Registry Editor Version 5.00\r\n"
        "\r\n"
        f"[{key_path}]\r\n"
        f'"Blob"=hex:{hex_str}\r\n'
    )
    return b"\xff\xfe" + body.encode("utf-16-le")


def macos_install_oneliner() -> str:
    """Return a self-contained shell one-liner that installs the root CA into
    the macOS System keychain with full SSL trust.

    The cert is embedded in the command itself (heredoc), so the user doesn't
    need a downloaded file at any specific path. They just paste this into
    Terminal and enter their Mac password. Idempotent — safe to re-run.
    """
    pem = root_ca_pem_bytes().decode().strip()
    return (
        "TMPF=$(mktemp) && cat > \"$TMPF\" <<'CERT_PEM'\n"
        f"{pem}\n"
        "CERT_PEM\n"
        "sudo security add-trusted-cert -d -r trustRoot -p ssl "
        "-k /Library/Keychains/System.keychain \"$TMPF\" && rm \"$TMPF\""
    )


def linux_install_oneliner() -> str:
    """Return a self-contained shell one-liner for Debian/Ubuntu / similar.

    Other distros (Fedora, Arch, Alpine) use different paths; users can adapt.
    """
    pem = root_ca_pem_bytes().decode().strip()
    return (
        "sudo bash -c 'cat > /usr/local/share/ca-certificates/local-dev-ca.crt "
        "<<\"CERT_PEM\"\n"
        f"{pem}\n"
        "CERT_PEM\n"
        "update-ca-certificates'"
    )


def build_macos_install_command() -> str:
    """Build a ``.command`` shell script for one-click trust install on macOS.

    macOS Finder treats ``.command`` files as Terminal-runnable scripts. When
    the user double-clicks the downloaded file, Terminal opens and runs it.
    The script then asks for the user's Mac password (sudo prompt) and
    installs the CA into the System keychain with full SSL trust — fixing
    the unsigned-mobileconfig limitation where SSL trust isn't auto-granted.
    """
    pem = root_ca_pem_bytes().decode().strip()
    return f'''#!/bin/bash
# One-click installer for the Local Dev CA trust certificate.
# Double-click in Finder → Terminal opens → enter your Mac password → done.

set -e
clear
cat <<'BANNER'

  ┌─────────────────────────────────────────────────────────┐
  │  Installing the Local Dev CA trust certificate          │
  │                                                         │
  │  This grants your Mac permission to trust HTTPS from    │
  │  internal services signed by this CA.                   │
  │                                                         │
  │  You'll be asked for your Mac password once below.      │
  └─────────────────────────────────────────────────────────┘

BANNER

TMPFILE=$(mktemp /tmp/local-dev-ca.XXXXXX.pem)
trap "rm -f \\"$TMPFILE\\"" EXIT

cat > "$TMPFILE" <<'CERT_PEM'
{pem}
CERT_PEM

if sudo security add-trusted-cert -d -r trustRoot -p ssl \\
        -k /Library/Keychains/System.keychain "$TMPFILE"; then
    cat <<'OK'

  ✅ Done! The CA is now trusted system-wide.

  Refresh your browser tab — internal HTTPS sites will load
  with a green padlock and no warnings.

OK
else
    cat <<'FAIL'

  ❌ Install was cancelled or failed.

  If you cancelled the password prompt, just run this file
  again. If it failed for another reason, ask the person who
  sent you this file.

FAIL
fi

echo
read -n 1 -s -r -p "Press any key to close this window..."
echo
exit 0
'''
def build_mobileconfig(
    *,
    display_name: str = "Local Dev CA",
    organization: str = "Local Dev",
    identifier: str = "ocdkit.trust.localdevca",
    description: str = "Installs the Local Dev CA root certificate so browsers "
                       "trust HTTPS served by internal machines.",
) -> bytes:
    """Build an Apple Configuration Profile (.mobileconfig) that installs the
    root CA as a trusted root."""
    cert_der = root_ca_der_bytes()
    outer_uuid = str(uuid.uuid4()).upper()
    inner_uuid = str(uuid.uuid4()).upper()
    plist = {
        "PayloadContent": [{
            "PayloadCertificateFileName": "root_ca.crt",
            "PayloadContent": cert_der,
            "PayloadDescription": f"Installs {display_name} root certificate as trusted.",
            "PayloadDisplayName": display_name,
            "PayloadIdentifier": f"{identifier}.cert",
            "PayloadType": "com.apple.security.root",
            "PayloadUUID": inner_uuid,
            "PayloadVersion": 1,
        }],
        "PayloadDescription": description,
        "PayloadDisplayName": f"{display_name} Trust Profile",
        "PayloadIdentifier": identifier,
        "PayloadOrganization": organization,
        "PayloadRemovalDisallowed": False,
        "PayloadType": "Configuration",
        "PayloadUUID": outer_uuid,
        "PayloadVersion": 1,
    }
    return plistlib.dumps(plist, fmt=plistlib.FMT_XML)


def root_ca_fingerprint_sha256() -> str:
    """SHA-256 fingerprint of the root CA (lowercase hex with colons)."""
    der = root_ca_der_bytes()
    digest = hashlib.sha256(der).hexdigest().upper()
    return ":".join(digest[i:i+2] for i in range(0, len(digest), 2))


def root_ca_subject() -> str:
    """Human-readable Subject DN of the root CA."""
    cert = x509.load_pem_x509_certificate(root_ca_pem_bytes())
    return cert.subject.rfc4514_string()
