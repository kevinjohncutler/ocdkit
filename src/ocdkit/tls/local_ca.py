"""Pure-Python self-signed CA used to issue leaf certs.

Each ocdkit install on a machine has its own CA stored under :func:`ca_dir`.
Sharing trust across machines requires running the trust install flow on each
client device.
"""

from __future__ import annotations

import base64
import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .imports import *
from .paths import ca_dir, config_dir

_RENEWAL_THRESHOLD_HOURS = 8
_DEFAULT_CA_VALIDITY_YEARS = 10
_DEFAULT_LEAF_VALIDITY_DAYS = 30


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
