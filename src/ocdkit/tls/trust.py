"""Build downloadable trust installers for end-user devices.

Each helper here packages the local root CA so that an end user can install it
into their OS trust store with a single click — Apple Configuration Profile,
macOS ``.command`` script, Windows ``.reg`` file, or copy-pasteable shell
one-liners. The CA itself comes from :mod:`.local_ca`.
"""

from __future__ import annotations

import hashlib
import plistlib
import struct
import uuid

from .imports import *
from .local_ca import root_ca_der_bytes, root_ca_pem_bytes


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
