"""GET /trust/* — root CA download + install-walkthrough page.

Served on BOTH the main HTTPS app (for users who already trust and want to
re-download) and on the HTTP setup sidecar (for users hitting the site for the
first time who need to install trust before HTTPS works without warnings).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from ... import tls

router = APIRouter(prefix="/trust", tags=["trust"])


_INSTALL_PAGE_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Install Trust Certificate</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
         max-width: 560px; margin: 4rem auto; padding: 0 1.5rem; color: #111;
         background: #fafafa; line-height: 1.5; }
  h1 { font-size: 1.6rem; margin: 0 0 1rem; }
  p  { margin: 0.8rem 0; }
  .card { background: #fff; border: 1px solid #e2e2e2; border-radius: 10px;
          padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  a.button { display: inline-block; padding: 0.7rem 1.2rem; margin: 0.3rem 0;
             background: #0A84FF; color: #fff; text-decoration: none;
             border-radius: 6px; font-weight: 500; }
  a.button:hover { background: #0066cc; }
  a.button.secondary { background: #555; }
  .hint { color: #666; font-size: 0.88rem; }
  ol.hint { margin: 0.3rem 0 0.8rem 1.2rem; padding: 0; }
  ol.hint li { margin: 0.2rem 0; }
  code { background: #f0f0f0; padding: 0.1rem 0.3rem; border-radius: 3px;
         font-size: 0.9em; }
  .platform { display: none; }
  .platform.active { display: block; }
  .detected { padding: 0.4rem 0.7rem; background: #e8f5e9;
              border-left: 3px solid #2e7d32; border-radius: 4px;
              margin: 0.5rem 0; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="card">
  <h1>Install Trust Certificate</h1>
  <p>You only need to do this <strong>once per device</strong>. After it's done,
     every internal service signed by the same CA will show a green padlock
     with no warnings.</p>

  <div id="mac" class="platform">
    <div class="detected">Detected: macOS</div>
    <p>
      <a class="button" id="mac-install-btn" href="./ca.pem" download>
        Download &amp; Copy Install Command</a>
    </p>
    <p id="mac-status" class="hint" style="display: none; color: #2e7d32; font-weight: 500;"></p>
    <p class="hint">
      One click here downloads the certificate to <code>~/Downloads/__PEM_FILENAME__</code>
      AND copies the install command to your clipboard. Then:
    </p>
    <ol class="hint">
      <li>Open Terminal (⌘+Space → type "Terminal" → Enter)</li>
      <li>Paste (⌘+V) and press Enter</li>
      <li>Type your Mac password when prompted</li>
      <li>Refresh this tab → green padlock</li>
    </ol>
    <p class="hint" style="margin-top: 1.2rem;">
      Alternatives if you'd rather:
      <a href="./ca-install.command" download><code>.command</code> file</a> (double-click runs Terminal — needs Gatekeeper bypass once),
      <a href="./ca.mobileconfig" download><code>.mobileconfig</code></a> (System Settings → Profiles, but doesn't grant SSL trust on unsigned).
    </p>
  </div>

  <script>
    (function() {
      const btn = document.getElementById("mac-install-btn");
      const status = document.getElementById("mac-status");
      if (!btn || !status) return;
      const cmd = "sudo security add-trusted-cert -d -r trustRoot -p ssl -k " +
                  "/Library/Keychains/System.keychain ~/Downloads/__PEM_FILENAME__";
      btn.addEventListener("click", function() {
        if (navigator.clipboard) {
          navigator.clipboard.writeText(cmd).then(function() {
            status.style.display = "block";
            status.textContent = "📋 Install command copied. Now open Terminal and paste.";
          }).catch(function() {
            status.style.display = "block";
            status.textContent = "⚠ Couldn't copy command. Run manually: " + cmd;
          });
        }
      });
    })();
  </script>

  <div id="ios" class="platform">
    <div class="detected">Detected: iOS / iPadOS</div>
    <p><a class="button" href="./ca.mobileconfig" download>Download Profile</a></p>
    <p class="hint">
      The profile downloads to Settings → go to
      <em>Settings → Profile Downloaded</em> to install, then
      <em>Settings → General → About → Certificate Trust Settings</em> to
      enable full trust for the Local Dev CA.
    </p>
  </div>

  <div id="windows" class="platform">
    <div class="detected">Detected: Windows</div>

    <p><strong>Recommended:</strong> download the registry installer — two clicks, no wizard.</p>
    <p>
      <a class="button" href="./ca.reg" download>Install (just me)</a>
      <a class="button secondary" href="./ca.reg?scope=machine" download>Install (all users, needs admin)</a>
    </p>
    <p class="hint">
      Double-click the downloaded <code>.reg</code> file. Windows will ask
      <em>"Are you sure you want to add this to the registry?"</em> — click
      <strong>Yes</strong>. That's it. Restart your browser, and every internal
      site signed by this CA will show up as trusted.
    </p>

    <p style="margin-top: 1.8rem;"><strong>Manual alternative:</strong> download the raw certificate.</p>
    <p><a class="button secondary" href="./ca.cer" download>Download .cer</a></p>
    <p class="hint">
      Double-click the <code>.cer</code> → <em>Install Certificate…</em> → pick
      <em>Current User</em> (or Local Machine for everyone). On the store page,
      <strong>do NOT</strong> leave "Automatically select" selected — Windows
      will file it as an Intermediate CA and browsers won't trust it. Instead:
    </p>
    <ol class="hint">
      <li>Select <em>Place all certificates in the following store</em></li>
      <li>Click <em>Browse…</em></li>
      <li>Choose <em>Trusted Root Certification Authorities</em></li>
      <li>OK → Next → Finish → Yes to the security warning</li>
    </ol>
  </div>

  <div id="linux" class="platform">
    <div class="detected">Detected: Linux</div>
    <p><a class="button" href="./ca.pem" download>Download PEM</a></p>
    <p class="hint">Install at the system level (Debian/Ubuntu):</p>
    <p><code>sudo cp root_ca.crt /usr/local/share/ca-certificates/local-dev-ca.crt
       &amp;&amp; sudo update-ca-certificates</code></p>
    <p class="hint">Firefox has its own store — import via
       Settings → Privacy &amp; Security → Certificates → View Certificates → Authorities → Import.</p>
  </div>

  <div id="other" class="platform">
    <p>Download the root CA in your preferred format:</p>
    <p>
      <a class="button" href="./ca.pem" download>PEM (.pem)</a>
      <a class="button secondary" href="./ca.cer" download>DER (.cer)</a>
      <a class="button secondary" href="./ca.mobileconfig" download>Apple (.mobileconfig)</a>
    </p>
  </div>

  <p class="hint" style="margin-top: 2rem; border-top: 1px solid #eee; padding-top: 1rem;">
    CA: <code>__CA_SUBJECT__</code><br>
    SHA-256 fingerprint:<br><code>__CA_FINGERPRINT__</code>
  </p>
</div>
<script>
  const ua = navigator.userAgent;
  let platform = 'other';
  if (/iPhone|iPad/.test(ua))          platform = 'ios';
  else if (/Mac OS X/.test(ua))        platform = 'mac';
  else if (/Windows/.test(ua))         platform = 'windows';
  else if (/Linux|CrOS/.test(ua))      platform = 'linux';
  document.getElementById(platform).classList.add('active');
</script>
</body>
</html>
"""


def _ca_subject_and_fingerprint() -> tuple[str, str]:
    try:
        return tls.root_ca_subject(), tls.root_ca_fingerprint_sha256()
    except Exception:
        return "Local Dev CA", "(unavailable)"


@router.get("/", response_class=HTMLResponse)
@router.get("/install", response_class=HTMLResponse)
def install_page(request: Request) -> HTMLResponse:
    subj, fp = _ca_subject_and_fingerprint()
    html = (_INSTALL_PAGE_HTML
            .replace("__CA_SUBJECT__", subj)
            .replace("__CA_FINGERPRINT__", fp)
            .replace("__PEM_FILENAME__", _PEM_FILENAME))
    return HTMLResponse(html)


# Filenames are referenced by the install-banner clipboard command — keep
# the DOWNLOAD filename in sync with what the JS pastes.
_PEM_FILENAME = "local-dev-ca.pem"
_CER_FILENAME = "local-dev-ca.cer"


@router.get("/ca.pem")
def ca_pem() -> Response:
    return Response(
        content=tls.root_ca_pem_bytes(),
        media_type="application/x-pem-file",
        headers={
            "Content-Disposition": f'attachment; filename="{_PEM_FILENAME}"',
        },
    )


@router.get("/ca.cer")
def ca_cer() -> Response:
    return Response(
        content=tls.root_ca_der_bytes(),
        media_type="application/pkix-cert",
        headers={
            "Content-Disposition": f'attachment; filename="{_CER_FILENAME}"',
        },
    )


@router.get("/ca.mobileconfig")
def ca_mobileconfig() -> Response:
    return Response(
        content=tls.build_mobileconfig(),
        media_type="application/x-apple-aspen-config",
        headers={"Content-Disposition": 'attachment; filename="local-dev-ca.mobileconfig"'},
    )


@router.get("/install-cmd-macos.txt", response_class=PlainTextResponse)
def install_cmd_macos() -> "PlainTextResponse":
    """Self-contained macOS install one-liner with embedded cert.
    Banner JS fetches this and copies to clipboard on click."""
    return PlainTextResponse(tls.macos_install_oneliner())


@router.get("/install-cmd-linux.txt", response_class=PlainTextResponse)
def install_cmd_linux() -> "PlainTextResponse":
    return PlainTextResponse(tls.linux_install_oneliner())


@router.get("/ca-install.command")
def ca_install_command() -> Response:
    """macOS .command shell script — preferred over .mobileconfig because
    unsigned config profiles can't grant SSL trust on macOS without admin/MDM.
    Double-click in Finder → Terminal runs → user types password → SSL trust
    granted system-wide."""
    return Response(
        content=tls.build_macos_install_command(),
        media_type="application/x-sh",
        headers={
            "Content-Disposition":
                'attachment; filename="install-local-dev-ca.command"',
        },
    )


_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c63000100000005000107a72fcf3300"
    "0000004945e442608200"
)


@router.get("/_pixel")
def trust_pixel() -> Response:
    """Tiny PNG used by the install banner to probe whether the browser
    actually trusts our CA (cross-origin image load forces fresh TLS check)."""
    return Response(
        content=_PIXEL_PNG,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/ca.reg")
def ca_reg(request: Request) -> Response:
    scope = request.query_params.get("scope", "user")
    if scope not in ("user", "machine"):
        scope = "user"
    return Response(
        content=tls.build_windows_reg(scope=scope),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition":
                f'attachment; filename="local-dev-ca-{scope}.reg"'
        },
    )
