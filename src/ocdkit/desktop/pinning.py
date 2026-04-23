"""
Cross-platform desktop integration for pywebview apps.

Provides taskbar/dock pinning, icon management, dark mode support, and
shell helpers so a pip-installed pywebview GUI behaves like a native app
on Windows, macOS, and Linux.

Core API is platform-agnostic — describe your app with :class:`AppIdentity`
and call :func:`setup_platform` before creating any windows::

    from ocdkit.desktop.pinning import AppIdentity, apply_early_dark_mode, setup_platform, set_window_icon

    APP = AppIdentity(
        name="MyApp",
        gui_entry_point="myapp-gui",
        icon_png=None,              # None → auto-generated default icon
        windows_app_id="Org.MyApp.Launcher",
        linux_app_id="myapp",
        macos_bundle_id="com.org.myapp",
    )

    apply_early_dark_mode()   # MUST be called before ``import webview``
    import webview

    setup_platform(APP)       # call before window creation
    window = webview.create_window(f"{APP.name} Launcher", ...)

    def on_start():
        set_window_icon(APP, window_title=f"{APP.name} Launcher")

    webview.start(func=on_start)

Dependencies
------------
The core module uses the standard library only; the default-icon generator
is the sole exception and lazily imports numpy + imagecodecs (already
ocdkit hard deps) on first use.  Platform-specific optional imports
(``ctypes``, ``winreg``, ``AppKit``, ``gi``) are imported lazily.
"""

from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "SYSTEM",
    "AppIdentity",
    "apply_early_dark_mode",
    "center_on_screen",
    "copy_to_clipboard",
    "is_dark_mode",
    "open_in_terminal",
    "setup_platform",
    "set_window_icon",
    "shell_executable",
]

SYSTEM = platform.system()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AppIdentity:
    """Everything that distinguishes one pywebview app from another.

    All platform integration functions accept an ``AppIdentity`` instead of
    reading module-level globals, making it trivial to reuse across apps.
    """

    name: str
    """Human-readable display name (e.g. ``"ocdkit-viewer"``)."""

    gui_entry_point: str
    """Name of the ``gui-scripts`` console entry point (e.g. ``"ocdkit-viewer-gui"``)."""

    windows_app_id: str
    """Dotted ``AppUserModelID`` for Windows taskbar grouping."""

    linux_app_id: str
    """Short lowercase token for ``WM_CLASS`` / ``StartupWMClass``."""

    macos_bundle_id: str
    """Reverse-DNS ``CFBundleIdentifier`` for the ``.app`` bundle."""

    icon_png: Optional[str] = None
    """Absolute path to the source PNG icon.  If ``None`` or missing on disk,
    a default icon (filled circle) is generated and cached at runtime."""

    version: str = "1.0"
    """Version string written into ``Info.plist`` / shortcut metadata."""

    description: str = ""
    """One-line description (Start Menu tooltip, ``.desktop`` Comment)."""

    categories: str = "Utility"
    """Semicolon-separated ``.desktop`` ``Categories`` value."""


# ---------------------------------------------------------------------------
# Derived paths (pure functions of AppIdentity — no module-level state)
# ---------------------------------------------------------------------------
def _local_dir(app: AppIdentity) -> str:
    """Writable directory for generated assets (ICO, wrapper exe, default icon, etc.)."""
    if SYSTEM == "Windows":
        root = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        return os.path.join(root, app.name)
    if SYSTEM == "Darwin":
        root = os.path.expanduser("~/Library/Application Support")
        return os.path.join(root, app.name)
    # Linux / other
    root = os.environ.get(
        "XDG_CACHE_HOME", os.path.expanduser("~/.cache")
    )
    return os.path.join(root, app.name)


def _icon_paths(app: AppIdentity):
    """Return (icon_png, icon_ico) paths inside the local dir."""
    d = _local_dir(app)
    return os.path.join(d, "icon.png"), os.path.join(d, "icon.ico")


def _generate_default_icon(dest_path: str, size: int = 256) -> bool:
    """Generate a simple default icon (filled circle on transparent background).

    Uses numpy + imagecodecs (both ocdkit hard dependencies) to write a PNG.
    Returns ``True`` on success.
    """
    try:
        import numpy as np
        import imagecodecs
    except ImportError:
        return False
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    cy = cx = size // 2
    radius = int(size * 0.44)
    yy, xx = np.ogrid[:size, :size]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    disk = dist2 <= radius ** 2
    # Feathered edge for a cleaner look at small sizes
    edge = (dist2 > (radius - 2) ** 2) & disk
    arr[disk] = (64, 128, 200, 255)   # blue-grey
    arr[edge] = (40, 96, 160, 255)    # slightly darker rim
    try:
        png_bytes = imagecodecs.png_encode(arr)
    except Exception:
        return False
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(png_bytes)
    except OSError:
        return False
    return True


def _resolve_source_png(app: AppIdentity) -> Optional[str]:
    """Return a usable source PNG path: the one on ``app.icon_png`` if it
    exists, otherwise a default icon auto-generated into the local dir.

    Returns ``None`` only if generation also failed (rare).
    """
    if app.icon_png and os.path.exists(app.icon_png):
        return app.icon_png
    # Fallback: auto-generate into local dir
    local_png, _ = _icon_paths(app)
    if os.path.exists(local_png):
        return local_png
    return local_png if _generate_default_icon(local_png) else None


def _resolve_gui_exe(entry_point: str):
    """Absolute path to a gui-scripts entry point, resolving Windows .bat shims."""
    exe = shutil.which(entry_point)
    if not exe:
        return None
    if SYSTEM == "Windows" and exe.lower().endswith(".bat"):
        scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
        real = os.path.join(scripts_dir, entry_point + ".exe")
        if os.path.exists(real):
            return real
    return exe


# ---------------------------------------------------------------------------
# Dark mode detection
# ---------------------------------------------------------------------------
def is_dark_mode() -> bool:
    """Return ``True`` if the OS is currently in dark mode."""
    try:
        if SYSTEM == "Darwin":
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip().lower() == "dark"
        elif SYSTEM == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            return val == 0
        elif SYSTEM == "Linux":
            for key in ("color-scheme", "gtk-theme"):
                result = subprocess.run(
                    ["gsettings", "get", "org.gnome.desktop.interface", key],
                    capture_output=True, text=True, timeout=5,
                )
                out = result.stdout.strip().lower()
                if key == "color-scheme" and "prefer-dark" in out:
                    return True
                if key == "gtk-theme" and "dark" in out:
                    return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Early dark mode init (Windows only — must run before ``import webview``)
# ---------------------------------------------------------------------------
def apply_early_dark_mode():
    """Opt the process into Windows dark mode before the webview backend loads.

    Call this **before** ``import webview``.  No-op on non-Windows platforms.
    """
    if SYSTEM != "Windows":
        return
    try:
        import ctypes
        ux = ctypes.WinDLL("uxtheme.dll")
        k32 = ctypes.WinDLL("kernel32.dll")
        k32.GetProcAddress.restype = ctypes.c_void_p
        k32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        # ordinal 135 = SetPreferredAppMode(AllowDark=1)
        ptr = k32.GetProcAddress(ux._handle, 135)
        if ptr:
            ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)(ptr)(1)
        # ordinal 136 = FlushMenuThemes
        ptr = k32.GetProcAddress(ux._handle, 136)
        if ptr:
            ctypes.WINFUNCTYPE(None)(ptr)()
        # ordinal 104 = RefreshImmersiveColorPolicyState
        ptr = k32.GetProcAddress(ux._handle, 104)
        if ptr:
            ctypes.WINFUNCTYPE(None)(ptr)()
    except Exception:
        pass


def _apply_windows_dark_mode_force():
    """Force dark mode on the process (``SetPreferredAppMode(ForceDark=2)``)."""
    try:
        import ctypes
        ux = ctypes.WinDLL("uxtheme.dll")
        k32 = ctypes.WinDLL("kernel32.dll")
        k32.GetProcAddress.restype = ctypes.c_void_p
        k32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ptr = k32.GetProcAddress(ux._handle, 135)
        if ptr:
            ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)(ptr)(2)
        ptr = k32.GetProcAddress(ux._handle, 136)
        if ptr:
            ctypes.WINFUNCTYPE(None)(ptr)()
        ptr = k32.GetProcAddress(ux._handle, 104)
        if ptr:
            ctypes.WINFUNCTYPE(None)(ptr)()
    except Exception:
        pass


def _allow_dark_mode_for_window(hwnd):
    """Apply per-window dark mode (ordinal 133 + ``DwmSetWindowAttribute``)."""
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32.dll")
        k32.GetProcAddress.restype = ctypes.c_void_p
        k32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ux = ctypes.WinDLL("uxtheme.dll")
        ptr = k32.GetProcAddress(ux._handle, 133)
        if ptr:
            ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool)(ptr)(hwnd, True)
        val = ctypes.c_int(1)
        ctypes.WinDLL("dwmapi.dll").DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(val), ctypes.sizeof(val))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Windows: ICO creation
# ---------------------------------------------------------------------------
def _ensure_ico(app: AppIdentity):
    """Copy PNG to local dir and create ``.ico`` wrapper (Windows only)."""
    if SYSTEM != "Windows":
        return
    local = _local_dir(app)
    os.makedirs(local, exist_ok=True)
    icon_png, icon_ico = _icon_paths(app)
    src_png = _resolve_source_png(app)
    if src_png and src_png != icon_png and not os.path.exists(icon_png):
        try:
            shutil.copy2(src_png, icon_png)
        except OSError:
            pass
    if os.path.exists(icon_ico):
        return
    if not os.path.exists(icon_png):
        return
    with open(icon_png, "rb") as f:
        png_data = f.read()
    header = struct.pack("<HHH", 0, 1, 1)
    data_offset = 6 + 16
    entry = struct.pack(
        "<BBBBHHII",
        128, 128, 0, 0, 1, 32,
        len(png_data), data_offset,
    )
    with open(icon_ico, "wb") as f:
        f.write(header + entry + png_data)


# ---------------------------------------------------------------------------
# Windows: C# wrapper exe
# ---------------------------------------------------------------------------
_CS_TEMPLATE = '''
using System;
using System.Diagnostics;
class P {{
    static void Main() {{
        try {{
            Process.Start(new ProcessStartInfo {{
                FileName = @"{target}",
                UseShellExecute = false,
                CreateNoWindow = true
            }});
        }} catch {{}}
    }}
}}
'''


def _build_wrapper_exe(app: AppIdentity):
    """Compile a tiny C# exe with the app icon embedded (Windows only)."""
    if SYSTEM != "Windows":
        return
    local = _local_dir(app)
    _, icon_ico = _icon_paths(app)
    wrapper_exe = os.path.join(local, f"{app.name}.exe")
    wrapper_target = os.path.join(local, ".wrapper_target")
    if not os.path.exists(icon_ico):
        return
    gui_exe = _resolve_gui_exe(app.gui_entry_point)
    if not gui_exe:
        return
    if os.path.exists(wrapper_exe) and os.path.exists(wrapper_target):
        try:
            with open(wrapper_target) as f:
                if f.read().strip() == gui_exe:
                    return
        except OSError:
            pass
    windir = os.environ.get("windir", r"C:\Windows")
    csc = None
    for arch in ("Framework64", "Framework"):
        net_dir = os.path.join(windir, "Microsoft.NET", arch)
        if not os.path.isdir(net_dir):
            continue
        for d in sorted(os.listdir(net_dir), reverse=True):
            candidate = os.path.join(net_dir, d, "csc.exe")
            if os.path.exists(candidate):
                csc = candidate
                break
        if csc:
            break
    if not csc:
        return
    os.makedirs(local, exist_ok=True)
    cs_path = os.path.join(local, "_launcher.cs")
    try:
        with open(cs_path, "w") as f:
            f.write(_CS_TEMPLATE.format(target=gui_exe))
        subprocess.run(
            [csc, "/nologo", "/target:winexe",
             f"/win32icon:{icon_ico}",
             f"/out:{wrapper_exe}",
             cs_path],
            capture_output=True, timeout=30,
        )
        if os.path.exists(wrapper_exe):
            with open(wrapper_target, "w") as f:
                f.write(gui_exe)
    except Exception:
        pass
    finally:
        if os.path.exists(cs_path):
            os.remove(cs_path)


# ---------------------------------------------------------------------------
# Windows: Start Menu shortcut
# ---------------------------------------------------------------------------
def _create_windows_shortcut(app: AppIdentity):
    """Create a Start Menu shortcut with matching ``AppUserModelID``."""
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return
    local = _local_dir(app)
    _, icon_ico = _icon_paths(app)
    wrapper_exe = os.path.join(local, f"{app.name}.exe")
    start_menu = os.path.join(
        appdata, "Microsoft", "Windows", "Start Menu", "Programs",
    )
    shortcut_path = os.path.join(start_menu, f"{app.name}.lnk")
    target = wrapper_exe if os.path.exists(wrapper_exe) else _resolve_gui_exe(app.gui_entry_point)
    if not target:
        return
    if os.path.exists(shortcut_path):
        return
    try:
        import win32com.client
        from win32com.shell import shellcon
        from win32com.propsys import propsys, pscon
        import pythoncom

        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = target
        shortcut.IconLocation = f"{icon_ico},0"
        shortcut.Description = app.description
        shortcut.save()

        store = propsys.SHGetPropertyStoreFromParsingName(
            shortcut_path, None, shellcon.GPS_READWRITE,
            propsys.IID_IPropertyStore,
        )
        store.SetValue(
            pscon.PKEY_AppUserModel_ID,
            propsys.PROPVARIANTType(app.windows_app_id, pythoncom.VT_LPWSTR),
        )
        store.Commit()
    except Exception:
        ps_script = (
            '$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{shortcut_path}"); '
            f'$s.TargetPath = "{target}"; '
            f'$s.IconLocation = "{icon_ico},0"; '
            f'$s.Description = "{app.description}"; '
            '$s.Save()'
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Windows: COM STA folder picker (no flash)
# ---------------------------------------------------------------------------
def pick_folder_sta():
    """Open ``IFileOpenDialog`` on a dedicated COM STA thread (Windows only).

    Returns the selected folder path, or ``None`` if cancelled.
    """
    import ctypes
    import threading
    from ctypes import (
        c_buffer, c_void_p, byref, sizeof, cast,
        WINFUNCTYPE, POINTER, HRESULT, windll,
    )
    from ctypes.wintypes import DWORD, HWND, LPWSTR

    CLSID_FileOpenDialog = c_buffer(
        b'\x9c\x5a\x1c\xdc\x8a\xe8\xde\x4d'
        b'\xa5\xa1\x60\xf8\x2a\x20\xae\xf7', 16)
    IID_IFileOpenDialog = c_buffer(
        b'\x88\x72\x7c\xd5\xad\xd4\x68\x47'
        b'\xbe\x02\x9d\x96\x95\x32\xd9\x60', 16)

    PSIZE = sizeof(POINTER(c_void_p))
    c_mem_p = POINTER(c_void_p)

    def _vtfunc(obj, idx, res, *args):
        addr = cast(obj.contents.value + idx * PSIZE, POINTER(WINFUNCTYPE(res, *args)))
        return addr.contents

    def _release(obj):
        if cast(obj, c_void_p).value:
            cast(obj.contents.value + 2 * PSIZE, POINTER(WINFUNCTYPE(HRESULT, c_mem_p))).contents(obj)

    result = [None]

    def _run():
        try:
            ole32 = windll.ole32
            ole32.CoInitializeEx(None, 0x2)
            try:
                com = c_mem_p()
                hr = ole32.CoCreateInstance(
                    byref(CLSID_FileOpenDialog), None, 0x1,
                    byref(IID_IFileOpenDialog), byref(com))
                if hr < 0:
                    return
                try:
                    flags = DWORD()
                    _vtfunc(com, 10, HRESULT, c_mem_p, POINTER(DWORD))(com, byref(flags))
                    flags.value |= 0x20 | 0x40 | 0x800
                    _vtfunc(com, 9, HRESULT, c_mem_p, DWORD)(com, flags)
                    _vtfunc(com, 17, HRESULT, c_mem_p, LPWSTR)(com, LPWSTR("Select Data Folder"))
                    hr = _vtfunc(com, 3, HRESULT, c_mem_p, HWND)(com, HWND(0))
                    if hr < 0:
                        return
                    result_item = c_mem_p()
                    hr = _vtfunc(com, 20, HRESULT, c_mem_p, POINTER(c_mem_p))(com, byref(result_item))
                    if hr < 0 or not cast(result_item, c_void_p).value:
                        return
                    try:
                        path = LPWSTR()
                        hr = _vtfunc(result_item, 5, HRESULT, c_mem_p, DWORD, POINTER(LPWSTR))(
                            result_item, DWORD(0x80058000), byref(path))
                        if hr >= 0 and path.value:
                            result[0] = path.value
                            ole32.CoTaskMemFree(path)
                    finally:
                        _release(result_item)
                finally:
                    _release(com)
            finally:
                ole32.CoUninitialize()
        except Exception:
            pass

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    return result[0]


# ---------------------------------------------------------------------------
# Linux: .desktop file
# ---------------------------------------------------------------------------
def _create_linux_desktop_entry(app: AppIdentity):
    """Install a ``.desktop`` file so GNOME/KDE match the window to its icon."""
    target = _resolve_gui_exe(app.gui_entry_point)
    if not target:
        return
    icon_png = _resolve_source_png(app)
    if not icon_png:
        return
    desktop_file = os.path.expanduser(
        f"~/.local/share/applications/{app.linux_app_id}.desktop"
    )
    desired = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={app.name}\n"
        f"Comment={app.description}\n"
        f"Exec={target}\n"
        f"Icon={icon_png}\n"
        "Terminal=false\n"
        f"Categories={app.categories};\n"
        f"StartupWMClass={app.linux_app_id}\n"
    )
    try:
        os.makedirs(os.path.dirname(desktop_file), exist_ok=True)
        if os.path.exists(desktop_file):
            with open(desktop_file) as f:
                if f.read() == desired:
                    return
        with open(desktop_file, "w") as f:
            f.write(desired)
        os.chmod(desktop_file, 0o755)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# macOS: .icns + .app bundle
# ---------------------------------------------------------------------------
def _create_macos_icns(dest_path: str, icon_png_src: str) -> bool:
    """Convert a PNG to ``.icns`` using macOS built-in ``sips`` + ``iconutil``."""
    if not os.path.exists(icon_png_src):
        return os.path.exists(dest_path)
    if os.path.exists(dest_path):
        if os.path.getmtime(dest_path) >= os.path.getmtime(icon_png_src):
            return True
        try:
            os.remove(dest_path)
        except OSError:
            return True
    import tempfile

    iconset = tempfile.mkdtemp(suffix=".iconset")
    try:
        for size in (16, 32, 128, 256, 512):
            subprocess.run(
                ["sips", "-z", str(size), str(size), icon_png_src,
                 "--out", os.path.join(iconset, f"icon_{size}x{size}.png")],
                capture_output=True, timeout=10,
            )
            if size <= 256:
                subprocess.run(
                    ["sips", "-z", str(size * 2), str(size * 2), icon_png_src,
                     "--out", os.path.join(iconset, f"icon_{size}x{size}@2x.png")],
                    capture_output=True, timeout=10,
                )
        result = subprocess.run(
            ["iconutil", "-c", "icns", iconset, "-o", dest_path],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(iconset, ignore_errors=True)


def _create_macos_app_bundle(app: AppIdentity):
    """Create ``~/Applications/<name>.app`` for dock pinning."""
    target = _resolve_gui_exe(app.gui_entry_point)
    if not target:
        return
    src_png = _resolve_source_png(app)
    if not src_png:
        return

    bundle = os.path.expanduser(f"~/Applications/{app.name}.app")
    contents = os.path.join(bundle, "Contents")
    macos_dir = os.path.join(contents, "MacOS")
    resources_dir = os.path.join(contents, "Resources")
    launcher = os.path.join(macos_dir, app.name)
    icns = os.path.join(resources_dir, "icon.icns")
    plist = os.path.join(contents, "Info.plist")

    if os.path.exists(launcher) and os.path.exists(icns):
        icns_fresh = (
            not os.path.exists(src_png)
            or os.path.getmtime(icns) >= os.path.getmtime(src_png)
        )
        if icns_fresh:
            try:
                with open(launcher) as f:
                    if target in f.read():
                        return
            except OSError:
                pass

    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)

    try:
        with open(launcher, "w") as f:
            f.write(f"#!/bin/bash\nexec \"{target}\"\n")
        os.chmod(launcher, 0o755)
    except Exception:
        return

    _create_macos_icns(icns, src_png)

    try:
        with open(plist, "w") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
                ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                '<plist version="1.0">\n<dict>\n'
                '  <key>CFBundleExecutable</key>\n'
                f'  <string>{app.name}</string>\n'
                '  <key>CFBundleIconFile</key>\n'
                '  <string>icon</string>\n'
                '  <key>CFBundleIdentifier</key>\n'
                f'  <string>{app.macos_bundle_id}</string>\n'
                '  <key>CFBundleName</key>\n'
                f'  <string>{app.name}</string>\n'
                '  <key>CFBundlePackageType</key>\n'
                '  <string>APPL</string>\n'
                '  <key>CFBundleShortVersionString</key>\n'
                f'  <string>{app.version}</string>\n'
                '  <key>LSMinimumSystemVersion</key>\n'
                '  <string>10.13</string>\n'
                '</dict>\n</plist>\n'
            )
    except Exception:
        return

    try:
        subprocess.run(
            ["/System/Library/Frameworks/CoreServices.framework/Frameworks/"
             "LaunchServices.framework/Support/lsregister",
             "-f", bundle],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API: setup_platform / set_window_icon
# ---------------------------------------------------------------------------
def setup_platform(app: AppIdentity):
    """One-call platform integration — invoke before creating any windows.

    * **Windows** — sets ``AppUserModelID``, applies dark mode, creates
      wrapper exe + Start Menu shortcut, generates ``.ico``.
    * **macOS** — creates ``~/Applications/<name>.app`` bundle.
    * **Linux** — pins ``WM_CLASS`` via ``GLib.set_prgname()``, installs
      ``.desktop`` file.
    """
    if SYSTEM == "Windows":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                app.windows_app_id)
        except Exception:
            pass
        if is_dark_mode():
            _apply_windows_dark_mode_force()
        _ensure_ico(app)
        _build_wrapper_exe(app)
        _create_windows_shortcut(app)
    elif SYSTEM == "Darwin":
        _create_macos_app_bundle(app)
    elif SYSTEM == "Linux":
        try:
            import gi
            gi.require_version("GLib", "2.0")
            from gi.repository import GLib
            GLib.set_prgname(app.linux_app_id)
        except Exception:
            pass
        _create_linux_desktop_entry(app)


def set_window_icon(app: AppIdentity, *, window_title: str = ""):
    """Set the window/dock icon after the pywebview window is fully created.

    Parameters
    ----------
    app:
        The application identity.
    window_title:
        The exact window title (used on Windows to find the HWND via
        ``FindWindowW``).  Ignored on other platforms.
    """
    icon_png, icon_ico = _icon_paths(app)
    src_png = _resolve_source_png(app)

    if SYSTEM == "Darwin":
        # When launched from ~/Applications/<name>.app, LaunchServices already
        # associates this process with the bundle's .icns — overriding via
        # setApplicationIconImage_ replaces it with a single-resolution
        # rasterized copy that the Dock renders differently. Skip the override
        # when a bundle is present; only fall back to the PNG for direct
        # terminal launches where no bundle association exists.
        bundle_path = os.path.expanduser(f"~/Applications/{app.name}.app")
        if not os.path.exists(bundle_path) and src_png:
            try:
                from AppKit import NSApplication, NSImage
                ns_image = NSImage.alloc().initWithContentsOfFile_(src_png)
                if ns_image:
                    NSApplication.sharedApplication().setApplicationIconImage_(ns_image)
            except Exception:
                pass

    elif SYSTEM == "Windows":
        try:
            import ctypes
            import time

            user32 = ctypes.windll.user32
            time.sleep(0.4)
            hwnd = user32.FindWindowW(None, window_title) if window_title else 0
            if not hwnd:
                return
            IMAGE_ICON = 1
            LR_LOADFROMFILE = 0x00000010
            icon_h = user32.LoadImageW(
                None, icon_ico, IMAGE_ICON, 0, 0, LR_LOADFROMFILE
            )
            if icon_h:
                WM_SETICON = 0x0080
                user32.SendMessageW(hwnd, WM_SETICON, 0, icon_h)
                user32.SendMessageW(hwnd, WM_SETICON, 1, icon_h)
            if is_dark_mode():
                _allow_dark_mode_for_window(hwnd)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shell / terminal helpers
# ---------------------------------------------------------------------------
def shell_executable() -> str:
    """Return the preferred shell (``pwsh`` / ``cmd`` on Windows, ``$SHELL`` elsewhere)."""
    if SYSTEM == "Windows":
        for ps in ("pwsh", "powershell"):
            if any(
                os.path.isfile(os.path.join(d, ps + ".exe"))
                for d in os.environ.get("PATH", "").split(os.pathsep)
            ):
                return ps
        return "cmd"
    return os.environ.get("SHELL", "/bin/bash")


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def open_in_terminal(command: str, cwd: str):
    """Open *command* in a new platform-native terminal window."""
    if SYSTEM == "Darwin":
        script = (
            f'tell application "Terminal"\n'
            f"  activate\n"
            f'  do script "cd {_sh_quote(cwd)} && {command}"\n'
            f"end tell"
        )
        subprocess.Popen(["osascript", "-e", script])

    elif SYSTEM == "Windows":
        import base64
        shell = shell_executable()
        if shell == "cmd":
            subprocess.Popen(
                f'start cmd /k "cd /d "{cwd}" && {command}"', shell=True
            )
        else:
            ps_code = f"Set-Location '{cwd}'; {command}"
            encoded = base64.b64encode(
                ps_code.encode("utf-16-le")
            ).decode("ascii")
            subprocess.Popen(
                f"start {shell} -NoExit -EncodedCommand {encoded}",
                shell=True,
            )
    else:
        for term in ("gnome-terminal", "xfce4-terminal", "konsole", "xterm"):
            if any(
                os.path.isfile(os.path.join(d, term))
                for d in os.environ.get("PATH", "").split(os.pathsep)
            ):
                if term == "gnome-terminal":
                    subprocess.Popen(
                        [term, "--working-directory", cwd, "--",
                         "bash", "-c", command + "; exec bash"],
                    )
                elif term == "xterm":
                    subprocess.Popen(
                        [term, "-hold", "-e",
                         f"cd {_sh_quote(cwd)} && {command}"],
                    )
                else:
                    subprocess.Popen(
                        [term, "-e",
                         f"cd {_sh_quote(cwd)} && {command}"],
                    )
                return
        subprocess.Popen(["bash", "-c", command], cwd=cwd)


def copy_to_clipboard(text: str) -> bool:
    """Copy *text* to the system clipboard.  Returns ``True`` on success."""
    try:
        if SYSTEM == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        elif SYSTEM == "Windows":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(
                ["clip"], input=text.encode(), check=True,
                startupinfo=si,
            )
        else:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode(), check=True,
            )
        return True
    except Exception:
        return False


def center_on_screen(width: int, height: int):
    """Return ``(x, y)`` to centre a window of *width* x *height*, or ``(None, None)``."""
    try:
        if SYSTEM == "Darwin":
            from AppKit import NSScreen
            frame = NSScreen.mainScreen().frame()
            sx, sy = int(frame.size.width), int(frame.size.height)
        elif SYSTEM == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            sx, sy = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        else:
            sx, sy = 1920, 1080
        return (sx - width) // 2, (sy - height) // 2
    except Exception:
        return None, None
