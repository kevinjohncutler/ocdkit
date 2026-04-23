# Making a pywebview App Look Like a Native Desktop App

> **Implementation**: all of the platform-integration code described in
> this guide is implemented in [`ocdkit.desktop.pinning`](../src/ocdkit/desktop/pinning.py).
> Use it via::
>
> ```python
> from ocdkit.desktop.pinning import (
>     AppIdentity, apply_early_dark_mode, setup_platform,
>     set_window_icon, relaunch_via_bundle,
> )
> ```
>
> This doc captures the *why* — the OS-specific quirks and the reasoning
> behind each step — so future maintainers can understand the design.

This guide describes how to make a pip-installed Python GUI app (using pywebview) appear as a proper desktop application with a custom icon on each OS:

- **Windows** — pinnable taskbar icon via `AppUserModelID`, a C# wrapper exe, and a Start Menu shortcut.
- **macOS** — pinnable dock icon via a `.app` bundle with `.icns` icon and `Info.plist`.
- **Linux** — correct icon in the GNOME dock / KDE taskbar via a `.desktop` file and a stable `WM_CLASS`.

> Each platform section is independent — wire up only the ones you ship to.

---

## Windows: Pinnable Taskbar Icon

## The Problem

When you run a pywebview app via a pip-installed entry point (e.g., `myapp.exe` in Python's `Scripts/` directory), Windows shows the generic Python icon in the taskbar. Pinning it pins "Python", not your app. Clicking the pinned icon doesn't relaunch your app.

## The Solution (3 Parts)

You need three things working together:

1. **AppUserModelID** on the running process (so Windows treats it as its own app)
2. **A compiled wrapper `.exe`** with your icon embedded (so the taskbar shows your icon)
3. **A Start Menu shortcut** with matching AppUserModelID (so pinning works correctly)

### Prerequisites

- `pywin32` — for setting AppUserModelID on the shortcut via `IPropertyStore`
- `.NET Framework` — ships with Windows; provides `csc.exe` for compiling the wrapper
- An `.ico` file — your app icon

---

## Step 1: Set AppUserModelID on the Process

Call this **before** creating any windows:

```python
import sys
import platform

APP_ID = "YourOrg.YourApp.Launcher"  # unique dotted string

if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass
```

## Step 2: Create a `gui-scripts` Entry Point

In `pyproject.toml`, add a `gui-scripts` entry alongside your regular `scripts`. The `gui-scripts` entry uses `pythonw.exe` (no console window):

```toml
[project.scripts]
myapp = "myapp.cli:main"

[project.gui-scripts]
myapp-gui = "myapp.launcher:main"
```

The user types `myapp` in terminal. The shortcut targets `myapp-gui.exe` (no console flash).

## Step 3: Compile a Wrapper `.exe` with Embedded Icon

The `myapp-gui.exe` created by pip is a generic setuptools stub with Python's icon. You can't modify it without breaking it. Instead, compile a tiny C# wrapper that launches `myapp-gui.exe` and has your icon embedded:

```python
import os
import shutil
import struct
import subprocess
import sys

LOCAL_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "YourApp",
)
ICON_ICO = os.path.join(LOCAL_DIR, "icon.ico")
WRAPPER_EXE = os.path.join(LOCAL_DIR, "YourApp.exe")

CS_TEMPLATE = '''
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

def build_wrapper_exe():
    """Compile a C# exe with embedded icon on first launch."""
    if os.path.exists(WRAPPER_EXE):
        return
    gui_exe = shutil.which("myapp-gui")
    if not gui_exe or not os.path.exists(ICON_ICO):
        return
    # Find csc.exe from .NET Framework
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
    os.makedirs(LOCAL_DIR, exist_ok=True)
    cs_path = os.path.join(LOCAL_DIR, "_launcher.cs")
    try:
        with open(cs_path, "w") as f:
            f.write(CS_TEMPLATE.format(target=gui_exe))
        subprocess.run(
            [csc, "/nologo", "/target:winexe",
             f"/win32icon:{ICON_ICO}",
             f"/out:{WRAPPER_EXE}",
             cs_path],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass
    finally:
        if os.path.exists(cs_path):
            os.remove(cs_path)
```

### Creating the `.ico` File

If you have a PNG icon, wrap it in ICO format:

```python
def write_ico_from_png(png_path, ico_path):
    """Wrap a PNG in a single-image ICO container (Vista+ format)."""
    with open(png_path, "rb") as f:
        png_data = f.read()
    header = struct.pack("<HHH", 0, 1, 1)
    data_offset = 6 + 16
    entry = struct.pack(
        "<BBBBHHII",
        0, 0, 0, 0, 1, 32,  # 0 width/height = 256x256
        len(png_data), data_offset,
    )
    with open(ico_path, "wb") as f:
        f.write(header + entry + png_data)
```

## Step 4: Create a Start Menu Shortcut with Matching AppUserModelID

This is the critical step. The shortcut must have the **same AppUserModelID** as the running process. The standard `WScript.Shell` COM object cannot set AppUserModelID — you need `IPropertyStore` via `pywin32`:

```python
def create_start_menu_shortcut():
    """Create a Start Menu shortcut with AppUserModelID for taskbar pinning."""
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return
    start_menu = os.path.join(
        appdata, "Microsoft", "Windows", "Start Menu", "Programs",
    )
    shortcut_path = os.path.join(start_menu, "YourApp.lnk")
    target = WRAPPER_EXE if os.path.exists(WRAPPER_EXE) else shutil.which("myapp-gui")
    if not target or os.path.exists(shortcut_path):
        return
    try:
        import win32com.client
        from win32com.shell import shellcon
        from win32com.propsys import propsys, pscon
        import pythoncom

        # Step 1: Create the .lnk
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = target
        shortcut.IconLocation = f"{ICON_ICO},0"
        shortcut.Description = "Your App Description"
        shortcut.save()

        # Step 2: Stamp AppUserModelID via IPropertyStore
        store = propsys.SHGetPropertyStoreFromParsingName(
            shortcut_path, None, shellcon.GPS_READWRITE,
            propsys.IID_IPropertyStore,
        )
        store.SetValue(
            pscon.PKEY_AppUserModel_ID,
            propsys.PROPVARIANTType(APP_ID, pythoncom.VT_LPWSTR),
        )
        store.Commit()
    except Exception:
        # Fallback without AppUserModelID (pinning icon may revert to Python)
        ps_script = (
            '$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{shortcut_path}"); '
            f'$s.TargetPath = "{target}"; '
            f'$s.IconLocation = "{ICON_ICO},0"; '
            '$s.Save()'
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
```

## Step 5: Wire It Up in Your Launcher

Call the setup functions early in your app's `main()`:

```python
def main():
    if platform.system() == "Windows":
        # Set AppUserModelID
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass
        # Build wrapper exe and create shortcut on first launch
        build_wrapper_exe()
        create_start_menu_shortcut()

    # ... create your pywebview window ...
    window = webview.create_window("Your App", ...)
    webview.start()
```

### How It Works

```text
User clicks pinned taskbar icon
  → Windows launches Start Menu shortcut
  → Shortcut targets YourApp.exe (C# wrapper with embedded icon)
  → Wrapper launches myapp-gui.exe (pip gui-scripts entry point)
  → myapp-gui.exe runs pythonw.exe → your Python launcher code
  → AppUserModelID matches → Windows associates the window with the pin
  → Taskbar shows your custom icon
```

### User Experience

After first launch:

1. Search "YourApp" in Start Menu — shows your custom icon
2. Right-click → **Pin to taskbar** — keeps the custom icon
3. Close the app, click the pinned icon — relaunches correctly

### Important Notes

- **Icon path must be local** — Windows shortcuts can't use UNC/network paths for icons. Copy the `.ico` to `%LOCALAPPDATA%\YourApp\` on first launch.
- **Don't modify setuptools exes** — Using `UpdateResource` on pip-generated `.exe` files corrupts the appended zip archive, causing "unable to find an appended archive" errors.
- **The wrapper exe bakes in the path** to `myapp-gui.exe`. If the user changes Python versions, the wrapper needs to be rebuilt. Track this with a marker file.
- **`pywin32` is required** for setting AppUserModelID on the shortcut. Without it, pinning will show Python's icon instead of yours. Include it as a Windows-only dependency:

  ```toml
  pywin32; sys_platform == "win32"
  ```

### Files Created on the User's Machine

```text
%LOCALAPPDATA%\YourApp\
  icon.png          # copied from package data
  icon.ico          # generated from PNG
  YourApp.exe       # compiled C# wrapper (5-6 KB)
  .wrapper_target   # tracks which myapp-gui.exe path is baked in

%APPDATA%\Microsoft\Windows\Start Menu\Programs\
  YourApp.lnk       # shortcut with AppUserModelID
```

---

## macOS: Dock Icon

### The Problem

When you run a pywebview app via a pip-installed entry point (e.g., `hiprcount-gui`), macOS shows "Python" in the dock. You can set the dock icon at runtime via `NSApplication.setApplicationIconImage_()` (and pywebview does look reasonable while running), but:

- Right-clicking shows "Python", not your app name
- **Keep in Dock** pins Python, not your app
- After quitting, the icon disappears — there's nothing to click to relaunch

macOS resolves dock identity from a `.app` bundle on disk. Without one, the process is just "Python" no matter what icon you set at runtime.

### The Solution

Create a lightweight `.app` bundle in `~/Applications/` — the macOS equivalent of a Linux `.desktop` file. The bundle contains:

1. **`Info.plist`** — metadata with `CFBundleIdentifier`, `CFBundleName`, and icon reference
2. **A shell script launcher** — `exec`s the pip-installed `hiprcount-gui` entry point
3. **An `.icns` icon** — converted from your PNG using macOS built-in tools

No Xcode, no code signing, no notarization needed for a local `~/Applications/` bundle.

### Step 1: Convert PNG to `.icns`

macOS requires icons in Apple's `.icns` format. Use the built-in `sips` (image resizer) and `iconutil` (icon compiler):

```python
import os
import shutil
import subprocess
import tempfile

ICON_PNG_SRC = "/abs/path/to/your/icon.png"

def create_icns(dest_path):
    """Convert a PNG to .icns using macOS built-in sips + iconutil."""
    if os.path.exists(dest_path):
        return True
    if not os.path.exists(ICON_PNG_SRC):
        return False
    iconset = tempfile.mkdtemp(suffix=".iconset")
    try:
        for size in (16, 32, 128, 256, 512):
            subprocess.run(
                ["sips", "-z", str(size), str(size), ICON_PNG_SRC,
                 "--out", os.path.join(iconset, f"icon_{size}x{size}.png")],
                capture_output=True, timeout=10,
            )
            # @2x variants for Retina displays
            if size <= 256:
                subprocess.run(
                    ["sips", "-z", str(size * 2), str(size * 2), ICON_PNG_SRC,
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
```

`sips` and `iconutil` ship with every macOS install (they're part of the OS, not Xcode). The `.iconset` directory must use the `.iconset` extension — `iconutil` refuses to process it otherwise.

### Step 2: Create the `.app` Bundle

The bundle is a directory tree that macOS recognizes as an application:

```python
MACOS_BUNDLE_ID = "com.yourorg.yourapp"
MACOS_APP_BUNDLE = os.path.expanduser("~/Applications/YourApp.app")

def create_app_bundle(gui_exe_path):
    """Create ~/Applications/YourApp.app for dock pinning."""
    if not gui_exe_path:
        return

    contents = os.path.join(MACOS_APP_BUNDLE, "Contents")
    macos_dir = os.path.join(contents, "MacOS")
    resources_dir = os.path.join(contents, "Resources")
    launcher = os.path.join(macos_dir, "YourApp")
    icns = os.path.join(resources_dir, "icon.icns")
    plist = os.path.join(contents, "Info.plist")

    # Skip rebuild if launcher already targets the same exe
    if os.path.exists(launcher) and os.path.exists(icns):
        try:
            with open(launcher) as f:
                if gui_exe_path in f.read():
                    return
        except OSError:
            pass

    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)

    # Launcher shell script
    with open(launcher, "w") as f:
        f.write(f'#!/bin/bash\nexec "{gui_exe_path}"\n')
    os.chmod(launcher, 0o755)

    # Icon
    create_icns(icns)

    # Info.plist
    with open(plist, "w") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
            ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n<dict>\n'
            '  <key>CFBundleExecutable</key>\n'
            '  <string>YourApp</string>\n'
            '  <key>CFBundleIconFile</key>\n'
            '  <string>icon</string>\n'
            '  <key>CFBundleIdentifier</key>\n'
            f'  <string>{MACOS_BUNDLE_ID}</string>\n'
            '  <key>CFBundleName</key>\n'
            '  <string>YourApp</string>\n'
            '  <key>CFBundlePackageType</key>\n'
            '  <string>APPL</string>\n'
            '  <key>CFBundleShortVersionString</key>\n'
            '  <string>1.0</string>\n'
            '  <key>LSMinimumSystemVersion</key>\n'
            '  <string>10.13</string>\n'
            '</dict>\n</plist>\n'
        )

    # Register with Launch Services so Spotlight / Launchpad discover it
    try:
        subprocess.run(
            ["/System/Library/Frameworks/CoreServices.framework/Frameworks/"
             "LaunchServices.framework/Support/lsregister",
             "-f", MACOS_APP_BUNDLE],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass
```

### Step 3: Set the Dock Icon at Runtime (Terminal Launches Only)

When the user launches from the terminal (not the `.app`), the dock still shows "Python". You can override the dock icon at runtime using PyObjC (which ships with macOS's system Python and is available via pip) — **but only do this when no `.app` bundle is present**:

```python
def set_dock_icon():
    """Set the dock icon for a terminal-launched process.

    When launched from ~/Applications/<name>.app, LaunchServices already
    associates this process with the bundle's .icns and the Dock renders it
    through Icon Services. `setApplicationIconImage_` bypasses Icon Services,
    so calling it — even with the bundle's own .icns — visibly downgrades the
    icon at launch. Skip the override whenever a bundle exists.
    """
    if os.path.exists(MACOS_APP_BUNDLE):
        return
    try:
        from AppKit import NSApplication, NSImage
        app = NSApplication.sharedApplication()
        ns_image = NSImage.alloc().initWithContentsOfFile_(ICON_PNG_SRC)
        if ns_image:
            app.setApplicationIconImage_(ns_image)
    except Exception:
        pass
```

This only affects the current session — the dock reverts to "Python" on next launch. The `.app` bundle is what makes **Keep in Dock** work permanently.

### Why `setApplicationIconImage_` Visibly Downgrades the Pinned Icon

When a `.app` bundle is launched, the Dock renders its icon through one of two pipelines:

| Path | When it runs | NSImage rep class | What the Dock gets |
|------|--------------|-------------------|--------------------|
| **Icon Services (bundle path)** | Process is associated with an `.app` bundle AND `setApplicationIconImage_` has not been called | `NSISIconImageRep` | Full Icon Services rendering — system shadow, tile padding, sub-pixel positioning, and additional synthesized standard sizes (18, 24, 36, 48 alongside the 16/32/128/256/512 in the `.icns`) |
| **Raw bitmap (runtime override)** | `setApplicationIconImage_` has been called | `NSBitmapImageRep` | Whatever bits are in the NSImage, blitted directly — no Icon Services compositing |

The shell-script `exec` in `Contents/MacOS/<Name>` preserves the LaunchServices bundle association across process replacement, so the Icon Services path is what's active when you click a pinned `.app`. Calling `setApplicationIconImage_` — even with `NSImage.initWithContentsOfFile_(icon.icns)` as the source — switches the Dock to the raw-bitmap path and you immediately lose the Icon Services compositing. Quitting the app releases the runtime override and the Dock reverts to the Icon Services rendering.

You can verify the pipeline difference at the REPL:

```python
from AppKit import NSImage, NSWorkspace
bundle = "/Users/you/Applications/YourApp.app"

ws = NSWorkspace.sharedWorkspace().iconForFile_(bundle)
raw = NSImage.alloc().initWithContentsOfFile_(bundle + "/Contents/Resources/icon.icns")

print([type(r).__name__ for r in ws.representations()])
# → ['NSISIconImageRep', 'NSISIconImageRep', ...]  (Icon Services)

print([type(r).__name__ for r in raw.representations()])
# → ['NSBitmapImageRep', 'NSBitmapImageRep', ...]  (raw bitmap)
```

The `NSISIconImageRep` class is private and cannot be constructed or copied by app code — which is why there is **no way to hand an Icon-Services-quality image to `setApplicationIconImage_`**. The only way to keep that rendering is to not call `setApplicationIconImage_` at all. Hence the bundle-exists guard in `set_dock_icon()` above.

Note that `NSWorkspace.iconForFile_()` does return an NSImage backed by `NSISIconImageRep`, but passing it to `setApplicationIconImage_` still produces degraded Dock rendering — the Icon Services path is gated on the Dock's own bundle lookup, not on the rep class of whatever image you hand it.

### Step 4: Wire It Up

```python
def main():
    if platform.system() == "Darwin":
        create_app_bundle(shutil.which("myapp-gui"))

    window = webview.create_window("Your App", ...)

    def on_start():
        set_dock_icon()

    webview.start(func=on_start)
```

### How It Works

```text
User right-clicks dock icon → Options → Keep in Dock
  → macOS remembers ~/Applications/YourApp.app
  → User quits the app — icon stays in the dock
  → User clicks pinned dock icon
  → macOS launches YourApp.app/Contents/MacOS/YourApp (shell script)
  → Shell script exec's hiprcount-gui (pip entry point)
  → Python + pywebview starts → window appears with custom dock icon
```

### Important Notes

- **`~/Applications/` is a standard macOS location** — it's searched by Spotlight and Launchpad alongside `/Applications/`. No admin privileges needed.
- **`CFBundleIconFile` omits the extension** — macOS appends `.icns` automatically when looking in `Resources/`.
- **The launcher script bakes in the path** to `myapp-gui`. If the user switches Python versions or venvs, the bundle needs rebuilding. Track this by checking the script content on each launch (same pattern as the Windows wrapper).
- **`lsregister`** forces Launch Services to index the bundle immediately. Without it, Spotlight/Launchpad may not find the app until the next periodic scan.
- **No code signing needed** — unsigned `.app` bundles work fine in `~/Applications/` for local use. Gatekeeper only blocks unsigned apps downloaded from the internet.
- **The `.icns` requires specific filenames** in the `.iconset` directory: `icon_16x16.png`, `icon_16x16@2x.png`, `icon_32x32.png`, etc. `iconutil` will error if the naming convention is wrong.
- **macOS caches dock icons aggressively** — if you update the `.icns` and the dock still shows the old icon, `touch` the `.app` bundle and run `lsregister -f` again, or log out and back in.

### Files Created on the User's Machine

```text
~/Applications/HiPR-Count.app/
  Contents/
    Info.plist           # bundle metadata with CFBundleIdentifier
    MacOS/
      HiPR-Count         # shell script that exec's hiprcount-gui
    Resources/
      icon.icns           # converted from PNG via sips + iconutil
```

---

## Linux: Dock / Taskbar Icon

### The Problem

Unlike macOS (which bundles a WebKit backend) and Windows (which uses Edge WebView2), **pywebview on Linux ships no rendering backend** — you must install one yourself, via either GTK+WebKit2 or Qt+QtWebEngine. A bare `pip install pywebview` will fail at runtime on Linux with a `ModuleNotFoundError: No module named 'gi'` (GTK path) or a Qt equivalent.

Even once a backend is installed, pywebview's `webview.start(icon=...)` parameter calls `Gtk.Window.set_icon_from_file()`. On **X11** this sets `_NET_WM_ICON` and works in the taskbar. On **Wayland + modern GNOME** (the Ubuntu 22.04+ default), GNOME Shell ignores per-window icons entirely — it resolves dock and Activities-overview icons by matching the window's `app_id` against an installed `.desktop` file. No `.desktop` file = generic fallback icon.

### The Solution (3 Parts)

1. **Install a GUI backend** — document a one-time apt + pip setup for Linux users.
2. **Pin a stable `app_id` / `WM_CLASS`** via `GLib.set_prgname()` *before* Gtk initialises. Without this, the app_id defaults to `argv[0]`, which differs depending on whether users launch via the CLI entry point or the `gui-scripts` entry point.
3. **Install a `.desktop` file** in `~/.local/share/applications/` with `StartupWMClass` matching the app_id above and `Icon=` pointing to an absolute PNG path.

### Step 1: Document the Backend Install

Add to your README's Linux install section (example is Ubuntu/Debian + GTK):

```bash
sudo apt install libgirepository-2.0-dev libcairo2-dev \
                 gir1.2-gtk-3.0 gir1.2-webkit2-4.1
$(pyenv which python) -m pip install PyGObject
```

The apt packages split into two groups:

| Package | Role |
|---------|------|
| `libgirepository-2.0-dev`, `libcairo2-dev` | Build headers so `pip install PyGObject` can compile from source |
| `gir1.2-gtk-3.0`, `gir1.2-webkit2-4.1` | Runtime GObject Introspection typelibs that pywebview loads via `gi.require_version()` |

Note the `-2.0-dev` is required by PyGObject ≥ 3.52 (older versions used `-1.0-dev`). The WebKit2 typelib package also pulls in the WebKitGTK runtime (~80–100 MB).

**Why pip alone can't do this**: typelib files (`.typelib`) are generated by distro tooling from C library metadata. They cannot be shipped on PyPI. pywebview's GTK backend imports them at runtime via `gi.repository`, so they must come from the system package manager.

Qt is a valid alternative (`pip install PyQt6 PyQt6-WebEngine`, no apt needed) — it's heavier (~150 MB of wheels) but avoids the system-package dance. Pick one; don't install both or pywebview will try GTK first and the Qt install becomes dead weight.

### Step 2: Pin a Stable `app_id`

pywebview's GTK backend uses plain `Gtk.Window`, so the X11 `WM_CLASS` / Wayland `app_id` is whatever GLib inherited from `argv[0]`. A user running `myapp` gets a different app_id than one running `myapp-gui`, which breaks the `.desktop` file matching.

Force a known name before pywebview starts Gtk:

```python
import platform

LINUX_APP_ID = "myapp"  # short, lowercase, no dots

def setup_linux_early():
    if platform.system() != "Linux":
        return
    try:
        import gi
        gi.require_version("GLib", "2.0")
        from gi.repository import GLib
        GLib.set_prgname(LINUX_APP_ID)
    except Exception:
        pass
```

Call `setup_linux_early()` **before** `webview.create_window(...)` — after `Gtk.init()` runs (which happens inside pywebview's backend), the prgname is locked in.

### Step 3: Install the `.desktop` File

Write it on every launch, but skip the write if the contents already match (so repeated launches don't churn the file's mtime):

```python
import os

ICON_PNG = "/abs/path/to/your/icon.png"  # must be absolute
LINUX_DESKTOP_FILE = os.path.expanduser(
    f"~/.local/share/applications/{LINUX_APP_ID}.desktop"
)

def create_linux_desktop_entry(exe_path):
    """Install ~/.local/share/applications/<app>.desktop so the shell can
    match the running window (via StartupWMClass) to its icon."""
    if not exe_path:
        return
    desired = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=MyApp\n"
        "Comment=MyApp GUI\n"
        f"Exec={exe_path}\n"
        f"Icon={ICON_PNG}\n"
        "Terminal=false\n"
        "Categories=Science;\n"
        f"StartupWMClass={LINUX_APP_ID}\n"
    )
    try:
        os.makedirs(os.path.dirname(LINUX_DESKTOP_FILE), exist_ok=True)
        if os.path.exists(LINUX_DESKTOP_FILE):
            with open(LINUX_DESKTOP_FILE) as f:
                if f.read() == desired:
                    return
        with open(LINUX_DESKTOP_FILE, "w") as f:
            f.write(desired)
        os.chmod(LINUX_DESKTOP_FILE, 0o755)
    except Exception:
        pass
```

Resolve `exe_path` via `shutil.which("myapp-gui")` — the same entry point you point a Windows shortcut at. Factoring this into a `_resolve_gui_exe()` helper lets the Windows and Linux code share it.

### How It Works

```text
User launches myapp
  → set_prgname("myapp") runs → Wayland app_id = "myapp"
  → .desktop file installed to ~/.local/share/applications/myapp.desktop
  → pywebview → Gtk → window appears
  → GNOME Shell sees app_id="myapp", looks up matching .desktop
  → finds StartupWMClass=myapp → reads Icon= → renders your PNG in the dock
```

### Important Notes

- **Absolute paths only** — `Exec=` and `Icon=` in a `.desktop` file must be absolute. Don't use `~` or `$HOME`.
- **Wayland sessions cache aggressively** — if the dock shows a generic icon the first time, the shell may have cached the no-.desktop state. Log out/in or restart the shell (`killall -HUP gnome-shell` works on X11 only). Subsequent launches pick it up immediately.
- **`StartupWMClass` is case-sensitive** on Wayland. If you set `GLib.set_prgname("MyApp")` but write `StartupWMClass=myapp`, matching silently fails.
- **Don't call `update-desktop-database`** — it's only needed for system-wide install paths. GNOME watches the user's applications dir via inotify.
- **Qt backend is equivalent** — if you use Qt instead of GTK, `QApplication.setApplicationName()` plays the role of `set_prgname()`. Same `.desktop` file, same `StartupWMClass`.

### Files Created on the User's Machine

```text
~/.local/share/applications/
  myapp.desktop     # installed on first launch, idempotent on subsequent launches
```

No generated wrapper binary is needed on Linux — the `.desktop` file points directly at `hiprcount-gui` (or equivalent), and GLib handles the rest.
