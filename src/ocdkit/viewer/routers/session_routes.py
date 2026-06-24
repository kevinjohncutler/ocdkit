"""Image / session routes: open/navigate, native dialog, save state, raw bytes."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response

from ..dependencies import get_session_state
from ..exceptions import BadRequest, NotFound, UnknownSession
from ..routes import _choose_path_osascript
from ..schemas import (
    OpenImageFolderPayload,
    OpenImagePayload,
    OpenMaskPayload,
    SaveStatePayload,
    SessionPayload,
)
from ..session import SESSION_MANAGER, SessionState

router = APIRouter(prefix="/api")

# Per-process scratch directory for uploaded images. Lives until the process
# exits (uvicorn reload re-creates one). Avoids shipping client-side bytes
# to a permanent location.
_UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="ocdkit_upload_"))


# ----- helpers ------------------------------------------------------------


def _native_picker(kind: str) -> Optional[str]:
    """Open the OS-native file/folder picker. Returns the selected path or None."""
    if sys.platform == "darwin":
        return _choose_path_osascript(kind)
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            root.update()
        except Exception:
            pass
        if kind == "file":
            return filedialog.askopenfilename(
                title="Select image",
                filetypes=[
                    ("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                    ("All files", "*.*"),
                ],
                parent=root,
            ) or None
        return filedialog.askdirectory(parent=root) or None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


# ----- routes -------------------------------------------------------------


@router.post("/open_image")
def api_open_image(
    payload: OpenImagePayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    try:
        if payload.path:
            SESSION_MANAGER.set_image(state, Path(payload.path))
        else:
            target = SESSION_MANAGER.navigate(
                state, 1 if payload.direction == "next" else -1
            )
            if target is None:
                raise NotFound("no_image")
            SESSION_MANAGER.set_image(state, target)
    except FileNotFoundError as exc:
        raise NotFound("file_not_found", detail=str(exc)) from exc
    config = SESSION_MANAGER.build_config(state, embed_image=False)
    return {"ok": True, "config": config}


@router.post("/open_image_folder")
def api_open_image_folder(
    payload: OpenImageFolderPayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    folder_path = Path(payload.path).expanduser().resolve()
    if folder_path.is_file():
        folder_path = folder_path.parent
    if not folder_path.exists() or not folder_path.is_dir():
        raise BadRequest("not_a_directory")
    files = SESSION_MANAGER._list_directory_images(folder_path)
    if not files:
        raise NotFound("no_images")
    target = state.current_path if (
        state.current_path and state.current_path.parent == folder_path
    ) else files[0]
    SESSION_MANAGER.set_image(state, target)
    config = SESSION_MANAGER.build_config(state, embed_image=False)
    return {"ok": True, "config": config}


@router.post("/select_image_file")
def api_select_image_file(
    payload: SessionPayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    file_path = _native_picker("file")
    if not file_path:
        raise BadRequest("cancelled")
    path_obj = Path(file_path).expanduser().resolve()
    if not path_obj.exists() or not path_obj.is_file():
        raise NotFound("file_not_found")
    SESSION_MANAGER.set_image(state, path_obj)
    config = SESSION_MANAGER.build_config(state, embed_image=False)
    return {"ok": True, "config": config}


@router.post("/upload_image")
async def api_upload_image(
    sessionId: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Accept a browser-uploaded image, save it to scratch, open it as the
    current image. Used by the client-side file picker so remote viewers
    work without a server-side native dialog (which needs a display)."""
    state = SESSION_MANAGER.get_or_create(sessionId)
    suffix = Path(file.filename or "upload.tif").suffix or ".tif"
    safe_stem = Path(file.filename or "upload").stem.replace("/", "_")
    target = _UPLOAD_DIR / f"{safe_stem}{suffix}"
    data = await file.read()
    target.write_bytes(data)
    SESSION_MANAGER.set_image(state, target)
    config = SESSION_MANAGER.build_config(state, embed_image=False)
    return {"ok": True, "config": config}


@router.post("/select_image_folder")
def api_select_image_folder(
    payload: SessionPayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    folder = _native_picker("folder")
    if not folder:
        raise BadRequest("cancelled")
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise BadRequest("not_a_directory")
    files = SESSION_MANAGER._list_directory_images(folder_path)
    if not files:
        raise NotFound("no_images")
    SESSION_MANAGER.set_image(state, files[0])
    config = SESSION_MANAGER.build_config(state, embed_image=False)
    return {"ok": True, "config": config}


@router.get("/image/{session_id}")
def api_image(session_id: str) -> Response:
    try:
        state = SESSION_MANAGER.get(session_id)
    except KeyError as exc:
        raise UnknownSession() from exc
    if not state.encoded_image_bytes:
        raise NotFound("no_image")
    return Response(
        content=state.encoded_image_bytes,
        media_type=state.encoded_image_mime,
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/volume_slice/{session_id}")
def api_volume_slice(session_id: str, z: int = 0, axis: int = 0) -> Response:
    """Serve slice ``z`` along ``axis`` (0=Z,1=Y,2=X) as PNG (2.5D navigation)."""
    try:
        state = SESSION_MANAGER.get(session_id)
    except KeyError as exc:
        raise UnknownSession() from exc
    png = SESSION_MANAGER.encode_slice_png(state, z, axis)
    if png is None:
        raise NotFound("not_a_volume")
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/open_mask")
def api_open_mask(
    payload: OpenMaskPayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    """Attach a label volume (from a file) to the current volume session."""
    try:
        SESSION_MANAGER.set_mask(state, Path(payload.path))
    except FileNotFoundError as exc:
        raise NotFound("file_not_found", detail=str(exc)) from exc
    except ValueError as exc:
        raise BadRequest("mask_mismatch", detail=str(exc)) from exc
    return {"ok": True}


@router.post("/select_mask_file")
def api_select_mask_file(
    payload: SessionPayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    """Native file picker → attach the chosen label volume."""
    file_path = _native_picker("file")
    if not file_path:
        raise BadRequest("cancelled")
    try:
        SESSION_MANAGER.set_mask(state, Path(file_path))
    except FileNotFoundError as exc:
        raise NotFound("file_not_found", detail=str(exc)) from exc
    except ValueError as exc:
        raise BadRequest("mask_mismatch", detail=str(exc)) from exc
    return {"ok": True}


@router.get("/mask_slice/{session_id}")
def api_mask_slice(session_id: str, z: int = 0, axis: int = 0) -> Response:
    """Raw label bytes for mask slice ``z`` along ``axis`` (2D mask overlay)."""
    try:
        state = SESSION_MANAGER.get(session_id)
    except KeyError as exc:
        raise UnknownSession() from exc
    res = SESSION_MANAGER.mask_slice(state, z, axis)
    if res is None:
        raise NotFound("no_mask")
    data, width, height, dtype = res
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "X-Mask-Width": str(width),
            "X-Mask-Height": str(height),
            "X-Mask-Dtype": dtype,
            "Cache-Control": "no-cache",
        },
    )


@router.post("/mask_slice/{session_id}")
async def api_put_mask_slice(session_id: str, request: Request, z: int = 0, axis: int = 0) -> dict:
    """Persist edited label bytes back into mask slice ``z`` along ``axis``."""
    try:
        state = SESSION_MANAGER.get(session_id)
    except KeyError as exc:
        raise UnknownSession() from exc
    dtype = request.headers.get("X-Mask-Dtype", "uint32")
    data = await request.body()
    try:
        SESSION_MANAGER.set_mask_slice(state, z, data, dtype, axis)
    except ValueError as exc:
        raise BadRequest("mask_write_failed", detail=str(exc)) from exc
    return {"ok": True}


@router.get("/volume_bundle/{session_id}")
def api_volume_bundle(session_id: str) -> dict:
    """Intensity-only 3D bundle for the loaded volume (renders before masks)."""
    try:
        state = SESSION_MANAGER.get(session_id)
    except KeyError as exc:
        raise UnknownSession() from exc
    bundle = SESSION_MANAGER.encode_volume_bundle(state)
    if bundle is None:
        raise NotFound("not_a_volume")
    return bundle


@router.post("/save_state")
def api_save_state(
    payload: SaveStatePayload,
    state: SessionState = Depends(get_session_state),
) -> dict:
    path_obj = Path(payload.imagePath).expanduser().resolve() if payload.imagePath else None
    SESSION_MANAGER.save_viewer_state(state, path_obj, payload.viewerState)
    return {"ok": True}
