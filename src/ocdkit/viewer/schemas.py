"""Pydantic request/response schemas for the viewer's HTTP API.

Consolidating all request models here gives:
* one place to evolve the wire format
* automatic OpenAPI docs at ``/docs``
* consistent 422 validation errors instead of hand-rolled 400s

Most request models accept ``model_config = ConfigDict(extra="allow")`` so that
forward-compatible payload extensions (e.g. plugin-specific knobs) don't need
schema updates here. Keys we actually consume are typed; everything else
passes through.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ----- shared envelope -----------------------------------------------------


class ErrorBody(BaseModel):
    """Standard error response envelope (returned by exception handlers)."""

    ok: bool = False
    error: str
    detail: Optional[Any] = None


class OkBody(BaseModel):
    """Standard success envelope used where there's no other payload."""

    ok: bool = True


# ----- log routes ----------------------------------------------------------


class LogPayload(BaseModel):
    """Accept either a batch (entries / messages) or a single message."""

    entries: Optional[list[Any]] = None
    messages: Optional[list[Any]] = None
    message: Optional[str] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _at_least_one(self) -> "LogPayload":
        if self.entries is None and self.messages is None and self.message is None:
            raise ValueError("LogPayload requires one of: entries, messages, message")
        return self


# ----- system routes -------------------------------------------------------


class UseGpuPayload(BaseModel):
    use_gpu: bool


# ----- session / image routes ---------------------------------------------


class SessionPayload(BaseModel):
    """Minimum payload that carries a session id."""

    sessionId: str = Field(min_length=1)
    model_config = ConfigDict(extra="allow")


class OpenImagePayload(SessionPayload):
    path: Optional[str] = None
    direction: Optional[str] = None  # "next" | "prev"

    @model_validator(mode="after")
    def _path_or_direction(self) -> "OpenImagePayload":
        if not self.path and self.direction not in {"next", "prev"}:
            raise ValueError("path or direction (next/prev) required")
        return self


class OpenImageFolderPayload(SessionPayload):
    path: str = Field(min_length=1)


class SaveStatePayload(SessionPayload):
    viewerState: dict[str, Any]
    imagePath: Optional[str] = None


# ----- segmentation routes ------------------------------------------------


class SegmentPayload(SessionPayload):
    """Settings for /api/segment and /api/resegment.

    Plugin-specific keys (mask_threshold, flow_threshold, niter, etc.) are
    accepted via ``extra="allow"`` and forwarded to ``plugin.run(image, params)``.
    """

    mode: Optional[str] = None  # "recompute" routes through resegment hook
    use_gpu: Optional[bool] = None


# ----- plugin routes ------------------------------------------------------


class SelectPluginPayload(BaseModel):
    name: str = Field(min_length=1)


class WarmupPayload(BaseModel):
    model: str = Field(min_length=1)


class OneShotSegmentPayload(BaseModel):
    """One-shot /api/plugins/{name}/segment — image as base64 + free-form params."""

    image: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


# ----- mask post-processing routes ----------------------------------------


class MaskOpPayload(BaseModel):
    """Common shape: a base64 mask + its dimensions."""

    mask: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    model_config = ConfigDict(extra="allow")


class NColorFromMaskPayload(MaskOpPayload):
    expand: bool = True


class AffinityGraphFragment(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    steps: list[list[int]]
    encoded: str = Field(min_length=1)


class RelabelFromAffinityPayload(MaskOpPayload):
    affinityGraph: AffinityGraphFragment
