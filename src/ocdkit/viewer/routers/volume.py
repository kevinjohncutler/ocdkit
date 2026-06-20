"""Volume routes — build the 3D viewer bundle from a label volume on disk.

Delegates to the active plugin's ``build_volume_bundle`` capability (only
volumetric plugins like omnipose implement it). The bundle carries the flow
field, distance, spatial affinity graph, cell-sink points, and temporal
trajectories the 3D / 2.5D viewer renders.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..dependencies import require_plugin_capability
from ..plugins.base import SegmentationPlugin

router = APIRouter(prefix="/api")


class VolumePayload(BaseModel):
    masksPath: str
    rawPath: str | None = None
    linksPath: str | None = None
    doFlow: bool = True
    doAffinity: bool = True
    doTrajectories: bool = True
    embedVolumes: bool = True
    embedAffinity: bool = True


@router.post("/volume")
async def api_volume(
    payload: VolumePayload,
    plugin: SegmentationPlugin = Depends(require_plugin_capability("build_volume_bundle")),
) -> dict:
    fn = plugin.build_volume_bundle
    bundle = await asyncio.to_thread(
        fn,
        payload.masksPath,
        payload.rawPath,
        links_path=payload.linksPath,
        do_flow=payload.doFlow,
        do_affinity=payload.doAffinity,
        do_trajectories=payload.doTrajectories,
        embed_volumes=payload.embedVolumes,
        embed_affinity=payload.embedAffinity,
    )
    return dict(bundle)
