"""3D/4D reconstruction tool wrapping depth estimation GPU models."""

from __future__ import annotations
from typing import Any, List, Optional

import numpy as np

from spatialcrew.tools.base import GPUTool
from spatialcrew.types.spatial import (
    PerFrameDepth,
    PerFramePointMap,
    Reconstruction,
)


class ReconstructTool(GPUTool):
    """Client wrapper for the 3D reconstruction GPU service."""
    TOOL_NAME = "Reconstruct"
    DOMAIN = "geometry"
    TOOL_PROMPT_DESCRIPTION = """### tools.Reconstruct - 3D/4D Reconstruction (GPU)

**tools.Reconstruct.reconstruct(frames: InputImages) -> Reconstruction**
Performs monocular depth estimation and 3D reconstruction on the input frames.

Returns a `Reconstruction` object with attributes:
- `depths`: list of `PerFrameDepth` (each has `.depth` as (H,W) float32 array, `.min_depth`, `.max_depth`)
- `point_maps`: list of `PerFramePointMap` (each has `.points` as (H,W,3) float32 array in world coords)
- `intrinsics`: (3,3) camera intrinsics matrix
- `extrinsics`: list of (4,4) camera-to-world matrices
- `frame_indices`: list of frame indices

Example:
```python
recon = tools.Reconstruct.reconstruct(InputImages)
depth_0 = recon.depths[0].depth  # (H, W) float32
points_0 = recon.point_maps[0].points  # (H, W, 3) float32
```
"""

    async def reconstruct(self, frames) -> Reconstruction:
        """Reconstruct 3D scene from input frames."""
        images = [f.image if hasattr(f, "image") else f for f in frames]
        frame_indices = [f.frame_index if hasattr(f, "frame_index") else i for i, f in enumerate(frames)]

        result = await self._rpc_call(
            "spatial_Reconstruct", "reconstruct",
            images=images, frame_indices=frame_indices,
        )

        depths, point_maps = [], []
        for i, fi in enumerate(frame_indices):
            if "depths" in result and i < len(result["depths"]):
                d = result["depths"][i]
                depths.append(PerFrameDepth(frame_index=fi, depth=d))
            if "point_maps" in result and i < len(result["point_maps"]):
                p = result["point_maps"][i]
                point_maps.append(PerFramePointMap(frame_index=fi, points=p))

        return Reconstruction(
            depths=depths,
            point_maps=point_maps,
            intrinsics=result.get("intrinsics"),
            extrinsics=result.get("extrinsics"),
            frame_indices=frame_indices,
        )
