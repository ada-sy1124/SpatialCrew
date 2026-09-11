"""SAM 3 segmentation tool wrapping the GPU segmentation model."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spatialcrew.tools.base import GPUTool
from spatialcrew.types.spatial import PerFrameMask


class SAM3Tool(GPUTool):
    """Client wrapper for SAM 3 segmentation GPU service."""
    TOOL_NAME = "SAM3"
    DOMAIN = "perception"
    TOOL_PROMPT_DESCRIPTION = """### tools.SAM3 - Segmentation (GPU)

**tools.SAM3.segment_image_by_text(image, text_prompt, frame_index=0) -> PerFrameMask**
Segment an object in a single image using a text description.

**tools.SAM3.segment_image_by_points(image, points, labels, frame_index=0) -> PerFrameMask**
Segment using point prompts. `points` is a list of [x, y] in pixel coords, `labels` is a list of 0/1.

**tools.SAM3.segment_image_by_box(image, box, frame_index=0) -> PerFrameMask**
Segment using a bounding box [x1, y1, x2, y2] in pixel coords.

**tools.SAM3.segment_video_by_text(images, text_prompt) -> List[PerFrameMask]**
Track and segment an object across video frames.

**tools.SAM3.is_object_exist(image, text_prompt) -> bool**
Check if an object exists in the image.

All segmentation methods return `PerFrameMask` with:
- `.mask`: (H, W) boolean array
- `.label`: text label
- `.score`: confidence score
- `.area`: pixel count
- `.area_ratio`: fraction of image area
"""

    async def segment_image_by_text(self, image, text_prompt: str, frame_index: int = 0) -> PerFrameMask:
        pil_img = image.image if hasattr(image, "image") else image
        result = await self._rpc_call(
            "spatial_SAM3", "detect",
            image=pil_img, text_prompt=text_prompt, mode="text",
        )
        mask = np.array(result.get("mask", np.zeros((1, 1), dtype=bool)))
        return PerFrameMask(frame_index=frame_index, mask=mask, label=text_prompt,
                           score=result.get("score", 1.0))

    async def segment_image_by_points(self, image, points, labels, frame_index: int = 0) -> PerFrameMask:
        pil_img = image.image if hasattr(image, "image") else image
        result = await self._rpc_call(
            "spatial_SAM3", "detect",
            image=pil_img, points=points, labels=labels, mode="points",
        )
        mask = np.array(result.get("mask", np.zeros((1, 1), dtype=bool)))
        return PerFrameMask(frame_index=frame_index, mask=mask, score=result.get("score", 1.0))

    async def segment_image_by_box(self, image, box, frame_index: int = 0) -> PerFrameMask:
        pil_img = image.image if hasattr(image, "image") else image
        result = await self._rpc_call(
            "spatial_SAM3", "detect",
            image=pil_img, box=box, mode="box",
        )
        mask = np.array(result.get("mask", np.zeros((1, 1), dtype=bool)))
        return PerFrameMask(frame_index=frame_index, mask=mask, score=result.get("score", 1.0))

    async def segment_video_by_text(self, images, text_prompt: str) -> List[PerFrameMask]:
        pil_imgs = [f.image if hasattr(f, "image") else f for f in images]
        result = await self._rpc_call(
            "spatial_SAM3", "segment_video",
            images=pil_imgs, text_prompt=text_prompt,
        )
        masks = result.get("masks", [])
        return [
            PerFrameMask(frame_index=i, mask=np.array(m), label=text_prompt)
            for i, m in enumerate(masks)
        ]

    async def is_object_exist(self, image, text_prompt: str) -> bool:
        pil_img = image.image if hasattr(image, "image") else image
        result = await self._rpc_call(
            "spatial_SAM3", "detect",
            image=pil_img, text_prompt=text_prompt, mode="text",
        )
        score = result.get("score", 0.0)
        return score > 0.3
