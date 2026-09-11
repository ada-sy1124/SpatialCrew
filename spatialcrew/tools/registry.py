"""Domain-isolated tool registry for specialist agents."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type

from spatialcrew.tools.base import BaseTool
from spatialcrew.tools.reconstruct import ReconstructTool
from spatialcrew.tools.sam3 import SAM3Tool
from spatialcrew.tools.geometry_utils import GeometryUtils
from spatialcrew.tools.mask_utils import MaskUtils
from spatialcrew.tools.draw_utils import DrawUtils
from spatialcrew.tools.time_utils import TimeUtils
from spatialcrew.tools.graph_utils import GraphDrawer

DOMAIN_TOOL_MAP: Dict[str, List[Type[BaseTool]]] = {
    "perception": [SAM3Tool, MaskUtils, DrawUtils],
    "geometry": [ReconstructTool, GeometryUtils],
    "temporal": [TimeUtils, GraphDrawer],
}

ALL_TOOL_CLASSES: List[Type[BaseTool]] = [
    SAM3Tool, ReconstructTool, MaskUtils, DrawUtils,
    GeometryUtils, TimeUtils, GraphDrawer,
]


class ToolRegistry:
    """Creates and manages domain-isolated tool instances."""

    def __init__(self, config, gpu_client=None, metadata=None):
        self._config = config
        self._gpu_client = gpu_client
        self._metadata = metadata
        self._tools: Dict[str, Any] = {}

    def create_tools_for_domain(self, domain: str) -> Dict[str, BaseTool]:
        """Create tool instances for a specific domain."""
        tool_classes = DOMAIN_TOOL_MAP.get(domain, [])
        tools = {}
        for cls in tool_classes:
            if issubclass(cls, (ReconstructTool, SAM3Tool)):
                tool = cls(config=self._config, gpu_client=self._gpu_client)
            elif cls == TimeUtils and self._metadata:
                tool = cls(
                    fps=self._metadata.fps,
                    total_frames=self._metadata.total_frames,
                )
            else:
                tool = cls()
            tools[cls.TOOL_NAME] = tool
        return tools

    def get_prompt_for_domain(self, domain: str) -> str:
        """Get concatenated tool prompt descriptions for a domain."""
        tool_classes = DOMAIN_TOOL_MAP.get(domain, [])
        parts = []
        for cls in tool_classes:
            desc = cls.TOOL_PROMPT_DESCRIPTION
            if desc:
                parts.append(desc.strip())
        return "\n\n".join(parts)

    def get_all_prompts(self) -> str:
        """Get all tool prompt descriptions (for coordinator overview)."""
        parts = []
        for cls in ALL_TOOL_CLASSES:
            desc = cls.TOOL_PROMPT_DESCRIPTION
            if desc:
                parts.append(desc.strip())
        return "\n\n".join(parts)
