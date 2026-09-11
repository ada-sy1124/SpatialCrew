"""Base classes for SpatialCrew tools."""

from __future__ import annotations
import logging
from abc import ABC
from typing import Any, ClassVar, Dict, Optional

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all tools."""
    TOOL_NAME: ClassVar[str] = ""
    TOOL_PROMPT_DESCRIPTION: ClassVar[str] = ""
    DOMAIN: ClassVar[str] = ""

    def get_prompt_description(self) -> str:
        return self.TOOL_PROMPT_DESCRIPTION


class GPUTool(BaseTool):
    """Base class for tools that require GPU server RPC calls."""

    def __init__(self, config, gpu_client=None):
        self._config = config
        self._gpu_client = gpu_client

    async def _rpc_call(self, service: str, method: str, **kwargs) -> Any:
        if self._gpu_client is None:
            raise RuntimeError(f"GPU client not available for {self.TOOL_NAME}")
        return await self._gpu_client.call(service, method, **kwargs)


class CPUTool(BaseTool):
    """Base class for tools that run locally on CPU."""
    pass
