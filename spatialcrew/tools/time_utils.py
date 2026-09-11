"""Temporal analysis utilities for video spatial reasoning."""

from __future__ import annotations
from typing import List, Optional, Tuple

from spatialcrew.tools.base import CPUTool


class TimeUtils(CPUTool):
    """CPU-based temporal utilities for video reasoning."""
    TOOL_NAME = "Time"
    DOMAIN = "temporal"
    TOOL_PROMPT_DESCRIPTION = """### tools.Time - Temporal Utilities

**tools.Time.frame_to_seconds(frame_index) -> float**
Convert a frame index to timestamp in seconds.

**tools.Time.seconds_to_frame(seconds) -> int**
Convert a timestamp in seconds to the nearest frame index.

**tools.Time.frame_range_to_seconds(start_frame, end_frame) -> Tuple[float, float]**
Convert a frame range to a time range in seconds.

**tools.Time.get_frame_at_time(seconds) -> int**
Get the frame index closest to the given time in seconds.
"""

    def __init__(self, fps: float = 30.0, total_frames: int = 0):
        self._fps = max(fps, 1e-6)
        self._total_frames = total_frames

    def frame_to_seconds(self, frame_index: int) -> float:
        return frame_index / self._fps

    def seconds_to_frame(self, seconds: float) -> int:
        frame = int(round(seconds * self._fps))
        return max(0, min(frame, self._total_frames - 1)) if self._total_frames > 0 else max(0, frame)

    def frame_range_to_seconds(self, start_frame: int, end_frame: int) -> Tuple[float, float]:
        return (self.frame_to_seconds(start_frame), self.frame_to_seconds(end_frame))

    def get_frame_at_time(self, seconds: float) -> int:
        return self.seconds_to_frame(seconds)
