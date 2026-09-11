"""Metadata for input samples."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Metadata:
    """Sample-level metadata passed to agents."""
    fps: float = 0.0
    total_frames: int = 0
    duration_sec: float = 0.0
    num_videos: int = 0
    image_width: int = 0
    image_height: int = 0
    question_type: str = ""
    choices: Optional[List[str]] = None
    extra: dict = field(default_factory=dict)
