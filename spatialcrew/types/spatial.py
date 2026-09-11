"""Spatial data types for 3D reconstruction and per-frame analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class PerFrameDepth:
    """Per-frame depth map."""
    frame_index: int
    depth: np.ndarray  # (H, W) float32
    min_depth: float = 0.0
    max_depth: float = 0.0

    def __post_init__(self):
        if self.depth is not None and self.min_depth == 0 and self.max_depth == 0:
            valid = self.depth[self.depth > 0]
            if len(valid) > 0:
                self.min_depth = float(valid.min())
                self.max_depth = float(valid.max())


@dataclass
class PerFrameMask:
    """Per-frame segmentation mask."""
    frame_index: int
    mask: np.ndarray  # (H, W) bool
    label: str = ""
    score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def area_ratio(self) -> float:
        return self.area / max(self.mask.size, 1)


@dataclass
class PerFramePointMap:
    """Per-frame 3D point map."""
    frame_index: int
    points: np.ndarray  # (H, W, 3) float32


@dataclass
class Reconstruction:
    """3D/4D reconstruction output from depth estimation."""
    depths: List[PerFrameDepth] = field(default_factory=list)
    point_maps: List[PerFramePointMap] = field(default_factory=list)
    intrinsics: Optional[np.ndarray] = None  # (3, 3)
    extrinsics: Optional[List[np.ndarray]] = None  # list of (4, 4)
    frame_indices: List[int] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.depths)

    def get_depth(self, frame_index: int) -> Optional[PerFrameDepth]:
        for d in self.depths:
            if d.frame_index == frame_index:
                return d
        return None

    def get_point_map(self, frame_index: int) -> Optional[PerFramePointMap]:
        for p in self.point_maps:
            if p.frame_index == frame_index:
                return p
        return None
