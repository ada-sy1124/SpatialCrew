"""Mask manipulation utilities for segmentation results."""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

from spatialcrew.tools.base import CPUTool


class MaskUtils(CPUTool):
    """CPU-based mask manipulation utilities."""
    TOOL_NAME = "Mask"
    DOMAIN = "perception"
    TOOL_PROMPT_DESCRIPTION = """### tools.Mask - Mask Utilities

**tools.Mask.centroid(mask) -> Tuple[float, float]**
Compute the centroid (cy, cx) of a boolean mask.

**tools.Mask.area(mask) -> int**
Count the number of True pixels in a mask.

**tools.Mask.bounding_box(mask) -> Tuple[int, int, int, int]**
Return (y_min, x_min, y_max, x_max) bounding box of a mask.

**tools.Mask.iou(mask1, mask2) -> float**
Compute Intersection over Union of two masks.

**tools.Mask.intersection(mask1, mask2) -> np.ndarray**
Return the intersection of two boolean masks.
"""

    @staticmethod
    def centroid(mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return (0.0, 0.0)
        return (float(ys.mean()), float(xs.mean()))

    @staticmethod
    def area(mask: np.ndarray) -> int:
        return int(np.sum(mask))

    @staticmethod
    def bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return (0, 0, 0, 0)
        return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))

    @staticmethod
    def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(inter / max(union, 1))

    @staticmethod
    def intersection(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        return np.logical_and(mask1, mask2)
