"""Geometry computation utilities for spatial reasoning."""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

from spatialcrew.tools.base import CPUTool


class GeometryUtils(CPUTool):
    """CPU-based geometry computation utilities."""
    TOOL_NAME = "Geometry"
    DOMAIN = "geometry"
    TOOL_PROMPT_DESCRIPTION = """### tools.Geometry - Spatial Measurement Utilities

**tools.Geometry.euclidean_distance(p1, p2) -> float**
Compute 3D Euclidean distance between two points.

**tools.Geometry.angle_between_vectors(v1, v2) -> float**
Compute angle (in degrees) between two 3D vectors.

**tools.Geometry.project_point_to_camera(point_3d, intrinsics, extrinsics) -> Tuple[float, float]**
Project a 3D world point to 2D pixel coordinates.

**tools.Geometry.fit_ground_plane_ransac(points, threshold=0.02) -> Tuple[np.ndarray, float]**
Fit a ground plane to 3D points using RANSAC. Returns (normal_vector, offset).

**tools.Geometry.transform_points(points, matrix) -> np.ndarray**
Apply a 4x4 transformation matrix to an array of 3D points.
"""

    @staticmethod
    def euclidean_distance(p1, p2) -> float:
        p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def angle_between_vectors(v1, v2) -> float:
        v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    @staticmethod
    def project_point_to_camera(
        point_3d: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> Tuple[float, float]:
        pt = np.asarray(point_3d, dtype=float)
        if pt.shape == (3,):
            pt = np.append(pt, 1.0)
        cam_pt = np.linalg.inv(extrinsics) @ pt
        px = intrinsics @ cam_pt[:3]
        px = px / (px[2] + 1e-12)
        return float(px[0]), float(px[1])

    @staticmethod
    def fit_ground_plane_ransac(
        points: np.ndarray,
        threshold: float = 0.02,
        max_iterations: int = 1000,
    ) -> Tuple[np.ndarray, float]:
        best_inliers = 0
        best_normal = np.array([0.0, 1.0, 0.0])
        best_d = 0.0
        n = len(points)

        for _ in range(max_iterations):
            idx = np.random.choice(n, 3, replace=False)
            p0, p1, p2 = points[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue
            normal = normal / norm_len
            d = -np.dot(normal, p0)
            distances = np.abs(points @ normal + d)
            inliers = np.sum(distances < threshold)
            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_d = d

        return best_normal, best_d

    @staticmethod
    def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts[:, :3], ones])
        transformed = (matrix @ pts_h.T).T
        return transformed[:, :3]
