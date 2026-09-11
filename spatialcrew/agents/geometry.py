"""Geometry specialist agent for 3D spatial computation."""

from __future__ import annotations
from typing import List

from spatialcrew.agents.base import BaseSpecialist
from spatialcrew.kernel.manager import ExecutionResult
from spatialcrew.types.results import SubTaskSpec, VerificationIssue


class GeometryAgent(BaseSpecialist):
    """Specialist agent for geometry tasks (reconstruction, distance, angles)."""
    DOMAIN = "geometry"

    def _verify(self, exec_result: ExecutionResult, subtask: SubTaskSpec) -> List[VerificationIssue]:
        issues = super()._verify(exec_result, subtask)
        if exec_result.success and exec_result.stdout:
            out = exec_result.stdout.lower()
            if "distance" in out:
                try:
                    import re
                    distances = re.findall(r"distance[:\s]*(-?[0-9.]+)", out)
                    for d_str in distances:
                        d = float(d_str)
                        if d < 0:
                            issues.append(VerificationIssue(
                                severity="error",
                                message=f"Negative distance detected: {d}",
                                suggestion="Check coordinate extraction and measurement logic.",
                            ))
                        elif d > 1000:
                            issues.append(VerificationIssue(
                                severity="warning",
                                message=f"Very large distance: {d}m",
                                suggestion="Verify units and reconstruction scale.",
                            ))
                except (ValueError, IndexError):
                    pass
        return issues
