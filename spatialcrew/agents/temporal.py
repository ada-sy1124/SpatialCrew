"""Temporal specialist agent for video temporal analysis."""

from __future__ import annotations
from typing import List

from spatialcrew.agents.base import BaseSpecialist
from spatialcrew.kernel.manager import ExecutionResult
from spatialcrew.types.results import SubTaskSpec, VerificationIssue


class TemporalAgent(BaseSpecialist):
    """Specialist agent for temporal tasks (tracking, speed, events)."""
    DOMAIN = "temporal"

    def _verify(self, exec_result: ExecutionResult, subtask: SubTaskSpec) -> List[VerificationIssue]:
        issues = super()._verify(exec_result, subtask)
        if exec_result.success and exec_result.stdout:
            out = exec_result.stdout.lower()
            if "speed" in out or "velocity" in out:
                try:
                    import re
                    speeds = re.findall(r"speed[:\s]*(-?[0-9.]+)", out)
                    for s_str in speeds:
                        s = float(s_str)
                        if s < 0:
                            issues.append(VerificationIssue(
                                severity="warning",
                                message=f"Negative speed detected: {s}",
                                suggestion="Speed should be non-negative. Check direction computation.",
                            ))
                        elif s > 200:
                            issues.append(VerificationIssue(
                                severity="warning",
                                message=f"Unusually high speed: {s} m/s",
                                suggestion="Verify frame rate and distance units.",
                            ))
                except (ValueError, IndexError):
                    pass
        return issues
