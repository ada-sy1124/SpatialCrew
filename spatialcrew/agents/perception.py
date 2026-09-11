"""Perception specialist agent for visual evidence extraction."""

from __future__ import annotations
from typing import List

from spatialcrew.agents.base import BaseSpecialist
from spatialcrew.kernel.manager import ExecutionResult
from spatialcrew.types.results import SubTaskSpec, VerificationIssue


class PerceptionAgent(BaseSpecialist):
    """Specialist agent for perception tasks (segmentation, depth, visual grounding)."""
    DOMAIN = "perception"

    def _verify(self, exec_result: ExecutionResult, subtask: SubTaskSpec) -> List[VerificationIssue]:
        issues = super()._verify(exec_result, subtask)
        if exec_result.success and exec_result.stdout:
            out = exec_result.stdout.lower()
            if "area_ratio" in out:
                try:
                    import re
                    ratios = re.findall(r"area_ratio[:\s]*([0-9.]+)", out)
                    for r_str in ratios:
                        r = float(r_str)
                        if r < 0.001:
                            issues.append(VerificationIssue(
                                severity="warning",
                                message=f"Mask area ratio {r:.4f} is very small (<0.1%)",
                                suggestion="Try a different segmentation prompt or use point prompts.",
                            ))
                        elif r > 0.8:
                            issues.append(VerificationIssue(
                                severity="warning",
                                message=f"Mask area ratio {r:.2f} is very large (>80%)",
                                suggestion="The mask may cover the entire image. Refine the prompt.",
                            ))
                except (ValueError, IndexError):
                    pass
        return issues
