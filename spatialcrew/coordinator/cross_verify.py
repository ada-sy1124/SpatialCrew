"""Cross-agent verification logic."""

from __future__ import annotations
import logging
from typing import Dict, List

from spatialcrew.types.results import AgentResult

logger = logging.getLogger(__name__)


def cross_verify_results(results: Dict[str, AgentResult]) -> List[str]:
    """Verify consistency across specialist agent results."""
    issues = []

    for task_id, result in results.items():
        if result.is_low_confidence:
            issues.append(f"Task {task_id}: low confidence ({result.confidence:.2f})")

        if not result.success:
            issues.append(f"Task {task_id}: execution failed")

        for warning in result.warnings:
            issues.append(f"Task {task_id} warning: {warning}")

    return issues
