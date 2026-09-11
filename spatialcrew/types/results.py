"""Result types for multi-agent communication."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class AgentDomain(str, Enum):
    PERCEPTION = "perception"
    GEOMETRY = "geometry"
    TEMPORAL = "temporal"


@dataclass
class SubTaskSpec:
    """Specification for a sub-task dispatched to a specialist agent."""
    task_id: str
    domain: AgentDomain
    instruction: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    depends_on: List[str] = field(default_factory=list)
    check_conditions: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Structured result returned by a specialist agent."""
    task_id: str
    domain: AgentDomain
    result: Any = None
    confidence: float = 0.0
    verification_log: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    steps_used: int = 0
    success: bool = True

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.5

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass
class ReturnAnswer:
    """Sentinel for the final answer submitted by the coordinator."""
    answer: str
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ReturnAnswer(answer={self.answer!r})"


@dataclass
class VerificationIssue:
    """A single verification issue found during WEVA verify step."""
    severity: str  # "error", "warning", "info"
    message: str
    source: str = ""
    suggestion: str = ""
