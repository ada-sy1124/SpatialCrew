"""Base benchmark and evaluation utilities."""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """A single benchmark sample."""
    sample_id: str
    question: str
    images: Any = None
    metadata: Any = None
    ground_truth: str = ""
    choices: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""
    sample_id: str
    prediction: str = ""
    ground_truth: str = ""
    correct: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    """Abstract base class for spatial reasoning benchmarks."""

    @abstractmethod
    def load(self, subset: str = "") -> None:
        """Load the dataset."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples."""

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples."""

    @abstractmethod
    def evaluate(self, predictions: Dict[str, str]) -> Dict[str, float]:
        """Compute aggregate metrics from predictions."""


def compute_accuracy(predictions: Dict[str, str], ground_truths: Dict[str, str]) -> float:
    """Compute exact-match accuracy."""
    if not ground_truths:
        return 0.0
    correct = sum(
        1 for sid, pred in predictions.items()
        if sid in ground_truths and _normalize_answer(pred) == _normalize_answer(ground_truths[sid])
    )
    return correct / len(ground_truths)


def compute_mra(
    predictions: Dict[str, str],
    ground_truths: Dict[str, str],
    choices_map: Dict[str, List[str]],
) -> float:
    """Compute MRA (Normalized Accuracy) that adjusts for random guessing."""
    if not ground_truths:
        return 0.0

    total_na = 0.0
    count = 0
    for sid, gt in ground_truths.items():
        pred = predictions.get(sid, "")
        n_choices = len(choices_map.get(sid, ["A", "B", "C", "D"]))
        random_acc = 1.0 / max(n_choices, 1)
        is_correct = _normalize_answer(pred) == _normalize_answer(gt)
        na = (float(is_correct) - random_acc) / (1.0 - random_acc)
        total_na += max(na, 0.0)
        count += 1

    return total_na / max(count, 1)


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip().upper()
    if answer and answer[0] in "ABCDEFGH":
        return answer[0]
    return answer
