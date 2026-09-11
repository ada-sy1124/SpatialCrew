"""Evaluation and benchmarking for SpatialCrew."""

from spatialcrew.evals.base import BaseBenchmark, Sample, EvalResult, compute_accuracy, compute_mra
from spatialcrew.evals.spbench import SPBench

__all__ = [
    "BaseBenchmark", "Sample", "EvalResult",
    "compute_accuracy", "compute_mra", "SPBench",
]
