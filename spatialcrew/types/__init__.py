"""Data types for SpatialCrew multi-agent framework."""

from spatialcrew.types.frames import FrameImage, InputImages
from spatialcrew.types.metadata import Metadata
from spatialcrew.types.results import (
    AgentDomain,
    AgentResult,
    ReturnAnswer,
    SubTaskSpec,
    VerificationIssue,
)
from spatialcrew.types.spatial import (
    PerFrameDepth,
    PerFrameMask,
    PerFramePointMap,
    Reconstruction,
)

__all__ = [
    "FrameImage", "InputImages", "Metadata",
    "AgentDomain", "AgentResult", "ReturnAnswer", "SubTaskSpec", "VerificationIssue",
    "PerFrameDepth", "PerFrameMask", "PerFramePointMap", "Reconstruction",
]
