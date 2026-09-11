"""Specialist agents for domain-specific spatial reasoning."""

from spatialcrew.agents.base import BaseSpecialist
from spatialcrew.agents.perception import PerceptionAgent
from spatialcrew.agents.geometry import GeometryAgent
from spatialcrew.agents.temporal import TemporalAgent

AGENT_REGISTRY = {
    "perception": PerceptionAgent,
    "geometry": GeometryAgent,
    "temporal": TemporalAgent,
}

__all__ = [
    "BaseSpecialist", "PerceptionAgent", "GeometryAgent", "TemporalAgent",
    "AGENT_REGISTRY",
]
