"""Prompt templates for SpatialCrew agents."""

from spatialcrew.llm.prompts.common import (
    CODE_RULES_SECTION,
    COORDINATE_SYSTEM_SECTION,
    EVIDENCE_HIERARCHY_SECTION,
    RETURN_ANSWER_SECTION,
    WEVA_FORMAT_SECTION,
)
from spatialcrew.llm.prompts.coordinator import COORDINATOR_SYSTEM_PROMPT
from spatialcrew.llm.prompts.geometry import GEOMETRY_SYSTEM_PROMPT
from spatialcrew.llm.prompts.perception import PERCEPTION_SYSTEM_PROMPT
from spatialcrew.llm.prompts.temporal import TEMPORAL_SYSTEM_PROMPT


def build_specialist_prompt(domain: str, tool_descriptions: str) -> str:
    """Build a complete system prompt for a specialist agent."""
    templates = {
        "perception": PERCEPTION_SYSTEM_PROMPT,
        "geometry": GEOMETRY_SYSTEM_PROMPT,
        "temporal": TEMPORAL_SYSTEM_PROMPT,
    }
    template = templates.get(domain, PERCEPTION_SYSTEM_PROMPT)
    return template.format(
        tool_descriptions=tool_descriptions,
        weva_format=WEVA_FORMAT_SECTION,
        code_rules=CODE_RULES_SECTION,
        coordinate_system=COORDINATE_SYSTEM_SECTION,
    )


__all__ = [
    "COORDINATOR_SYSTEM_PROMPT",
    "build_specialist_prompt",
    "WEVA_FORMAT_SECTION",
    "CODE_RULES_SECTION",
    "COORDINATE_SYSTEM_SECTION",
    "EVIDENCE_HIERARCHY_SECTION",
    "RETURN_ANSWER_SECTION",
]
