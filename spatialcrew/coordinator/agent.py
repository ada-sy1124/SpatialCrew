"""Coordinator agent for multi-agent orchestration."""

from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Optional

from spatialcrew.agents import AGENT_REGISTRY
from spatialcrew.kernel.pool import KernelPool
from spatialcrew.llm.client import LLMClient
from spatialcrew.llm.prompts.coordinator import COORDINATOR_SYSTEM_PROMPT
from spatialcrew.tools.registry import ToolRegistry
from spatialcrew.types.results import AgentDomain, AgentResult, ReturnAnswer, SubTaskSpec

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """Orchestrates specialist agents for spatial reasoning tasks."""

    def __init__(self, config, llm_client: LLMClient, kernel_pool: KernelPool, tool_registry: ToolRegistry):
        self._config = config
        self._llm = llm_client
        self._kernel_pool = kernel_pool
        self._tool_registry = tool_registry

        self._specialists = {}
        for domain, agent_cls in AGENT_REGISTRY.items():
            self._specialists[domain] = agent_cls(
                config=config,
                llm_client=llm_client,
                kernel_pool=kernel_pool,
                tool_registry=tool_registry,
            )

    async def solve(self, question: str, images: Any, metadata: Any = None) -> ReturnAnswer:
        """Solve a spatial reasoning question using multi-agent collaboration."""

        # Step 1: Task decomposition
        subtasks = await self._decompose(question, images, metadata)
        logger.info(f"Decomposed into {len(subtasks)} sub-tasks: {[s.task_id for s in subtasks]}")

        # Step 2: Execute sub-tasks respecting dependencies
        results: Dict[str, AgentResult] = {}
        for round_num in range(self._config.max_dispatch_rounds):
            pending = [st for st in subtasks if st.task_id not in results]
            if not pending:
                break

            for subtask in pending:
                if all(dep in results for dep in subtask.depends_on):
                    context = {dep: results[dep] for dep in subtask.depends_on}
                    specialist = self._specialists.get(subtask.domain.value)
                    if specialist is None:
                        logger.warning(f"No specialist for domain {subtask.domain}")
                        continue
                    result = await specialist.run(subtask, context)
                    results[subtask.task_id] = result
                    logger.info(
                        f"Task {subtask.task_id} ({subtask.domain.value}): "
                        f"success={result.success}, confidence={result.confidence:.2f}"
                    )

            # Step 3: Cross-verification
            if self._config.enable_cross_verification:
                issues = self._cross_verify(results)
                if issues:
                    logger.warning(f"Cross-verification issues: {issues}")
                    low_conf = [tid for tid, r in results.items() if r.is_low_confidence]
                    for tid in low_conf:
                        del results[tid]
                    if not low_conf:
                        break
                else:
                    break
            else:
                break

        # Step 4: Synthesize final answer
        return await self._synthesize(question, results, images, metadata)

    async def _decompose(self, question: str, images: Any, metadata: Any) -> List[SubTaskSpec]:
        """Decompose the question into sub-tasks using the LLM."""
        tool_overview = self._tool_registry.get_all_prompts()

        messages = [
            {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"## Question\n{question}\n\n"
                f"## Available Tools Overview\n{tool_overview}\n\n"
                f"## Input\n- Number of images: {len(images) if images else 0}\n"
                f"- Metadata: {metadata}\n\n"
                "Please decompose this question into sub-tasks. "
                "Output ONLY a JSON array of sub-task objects."
            )},
        ]

        response = await self._llm.generate(messages, role="coordinator")
        return self._parse_subtasks(response)

    def _parse_subtasks(self, response: str) -> List[SubTaskSpec]:
        """Parse LLM response into SubTaskSpec objects."""
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            logger.warning("Could not parse sub-tasks from LLM response, using default single task.")
            return [SubTaskSpec(
                task_id="t1",
                domain=AgentDomain.GEOMETRY,
                instruction="Analyze the image and answer the question.",
            )]

        try:
            tasks_data = json.loads(json_match.group())
            subtasks = []
            for t in tasks_data:
                subtasks.append(SubTaskSpec(
                    task_id=t.get("task_id", f"t{len(subtasks)+1}"),
                    domain=AgentDomain(t.get("domain", "geometry")),
                    instruction=t.get("instruction", ""),
                    expected_output=t.get("expected_output", ""),
                    depends_on=t.get("depends_on", []),
                    check_conditions=t.get("check_conditions", []),
                ))
            return subtasks
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse sub-tasks JSON: {e}")
            return [SubTaskSpec(
                task_id="t1",
                domain=AgentDomain.GEOMETRY,
                instruction="Analyze the image and answer the question.",
            )]

    def _cross_verify(self, results: Dict[str, AgentResult]) -> List[str]:
        """Check cross-agent consistency."""
        issues = []
        low_conf = [tid for tid, r in results.items() if r.is_low_confidence]
        if low_conf:
            issues.append(f"Low confidence results: {low_conf}")
        warning_tasks = [tid for tid, r in results.items() if r.has_warnings]
        if warning_tasks:
            issues.append(f"Tasks with warnings: {warning_tasks}")
        return issues

    async def _synthesize(
        self, question: str, results: Dict[str, AgentResult],
        images: Any, metadata: Any,
    ) -> ReturnAnswer:
        """Synthesize the final answer from all specialist results."""
        results_summary = []
        for tid, r in results.items():
            results_summary.append(
                f"- **{tid}** ({r.domain.value}): result={r.result}, "
                f"confidence={r.confidence:.2f}, warnings={r.warnings}"
            )

        messages = [
            {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"## Original Question\n{question}\n\n"
                f"## Specialist Results\n" + "\n".join(results_summary) + "\n\n"
                "Based on the specialist results above, provide the final answer. "
                "Output ONLY the answer value (e.g., 'A', '3.5', 'yes')."
            )},
        ]

        response = await self._llm.generate(messages, role="coordinator")
        answer = response.strip().strip("'\"")

        return ReturnAnswer(
            answer=answer,
            reasoning=f"Synthesized from {len(results)} specialist results.",
            evidence=[f"{tid}: {r.result}" for tid, r in results.items()],
        )
