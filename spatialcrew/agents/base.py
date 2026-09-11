"""Base specialist agent with the WEVA (Write-Execute-Verify-Adjust) loop."""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional

from spatialcrew.kernel.manager import ExecutionResult, JupyterKernelManager
from spatialcrew.kernel.pool import KernelPool
from spatialcrew.kernel.sandbox import CodeSandbox
from spatialcrew.llm.client import LLMClient
from spatialcrew.llm.prompts import build_specialist_prompt
from spatialcrew.tools.registry import ToolRegistry
from spatialcrew.types.results import AgentDomain, AgentResult, SubTaskSpec, VerificationIssue

logger = logging.getLogger(__name__)


class BaseSpecialist:
    """Base class for specialist agents implementing the WEVA loop."""

    DOMAIN: str = ""

    def __init__(
        self,
        config,
        llm_client: LLMClient,
        kernel_pool: KernelPool,
        tool_registry: ToolRegistry,
    ):
        self._config = config
        self._llm = llm_client
        self._kernel_pool = kernel_pool
        self._tool_registry = tool_registry
        self._sandbox = CodeSandbox()
        self._max_steps = config.max_steps

    async def run(self, subtask: SubTaskSpec, context: Dict[str, Any] = None) -> AgentResult:
        """Execute a sub-task using the WEVA loop."""
        kernel = await self._kernel_pool.acquire()
        try:
            return await self._weva_loop(kernel, subtask, context or {})
        finally:
            await self._kernel_pool.release(kernel)

    async def _weva_loop(
        self,
        kernel: JupyterKernelManager,
        subtask: SubTaskSpec,
        context: Dict[str, Any],
    ) -> AgentResult:
        """The Write-Execute-Verify-Adjust loop."""
        tools = self._tool_registry.create_tools_for_domain(self.DOMAIN)
        init_code = self._build_init_code(tools, context)
        kernel.set_init_code(init_code)
        await kernel.execute(init_code)

        tool_prompts = self._tool_registry.get_prompt_for_domain(self.DOMAIN)
        system_prompt = build_specialist_prompt(self.DOMAIN, tool_prompts)

        history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_task_message(subtask)},
        ]

        verification_log = []
        warnings = []
        final_result = None
        confidence = 0.0
        steps_used = 0

        for step in range(self._max_steps):
            steps_used = step + 1

            # [Write] Generate code from LLM
            response = await self._llm.generate(
                messages=history, role="specialist"
            )
            history.append({"role": "assistant", "content": response})

            code = self._extract_code(response)
            if not code:
                history.append({
                    "role": "user",
                    "content": "No executable code found. Please provide a Python code block.",
                })
                continue

            violations = self._sandbox.validate(code)
            if violations:
                msg = "Security violations:\n" + "\n".join(f"- {v}" for v in violations)
                history.append({"role": "user", "content": msg})
                warnings.append(f"Step {step}: security violation")
                continue

            # [Execute] Run code in kernel
            exec_result = await kernel.execute(code)

            feedback = self._build_feedback(exec_result)

            if "ReturnAnswer" in exec_result.stdout or "ReturnAnswer" in code:
                answer_match = re.search(
                    r'ReturnAnswer\s*\(\s*answer\s*=\s*["\'](.+?)["\']',
                    code + "\n" + exec_result.stdout,
                )
                if answer_match:
                    final_result = answer_match.group(1)
                    confidence = 0.9
                    verification_log.append(f"Step {step}: answer submitted")
                    break

            # [Verify] Check execution result
            issues = self._verify(exec_result, subtask)
            for issue in issues:
                verification_log.append(f"Step {step}: {issue.severity} - {issue.message}")
                if issue.severity == "warning":
                    warnings.append(issue.message)

            if exec_result.success and not any(i.severity == "error" for i in issues):
                feedback += "\nExecution successful. Continue with the next step or submit your answer."
            else:
                # [Adjust] Provide feedback for the next iteration
                if issues:
                    feedback += "\n\nVerification issues found:\n"
                    for issue in issues:
                        feedback += f"- [{issue.severity}] {issue.message}"
                        if issue.suggestion:
                            feedback += f" Suggestion: {issue.suggestion}"
                        feedback += "\n"
                    feedback += "\nPlease adjust your approach and try again."

            history.append({"role": "user", "content": feedback})

        return AgentResult(
            task_id=subtask.task_id,
            domain=AgentDomain(self.DOMAIN),
            result=final_result,
            confidence=confidence,
            verification_log=verification_log,
            warnings=warnings,
            steps_used=steps_used,
            success=final_result is not None,
        )

    def _build_init_code(self, tools: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build initialization code to inject into the kernel."""
        lines = [
            "import numpy as np",
            "from scipy import spatial as sp_spatial",
            "import matplotlib; matplotlib.use('Agg')",
            "import matplotlib.pyplot as plt",
        ]
        return "\n".join(lines)

    def _build_task_message(self, subtask: SubTaskSpec) -> str:
        """Build the initial task message for the LLM."""
        parts = [f"## Task: {subtask.instruction}"]
        if subtask.expected_output:
            parts.append(f"\n**Expected Output**: {subtask.expected_output}")
        if subtask.check_conditions:
            parts.append("\n**Verification Conditions**:")
            for cond in subtask.check_conditions:
                parts.append(f"- {cond}")
        return "\n".join(parts)

    def _verify(self, exec_result: ExecutionResult, subtask: SubTaskSpec) -> List[VerificationIssue]:
        """Verify execution results against task specifications. Override in subclasses."""
        issues = []
        if exec_result.error:
            issues.append(VerificationIssue(
                severity="error",
                message=f"Execution error: {exec_result.error[:200]}",
                suggestion="Fix the error and re-execute.",
            ))
        return issues

    @staticmethod
    def _extract_code(response: str) -> str:
        """Extract Python code block from LLM response."""
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return ""

    @staticmethod
    def _build_feedback(exec_result: ExecutionResult) -> str:
        """Build a feedback message from execution results."""
        parts = [f"**Execution** ({exec_result.execution_time_sec:.1f}s)"]
        if exec_result.stdout:
            parts.append(f"```\n{exec_result.stdout[:2000]}\n```")
        if exec_result.stderr:
            parts.append(f"**Warnings:**\n```\n{exec_result.stderr[:1000]}\n```")
        if exec_result.error:
            parts.append(f"**Error:**\n```\n{exec_result.error[:1500]}\n```")
        return "\n".join(parts)
