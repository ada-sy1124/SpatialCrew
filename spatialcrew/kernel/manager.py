"""Jupyter kernel lifecycle management for code execution."""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jupyter_client import AsyncKernelManager

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing code in a Jupyter kernel."""
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    display_data: List[Any] = field(default_factory=list)
    execution_time_sec: float = 0.0
    variables_changed: Dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    def summary(self, max_len: int = 500) -> str:
        parts = []
        if self.stdout:
            parts.append(f"[stdout] {self.stdout[:max_len]}")
        if self.stderr:
            parts.append(f"[stderr] {self.stderr[:max_len]}")
        if self.error:
            parts.append(f"[error] {self.error[:max_len]}")
        return "\n".join(parts) if parts else "[no output]"


class JupyterKernelManager:
    """Manages a persistent Jupyter (IPython) kernel for agent code execution."""

    def __init__(self, startup_timeout: float = 60.0):
        self._km: Optional[AsyncKernelManager] = None
        self._kc = None
        self._startup_timeout = startup_timeout
        self._init_code: Optional[str] = None
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._started and self._km is not None

    async def start(self, retries: int = 3) -> None:
        """Start the kernel with retry logic."""
        for attempt in range(retries):
            try:
                self._km = AsyncKernelManager(kernel_name="python3")
                await self._km.start_kernel()
                self._kc = self._km.client()
                self._kc.start_channels()
                await self._kc.wait_for_ready(timeout=self._startup_timeout)
                self._started = True
                logger.info("Jupyter kernel started successfully.")
                if self._init_code:
                    await self.execute(self._init_code)
                return
            except Exception as e:
                logger.warning(f"Kernel start attempt {attempt+1}/{retries} failed: {e}")
                await self._cleanup()
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to start Jupyter kernel after {retries} attempts.")

    async def execute(self, code: str, timeout: float = 120.0) -> ExecutionResult:
        """Execute code in the kernel and collect outputs."""
        if not self.is_running:
            raise RuntimeError("Kernel is not running.")

        t0 = time.monotonic()
        msg_id = self._kc.execute(code)

        stdout_parts, stderr_parts, display_data = [], [], []
        error_text = None

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error_text = f"Execution timed out after {timeout}s"
                break
            try:
                msg = await asyncio.wait_for(
                    self._kc.get_iopub_msg(), timeout=min(remaining, 5.0)
                )
            except asyncio.TimeoutError:
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                if content["name"] == "stdout":
                    stdout_parts.append(content["text"])
                else:
                    stderr_parts.append(content["text"])
            elif msg_type == "error":
                tb = content.get("traceback", [])
                error_text = "\n".join(tb) if tb else content.get("evalue", "Unknown error")
            elif msg_type in ("display_data", "execute_result"):
                display_data.append(content.get("data", {}))
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        elapsed = time.monotonic() - t0
        return ExecutionResult(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            error=error_text,
            display_data=display_data,
            execution_time_sec=elapsed,
        )

    def set_init_code(self, code: str) -> None:
        """Store initialization code to re-inject on kernel restart."""
        self._init_code = code

    async def reset_namespace(self) -> None:
        """Clear user-defined variables while keeping injected objects."""
        if self.is_running:
            await self.execute("%reset -f")
            if self._init_code:
                await self.execute(self._init_code)

    async def shutdown(self) -> None:
        """Gracefully shut down the kernel."""
        await self._cleanup()
        logger.info("Jupyter kernel shut down.")

    async def _cleanup(self) -> None:
        self._started = False
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
            self._kc = None
        if self._km is not None:
            try:
                await self._km.shutdown_kernel(now=True)
            except Exception:
                pass
            self._km = None
