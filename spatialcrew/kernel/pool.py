"""Pool of reusable Jupyter kernel managers."""

from __future__ import annotations
import asyncio
import logging
from typing import List

from spatialcrew.kernel.manager import JupyterKernelManager

logger = logging.getLogger(__name__)


class KernelPool:
    """Pool of reusable Jupyter kernels to avoid cold starts."""

    def __init__(self, pool_size: int = 4, startup_timeout: float = 60.0):
        self._pool_size = pool_size
        self._startup_timeout = startup_timeout
        self._available: asyncio.Queue[JupyterKernelManager] = asyncio.Queue()
        self._all_kernels: List[JupyterKernelManager] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Pre-start all kernels in the pool."""
        if self._initialized:
            return
        tasks = []
        for _ in range(self._pool_size):
            km = JupyterKernelManager(startup_timeout=self._startup_timeout)
            self._all_kernels.append(km)
            tasks.append(km.start())
        await asyncio.gather(*tasks)
        for km in self._all_kernels:
            await self._available.put(km)
        self._initialized = True
        logger.info(f"Kernel pool initialized with {self._pool_size} kernels.")

    async def acquire(self) -> JupyterKernelManager:
        """Acquire a kernel from the pool (blocks if none available)."""
        if not self._initialized:
            await self.initialize()
        return await self._available.get()

    async def release(self, kernel: JupyterKernelManager) -> None:
        """Return a kernel to the pool after resetting its namespace."""
        try:
            await kernel.reset_namespace()
        except Exception as e:
            logger.warning(f"Failed to reset kernel namespace: {e}")
        await self._available.put(kernel)

    async def shutdown_all(self) -> None:
        """Shut down all kernels in the pool."""
        for km in self._all_kernels:
            try:
                await km.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down kernel: {e}")
        self._all_kernels.clear()
        self._initialized = False
        logger.info("All kernels shut down.")
