"""Executors for sandboxed code execution.

Keep this small and dependency-light so it can be imported by the pipeline
without pulling in optional server runtimes.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: str = ""
    execution_time: float = 0.0


@dataclass
class ExecutionConfig:
    timeout: float = 10.0
    memory_limit_mb: int = 16384
    temp_dir: str = "/tmp"
    python_bin: str = "python"
    max_concurrency: Optional[int] = None


class CodeExecutor(ABC):
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        limit = self.config.max_concurrency
        if not isinstance(limit, int) or limit <= 0:
            limit = (os.cpu_count() or 16) * 4
        self._semaphore = asyncio.Semaphore(limit)

    @abstractmethod
    async def execute_single(self, code: str) -> ExecutionResult: ...

    async def execute_multiple(self, codes: List[str]) -> List[ExecutionResult]:
        if not codes:
            return []

        async def _bounded(code_str: str) -> ExecutionResult:
            async with self._semaphore:
                return await self.execute_single(code_str)

        results = await asyncio.gather(*(_bounded(c) for c in codes), return_exceptions=True)
        out: List[ExecutionResult] = []
        for res in results:
            if isinstance(res, Exception):
                out.append(
                    ExecutionResult(
                        success=False,
                        output="",
                        error=f"executor_error: {type(res).__name__}: {res}",
                        execution_time=0.0,
                    )
                )
            else:
                out.append(res)
        return out

    def validate_code(self, code: str) -> bool:
        return isinstance(code, str) and bool(code.strip())

    def preprocess_code(self, code: str) -> str:
        return code.strip()
