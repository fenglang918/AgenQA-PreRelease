"""Code verifier utilities (local + optional MCP server).

This module provides a Python-only code execution sandbox that can be used by
the pipeline to score/verify generated code. An optional MCP server wrapper is
available when the `fastmcp` dependency is installed.
"""

from .executor import ExecutionConfig, ExecutionResult
from .python_executor import PythonExecutor

__all__ = [
    "ExecutionConfig",
    "ExecutionResult",
    "PythonExecutor",
]
