"""Optional MCP server wrapper for the Python code verifier.

This is intentionally isolated from the rest of infra so that missing optional
dependencies (fastmcp) won't break the core AgenQA pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .executor import ExecutionConfig
from .python_executor import PythonExecutor


def _require_fastmcp():
    # FastMCP's CLI banner may attempt an online version check via httpx.
    # In environments with SOCKS proxies (and without httpx[socks]) this can crash
    # the server at startup. Disable update checks by default for robustness.
    os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")
    try:
        from fastmcp import FastMCP  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing optional dependency `fastmcp`.\n"
            "Install with: `pip install -r requirements_code_verifier.txt`"
        ) from e


def create_server(
    *,
    name: str = "AgenQA-CodeVerifier",
    temp_dir: str = "/tmp",
    memory_limit_mb: int = 16384,
    default_timeout: float = 10.0,
    python_bin: str = sys.executable,
):
    _require_fastmcp()
    from fastmcp import FastMCP

    mcp = FastMCP(name=name)

    base_conf = ExecutionConfig(
        timeout=float(default_timeout),
        memory_limit_mb=int(memory_limit_mb),
        temp_dir=str(temp_dir),
        python_bin=str(python_bin),
    )

    @mcp.tool
    async def execute_code(code: str, timeout: Optional[float] = None) -> str:
        """Execute a Python snippet and return a JSON string."""
        conf = base_conf
        if isinstance(timeout, (int, float)) and float(timeout) > 0:
            conf = ExecutionConfig(**{**base_conf.__dict__, "timeout": float(timeout)})
        executor = PythonExecutor(conf)
        result = await executor.execute_single(code)
        return json.dumps(
            {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "execution_time": result.execution_time,
            },
            ensure_ascii=False,
        )

    @mcp.tool
    async def execute_codes_parallel(code_list: List[str], timeout: Optional[float] = None) -> str:
        conf = base_conf
        if isinstance(timeout, (int, float)) and float(timeout) > 0:
            conf = ExecutionConfig(**{**base_conf.__dict__, "timeout": float(timeout)})
        executor = PythonExecutor(conf)
        results = await executor.execute_multiple(list(code_list or []))
        return json.dumps(
            [
                {
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                    "execution_time": r.execution_time,
                }
                for r in results
            ],
            ensure_ascii=False,
        )

    @mcp.tool
    async def list_supported_languages() -> str:
        return json.dumps(["python"], ensure_ascii=False)

    return mcp


def main(argv: Optional[List[str]] = None) -> None:
    default_temp_dir = str((Path(__file__).resolve().parent / "data").resolve())
    parser = argparse.ArgumentParser(description="AgenQA Python code verifier (MCP server)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--name", type=str, default="AgenQA-CodeVerifier")
    parser.add_argument("--temp-dir", type=str, default=default_temp_dir)
    parser.add_argument("--memory-mb", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    args = parser.parse_args(argv)

    mcp = create_server(
        name=args.name,
        temp_dir=args.temp_dir,
        memory_limit_mb=args.memory_mb,
        default_timeout=args.timeout,
        python_bin=args.python_bin,
    )
    mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
