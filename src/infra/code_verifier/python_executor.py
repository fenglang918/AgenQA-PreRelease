"""Python executor: run untrusted code with time/memory limits."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from .executor import CodeExecutor, ExecutionConfig, ExecutionResult


def _remove_python_code_block_markers(code: str) -> str:
    stripped = (code or "").strip()
    if not stripped.startswith("```"):
        return stripped
    import re

    m = re.match(r"```(?:python|py)?\n([\s\S]*?)```$", stripped)
    if m:
        return (m.group(1) or "").strip()
    return stripped


def _resolve_writable_temp_dir(configured: str) -> Path:
    """Resolve a writable temp directory.

    Some configs are shared between Linux servers and macOS laptops. When a config
    points temp_dir to a server-only path (e.g. /data2/...), we fall back to a
    local writable directory instead of failing the whole pipeline.
    """

    def _try_dir(path_str: str) -> Optional[Path]:
        if not isinstance(path_str, str):
            return None
        s = path_str.strip()
        if not s:
            return None
        try:
            p = Path(s).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            probe = p / f".agenqa_write_probe_{uuid.uuid4().hex}"
            probe.mkdir(parents=False, exist_ok=False)
            probe.rmdir()
            return p
        except Exception:
            return None

    # Env override first (useful when running a server config on a laptop).
    env_override = os.environ.get("CODE_VERIFIER_TEMP_DIR") or os.environ.get("AGENT_CODE_VERIFIER_TEMP_DIR")
    for cand in [
        env_override,
        configured,
        os.environ.get("TMPDIR", ""),
        tempfile.gettempdir(),
    ]:
        got = _try_dir(str(cand) if cand is not None else "")
        if got is not None:
            return got

    # Final fallback: always try to create a dedicated folder under system temp.
    fallback = Path(tempfile.gettempdir()) / "agenqa_code_verifier"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return fallback


def _resolve_python_bin(configured: str) -> str:
    s = str(configured or "").strip()
    if not s:
        return sys.executable
    # If it looks like a path, trust the caller.
    if "/" in s or os.path.sep in s:
        return s
    if shutil.which(s):
        return s
    return sys.executable


class PythonExecutor(CodeExecutor):
    def __init__(self, config: Optional[ExecutionConfig] = None):
        super().__init__(config)
        resolved = _resolve_writable_temp_dir(getattr(self.config, "temp_dir", "/tmp"))
        self.config.temp_dir = str(resolved)
        self.config.python_bin = _resolve_python_bin(getattr(self.config, "python_bin", "python"))

    async def execute_single(self, code: str) -> ExecutionResult:
        if not self.validate_code(code):
            return ExecutionResult(success=False, output="", error="code_validation_failed")

        code = self.preprocess_code(code)
        start = time.time()

        base_temp_dir = Path(self.config.temp_dir)
        session_dir = base_temp_dir / f"verify_session_{uuid.uuid4().hex}"
        script_path = session_dir / "script.py"

        process: asyncio.subprocess.Process | None = None
        try:
            # Keep setup synchronous to avoid forking a multi-threaded process.
            session_dir.mkdir(parents=True, exist_ok=True)
            script_path.write_text(code, encoding="utf-8")

            def _set_limits() -> None:
                try:
                    import resource

                    limit_bytes = int(self.config.memory_limit_mb) * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                except Exception:
                    pass

            # Use absolute paths to avoid cwd-relative path pitfalls when temp_dir is relative.
            session_dir_abs = session_dir.resolve()
            script_path_abs = script_path.resolve()
            cmd = [str(self.config.python_bin or "python"), str(script_path_abs)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_set_limits if os.name == "posix" else None,
                cwd=str(session_dir_abs),
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=float(self.config.timeout))
            except asyncio.TimeoutError:
                if process:
                    try:
                        process.kill()
                        await process.communicate()
                    except Exception:
                        pass
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"timeout_after_{self.config.timeout}_seconds",
                    execution_time=time.time() - start,
                )

            out = (stdout or b"").decode("utf-8", errors="replace")
            err = (stderr or b"").decode("utf-8", errors="replace")
            ok = (process.returncode or 0) == 0
            if not ok:
                return ExecutionResult(
                    success=False,
                    output=out,
                    error=f"nonzero_exit({process.returncode})\n---STDERR---\n{err}",
                    execution_time=time.time() - start,
                )
            return ExecutionResult(success=True, output=out, error="", execution_time=time.time() - start)
        except asyncio.CancelledError:
            if process:
                try:
                    process.kill()
                    await process.communicate()
                except Exception:
                    pass
            raise
        except Exception:
            return ExecutionResult(
                success=False,
                output="",
                error=traceback.format_exc(),
                execution_time=time.time() - start,
            )
        finally:
            if session_dir.exists():
                try:
                    shutil.rmtree(session_dir)
                except Exception:
                    pass

    def validate_code(self, code: str) -> bool:
        if not super().validate_code(code):
            return False
        try:
            compile(_remove_python_code_block_markers(code), "<verifier>", "exec")
            return True
        except SyntaxError:
            return False

    def preprocess_code(self, code: str) -> str:
        return _remove_python_code_block_markers(super().preprocess_code(code))
