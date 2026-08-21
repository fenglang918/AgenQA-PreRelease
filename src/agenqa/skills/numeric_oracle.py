"""NumericOracle role: generate oracle code + tolerance for Numeric questions.

This is an internal helper for semantic Numeric questions:
- It produces deterministic Python code that computes the ground-truth value.
- It produces per-question tolerance/precision conventions (operator-defined).

The executor is sandboxed and must not leak oracle outputs into path prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional, Tuple

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used, snapshot_rendered_prompt
from infra.llm.service_client import LLMServiceSession
from agenqa.domain.numeric_oracle_schema import (
    FIELD_ABS_TOL,
    FIELD_NOTES,
    FIELD_ORACLE_CODE,
    FIELD_REL_TOL,
    FIELD_SIG_FIGS,
    FIELD_UNIT,
)
from infra.text.json_policy import clean_json_text
from agenqa.skills.base import BaseSkillRunner
from infra.text.json_sanitize import sanitize_json_text

from infra.code_verifier.executor import ExecutionConfig
from infra.code_verifier.python_executor import PythonExecutor

from agenqa.prompts.numeric_oracle import NUMERIC_ORACLE_V1, NUMERIC_ORACLE_V1_EN

logger = logging.getLogger(__name__)


@dataclass
class NumericOracleInput:
    step: int
    question: str
    solution: str


@dataclass
class NumericOracleOutput:
    abs_tol: Optional[float] = None
    rel_tol: Optional[float] = None
    sig_figs: Optional[int] = None
    unit: str = ""
    oracle_code: str = ""
    gt_value: Optional[float] = None
    exec_payload: Optional[Dict[str, Any]] = None
    notes: str = ""


@dataclass
class NumericOracleConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    # Execution limits
    timeout_seconds: float = 10.0
    memory_limit_mb: int = 4096
    temp_dir: str = "/tmp"
    python_bin: str = "python"


def _extract_fenced_json(text: str) -> str:
    candidate = (text or "").strip()
    think_end = "</think>"
    if think_end in candidate:
        candidate = candidate.split(think_end, 1)[1].strip()
    idx = candidate.find("```json")
    if idx != -1:
        end = candidate.find("```", idx + len("```json"))
        if end != -1:
            return candidate[idx + len("```json") : end].strip()
        return candidate[idx + len("```json") :].strip()
    idx = candidate.find("```")
    if idx != -1:
        end = candidate.find("```", idx + len("```"))
        if end != -1:
            return candidate[idx + len("```") : end].strip()
        return candidate[idx + len("```") :].strip()
    return candidate


def _as_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        out = float(val)
        return out if out > 0 else None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            out = float(s)
            return out if out > 0 else None
        except Exception:
            return None
    return None


def _as_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, int):
        return val if val > 0 else None
    if isinstance(val, float):
        try:
            iv = int(val)
            return iv if iv > 0 else None
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            iv = int(float(s))
            return iv if iv > 0 else None
        except Exception:
            return None
    return None


def _validate_tolerance(abs_tol: Optional[float], rel_tol: Optional[float], sig_figs: Optional[int]) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Enforce the \"choose one convention\" contract with safe fallbacks.

    - Prefer sig_figs if provided.
    - Otherwise require at least one of abs_tol/rel_tol.
    """
    if sig_figs is not None:
        return None, None, sig_figs
    if abs_tol is None and rel_tol is None:
        # Conservative fallback: require at least some tolerance to keep eval well-defined.
        return 1e-6, 1e-4, None
    return abs_tol, rel_tol, None


def _parse_executor_stdout(stdout: str) -> Optional[float]:
    txt = (stdout or "").strip()
    if not txt:
        return None
    # Prefer JSON object with {"value": ...}
    try:
        obj = json.loads(txt.splitlines()[-1])
        if isinstance(obj, dict) and "value" in obj:
            val = obj.get("value")
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val.strip())
                except Exception:
                    return None
    except Exception:
        pass
    # Fallback: pick the first scalar in output.
    m = re.search(r"[-+]?(?:\\d+(?:\\.\\d+)?|\\.\\d+)(?:[eE][-+]?\\d+)?", txt)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def execute_oracle_code(
    code: str,
    *,
    timeout_seconds: float = 10.0,
    memory_limit_mb: int = 4096,
    temp_dir: str = "/tmp",
    python_bin: str = "python",
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Execute deterministic python oracle code and parse a single numeric value.

    Contract:
    - Code should be deterministic and sandbox-safe (no network / external file I/O).
    - stdout should print a JSON object line: {"value": <number>}.
    """
    exec_cfg = ExecutionConfig(
        timeout=float(timeout_seconds),
        memory_limit_mb=int(memory_limit_mb),
        temp_dir=str(temp_dir),
        python_bin=str(python_bin),
    )
    executor = PythonExecutor(exec_cfg)

    async def _run() -> Any:
        return await executor.execute_single(code)

    try:
        res = asyncio.run(_run())
    except RuntimeError as exc:
        raise RuntimeError(
            "execute_oracle_code() must run in a synchronous context (no running event loop in the same thread)"
        ) from exc

    payload = {
        "success": bool(getattr(res, "success", False)),
        "output": getattr(res, "output", ""),
        "error": getattr(res, "error", ""),
        "execution_time": getattr(res, "execution_time", 0.0),
    }
    if not payload["success"]:
        return None, payload
    val = _parse_executor_stdout(payload["output"])
    return val, payload


def format_numeric_answer(value: float, *, sig_figs: Optional[int] = None) -> str:
    """Format a numeric value into a stable `\\boxed{...}` answer string."""
    if sig_figs is not None and sig_figs > 0:
        s = f"{value:.{int(sig_figs)}g}"
    else:
        # Default: a compact but stable representation.
        s = f"{value:.12g}"
    return f"\\\\boxed{{{s}}}"


def build_numeric_answer_format_sentence(
    *,
    abs_tol: Optional[float],
    rel_tol: Optional[float],
    sig_figs: Optional[int],
    lang: str | None,
) -> str:
    """Build a deterministic tolerance sentence for solver-visible contract text/logs."""
    use_en = str(lang or "").strip().lower() in {"en", "english"}
    if sig_figs is not None and sig_figs > 0:
        if use_en:
            return f"Answer format: give a single numeric value in LaTeX \\\\boxed{{...}} with {int(sig_figs)} significant figures."
        return f"答案格式：请给出一个数值，用 LaTeX \\\\boxed{{...}} 包裹；保留 {int(sig_figs)} 位有效数字。"

    parts = []
    if abs_tol is not None:
        parts.append(f"abs_tol={abs_tol:g}")
    if rel_tol is not None:
        parts.append(f"rel_tol={rel_tol:g}")
    tol_text = ", ".join(parts) if parts else "abs_tol=1e-6, rel_tol=1e-4"
    if use_en:
        return f"Answer format: give a single numeric value in LaTeX \\\\boxed{{...}}; acceptable error: {tol_text}."
    return f"答案格式：请给出一个数值，用 LaTeX \\\\boxed{{...}} 包裹；允许误差口径：{tol_text}。"


class NumericOracleRunner:
    def __init__(self, config: NumericOracleConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = NUMERIC_ORACLE_V1_EN if use_en else NUMERIC_ORACLE_V1
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(self, oracle_in: NumericOracleInput) -> str:
        payload: Dict[str, Any] = {
            "step": int(oracle_in.step),
            "question": oracle_in.question or "",
            "solution": oracle_in.solution or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        lang = (self.config.lang or "zh").lower()
        header = f"Role: NumericOracle | Language: {lang}" if lang in {"en", "english"} else f"【NumericOracle 角色】语言: {lang}"
        return "\n".join([header, "", body])

    def _parse_llm_output(self, text: str) -> NumericOracleOutput:
        candidate = _extract_fenced_json(text)
        cleaned = clean_json_text(
            candidate,
            generator=self.config.generator,
            task_name="numeric_oracle",
            lang=self.config.lang or "zh",
            required_keys=[[FIELD_ORACLE_CODE]],
            prompt_body=None,
            snapshot_dir=None,
            allow_python=False,
        )
        if not cleaned:
            cleaned = candidate
        try:
            data = json.loads(cleaned)
        except Exception:
            data = json.loads(sanitize_json_text(cleaned))
        if not isinstance(data, dict):
            raise ValueError("NumericOracle output is not a JSON object")

        oracle_code = data.get(FIELD_ORACLE_CODE)
        if not isinstance(oracle_code, str) or not oracle_code.strip():
            raise ValueError("numeric_oracle.oracle_code missing")

        abs_tol = _as_float(data.get(FIELD_ABS_TOL))
        rel_tol = _as_float(data.get(FIELD_REL_TOL))
        sig_figs = _as_int(data.get(FIELD_SIG_FIGS))
        unit = str(data.get(FIELD_UNIT) or "").strip()
        notes = str(data.get(FIELD_NOTES) or "").strip()
        abs_tol, rel_tol, sig_figs = _validate_tolerance(abs_tol, rel_tol, sig_figs)

        return NumericOracleOutput(
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            sig_figs=sig_figs,
            unit=unit,
            oracle_code=oracle_code.strip(),
            gt_value=None,
            notes=notes,
        )

    def _execute_oracle_code(self, code: str) -> Tuple[Optional[float], Dict[str, Any]]:
        return execute_oracle_code(
            code,
            timeout_seconds=float(self.config.timeout_seconds),
            memory_limit_mb=int(self.config.memory_limit_mb),
            temp_dir=str(self.config.temp_dir),
            python_bin=str(self.config.python_bin),
        )

    def run_one(
        self,
        oracle_in: NumericOracleInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.numeric_oracle.",
    ) -> NumericOracleOutput:
        prompt_body = self._build_prompt(oracle_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "NumericOracle")
        raw_text = self.session.extract_text(response, default="") or ""

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                (snap_path / "input_view.json").write_text(
                    json.dumps(
                        {
                            "step": oracle_in.step,
                            "question": oracle_in.question,
                            "solution": oracle_in.solution,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                snapshot_prompt_used(
                    Path(str(self.config.prompt_path)),
                    snap_path,
                    content=self.prompt_text,
                    name_prefix="prompt_used.",
                    logger=logger,
                )
                snapshot_rendered_prompt(prompt_body, snap_path, logger=logger)
                (snap_path / "llm_output.txt").write_text(raw_text, encoding="utf-8")
            except Exception:
                snap_path = None

        out = self._parse_llm_output(raw_text)
        gt, exec_payload = self._execute_oracle_code(out.oracle_code)
        out.gt_value = gt
        out.exec_payload = exec_payload

        if snap_path is not None:
            try:
                (snap_path / "parsed_output.json").write_text(
                    json.dumps(
                        {
                            FIELD_ABS_TOL: out.abs_tol,
                            FIELD_REL_TOL: out.rel_tol,
                            FIELD_SIG_FIGS: out.sig_figs,
                            FIELD_UNIT: out.unit,
                            FIELD_NOTES: out.notes,
                            "gt_value": out.gt_value,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (snap_path / "oracle_code.py").write_text(out.oracle_code, encoding="utf-8")
                (snap_path / "executor_result.json").write_text(
                    json.dumps(exec_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # Persist prompt snapshot into unified directory (if provided).
        if unified_prompt_dir is not None:
            try:
                snapshot_prompt_used(
                    Path(str(self.config.prompt_path)),
                    Path(unified_prompt_dir),
                    content=self.prompt_text,
                    name_prefix=name_prefix,
                    logger=logger,
                )
            except Exception:
                pass

        return out


__all__ = [
    "NumericOracleInput",
    "NumericOracleOutput",
    "NumericOracleConfig",
    "NumericOracleRunner",
    "execute_oracle_code",
    "format_numeric_answer",
    "build_numeric_answer_format_sentence",
]
