"""Tool-enabled solver runner.

This runner is used by an optional solver tier that may output executable Python
code (ToolCode). The pipeline will execute ToolCode to obtain a numeric value
for verification and/or as the solver's predicted answer.

Design notes:
- Tool artifacts are internal (edge-only). They should never be injected into
  path prompts.
- This runner keeps output rows compatible with existing solve_*.jsonl rows,
  and adds a `tool` sub-object for debugging/analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, Optional

from infra.data.io import read_jsonl, write_jsonl
from infra.llm.inference import resolve_inference
from infra.llm.service_client import LLMServiceSession
from utils import ensure_dir

from infra.code_verifier.executor import ExecutionConfig
from infra.code_verifier.python_executor import PythonExecutor

from agenqa.domain.known_utils import format_known_for_solver

from agenqa.domain.solver_schema import (
    FIELD_ANSWER,
    FIELD_SOLVER_REASONING,
    FIELD_QUESTION_WELL_POSED,
    FIELD_CORRECTNESS_FEEDBACK,
    FIELD_DIFFICULTY_FEEDBACK,
    FIELD_KEY_CONCLUSION,
)
from agenqa.domain.solver_tool_schema import (
    FIELD_TOOL_CODE,
    FIELD_TOOL_NAME,
    FIELD_TOOL_NOTES,
    FIELD_TOOL_USED,
)
from agenqa.skills.base import BaseSkillRunner
from agenqa.skills.solving import _solver_status_from_outcome
from infra.text.json_sanitize import sanitize_json_text
from agenqa.skills.solving import _estimate_tokens, _fix_unclosed_boxed

from agenqa.prompts.solver_tool import SOLVER_TOOL_PROMPT, SOLVER_TOOL_PROMPT_EN

logger = logging.getLogger(__name__)


def _iter_kqa_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Iterate KQA rows from JSONL."""
    try:
        yield from read_jsonl(path, schema=None)
    except Exception:
        return


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


def _extract_first_brace_block(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


_FLOAT_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_executor_stdout(stdout: str) -> Optional[float]:
    txt = (stdout or "").strip()
    if not txt:
        return None
    # Prefer JSON object with {"value": ...} on the last line.
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
    # Fallback: pick the first scalar token in output.
    m = _FLOAT_TOKEN_RE.search(txt)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def _format_numeric_boxed(value: float) -> str:
    s = f"{value:.12g}"
    return f"\\\\boxed{{{s}}}"


def _extract_finish_reason(response: Any) -> Optional[str]:
    try:
        choices: Any = None
        if isinstance(response, dict):
            choices = response.get("choices")
        else:
            choices = getattr(response, "choices", None)
        if not choices or not isinstance(choices, list):
            return None
        first = choices[0] if choices else None
        if isinstance(first, dict):
            val = first.get("finish_reason")
        else:
            val = getattr(first, "finish_reason", None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        return None
    return None


def _extract_gateway_finish_reason(response: Any) -> Optional[str]:
    try:
        if not isinstance(response, dict):
            return None
        gw = response.get("_gateway")
        if not isinstance(gw, dict):
            return None
        val = gw.get("finish_reason_raw") or gw.get("stop_reason")
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        return None
    return None


def _normalize_solver_error_code(
    error_msg: Optional[str],
    *,
    finish_reason: Optional[str],
    text: str,
) -> Optional[str]:
    if isinstance(finish_reason, str) and finish_reason.lower() == "length":
        return "truncated"
    if not (text or "").strip() and not (error_msg or "").strip():
        return "empty_content"
    if not isinstance(error_msg, str) or not error_msg.strip():
        return None
    msg = error_msg.strip().lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg or "connect" in msg or "ssl" in msg:
        return "connection_error"
    return "exception"


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _as_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float, bool)):
        return str(val)
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


@dataclass
class SolverToolConfig:
    generator: Dict[str, Any]
    prompt_path: Any  # Path-like
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    prompt_vars: Optional[Dict[str, Any]] = None
    # Execution limits for ToolCode
    timeout_seconds: float = 10.0
    memory_limit_mb: int = 4096
    temp_dir: str = "/tmp"
    python_bin: str = "python"


class SolverToolRunner:
    """A tool-enabled solver that can emit executable Python code."""

    def __init__(self, config: SolverToolConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = SOLVER_TOOL_PROMPT_EN if use_en else SOLVER_TOOL_PROMPT
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)

    def _render_prompt(self, known: str, question: str) -> str:
        known_text = format_known_for_solver(known)
        payload = {
            "known": str(known_text or ""),
            "question": str(question or ""),
            "answer_output_spec": "",
        }
        extra_vars = getattr(self.config, "prompt_vars", None)
        if isinstance(extra_vars, dict):
            payload.update(extra_vars)
        return self.prompt_template.safe_substitute(payload)

    def _parse_json_obj(self, raw_text: str) -> Dict[str, Any] | None:
        candidate = _extract_fenced_json(raw_text)
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            try:
                obj = json.loads(sanitize_json_text(candidate))
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
        # Fallback: first balanced {...}
        block = _extract_first_brace_block(candidate)
        if not block:
            return None
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            try:
                obj = json.loads(sanitize_json_text(block))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

    def _extract_tool_fields(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        used = _as_bool(obj.get(FIELD_TOOL_USED) or obj.get("tool_used") or obj.get("use_tool"))
        name = _as_str(obj.get(FIELD_TOOL_NAME) or obj.get("tool_name") or "python_executor").strip() or "python_executor"
        code = _as_str(obj.get(FIELD_TOOL_CODE) or obj.get("tool_code") or obj.get("oracle_code") or obj.get("code")).strip()
        notes = _as_str(obj.get(FIELD_TOOL_NOTES) or obj.get("tool_notes") or "").strip()
        # If code is present, treat as used (even if the model forgot ToolUsed).
        if code and not used:
            used = True
        return {
            "used": used,
            "name": name,
            "code": code,
            "notes": notes,
        }

    def _execute_tool_code(self, code: str) -> tuple[Optional[float], Dict[str, Any]]:
        exe = PythonExecutor(
            ExecutionConfig(
                timeout=float(self.config.timeout_seconds),
                memory_limit_mb=int(self.config.memory_limit_mb),
                temp_dir=str(self.config.temp_dir),
                python_bin=str(self.config.python_bin),
            )
        )
        res = asyncio.run(exe.execute_single(code))
        payload = {
            "success": bool(getattr(res, "success", False)),
            "output": getattr(res, "output", "") or "",
            "error": getattr(res, "error", "") or "",
            "execution_time": getattr(res, "execution_time", 0.0),
        }
        if not payload["success"]:
            return None, payload
        val = _parse_executor_stdout(payload["output"])
        return val, payload

    def run(self, kqa_path: Path, output_path: Path, append: bool = False) -> Path:
        output_dir = ensure_dir(str(output_path.parent))
        raw_path = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix or '.jsonl'}")

        solved_rows: list[Dict[str, Any]] = []
        raw_rows: list[Dict[str, Any]] = []

        for record in _iter_kqa_records(kqa_path):
            known = record.get("known") or record.get("Known") or ""
            query = record.get("question") or ""
            gt = record.get("answer") or record.get("Answer") or ""
            if not (known and query):
                continue

            prompt = self._render_prompt(str(known), str(query))
            response = None
            text = ""
            error_msg: Optional[str] = None
            finish_reason: Optional[str] = None
            gateway_finish_reason: Optional[str] = None
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.session.chat(messages, **self._chat_args)
                finish_reason = _extract_finish_reason(response)
                gateway_finish_reason = _extract_gateway_finish_reason(response)
                text = self.session.extract_text(response, default="")
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                gateway_finish_reason = None
            error_code = _normalize_solver_error_code(error_msg, finish_reason=finish_reason, text=text)
            error_detail = (
                (error_msg.strip() if isinstance(error_msg, str) and error_msg.strip() else None)
                if error_code in {"timeout", "connection_error", "exception"}
                else None
            )

            obj = self._parse_json_obj(text) if text else None
            answer_reported = ""
            if isinstance(obj, dict):
                answer_reported = _as_str(obj.get(FIELD_ANSWER)).strip()
            if not answer_reported:
                # Fallback: keep raw text (truncated) so downstream can still inspect.
                answer_reported = (text or "").strip()

            tool = self._extract_tool_fields(obj or {}) if isinstance(obj, dict) else {"used": False, "name": "python_executor", "code": "", "notes": ""}
            tool_value: Optional[float] = None
            exec_payload: Optional[Dict[str, Any]] = None
            answer_from_tool: Optional[str] = None
            if tool.get("used") and isinstance(tool.get("code"), str) and tool["code"].strip():
                tool_value, exec_payload = self._execute_tool_code(tool["code"])
                if tool_value is not None:
                    answer_from_tool = _format_numeric_boxed(tool_value)

            gt_fixed = _fix_unclosed_boxed(str(gt or ""))
            pred_fixed = _fix_unclosed_boxed(answer_from_tool or answer_reported or "")
            # 正确性判定由上游 solve 节点的 LLM judge 统一裁决；tool runner 仅产出预测与可观测信息。
            ok = None
            if error_code:
                ok = False
                pred_fixed = ""

            # Policy (P3): if the solver claims tool usage and the tool run fails, treat as invalid.
            # - Mark `correct=False` and do NOT allow the reported answer to be used as a fallback prediction.
            # - Surface as an error so downstream (consensus / flow control) can retry or switch solvers.
            tool_used = bool(tool.get("used"))
            tool_code_present = isinstance(tool.get("code"), str) and bool(tool["code"].strip())
            tool_exec_success = (
                bool(exec_payload.get("success")) if isinstance(exec_payload, dict) and "success" in exec_payload else None
            )
            tool_value_parsed = tool_value is not None
            tool_claimed_but_missing_code = tool_used and (not tool_code_present)
            tool_executed_but_failed = (tool_exec_success is False) or (
                tool_code_present and (exec_payload is not None) and (tool_exec_success is True) and (not tool_value_parsed)
            )
            if tool_claimed_but_missing_code or tool_executed_but_failed:
                ok = False
                pred_fixed = ""
                error_code = error_code or "tool_exec_failed"
            solver_status, solver_failure_code, solver_failure_stage = _solver_status_from_outcome(
                error_code=error_code,
                answer_pred=pred_fixed,
                raw_text=text,
            )
            solver_failure_detail = error_detail

            usage = (response or {}).get("usage") if isinstance(response, dict) else {}
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            if not isinstance(completion_tokens, int):
                completion_tokens = _estimate_tokens(text or pred_fixed)
            kq_tokens = _estimate_tokens(f"{known}\n{query}")
            token_ratio = None

            pid_val = record.get("paper_id")
            step_val = record.get("step")
            try:
                step_int = int(step_val) if step_val is not None else None
            except Exception:
                step_int = None
            chain_value = record.get("chain") or (f"k{step_int},q{step_int},a{step_int}" if step_int is not None else None)

            # Feedback fields (optional)
            question_well_posed = None
            correctness_feedback = None
            difficulty_feedback = None
            key_conclusion = None
            solver_reasoning = None
            if isinstance(obj, dict):
                question_well_posed = obj.get(FIELD_QUESTION_WELL_POSED)
                correctness_feedback = obj.get(FIELD_CORRECTNESS_FEEDBACK)
                difficulty_feedback = obj.get(FIELD_DIFFICULTY_FEEDBACK)
                key_conclusion = obj.get(FIELD_KEY_CONCLUSION)
                solver_reasoning = obj.get(FIELD_SOLVER_REASONING)

            solved_rows.append(
                {
                    "known": known,
                    "question": query,
                    "answer": gt_fixed,
                    "solve": pred_fixed,
                    "answer_ref": gt_fixed,
                    "answer_pred": pred_fixed,
                    "solver_status": solver_status,
                    "solve_reported": answer_reported,
                    "correct": ok,
                    "token_ratio": token_ratio,
                    "metrics": {
                        "kq_tokens": kq_tokens,
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": (usage.get("prompt_tokens") if isinstance(usage, dict) else None),
                    },
                    "paper_id": pid_val,
                    "step": step_int,
                    "chain": chain_value,
                    "question_well_posed": question_well_posed if isinstance(question_well_posed, bool) else None,
                    "correctness_feedback": _as_str(correctness_feedback).strip() if isinstance(correctness_feedback, str) else None,
                    "difficulty_feedback": _as_str(difficulty_feedback).strip() if isinstance(difficulty_feedback, str) else None,
                    "key_conclusion": _as_str(key_conclusion).strip() if isinstance(key_conclusion, str) else None,
                    "solver_reasoning": _as_str(solver_reasoning).strip() if isinstance(solver_reasoning, str) else None,
                    "tool": {
                        "used": bool(tool.get("used")),
                        "name": tool.get("name"),
                        "value": tool_value,
                        "exec": exec_payload,
                        "code": tool.get("code"),
                        "notes": tool.get("notes"),
                    },
                    "model": self.session.model_name,
                    "service_id": self.session.service_id,
                    "finish_reason": finish_reason,
                    "gateway_finish_reason": gateway_finish_reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **(
                        {"error": error_code} if error_code else {}
                    ),
                    **({"error_detail": error_detail} if error_detail else {}),
                    **({"solver_failure_code": solver_failure_code} if solver_failure_code else {}),
                    **({"solver_failure_stage": solver_failure_stage} if solver_failure_stage else {}),
                    **({"solver_failure_detail": solver_failure_detail} if solver_failure_detail else {}),
                }
            )

            raw_rows.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input": {"known": known, "question": query},
                    "prompt": prompt,
                    "response": response,
                    "text": text,
                    "usage": usage,
                    "gen_params": {"model": self.session.model_name, "service_id": self.session.service_id},
                    "finish_reason": finish_reason,
                    "gateway_finish_reason": gateway_finish_reason,
                    "tool_exec": exec_payload,
                    **({"error": error_code} if error_code else {}),
                    **({"error_detail": error_detail} if error_detail else {}),
                }
            )

        write_jsonl(solved_rows, output_path, schema=None, append=append)
        write_jsonl(raw_rows, raw_path, schema=None, append=append)
        logger.info("SolverTool completed, output=%s", str(output_path))
        return output_dir / output_path.name


__all__ = [
    "SolverToolConfig",
    "SolverToolRunner",
]
