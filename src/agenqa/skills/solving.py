"""KQA 求解器。

对 KnownInit/Extend 生成的 K/Q/A 记录进行“看不到 GT answer”的求解，
并将结果以 known/question/answer/solve 写出。

注意：
本仓库的 semantic/unified pipeline 默认不在 SolverRunner 内做“程序化正确性判分”：
正确性由上游 solve 节点的 LLM judge（expression_judge / numeric_judge）统一裁决，
避免 LaTeX 排版噪声、等价变形、单位换算等导致的误判与隐式 fallback。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import re
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Set
import os

from infra.data.io import read_jsonl, write_jsonl
from infra.llm.service_client import LLMServiceSession
from infra.llm.inference import resolve_inference
from infra.llm.service_loader import load_llm_service_full_config
from utils import ensure_dir
from infra.prompt.prompt_builder import load_prompt_fragment
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from agenqa.prompts.agent_prompts import SOLVER_PROMPT
from agenqa.domain.contracts.solver_contract_text import compose_solver_question
from agenqa.domain.known_utils import format_known_for_solver
from agenqa.domain.solver_schema import (
    FIELD_ANSWER,
    FIELD_SOLVER_REASONING,
    FIELD_FEEDBACK,
    FIELD_HARDER_SUGGESTION,
    FIELD_KEY_CONCLUSION,
    FIELD_QUESTION_WELL_POSED,
    FIELD_CORRECTNESS_FEEDBACK,
    FIELD_DIFFICULTY_FEEDBACK,
    solver_output_base,
)


logger = logging.getLogger(__name__)


DEFAULT_LLM_SERVICE_CONFIG = Path(
    os.getenv("LLM_SERVICES_JSON", "config/services.json")
)
DEFAULT_SERVICE_ID = "remote:qwen3-30b-a3b-thinking"


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


def _solver_status_from_outcome(
    *,
    error_code: Optional[str],
    answer_pred: Optional[str],
    raw_text: str,
) -> tuple[str, Optional[str], Optional[str]]:
    """Classify solver availability for downstream judge/consensus.

    Returns:
      (solver_status, failure_code, failure_stage)
    """
    if isinstance(error_code, str) and error_code.strip():
        return "request_failed", error_code.strip(), "request"
    if isinstance(answer_pred, str) and answer_pred.strip():
        return "success", None, None
    if isinstance(raw_text, str) and raw_text.strip():
        return "parse_failed", "missing_answer_field", "extraction"
    return "request_failed", "empty_content", "request"


def _load_service_from_llm_config(config_path: Path, service_id: str) -> Dict[str, Any]:
    """保持原接口名，委托给共享的 service_loader。"""
    return load_llm_service_full_config(config_path, service_id)


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


def _iter_kqa_records(path: Path) -> Iterable[Dict[str, Any]]:
    """迭代 KQA 记录，支持 JSONL 与连续 JSON 对象两种形式。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        yield from read_jsonl(path, schema=None)
        return

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    yielded = 0
    while True:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                yielded += 1
                yield obj
            idx = end
        except json.JSONDecodeError:
            yielded = 0
            break

    if yielded == 0:
        yield from read_jsonl(path, schema=None)


def _load_solved_keys(path: Path) -> Set[str]:
    """从已存在的 solve 输出中加载已完成的 (paper_id, step) 集合。

    规则：
    - key 形如 "paper_id::step"（step 缺失时用 "None" 占位）；
    - 包含 error 的记录视为未完成（允许重试）。
    """
    solved: Set[str] = set()
    try:
        for row in read_jsonl(path, schema=None):
            if not isinstance(row, dict):
                continue
            if row.get("error"):
                continue
            pid = row.get("paper_id")
            step_val = row.get("step")
            step_str = str(step_val) if step_val is not None else "None"
            if isinstance(pid, str) and pid:
                solved.add(f"{pid}::{step_str}")
    except Exception:
        # 容错：读取失败不影响后续
        pass
    return solved


@dataclass
class SolverConfig:
    """求解配置。"""

    prompt_path: Path
    # 可选：直接提供模板文本（Template 语法），优先于 prompt_path。
    prompt_text: Optional[str] = None
    # 可选：控制提示语言（例如 "zh" 或 "en"），用于选择通用片段语言等。
    lang: Optional[str] = None
    # 额外 prompt 变量，用于注入 solver-visible 的上下文片段。
    prompt_vars: Optional[Dict[str, Any]] = None
    # 如未提供，将自动从 llm_service 的 services.json 中按 DEFAULT_SERVICE_ID 解析
    generator: Optional[Dict[str, Any]] = None
    service_config_path: Path = DEFAULT_LLM_SERVICE_CONFIG
    service_id: str = DEFAULT_SERVICE_ID


class SolverRunner(BaseSkillRunner):
    """基于 Known + Question 求解（看不到 GT）。"""

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        # 解析生成器配置
        generator = config.generator or _load_service_from_llm_config(
            config.service_config_path, config.service_id
        )
        super().__init__(generator)
        resolved = resolve_inference(generator)
        self.session: LLMServiceSession = resolved.session
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            use_code_prompt = os.getenv("SCICLONE_USE_CODE_PROMPTS", "").strip() == "1"
            if use_code_prompt:
                base_text = SOLVER_PROMPT
            else:
                try:
                    base_text = Path(config.prompt_path).read_text(encoding="utf-8")
                except FileNotFoundError:
                    base_text = SOLVER_PROMPT
        self.prompt_template: Template = Template(base_text)
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

    def run(self, kqa_path: Path, output_path: Path, append: bool = False, concurrency: int = 1) -> Path:
        output_dir = ensure_dir(str(output_path.parent))
        solved: List[Dict[str, Any]] = []
        raw_entries: List[Dict[str, Any]] = []
        # 并发模式：委托给并发实现
        if int(concurrency) > 1:
            return self._run_concurrent(kqa_path, output_path, append, int(concurrency))

        # 增量写入参数
        raw_path = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix or '.jsonl'}")
        append_flag = bool(append or output_path.exists())
        # 已存在进度：按 (paper_id, step) 跳过，允许同一论文不同 step 多次求解
        solved_keys: Set[str] = _load_solved_keys(output_path) if output_path.exists() else set()
        BATCH = 50

        for record in _iter_kqa_records(kqa_path):
            known_raw = record.get("known") or record.get("Known") or ""
            # Prefer the pre-composed solver question when available; otherwise compose it on the fly.
            query = (
                record.get("question_for_solver")
                or compose_solver_question(record.get("question") or "", record.get("world_contract_text"))
                or ""
            )
            # GT 来自输入记录，但不参与提示
            gt = record.get("answer") or record.get("Answer") or ""

            if not (known_raw and query):
                logger.warning("记录缺少 known/question 字段，已跳过: %s", str(record)[:160])
                continue
            known_text = format_known_for_solver(known_raw)

            error_msg: Optional[str] = None
            finish_reason: Optional[str] = None
            gateway_finish_reason: Optional[str] = None
            response = None
            text = ""
            try:
                prompt = self._render_prompt(known_text, query)
                messages = [{"role": "user", "content": prompt}]
                response = self.session.chat(messages, **self._chat_args)
                finish_reason = _extract_finish_reason(response)
                gateway_finish_reason = _extract_gateway_finish_reason(response)
                text = self.session.extract_text(response, default="")
            except Exception as e:  # noqa: BLE001
                error_msg = str(e)
                prompt = self._render_prompt(known_text, query)
                finish_reason = None
                gateway_finish_reason = None
            error_code = _normalize_solver_error_code(error_msg, finish_reason=finish_reason, text=text)
            error_detail = (
                (error_msg.strip() if isinstance(error_msg, str) and error_msg.strip() else None)
                if error_code in {"timeout", "connection_error", "exception"}
                else None
            )
            answer = self._parse_answer(text) if text else None
            fb_sug = self._parse_feedback_and_suggestion(text) if text else {
                "QuestionWellPosed": False,
                "CorrectnessFeedback": None,
                "DifficultyFeedback": None,
                "KeyConclusion": None,
                "SolverReasoning": None,
                "Feedback": None,
                "HarderSuggestion": None,
            }
            if not answer and text:
                logger.warning("solver parse failed: missing Answer field, raw response preview: %s", text[:200])
                answer = ""

            # 修复可能缺失的 \boxed{...} 右花括号
            gt = _fix_unclosed_boxed(gt)
            answer = _fix_unclosed_boxed(answer)
            solver_status, solver_failure_code, solver_failure_stage = _solver_status_from_outcome(
                error_code=error_code,
                answer_pred=answer,
                raw_text=text,
            )
            solver_failure_detail = error_detail

            # 记录 tokens 统计（用于后续 LLM judge 覆盖 correct/token_ratio 时重算）
            usage = (response or {}).get("usage") if isinstance(response, dict) else {}
            completion_tokens = (
                usage.get("completion_tokens")
                if isinstance(usage, dict)
                else None
            )
            if not isinstance(completion_tokens, int):
                # 回退：以 answer 文本估算输出 tokens
                completion_tokens = _estimate_tokens(answer)
            kq_tokens = _estimate_tokens(f"{known_text}\n{query}")
            token_ratio = None

            # 确定链路标识
            step_val = record.get("step")
            try:
                step_int = int(step_val) if step_val is not None else None
            except Exception:
                step_int = None
            chain_value = record.get("chain") or (f"k{step_int},q{step_int},a{step_int}" if step_int is not None else None)

            # 确定链路标识
            step_val = record.get("step")
            try:
                step_int = int(step_val) if step_val is not None else None
            except Exception:
                step_int = None
            chain_value = record.get("chain") or (f"k{step_int},q{step_int},a{step_int}" if step_int is not None else None)

            # 按 (paper_id, step) 写入求解结果
            pid_val = record.get("paper_id")
            step_str = str(step_int) if step_int is not None else "None"
            key = f"{pid_val}::{step_str}" if pid_val else None
            if key and key in solved_keys:
                continue
            if key:
                solved_keys.add(key)

            # 解析新字段和旧字段
            question_well_posed = fb_sug.get("QuestionWellPosed", False)
            correctness_feedback = fb_sug.get("CorrectnessFeedback")
            difficulty_feedback = fb_sug.get("DifficultyFeedback")
            key_conclusion = fb_sug.get("KeyConclusion")
            solver_reasoning = fb_sug.get("SolverReasoning")

            # 向后兼容：使用旧字段作为回退
            old_feedback = fb_sug.get("Feedback")
            old_harder_suggestion = fb_sug.get("HarderSuggestion")

            # 字段传播逻辑：
            # - correctness_feedback：即使 is_correct=False 也保留（便于 Revise-Correctness 使用）
            # - difficulty_feedback：仅在 well-posed 且 is_correct=True 时保留
            # - legacy 字段（Feedback/HarderSuggestion）仅作为回退来源，不再落盘写入
            correctness_feedback_used = (
                correctness_feedback if isinstance(correctness_feedback, str) and correctness_feedback.strip() else None
            ) or (old_feedback if isinstance(old_feedback, str) and old_feedback.strip() else None)
            difficulty_feedback_used = None
            if question_well_posed:
                if isinstance(difficulty_feedback, str) and difficulty_feedback.strip():
                    difficulty_feedback_used = difficulty_feedback
                elif isinstance(old_harder_suggestion, str) and old_harder_suggestion.strip():
                    difficulty_feedback_used = old_harder_suggestion
            key_conclusion_used = key_conclusion if isinstance(key_conclusion, str) and key_conclusion.strip() else None
            solver_reasoning_used = solver_reasoning if isinstance(solver_reasoning, str) and solver_reasoning.strip() else None
            if isinstance(solver_reasoning_used, str) and len(solver_reasoning_used) > 4000:
                solver_reasoning_used = solver_reasoning_used[:4000] + " ...[truncated]"

            solved.append(
                {
                    "known": known_raw,
                    "question": query,
                    "answer": gt,
                    "solve": answer,
                    "answer_ref": gt,
                    "answer_pred": answer,
                    "solver_status": solver_status,
                    "correct": None,
                    "token_ratio": token_ratio,
                    "metrics": {
                        "kq_tokens": kq_tokens,
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": (usage.get("prompt_tokens") if isinstance(usage, dict) else None),
                    },
                    # Propagate context for correlation/analysis
                    "paper_id": pid_val,
                    "step": step_int,
                    "chain": chain_value,
                    # 新字段：正确性与难度反馈
                    "question_well_posed": question_well_posed,
                    "correctness_feedback": correctness_feedback_used,
                    "difficulty_feedback": difficulty_feedback_used,
                    "key_conclusion": key_conclusion_used,
                    "solver_reasoning": solver_reasoning_used,
                    "model": self.session.model_name,
                    "service_id": self.session.service_id,
                    "finish_reason": finish_reason,
                    "gateway_finish_reason": gateway_finish_reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **({"error": error_code} if error_code else {}),
                    **({"error_detail": error_detail} if error_detail else {}),
                    **({"solver_failure_code": solver_failure_code} if solver_failure_code else {}),
                    **({"solver_failure_stage": solver_failure_stage} if solver_failure_stage else {}),
                    **({"solver_failure_detail": solver_failure_detail} if solver_failure_detail else {}),
                }
            )

            raw_entries.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input": {"known": known_raw, "question": query},
                    "prompt": prompt,
                    "response": response,
                    "text": text,
                    "usage": usage,
                    "gen_params": {
                        "model": self.session.model_name,
                        "service_id": self.session.service_id,
                    },
                    "finish_reason": finish_reason,
                    "gateway_finish_reason": gateway_finish_reason,
                    **({"error": error_code} if error_code else {}),
                    **({"error_detail": error_detail} if error_detail else {}),
                }
            )

            # 达到批次阈值则增量写入
            if len(solved) >= BATCH:
                write_jsonl(solved, output_path, schema=None, append=append_flag)
                solved.clear()
                append_flag = True
            if len(raw_entries) >= BATCH:
                # 原始响应始终追加
                write_jsonl(raw_entries, raw_path, schema=None, append=append_flag)
                raw_entries.clear()
                append_flag = True

        # 收尾写入
        if solved:
            write_jsonl(solved, output_path, schema=None, append=append_flag)
            append_flag = True
        if raw_entries:
            write_jsonl(raw_entries, raw_path, schema=None, append=append_flag)

        logger.info("KQA 求解完成，输出: %s", output_path)
        return output_dir / output_path.name

    def _run_concurrent(self, kqa_path: Path, output_path: Path, append: bool, concurrency: int) -> Path:
        # 与 legacy init/extend-upgrade 一致的线程池并发实现；每个线程持有独立的会话
        output_dir = ensure_dir(str(output_path.parent))
        # 增量写入参数
        raw_path = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix or '.jsonl'}")
        append_flag = bool(append or output_path.exists())
        # 已存在进度：按 (paper_id, step) 跳过，允许同一论文不同 step 多次求解
        solved_keys: Set[str] = _load_solved_keys(output_path) if output_path.exists() else set()
        def worker(record: Dict[str, Any]):
            known_raw = record.get("known") or record.get("Known") or ""
            query = record.get("question") or ""
            gt = record.get("answer") or record.get("Answer") or ""
            if not (known_raw and query):
                return None, None
            pid_val = record.get("paper_id")
            step_val = record.get("step")
            try:
                step_int = int(step_val) if step_val is not None else None
            except Exception:
                step_int = None
            step_str = str(step_int) if step_int is not None else "None"
            key = f"{pid_val}::{step_str}" if isinstance(pid_val, str) and pid_val else None
            if key and key in solved_keys:
                return None, None
            known_text = format_known_for_solver(known_raw)
            prompt = self._render_prompt(known_text, query)
            sess, chat_args = self._session_pool.get()
            messages = [{"role": "user", "content": prompt}]
            response = None
            text = ""
            error_msg: Optional[str] = None
            finish_reason: Optional[str] = None
            gateway_finish_reason: Optional[str] = None
            try:
                response = sess.chat(messages, **chat_args)
                finish_reason = _extract_finish_reason(response)
                gateway_finish_reason = _extract_gateway_finish_reason(response)
                text = sess.extract_text(response, default="")
                answer = self._parse_answer(text) if text else None
                if not answer and text:
                    logger.warning("solver parse failed: missing Answer field, raw response preview: %s", text[:200])
                    answer = ""
            except Exception as e:  # noqa: BLE001
                error_msg = str(e)
                answer = ""
                finish_reason = None
                gateway_finish_reason = None
            error_code = _normalize_solver_error_code(error_msg, finish_reason=finish_reason, text=text)
            error_detail = (
                (error_msg.strip() if isinstance(error_msg, str) and error_msg.strip() else None)
                if error_code in {"timeout", "connection_error", "exception"}
                else None
            )

            gtf = _fix_unclosed_boxed(gt)
            ansf = _fix_unclosed_boxed(answer)
            solver_status, solver_failure_code, solver_failure_stage = _solver_status_from_outcome(
                error_code=error_code,
                answer_pred=ansf,
                raw_text=text,
            )
            solver_failure_detail = error_detail
            usage = (response or {}).get("usage") if isinstance(response, dict) else {}
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            if not isinstance(completion_tokens, int):
                completion_tokens = _estimate_tokens(ansf)
            kq_tokens = _estimate_tokens(f"{known_text}\n{query}")
            token_ratio = None

            chain_value = record.get("chain") or (f"k{step_int},q{step_int},a{step_int}" if step_int is not None else None)

            # 解析反馈字段
            fb_sug = self._parse_feedback_and_suggestion(text) if text else {
                "QuestionWellPosed": False,
                "CorrectnessFeedback": None,
                "DifficultyFeedback": None,
                "KeyConclusion": None,
                "SolverReasoning": None,
                "Feedback": None,
                "HarderSuggestion": None,
            }

            question_well_posed = fb_sug.get("QuestionWellPosed", False)
            correctness_feedback = fb_sug.get("CorrectnessFeedback")
            difficulty_feedback = fb_sug.get("DifficultyFeedback")
            key_conclusion = fb_sug.get("KeyConclusion")
            solver_reasoning = fb_sug.get("SolverReasoning")
            old_feedback = fb_sug.get("Feedback")
            old_harder_suggestion = fb_sug.get("HarderSuggestion")

            # 字段传播逻辑（与单线程版本一致）
            correctness_feedback_used = (
                correctness_feedback if isinstance(correctness_feedback, str) and correctness_feedback.strip() else None
            ) or (old_feedback if isinstance(old_feedback, str) and old_feedback.strip() else None)
            difficulty_feedback_used = None
            if question_well_posed:
                if isinstance(difficulty_feedback, str) and difficulty_feedback.strip():
                    difficulty_feedback_used = difficulty_feedback
                elif isinstance(old_harder_suggestion, str) and old_harder_suggestion.strip():
                    difficulty_feedback_used = old_harder_suggestion
            key_conclusion_used = key_conclusion if isinstance(key_conclusion, str) and key_conclusion.strip() else None
            solver_reasoning_used = solver_reasoning if isinstance(solver_reasoning, str) and solver_reasoning.strip() else None
            if isinstance(solver_reasoning_used, str) and len(solver_reasoning_used) > 4000:
                solver_reasoning_used = solver_reasoning_used[:4000] + " ...[truncated]"

            # 标记当前 (paper_id, step) 已求解，避免重复
            if key:
                solved_keys.add(key)

            solved_item = {
                "known": known_raw,
                "question": query,
                "answer": gtf,
                "solve": ansf,
                "answer_ref": gtf,
                "answer_pred": ansf,
                "solver_status": solver_status,
                "correct": None,
                "token_ratio": token_ratio,
                "metrics": {
                    "kq_tokens": kq_tokens,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": (usage.get("prompt_tokens") if isinstance(usage, dict) else None),
                },
                "paper_id": pid_val,
                "step": step_int,
                "chain": chain_value,
                # 新字段：正确性与难度反馈
                "question_well_posed": question_well_posed,
                "correctness_feedback": correctness_feedback_used,
                "difficulty_feedback": difficulty_feedback_used,
                "key_conclusion": key_conclusion_used,
                "solver_reasoning": solver_reasoning_used,
                "model": sess.model_name,
                "service_id": sess.service_id,
                "finish_reason": finish_reason,
                "gateway_finish_reason": gateway_finish_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if error_code:
                solved_item["error"] = error_code
            if error_detail:
                solved_item["error_detail"] = error_detail
            if solver_failure_code:
                solved_item["solver_failure_code"] = solver_failure_code
            if solver_failure_stage:
                solved_item["solver_failure_stage"] = solver_failure_stage
            if solver_failure_detail:
                solved_item["solver_failure_detail"] = solver_failure_detail

            raw_item = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": {"known": known_raw, "question": query},
                "prompt": prompt,
                "response": response,
                "text": text,
                "usage": usage,
                "gen_params": {"model": sess.model_name, "service_id": sess.service_id},
                "finish_reason": finish_reason,
                "gateway_finish_reason": gateway_finish_reason,
                **({"error": error_code} if error_code else {}),
                **({"error_detail": error_detail} if error_detail else {}),
            }
            return solved_item, raw_item

        buffer_solved: List[Dict[str, Any]] = []
        buffer_raw: List[Dict[str, Any]] = []
        BATCH = max(1, int(concurrency))
        for s_item, r_item in self._concurrent_map(_iter_kqa_records(kqa_path), worker, concurrency):
            if s_item:
                buffer_solved.append(s_item)
            if r_item:
                buffer_raw.append(r_item)
            if len(buffer_solved) >= BATCH:
                write_jsonl(buffer_solved, output_path, schema=None, append=append_flag)
                buffer_solved.clear()
                append_flag = True
            if len(buffer_raw) >= BATCH:
                write_jsonl(buffer_raw, raw_path, schema=None, append=append_flag)
                buffer_raw.clear()
                append_flag = True

        # 收尾写入
        if buffer_solved:
            write_jsonl(buffer_solved, output_path, schema=None, append=append_flag)
            append_flag = True
        if buffer_raw:
            write_jsonl(buffer_raw, raw_path, schema=None, append=append_flag)

        logger.info("KQA 求解完成（并发=%s），输出: %s", concurrency, output_path)
        return output_dir / output_path.name

    def _render_prompt(self, known_text: str, query: str) -> str:
        lang = (getattr(self.config, "lang", None) or "zh").lower().strip() or "zh"
        payload = {
            "known": str(known_text or ""),
            "question": query,
            "query": query,
            "answer_schema": load_prompt_fragment("answer_schema", lang=lang),
            "question_types": load_prompt_fragment("question_types", lang=lang),
        }
        extra_vars = getattr(self.config, "prompt_vars", None)
        if isinstance(extra_vars, dict):
            payload.update(extra_vars)
        return self.prompt_template.safe_substitute(payload)

    def _parse_answer(self, text: str) -> Optional[str]:
        candidate = text.strip()

        # Prefer a fenced JSON block if present (can appear anywhere in the output).
        fenced = extract_preferred_fenced_block(candidate, preferred_langs=("json",))
        if fenced:
            candidate = fenced

        # 优先 JSON 解析
        def _try_json_parse(text_block: str) -> Optional[str]:
            try:
                obj = json.loads(text_block)
                if isinstance(obj, dict) and "Answer" in obj:
                    return str(obj["Answer"]).strip()
            except Exception:
                return None
            return None

        ans = _try_json_parse(candidate)
        if ans:
            return ans

        # 扫描所有平衡的 {...} 片段，寻找包含 Answer 的 JSON
        blocks: list[str] = []
        text_iter = candidate
        while True:
            blk = _extract_first_brace_block(text_iter)
            if not blk:
                break
            blocks.append(blk)
            # 从当前块结束位置之后继续扫描
            try:
                end_pos = text_iter.index(blk) + len(blk)
                text_iter = text_iter[end_pos:]
            except Exception:
                break
        for blk in blocks:
            ans = _try_json_parse(blk)
            if ans:
                return ans

        # 若 JSON 解析均失败，退而求其次：直接用正则从文本中抓取 "Answer": "..."
        try:
            import re as _re

            m_ans = _re.search(r'"Answer"\s*:\s*"(.+?)"', candidate, flags=_re.DOTALL)
            if m_ans:
                raw_ans = m_ans.group(1).strip()
                if raw_ans:
                    return raw_ans
        except Exception:
            pass

        # 兜底：抓取 \boxed{...}
        marker = "\\boxed{"
        if marker in candidate:
            start = candidate.find(marker) + len(marker)
            end = candidate.find("}", start)
            if end != -1:
                return f"\\boxed{{{candidate[start:end]}}}"
        return None

    def _parse_feedback_and_suggestion(self, text: str) -> Dict[str, Any]:
        """从模型输出中解析题目反馈与改难建议。

        支持新字段（QuestionWellPosed, CorrectnessFeedback, DifficultyFeedback）和旧字段（Feedback, HarderSuggestion）。
        优先使用新字段，若缺失则回退到旧字段以保持向后兼容。
        """
        result: Dict[str, Any] = {
            # 新字段
            "QuestionWellPosed": False,
            "CorrectnessFeedback": None,
            "DifficultyFeedback": None,
            "KeyConclusion": None,
            "SolverReasoning": None,
            # 向后兼容字段
            "Feedback": None,
            "HarderSuggestion": None,
        }
        candidate = text.strip()

        # 去除 ```json / ``` 包裹
        if candidate.startswith("```"):
            fence = "```json" if candidate.startswith("```json") else "```"
            end = candidate.find("```", len(fence))
            if end != -1:
                candidate = candidate[len(fence) : end].strip()

        # 尝试直接 JSON 解析
        obj: Any
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            # 尝试提取首个 JSON 对象
            block = _extract_first_brace_block(candidate)
            if not block:
                obj = None
            try:
                if block:
                    obj = json.loads(block)
            except json.JSONDecodeError:
                obj = None

        if isinstance(obj, dict):
            # 优先解析新字段
            qwp = obj.get(FIELD_QUESTION_WELL_POSED)
            if isinstance(qwp, bool):
                result["QuestionWellPosed"] = qwp
            elif isinstance(qwp, str):
                # 容错：字符串 "true"/"false" 也接受
                result["QuestionWellPosed"] = qwp.strip().lower() in ("true", "1", "yes")

            cf = obj.get(FIELD_CORRECTNESS_FEEDBACK)
            if isinstance(cf, str) and cf.strip():
                result["CorrectnessFeedback"] = cf.strip()

            df = obj.get(FIELD_DIFFICULTY_FEEDBACK)
            if isinstance(df, str) and df.strip():
                result["DifficultyFeedback"] = df.strip()
            elif df is None:
                # 明确为 null 时保持 None
                result["DifficultyFeedback"] = None

            kc = obj.get(FIELD_KEY_CONCLUSION) or obj.get("HarderSuggestion_KeyConclusion") or obj.get("KeyPoint")
            if isinstance(kc, str) and kc.strip():
                result["KeyConclusion"] = kc.strip()

            sr = obj.get(FIELD_SOLVER_REASONING)
            if isinstance(sr, str) and sr.strip():
                result["SolverReasoning"] = sr.strip()
            elif sr is None:
                result["SolverReasoning"] = None

            # 向后兼容：解析旧字段（仅在新字段缺失时使用）
            if result["CorrectnessFeedback"] is None:
                fb = obj.get(FIELD_FEEDBACK)
                if isinstance(fb, str) and fb.strip():
                    result["Feedback"] = fb.strip()
                    # 映射到新字段（作为回退）
                    result["CorrectnessFeedback"] = fb.strip()

            if result["DifficultyFeedback"] is None:
                sug = obj.get(FIELD_HARDER_SUGGESTION)
                if isinstance(sug, str) and sug.strip():
                    result["HarderSuggestion"] = sug.strip()
                    # 映射到新字段（作为回退，但仅在 well-posed 时）
                    if result["QuestionWellPosed"]:
                        result["DifficultyFeedback"] = sug.strip()

            return result

        # JSON 解析失败时，退而求其次：用正则从文本中抓取字段
        try:
            import re as _re

            # 新字段
            m_qwp = _re.search(rf'"{FIELD_QUESTION_WELL_POSED}"\s*:\s*(true|false)', candidate, flags=_re.IGNORECASE)
            if m_qwp:
                result["QuestionWellPosed"] = m_qwp.group(1).lower() == "true"

            m_cf = _re.search(rf'"{FIELD_CORRECTNESS_FEEDBACK}"\s*:\s*"(.+?)"', candidate, flags=_re.DOTALL)
            if m_cf and m_cf.group(1).strip():
                result["CorrectnessFeedback"] = m_cf.group(1).strip()

            m_df = _re.search(rf'"{FIELD_DIFFICULTY_FEEDBACK}"\s*:\s*"(.+?)"', candidate, flags=_re.DOTALL)
            if m_df and m_df.group(1).strip():
                result["DifficultyFeedback"] = m_df.group(1).strip()

            m_kc = _re.search(r'"(?:HarderSuggestion_KeyConclusion|KeyConclusion|KeyPoint)"\s*:\s*"(.+?)"', candidate, flags=_re.DOTALL)
            if m_kc and m_kc.group(1).strip():
                result["KeyConclusion"] = m_kc.group(1).strip()

            m_sr = _re.search(rf'"{FIELD_SOLVER_REASONING}"\s*:\s*"(.+?)"', candidate, flags=_re.DOTALL)
            if m_sr and m_sr.group(1).strip():
                result["SolverReasoning"] = m_sr.group(1).strip()

            # 向后兼容：旧字段
            if result["CorrectnessFeedback"] is None:
                m_fb = _re.search(r'"Feedback"\s*:\s*"(.+?)"', candidate, flags=_re.DOTALL)
                if m_fb and m_fb.group(1).strip():
                    result["Feedback"] = m_fb.group(1).strip()
                    result["CorrectnessFeedback"] = m_fb.group(1).strip()

            if result["DifficultyFeedback"] is None:
                m_sug = _re.search(r'"HarderSuggestion"\s*:\s*"(.+?)"', candidate, flags=_re.DOTALL)
                if m_sug and m_sug.group(1).strip():
                    result["HarderSuggestion"] = m_sug.group(1).strip()
                    if result["QuestionWellPosed"]:
                        result["DifficultyFeedback"] = m_sug.group(1).strip()
        except Exception:
            pass
        return result


def _fix_unclosed_boxed(s: str) -> str:
    """若字符串包含 LaTeX \\boxed{...} 但末尾缺失对应的右花括号，则补齐一个。"""
    if not s or "\\boxed{" not in s:
        return s

    last = s.rfind("\\boxed{")
    # 从 last 起向后扫描花括号配平
    depth = 0
    i = last
    # 进入时遇到 \boxed{ 视为先消耗一个 '{'
    i = s.find("{", last)
    if i == -1:
        return s
    depth = 1
    i += 1
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # 找到了成对的 '}'
                return s
        i += 1
    # 未能配平，补一个右花括号
    return s + "}"


def _estimate_tokens(text: str) -> int:
    """估算文本 token 数：优先用 tiktoken，缺失则用简单启发式。"""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        # 回退：按单词/符号粗算
        tokens = re.findall(r"\w+|[^\w\s]", text or "", re.UNICODE)
        return len(tokens)
