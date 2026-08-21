"""Shared expression equivalence judge utilities.

This module centralizes how we:
- load the expression_judge generator config; and
- build/parse/run a lightweight LLM judge for expression equivalence.

It is used by:
- agenqa.nodes.evaluators.solve (judge answer_ref vs answer_pred)
- agenqa.nodes.evaluators.consensus (judge equivalence across strong solvers)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from infra.llm.inference import resolve_inference
from infra.llm.service_loader import load_llm_service_full_config

DEFAULT_SERVICE_CONFIG = Path(
    os.getenv(
        "LLM_SERVICES_JSON",
        "config/services.json",
    )
)


def load_expression_judge_generator(agent_conf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load expression_judge generator config from agent_conf.

    Priority:
    1) agent_conf['expression_judge']['generator'] (full generator dict)
    2) agent_conf['expression_judge']['service_id'] (+ optional service_config / service_model)
    """
    ej_block = agent_conf.get("expression_judge") or {}
    if not isinstance(ej_block, dict):
        return None
    generator = ej_block.get("generator")
    if isinstance(generator, dict) and generator:
        return generator
    service_id = ej_block.get("service_id")
    if not isinstance(service_id, str) or not service_id.strip():
        return None
    service_cfg_path = Path(ej_block.get("service_config") or DEFAULT_SERVICE_CONFIG)
    return load_llm_service_full_config(service_cfg_path, service_id, explicit_model=ej_block.get("service_model"))


def build_expression_equivalence_prompt_zh(
    *,
    known: str,
    question: str,
    answer_ref: str,
    answer_pred: str,
    answer_contract: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a judge prompt for expression equivalence (ZH)."""
    contract_block = ""
    if isinstance(answer_contract, dict) and answer_contract:
        try:
            contract_json = json.dumps(answer_contract, ensure_ascii=False, indent=2)
        except Exception:
            contract_json = str(answer_contract)
        contract_block = (
            "附加判对上下文 AnswerContractContext（内部结构化约束；用于判断题目语境下的答案等价性，不是求解输入）：\n"
            f"{contract_json}\n\n"
        )
    return (
        "你的任务是判断两个答案表达式在给定条件下是否数学等价。\n"
        "请只根据解析形式判断等价性：允许符号重命名和等价的代数变形；\n"
        "不要因为写法不同（如 (Re/Re0)^α 与 exp(α ln(Re/Re0))）而误判为不等价。\n\n"
        "已知条件 Known：\n"
        f"{known}\n\n"
        f"{contract_block}"
        "题目 Question：\n"
        f"{question}\n\n"
        "参考答案表达式 ReferenceAnswer：\n"
        f"{answer_ref}\n\n"
        "待比较答案表达式 PredictedAnswer：\n"
        f"{answer_pred}\n\n"
        "请只输出一个 JSON 对象（不要输出任何额外文字），字段要求：\n"
        "- 字段 equivalent: 取值必须为 \"yes\" / \"no\" / \"uncertain\" 三者之一；\n"
        "- 字段 reason: 给出一句简要中文说明。\n"
        "其中：若两者可以确定在题目语境下数学等价，则 equivalent= \"yes\"；"
        "若可以确定不等价，则 equivalent= \"no\"；"
        "若无法判断或需要具体数值才能比较，则 equivalent= \"uncertain\"。\n"
    )


def build_expression_equivalence_prompt_en(
    *,
    known: str,
    question: str,
    answer_ref: str,
    answer_pred: str,
    answer_contract: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a judge prompt for expression equivalence (EN)."""
    contract_block = ""
    if isinstance(answer_contract, dict) and answer_contract:
        try:
            contract_json = json.dumps(answer_contract, ensure_ascii=False, indent=2)
        except Exception:
            contract_json = str(answer_contract)
        contract_block = (
            "Additional AnswerContractContext (internal structured constraints for equivalence under task context; "
            "not solver input):\n"
            f"{contract_json}\n\n"
        )
    return (
        "Your task is to determine whether two answer expressions are mathematically equivalent under the given context.\n"
        "Judge by semantic/formal equivalence: allow symbol renaming and algebraic transformations.\n"
        "Do not mark non-equivalent only because of different surface forms "
        "(e.g., (Re/Re0)^alpha vs exp(alpha ln(Re/Re0))).\n\n"
        "Known:\n"
        f"{known}\n\n"
        f"{contract_block}"
        "Question:\n"
        f"{question}\n\n"
        "ReferenceAnswer:\n"
        f"{answer_ref}\n\n"
        "PredictedAnswer:\n"
        f"{answer_pred}\n\n"
        "Output one JSON object only (no extra text), with fields:\n"
        "- equivalent: one of \"yes\" / \"no\" / \"uncertain\";\n"
        "- reason: one short English explanation.\n"
        "Use: equivalent=\"yes\" if confidently equivalent in context; "
        "\"no\" if confidently not equivalent; "
        "\"uncertain\" if not decidable from given context.\n"
    )


def parse_expression_judge_output(text: str) -> Tuple[Optional[bool], Optional[str]]:
    """Parse judge output, return (equivalent: True/False/None, reason)."""
    if not text:
        return None, None
    candidate = text.strip()

    # Strip ```json / ``` fences if present
    if candidate.startswith("```"):
        fence = "```json" if candidate.startswith("```json") else "```"
        end = candidate.rfind("```")
        if end != -1:
            candidate = candidate[len(fence) : end].strip()

    def _try_load_json(block: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    obj = _try_load_json(candidate)
    if obj is None:
        start = candidate.find("{")
        if start != -1:
            depth = 0
            end_idx = None
            for idx, ch in enumerate(candidate[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = idx + 1
                        break
            if end_idx is not None:
                obj = _try_load_json(candidate[start:end_idx])

    if not obj:
        return None, None

    eq_raw = str(obj.get("equivalent") or "").strip().lower()
    reason = obj.get("reason") or obj.get("explanation") or ""
    if isinstance(reason, (dict, list)):
        try:
            reason = json.dumps(reason, ensure_ascii=False)
        except Exception:
            reason = str(reason)
    if not isinstance(reason, str):
        reason = str(reason)

    if eq_raw == "yes":
        return True, reason or None
    if eq_raw == "no":
        return False, reason or None
    if eq_raw == "uncertain":
        return None, reason or None
    return None, reason or None


def run_expression_equivalence_judge(
    generator: Dict[str, Any],
    *,
    known: str,
    question: str,
    answer_ref: str,
    answer_pred: str,
    answer_contract: Optional[Dict[str, Any]] = None,
    lang: Optional[str] = None,
) -> Tuple[Optional[bool], Optional[str]]:
    """Call the LLM judge and return (equivalent, reason)."""
    resolved = resolve_inference(generator)
    session = resolved.session
    chat_args = dict(resolved.chat_args)
    lang_norm = str(lang or "").strip().lower()
    use_en = lang_norm in {"en", "english"}
    if use_en:
        prompt = build_expression_equivalence_prompt_en(
            known=str(known),
            question=str(question),
            answer_ref=str(answer_ref),
            answer_pred=str(answer_pred),
            answer_contract=answer_contract if isinstance(answer_contract, dict) else None,
        )
    else:
        prompt = build_expression_equivalence_prompt_zh(
            known=str(known),
            question=str(question),
            answer_ref=str(answer_ref),
            answer_pred=str(answer_pred),
            answer_contract=answer_contract if isinstance(answer_contract, dict) else None,
        )
    messages = [{"role": "user", "content": prompt}]
    response = session.chat(messages, **chat_args)
    text = session.extract_text(response, default="")
    return parse_expression_judge_output(text)


__all__ = [
    "DEFAULT_SERVICE_CONFIG",
    "build_expression_equivalence_prompt_en",
    "build_expression_equivalence_prompt_zh",
    "load_expression_judge_generator",
    "parse_expression_judge_output",
    "run_expression_equivalence_judge",
]
