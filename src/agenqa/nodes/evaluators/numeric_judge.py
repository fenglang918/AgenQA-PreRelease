"""Numeric equivalence judge utilities.

Used by consensus aggregation for Numeric question types when answer_judge=llm.

Contract (2026-02-01, see docs/design/active/semantic_numeric_oracle.md):
- Strict binary output: equivalent must be boolean true/false (no uncertain).
- Any judge failure (parse/timeout/refusal/etc.) must raise (no silent degrade).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from infra.text.json_sanitize import sanitize_json_text


def build_numeric_equivalence_prompt_zh(
    *,
    known: str,
    question: str,
    answer_ref: str,
    answer_pred: str,
) -> str:
    """Build a judge prompt for numeric answer equivalence (ZH)."""
    return (
        "你的任务是判断两个【数值答案】在给定题目语境下是否等价。\n"
        "- 若题干明确给出容差口径（例如 abs_tol / rel_tol，或“保留 N 位小数/有效数字/误差不超过…”），请严格按题干口径判定。\n"
        "- 若题干未给出容差口径：不要自行引入固定默认阈值；请基于题干与上下文判断二者是否可视为同一答案。\n"
        "- 允许常见格式噪声：不同小数位、科学计数法、空格、以及单位写法差异（如 \"us\"/\"μs\"/\"microsecond\"）等。\n"
        "- 若两者单位不同但**同量纲且可换算**，请先换算到同一单位后再判等（例如 3.91 us 等价于 3910 ns）。\n"
        "- 若单位不同且无法确认可换算关系，默认视为不等价，并在 reason 中说明“单位/量纲不一致或不可换算”。\n"
        "- 必须输出严格二分类结论：等价=true 或 不等价=false（不允许 uncertain）。\n\n"
        "已知条件 Known：\n"
        f"{known}\n\n"
        "题目 Question：\n"
        f"{question}\n\n"
        "参考答案 ReferenceAnswer：\n"
        f"{answer_ref}\n\n"
        "待比较答案 PredictedAnswer：\n"
        f"{answer_pred}\n\n"
        "请只输出一个 JSON 对象（不要输出任何额外文字），字段要求：\n"
        "- equivalent: boolean（true/false）；\n"
        "- reason: 给出一句简要中文说明（需可审计）。\n"
    )


def build_numeric_equivalence_prompt_en(
    *,
    known: str,
    question: str,
    answer_ref: str,
    answer_pred: str,
) -> str:
    """Build a judge prompt for numeric answer equivalence (EN)."""
    return (
        "Your task is to decide whether two numeric answers are equivalent under the problem context.\n"
        "- If tolerance rules are explicitly stated in the question (e.g., abs_tol / rel_tol, "
        "\"N decimal places/significant figures\", or error bounds), apply them strictly.\n"
        "- If tolerance rules are not explicitly stated, do not invent a fixed default threshold; "
        "judge from the question/context whether they should be treated as the same value.\n"
        "- Allow common formatting noise: decimal places, scientific notation, whitespace, and unit wording "
        "(e.g., \"us\"/\"μs\"/\"microsecond\").\n"
        "- If units differ but are convertible under the same dimension, convert before judging "
        "(e.g., 3.91 us equals 3910 ns).\n"
        "- If units differ and convertibility cannot be established, treat as not equivalent and explain why.\n"
        "- Output must be strict binary: equivalent=true or equivalent=false (no uncertain).\n\n"
        "Known:\n"
        f"{known}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "ReferenceAnswer:\n"
        f"{answer_ref}\n\n"
        "PredictedAnswer:\n"
        f"{answer_pred}\n\n"
        "Output one JSON object only (no extra text), with fields:\n"
        "- equivalent: boolean (true/false);\n"
        "- reason: one short English explanation.\n"
    )


def _strip_fences(text: str) -> str:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        fence = "```json" if candidate.startswith("```json") else "```"
        end = candidate.rfind("```")
        if end != -1:
            candidate = candidate[len(fence) : end].strip()
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


def parse_numeric_judge_output(text: str) -> Tuple[bool, Optional[str]]:
    """Parse judge output, return (equivalent: True/False, reason).

    Raises:
      ValueError: when output is missing/invalid or equivalent is not a boolean.
    """
    candidate = _strip_fences(text)
    if not candidate:
        raise ValueError("numeric_judge: empty output")

    obj: Dict[str, Any] | None = None
    try:
        loaded = json.loads(candidate)
        obj = loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        try:
            loaded = json.loads(sanitize_json_text(candidate))
            obj = loaded if isinstance(loaded, dict) else None
        except Exception:
            obj = None

    if obj is None:
        block = _extract_first_brace_block(candidate)
        if not block:
            raise ValueError("numeric_judge: cannot find JSON object")
        try:
            loaded = json.loads(block)
            obj = loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            loaded = json.loads(sanitize_json_text(block))
            obj = loaded if isinstance(loaded, dict) else None

    if not obj:
        raise ValueError("numeric_judge: invalid JSON object")

    if "equivalent" not in obj:
        raise ValueError("numeric_judge: missing field 'equivalent'")
    eq = obj.get("equivalent")
    if not isinstance(eq, bool):
        raise ValueError("numeric_judge: field 'equivalent' must be boolean")

    reason = obj.get("reason") or obj.get("explanation") or None
    if reason is not None and not isinstance(reason, str):
        try:
            reason = json.dumps(reason, ensure_ascii=False)
        except Exception:
            reason = str(reason)
    return eq, (reason.strip() if isinstance(reason, str) and reason.strip() else None)


def run_numeric_equivalence_judge(
    generator: Dict[str, object],
    *,
    known: str,
    question: str,
    answer_ref: str,
    answer_pred: str,
    lang: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Call the LLM judge and return (equivalent, reason)."""
    # Local import to avoid a hard dependency loop.
    from infra.llm.inference import resolve_inference

    resolved = resolve_inference(generator)  # type: ignore[arg-type]
    session = resolved.session
    chat_args = dict(resolved.chat_args)
    lang_norm = str(lang or "").strip().lower()
    use_en = lang_norm in {"en", "english"}
    if use_en:
        prompt = build_numeric_equivalence_prompt_en(
            known=str(known),
            question=str(question),
            answer_ref=str(answer_ref),
            answer_pred=str(answer_pred),
        )
    else:
        prompt = build_numeric_equivalence_prompt_zh(
            known=str(known),
            question=str(question),
            answer_ref=str(answer_ref),
            answer_pred=str(answer_pred),
        )
    messages = [{"role": "user", "content": prompt}]
    response = session.chat(messages, **chat_args)
    text = session.extract_text(response, default="")
    return parse_numeric_judge_output(text)


__all__ = [
    "build_numeric_equivalence_prompt_en",
    "build_numeric_equivalence_prompt_zh",
    "parse_numeric_judge_output",
    "run_numeric_equivalence_judge",
]
