from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*([A-D])\s*\}")
_MCQ_OPTION_RE = re.compile(r"^\s*([A-D])\s*[\.\:：\)\]]\s*(.*)$")


def extract_boxed_letter(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    m = _BOXED_LETTER_RE.search(text)
    if not m:
        return None
    return m.group(1)


def parse_mcq_options(question: str) -> Dict[str, str]:
    """Parse options like 'A. ...' / 'B: ...' from a question string."""
    if not isinstance(question, str) or not question:
        return {}
    options: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw in question.splitlines():
        line = raw.rstrip("\n")
        m = _MCQ_OPTION_RE.match(line)
        if m:
            current = m.group(1)
            options.setdefault(current, []).append(m.group(2).strip())
            continue
        if current:
            options[current].append(line.strip())
    return {k: "\n".join(v).strip() for k, v in options.items() if k and v}


def is_mcq_question(question: str) -> bool:
    opts = parse_mcq_options(question)
    return len(opts) >= 4 and all(k in opts for k in ("A", "B", "C", "D"))

def validate_prev_step_reused_refs(
    *,
    prev_step: int,
    prev_question: str,
    prev_answer: str,
    reused_refs: object,
) -> Tuple[bool, List[str]]:
    """Protocol-level structural validation for reuse references.

    This intentionally does NOT attempt any string/LaTeX containment matching.
    It only checks that:
    - the next step declares it reuses the previous step (source_step == prev_step)
    - if previous is MCQ with boxed letter, the reuse ref carries the same mcq_choice.
    """
    try:
        prev_step_i = int(prev_step)
    except Exception:
        prev_step_i = -1
    if prev_step_i < 0:
        return True, []

    refs: List[Dict[str, object]] = []
    if isinstance(reused_refs, list):
        refs = [r for r in reused_refs if isinstance(r, dict)]
    elif isinstance(reused_refs, dict):
        refs = [reused_refs]

    if not refs:
        return (
            False,
            [f"复用结构错误：缺少 reused_refs（必须至少引用上一步 step={prev_step_i}）"],
        )

    hit = None
    for r in refs:
        try:
            if int(r.get("source_step")) == prev_step_i:
                hit = r
                break
        except Exception:
            continue

    if hit is None:
        return (
            False,
            [f"复用结构错误：reused_refs 未包含 source_step={prev_step_i} 的引用（必须引用上一步）"],
        )

    # If previous is MCQ with boxed letter, require mcq_choice matches.
    if is_mcq_question(prev_question):
        letter = extract_boxed_letter(prev_answer or "")
        if letter:
            raw_choice = hit.get("mcq_choice") or hit.get("choice") or hit.get("option")
            choice = str(raw_choice).strip().upper() if raw_choice is not None else ""
            if choice != letter:
                return (
                    False,
                    [
                        "复用结构错误：上一步为 MCQ，但 reused_refs 中 mcq_choice 未与上一步答案一致；"
                        f"期望 {letter}，实际 {choice or '<missing>'}"
                    ],
                )

    return True, []
