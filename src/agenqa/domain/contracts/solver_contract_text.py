"""Helpers for solver-visible contract text.

This module keeps a strict separation between:
- the core question body
- the solver-visible contract block derived from structured world_contract plus
  the public L4 answer-output slice

The final solver input may concatenate the two, but storage/history should keep
them as separate fields to avoid baking contract governance into question text.
"""

from __future__ import annotations

from typing import Any, Dict, List
import json
import re

from .world_contract import normalize_world_contract
from .answer_contract_bank import extract_answer_output_spec_context

_WC_HEADING_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:Semantics\s*/\s*World Contract|World Contract|语义\s*/\s*World Contract|语义契约|答案要求|Answer Requirements|Answer Contract)\s*(?:\*\*)?\s*$",
    re.IGNORECASE,
)


def _is_en(lang: str | None) -> bool:
    return str(lang or "").strip().lower() in {"en", "english"}


def _compact_json(val: Any) -> str:
    if isinstance(val, str):
        return val.strip()
    try:
        return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(val).strip()


def strip_embedded_contract_blocks(question_text: str) -> str:
    """Best-effort removal of embedded contract sections from question text.

    We only strip obvious tail blocks introduced as separate headings, keeping the
    question body intact. This is a defensive cleanup for legacy prompts / runs.
    """
    text = str(question_text or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    cut_idx: int | None = None
    for i, line in enumerate(lines):
        if _WC_HEADING_RE.match(str(line or "").strip()):
            cut_idx = i
            break
    if cut_idx is None:
        return text
    kept = "\n".join(lines[:cut_idx]).rstrip()
    return kept


def _render_point(level: str, point: Dict[str, Any]) -> str:
    axis = str(point.get("axis") or "").strip()
    choice = _compact_json(point.get("choice"))
    note = str(point.get("note") or "").strip()
    base = f"{level}.{axis}: {choice}" if axis else f"{level}: {choice}"
    if note:
        base = f"{base} ({note})"
    return base


def render_world_contract_text(
    world_contract: Any,
    *,
    answer_output_specs: List[Dict[str, Any]] | None = None,
    lang: str | None = None,
) -> str:
    wc = normalize_world_contract(world_contract)
    sections = wc.get("sections")
    if not isinstance(sections, list):
        sections = []

    use_en = _is_en(lang)
    lines: List[str] = ["**World Contract**" if use_en else "**World Contract**"]

    has_any = False
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        level = str(sec.get("level") or "").strip().upper()
        points = sec.get("points")
        if not isinstance(points, list):
            continue
        rendered = []
        for point in points:
            if not isinstance(point, dict):
                continue
            axis = str(point.get("axis") or "").strip()
            if level == "L4" and axis.startswith("type2."):
                continue
            row = _render_point(level, point).strip()
            if row:
                rendered.append(row)
        if not rendered:
            continue
        has_any = True
        for row in rendered:
            lines.append(f"- {row}")

    if isinstance(answer_output_specs, list):
        specs = [x for x in answer_output_specs if isinstance(x, dict) and x]
    else:
        specs = []
    if specs:
        has_any = True
        for idx, spec in enumerate(specs, start=1):
            lines.append(f"- L4.answer_output_spec_{idx}: {_compact_json(spec)}")

    if not has_any:
        return ""
    return "\n".join(lines).strip()


def extract_solver_world_contract_text(
    memory: Dict[str, Any] | None,
    *,
    step: int,
    lang: str | None = None,
    explicit_world_contract_text: str | None = None,
) -> str:
    mem = dict(memory or {})
    ctx = extract_answer_output_spec_context(mem, step=int(step))
    raw_specs = ctx.get("answer_output_specs")
    specs = raw_specs if isinstance(raw_specs, list) else []
    explicit = str(explicit_world_contract_text or "").strip()
    if explicit:
        if specs and "L4.answer_output_spec_" not in explicit:
            lines = [explicit]
            for idx, spec in enumerate([x for x in specs if isinstance(x, dict)], start=1):
                lines.append(f"- L4.answer_output_spec_{idx}: {_compact_json(spec)}")
            return "\n".join(lines).strip()
        return explicit
    return render_world_contract_text(
        mem.get("world_contract"),
        answer_output_specs=[x for x in specs if isinstance(x, dict)],
        lang=lang,
    )


def compose_solver_question(question: str, world_contract_text: str | None) -> str:
    q = str(question or "").strip()
    wc = str(world_contract_text or "").strip()
    if q and wc:
        return f"{q}\n\n{wc}"
    return q or wc


__all__ = [
    "strip_embedded_contract_blocks",
    "render_world_contract_text",
    "extract_solver_world_contract_text",
    "compose_solver_question",
]
