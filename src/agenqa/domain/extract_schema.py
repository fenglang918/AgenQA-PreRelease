"""Extract 角色输出字段的单一来源定义。"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, List

if TYPE_CHECKING:
    from agenqa.skills.extracting import ExtractOutput

FIELD_EXAM_POINTS = "exam_points"
FIELD_CHAIN_POTENTIAL = "chain_potential"

EXTRACT_OUTPUT_FIELDS = [
    FIELD_EXAM_POINTS,
    FIELD_CHAIN_POTENTIAL,
]


def extract_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    if use_en:
        return (
            f"- {FIELD_EXAM_POINTS}: string[]  # 2–5 exam points suitable for question generation\n"
            f"- {FIELD_CHAIN_POTENTIAL}: string  # relationships among points and chain potential"
        )
    return (
        f"- {FIELD_EXAM_POINTS}: string[]  # 2-5 个可出题的考点\n"
        f"- {FIELD_CHAIN_POTENTIAL}: string  # 说明这些考点之间的逻辑关系与链式出题潜力"
    )


def extract_output_to_dict(out: "ExtractOutput") -> Dict[str, Any]:
    return {
        FIELD_EXAM_POINTS: out.exam_points,
        FIELD_CHAIN_POTENTIAL: out.chain_potential,
    }
