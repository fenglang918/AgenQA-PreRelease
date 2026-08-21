"""Extend/Compress 输出字段单一来源定义。"""

from __future__ import annotations

from typing import Any, Dict

FIELD_STEP = "Step"
FIELD_KNOWN = "Known"
FIELD_QUESTION = "Question"
FIELD_ANSWER = "Answer"
FIELD_SOLUTION = "Solution"
FIELD_NEW_BACKGROUND = "NewBackground"
FIELD_DERIVED_FACTS = "DerivedFacts"

EXTEND_OUTPUT_FIELDS = [
    FIELD_STEP,
    FIELD_QUESTION,
    FIELD_SOLUTION,
    FIELD_ANSWER,
    FIELD_NEW_BACKGROUND,
    FIELD_DERIVED_FACTS,
]


def extend_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    new_bg = "string or string[] (optional)" if use_en else "string 或 string[]（可选）"
    derived = "string[] (optional)" if use_en else "string[]（可选）"
    return "\n".join(
        [
            f"- {FIELD_STEP}: integer (= i)",
            f"- {FIELD_QUESTION}: string",
            f"- {FIELD_SOLUTION}: string",
            f"- {FIELD_ANSWER}: string",
            f"- {FIELD_NEW_BACKGROUND}: {new_bg}",
            f"- {FIELD_DERIVED_FACTS}: {derived}",
        ]
    )
