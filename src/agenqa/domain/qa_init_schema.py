"""QA-Init 输出字段单一来源定义。"""

from __future__ import annotations

from typing import Any, Dict, List

FIELD_STEP = "Step"
FIELD_SUBJECT = "Subject"
FIELD_KNOWN = "Known"
FIELD_QUESTION = "Question"
FIELD_SOLUTION = "Solution"
FIELD_ANSWER = "Answer"

KNOWN_FIELD_KNOWN0 = "known_0"
KNOWN_FIELD_HISTORY = "history"

QA_INIT_OUTPUT_FIELDS = [
    FIELD_STEP,
    FIELD_SUBJECT,
    FIELD_KNOWN,
    FIELD_QUESTION,
    FIELD_SOLUTION,
    FIELD_ANSWER,
]


def qa_init_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    known_desc = (
        f"object containing {KNOWN_FIELD_KNOWN0} and {KNOWN_FIELD_HISTORY}"
        if use_en
        else f"object 包含 {KNOWN_FIELD_KNOWN0} 与 {KNOWN_FIELD_HISTORY}"
    )
    return "\n".join(
        [
            f"- {FIELD_STEP}: integer (=0)",
            f"- {FIELD_SUBJECT}: string",
            f"- {FIELD_KNOWN}: {known_desc}",
            f"- {FIELD_QUESTION}: string",
            f"- {FIELD_SOLUTION}: string",
            f"- {FIELD_ANSWER}: string",
        ]
    )
