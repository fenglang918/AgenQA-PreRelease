"""Format 角色输出字段的单一来源定义。"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, List

if TYPE_CHECKING:
    from agenqa.skills.formatting import FormatOutput

FIELD_STEP = "Step"
FIELD_QUESTION = "Question"
FIELD_WORLD_CONTRACT = "WorldContract"
FIELD_SOLUTION = "Solution"
FIELD_ANSWER = "Answer"
FIELD_VALIDATION_PASSED = "validation_passed"
FIELD_VALIDATION_ERRORS = "validation_errors"

FORMAT_OUTPUT_FIELDS = [
    FIELD_STEP,
    FIELD_QUESTION,
    FIELD_WORLD_CONTRACT,
    FIELD_SOLUTION,
    FIELD_ANSWER,
    FIELD_VALIDATION_PASSED,
    FIELD_VALIDATION_ERRORS,
]


def format_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    _ = use_en
    return "\n".join(
        [
            f"- {FIELD_STEP}: integer",
            f"- {FIELD_QUESTION}: string",
            f"- {FIELD_WORLD_CONTRACT}: string",
            f"- {FIELD_SOLUTION}: string",
            f"- {FIELD_ANSWER}: string",
            f"- {FIELD_VALIDATION_PASSED}: boolean",
            f"- {FIELD_VALIDATION_ERRORS}: string[]",
        ]
    )


def format_output_to_dict(out: "FormatOutput") -> Dict[str, Any]:
    return {
        FIELD_STEP: out.step,
        FIELD_QUESTION: out.question,
        FIELD_WORLD_CONTRACT: out.world_contract,
        FIELD_SOLUTION: out.solution,
        FIELD_ANSWER: out.answer,
        FIELD_VALIDATION_PASSED: out.validation_passed,
        FIELD_VALIDATION_ERRORS: out.validation_errors,
    }
