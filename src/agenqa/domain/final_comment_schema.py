"""Final Commenter 输出字段的单一来源定义。"""

from __future__ import annotations


FIELD_WELL_POSED = "well_posed"
FIELD_QUESTION_LEAKS_PREVIOUS_CONCLUSIONS = "question_leaks_previous_conclusions"
FIELD_INTERMEDIATE_STEPS_NECESSARY = "intermediate_steps_necessary"
FIELD_DIFFICULTY = "difficulty"
FIELD_EVIDENCE = "evidence"
FIELD_SUGGESTIONS = "suggestions"

FINAL_COMMENT_OUTPUT_FIELDS = [
    FIELD_WELL_POSED,
    FIELD_QUESTION_LEAKS_PREVIOUS_CONCLUSIONS,
    FIELD_INTERMEDIATE_STEPS_NECESSARY,
    FIELD_DIFFICULTY,
    FIELD_EVIDENCE,
    FIELD_SUGGESTIONS,
]


def final_comment_output_schema_text() -> str:
    return (
        f"- {FIELD_WELL_POSED}: boolean | null\n"
        f"- {FIELD_QUESTION_LEAKS_PREVIOUS_CONCLUSIONS}: boolean | null\n"
        f"- {FIELD_INTERMEDIATE_STEPS_NECESSARY}: boolean | null\n"
        f"- {FIELD_DIFFICULTY}: string\n"
        f"- {FIELD_EVIDENCE}: string[]\n"
        f"- {FIELD_SUGGESTIONS}: string[]"
    )
