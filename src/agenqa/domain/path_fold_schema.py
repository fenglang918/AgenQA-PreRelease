"""Path-Fold role output schema (code truth source).

This role generates folded path questions for a tail step N:
- a scaffolded version (with intermediate sub-goals hinted)
- a direct version (ask final target only)
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from agenqa.skills.path_fold import PathFoldOutput


FIELD_QUESTION_SCAFFOLDED = "question_scaffolded"
FIELD_QUESTION_DIRECT = "question_direct"
FIELD_FOLD_NOTES = "fold_notes"

PATH_FOLD_OUTPUT_FIELDS = [
    FIELD_QUESTION_SCAFFOLDED,
    FIELD_QUESTION_DIRECT,
    FIELD_FOLD_NOTES,
]


def path_fold_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    notes_comment = "short folding notes" if use_en else "折叠说明（简短）"
    return "\n".join(
        [
            f"- {FIELD_QUESTION_SCAFFOLDED}: string  # {'scaffolded (with intermediate hints)' if use_en else '带中间目标提示的版本'}",
            f"- {FIELD_QUESTION_DIRECT}: string  # {'direct (final ask only)' if use_en else '直接问最终目标的版本'}",
            f"- {FIELD_FOLD_NOTES}: string  # {notes_comment}",
        ]
    )


def path_fold_output_to_dict(out: "PathFoldOutput") -> Dict[str, Any]:
    return {
        FIELD_QUESTION_SCAFFOLDED: out.question_scaffolded,
        FIELD_QUESTION_DIRECT: out.question_direct,
        FIELD_FOLD_NOTES: out.fold_notes,
    }


__all__ = [
    "FIELD_QUESTION_SCAFFOLDED",
    "FIELD_QUESTION_DIRECT",
    "FIELD_FOLD_NOTES",
    "PATH_FOLD_OUTPUT_FIELDS",
    "path_fold_output_schema_text",
    "path_fold_output_to_dict",
]
