"""Diagnose 角色输出字段的单一来源定义。"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, List

if TYPE_CHECKING:
    from agenqa.skills.diagnosing import DiagnoseOutput

FIELD_ISSUES = "issues"
FIELD_FIX_SUGGESTIONS = "fix_suggestions"
FIELD_DIAGNOSIS = "diagnosis"

DIAGNOSE_OUTPUT_FIELDS = [FIELD_ISSUES, FIELD_FIX_SUGGESTIONS, FIELD_DIAGNOSIS]


def diagnose_output_schema_text() -> str:
    return (
        f"- {FIELD_ISSUES}: string[]\n"
        f"- {FIELD_FIX_SUGGESTIONS}: string[]\n"
        f"- {FIELD_DIAGNOSIS}: string"
    )


def diagnose_output_to_dict(out: "DiagnoseOutput") -> Dict[str, Any]:
    return {
        FIELD_ISSUES: out.issues,
        FIELD_FIX_SUGGESTIONS: out.fix_suggestions,
        FIELD_DIAGNOSIS: out.diagnosis,
    }
