"""Solver 输出字段单一来源定义。"""

from __future__ import annotations

from typing import Any, Dict

FIELD_ANSWER = "Answer"
FIELD_SOLVER_REASONING = "SolverReasoning"
FIELD_FEEDBACK = "Feedback"
FIELD_HARDER_SUGGESTION = "HarderSuggestion"
FIELD_KEY_CONCLUSION = "KeyConclusion"
FIELD_QUESTION_WELL_POSED = "QuestionWellPosed"
FIELD_CORRECTNESS_FEEDBACK = "CorrectnessFeedback"
FIELD_DIFFICULTY_FEEDBACK = "DifficultyFeedback"

SOLVER_OUTPUT_FIELDS = [
    FIELD_ANSWER,
    FIELD_SOLVER_REASONING,
    FIELD_FEEDBACK,
    FIELD_HARDER_SUGGESTION,
    FIELD_KEY_CONCLUSION,
    FIELD_QUESTION_WELL_POSED,
    FIELD_CORRECTNESS_FEEDBACK,
    FIELD_DIFFICULTY_FEEDBACK,
]


def solver_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    if use_en:
        return "\n".join(
            [
                f'- {FIELD_ANSWER}: "... \\\\boxed{{}} ..."',
                f"- {FIELD_SOLVER_REASONING}: string (optional; brief derivation summary for Diagnose)",
                f"- {FIELD_QUESTION_WELL_POSED}: boolean (whether the question is well-posed: consistent, sufficient info, uniquely solvable)",
                f"- {FIELD_CORRECTNESS_FEEDBACK}: string (required; if QuestionWellPosed=false, state issues + fix suggestions; if true, briefly confirm)",
                f"- {FIELD_DIFFICULTY_FEEDBACK}: string | null (only when QuestionWellPosed=true; otherwise null)",
                f"- {FIELD_KEY_CONCLUSION}: string (optional; key intermediate conclusion)",
                f"- {FIELD_FEEDBACK}: string (optional; legacy; prefer CorrectnessFeedback)",
                f"- {FIELD_HARDER_SUGGESTION}: string (optional; legacy; prefer DifficultyFeedback)",
            ]
        )
    return "\n".join(
        [
            f'- {FIELD_ANSWER}: "... \\\\boxed{{}} ..."',
            f"- {FIELD_SOLVER_REASONING}: string（可选，推导过程摘要：用少量句子概括关键步骤与中间结论，供 Diagnose 参考）",
            f"- {FIELD_QUESTION_WELL_POSED}: boolean（题目是否 well-posed：条件自洽、信息充分、可唯一求解）",
            f"- {FIELD_CORRECTNESS_FEEDBACK}: string（题目正确性/well-posedness 反馈：若 QuestionWellPosed=false，必须指出问题点与修复建议；若为 true，可简要确认自洽性）",
            f"- {FIELD_DIFFICULTY_FEEDBACK}: string | null（难度反馈与加难建议：仅在 QuestionWellPosed=true 时填写，否则为 null）",
            f"- {FIELD_KEY_CONCLUSION}: string（可选，关键中间结论）",
            # 向后兼容字段（保留但标记为可选）
            f"- {FIELD_FEEDBACK}: string（可选，向后兼容，建议使用 CorrectnessFeedback）",
            f"- {FIELD_HARDER_SUGGESTION}: string（可选，向后兼容，建议使用 DifficultyFeedback）",
        ]
    )


def solver_output_base() -> Dict[str, Any]:
    """返回带空字段的基础 dict，用于初始化/合并。"""
    return {
        FIELD_ANSWER: "",
        FIELD_SOLVER_REASONING: None,
        FIELD_QUESTION_WELL_POSED: False,
        FIELD_CORRECTNESS_FEEDBACK: None,
        FIELD_DIFFICULTY_FEEDBACK: None,
        FIELD_KEY_CONCLUSION: None,
        # 向后兼容字段
        FIELD_FEEDBACK: None,
        FIELD_HARDER_SUGGESTION: None,
    }
