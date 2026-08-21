"""Draft 角色的输出 Schema 定义（Single Source of Truth）。

所有涉及 Draft 输出字段名的地方都应引用此模块的常量，
避免字段名散落在 prompt/解析/输出代码中造成不一致。

用法：
1. 解析 JSON 时使用 FIELD_* 常量作为 key
2. 输出时使用 draft_output_to_dict(draft_out) 转换
3. Prompt 中使用 draft_output_schema_text() 生成字段描述
"""

from __future__ import annotations

from typing import Any, Dict

# === Draft 输出字段名 ===
# 修改这里的值，会自动影响 prompt 描述、JSON 解析、节点输出
FIELD_DRAFT_QUESTION_EXPLICIT = "draft_question_explicit"  # 显式版本（写出复用关系）
FIELD_DRAFT_QUESTION = "draft_question"  # 隐藏版本（不透露前序结论）
FIELD_DRAFT_SOLUTION = "draft_solution"
FIELD_DRAFT_ANSWER = "draft_answer"
FIELD_DRAFT_BACKGROUND = "draft_background"
FIELD_REUSED_CONCLUSIONS = "reused_conclusions"
FIELD_REUSED_REFS = "reused_refs"
FIELD_GROUNDING_CHECK = "grounding_check"

# 兼容旧字段名（用于解析历史数据）
FIELD_GROUNDING_CHECK_LEGACY = "physics_check"
FIELD_DRAFT_BACKGROUND_LEGACY = "new_assumptions"

# 字段列表（用于生成 prompt 描述）
DRAFT_OUTPUT_FIELDS = [
    FIELD_DRAFT_QUESTION_EXPLICIT,
    FIELD_DRAFT_QUESTION,
    FIELD_DRAFT_SOLUTION,
    FIELD_DRAFT_ANSWER,
    FIELD_DRAFT_BACKGROUND,
    FIELD_REUSED_CONCLUSIONS,
    FIELD_REUSED_REFS,
    FIELD_GROUNDING_CHECK,
]


def draft_output_schema_text(lang: str | None = None) -> str:
    """Generate the Draft output schema text for prompts."""
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    lines = []
    for f in DRAFT_OUTPUT_FIELDS:
        if f == FIELD_DRAFT_BACKGROUND or f == FIELD_REUSED_CONCLUSIONS:
            lines.append(f"    - {f}: string[]")
        elif f == FIELD_REUSED_REFS:
            comment = (
                "structured reuse references (must include source_step:int; for MCQ also mcq_choice:A-D)"
                if use_en
                else "结构化复用引用（至少包含 source_step:int；MCQ 时包含 mcq_choice:A-D）"
            )
            lines.append(f"    - {f}: object[]  # {comment}")
        elif f == FIELD_DRAFT_QUESTION_EXPLICIT:
            comment = (
                "explicit version: shows how prior result is used (for verification)"
                if use_en
                else "显式版本：写出如何使用前序结论（用于验证）"
            )
            lines.append(f"    - {f}: string  # {comment}")
        elif f == FIELD_DRAFT_QUESTION:
            comment = (
                "hidden version: same question but WITHOUT mentioning prior result (actual output)"
                if use_en
                else "隐藏版本：同一问题但不提及前序结论（实际输出）"
            )
            lines.append(f"    - {f}: string  # {comment}")
        else:
            lines.append(f"    - {f}: string")
    return "\n".join(lines)


def draft_output_to_dict(draft_out: Any, is_first_step: bool = False) -> Dict[str, Any]:
    """将 DraftOutput 转换为 dict（用于 JSON 输出）。

    节点文件应调用此函数而不是手动构建 dict，确保字段名一致。

    Args:
        draft_out: Draft 输出对象
        is_first_step: 是否是第一题（用于添加说明注释）
    """
    result = {
        FIELD_DRAFT_QUESTION_EXPLICIT: getattr(draft_out, "draft_question_explicit", "") or "",
        FIELD_DRAFT_QUESTION: draft_out.draft_question,
        FIELD_DRAFT_SOLUTION: draft_out.draft_solution,
        FIELD_DRAFT_ANSWER: draft_out.draft_answer,
        FIELD_DRAFT_BACKGROUND: draft_out.draft_background,
        FIELD_REUSED_CONCLUSIONS: draft_out.reused_conclusions,
        FIELD_REUSED_REFS: getattr(draft_out, "reused_refs", []) or [],
        FIELD_GROUNDING_CHECK: draft_out.grounding_check,
    }

    # 为第一题添加说明注释
    if is_first_step and not result[FIELD_DRAFT_QUESTION_EXPLICIT]:
        result["_comment"] = "First step: draft_question_explicit is empty because there are no prior conclusions to reuse"

    return result
