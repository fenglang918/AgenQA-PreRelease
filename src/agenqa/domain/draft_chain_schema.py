"""DraftChain 角色输出 Schema 定义（Single Source of Truth）。"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, List, Literal

from agenqa.domain.numeric_oracle_schema import (
    FIELD_ABS_TOL,
    FIELD_NOTES,
    FIELD_ORACLE_CODE,
    FIELD_REL_TOL,
    FIELD_SIG_FIGS,
    FIELD_UNIT,
)

if TYPE_CHECKING:
    from agenqa.skills.draft_chain import DraftChainOutput

FIELD_DRAFT_QUESTION_EXPLICIT = "draft_question_explicit"
FIELD_DRAFT_SOLUTION_OUTLINE = "draft_solution_outline"
FIELD_DRAFT_ANSWER = "draft_answer"
FIELD_SUBTASKS = "subtasks"
FIELD_FINAL_SUBTASK_ID = "final_subtask_id"
FIELD_DEPENDENCIES = "dependencies"
FIELD_REQUIRED_FACT_IDS = "required_fact_ids"
FIELD_PRIMARY_REQUIRED_FACT_ID = "primary_required_fact_id"
FIELD_REUSE_PLAN = "reuse_plan"
FIELD_WORLD_CONTRACT = "world_contract"

_DRAFT_CHAIN_NUMERIC_TOOL_FIELDS = [
    FIELD_ABS_TOL,
    FIELD_REL_TOL,
    FIELD_SIG_FIGS,
    FIELD_UNIT,
    FIELD_ORACLE_CODE,
    FIELD_NOTES,
]

DRAFT_CHAIN_OUTPUT_FIELDS = [
    FIELD_SUBTASKS,
    FIELD_FINAL_SUBTASK_ID,
    FIELD_DEPENDENCIES,
    FIELD_DRAFT_QUESTION_EXPLICIT,
    FIELD_DRAFT_SOLUTION_OUTLINE,
    FIELD_DRAFT_ANSWER,
    FIELD_REQUIRED_FACT_IDS,
    FIELD_PRIMARY_REQUIRED_FACT_ID,
    FIELD_REUSE_PLAN,
]


def draft_chain_output_schema_text(
    lang: str | None = None,
    *,
    question_type: str | None = None,
    world_contract_policy: Literal["omit", "optional", "required"] = "optional",
) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    qt = (question_type or "").strip().lower()
    include_numeric_tool = qt == "numeric"
    reuse_comment = (
        "list of how each required_fact is reused"
        if use_en
        else "逐条说明每个 required_fact 如何被复用"
    )
    subtasks_comment = (
        "object[]; each item: {id, description, result}. For step>=2, prefer exactly 2 coarse subtasks: sub_prev, sub_step."
        if use_en
        else "object[]；每项：{id, description, result}。建议 step>=2 时恰好 2 个粗粒度子任务：sub_prev、sub_step。"
    )
    deps_comment = (
        "object mapping subtask_id -> string[] of prerequisite subtask ids"
        if use_en
        else "object；子任务依赖映射：subtask_id -> string[]（前置子任务 id 列表）"
    )
    lines = [
        f"- {FIELD_SUBTASKS}: object[]  # {subtasks_comment}",
        f"- {FIELD_FINAL_SUBTASK_ID}: string",
        f"- {FIELD_DEPENDENCIES}: object  # {deps_comment}",
        f"- {FIELD_DRAFT_QUESTION_EXPLICIT}: string",
        f"- {FIELD_DRAFT_SOLUTION_OUTLINE}: string",
        f"- {FIELD_DRAFT_ANSWER}: string",
        f"- {FIELD_REQUIRED_FACT_IDS}: string[]",
        f"- {FIELD_PRIMARY_REQUIRED_FACT_ID}: string",
        f"- {FIELD_REUSE_PLAN}: string[]  # {reuse_comment}",
    ]
    if world_contract_policy != "omit":
        if world_contract_policy == "required":
            comment = (
                "required; structured world_contract object for Type1 world-view fixing (layered sections L1-L4 + points)"
                if use_en
                else "必填；Type1 语义世界观治理的结构化对象（分层 sections L1-L4 + points）"
            )
            lines.append(f"- {FIELD_WORLD_CONTRACT}: object  # {comment}")
        else:
            comment = (
                "optional; structured world_contract object (Type1; layered sections+points); use null when not applicable"
                if use_en
                else "可选；结构化 world_contract（Type1；sections+points）；不适用时置为 null"
            )
            lines.append(f"- {FIELD_WORLD_CONTRACT}: object | null  # {comment}")
    if include_numeric_tool:
        # For Numeric questions, DraftChain may additionally emit a deterministic oracle_code + tolerance.
        if use_en:
            lines.extend(
                [
                    f"- {FIELD_ABS_TOL}: number | null  # absolute error tolerance (optional if sig_figs is set)",
                    f"- {FIELD_REL_TOL}: number | null  # relative error tolerance (optional if sig_figs is set)",
                    f"- {FIELD_SIG_FIGS}: integer | null  # significant figures (alternative to abs/rel tol)",
                    f"- {FIELD_UNIT}: string  # unit label (optional; must not include expected value)",
                    f"- {FIELD_ORACLE_CODE}: string  # deterministic Python; stdout prints {{\"value\": <number>}}",
                    f"- {FIELD_NOTES}: string  # brief non-CoT notes; may be empty",
                ]
            )
        else:
            lines.extend(
                [
                    f"- {FIELD_ABS_TOL}: number | null  # 绝对误差阈值（若设置 sig_figs 则可为空）",
                    f"- {FIELD_REL_TOL}: number | null  # 相对误差阈值（若设置 sig_figs 则可为空）",
                    f"- {FIELD_SIG_FIGS}: integer | null  # 有效数字位数（abs/rel 的替代口径）",
                    f"- {FIELD_UNIT}: string  # 输出单位标注（可为空；不要包含期望数值）",
                    f"- {FIELD_ORACLE_CODE}: string  # 确定性 Python；stdout 打印 {{\"value\": <number>}}",
                    f"- {FIELD_NOTES}: string  # 简短、非 CoT 说明，可为空",
                ]
            )
    return "\n".join(lines)


def draft_chain_output_to_dict(out: "DraftChainOutput") -> Dict[str, Any]:
    d: Dict[str, Any] = {
        FIELD_SUBTASKS: getattr(out, "subtasks", []),
        FIELD_FINAL_SUBTASK_ID: getattr(out, "final_subtask_id", ""),
        FIELD_DEPENDENCIES: getattr(out, "dependencies", {}),
        FIELD_DRAFT_QUESTION_EXPLICIT: out.draft_question_explicit,
        FIELD_DRAFT_SOLUTION_OUTLINE: out.draft_solution_outline,
        FIELD_DRAFT_ANSWER: out.draft_answer,
        FIELD_REQUIRED_FACT_IDS: out.required_fact_ids,
        FIELD_PRIMARY_REQUIRED_FACT_ID: out.primary_required_fact_id,
        FIELD_REUSE_PLAN: out.reuse_plan,
        FIELD_WORLD_CONTRACT: getattr(out, "world_contract", None),
    }
    for f in _DRAFT_CHAIN_NUMERIC_TOOL_FIELDS:
        if hasattr(out, f):
            d[f] = getattr(out, f)
    return d


__all__ = [
    "FIELD_DRAFT_QUESTION_EXPLICIT",
    "FIELD_DRAFT_SOLUTION_OUTLINE",
    "FIELD_DRAFT_ANSWER",
    "FIELD_SUBTASKS",
    "FIELD_FINAL_SUBTASK_ID",
    "FIELD_DEPENDENCIES",
    "FIELD_REQUIRED_FACT_IDS",
    "FIELD_PRIMARY_REQUIRED_FACT_ID",
    "FIELD_REUSE_PLAN",
    "FIELD_WORLD_CONTRACT",
    "FIELD_ABS_TOL",
    "FIELD_REL_TOL",
    "FIELD_SIG_FIGS",
    "FIELD_UNIT",
    "FIELD_ORACLE_CODE",
    "FIELD_NOTES",
    "DRAFT_CHAIN_OUTPUT_FIELDS",
    "draft_chain_output_schema_text",
    "draft_chain_output_to_dict",
]
