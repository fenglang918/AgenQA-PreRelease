"""AnswerContractBuilder 角色输出 Schema 定义。"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from agenqa.skills.answer_contract_builder import AnswerContractBuilderOutput

FIELD_ANSWER_STYLE = "answer_style"
FIELD_ANSWER_SEMANTICS = "answer_semantics"
FIELD_SUPPORT_WITNESS = "support_witness"

ANSWER_CONTRACT_BUILDER_OUTPUT_FIELDS = [
    FIELD_ANSWER_STYLE,
    FIELD_ANSWER_SEMANTICS,
    FIELD_SUPPORT_WITNESS,
]


def answer_contract_builder_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    style_desc = (
        "public answer-style constraints (boxed, form, rendering_notes)"
        if use_en
        else "公开的答案写法约束（boxed、form、rendering_notes）"
    )
    semantics_desc = (
        "answer-acceptance semantics (answer_object, acceptance_mode, branch_policy, allowed_symbols, required_qualifiers, equivalence_rules)"
        if use_en
        else "答案接受语义（answer_object、acceptance_mode、branch_policy、allowed_symbols、required_qualifiers、equivalence_rules）"
    )
    witness_desc = (
        "judge-side local witness list; each item: {type, statement}"
        if use_en
        else "judge 侧局部 witness 列表；每项：{type, statement}"
    )
    return "\n".join(
        [
            f"- {FIELD_ANSWER_STYLE}: object  # {style_desc}",
            f"- {FIELD_ANSWER_SEMANTICS}: object  # {semantics_desc}",
            f"- {FIELD_SUPPORT_WITNESS}: object[]  # {witness_desc}",
        ]
    )


def answer_contract_builder_output_to_dict(out: "AnswerContractBuilderOutput") -> Dict[str, Any]:
    return {
        FIELD_ANSWER_STYLE: out.answer_style,
        FIELD_ANSWER_SEMANTICS: out.answer_semantics,
        FIELD_SUPPORT_WITNESS: out.support_witness,
    }


__all__ = [
    "FIELD_ANSWER_STYLE",
    "FIELD_ANSWER_SEMANTICS",
    "FIELD_SUPPORT_WITNESS",
    "ANSWER_CONTRACT_BUILDER_OUTPUT_FIELDS",
    "answer_contract_builder_output_schema_text",
    "answer_contract_builder_output_to_dict",
]
