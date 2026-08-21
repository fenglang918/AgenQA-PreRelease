"""StepCertBuilder 角色输出 Schema 定义。"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from agenqa.skills.step_cert_builder import StepCertOutput

FIELD_PREMISE_DELTA = "premise_delta"
FIELD_FACT_DELTA = "fact_delta"
FIELD_STEP_CERT = "step_cert"
FIELD_KEY_FACT_ID = "key_fact_id"

STEP_CERT_OUTPUT_FIELDS = [
    FIELD_PREMISE_DELTA,
    FIELD_FACT_DELTA,
    FIELD_STEP_CERT,
    FIELD_KEY_FACT_ID,
]


def step_cert_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    premise_desc = "premise entries (id, kind, text, source_step, provenance)" if use_en else "前提条目（id, kind, text, source_step, provenance）"
    fact_desc = (
        "fact entries (id, kind, text/statement, tags, source_step, provenance; for MCQ key_fact also include mcq_choice, mcq_choice_text)"
        if use_en
        else "事实条目（id, kind, text/statement, tags, source_step, provenance；MCQ 的 key_fact 还需 mcq_choice, mcq_choice_text）"
    )
    cert_desc = "step certificate (step, uses_premise_ids, uses_fact_ids, produces_fact_ids, key_fact_id, cert_text)" if use_en else "推理证书（step, uses_premise_ids, uses_fact_ids, produces_fact_ids, key_fact_id, cert_text）"
    return "\n".join(
        [
            f"- {FIELD_PREMISE_DELTA}: object[]  # {premise_desc}",
            f"- {FIELD_FACT_DELTA}: object[]  # {fact_desc}",
            f"- {FIELD_STEP_CERT}: object  # {cert_desc}",
            f"- {FIELD_KEY_FACT_ID}: string",
        ]
    )


def step_cert_output_to_dict(out: "StepCertOutput") -> Dict[str, Any]:
    return {
        FIELD_PREMISE_DELTA: out.premise_delta,
        FIELD_FACT_DELTA: out.fact_delta,
        FIELD_STEP_CERT: out.step_cert,
        FIELD_KEY_FACT_ID: out.key_fact_id,
    }


__all__ = [
    "FIELD_PREMISE_DELTA",
    "FIELD_FACT_DELTA",
    "FIELD_STEP_CERT",
    "FIELD_KEY_FACT_ID",
    "STEP_CERT_OUTPUT_FIELDS",
    "step_cert_output_schema_text",
    "step_cert_output_to_dict",
]
