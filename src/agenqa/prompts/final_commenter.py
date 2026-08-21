"""Final Commenter prompt (Python style).

This role comments on the final Path question:
- well-posedness
- difficulty / reasoning footprint
- whether intermediate steps are necessary (based on edge vs path signals)
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict
import json

from ._base import PromptSection, PromptTemplate
from agenqa.domain.final_comment_schema import final_comment_output_schema_text
from .common import COMMON_EDGE_QA_VS_PATH, COMMON_EDGE_QA_VS_PATH_EN

__all__ = ["FINAL_COMMENTER_TEMPLATE", "FINAL_COMMENTER_TEMPLATE_EN", "build_final_commenter_v1_body"]


FINAL_COMMENTER_ROLE_SECTION = PromptSection(
    text=dedent(
        """\
        You are the Final Commenter.

        Your task: comment on the final question under the Path view:
        - Is the question well-posed (unique, checkable answer; sufficient info for a solver)?
        - Does the Question explicitly leak previous-step conclusions (this invalidates path-fold)?
        - Is the chain's intermediate reasoning likely necessary (based on edge vs path signals)?
        - Give a qualitative difficulty assessment.

        Keep comments concise and evidence-based. Do not rewrite the question. Do not add new assumptions.
        """
    )
)


FINAL_COMMENTER_INPUT_SECTION = PromptSection(
    text=dedent(
        """\
        Below are the inputs (JSON):
        - edge_kqa (solver sees edge-visible Known):
        {edge_kqa_json}

        - path_kqa (solver sees path-visible Known):
        {path_kqa_json}

        - solver_summary (edge vs path; medium/strong):
        {solver_summary_json}

        Output language: {lang_hint}
        """
    )
)


FINAL_COMMENTER_OUTPUT_SECTION = PromptSection(
    text=dedent(
        """\
        Output requirement:
        - Output a single strict JSON object (no Markdown, no code fences).
        - Follow this schema (keys must match exactly):
        {schema_text}
        """
    )
)


FINAL_COMMENTER_TEMPLATE = PromptTemplate(
    name="final_commenter_v1",
    sections=[
        FINAL_COMMENTER_ROLE_SECTION,
        COMMON_EDGE_QA_VS_PATH,
        FINAL_COMMENTER_INPUT_SECTION,
        FINAL_COMMENTER_OUTPUT_SECTION,
    ],
)


FINAL_COMMENTER_TEMPLATE_EN = PromptTemplate(
    name="final_commenter_v1_en",
    sections=[
        FINAL_COMMENTER_ROLE_SECTION,
        COMMON_EDGE_QA_VS_PATH_EN,
        FINAL_COMMENTER_INPUT_SECTION,
        FINAL_COMMENTER_OUTPUT_SECTION,
    ],
)


def build_final_commenter_v1_body(payload: Dict[str, Any], *, lang: str | None) -> str:
    """Render Final Commenter v1 user prompt body from structured payload."""
    lang_norm = (lang or "").strip().lower()
    lang_hint = "English" if lang_norm in {"en", "english"} else "中文"
    template = FINAL_COMMENTER_TEMPLATE_EN if lang_norm in {"en", "english"} else FINAL_COMMENTER_TEMPLATE
    ctx: Dict[str, Any] = {
        "edge_kqa_json": json.dumps(payload.get("edge_kqa") or {}, ensure_ascii=False, indent=2),
        "path_kqa_json": json.dumps(payload.get("path_kqa") or {}, ensure_ascii=False, indent=2),
        "solver_summary_json": json.dumps(payload.get("solver_summary") or {}, ensure_ascii=False, indent=2),
        "lang_hint": lang_hint,
        "schema_text": final_comment_output_schema_text(),
    }
    return template.render_body(ctx)
