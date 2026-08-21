"""Solver prompt (Python style)."""

from __future__ import annotations
from textwrap import dedent

from ._base import PromptSection, PromptTemplate
from .common import COMMON_ANSWER_SCHEMA, COMMON_ANSWER_SCHEMA_EN
from agenqa.domain.solver_schema import (
    FIELD_ANSWER,
    FIELD_SOLVER_REASONING,
    FIELD_FEEDBACK,
    FIELD_HARDER_SUGGESTION,
    FIELD_QUESTION_WELL_POSED,
    FIELD_CORRECTNESS_FEEDBACK,
    FIELD_DIFFICULTY_FEEDBACK,
    FIELD_KEY_CONCLUSION,
)

__all__ = ["SOLVER_PROMPT", "SOLVER_TEMPLATE", "SOLVER_PROMPT_EN", "SOLVER_TEMPLATE_EN"]

_SOL_INPUT_HEADER = PromptSection(
    text=dedent(
        """\
        # Input
        {
          Known: $known,
          Question: $question
        }

        根据上述的 Known 与 Question 求解该题目。
        """
    )
)

_SOL_INPUT_HEADER_EN = PromptSection(
    text=dedent(
        """\
        # Input
        {
          Known: $known,
          Question: $question
        }

        Solve the question based on the Known and Question provided above.
        All output text must be in English.
        """
    )
)

_SOL_OUTPUT_REQ = PromptSection(
    text=dedent(
        """\
        # Output
        仅输出一个合法 JSON，必须用 ```json 代码块包裹，不得包含额外解释或标记。

        请严格按如下要求作答：
        - 遵守统一答案格式：
        """
    )
)

_SOL_OUTPUT_REQ_EN = PromptSection(
    text=dedent(
        """\
        # Output
        Output one valid JSON object only, wrapped in a ```json code block. Do not include any extra text or markers.

        Follow the unified answer format:
        """
    )
)

_SOL_REQ_SUFFIX = PromptSection(
    text=dedent(
        """\
        ## Output 字段含义

        - `Answer`：最终答案，用 `\\\\boxed{{}}` 包裹。
        - `SolverReasoning`：推导路径摘要（2-3 句话），供 Diagnose 参考。
        - `QuestionWellPosed`：题目是否 well-posed（条件自洽、信息充分、答案唯一）。
        - `CorrectnessFeedback`：
          * 若 `QuestionWellPosed=false`：具体问题点与修复建议。
          * 若 `QuestionWellPosed=true`：简要确认自洽性。
        - `DifficultyFeedback`：
          * 仅当 `QuestionWellPosed=true` 时：难度评价 + 1-3 条加难建议。
          * 否则为 `null`。
        - `KeyConclusion`（可选）：可复用的关键中间结论。
        """
    )
)

_SOL_REQ_SUFFIX_EN = PromptSection(
    text=dedent(
        """\
        ## Output field semantics

        - `Answer`: final answer, wrapped in `\\\\boxed{{...}}`.
        - `SolverReasoning`: derivation summary (2-3 sentences), for Diagnose reference.
        - `QuestionWellPosed`: whether the problem is well-posed (consistent, sufficient info, unique answer).
        - `CorrectnessFeedback`:
          * If `QuestionWellPosed=false`: concrete issues + fix suggestion.
          * If `QuestionWellPosed=true`: briefly confirm consistency.
        - `DifficultyFeedback`:
          * Only if `QuestionWellPosed=true`: difficulty assessment + 1-3 harder suggestions.
          * Otherwise `null`.
        - `KeyConclusion` (optional): reusable key intermediate conclusion.
        """
    )
)

_SOL_JSON = PromptSection(
    text=dedent(
        f"""\
        JSON 格式（必须用 ```json 代码块包裹）：

        ```json
        {{
          "{FIELD_ANSWER}": "...最终答案（用 \\\\boxed{{}} 包裹）...",
          "{FIELD_SOLVER_REASONING}": "...推导过程摘要（2–3 句话，供 Diagnose 参考，可为 null）...",
          "{FIELD_QUESTION_WELL_POSED}": true/false,
          "{FIELD_CORRECTNESS_FEEDBACK}": "...题目正确性/well-posedness 反馈（必填）...",
          "{FIELD_DIFFICULTY_FEEDBACK}": "...难度反馈与加难建议（仅当 QuestionWellPosed=true 时填写，否则为 null）...",
          "{FIELD_KEY_CONCLUSION}": "...关键中间结论（可选）..."
        }}
        ```

        字段说明：
        - {FIELD_ANSWER}：必填，最终答案，用 \\\\boxed{{}} 包裹。
        - {FIELD_SOLVER_REASONING}：可选，字符串或 null。用少量句子概括推导路径与关键中间结论，便于 Diagnose 分析。
        - {FIELD_QUESTION_WELL_POSED}：必填，布尔值，表示题目是否 well-posed。
        - {FIELD_CORRECTNESS_FEEDBACK}：必填，字符串。若 QuestionWellPosed=false，必须具体指出问题点与修复建议；若为 true，简要确认自洽性。
        - {FIELD_DIFFICULTY_FEEDBACK}：条件必填。仅当 QuestionWellPosed=true 时填写难度评价与 1–3 条加难建议；否则必须为 null。
        - {FIELD_KEY_CONCLUSION}：可选，字符串，指出可复用的关键中间结论。
        """
    )
)

_SOL_JSON_EN = PromptSection(
    text=dedent(
        f"""\
        JSON format (must be wrapped in a ```json code block):

        ```json
        {{
          "{FIELD_ANSWER}": "...final answer (wrapped in \\\\boxed{{...}})...",
          "{FIELD_SOLVER_REASONING}": "...brief derivation summary (2–3 sentences; may be null)...",
          "{FIELD_QUESTION_WELL_POSED}": true/false,
          "{FIELD_CORRECTNESS_FEEDBACK}": "...required well-posedness/correctness feedback...",
          "{FIELD_DIFFICULTY_FEEDBACK}": "...difficulty feedback + harder suggestions (only if QuestionWellPosed=true; otherwise null)...",
          "{FIELD_KEY_CONCLUSION}": "...optional key intermediate conclusion..."
        }}
        ```

        Field notes:
        - {FIELD_ANSWER}: required; wrap in \\\\boxed{{...}}.
        - {FIELD_SOLVER_REASONING}: optional; string or null; concise reasoning summary for Diagnose.
        - {FIELD_QUESTION_WELL_POSED}: required; boolean.
        - {FIELD_CORRECTNESS_FEEDBACK}: required; if QuestionWellPosed=false, state concrete issues + fix suggestion; if true, briefly confirm.
        - {FIELD_DIFFICULTY_FEEDBACK}: required conditionally; only fill when QuestionWellPosed=true; otherwise must be null.
        - {FIELD_KEY_CONCLUSION}: optional; key reusable intermediate conclusion.
        """
    )
)

SOLVER_TEMPLATE = PromptTemplate(
    name="solver_v1_mod",
    sections=[
        _SOL_INPUT_HEADER,
        # 对 Known 结构的说明在 _SOL_INPUT_HEADER 中已给出，避免引入 Extend/Revise 相关指令

        _SOL_OUTPUT_REQ,
        COMMON_ANSWER_SCHEMA,  # Insert Answer Schema
        # 注：Solver 不需要 COMMON_QUESTION_TYPES（题型口径定义），只需按题干理解并作答
        _SOL_REQ_SUFFIX,
        _SOL_JSON,
    ],
)

SOLVER_PROMPT = SOLVER_TEMPLATE.render_body({})


SOLVER_TEMPLATE_EN = PromptTemplate(
    name="solver_v1_en",
    sections=[
        _SOL_INPUT_HEADER_EN,
        _SOL_OUTPUT_REQ_EN,
        COMMON_ANSWER_SCHEMA_EN,
        _SOL_REQ_SUFFIX_EN,
        _SOL_JSON_EN,
    ],
)

SOLVER_PROMPT_EN = SOLVER_TEMPLATE_EN.render_body({})
