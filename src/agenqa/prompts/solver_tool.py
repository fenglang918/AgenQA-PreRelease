"""Tool-enabled solver prompt (Python style).

This prompt is used by an optional solver tier that may output executable
Python code for Numeric questions. The pipeline will execute the code to
verify and/or obtain the numeric value.
"""

from __future__ import annotations

from textwrap import dedent, indent

from ._base import PromptSection, PromptTemplate
from .common import COMMON_ANSWER_SCHEMA, COMMON_ANSWER_SCHEMA_EN
from agenqa.domain.solver_schema import (
    FIELD_ANSWER,
    FIELD_SOLVER_REASONING,
    FIELD_QUESTION_WELL_POSED,
    FIELD_CORRECTNESS_FEEDBACK,
    FIELD_DIFFICULTY_FEEDBACK,
    FIELD_KEY_CONCLUSION,
)
from agenqa.domain.solver_tool_schema import (
    FIELD_TOOL_USED,
    FIELD_TOOL_NAME,
    FIELD_TOOL_CODE,
    FIELD_TOOL_NOTES,
    SOLVER_TOOL_GUIDE_ZH,
    SOLVER_TOOL_GUIDE_EN,
    solver_tool_output_schema_text,
)

__all__ = [
    "SOLVER_TOOL_PROMPT",
    "SOLVER_TOOL_PROMPT_EN",
    "SOLVER_TOOL_TEMPLATE",
    "SOLVER_TOOL_TEMPLATE_EN",
]


_INPUT_HEADER = PromptSection(
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

_INPUT_HEADER_EN = PromptSection(
    text=dedent(
        """\
        # Input
        {
          Known: $known,
          Question: $question
        }

        Solve the question based on the Known and Question above.
        All output text must be in English.
        """
    )
)

_OUTPUT_REQ = PromptSection(
    text=dedent(
        """\
        # Output
        仅输出一个合法 JSON，必须用 ```json 代码块包裹，不得包含额外解释或标记。

        你可以（但不强制）在 Numeric 题上使用工具：输出一段可执行 Python 代码，我们会运行它并获取结果用于验证。
        """
    )
)

_OUTPUT_REQ_EN = PromptSection(
    text=dedent(
        """\
        # Output
        Output one valid JSON object only, wrapped in a ```json code block. Do not include any extra text or markers.

        You MAY (but are not required to) use a tool for Numeric questions by outputting executable Python code.
        We will execute it and use the result for verification.
        """
    )
)

_FIELD_SEMANTICS = PromptSection(
    text=dedent(
        """\
        ## Output 字段含义

        - `Answer`：最终答案，用 `\\\\boxed{...}` 包裹。
        - `SolverReasoning`：推导路径摘要（2-3 句话），供 Diagnose 参考。
        - `QuestionWellPosed`：题目是否 well-posed（条件自洽、信息充分、答案唯一）。
        - `CorrectnessFeedback`：题目正确性/well-posedness 反馈（必填）。
        - `DifficultyFeedback`：难度反馈与加难建议（仅当 QuestionWellPosed=true 时填写，否则为 null）。
        - `KeyConclusion`（可选）：可复用的关键中间结论。

        - `ToolUsed/ToolName/ToolCode/ToolNotes`：可选的工具输出（仅当你确实使用工具时填写）。
        """
    )
)

_FIELD_SEMANTICS_EN = PromptSection(
    text=dedent(
        """\
        ## Output field semantics

        - `Answer`: final answer, wrapped in `\\\\boxed{...}`.
        - `SolverReasoning`: 2-3 sentence derivation summary for Diagnose reference.
        - `QuestionWellPosed`: whether the question is well-posed.
        - `CorrectnessFeedback`: required well-posedness/correctness feedback.
        - `DifficultyFeedback`: only when QuestionWellPosed=true; otherwise null.
        - `KeyConclusion` (optional): reusable key intermediate conclusion.

        - `ToolUsed/ToolName/ToolCode/ToolNotes`: optional tool output (only when you actually used the tool).
        """
    )
)

_JSON_FORMAT = PromptSection(
    text=dedent(
        f"""\
        JSON 格式（必须用 ```json 代码块包裹）：

        ```json
        {{
          "{FIELD_ANSWER}": "...最终答案（用 \\\\boxed{{...}} 包裹）...",
          "{FIELD_SOLVER_REASONING}": "...推导过程摘要（2–3 句话，可为 null）...",
          "{FIELD_QUESTION_WELL_POSED}": true/false,
          "{FIELD_CORRECTNESS_FEEDBACK}": "...必填...",
          "{FIELD_DIFFICULTY_FEEDBACK}": "...仅当 QuestionWellPosed=true 时填写，否则为 null...",
          "{FIELD_KEY_CONCLUSION}": "...可选...",

          "{FIELD_TOOL_USED}": true/false,
          "{FIELD_TOOL_NAME}": "python_executor",
          "{FIELD_TOOL_CODE}": "...python code string...",
          "{FIELD_TOOL_NOTES}": "...optional..."
        }}
        ```

        Tool 字段说明：
{indent(solver_tool_output_schema_text(), "        ")}
        """
    )
)

_JSON_FORMAT_EN = PromptSection(
    text=dedent(
        f"""\
        JSON format (must be wrapped in a ```json code block):

        ```json
        {{
          "{FIELD_ANSWER}": "...final answer (wrapped in \\\\boxed{{...}})...",
          "{FIELD_SOLVER_REASONING}": "...2–3 sentence summary (may be null)...",
          "{FIELD_QUESTION_WELL_POSED}": true/false,
          "{FIELD_CORRECTNESS_FEEDBACK}": "...required...",
          "{FIELD_DIFFICULTY_FEEDBACK}": "...only if QuestionWellPosed=true; otherwise null...",
          "{FIELD_KEY_CONCLUSION}": "...optional...",

          "{FIELD_TOOL_USED}": true/false,
          "{FIELD_TOOL_NAME}": "python_executor",
          "{FIELD_TOOL_CODE}": "...python code string...",
          "{FIELD_TOOL_NOTES}": "...optional..."
        }}
        ```

        Tool fields:
{indent(solver_tool_output_schema_text("en"), "        ")}
        """
    )
)


SOLVER_TOOL_TEMPLATE = PromptTemplate(
    name="solver_tool_v1_zh",
    sections=[
        _INPUT_HEADER,
        _OUTPUT_REQ,
        COMMON_ANSWER_SCHEMA,
        PromptSection(text=SOLVER_TOOL_GUIDE_ZH),
        _FIELD_SEMANTICS,
        _JSON_FORMAT,
    ],
)

SOLVER_TOOL_PROMPT = SOLVER_TOOL_TEMPLATE.render_body({})


SOLVER_TOOL_TEMPLATE_EN = PromptTemplate(
    name="solver_tool_v1_en",
    sections=[
        _INPUT_HEADER_EN,
        _OUTPUT_REQ_EN,
        COMMON_ANSWER_SCHEMA_EN,
        PromptSection(text=SOLVER_TOOL_GUIDE_EN),
        _FIELD_SEMANTICS_EN,
        _JSON_FORMAT_EN,
    ],
)

SOLVER_TOOL_PROMPT_EN = SOLVER_TOOL_TEMPLATE_EN.render_body({})
