"""SolverTool schema (single source of truth).

This schema defines the structured output contract for a tool-enabled solver tier.
The core idea:
- The solver still outputs a final `Answer` in `\\boxed{...}`.
- Optionally, it may provide deterministic `ToolCode` that we will execute to obtain
  a numeric value for verification (and/or as its predicted answer).

Tool artifacts must be treated as internal (edge-only) and must not leak into path prompts.
"""

from __future__ import annotations

from textwrap import dedent

FIELD_TOOL_USED = "ToolUsed"
FIELD_TOOL_NAME = "ToolName"
FIELD_TOOL_CODE = "ToolCode"
FIELD_TOOL_NOTES = "ToolNotes"


def solver_tool_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").strip().lower()
    use_en = lang_norm in {"en", "english"}
    if use_en:
        return "\n".join(
            [
                f"- {FIELD_TOOL_USED}: boolean  # whether you used a tool/code",
                f"- {FIELD_TOOL_NAME}: string  # tool name (use \"python_executor\")",
                f"- {FIELD_TOOL_CODE}: string  # Python code; stdout must print one JSON line: {{\"value\": <number>}}",
                f"- {FIELD_TOOL_NOTES}: string  # brief non-CoT notes; may be empty",
            ]
        )
    return "\n".join(
        [
            f"- {FIELD_TOOL_USED}: boolean  # 是否使用工具/代码",
            f"- {FIELD_TOOL_NAME}: string  # 工具名（统一写 \"python_executor\"）",
            f"- {FIELD_TOOL_CODE}: string  # Python 代码；stdout 必须打印一行 JSON：{{\"value\": <number>}}",
            f"- {FIELD_TOOL_NOTES}: string  # 简短、非 CoT 说明，可为空",
        ]
    )


SOLVER_TOOL_GUIDE_ZH = dedent(
    """\
    ## Tool 约束（重要）

    - Tool 仅用于“数值计算/验证”：你可以写一段 Python 代码来计算最终数值答案。
    - 代码必须是**确定性**的；只允许使用 Python 标准库（math/decimal/fractions 等）。
    - 禁止网络访问；禁止读写外部文件；不要尝试读取环境变量或运行系统命令。
    - 代码运行后 stdout 必须只输出一行 JSON：`{"value": <number>}`。
      - 强烈建议使用 `json.dumps` 打印，避免手写 JSON / f-string 花括号转义等低级错误。例如：
        ```python
        import json
        print(json.dumps({"value": solve()}))
        ```
    - 你仍然必须给出最终 `Answer=\\boxed{...}`；Tool 只是辅助/可验证工件。
    """
).rstrip()

SOLVER_TOOL_GUIDE_EN = dedent(
    """\
    ## Tool constraints (important)

    - Tool use is ONLY for numeric computation/verification: you may write Python code to compute the final numeric answer.
    - The code must be deterministic; only Python stdlib is allowed (math/decimal/fractions, etc.).
    - No network; no external file I/O; do not read env vars or run system commands.
    - The code must print exactly one JSON line to stdout: `{"value": <number>}`.
      - Strongly recommended: print via `json.dumps` to avoid JSON formatting pitfalls, e.g.:
        ```python
        import json
        print(json.dumps({"value": solve()}))
        ```
    - You must still provide the final `Answer=\\boxed{...}`; the tool code is an internal, verifiable artifact.
    """
).rstrip()


__all__ = [
    "FIELD_TOOL_USED",
    "FIELD_TOOL_NAME",
    "FIELD_TOOL_CODE",
    "FIELD_TOOL_NOTES",
    "solver_tool_output_schema_text",
    "SOLVER_TOOL_GUIDE_ZH",
    "SOLVER_TOOL_GUIDE_EN",
]
