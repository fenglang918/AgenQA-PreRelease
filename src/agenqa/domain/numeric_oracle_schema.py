"""Numeric oracle schema (single source of truth).

This module defines the structured output contract for a Numeric-oracle helper role:
- It produces per-question tolerance (abs/rel or significant figures).
- It produces a small Python snippet to compute the ground-truth numeric value.

The oracle output is internal (edge-only) and should not be leaked into path prompts.
"""

from __future__ import annotations

from textwrap import dedent

FIELD_ABS_TOL = "abs_tol"
FIELD_REL_TOL = "rel_tol"
FIELD_SIG_FIGS = "sig_figs"
FIELD_UNIT = "unit"
FIELD_ORACLE_CODE = "oracle_code"
FIELD_NOTES = "notes"


def numeric_oracle_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").strip().lower()
    use_en = lang_norm in {"en", "english"}
    if use_en:
        return "\n".join(
            [
                f"- {FIELD_ABS_TOL}: number | null  # absolute error tolerance",
                f"- {FIELD_REL_TOL}: number | null  # relative error tolerance",
                f"- {FIELD_SIG_FIGS}: integer | null  # significant figures (alternative to abs/rel tol)",
                f"- {FIELD_UNIT}: string  # output unit label (optional; keep short; do not include expected value)",
                f"- {FIELD_ORACLE_CODE}: string  # Python code that prints a JSON object: {{\"value\": <number>}}",
                f"- {FIELD_NOTES}: string  # brief non-CoT notes; may be empty",
            ]
        )
    return "\n".join(
        [
            f"- {FIELD_ABS_TOL}: number | null  # 绝对误差阈值",
            f"- {FIELD_REL_TOL}: number | null  # 相对误差阈值",
            f"- {FIELD_SIG_FIGS}: integer | null  # 有效数字位数（abs/rel 的替代口径）",
            f"- {FIELD_UNIT}: string  # 输出单位标注（可为空；不要包含期望数值）",
            f"- {FIELD_ORACLE_CODE}: string  # Python 代码，运行后 stdout 打印 JSON：{{\"value\": <number>}}",
            f"- {FIELD_NOTES}: string  # 简短、非 CoT 的说明，可为空",
        ]
    )


NUMERIC_ORACLE_PROMPT_GUIDE_ZH = dedent(
    """\
    ## Numeric Oracle 约束（重要）

    - 你的目标：为一个 Numeric 题型生成可执行的 oracle_code，并给出题面判题所需的容差/精度口径。
    - oracle_code 必须是**确定性**的 Python 代码，且必须在 stdout 输出一行 JSON：
      `{"value": <number>}`
    - 禁止访问网络；禁止读取/写入外部文件；只允许使用 Python 标准库（如 math/decimal/fractions）。
    - 只输出工具需要的内容：不要在 notes 中泄露推导过程或期望答案。
    - 容差口径二选一：
      1) abs_tol/rel_tol（推荐，数值均为正数）；或
      2) sig_figs（正整数，表示有效数字位数）。
    """
).rstrip()

NUMERIC_ORACLE_PROMPT_GUIDE_EN = dedent(
    """\
    ## Numeric Oracle constraints (important)

    - Goal: for a Numeric question, produce deterministic oracle_code and the tolerance/precision convention.
    - oracle_code must be deterministic Python and must print exactly one JSON line to stdout:
      `{\"value\": <number>}`
    - No network; no external file I/O; only Python stdlib (e.g., math/decimal/fractions).
    - Do not leak derivations or expected answers in notes.
    - Choose ONE tolerance convention:
      1) abs_tol/rel_tol (recommended; positive numbers), or
      2) sig_figs (positive integer; significant figures).
    """
).rstrip()


__all__ = [
    "FIELD_ABS_TOL",
    "FIELD_REL_TOL",
    "FIELD_SIG_FIGS",
    "FIELD_UNIT",
    "FIELD_ORACLE_CODE",
    "FIELD_NOTES",
    "numeric_oracle_output_schema_text",
    "NUMERIC_ORACLE_PROMPT_GUIDE_ZH",
    "NUMERIC_ORACLE_PROMPT_GUIDE_EN",
]
