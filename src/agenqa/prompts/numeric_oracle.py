"""NumericOracle role prompt (Python style).

This role is an internal helper for `QuestionType=Numeric`:
- produce deterministic oracle_code to compute the numeric ground truth
- produce per-question tolerance/precision conventions (abs/rel or sig figs)
"""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.domain.numeric_oracle_schema import (
    NUMERIC_ORACLE_PROMPT_GUIDE_EN,
    NUMERIC_ORACLE_PROMPT_GUIDE_ZH,
    numeric_oracle_output_schema_text,
)

__all__ = [
    "NUMERIC_ORACLE_V1",
    "NUMERIC_ORACLE_V1_EN",
]


NUMERIC_ORACLE_V1 = dedent(
    f"""\
    # NumericOracle（Numeric 题的 oracle 与容差口径）

    ## Input
    - step: $step
    - question: $question
    - solution: $solution

    ## Task
    你需要基于 question/solution：
    1) 给出本题判题所需的容差/精度口径（abs/rel 或 sig_figs）；
    2) 生成可执行的 oracle_code，用于计算本题的 ground-truth 数值。

{indent(NUMERIC_ORACLE_PROMPT_GUIDE_ZH, "    ")}

    ## Output format (JSON)
    仅输出一个 JSON 对象，必须用 ```json 代码块包裹。字段如下：
{indent(numeric_oracle_output_schema_text(), "    ")}
    """
)


NUMERIC_ORACLE_V1_EN = dedent(
    f"""\
    # NumericOracle (oracle + tolerance for Numeric questions)

    ## Input
    - step: $step
    - question: $question
    - solution: $solution

    ## Task
    Based on question/solution:
    1) choose a tolerance/precision convention (abs/rel or sig_figs);
    2) write deterministic oracle_code that computes the ground-truth numeric value.

{indent(NUMERIC_ORACLE_PROMPT_GUIDE_EN, "    ")}

    ## Output format (JSON)
    Output one JSON object only, wrapped in ```json. Fields:
{indent(numeric_oracle_output_schema_text("en"), "    ")}
    """
)
