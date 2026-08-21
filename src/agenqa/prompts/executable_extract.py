"""ExecutableExtract role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.prompts.common import (
    COMMON_EXECUTABLE_EXTRACT_SCHEMA,
    COMMON_EXECUTABLE_EXTRACT_SCHEMA_EN,
)

__all__ = [
    "EXECUTABLE_EXTRACT_V1",
    "EXECUTABLE_EXTRACT_V1_EN",
]


EXECUTABLE_EXTRACT_V1 = dedent(
    f"""\
    # ExecutableExtract（从论文背景抽取可做的 executable 任务骨架）

    ## Input
    - director_notes: $director_notes
    - paper_background: $paper_background
    - problem_description: $problem_description
    - dependencies_whitelist: $dependencies_whitelist

    ## Task
    基于论文背景，判断是否适合构造“可执行/可评测”的多步 executable 题。
    若适合：给出任务概述与**最多 1 个**起步 sub-step（接口/目标），不要一次性生成整条链。

    ## Hard constraints (must follow)
    - 只输出抽象任务骨架与接口定义，不写任何 golden 代码。
    - sub_step 必须对应独立可测函数/接口（带 function_header 与 return_line）。
    - 依赖必须落在 dependencies_whitelist 内；不要引入不可用库。
    - 若 director_notes 给出偏好/风险提示（例如“avoid X / try Y”），默认应当优先遵循。
    - 输出必须是严格 JSON（用 ```json 代码块包裹），字段如下：
{indent(COMMON_EXECUTABLE_EXTRACT_SCHEMA.text, "    ")}
    """
)


EXECUTABLE_EXTRACT_V1_EN = dedent(
    f"""\
    # ExecutableExtract (extract a executable task skeleton)

    ## Input
    - director_notes: $director_notes
    - paper_background: $paper_background
    - problem_description: $problem_description
    - dependencies_whitelist: $dependencies_whitelist

    ## Task
    Decide whether the paper background is suitable for a multi-step, executable executable task.
    If suitable: provide a task sketch and **at most one** initial sub-step (interface/goal). Do NOT generate the full chain.

    ## Hard constraints (must follow)
    - Output only the task skeleton and interfaces; do NOT include golden code.
    - Each sub_step must be an independently testable function/interface (with function_header + return_line).
    - Dependencies must be within the whitelist.
    - If director_notes provides preferences / risk hints (e.g., “avoid X / try Y”), prefer to follow them by default.
    - Output must be strict JSON wrapped in ```json with fields:
{indent(COMMON_EXECUTABLE_EXTRACT_SCHEMA_EN.text, "    ")}
    """
)
