"""ExecutableReviseStep role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.prompts.common import (
    COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA,
    COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA_EN,
)

__all__ = [
    "EXECUTABLE_REVISE_STEP_V1",
    "EXECUTABLE_REVISE_STEP_V1_EN",
]


EXECUTABLE_REVISE_STEP_V1 = dedent(
    f"""\
    # ExecutableReviseStep（修复当前 step：改 spec 或改 golden，实现闭环）

    ## Input
    - step: $step
    - director_notes: $director_notes
    - task_sketch: $task_sketch
    - background: $background
    - prev_sub_steps_json: $prev_sub_steps_json
    - current_sub_step_json: $current_sub_step_json
    - current_golden_step_code: $current_golden_step_code
    - diagnose_json: $diagnose_json
    - dependencies_whitelist: $dependencies_whitelist

    ## Task
    你是在“Revise”模式下修复当前 step（不生成下一步）：
    - 保持 step_number 不变；
    - 可以修复 sub_step 的接口/描述（但必须保持可独立测试、且与实现一致）；
    - 修复 golden_step_code，使其满足确定性/可测性约束，并与接口契约一致；
    - 不得改写历史步骤，也不得在本步实现其它步骤的接口。

    ## Hard constraints
    - 代码必须确定性且无副作用：禁止网络/文件 I/O、随机性、sleep、环境依赖。
    - 依赖必须在 dependencies_whitelist 内；必要 imports 写在代码里。
    - 不输出任何测试代码；测试由系统后续推导。
    - 输出必须是严格 JSON（用 ```json 代码块包裹），字段如下：
{indent(COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA.text, "    ")}

    ## JSON 约束（避免解析失败）
    - 只输出一个 ```json 代码块；不要输出任何解释文本。
    - golden_step_code 字段必须是 JSON 字符串：不要再包 ```python；换行请用 \\n 形式表达。
    """
)


EXECUTABLE_REVISE_STEP_V1_EN = dedent(
    f"""\
    # ExecutableReviseStep (revise the current step: fix spec and/or golden code)

    ## Input
    - step: $step
    - director_notes: $director_notes
    - task_sketch: $task_sketch
    - background: $background
    - prev_sub_steps_json: $prev_sub_steps_json
    - current_sub_step_json: $current_sub_step_json
    - current_golden_step_code: $current_golden_step_code
    - diagnose_json: $diagnose_json
    - dependencies_whitelist: $dependencies_whitelist

    ## Task
    You are in Revise mode for the current step (do NOT generate the next step):
    - Keep the step_number unchanged;
    - You may fix the sub_step interface/description (must remain independently testable and consistent with the implementation);
    - Fix golden_step_code so it is deterministic/testable and matches the interface contract;
    - Do not rewrite prior steps; do not implement other steps in this step.

    ## Hard constraints
    - Code must be deterministic and side-effect-free: no network/file I/O, no randomness, no sleep, no environment-dependent behavior.
    - Dependencies must be within the whitelist; include required imports in code.
    - Do NOT output any tests; tests are derived by the system later.
    - Output must be strict JSON wrapped in ```json with fields:
{indent(COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA_EN.text, "    ")}

    ## JSON rules (avoid parse failures)
    - Output exactly one ```json code block; no extra prose.
    - golden_step_code must be a JSON string; do not wrap it in ```python. Use \\n escapes for newlines.
    """
)
