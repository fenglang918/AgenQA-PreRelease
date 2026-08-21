"""ExecutableDraftStep role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.prompts.common import (
    COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA,
    COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA_EN,
)

__all__ = [
    "EXECUTABLE_DRAFT_STEP_V1",
    "EXECUTABLE_DRAFT_STEP_V1_EN",
]


EXECUTABLE_DRAFT_STEP_V1 = dedent(
    f"""\
    # ExecutableDraftStep（逐步生成：只生成下一步的 sub-step 与本步 golden 代码）

    ## Input
    - step: $step
    - director_notes: $director_notes
    - task_sketch: $task_sketch
    - background: $background
    - prev_sub_steps_json: $prev_sub_steps_json
    - expected_primary_fact_id: $expected_primary_fact_id
    - dependencies_whitelist: $dependencies_whitelist

    ## Task
    基于已生成的 sub_steps，补全**下一步**的接口定义与本步实现（golden_step_code）。
    仅生成当前 step 的增量，不要一次性生成整条链或未来步骤。

    ## Hard constraints (must follow)
    - 输出必须只包含“下一步”的 sub_step 与本步代码；不得重写历史步骤。
    - golden_step_code 只实现当前 step 的函数/类，不能包含其它步骤的实现。
    - 不输出任何测试代码；测试由系统后续推导。
    - 代码必须确定性（deterministic）且无副作用：禁止网络/文件 I/O、随机性、sleep、环境依赖。
    - 依赖必须在 dependencies_whitelist 内；必要 imports 写在代码里。
    - 输出必须是严格 JSON（用 ```json 代码块包裹），字段如下：
{indent(COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA.text, "    ")}

    ## Director Notes（重要但非绝对）
    - director_notes 提供偏好/风险提示（例如“avoid X / try Y”）。默认应当优先遵循，避免无声忽略。
    - 若你认为无法遵循（例如会破坏可测性、确定性或与“单步递进”冲突），允许偏离，但必须在 dependencies 字段用 1–2 句说明原因与权衡。

    ## 单步递进（关键）
    - 本步只新增一个可独立测试的接口（sub_step），不得在一个 step 内展开长链。
    - 当存在 prev_sub_steps 时：本步应当复用上一步接口作为必要中间能力（例如调用上一步函数），而不是绕开它“直接做最终计算”。

    ## 链式复用声明（对齐 semantic draft_chain）
    - 你必须在输出中填写：required_fact_ids / primary_required_fact_id / reuse_plan。
    - 若 step=1：required_fact_ids 必须为空，primary_required_fact_id 必须为空字符串或 null。
    - 若 step>=2 且 expected_primary_fact_id 非空：
      - primary_required_fact_id 必须等于 expected_primary_fact_id
      - required_fact_ids 必须包含 expected_primary_fact_id
      - reuse_plan 需用 1–3 条短句说明如何复用该 fact（例如“把上一步函数输出作为本步输入的一部分”）

    ## JSON 约束（避免解析失败）
    - 只输出一个 ```json 代码块；不要输出任何解释文本。
    - golden_step_code 字段必须是 JSON 字符串：不要再包 ```python；换行请用 \\n 形式表达。
    """
)


EXECUTABLE_DRAFT_STEP_V1_EN = dedent(
    f"""\
    # ExecutableDraftStep (step-wise: generate only the next sub-step and its golden code)

    ## Input
    - step: $step
    - director_notes: $director_notes
    - task_sketch: $task_sketch
    - background: $background
    - prev_sub_steps_json: $prev_sub_steps_json
    - expected_primary_fact_id: $expected_primary_fact_id
    - dependencies_whitelist: $dependencies_whitelist

    ## Task
    Based on existing sub_steps, generate **only the next step**: its interface definition and golden_step_code.
    Do NOT generate the full chain or future steps.

    ## Hard constraints (must follow)
    - Output only the next sub_step and its code; do not rewrite previous steps.
    - golden_step_code must implement only the current step function/class.
    - Do NOT output any tests; tests are derived by the system later.
    - Code must be deterministic and side-effect-free: no network/file I/O, no randomness, no sleep, no environment-dependent behavior.
    - Dependencies must be within the whitelist; include required imports in code.
    - Output must be strict JSON wrapped in ```json with fields:
{indent(COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA_EN.text, "    ")}

    ## Director Notes (important, not absolute)
    - director_notes carries preferences / risk hints. Prefer to follow them by default; do not ignore silently.
    - If you deviate (e.g., it would break testability/determinism or the single-step constraint), briefly explain the tradeoff in the dependencies field (1–2 sentences).

    ## Single-step progression (critical)
    - Add exactly one testable interface (sub_step). Do not expand a long chain in one step.
    - If prev_sub_steps exist: this step should reuse the immediate previous step as a necessary intermediate capability (e.g., call the prior function), not bypass it.

    ## Reuse declaration (align semantic draft_chain)
    - You must fill: required_fact_ids / primary_required_fact_id / reuse_plan.
    - If step=1: required_fact_ids must be empty, and primary_required_fact_id must be empty string or null.
    - If step>=2 and expected_primary_fact_id is non-empty:
      - primary_required_fact_id must equal expected_primary_fact_id
      - required_fact_ids must include expected_primary_fact_id
      - reuse_plan: 1–3 short notes on how you reuse that fact (e.g., "use the previous function output as part of this step input")

    ## JSON rules (avoid parse failures)
    - Output exactly one ```json code block; no extra prose.
    - golden_step_code must be a JSON string; do not wrap it in ```python. Use \\n escapes for newlines.
    """
)
