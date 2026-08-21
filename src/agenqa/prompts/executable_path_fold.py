"""Executable Path-Fold role prompt (Python style).

This role folds a multi-step SciCode-like executable chain into a single path-view
spec for the current tail step, producing two variants:
- scaffolded: with intermediate sub-goals (easier)
- direct: minimal hints (harder)
"""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.domain.path_fold_schema import path_fold_output_schema_text

__all__ = [
    "EXECUTABLE_PATH_FOLD_V1",
    "EXECUTABLE_PATH_FOLD_V1_EN",
]


EXECUTABLE_PATH_FOLD_V1 = dedent(
    f"""\
    # ExecutablePathFold（Executable 路径折叠：生成 path 视角题面）

    ## Input
    - step: $step
    - question_type: $question_type
    - premise_bank_json: $premise_bank_json
    - history_json: $history_json

    说明：
    - premise_bank_json：path 起点 P 的 executable seed（仅前提/背景/依赖/接口，不含任何 ground-truth 代码）。
    - history_json：从起点到当前 tail 的编码链路信息（sub_steps 的接口与描述 + tail 的 tests 等），用于你理解“整条路径”的结构。

    ## Task
    你需要把“从 P 出发，经由多步编码链到达 tail step 的实现目标”折叠成一个**单步、可自包含**的 executable 题面，并输出两个版本：

    1) question_scaffolded（带中间目标提示版，较易）
       - 把原链条拆成若干子任务（例如 S1/S2/S3…），但不要用“第 N 步/上一题/历史”等内部指针描述；
       - 允许给出必要的函数/类签名与接口约定；
       - 最后一问与 tail step 的目标一致（语义等价），并确保题面足够自包含。

    2) question_direct（最少提示版，最难）
       - 直接描述最终目标与必要约束，不列出显式的子任务分解；
       - 只给出**必须**给出的接口（函数/类签名）与输入输出约定；
       - ❌ 禁止出现任何“根据上一步/根据 step X/如上所示”等指针式引用。

    ## Hard constraints (must follow)
    - 两个版本都必须自包含：不依赖“历史步骤/上一步结果/隐藏中间过程”才能理解与实现。
    - ❌ 不得输出任何 ground-truth / golden 代码；不得把完整实现作为题面的一部分。
    - ✅ 允许引用 premise_bank_json 中给定的背景/依赖/接口信息（但不要显式引用内部字段名）。
    - 输出必须是严格 JSON（用 ```json 代码块包裹），字段如下：
{indent(path_fold_output_schema_text(), '    ')}
    """
)


EXECUTABLE_PATH_FOLD_V1_EN = dedent(
    f"""\
    # ExecutablePathFold (Executable path folding: generate path-view specs)

    ## Input
    - step: $step
    - question_type: $question_type
    - premise_bank_json: $premise_bank_json
    - history_json: $history_json

    Notes:
    - premise_bank_json: the path start P executable seed (premises/background/dependencies/interfaces only; NO ground-truth code).
    - history_json: executable-chain info up to the current tail (interfaces + descriptions + tail tests) for understanding the full path.

    ## Task
    Fold the multi-step executable chain into a **single, self-contained** path-view executable prompt, and output two variants:

    1) question_scaffolded (easier, with intermediate sub-goals)
       - Decompose into sub-goals (e.g., S1/S2/S3), but do NOT use internal pointers like “step N / previous step / history”.
       - You may include necessary function/class signatures and interface contracts.
       - The final ask must match the tail step objective (answer-equivalent), and the prompt must be self-contained.

    2) question_direct (hardest, minimal hints)
       - Describe the final goal and required constraints directly; do NOT list explicit sub-goal decomposition.
       - Provide only the **necessary** interfaces (function/class signatures) and I/O contracts.
       - Forbidden: pointer-style references like “from the previous step”, “step X”, “as shown above”.

    ## Hard constraints (must follow)
    - Both variants must be self-contained (no reliance on hidden intermediate steps).
    - Forbidden: output any ground-truth / golden code; do NOT paste full solutions into the prompt.
    - Allowed: restate background/dependencies/interfaces from premise_bank_json (but do not cite internal field names).
    - Output must be strict JSON wrapped in ```json with fields:
{indent(path_fold_output_schema_text('en'), '    ')}
    """
)
