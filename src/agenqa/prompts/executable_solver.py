"""ExecutableSolver role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent

__all__ = [
    "EXECUTABLE_SOLVER_V1",
    "EXECUTABLE_SOLVER_V1_EN",
]


EXECUTABLE_SOLVER_V1 = dedent(
    """\
    # ExecutableSolver（为当前 step 生成可执行代码）

    ## Input
    - background: $background
    - known_json: $known_json
    - required_dependencies: $required_dependencies
    - sub_steps_json: $sub_steps_json
    - step_number: $step_number
    - function_header: $function_header
    - return_line: $return_line

    ## Task
    请实现当前 step（step_number 对应的最后一个 sub_step），输出完整可执行 Python 代码。
    你可以参考 sub_steps_json 中的前序接口定义，但不要调用任何“未定义符号”：
    - 只有当某个函数/类在你本次输出代码中实现，或明确由运行环境注入时，才允许调用；
    - 对于 `step_number="e2e"`：不要假设任何前序步骤实现存在（即使 sub_steps_json 里出现了中间步骤的接口脚手架），所有被调用的 helper 必须在本次输出中定义，或直接在 `solve(...)` 内完成计算。
    请优先遵循 known_json 中的 premises/contract 约定（例如坐标系、单位、latvec 约定、cutoff 口径、容差等）。

    ## Hard constraints (must follow)
    - 只输出 Python 代码（用 ```python 代码块包裹）。
    - 必须实现 function_header 指定的函数/类，并遵守 return_line 的语义。
    - 不要输出任何测试代码、示例、解释或额外文本。
    - 不要进行文件读写或网络请求；仅使用 required_dependencies 中允许的库。
    """
)


EXECUTABLE_SOLVER_V1_EN = dedent(
    """\
    # ExecutableSolver (generate executable code for the current step)

    ## Input
    - background: $background
    - known_json: $known_json
    - required_dependencies: $required_dependencies
    - sub_steps_json: $sub_steps_json
    - step_number: $step_number
    - function_header: $function_header
    - return_line: $return_line

    ## Task
    Implement the current step (the tail sub_step) and output runnable Python code.
    You may reference previous step interfaces in sub_steps_json, but do NOT call any undefined symbols:
    - Only call a function/class if it is implemented in your output code, or is explicitly injected by the runtime.
    - When `step_number="e2e"`: do not assume any prior-step implementations exist (even if sub_steps_json contains interface scaffolds). Define every helper you call in this output, or implement everything directly inside `solve(...)`.
    Prefer to follow the premises/contract conventions in known_json (e.g., coordinate frame, units, latvec convention, cutoff policy, tolerance).

    ## Hard constraints (must follow)
    - Output Python code only, wrapped in ```python.
    - Must implement the function/class specified by function_header and respect return_line semantics.
    - Do NOT output tests, examples, explanations, or extra text.
    - No file I/O or network access; only use libraries in required_dependencies.
    """
)
