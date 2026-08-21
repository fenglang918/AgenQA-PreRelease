"""ExecutableTestInputs role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.prompts.common import (
    COMMON_EXECUTABLE_TEST_INPUTS_SCHEMA,
    COMMON_EXECUTABLE_TEST_INPUTS_SCHEMA_EN,
)

__all__ = [
    "EXECUTABLE_TEST_INPUTS_V1",
    "EXECUTABLE_TEST_INPUTS_V1_EN",
]


EXECUTABLE_TEST_INPUTS_V1 = dedent(
    f"""\
    # ExecutableTestInputs（生成当前 step 的测试输入）

    ## Input
    - function_header: $function_header
    - return_line: $return_line
    - step_description: $step_description
    - step_background: $step_background

    ## Task
    为当前 step 生成一组**可 JSON 序列化**的输入样例，用于执行 golden 代码并派生测试断言。

    ## Hard constraints (must follow)
    - 只输出小规模、可序列化输入；避免超长数组/超大矩阵/外部文件依赖。
    - 每个 case 用 JSON 表示：{{"args": [...], "kwargs": {{...}}}}；kwargs 可为空对象。
    - args/kwargs 必须与 function_header 的签名对齐；若要传命名参数，放在 kwargs 里（不要把“参数名→值”的 dict 塞进 args 里）。
    - 输出必须是严格 JSON（用 ```json 代码块包裹），字段如下：
{indent(COMMON_EXECUTABLE_TEST_INPUTS_SCHEMA.text, "    ")}
    """
)


EXECUTABLE_TEST_INPUTS_V1_EN = dedent(
    f"""\
    # ExecutableTestInputs (generate test inputs for the current step)

    ## Input
    - function_header: $function_header
    - return_line: $return_line
    - step_description: $step_description
    - step_background: $step_background

    ## Task
    Generate a small set of **JSON-serializable** input cases for the current step.

    ## Hard constraints (must follow)
    - Only small, serializable inputs; avoid huge arrays/matrices and external files.
    - Each case must be a JSON object: {{"args": [...], "kwargs": {{...}}}}; kwargs may be empty.
    - args/kwargs must match the function_header signature; put named parameters in kwargs (do not pass a "param->value" dict as a positional arg).
    - Output must be strict JSON wrapped in ```json with fields:
{indent(COMMON_EXECUTABLE_TEST_INPUTS_SCHEMA_EN.text, "    ")}
    """
)
