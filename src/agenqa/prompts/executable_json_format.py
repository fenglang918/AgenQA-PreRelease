"""ExecutableJsonFormat role prompt (Python style).

This role is a small, structured "format/repair" stage for executable track:
it converts a possibly non-conforming model output into a strict JSON object.
"""

from __future__ import annotations

from textwrap import dedent

__all__ = [
    "EXECUTABLE_JSON_FORMAT_V1",
    "EXECUTABLE_JSON_FORMAT_V1_EN",
]


EXECUTABLE_JSON_FORMAT_V1 = dedent(
    """\
    # ExecutableJsonFormat（结构化输出格式化/修复）

    ## Input
    - task_name: $task_name
    - required_keys_json: $required_keys_json
    - parse_error: $parse_error
    - original_prompt: $original_prompt
    - raw_output: $raw_output

    ## Task
    将 raw_output 修复/整理为**严格 JSON**对象，用于后续解析与流水线执行。

    ## Hard constraints (must follow)
    - 只输出一个 JSON 对象文本（不要 ```json 代码块，不要任何解释文字）。
    - 输出 JSON 对象必须包含 required_keys_json 中列出的所有 key，且不要输出其它 key。
    - 保持语义一致：不要编造新内容；仅做格式化、字段对齐、转义修复、结构修复。
    - 如果 raw_output 明显缺失关键信息，无法在不编造的情况下补齐 required keys，则输出：INCOMPLETE
    """
)


EXECUTABLE_JSON_FORMAT_V1_EN = dedent(
    """\
    # ExecutableJsonFormat (structured output format/repair)

    ## Input
    - task_name: $task_name
    - required_keys_json: $required_keys_json
    - parse_error: $parse_error
    - original_prompt: $original_prompt
    - raw_output: $raw_output

    ## Task
    Repair/normalize raw_output into a **strict JSON object** for downstream parsing.

    ## Hard constraints (must follow)
    - Output ONLY one JSON object text (no ```json fences, no commentary).
    - The JSON object MUST contain all keys in required_keys_json, and MUST NOT contain any other keys.
    - Preserve meaning: do not invent new content; only fix formatting/escaping/structure/field alignment.
    - If the input is clearly missing critical information and cannot be repaired without inventing content, output exactly: INCOMPLETE
    """
)
