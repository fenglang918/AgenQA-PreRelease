"""Format role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent

from agenqa.domain.format_schema import format_output_schema_text
from .common import (
    COMMON_ANSWER_SCHEMA,
    COMMON_ANSWER_SCHEMA_EN,
    COMMON_QUESTION_TYPES,
    COMMON_QUESTION_TYPES_EN,
)

__all__ = [
    "FORMAT_V1",
    "FORMAT_V1_TAGGED",
    "FORMAT_V1_FIRST_STEP",
    "FORMAT_V1_TAGGED_FIRST_STEP",
    "FORMAT_V1_EN",
    "FORMAT_V1_TAGGED_EN",
    "FORMAT_V1_FIRST_STEP_EN",
    "FORMAT_V1_TAGGED_FIRST_STEP_EN",
]


FORMAT_V1 = dedent(
    f"""\
    # Format（题目格式化与自检）

    ## Input
    - prev_step: $prev_step
    - step: $step
    - question_type: $question_type
    - draft_json: $draft_json

    ## 目标
    基于 draft_json 生成最终 Question/Solution/Answer，并做结构化自检。

    ## 要求
    - 输出中的所有自然语言内容（Question/Solution）必须使用中文；Answer 保持题型要求的 LaTeX `\\boxed{{...}}`。
    - Now QA 允许显式表达对前序结论的依赖关系（例如“使用上一步的结论…”），但不要出现内部实现指针（如 history/fact_bank/step_certs/premise_bank/known_0）。
    - `draft_json` 中的 `world_contract` 与 `Question` 是分离字段：下游会把 `Question + World Contract` 一起交给 solver。不要把完整的 `World Contract` / `Answer Requirements` 块重复写回 `Question`。
    - `WorldContract` 是独立输出字段：用于承载给 solver 的附加 contract 文本块。`Question` 只保留题目主体；若没有额外 contract 需要显式输出，`WorldContract` 可为空字符串。
    - Solution 用 S1/S2/...，可复现推理链。
    - Answer 使用 \\boxed{{...}}。
    - 若 question_type 为 Derivation：最终 solver-visible 交付（`Question + World Contract`）必须主动消歧，而不是把关键约定留给 judge 猜。
      具体要求：
      1. 若目标量/函数的符号形式可能有歧义，必须在最终 solver-visible 交付中钉死其签名或参数形式（例如说明是 `\\phi(\\Delta t)` 还是 `\\phi(t_1,t_2)`）。
      2. 若会使用多参数函数/算子，必须在最终 solver-visible 交付中钉死参数顺序（例如“其中 `mem(m,s)` 按此参数顺序定义”）。
      3. 若题目涉及阈值、不等式、可行域或边界条件，必须在最终 solver-visible 交付中钉死严格性（例如“这里的 at least 对应 `\\ge`”）。
      4. 必须保证最终 solver-visible 交付明确答案形式要求；若该要求已在 `draft_json.world_contract` / 独立 contract 中表达，不要再把它重复写回 `Question`。
      5. 若要求“化简/闭式/单表达式”，必须在最终 solver-visible 交付中明确，而不是只在 Solution 里体现。
    - 若 question_type 为 Numeric：只要求 Answer 是一个 LaTeX `\\boxed{...}` 数值；不要在 `Question` 中编造/猜测误差阈值（容差口径应留在独立 world_contract/L4 answer-output spec 中，由 Numeric-oracle 工具链注入并落盘）。
    - 结构化自检时，若上述 Derivation 消歧信息在最终 solver-visible 交付中仍缺失，必须将 `validation_passed=false`，并在 `validation_errors` 中明确指出缺失的是“符号签名 / 参数顺序 / 边界严格性 / 答案形式要求”中的哪一类。

    ## 题型与答案规范
    {COMMON_QUESTION_TYPES.text}
    {COMMON_ANSWER_SCHEMA.text}

    ## 输出格式（JSON）
    仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段为：
    - JSON 字符串内若需出现双引号（"），必须转义为 \\\"，不可输出裸引号。
    {format_output_schema_text()}
    """
)


FORMAT_V1_TAGGED = dedent(
    f"""\
    # Format（题目格式化与自检）- Tagged

    ## Input
    - prev_step: $prev_step
    - step: $step
    - question_type: $question_type
    - draft_json: $draft_json

    ## 输出格式（带字段标记的纯文本）
    [Step]\n$step\n[/Step]\n
    [Question]\n...\n[/Question]\n
    [WorldContract]\n...\n[/WorldContract]\n
    [Solution]\n...\n[/Solution]\n
    [Answer]\n...\n[/Answer]\n
    [validation_passed]\ntrue/false\n[/validation_passed]\n
    [validation_errors]\n- error1\n- error2\n[/validation_errors]

    约束：
    - [Step] 必须只包含一个整数，且等于输入 step（不要写解释性文字）。
    - [validation_passed] 必须只包含 true/false。
    - `draft_json.world_contract` 与 [Question] 是分离字段；下游会把二者拼接给 solver。不要把完整的 World Contract / Answer Requirements 块重复写回 [Question]。
    - [WorldContract] 是独立输出字段，用于承载 solver-visible contract 文本。保持 [Question] 为题目主体；若无额外 contract，可令 [WorldContract] 为空。
    - 若 question_type 为 Derivation：最终 solver-visible 交付必须主动消歧：
      1. 若目标量/函数签名可能歧义，必须在 Question 或 world_contract 中显式写清；
      2. 若存在多参数函数/算子，必须在 Question 或 world_contract 中显式写清参数顺序；
      3. 若涉及阈值/不等式/边界条件，必须在 Question 或 world_contract 中显式写清严格性；
      4. 必须保证答案格式要求在最终 solver-visible 交付中明确；若 world_contract 已表达，不要在 [Question] 中重复抄写；
      5. 若以上任一项缺失，[validation_passed] 必须为 false，并在 [validation_errors] 中指出缺了哪一类。
    - 若 question_type 为 Numeric：只要求 [Answer] 是一个 LaTeX `\\boxed{...}` 数值；不要在 [Question] 中编造/猜测误差阈值（容差口径应留在独立 world_contract/L4 answer-output spec 中）。
    """
)


FORMAT_V1_EN = dedent(
    f"""\
    # Format (Finalize Question)

    ## Input
    - prev_step: $prev_step
    - step: $step
    - question_type: $question_type
    - draft_json: $draft_json

    ## Goal
    Produce the final Question/Solution/Answer and self-check.

    ## Constraints
    - All natural-language content (Question/Solution) MUST be in English; Answer must follow the required LaTeX `\\boxed{{...}}` format.
    - In Now QA, it is OK to explicitly reference prior-step results (e.g., “using the previous step’s result ...”), but do not mention internal system pointers such as history/fact_bank/step_certs/premise_bank/known_0.
    - `world_contract` inside `draft_json` is separate from `Question`: downstream concatenates `Question + World Contract` for solver consumption. Do not duplicate a full `World Contract` / `Answer Requirements` block inside `Question`.
    - `WorldContract` is a separate output field: it carries the solver-visible contract block, while `Question` stays focused on the core problem statement. If no extra contract block is needed, `WorldContract` may be an empty string.
    - Solution must be reproducible (S1/S2/...)
    - Answer must be wrapped in \\boxed{{...}}.
    - If question_type is Derivation: the final solver-visible deliverable (`Question + World Contract`) must resolve notation ambiguity instead of leaving it implicit.
      Requirements:
      1. If the target quantity/function signature could be ambiguous, explicitly state the intended signature in the final solver-visible deliverable.
      2. If any multi-argument function/operator is used, explicitly state the parameter order in the final solver-visible deliverable.
      3. If thresholds, inequalities, feasibility regions, or boundary conditions matter, explicitly state the strictness in the final solver-visible deliverable.
      4. Make sure the answer-format requirement is explicit in the final solver-visible deliverable; if it already exists in `draft_json.world_contract` / separate contract text, do not restate it verbatim in `Question`.
      5. If a simplified / closed-form / single-expression answer is required, say so somewhere in the final solver-visible deliverable.
    - If question_type is Numeric: only require the Answer to be a single LaTeX `\\boxed{...}` numeric value; do NOT invent tolerance thresholds inside `Question` (the Numeric-oracle toolchain should persist them into separate world-contract/L4 text).
    - In structural self-check, if any required Derivation disambiguation is still missing in the final solver-visible deliverable, set `validation_passed=false` and list the missing category explicitly in `validation_errors` (`symbol signature`, `parameter order`, `boundary strictness`, or `answer-form requirement`).

    ## Question type & answer rules
    {COMMON_QUESTION_TYPES_EN.text}
    {COMMON_ANSWER_SCHEMA_EN.text}

    ## Output format (JSON)
    Output one JSON object only, wrapped in a ```json code block. Top-level fields:
    - If a JSON string needs a double quote ("), it must be escaped as \\\" (no bare quotes).
    {format_output_schema_text('en')}
    """
)


FORMAT_V1_TAGGED_EN = dedent(
    """\
    # Format (Finalize Question) - Tagged

    ## Input
    - prev_step: $prev_step
    - step: $step
    - question_type: $question_type
    - draft_json: $draft_json

    ## Output format (tagged)
    [Step]\n$step\n[/Step]\n
    [Question]\n...\n[/Question]\n
    [WorldContract]\n...\n[/WorldContract]\n
    [Solution]\n...\n[/Solution]\n
    [Answer]\n...\n[/Answer]\n
    [validation_passed]\ntrue/false\n[/validation_passed]\n
    [validation_errors]\n- error1\n- error2\n[/validation_errors]

    Constraints:
    - [Step] must contain only one integer equal to the input step (no extra words).
    - [validation_passed] must contain only true/false.
    - `draft_json.world_contract` and [Question] are separate channels; downstream concatenates them for solver consumption. Do not duplicate a full World Contract / Answer Requirements block inside [Question].
    - [WorldContract] is a separate output field for solver-visible contract text. Keep [Question] as the core problem statement; if no extra contract block is needed, [WorldContract] may be empty.
    - If question_type is Derivation: the final solver-visible deliverable must resolve notation ambiguity explicitly:
      1. state the intended symbol/function signature in Question or world_contract when it could be ambiguous;
      2. state the parameter order for any multi-argument function/operator in Question or world_contract;
      3. state inequality / threshold / boundary strictness when relevant in Question or world_contract;
      4. ensure answer-format requirements are explicit in the final solver-visible deliverable; do not restate them in [Question] if world_contract already covers them;
      5. if any of these are missing, [validation_passed] must be false and [validation_errors] must name the missing category.
    - If question_type is Numeric: only require the [Answer] to be a single LaTeX `\\boxed{...}` numeric value; do NOT invent tolerance thresholds inside [Question] (the Numeric-oracle toolchain should persist them into separate world-contract/L4 text).
    """
)

# First-step variants (alias to the same content in v2)
FORMAT_V1_FIRST_STEP = FORMAT_V1
FORMAT_V1_TAGGED_FIRST_STEP = FORMAT_V1_TAGGED
FORMAT_V1_FIRST_STEP_EN = FORMAT_V1_EN
FORMAT_V1_TAGGED_FIRST_STEP_EN = FORMAT_V1_TAGGED_EN
