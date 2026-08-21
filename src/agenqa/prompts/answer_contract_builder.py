"""Answer Contract Builder role prompt."""

from __future__ import annotations

from textwrap import dedent

from agenqa.domain.contracts.answer_contract_builder_schema import answer_contract_builder_output_schema_text
from .common import (
    COMMON_ANSWER_CONTRACT_MODEL,
    COMMON_ANSWER_CONTRACT_MODEL_EN,
    COMMON_WORLD_CONTRACT_MODEL,
    COMMON_WORLD_CONTRACT_MODEL_EN,
)


__all__ = [
    "ANSWER_CONTRACT_BUILDER_V1",
    "ANSWER_CONTRACT_BUILDER_V1_EN",
]


ANSWER_CONTRACT_BUILDER_V1 = dedent(
    f"""\
    # Answer Contract Builder（答案契约提炼）

    ## Input
    - step: $step
    - question_type: $question_type
    - question: $question
    - answer: $answer

    {COMMON_WORLD_CONTRACT_MODEL.text}

    {COMMON_ANSWER_CONTRACT_MODEL.text}

    ## 你的任务
    - 阅读给定的格式化题目 (Question) 以及对应答案 (Answer)。
    - 你当前不是在定义题目语义，而是在提炼 Derivation 的答案接受层。
    - 当前实现限制下：
      - 如果题目是推导类题目（Derivation），输出 `answer_style / answer_semantics / support_witness`
      - 如果题目不是推导题（如纯数值题、选择题），返回空对象/空数组

    ## 当前 v2 工作含义
    - `answer_style`
      - 只表达 final answer 应怎么写，例如：
        - 是否必须 `\\boxed{{...}}`
        - 是否要求 `single_expression` / `single_equation` / `single_inequality`
        - 是否只输出最终结论
    - `answer_semantics`
      - 只表达在题目语义已经固定之后，什么 final answer 算可接受，例如：
        - `answer_object`
        - `acceptance_mode`
        - `branch_policy`
        - `allowed_symbols`
        - `required_qualifiers`
        - `equivalence_rules`
    - `support_witness`
      - 只表达 judge-side 的局部辅助 witness，每项形如 `{{type, statement}}`
      - 可用类型：`branch` / `boundary` / `signature` / `dependency` / `equivalence_cue`

    ## 边界要求
    - 你不能重新定义题目中的函数、对象、参数顺序、边界规则或范式。
    - 若某条规则改变后会让题目本身变成另一道题，它不属于 answer contract，而应属于 world contract / 题面显式化。
    - 尤其不要把以下 Type1/L3 语义补洞下沉为 answer contract：
      - 函数签名本身
      - 参数顺序本身
      - 边界规则本身（如 `>` vs `>=`）
      - 操作符/对象本体语义

    ## 提取规则
    - **`answer_style.boxed`**：题目或答案中是否要求输出被 LaTeX 的 `\\boxed{{...}}` 包裹。
    - **`answer_style.form`**：若题目明确要求单表达式/单方程/单不等式/集合/tuple，请填写最贴切的一个；否则可留空。
    - **`answer_style.rendering_notes`**：只放轻量写法要求，例如“只输出最终结论”。
    - **`answer_semantics.answer_object`**：judge 正在比较的 final answer 对象类型，例如 `symbolic_expr` / `equation` / `inequality` / `set` / `tuple`。
    - **`answer_semantics.acceptance_mode`**：默认偏 `exact`；只有题面真的允许近似时才填写 `approx` 或 `either`。
    - **`answer_semantics.branch_policy`**：仅在题面真的允许多分支或要求完整列举时填写。
    - **`answer_semantics.allowed_symbols`**：若题目要求答案“用 ... 表示”（in terms of $x$, $y$ 等），将其中数学符号原样提取为字符串数组。
    - **`answer_semantics.required_qualifiers`**：仅填写 final claim 中不能丢的限定，例如 `closed_form`、`branch_description`、`boundary_condition`。
    - **`answer_semantics.equivalence_rules`**：只填轻量 final-claim 级别规则，例如 `algebraic_rewrite_ok`、`branch_collapse_not_ok`、`boundary_drop_not_ok`。
    - **`support_witness`**：仅在 final claim 单独看不够稳时使用；不要把它写成完整 proof protocol。

    ## 保守性要求
    - 只有当 Question / Answer 已显式要求，或强约束推出时，才填写字段。
    - 不要为了让 judge 更方便而编造题面没有承诺的规则。
    - 不要把 Type1 语义问题错误地下沉为 answer contract。

    ## 输出格式（JSON）
    仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段为：
    - JSON 字符串内若需出现双引号（"），必须转义为 \\\"，不可输出裸引号。
    {{
      "{answer_contract_builder_output_schema_text('zh')}"
    }}
    示例：
    {{
      "answer_style": {{"boxed": true, "form": "single_expression", "rendering_notes": []}},
      "answer_semantics": {{
        "answer_object": "symbolic_expr",
        "acceptance_mode": "exact",
        "branch_policy": {{"allow_branches": false, "require_complete_enumeration": false}},
        "allowed_symbols": ["x", "y"],
        "required_qualifiers": ["closed_form"],
        "equivalence_rules": ["algebraic_rewrite_ok"]
      }},
      "support_witness": []
    }}
    """
)


ANSWER_CONTRACT_BUILDER_V1_EN = dedent(
    f"""\
    # Answer Contract Builder (Extract answer-contract constraints)

    ## Input
    - step: $step
    - question_type: $question_type
    - question: $question
    - answer: $answer

    {COMMON_WORLD_CONTRACT_MODEL_EN.text}

    {COMMON_ANSWER_CONTRACT_MODEL_EN.text}

    ## Your task
    - Read the finalized Question and its Answer.
    - You are not defining task semantics; you are extracting the answer-acceptance layer for Derivation.
    - Under the current implementation constraint:
      - if the question is a derivation task (Derivation), output `answer_style / answer_semantics / support_witness`
      - if it is not a derivation task (e.g. Numeric, MCQ), return empty objects / empty array

    ## What the v2 payload means
    - `answer_style`
      - how the final answer should be written
    - `answer_semantics`
      - what final answer counts as acceptable once task semantics are already fixed
    - `support_witness`
      - small judge-side local witnesses, each shaped like `{{type, statement}}`
      - valid types: `branch`, `boundary`, `signature`, `dependency`, `equivalence_cue`

    ## Boundary rules
    - You must not redefine task semantics such as function signatures, argument order, boundary rules, operator semantics, or paradigm choice.
    - If changing a rule would make the problem itself become a different problem, that rule belongs to world contract / question-side clarification, not answer contract.

    ## Extraction Rules
    - **`answer_style.boxed`**: whether the final output must be wrapped in LaTeX `\\boxed{{...}}`.
    - **`answer_style.form`**: use the closest label only when the task clearly requires it: `single_expression`, `single_equation`, `single_inequality`, `set`, or `tuple`.
    - **`answer_style.rendering_notes`**: lightweight rendering-only notes such as "final answer only".
    - **`answer_semantics.answer_object`**: the final answer object the judge compares, e.g. `symbolic_expr`, `equation`, `inequality`, `set`, `tuple`.
    - **`answer_semantics.acceptance_mode`**: prefer `exact` by default; use `approx` or `either` only when the task really permits approximation.
    - **`answer_semantics.branch_policy`**: fill only when the task explicitly allows multiple branches or requires complete enumeration.
    - **`answer_semantics.allowed_symbols`**: if the answer must be written in terms of specific symbols, extract them exactly.
    - **`answer_semantics.required_qualifiers`**: only include qualifiers that the final claim must preserve, such as `closed_form`, `branch_description`, or `boundary_condition`.
    - **`answer_semantics.equivalence_rules`**: only lightweight final-claim rules such as `algebraic_rewrite_ok`, `branch_collapse_not_ok`, or `boundary_drop_not_ok`.
    - **`support_witness`**: use only when the final claim alone is too fragile for stable judging; do not turn it into a full proof protocol.

    ## Conservatism
    - Only fill a field if it is explicitly required or strongly implied by the finalized Question / Answer.
    - Do not invent rules just to make judging easier.
    - Do not push Type1 semantic issues down into answer contract.

    ## Output format (JSON)
    Output one JSON object only, wrapped in a ```json code block. Top-level fields:
    - If a JSON string needs a double quotes ("), it must be escaped as \\\" (no bare quotes).
    {{
      "{answer_contract_builder_output_schema_text('en')}"
    }}
    Example:
    {{
      "answer_style": {{"boxed": true, "form": "single_expression", "rendering_notes": []}},
      "answer_semantics": {{
        "answer_object": "symbolic_expr",
        "acceptance_mode": "exact",
        "branch_policy": {{"allow_branches": false, "require_complete_enumeration": false}},
        "allowed_symbols": ["x", "y"],
        "required_qualifiers": ["closed_form"],
        "equivalence_rules": ["algebraic_rewrite_ok"]
      }},
      "support_witness": []
    }}
    """
)
