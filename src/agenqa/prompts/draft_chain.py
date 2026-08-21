"""DraftChain role prompt (Python style).

This prompt is question-type specific. The pipeline should pick the minimal
sufficient variant for the current QuestionType.
"""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.domain.draft_chain_schema import draft_chain_output_schema_text
from .common import COMMON_WORLD_CONTRACT_GUIDANCE, COMMON_WORLD_CONTRACT_GUIDANCE_EN

__all__ = [
    "DRAFT_CHAIN_MCQ",
    "DRAFT_CHAIN_DERIVATION",
    "DRAFT_CHAIN_NUMERIC",
    "DRAFT_CHAIN_MCQ_EN",
    "DRAFT_CHAIN_DERIVATION_EN",
    "DRAFT_CHAIN_NUMERIC_EN",
    "DRAFT_CHAIN_MCQ_CALC",
    "DRAFT_CHAIN_DERIVATION_CALC",
    "DRAFT_CHAIN_NUMERIC_CALC",
    "DRAFT_CHAIN_MCQ_CALC_EN",
    "DRAFT_CHAIN_DERIVATION_CALC_EN",
    "DRAFT_CHAIN_NUMERIC_CALC_EN",
    "get_draft_chain_prompt",
]


_BASE_ZH = dedent(
    f"""\
    # DraftChain（题目递进草稿：__QTYPE_LABEL__）

    ## Input
    - chain_view_json: $chain_view_json
    - prev_step: $prev_step
    - step: $step
    - question_type: $question_type
    - expected_primary_fact_id: $expected_primary_fact_id
    - director_notes: $director_notes

    ## 目标
    生成“显式草稿题面”并显式结构化推理路径：
    - 用 `subtasks` / `dependencies` 明确列出需要先推导的中间量，以及如何依赖它们得到最终答案；
    - `draft_question_explicit` 将直接作为本步 Now QA（edge QA）的题面来源（下游仅做 Format 整理，不再有额外的“改写隐藏复用”步骤）。
      - 允许用自然语言显式表达对前序结论的依赖（如“使用上一步的结论…”），但不要出现内部实现指针（history/fact_bank/step_certs/known_0 等）。
      - 不要在题面中写 fact_id；也尽量避免大段复制粘贴前序答案造成冗余或不一致。
      - 题面仍需保持 Answer 不泄露、结论唯一。

    ## 语言（重要）
    - 输出中的所有自然语言内容（尤其是 `draft_question_explicit` / `draft_solution_outline` / `reuse_plan`）必须使用中文。
    - 数学符号/LaTeX/代码块不受语言限制；JSON 键名必须保持英文（由 schema 决定）。

    ## 单步递进（关键）
    本步只设计“一条新边”（从 step-1 推到 step），不要在一个 step 内展开很长的多跳推导链。

    - 当 step>=2 时：`subtasks` **必须恰好 2 个**，且建议 id 固定为：`sub_prev`、`sub_step`。
      - `sub_prev`: 重建/引用上一步 key_fact（`expected_primary_fact_id`）作为必要中间量；
      - `sub_step`: 在 `sub_prev` 的基础上再推进一步，得到本步最终答案；
      - `dependencies`: `sub_step` 依赖 `sub_prev`；`final_subtask_id` 必须是 `sub_step`。
    - 当 step=1 时：`subtasks` 也必须保持粗粒度（1–2 个），不要拆成很多小步。
    - 不要把完整推导过程拆成很多 `subtasks`；推导细节写在 `draft_solution_outline` 即可。

    ## 关键约束
    - step>=2 时：`expected_primary_fact_id` 是唯一权威输入。你必须输出 `primary_required_fact_id="$expected_primary_fact_id"`（不得选择其他 fact_id），且 `required_fact_ids` 必须包含 `$expected_primary_fact_id`。
    - 如需额外复用更早的 fact（例如 step-2 的方法/公式），可以把这些 fact_id **追加**到 `required_fact_ids` / `reuse_plan`；但 `primary_required_fact_id` 仍必须保持为 `$expected_primary_fact_id`。
    - `reuse_plan` 必须包含一条说明 `$expected_primary_fact_id` 如何被复用（不要只写其他 fact）。
    - step=1 时：`required_fact_ids` 置为空数组，`primary_required_fact_id` 置为空字符串。
    - 题面必须可解、结论唯一；Answer 用 \\boxed{{...}}。
__MODE_CONSTRAINTS__

    ## Director Notes（重要，但非绝对）
    - `director_notes` 提供偏好/风险提示（例如“avoid X / try Y”）。默认应当优先遵循，避免无声忽略。
    - 若你认为无法遵循（例如会破坏可解性、与单步递进/题型约束冲突），允许偏离，但必须在 `draft_solution_outline` 中用 1–2 句说明原因与权衡。

    ## 题型硬约束（仅限本题型）
__QTYPE_RULES__

    ## 答案格式
__ANSWER_FORMAT__

    ## 输出格式（JSON）
    仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段为：
__OUTPUT_SCHEMA__
    """
)


_BASE_EN = dedent(
    f"""\
    # DraftChain (chained draft: __QTYPE_LABEL__)

    ## Input
    - chain_view_json: $chain_view_json
    - prev_step: $prev_step
    - step: $step
    - question_type: $question_type
    - expected_primary_fact_id: $expected_primary_fact_id
    - director_notes: $director_notes

    ## Goal
    Produce an explicit chained draft with a structured subtask sequence:
    - Use `subtasks` / `dependencies` to explicitly spell out the intermediate results and their dependency graph.
    - `draft_question_explicit` will be used directly as the source of the Now QA (edge QA) question (downstream only runs Format; there is no extra “rewrite to hide reuse” step).
      - It is OK to explicitly reference prior-step results in natural language (e.g., “using the previous step’s result ...”), but do not mention internal system pointers (history/fact_bank/step_certs/known_0).
      - Do not mention fact_id in the question text; avoid copy-pasting large chunks of prior answers that can become redundant or inconsistent.
      - The question must still avoid leaking the Answer and have a unique conclusion.

    ## Language (important)
    - All natural-language content in the output (especially `draft_question_explicit` / `draft_solution_outline` / `reuse_plan`) MUST be in English.
    - Math symbols/LaTeX/code blocks are language-agnostic; JSON keys must remain as defined by the schema.

    ## Single-edge constraint (critical)
    This step must add exactly one new “edge” from step-1 to step; do not expand a long multi-hop derivation inside one step.

    - If step>=2: `subtasks` must have **exactly 2** coarse-grained items, with recommended ids `sub_prev` and `sub_step`.
      - `sub_prev`: restate/reconstruct the previous step's key_fact (`expected_primary_fact_id`) as the required intermediate.
      - `sub_step`: one-step extension that uses `sub_prev` to reach this step’s final answer.
      - `dependencies`: `sub_step` depends on `sub_prev`; `final_subtask_id` must be `sub_step`.
    - If step==1: keep `subtasks` coarse-grained (1–2 items); do not break the full derivation into many subtasks.
    - Put derivation details in `draft_solution_outline`, not by inflating `subtasks`.

    ## Hard constraints
    - For step>=2: `expected_primary_fact_id` is the only authoritative input. You must output `primary_required_fact_id="$expected_primary_fact_id"` (never choose another fact_id), and `required_fact_ids` must include `$expected_primary_fact_id`.
    - If you also need older facts (e.g. step-2 methods), append them to `required_fact_ids` / `reuse_plan`, but keep `primary_required_fact_id` equal to `$expected_primary_fact_id`.
    - `reuse_plan` must include a line explaining how `$expected_primary_fact_id` is reused (do not list only other facts).
    - For step=1: set `required_fact_ids` to [] and `primary_required_fact_id` to "".
    - The question must be well-posed with a unique answer; wrap Answer in \\boxed{{...}}.
__MODE_CONSTRAINTS__

    ## Director Notes (important, not absolute)
    - `director_notes` carries preferences / risk hints (e.g., “avoid X / try Y”). Prefer to follow them by default; do not ignore silently.
    - If you decide to deviate (e.g., it would break well-posedness or conflict with the single-edge / question-type constraints), you may do so, but briefly explain the rationale/tradeoff in `draft_solution_outline` (1–2 sentences).

    ## Question-type constraints (this type only)
__QTYPE_RULES__

    ## Answer format
__ANSWER_FORMAT__

    ## Output format (JSON)
    Output one JSON object only, wrapped in a ```json code block. Top-level fields:
__OUTPUT_SCHEMA__
    """
)


_MCQ_RULES_ZH = dedent(
    """\
    - 题干必须明确写出“仅给出选项字母”（例如“请仅回答 A/B/C/D”）。
    - 必须提供至少 4 个互斥选项（A/B/C/D），且**有且仅有一个严格正确**。
    - 选项应有区分度：错误项必须来自常见混淆点/可解释的近似误差，而不是随机扰动。
    - 若涉及数值：在题干或选项中写清近似/舍入/单位口径，避免“看似不同但等价”的选项。
    """
).rstrip()

_MCQ_ANSWER_FORMAT_ZH = dedent(
    """\
    - `draft_answer` 中仅写唯一正确选项的字母（如 `\\\\boxed{A}`）。
    - 不要在 `draft_answer` 中附加解释、句子或额外符号（只给字母本身）。
    """
).rstrip()

_MCQ_RULES_EN = dedent(
    """\
    - The question must explicitly say “answer with the option letter only” (A/B/C/D).
    - Provide at least 4 mutually exclusive options, with **exactly one** strictly correct.
    - Distractors must be meaningful (common confusions / explainable approximation errors), not random noise.
    - If numeric values appear, state rounding/approximation/unit conventions to avoid equivalent options.
    """
).rstrip()

_MCQ_ANSWER_FORMAT_EN = dedent(
    """\
    - In `draft_answer`, output the unique correct option letter only (e.g., `\\\\boxed{A}`).
    - Do not add explanations/sentences in `draft_answer` (letter only).
    """
).rstrip()

_DERIVATION_RULES_ZH = dedent(
    """\
    - 必须是“推导/符号推理”题：最终答案为唯一可判定的解析式/符号结论。
    - 最终 solver-visible 交付（Question + World Contract）必须显式约束允许出现的符号集合（例如 `in terms of {a,b,\\gamma}`），且**不得引入新符号**。
    - 不要把 Derivation 退化成无推理的选择题；推导结构写在 subtasks/solution_outline。
    - Derivation 默认不应要求数值求值/小数近似/误差口径（例如“保留四位小数”“abs_tol=...”“给出精确数值”）；这类任务属于 Numeric。
    """
).rstrip()

_DERIVATION_ANSWER_FORMAT_ZH = dedent(
    """\
    - `draft_answer` 给出唯一可判定的解析表达式/符号结论，用 `\\\\boxed{...}` 包裹。
    - 不得引入未在最终 solver-visible 交付中定义的新符号；必要时通过 Question 或 World Contract 明确 “in terms of ...” 的允许符号集合。
    """
).rstrip()

_DERIVATION_RULES_EN = dedent(
    """\
    - Must be a derivation/symbolic reasoning question: the final answer is a uniquely checkable expression.
    - The final solver-visible deliverable (Question + World Contract) must explicitly constrain the allowed symbols (e.g., “in terms of {a,b,\\gamma}”) and must not introduce new symbols.
    - Do not degenerate Derivation into a trivial MCQ; keep the reasoning structure in subtasks/solution_outline.
    - By default, Derivation should not require numeric evaluation/decimal approximations/tolerance rules (e.g., “4 decimal places”, “abs_tol=...”); those belong to Numeric.
    """
).rstrip()

_DERIVATION_ANSWER_FORMAT_EN = dedent(
    """\
    - In `draft_answer`, give a uniquely checkable symbolic/analytic expression wrapped in `\\\\boxed{...}`.
    - Do not introduce new symbols not defined in the final solver-visible deliverable; specify the allowed symbols via Question or World Contract if needed ("in terms of ...").
    """
).rstrip()

_NUMERIC_RULES_ZH = dedent(
    """\

    ### 核心定义
    Numeric = 需要"建模 + 代码执行"才能可靠求解的**定量计算题**：
    - **Model（建模）**：需要建立物理/数学模型，将问题翻译成可计算形式
    - **Code（执行）**：需要代码执行才能可靠得到最终数值（避免 LLM 心算误差）
    - **Answer（数值）**：目标量可被唯一数值判定（带精度口径）

    ### 复杂度要求（鼓励）
    既然 solver 可以使用代码，Numeric 题应尽量考察以下能力之一（而非简单套公式代数值）：
    - **复合建模**：需要整合多个科学/数学概念才能得到答案
    - **数值方法**：积分/优化/迭代/采样（不是简单的算术运算）

    ### 精度与单位
    - 最终 solver-visible 交付必须写清单位/口径/近似（必要时），并且需要一个可机判的精度规则（绝对误差/相对误差/有效数字）。
      - 允许把"精度口径"写在 Question 或独立 World Contract 中；若需要让 Numeric-oracle 补全，请在 `draft_solution_outline` 中写清你希望的口径（abs/rel/sig_figs + 单位）。
      - 但**不能**留下"约等于/差不多"这类不可判定口径。
      - oracle_code 输出的数值必须与题干要求的**单位一致**。

    ### oracle_code 要求
    为避免出题阶段的心算/手算错误污染链路：本算子需要同时输出一段 **可执行的 `oracle_code`**（内部运行）来计算 ground-truth 数值，并给出容差口径。
    - `oracle_code` 必须是**确定性** Python，只允许标准库（math/decimal/fractions 等）；禁止网络与外部文件 I/O。
    - `oracle_code` 运行后 stdout 必须只输出一行 JSON：`{"value": <number>}`。
    - 你仍然需要写清 `draft_solution_outline`（推导与计算配方），但最终数值以 `oracle_code` 执行结果为准。
    - tool 工件（oracle_code/exec_payload/gt_value）不得写入题面或 premise_bank（内部会存到 step_certs 并对 LLM 视角脱敏）。
    """
).rstrip()

_NUMERIC_ANSWER_FORMAT_ZH = dedent(
    """\
    - `draft_answer` 给出可机判的数值结果，用 `\\\\boxed{...}` 包裹（只给数值，不附加句子）。
    - 误差/精度/单位口径应在最终 solver-visible 交付中明确声明；若采用分离式 contract，不要把整块重复写回题面。
    """
).rstrip()

_NUMERIC_RULES_EN = dedent(
    """\

    ### Core definition
    Numeric = a **quantitative computation** question that requires both "modeling + code execution" to solve reliably:
    - **Model**: build a physical/mathematical model and translate the problem into a computable form
    - **Code**: rely on deterministic code execution to obtain the final numeric value (avoid LLM mental-arithmetic errors)
    - **Answer**: the target must be uniquely numerically decidable (with a checkable precision convention)

    ### Complexity (encouraged)
    Since solvers can use code, Numeric questions should ideally test at least one of the following (not a trivial plug-and-chug):
    - **Composite modeling**: combine multiple scientific/mathematical concepts to derive the computation
    - **Numerical methods**: integration/optimization/iteration/sampling (not simple arithmetic)

    ### Precision and units
    - The final solver-visible deliverable must specify unit/conventions/approximations (as needed) AND a machine-checkable precision rule (abs/rel error or significant figures).
      - You may state the tolerance in the Question or in a separate World Contract block. If you want the Numeric-oracle to finalize it, specify your preference in `draft_solution_outline` (abs/rel/sig_figs + units).
      - Never leave vague “approximately / roughly” conventions that are not checkable.
      - The numeric value printed by `oracle_code` must use the **same unit** as required by the question.

    ### oracle_code requirements
    To avoid mental-arithmetic mistakes during question generation, this operator must also output an executable `oracle_code` (run internally) to compute the ground-truth numeric value, along with a tolerance convention.
    - `oracle_code` must be deterministic Python using stdlib only (math/decimal/fractions, etc.); no network or external file I/O.
    - The code must print exactly one JSON line to stdout: `{"value": <number>}`.
    - You should still provide a computationally explicit `draft_solution_outline`, but the final numeric value is determined by executing `oracle_code`.
    - Tool artifacts (oracle_code/exec_payload/gt_value) must not be leaked into the question or premise_bank (they are stored in step_certs and redacted for LLM views).
    """
).rstrip()

_NUMERIC_ANSWER_FORMAT_EN = dedent(
    """\
    - In `draft_answer`, output a machine-checkable numeric value wrapped in `\\\\boxed{...}` (value only; no extra sentences).
    - The precision/tolerance/unit convention must be explicitly stated in the final solver-visible deliverable; if a separate contract block is used, do not duplicate it verbatim in the question body.
    """
).rstrip()

_CALC_GUIDANCE_ZH = dedent(
    """\
    ## 难度侧重点（计算 + 推导）
    - 优先让 sub_step 引入“计算量显著”的推导（如代数消元/矩阵运算/级数展开/积分/极值/渐近/概率期望等），而不仅是概念性解释。
    - 难度优先来自可复现的复杂推导与计算；但务必基于论文背景（episode_seed 的 anchor；可选 subject/keywords + premise_bank），并结合必要的垂类知识点提升难度，避免脱离论文主题出“泛化玩具题”。
    - 若允许数值计算，尽量使用小整数/分数/根号等精确量，保证答案唯一并降低算错风险。
    """
).rstrip()

_CALC_GUIDANCE_EN = dedent(
    """\
    ## Difficulty focus (computation + derivation)
    - Prefer a computation-heavy sub_step (e.g., elimination/algebra, matrix ops, series expansions, integrals, extrema, asymptotics, expectations), not just conceptual explanations.
    - Prioritize difficulty from reproducible, computation-heavy derivation; but stay grounded in the paper context (episode_seed anchor; optional subject/keywords + premise_bank) and use necessary domain knowledge accordingly—avoid drifting into generic toy problems unrelated to the paper.
    - If numeric computation is allowed, prefer small exact values (integers/fractions/roots) to keep the answer unique and reduce arithmetic mistakes.
    """
).rstrip()


def _render(
    base: str,
    *,
    qtype_label: str,
    qtype_rules: str,
    answer_format: str,
    output_schema: str,
    calc_guidance: str | None = None,
    mode_constraints: str | None = None,
) -> str:
    out = (
        base.replace("__QTYPE_LABEL__", qtype_label)
        .replace("__QTYPE_RULES__", indent(qtype_rules, "    "))
        .replace("__ANSWER_FORMAT__", indent(answer_format, "    "))
        .replace("__OUTPUT_SCHEMA__", indent(output_schema, "    "))
    )
    out = out.replace("__MODE_CONSTRAINTS__", indent(mode_constraints, "    ") if mode_constraints else "")
    if calc_guidance and calc_guidance.strip():
        out = f"{out}\n\n{calc_guidance}\n"
    return out


DRAFT_CHAIN_MODE_DEFAULT_ZH = ""
DRAFT_CHAIN_MODE_DEFAULT_EN = ""

_WORLD_CONTRACT_CONSTRAINTS_ZH = COMMON_WORLD_CONTRACT_GUIDANCE.text

_WORLD_CONTRACT_CONSTRAINTS_EN = COMMON_WORLD_CONTRACT_GUIDANCE_EN.text

DRAFT_CHAIN_MCQ = _render(
    _BASE_ZH,
    qtype_label="MCQ",
    qtype_rules=_MCQ_RULES_ZH,
    answer_format=_MCQ_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(world_contract_policy="omit"),
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_ZH,
)
DRAFT_CHAIN_DERIVATION = _render(
    _BASE_ZH,
    qtype_label="Derivation",
    qtype_rules=_DERIVATION_RULES_ZH,
    answer_format=_DERIVATION_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(world_contract_policy="omit"),
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_ZH,
)
DRAFT_CHAIN_NUMERIC = _render(
    _BASE_ZH,
    qtype_label="Numeric",
    qtype_rules=_NUMERIC_RULES_ZH,
    answer_format=_NUMERIC_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(question_type="Numeric", world_contract_policy="omit"),
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_ZH,
)

DRAFT_CHAIN_MCQ_EN = _render(
    _BASE_EN,
    qtype_label="MCQ",
    qtype_rules=_MCQ_RULES_EN,
    answer_format=_MCQ_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", world_contract_policy="omit"),
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_EN,
)
DRAFT_CHAIN_DERIVATION_EN = _render(
    _BASE_EN,
    qtype_label="Derivation",
    qtype_rules=_DERIVATION_RULES_EN,
    answer_format=_DERIVATION_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", world_contract_policy="omit"),
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_EN,
)
DRAFT_CHAIN_NUMERIC_EN = _render(
    _BASE_EN,
    qtype_label="Numeric",
    qtype_rules=_NUMERIC_RULES_EN,
    answer_format=_NUMERIC_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", question_type="Numeric", world_contract_policy="omit"),
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_EN,
)

DRAFT_CHAIN_MCQ_WORLD_CONTRACT = _render(
    _BASE_ZH,
    qtype_label="MCQ",
    qtype_rules=_MCQ_RULES_ZH,
    answer_format=_MCQ_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(world_contract_policy="required"),
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_ZH,
)
DRAFT_CHAIN_DERIVATION_WORLD_CONTRACT = _render(
    _BASE_ZH,
    qtype_label="Derivation",
    qtype_rules=_DERIVATION_RULES_ZH,
    answer_format=_DERIVATION_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(world_contract_policy="required"),
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_ZH,
)
DRAFT_CHAIN_NUMERIC_WORLD_CONTRACT = _render(
    _BASE_ZH,
    qtype_label="Numeric",
    qtype_rules=_NUMERIC_RULES_ZH,
    answer_format=_NUMERIC_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(question_type="Numeric", world_contract_policy="required"),
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_ZH,
)

DRAFT_CHAIN_MCQ_WORLD_CONTRACT_EN = _render(
    _BASE_EN,
    qtype_label="MCQ",
    qtype_rules=_MCQ_RULES_EN,
    answer_format=_MCQ_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", world_contract_policy="required"),
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_EN,
)
DRAFT_CHAIN_DERIVATION_WORLD_CONTRACT_EN = _render(
    _BASE_EN,
    qtype_label="Derivation",
    qtype_rules=_DERIVATION_RULES_EN,
    answer_format=_DERIVATION_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", world_contract_policy="required"),
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_EN,
)
DRAFT_CHAIN_NUMERIC_WORLD_CONTRACT_EN = _render(
    _BASE_EN,
    qtype_label="Numeric",
    qtype_rules=_NUMERIC_RULES_EN,
    answer_format=_NUMERIC_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", question_type="Numeric", world_contract_policy="required"),
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_EN,
)

DRAFT_CHAIN_MCQ_CALC = _render(
    _BASE_ZH,
    qtype_label="MCQ",
    qtype_rules=_MCQ_RULES_ZH,
    answer_format=_MCQ_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(world_contract_policy="omit"),
    calc_guidance=_CALC_GUIDANCE_ZH,
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_ZH,
)
DRAFT_CHAIN_DERIVATION_CALC = _render(
    _BASE_ZH,
    qtype_label="Derivation",
    qtype_rules=_DERIVATION_RULES_ZH,
    answer_format=_DERIVATION_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(world_contract_policy="omit"),
    calc_guidance=_CALC_GUIDANCE_ZH,
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_ZH,
)
DRAFT_CHAIN_NUMERIC_CALC = _render(
    _BASE_ZH,
    qtype_label="Numeric",
    qtype_rules=_NUMERIC_RULES_ZH,
    answer_format=_NUMERIC_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(question_type="Numeric", world_contract_policy="omit"),
    calc_guidance=_CALC_GUIDANCE_ZH,
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_ZH,
)

DRAFT_CHAIN_MCQ_CALC_EN = _render(
    _BASE_EN,
    qtype_label="MCQ",
    qtype_rules=_MCQ_RULES_EN,
    answer_format=_MCQ_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", world_contract_policy="omit"),
    calc_guidance=_CALC_GUIDANCE_EN,
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_EN,
)
DRAFT_CHAIN_DERIVATION_CALC_EN = _render(
    _BASE_EN,
    qtype_label="Derivation",
    qtype_rules=_DERIVATION_RULES_EN,
    answer_format=_DERIVATION_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", world_contract_policy="omit"),
    calc_guidance=_CALC_GUIDANCE_EN,
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_EN,
)
DRAFT_CHAIN_NUMERIC_CALC_EN = _render(
    _BASE_EN,
    qtype_label="Numeric",
    qtype_rules=_NUMERIC_RULES_EN,
    answer_format=_NUMERIC_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", question_type="Numeric", world_contract_policy="omit"),
    calc_guidance=_CALC_GUIDANCE_EN,
    mode_constraints=DRAFT_CHAIN_MODE_DEFAULT_EN,
)

DRAFT_CHAIN_MCQ_WORLD_CONTRACT_CALC = _render(
    _BASE_ZH,
    qtype_label="MCQ",
    qtype_rules=_MCQ_RULES_ZH,
    answer_format=_MCQ_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(world_contract_policy="required"),
    calc_guidance=_CALC_GUIDANCE_ZH,
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_ZH,
)
DRAFT_CHAIN_DERIVATION_WORLD_CONTRACT_CALC = _render(
    _BASE_ZH,
    qtype_label="Derivation",
    qtype_rules=_DERIVATION_RULES_ZH,
    answer_format=_DERIVATION_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(world_contract_policy="required"),
    calc_guidance=_CALC_GUIDANCE_ZH,
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_ZH,
)
DRAFT_CHAIN_NUMERIC_WORLD_CONTRACT_CALC = _render(
    _BASE_ZH,
    qtype_label="Numeric",
    qtype_rules=_NUMERIC_RULES_ZH,
    answer_format=_NUMERIC_ANSWER_FORMAT_ZH,
    output_schema=draft_chain_output_schema_text(question_type="Numeric", world_contract_policy="required"),
    calc_guidance=_CALC_GUIDANCE_ZH,
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_ZH,
)

DRAFT_CHAIN_MCQ_WORLD_CONTRACT_CALC_EN = _render(
    _BASE_EN,
    qtype_label="MCQ",
    qtype_rules=_MCQ_RULES_EN,
    answer_format=_MCQ_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", world_contract_policy="required"),
    calc_guidance=_CALC_GUIDANCE_EN,
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_EN,
)
DRAFT_CHAIN_DERIVATION_WORLD_CONTRACT_CALC_EN = _render(
    _BASE_EN,
    qtype_label="Derivation",
    qtype_rules=_DERIVATION_RULES_EN,
    answer_format=_DERIVATION_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", world_contract_policy="required"),
    calc_guidance=_CALC_GUIDANCE_EN,
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_EN,
)
DRAFT_CHAIN_NUMERIC_WORLD_CONTRACT_CALC_EN = _render(
    _BASE_EN,
    qtype_label="Numeric",
    qtype_rules=_NUMERIC_RULES_EN,
    answer_format=_NUMERIC_ANSWER_FORMAT_EN,
    output_schema=draft_chain_output_schema_text("en", question_type="Numeric", world_contract_policy="required"),
    calc_guidance=_CALC_GUIDANCE_EN,
    mode_constraints=_WORLD_CONTRACT_CONSTRAINTS_EN,
)


def get_draft_chain_prompt(*, question_type: str, use_en: bool, calc: bool, world_contract: bool = False) -> str:
    qt = str(question_type or "").strip().lower()
    if qt == "mcq":
        if world_contract:
            return (
                (DRAFT_CHAIN_MCQ_WORLD_CONTRACT_CALC_EN if use_en else DRAFT_CHAIN_MCQ_WORLD_CONTRACT_CALC)
                if calc
                else (DRAFT_CHAIN_MCQ_WORLD_CONTRACT_EN if use_en else DRAFT_CHAIN_MCQ_WORLD_CONTRACT)
            )
        return (
            (DRAFT_CHAIN_MCQ_CALC_EN if use_en else DRAFT_CHAIN_MCQ_CALC)
            if calc
            else (DRAFT_CHAIN_MCQ_EN if use_en else DRAFT_CHAIN_MCQ)
        )
    if qt == "numeric":
        if world_contract:
            return (
                (DRAFT_CHAIN_NUMERIC_WORLD_CONTRACT_CALC_EN if use_en else DRAFT_CHAIN_NUMERIC_WORLD_CONTRACT_CALC)
                if calc
                else (DRAFT_CHAIN_NUMERIC_WORLD_CONTRACT_EN if use_en else DRAFT_CHAIN_NUMERIC_WORLD_CONTRACT)
            )
        return (
            (DRAFT_CHAIN_NUMERIC_CALC_EN if use_en else DRAFT_CHAIN_NUMERIC_CALC)
            if calc
            else (DRAFT_CHAIN_NUMERIC_EN if use_en else DRAFT_CHAIN_NUMERIC)
        )
    # Default to Derivation if unknown.
    if world_contract:
        return (
            (DRAFT_CHAIN_DERIVATION_WORLD_CONTRACT_CALC_EN if use_en else DRAFT_CHAIN_DERIVATION_WORLD_CONTRACT_CALC)
            if calc
            else (DRAFT_CHAIN_DERIVATION_WORLD_CONTRACT_EN if use_en else DRAFT_CHAIN_DERIVATION_WORLD_CONTRACT)
        )
    return (
        (DRAFT_CHAIN_DERIVATION_CALC_EN if use_en else DRAFT_CHAIN_DERIVATION_CALC)
        if calc
        else (DRAFT_CHAIN_DERIVATION_EN if use_en else DRAFT_CHAIN_DERIVATION)
    )
