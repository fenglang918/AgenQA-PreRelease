"""Path-Fold role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.domain.path_fold_schema import path_fold_output_schema_text

__all__ = [
    "PATH_FOLD_V1",
    "PATH_FOLD_V1_EN",
]


PATH_FOLD_V1 = dedent(
    f"""\
    # PathFold（路径折叠：生成 path 题面）

    ## Input
    - step: $step
    - question_type: $question_type
    - premise_bank_json: $premise_bank_json
    - history_json: $history_json

    说明：
    - premise_bank_json：当前 tail step 的 head-tail 起点 P（仅前提/定义/条件），JSON 列表，每项形如 {{id,text}}。
    - history_json：一个结构化对象，包含：
      - `recent_steps`：最近若干步的精确 `{{step, question, answer}}` 序列；
      - `older_summary`：更早 steps 的压缩摘要（仅用于粗粒度理解链路，不是精确真源）。
    - 你必须把 `recent_steps` 的最后一项视为 tail step；只有它的 `Answer` 才是本次 Path-Fold 的真源。

    ## Task
    你需要把“从 P 出发，经由多步推理链到达 Answer_$step”这条路径折叠成一个等价的单步 path 问题，并输出两个版本：

    1) question_scaffolded（带提示版，较易）
       - 把原链条拆成若干子任务（可用 (a)(b)(c)… 或 S1/S2/S3… 的形式）；
       - 每个子任务只要求"推导/证明/求出"中间量，不要把中间结论写成已知；
       - 最后一问与原 tail 的目标一致（Answer 等价），并保持题型一致（例如 MCQ 仍是 A/B/C/D）。

    2) question_direct（无任何提示版，最难）
       - 直接问最终目标，不列出任何中间子任务；
       - 只允许重述 premise_bank 中的原始定义/条件；
       - ❌ 严禁任何形式的"提示性内容"，包括但不限于：
         - "(Note: ...)", "(Hint: ...)", "Recall that ...", "(Important: ...)" 等辅助文本；
         - 透露解题所需的技术/方法名称（如"measure change", "Fourier transform"）；
         - 透露变换方向或符号（如"u-i", "+1", "share measure"）；
         - 任何能让 solver 跳过推导、直接匹配答案的信息。
       - 目标：solver 必须独立发现推理路径，不能从题面获得任何"捷径"。

    ## Hard constraints (must follow)
    - **输出目标一致性（最重要）**：
      - 你必须以 `history_json.recent_steps` 中 **最后一步（tail step）** 的 Answer 为“真源”，并据此确定本次 Path-Fold 的**必交付输出项**（required outputs）。
      - 两个版本（scaffolded/direct）在“必交付输出项”和“最终答案格式”上必须**完全一致**；差异只能来自是否提供子任务脚手架（提示密度），而不是改变问题要求集合。
      - ❌ 禁止 direct 版本“少问一部分”，导致 solver 只需回答部分输出；这会造成题面与参考答案不对齐，评测必然失真。
      - ❌ 禁止新增 tail Answer 中不存在、且无法被 tail Answer 表达的额外要求（例如让 solver 输出趋势分析文字，但 tail Answer 只有表达式）。
      - 若 tail step 的 Question 含“分析/解释/比较”等文字性要求，但 tail Answer 没有提供可机判的等价表达（例如只给了一个表达式），则 Path-Fold 题面必须以 tail Answer 为准，**不要**强行要求额外的文字输出。
    - 两个版本都必须与 tail step 的 Answer 语义等价（保持同一个可机判答案）。
    - scaffolded 与 direct 的难度差异应该显著：scaffolded 有子任务引导，direct 完全无提示。
    - Question 正文中：
      - ❌ 禁止出现 "上一步/前序/step/history/fact_bank/premise_bank/known_0" 等内部指针。
      - ❌ 禁止粘贴 history 中任何一步的 Answer（选项字母、数值、闭式表达式、结论句）作为已知条件。
      - ✅ 允许引用 premise_bank 中的定义/符号（但不要显式引用 premise id）。
    - question_direct 额外约束：
      - ❌ 禁止出现任何 "(Note: ...)", "(Hint: ...)", "Recall that ..." 等提示语句。
      - ❌ 禁止透露推导过程中的关键洞察（技术名词、变换类型、符号方向等）。
    - 若 question_type 为 MCQ：必须提供 A/B/C/D 四个选项，并说明"Answer with the option letter only."
    - 若 question_type 为 Derivation：两个版本的题面末尾都必须追加一句“答案格式声明”，明确要求“最终答案写成一个 LaTeX `\\boxed{...}` 表达式，并显式写 `in terms of ...` 列出允许出现的符号（必须来自 premise_bank/题面已定义，不得引入新符号）”。
    - 若 question_type 为 Numeric：两个版本的题面末尾都必须追加一句“答案格式声明”，明确要求“最终答案写成一个 LaTeX `\\boxed{...}` 数值，并显式声明误差/精度口径（绝对/相对误差阈值，或有效数字），与题面保持一致”。
    - 输出必须是严格 JSON（用 ```json 代码块包裹），字段如下：
{indent(path_fold_output_schema_text(), '    ')}
    """
)


PATH_FOLD_V1_EN = dedent(
    f"""\
    # PathFold (Path folding: generate path questions)

    ## Input
    - step: $step
    - question_type: $question_type
    - premise_bank_json: $premise_bank_json
    - history_json: $history_json

    Notes:
    - premise_bank_json: the head-tail start P (premises/definitions/conditions only), as a JSON list of {{id,text}}.
    - history_json: a structured object with:
      - `recent_steps`: exact recent `{{step, question, answer}}` items;
      - `older_summary`: compressed summaries of earlier steps for coarse context only.
    - Treat the last item in `recent_steps` as the tail step; only its Answer is the source of truth.

    ## Task
    Fold the multi-step path “P → Answer_$step” into an equivalent single path-folded problem, and output two variants:

    1) question_scaffolded (with intermediate hints, easier)
       - Decompose into sub-goals (e.g., (a)(b)(c) or S1/S2/S3).
       - Each sub-goal asks to *derive* an intermediate quantity; do NOT state intermediate results as givens.
       - The final ask must match the original tail objective (answer-equivalent), keeping the same question type (e.g., MCQ stays A/B/C/D).

    2) question_direct (ZERO hints, hardest)
       - Ask the final target directly; do NOT list any intermediate sub-goals.
       - You may ONLY restate primitive definitions/conditions from premise_bank.
       - ❌ STRICTLY FORBIDDEN - any form of "helper text" including but not limited to:
         - "(Note: ...)", "(Hint: ...)", "Recall that ...", "(Important: ...)" or similar;
         - Revealing technique/method names needed (e.g., "measure change", "Fourier transform");
         - Revealing transformation direction or sign (e.g., "u-i", "+1", "share measure");
         - Any text that lets the solver shortcut the derivation or pattern-match the answer.
       - Goal: the solver must independently discover the reasoning path with no shortcuts from the question text.

    ## Hard constraints (must follow)
    - **Output target consistency (MOST IMPORTANT)**:
      - Treat the tail-step Answer (the last item in `history_json.recent_steps`) as the single source of truth for the **required outputs**.
      - Both variants (scaffolded/direct) must demand the **exact same** required outputs and final answer format. The ONLY difference is hint density (sub-goal scaffolding), not the target set.
      - ❌ The direct variant must NOT drop any required outputs (asking for only a subset). This breaks question–reference alignment and makes evaluation unreliable.
      - ❌ Do NOT add extra requirements that are not represented by the tail Answer (e.g., ask for prose trend analysis while the tail Answer is only an expression).
      - If the tail Question contains prose-style asks (analysis/explain/compare) but the tail Answer does not encode them in a machine-checkable way, align the folded questions to the tail Answer. Do NOT force extra prose output.
    - Both variants must be semantically equivalent to the tail Answer (same machine-checkable answer).
    - The difficulty gap between scaffolded and direct must be significant: scaffolded has sub-goal guidance, direct has absolutely no hints.
    - In the Question text:
      - Forbidden: internal pointers like "previous step", "step X", "history", "fact_bank", "premise_bank", "known_0".
      - Forbidden: pasting any step's Answer (MCQ letter, numbers, closed forms, conclusion sentences) as a given.
      - Allowed: use definitions/symbols from premise_bank (but do not cite premise IDs).
    - Additional constraints for question_direct:
      - Forbidden: any "(Note: ...)", "(Hint: ...)", "Recall that ..." helper sentences.
      - Forbidden: revealing key insights from the derivation (technique names, transformation types, sign/direction choices).
    - If question_type is MCQ: include exactly 4 options (A/B/C/D) and say "Answer with the option letter only."
    - If question_type is Derivation: both variants must end with one short answer-format sentence requiring a single LaTeX `\\boxed{...}` expression, and explicitly include an `in terms of ...` symbol list (symbols must be defined in premise_bank/the question; do not introduce new symbols).
    - If question_type is Numeric: both variants must end with one short answer-format sentence requiring a single LaTeX `\\boxed{...}` numeric value, and explicitly state the tolerance/precision convention (absolute/relative error, or significant figures) consistent with the question.
    - Output must be strict JSON wrapped in ```json with fields:
{indent(path_fold_output_schema_text('en'), '    ')}
    """
)
