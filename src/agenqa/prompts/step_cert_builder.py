"""StepCertBuilder role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.domain.step_cert_schema import step_cert_output_schema_text

__all__ = [
    "STEP_CERT_BUILDER_V1",
    "STEP_CERT_BUILDER_V1_EN",
]


STEP_CERT_BUILDER_V1 = dedent(
    f"""\
    # StepCertBuilder（推理证书生成）

    ## Input
    - step: $step
    - question: $question
    - solution: $solution
    - answer: $answer
    - question_type: $question_type
    - memory_json: $memory_json

    ## 关于 memory_json
    - memory_json 为 step_cert_builder 专用的最小视图，仅包含 < step 的：
      - premise_bank: {{id, text}}
      - fact_bank: {{id, text}}（text 来自 statement 或 text）
    - 用途：
      - 让你能把“语义”对齐到可引用的 ID（用于 uses_*_ids）；
      - 让你能避免新 ID 与既有 ID 冲突。

    ## 目标
    从本步最终 QA 中抽取：
    - premise_delta：本步新增前提（definition/assumption/condition），可为空；
    - fact_delta：本步新增结论（可复用的中间结论/答案等价锚点）；
    - step_cert：本步推理证书（结构化记录依赖与产出）；
    - key_fact_id：指向 fact_delta 中“与本步 Answer 等价”的条目。

    ## 记忆写入策略（保守）
    - 你的输出会写入长期记忆并被后续步骤引用；因此优先保证“稳”和“少”。
    - 只写入未来可能被复用的、稳定的语义锚点：
      - 强优先：与 Answer 等价的 key_fact（必须有）。
      - 可选：少量关键定义/定理/中间结论（确实能复用、且你有把握正确）。
    - **premise_delta 只用于“真前提”**（长期有效的定义/假设/条件），不要把任何推导结论“伪装成定义”写进 premise_delta：
      - ✅ 允许：符号约定、变量替换、度量/概率测度的定义、明确新增的外生假设。
      - ❌ 禁止：与本步或前序 step 已推导出的关键结论语义等价的公式/数值/闭式表达式（这属于 fact，应写入 fact_delta）。
      - 背景：Path-Fold head-tail 的题面由专门的折叠器生成，不需要用 premise_bank 来“补齐可解性”；premise_bank 的污染会长期降低后续 head-tail 的有效性。
    - 避免把脆弱的推导细节固化成可复用 fact（例如：积分收敛/发散判别、边界分类的细枝末节、长链代数化简）。
      - 如果某个中间结论你不确定是否严格正确：不要写入 fact_delta（宁可缺失也不要固化错误）。
    - 避免把“题干/解答叙述的重复句”写入 premise_delta/fact_delta，除非它确实是可复用的定义或约束。

    ## 强约束
    - premise_delta 与 fact_delta 的每个条目必须包含 id；并填写 source_step。
    - premise_delta/fact_delta 内部 ID 不能重复；premise_delta 与 fact_delta 之间也不能复用同一个 ID。
    - premise_delta/fact_delta 的新 ID 不能与 memory_json 中已有的 premise/fact ID 冲突。
    - key_fact_id 必须指向 fact_delta 中的条目。
    - step_cert 引用约束：
      - uses_premise_ids：只能引用 memory_json.premise_bank 的 id 或 premise_delta 的 id。
      - uses_fact_ids：只能引用 memory_json.fact_bank 的 id 或 fact_delta 的 id。
      - produces_fact_ids：必须全部来自 fact_delta 的 id。
    - 若本题为 MCQ：key_fact 条目必须包含 mcq_choice / mcq_choice_text / statement，且 statement 同时包含选项字母与选项内容。

    ## 输出格式（JSON）
    仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段为：
{indent(step_cert_output_schema_text(), '    ')}
    """
)


STEP_CERT_BUILDER_V1_EN = dedent(
    f"""\
    # StepCertBuilder (Reasoning Certificate)

    ## Input
    - step: $step
    - question: $question
    - solution: $solution
    - answer: $answer
    - question_type: $question_type
    - memory_json: $memory_json

    ## About memory_json
    - memory_json is a minimal view for step_cert_builder, containing only < step entries:
      - premise_bank: {{id, text}}
      - fact_bank: {{id, text}} (text is derived from statement or text)
    - Purposes:
      - Map semantics to referenceable IDs (for uses_*_ids).
      - Avoid ID collisions with existing memory IDs.

    ## Goal
    Extract from the final QA:
    - premise_delta: new premises (definition/assumption/condition), optional.
    - fact_delta: new reusable facts (including the answer-equivalent anchor).
    - step_cert: structured reasoning certificate for this step.
    - key_fact_id: points to the answer-equivalent fact in fact_delta.

    ## Memory write policy (conservative)
    - Your output is written into long-term memory and may be reused by later steps; prioritize being correct and minimal.
    - Only write stable semantic anchors that are likely to be reused:
      - Highest priority: the key_fact equivalent to Answer (required).
      - Optional: a small number of key definitions/theorems/intermediate results (truly reusable and you are confident they are correct).
    - **premise_delta is for true premises only** (durable definitions/assumptions/conditions). Do NOT smuggle derived results or solution methods into premise_delta:
      - ✅ Allowed: explicit new exogenous assumptions (e.g., "Assume interest rate is constant"), global notation conventions.
      - ❌ Forbidden:
        - Any formula/number/closed form that is semantically equivalent to a derived conclusion (put these in fact_delta).
        - **Method-specific definitions** that reveal the solution path (e.g., specific ansatz forms, auxiliary variables for solving ODEs/integrals like $\\beta, d, r_{{\\pm}}$, substitution tricks).
      - Rationale: head-tail is evaluated via a separate Path-Fold question; adding solution steps as "premises" leaks the solution path and degrades the evaluation.
    - Do NOT solidify fragile derivation details as reusable facts (e.g., integral convergence tests, boundary-classification fine print, long algebraic manipulations).
      - If you are not sure an intermediate statement is strictly correct: do not include it in fact_delta.
    - Avoid adding premises/facts that merely restate the question/solution narration unless they are genuinely reusable definitions/constraints.

    ## Hard constraints
    - Every premise_delta / fact_delta entry must include id and source_step.
    - IDs must be unique within premise_delta, within fact_delta, and must not overlap between premise_delta and fact_delta.
    - New IDs in premise_delta/fact_delta must not conflict with existing IDs in memory_json.
    - key_fact_id must refer to an entry in fact_delta.
    - step_cert reference constraints:
      - uses_premise_ids must reference IDs from memory_json.premise_bank or premise_delta.
      - uses_fact_ids must reference IDs from memory_json.fact_bank or fact_delta.
      - produces_fact_ids must all come from fact_delta IDs.
    - If MCQ: key_fact must include mcq_choice / mcq_choice_text / statement, and statement must contain both the letter and the option text.

    ## Output format (JSON)
    Output one JSON object only, wrapped in a ```json code block. Top-level fields:
{indent(step_cert_output_schema_text('en'), '    ')}
    """
)
