"""ExecutableStepCertBuilder role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.domain.step_cert_schema import step_cert_output_schema_text
from agenqa.domain.executable_step_cert_schema import executable_chain_cert_guide_text

__all__ = [
    "EXECUTABLE_STEP_CERT_BUILDER_V1",
    "EXECUTABLE_STEP_CERT_BUILDER_V1_EN",
]


EXECUTABLE_STEP_CERT_BUILDER_V1 = dedent(
    f"""\
    # ExecutableStepCertBuilder（Executable 链式证书生成：对齐 KnownTree v2）

    ## Input
    - step: $step
    - director_notes: $director_notes
    - task_sketch: $task_sketch
    - background: $background
    - prev_sub_steps_json: $prev_sub_steps_json
    - tail_sub_step_json: $tail_sub_step_json
    - golden_step_code: $golden_step_code
    - memory_json: $memory_json
    - expected_primary_fact_id: $expected_primary_fact_id
    - observed_required_fact_ids_json: $observed_required_fact_ids_json
    - observed_primary_required_fact_id: $observed_primary_required_fact_id

    ## 关于 memory_json（重要）
    - memory_json 为最小视图，仅包含 < step 的：
      - premise_bank: {{id, text}}
      - fact_bank: {{id, text}}
    - 你的输出会写入长期记忆（KnownTree v2），因此必须：
      - 避免新 ID 与 memory_json 已有 ID 冲突；
      - uses_*_ids 只能引用 memory_json 中的 id 或本步新增 delta 的 id。

    ## 目标
    为当前 executable step 生成链式证书（premise/fact/cert），使下一步能稳定复用本步 `key_fact_id`，并让 contract（口径）显式可控。

    ## 语义约定（Executable ↔ Semantic 对齐）
{indent(executable_chain_cert_guide_text(), "    ")}

    ## 强约束
    - 输出必须是严格 JSON（仅一个 ```json 代码块），字段如下：
{indent(step_cert_output_schema_text(), "    ")}
    - step_cert.kind 必须为 "executable_chain_cert"（用于与评测证书 kind="executable_eval_cert" 区分）。
    - key_fact_id 必须指向 fact_delta 中的一个条目，且属于 step_cert.produces_fact_ids。
    - 若 expected_primary_fact_id 非空（step>=2）：
      - step_cert.uses_fact_ids 必须包含 expected_primary_fact_id（否则链式复用断裂）。
    - 不要把 golden 代码/测试期望/具体数值答案写入 premise_delta/fact_delta。

    ## E2E Differential Oracle（方案 B：整题入口 solve + 显式映射）
    - 你必须在 step_cert 中额外给出 `e2e_spec`（用于后续生成整题入口 `solve(...)` 与 e2e tests；path 视角不可见）。
    - 约束：
      - `e2e_spec.function_name` 必须为 "solve"
      - `e2e_spec.params`：参数列表（按顺序），每项 `{{name, optional default}}`；name 必须是合法 Python 标识符
      - `e2e_spec.calls`：按顺序串联子步骤；每项：
        - `step_number`: 对应 record.sub_steps 的 step_number（字符串）
        - `kwargs`: 显式入参映射；value 只能是 `"param:<name>"` 或 `"var:<name>"`
        - 非最后一项必须有 `bind`（保存该步输出变量名）；最后一项必须 `return=true` 且其 step_number 必须等于 tail step_number
    """
)


EXECUTABLE_STEP_CERT_BUILDER_V1_EN = dedent(
    f"""\
    # ExecutableStepCertBuilder (Executable chain certificate aligned with KnownTree v2)

    ## Input
    - step: $step
    - director_notes: $director_notes
    - task_sketch: $task_sketch
    - background: $background
    - prev_sub_steps_json: $prev_sub_steps_json
    - tail_sub_step_json: $tail_sub_step_json
    - golden_step_code: $golden_step_code
    - memory_json: $memory_json
    - expected_primary_fact_id: $expected_primary_fact_id
    - observed_required_fact_ids_json: $observed_required_fact_ids_json
    - observed_primary_required_fact_id: $observed_primary_required_fact_id

    ## About memory_json (important)
    - memory_json is a minimal view containing only < step entries:
      - premise_bank: {{id, text}}
      - fact_bank: {{id, text}}
    - Your output is written into long-term memory (KnownTree v2). You must:
      - avoid ID collisions with memory_json;
      - reference IDs only from memory_json or the current step deltas.

    ## Goal
    Produce a chain certificate (premise/fact/cert) so the next step can reliably reuse `key_fact_id`, and make the contract conventions explicit and controllable.

    ## Semantics (Executable ↔ Semantic alignment)
{indent(executable_chain_cert_guide_text("en"), "    ")}

    ## Hard constraints
    - Output must be strict JSON in a single ```json block with fields:
{indent(step_cert_output_schema_text("en"), "    ")}
    - step_cert.kind must be \"executable_chain_cert\" (to distinguish from eval cert kind=\"executable_eval_cert\").
    - key_fact_id must point to an entry in fact_delta and be included in step_cert.produces_fact_ids.
    - If expected_primary_fact_id is non-empty (step>=2):
      - step_cert.uses_fact_ids must include expected_primary_fact_id.
    - Do NOT include golden code, test expectations, or concrete answer values inside premise_delta/fact_delta.

    ## E2E Differential Oracle (Plan B: whole-problem solve + explicit mapping)
    - You must additionally provide `e2e_spec` inside step_cert (used to generate `solve(...)` and e2e tests; hidden from path view).
    - Constraints:
      - `e2e_spec.function_name` must be \"solve\"
      - `e2e_spec.params`: ordered param list; each item `{{name, optional default}}`; name must be a valid Python identifier
      - `e2e_spec.calls`: ordered chain; each item:
        - `step_number`: must match a record.sub_steps step_number (string)
        - `kwargs`: explicit mapping; values must be either `\"param:<name>\"` or `\"var:<name>\"`
        - non-last calls must set `bind`; last call must set `return=true` and its step_number must equal the tail step_number
    """
)
