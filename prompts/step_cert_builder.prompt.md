# Step Certificate Builder Prompt

```text
# StepCertBuilder（推理证书生成）

## Input

- step: $step
- question: $question
- solution: $solution
- answer: $answer
- question_type: $question_type
- memory_json: $memory_json

## 关于 memory_json

memory_json 为 step_cert_builder 专用的最小视图，仅包含 < step 的：

- premise_bank: {id, text}
- fact_bank: {id, text}

用途：

- 让你能把“语义”对齐到可引用的 ID（用于 uses_*_ids）。
- 让你能避免新 ID 与既有 ID 冲突。

## 目标

从本步最终 QA 中抽取：

- premise_delta：本步新增前提（definition/assumption/condition），可为空。
- fact_delta：本步新增结论（可复用的中间结论/答案等价锚点）。
- step_cert：本步推理证书（结构化记录依赖与产出）。
- key_fact_id：指向 fact_delta 中“与本步 Answer 等价”的条目。

## 记忆写入策略（保守）

- 你的输出会写入长期记忆并被后续步骤引用；因此优先保证“稳”和“少”。
- 只写入未来可能被复用的、稳定的语义锚点：
  - 强优先：与 Answer 等价的 key_fact（必须有）。
  - 可选：少量关键定义/定理/中间结论。

- premise_delta 只用于“真前提”，不要把任何推导结论伪装成定义写进 premise_delta：
  - 允许：符号约定、变量替换、度量/概率测度的定义、明确新增的外生假设。
  - 禁止：与本步或前序 step 已推导出的关键结论语义等价的公式/数值/闭式表达式。
  - 这些属于 fact，应写入 fact_delta。

- 避免把脆弱的推导细节固化成可复用 fact。
- 如果某个中间结论你不确定是否严格正确：不要写入 fact_delta。
- 避免把题干/解答叙述的重复句写入 premise_delta/fact_delta。

## 强约束

- premise_delta 与 fact_delta 的每个条目必须包含 id，并填写 source_step。
- premise_delta/fact_delta 内部 ID 不能重复。
- premise_delta 与 fact_delta 之间也不能复用同一个 ID。
- 新 ID 不能与 memory_json 中已有的 premise/fact ID 冲突。
- key_fact_id 必须指向 fact_delta 中的条目。

step_cert 引用约束：

- uses_premise_ids：只能引用 memory_json.premise_bank 的 id 或 premise_delta 的 id。
- uses_fact_ids：只能引用 memory_json.fact_bank 的 id 或 fact_delta 的 id。
- produces_fact_ids：必须全部来自 fact_delta 的 id。

## 输出格式

仅输出一个 JSON 对象：

{
  "premise_delta": [],
  "fact_delta": [],
  "step_cert": {
    "step": "$step",
    "uses_premise_ids": [],
    "uses_fact_ids": [],
    "produces_fact_ids": [],
    "key_fact_id": "...",
    "cert_text": "..."
  },
  "key_fact_id": "..."
}
```
