# Path-Fold Prompt

```text
# PathFold（路径折叠：生成 path 题面）

## Input

- step: $step
- question_type: $question_type
- premise_bank_json: $premise_bank_json
- history_json: $history_json

说明：
- premise_bank_json：当前 tail step 的 head-tail 起点 P（仅前提/定义/条件），JSON 列表，每项形如 {id,text}。
- history_json：一个结构化对象，包含：
  - recent_steps：最近若干步的精确 {step, question, answer} 序列；
  - older_summary：更早 steps 的压缩摘要。
- 你必须把 recent_steps 的最后一项视为 tail step；只有它的 Answer 才是本次 Path-Fold 的真源。

## Task

你需要把“从 P 出发，经由多步推理链到达 Answer_$step”这条路径折叠成一个等价的单步 path 问题，并输出两个版本：

1. question_scaffolded（带提示版，较易）
   - 把原链条拆成若干子任务。
   - 每个子任务只要求“推导/证明/求出”中间量，不要把中间结论写成已知。
   - 最后一问与原 tail 的目标一致（Answer 等价），并保持题型一致。

2. question_direct（无任何提示版，最难）
   - 直接问最终目标，不列出任何中间子任务。
   - 只允许重述 premise_bank 中的原始定义/条件。
   - 严禁任何形式的提示性内容，包括：
     - "(Note: ...)", "(Hint: ...)", "Recall that ..." 等辅助文本；
     - 透露解题所需的技术/方法名称；
     - 透露变换方向或符号；
     - 任何能让 solver 跳过推导、直接匹配答案的信息。
   - 目标：solver 必须独立发现推理路径，不能从题面获得捷径。

## Hard constraints

- 输出目标一致性
  - 必须以 history_json.recent_steps 中最后一步的 Answer 为真源。
  - 两个版本在必交付输出项和最终答案格式上必须完全一致。
  - 禁止 direct 版本少问一部分，导致 solver 只需回答部分输出。
  - 禁止新增 tail Answer 中不存在、且无法被 tail Answer 表达的额外要求。

- 两个版本都必须与 tail step 的 Answer 语义等价。
- scaffolded 与 direct 的难度差异应该显著。

- Question 正文中：
  - 禁止出现“上一步/前序/step/history/fact_bank/premise_bank/known_0”等内部指针。
  - 禁止粘贴 history 中任何一步的 Answer 作为已知条件。
  - 允许引用 premise_bank 中的定义/符号，但不要显式引用 premise id。

- question_direct 额外约束：
  - 禁止出现任何提示语句。
  - 禁止透露推导过程中的关键洞察。

## 输出格式

输出必须是严格 JSON：

{
  "question_scaffolded": "...",
  "question_direct": "...",
  "fold_notes": "..."
}
```
