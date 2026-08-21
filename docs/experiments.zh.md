# 阶段性评测结果

[English](./experiments.md) · [机器可读 CSV 快照](../results/README.zh.md)

本页整理若干 pre-release evaluation snapshots，用于说明 AgenQA 生成的 **Path View** 题目已经呈现出可用的 benchmark-construction signal，并初步显示少量 AgenQA 数据可作为下游训练信号。

AgenQA 的论文框架强调：Edge View 支持 step-level correctness control，而经过 Path-Fold 的 solver-facing Path View 用于观察 benchmark 行为。本页保留三张互补表：前两张展示 benchmark-construction signal，包括强模型区间的难度/区分区域和 Qwen-family 内的 scale-consistency sanity check；第三张展示一个早期下游训练结果，用于观察少量 AgenQA 数据是否能带来跨 benchmark 的 reasoning transfer。

## Table 1. SOTA Solvers on Path View Questions

该表来自一个更明确的问题规约版本：题面约束、答案等价性和输出边界被写得更清楚。评测覆盖 `96` 次 synthesis runs / `547` 道 solver-facing Path View questions。诊断子集指至少一个 solver 未能正确回答的题目集合，用于观察更有区分度的题目区域。

| Solver | All Path View questions | Accuracy | 诊断子集 | Diagnostic accuracy |
| --- | ---: | ---: | ---: | ---: |
| claude-sonnet-4-6 | 466 / 574 | 81.18% | 63 / 150 | 42.00% |
| gemini-3.1-pro-preview | 254 / 294 | 86.39% | 47 / 78 | 60.26% |
| glm-5 | 494 / 574 | 86.06% | 89 / 150 | 59.33% |
| gpt-5.4-2026-03-05 | 469 / 574 | 81.71% | 62 / 150 | 41.33% |
| qwen3.5-plus | 498 / 575 | 86.61% | 90 / 150 | 60.00% |
| All solvers | 2181 / 2591 | 84.18% | 351 / 678 | 51.77% |

## Table 2. Qwen-Family 模型规模正相关信号

该表来自较早的一组 `37` 次 synthesis runs / `175` 道 Path View questions。它展示在同一模型家族内，同一 benchmark 的 accuracy 与模型规模/理论能力呈正相关。这里的核心不是强模型区分度，而是 scale-consistency sanity check：benchmark 分数是否能匹配不同尺寸模型的预期能力顺序，并低成本补充人工逐题审查。

| Solver | All Path View accuracy | Diagnostic-subset accuracy | 说明 |
| --- | ---: | ---: | --- |
| qwen3-4b | 48.00% | 21.55% | lower-capacity baseline |
| qwen3-8b | 56.57% | 34.48% | mid-scale improvement |
| Qwen3-32B | 66.86% | 50.00% | larger-model improvement |
| All solvers | 59.54% | 38.97% | family-level aggregate |

## Table 3. 2K AgenQA 数据的早期训练迁移信号

该表来自一个早期下游训练评测：以 `Qwen3-4B-Instruct` 为 base model，使用约 `2,000` 条 AgenQA 数据训练 `instruct-gspo` 和 `instruct-grpo`，并在外部数学/科学 benchmark 上做 `32` 次平均评测。它不作为最终 paper claim，而是作为 training utility 的早期 evidence：少量结构化 AgenQA 数据是否能在不明显损伤 GPQA 系列的情况下带来正迁移。

| Model | aim24 | HMMT-FEB | GPQA-Diamond | GPQA | SciBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct baseline | 59.62 | 25.00 | 59.90 | 54.95 | 31.71 |
| instruct-gspo | 61.98 | 25.94 | 59.85 | 54.99 | 32.76 |
| instruct-grpo | 61.98 | 26.87 | 59.79 | 55.15 | 32.12 |

相对 baseline 的变化：

| Model | aim24 | HMMT-FEB | GPQA-Diamond | GPQA | SciBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| instruct-gspo | +2.36 | +0.94 | -0.05 | +0.04 | +1.05 |
| instruct-grpo | +2.36 | +1.87 | -0.11 | +0.20 | +0.41 |

初步解读：`aim24` 的 `+2.36` 是最清楚的正向信号；`HMMT-FEB` 尤其在 `instruct-grpo` 上有明显提升；`SciBench` 对 `instruct-gspo` 有一定正迁移。`GPQA` 和 `GPQA-Diamond` 基本持平，其中 `GPQA-Diamond` 的 `-0.05 / -0.11` 不应在缺少方差、置信区间和 paired test 的情况下解读为真实退化。

三张表共同说明：AgenQA 的 Path-Fold 不只是产生更长题面，而是在保留生成侧 dependency state 的同时，为 solver-facing benchmark 构造出可测的难度、强模型区分信号和模型尺寸一致性信号；同时，少量 AgenQA 数据已经显示出 early downstream training utility signal。
