# 阶段性评测结果

[English](./experiments.md)

本页整理自 2026-03-26 的 evaluation snapshot，用于说明 AgenQA 生成的 **Path View** 题目已经呈现出可用的 benchmark-construction signal。

AgenQA 的论文框架强调：Edge View 支持 step-level correctness control，而经过 Path-Fold 的 solver-facing Path View 用于观察 benchmark 行为。本页只保留两张互补表：一张展示强模型在 Path View 题目上的难度与区分区域，另一张展示 Qwen-family 内 benchmark accuracy 与模型规模/理论能力的正相关关系，作为 benchmark 合理性的 scale-consistency sanity check。

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

两张表共同说明：AgenQA 的 Path-Fold 不只是产生更长题面，而是在保留生成侧 dependency state 的同时，为 solver-facing benchmark 构造出可测的难度、强模型区分信号和模型尺寸一致性信号。
