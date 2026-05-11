# 非正式评测结果

[English](./experiments.md)

> 本页整理自 2026-03-26 项目汇报中的内部评测快照，用于展示 AgenQA 作为 benchmark-synthesis framework 的阶段性信号。它不是论文最终实验表，也不代表正式 benchmark release。

AgenQA 当前最值得展示的评测信号很简单：一是 stronger-contract setting 下的 SOTA 评测结果，二是 Qwen 系列内部随模型规模上升出现的解题梯度。

## Table 1. SOTA 评测结果

Batch 2 使用优化后的题面 contract，统计 `96` 次 run / `547` 道 path questions。Hard subset 指至少一个模型出现 `F` 的题目子集。

| Model | Full set | Full accuracy | Hard subset | Hard accuracy |
| --- | ---: | ---: | ---: | ---: |
| claude-sonnet-4-6 | 466 / 574 | 81.18% | 63 / 150 | 42.00% |
| gemini-3.1-pro-preview | 254 / 294 | 86.39% | 47 / 78 | 60.26% |
| glm-5 | 494 / 574 | 86.06% | 89 / 150 | 59.33% |
| gpt-5.4-2026-03-05 | 469 / 574 | 81.71% | 62 / 150 | 41.33% |
| qwen3.5-plus | 498 / 575 | 86.61% | 90 / 150 | 60.00% |
| ALL_MODELS | 2181 / 2591 | 84.18% | 351 / 678 | 51.77% |

## Table 2. Qwen 梯度结果

Batch 1 使用 `37` 次 synthesis runs / `175` 道 path questions，展示同系列 Qwen 模型在全量题目与 hard subset 上的梯度。

| Model | Full accuracy | Hard-subset accuracy | Signal |
| --- | ---: | ---: | --- |
| qwen3-4b | 48.00% | 21.55% | small-model baseline |
| qwen3-8b | 56.57% | 34.48% | mid-scale improvement |
| Qwen3-32B | 66.86% | 50.00% | larger-model improvement |
| ALL_MODELS | 59.54% | 38.97% | Qwen group aggregate |

这两张表分别说明：AgenQA path questions 在强模型上仍有可观测的 hard subset，且在 Qwen 系列内部能体现随模型能力上升的准确率梯度。

本页仍不公开完整原始题库、source papers、raw solver responses、consensus JSON、model API configuration 或 prompt snapshots。
