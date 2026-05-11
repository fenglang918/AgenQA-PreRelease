# 非正式评测结果

[English](./experiments.md)

> 本页整理自 2026-03-26 项目汇报中的内部评测快照，用于展示 AgenQA 作为 benchmark-synthesis framework 的阶段性信号。它不是论文最终实验表，也不代表正式 benchmark release。

## 评测想回答什么

AgenQA 在当前阶段主要被评估为 benchmark-synthesis framework。核心问题包括：

1. 生成出的 path 题目能否区分模型能力，而不是过于简单或所有模型都解不出？
2. SOTA 组是否明显强于同系列较小模型组，从而说明题目不是随机噪声？
3. 存在失败模型的 hard subset 是否能保留足够难度，用于后续模型诊断？
4. 更强的 contract / 题面约束是否能减少格式性误判，并提升评测稳定性？

## 口径说明

这里的两批数据都统计 path 题目，但评测口径不完全一致：

- 第一批数据统计 path 题目上的最终标签结果，后续经过再次 eval；`T/T(-)` 记为正确，`F/F(-)/C` 记为错误。
- 第二批数据统计优化 contract 后的 path 题目，直接提取过程中 strong solver 的作答信号。
- 两批数据的 contract 强度、样本规模、模型集合和过滤规则不同，因此横向比较应视为趋势参考，不是严格的一一对应对照实验。

## 核心结论

- 第一批数据中，`SOTA` 组整体显著强于 `QWEN` 组。全量题目 `ALL_MODELS` 正确率分别为 `71.31%` 和 `59.54%`。
- 在第一批数据的 hard subset 中，`SOTA` 组 `ALL_MODELS` 正确率为 `47.71%`，高于 `QWEN` 组的 `38.97%`。
- 第二批数据在更强 contract 约束下扩展到 `96` 次 run / `547` 道 path 题目，`ALL_MODELS` 全量正确率为 `84.18%`，存在 `F` 的题目子集正确率为 `51.77%`。
- 更强的 contract 约束使模型输出更符合题面规范，减少了因题目限制不严格、输出边界不清而造成的格式性误判。

## 阶段总览

| Batch | Benchmark group | Runs | Path questions | Overall ALL_MODELS accuracy | Hard subset ratio | Hard-subset ALL_MODELS accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 第一批数据 | QWEN | 37 | 175 | 59.54% | 66.29% | 38.97% |
| 第一批数据 | SOTA | 37 | 175 | 71.31% | 54.86% | 47.71% |
| 第二批数据 | SOTA | 96 | 547 | 84.18% | 27.42% | 51.77% |

说明：

- 第一批数据的 `hard_subset_ratio` 表示“至少一个模型出现 `F/F(-)/C`”的 path 题目占比。
- 第二批数据的 `hard_subset_ratio` 表示“至少一个模型出现 `F`”的题目占比。
- 第二批数据中不同模型的有效题目数略有差异，所以下方模型表的 per-model denominator 不完全相同。

## 第一批数据：模型聚合结果

### QWEN 组

| Model | Full accuracy | Hard-subset accuracy |
| --- | ---: | ---: |
| Qwen3-32B | 66.86% | 50.00% |
| qwen3-8b | 56.57% | 34.48% |
| qwen3-4b | 48.00% | 21.55% |
| qwen3-next-80b-a3b-instruct | 66.29% | 49.14% |
| qwq-32b | 60.00% | 39.66% |
| ALL_MODELS | 59.54% | 38.97% |

### SOTA 组

| Model | Full accuracy | Hard-subset accuracy |
| --- | ---: | ---: |
| DeepSeek-V3.2-Exp | 65.14% | 36.46% |
| claude_sonnet4_5 | 74.29% | 53.12% |
| gpt-5.2-1211-global | 70.29% | 45.83% |
| kimi-k2-thinking | 74.86% | 54.17% |
| qwen3-max | 72.00% | 48.96% |
| ALL_MODELS | 71.31% | 47.71% |

## 第二批数据：优化 contract 后的聚合结果

### 全量题目

| Model | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| claude-sonnet-4-6 | 466 | 574 | 81.18% |
| gemini-3.1-pro-preview | 254 | 294 | 86.39% |
| glm-5 | 494 | 574 | 86.06% |
| gpt-5.4-2026-03-05 | 469 | 574 | 81.71% |
| qwen3.5-plus | 498 | 575 | 86.61% |
| ALL_MODELS | 2181 | 2591 | 84.18% |

### 存在 F 的题目子集

| Model | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| claude-sonnet-4-6 | 63 | 150 | 42.00% |
| gemini-3.1-pro-preview | 47 | 78 | 60.26% |
| glm-5 | 89 | 150 | 59.33% |
| gpt-5.4-2026-03-05 | 62 | 150 | 41.33% |
| qwen3.5-plus | 90 | 150 | 60.00% |
| ALL_MODELS | 351 | 678 | 51.77% |

## 如何解读这些结果

这些结果主要支持三点：

- AgenQA 生成的 path 题目具有可观测的模型区分度：同一批题上，SOTA 组整体优于 QWEN 组。
- hard subset 保留了诊断价值：即使强模型组，全体模型在 hard subset 上也明显低于全量准确率。
- contract 强化对 benchmark construction 很关键：它能减少题面边界不清导致的格式性误判，让评测更接近真正的推理能力差异。

## 本页仍不包含

本页只公开阶段性评测结果，不公开：

- 完整原始题库；
- source papers 或 source documents；
- raw solver responses；
- consensus JSON files；
- model API configuration；
- prompt snapshots。
