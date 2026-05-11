# Informal Evaluation Results

[中文](./experiments.zh.md)

> This page summarizes an internal evaluation snapshot from the March 26, 2026 project report. It is intended to show the current evidence behind AgenQA as a benchmark-synthesis framework. It is not the final experiment table for the paper and is not a formal benchmark release.

## What the Evaluation Tries to Show

AgenQA is currently evaluated as a benchmark-synthesis framework. The central questions are:

1. Can the generated path questions separate model capabilities rather than becoming trivially easy or uniformly unsolved?
2. Does the SOTA group outperform smaller same-family models, suggesting that the tasks are not random noise?
3. Does the hard subset, where at least one model fails, retain enough difficulty for model diagnosis?
4. Do stronger contracts and clearer problem boundaries reduce format-related misjudgment and stabilize evaluation?

## Protocol Notes

Both batches below evaluate path questions, but they do not use exactly the same protocol:

- Batch 1 reports final labels on path questions after a later re-evaluation pass. `T/T(-)` is counted as correct, while `F/F(-)/C` is counted as incorrect.
- Batch 2 reports path questions after stronger contract optimization and extracts answer signals directly from strong-solver runs.
- The two batches differ in contract strength, sample size, model pool, and filtering rules. Cross-batch comparison should therefore be read as trend evidence, not a strict controlled comparison.

## Main Takeaways

- In Batch 1, the `SOTA` group is clearly stronger than the `QWEN` group. Overall `ALL_MODELS` accuracy is `71.31%` for SOTA and `59.54%` for QWEN.
- On Batch 1's hard subset, `ALL_MODELS` accuracy is `47.71%` for SOTA and `38.97%` for QWEN.
- Batch 2 scales the stronger-contract setting to `96` runs / `547` path questions. Overall `ALL_MODELS` accuracy is `84.18%`, while the subset with at least one `F` has `51.77%` accuracy.
- Stronger contracts make model outputs better aligned with problem constraints, reducing errors caused by ambiguous output boundaries rather than reasoning failure.

## Stage Overview

| Batch | Benchmark group | Runs | Path questions | Overall ALL_MODELS accuracy | Hard subset ratio | Hard-subset ALL_MODELS accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | QWEN | 37 | 175 | 59.54% | 66.29% | 38.97% |
| Batch 1 | SOTA | 37 | 175 | 71.31% | 54.86% | 47.71% |
| Batch 2 | SOTA | 96 | 547 | 84.18% | 27.42% | 51.77% |

Notes:

- In Batch 1, `hard_subset_ratio` means the share of path questions where at least one model is labeled `F/F(-)/C`.
- In Batch 2, `hard_subset_ratio` means the share of questions where at least one model is labeled `F`.
- In Batch 2, the effective denominator differs slightly across models, so the per-model totals below are not identical.

## Batch 1: Model-Aggregated Results

### QWEN Group

| Model | Full accuracy | Hard-subset accuracy |
| --- | ---: | ---: |
| Qwen3-32B | 66.86% | 50.00% |
| qwen3-8b | 56.57% | 34.48% |
| qwen3-4b | 48.00% | 21.55% |
| qwen3-next-80b-a3b-instruct | 66.29% | 49.14% |
| qwq-32b | 60.00% | 39.66% |
| ALL_MODELS | 59.54% | 38.97% |

### SOTA Group

| Model | Full accuracy | Hard-subset accuracy |
| --- | ---: | ---: |
| DeepSeek-V3.2-Exp | 65.14% | 36.46% |
| claude_sonnet4_5 | 74.29% | 53.12% |
| gpt-5.2-1211-global | 70.29% | 45.83% |
| kimi-k2-thinking | 74.86% | 54.17% |
| qwen3-max | 72.00% | 48.96% |
| ALL_MODELS | 71.31% | 47.71% |

## Batch 2: Stronger-Contract Aggregate Results

### Full Set

| Model | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| claude-sonnet-4-6 | 466 | 574 | 81.18% |
| gemini-3.1-pro-preview | 254 | 294 | 86.39% |
| glm-5 | 494 | 574 | 86.06% |
| gpt-5.4-2026-03-05 | 469 | 574 | 81.71% |
| qwen3.5-plus | 498 | 575 | 86.61% |
| ALL_MODELS | 2181 | 2591 | 84.18% |

### Subset With At Least One F

| Model | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| claude-sonnet-4-6 | 63 | 150 | 42.00% |
| gemini-3.1-pro-preview | 47 | 78 | 60.26% |
| glm-5 | 89 | 150 | 59.33% |
| gpt-5.4-2026-03-05 | 62 | 150 | 41.33% |
| qwen3.5-plus | 90 | 150 | 60.00% |
| ALL_MODELS | 351 | 678 | 51.77% |

## How to Read These Results

The snapshot supports three practical claims:

- AgenQA path questions show observable discriminative power: on the same batch, the SOTA group performs better than the QWEN group.
- The hard subset remains useful for diagnosis: even stronger models drop substantially from full-set accuracy.
- Contract strength matters for benchmark construction: clearer constraints reduce format-driven misjudgment and make the signal closer to reasoning ability.

## Still Not Included Here

This page publishes evaluation numbers, but it still does not release:

- the full raw benchmark set;
- source papers or source documents;
- raw solver responses;
- consensus JSON files;
- model API configuration;
- prompt snapshots.
