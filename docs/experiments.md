# Informal Evaluation Results

[中文](./experiments.zh.md)

> This page summarizes an internal evaluation snapshot from the March 26, 2026 project report. It is intended to show the current evidence behind AgenQA as a benchmark-synthesis framework. It is not the final experiment table for the paper and is not a formal benchmark release.

The most useful public signal is simple: one table shows SOTA evaluation under the stronger-contract setting, and the other shows the Qwen-family scale gradient.

## Table 1. SOTA Evaluation Results

Batch 2 uses stronger problem contracts and covers `96` runs / `547` path questions. The hard subset contains questions where at least one model received an `F`.

| Model | Full set | Full accuracy | Hard subset | Hard accuracy |
| --- | ---: | ---: | ---: | ---: |
| claude-sonnet-4-6 | 466 / 574 | 81.18% | 63 / 150 | 42.00% |
| gemini-3.1-pro-preview | 254 / 294 | 86.39% | 47 / 78 | 60.26% |
| glm-5 | 494 / 574 | 86.06% | 89 / 150 | 59.33% |
| gpt-5.4-2026-03-05 | 469 / 574 | 81.71% | 62 / 150 | 41.33% |
| qwen3.5-plus | 498 / 575 | 86.61% | 90 / 150 | 60.00% |
| ALL_MODELS | 2181 / 2591 | 84.18% | 351 / 678 | 51.77% |

## Table 2. Qwen Scale Gradient

Batch 1 uses `37` synthesis runs / `175` path questions and shows the within-family Qwen gradient on the full set and hard subset.

| Model | Full accuracy | Hard-subset accuracy | Signal |
| --- | ---: | ---: | --- |
| qwen3-4b | 48.00% | 21.55% | small-model baseline |
| qwen3-8b | 56.57% | 34.48% | mid-scale improvement |
| Qwen3-32B | 66.86% | 50.00% | larger-model improvement |
| ALL_MODELS | 59.54% | 38.97% | Qwen group aggregate |

Together, the two tables show that AgenQA path questions retain a measurable hard subset even for strong models, while also producing a clear accuracy gradient within the Qwen family.

This page still does not release the full raw benchmark set, source papers, raw solver responses, consensus JSON, model API configuration, or prompt snapshots.
