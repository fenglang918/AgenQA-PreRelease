# Evaluation Snapshot

[中文](./experiments.zh.md)

This page summarizes several pre-release evaluation snapshots. They show that AgenQA-generated **Path View** questions already provide useful benchmark-construction signal and early evidence that a small amount of AgenQA data can serve as downstream training signal.

The paper framing of AgenQA separates the roles of Edge and Path views: Edge Views support step-level correctness control, while Path-Fold creates solver-facing Path View questions. This page keeps three complementary checks: two benchmark-construction signals, including the difficulty / discriminative region among strong solvers and the Qwen-family scale-consistency sanity check; and one early downstream training result that probes whether a small amount of AgenQA data transfers to external reasoning benchmarks.

## Table 1. SOTA Solvers on Path View Questions

This table comes from a stronger problem-specification setting where problem constraints, answer equivalence, and output boundaries were written more explicitly. The evaluation covers `96` synthesis runs / `547` solver-facing Path View questions. The diagnostic subset contains questions missed by at least one solver and highlights the more discriminative region of the benchmark.

| Solver | All Path View questions | Accuracy | Diagnostic subset | Diagnostic accuracy |
| --- | ---: | ---: | ---: | ---: |
| claude-sonnet-4-6 | 466 / 574 | 81.18% | 63 / 150 | 42.00% |
| gemini-3.1-pro-preview | 254 / 294 | 86.39% | 47 / 78 | 60.26% |
| glm-5 | 494 / 574 | 86.06% | 89 / 150 | 59.33% |
| gpt-5.4-2026-03-05 | 469 / 574 | 81.71% | 62 / 150 | 41.33% |
| qwen3.5-plus | 498 / 575 | 86.61% | 90 / 150 | 60.00% |
| All solvers | 2181 / 2591 | 84.18% | 351 / 678 | 51.77% |

## Table 2. Qwen-Family Model-Scale Correlation

This table comes from an earlier set of `37` synthesis runs / `175` Path View questions. It shows that, within the same model family, benchmark accuracy is positively correlated with model scale / expected capability. This is not the main strong-model discrimination claim; it is a scale-consistency sanity check that the benchmark behaves in the expected capability order and can complement item-level human review.

| Solver | All Path View accuracy | Diagnostic-subset accuracy | Interpretation |
| --- | ---: | ---: | --- |
| qwen3-4b | 48.00% | 21.55% | lower-capacity baseline |
| qwen3-8b | 56.57% | 34.48% | mid-scale improvement |
| Qwen3-32B | 66.86% | 50.00% | larger-model improvement |
| All solvers | 59.54% | 38.97% | family-level aggregate |

## Table 3. Early Training Transfer from 2K AgenQA Examples

This table comes from an early downstream training evaluation. Starting from `Qwen3-4B-Instruct`, `instruct-gspo` and `instruct-grpo` were trained with around `2,000` AgenQA examples and evaluated on external math / science benchmarks with `32` averaged evaluation runs. This is not presented as a final paper claim; it is an early training-utility signal: whether a small amount of structured AgenQA data can produce positive transfer without a visible GPQA-family trade-off.

| Model | aim24 | HMMT-FEB | GPQA-Diamond | GPQA | SciBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct baseline | 59.62 | 25.00 | 59.90 | 54.95 | 31.71 |
| instruct-gspo | 61.98 | 25.94 | 59.85 | 54.99 | 32.76 |
| instruct-grpo | 61.98 | 26.87 | 59.79 | 55.15 | 32.12 |

Changes relative to the baseline:

| Model | aim24 | HMMT-FEB | GPQA-Diamond | GPQA | SciBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| instruct-gspo | +2.36 | +0.94 | -0.05 | +0.04 | +1.05 |
| instruct-grpo | +2.36 | +1.87 | -0.11 | +0.20 | +0.41 |

The clearest positive signal is `aim24`, where both trained models improve by `+2.36`. `HMMT-FEB` improves most under `instruct-grpo`, while `SciBench` shows a positive signal especially under `instruct-gspo`. `GPQA` and `GPQA-Diamond` remain essentially unchanged; the `-0.05 / -0.11` differences on `GPQA-Diamond` should not be interpreted as a real degradation without variance, confidence intervals, or a paired test.

Together, the three tables show that AgenQA's Path-Fold does more than create longer problem statements: it preserves a generator-side dependency state while producing solver-facing benchmark items with measurable difficulty, strong-solver discrimination, and model-scale consistency. The 2K training snapshot also provides an early downstream training-utility signal.
