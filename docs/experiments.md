# Evaluation Snapshot

[中文](./experiments.zh.md)

This page summarizes an evaluation snapshot from the March 26, 2026 project report. It shows that AgenQA-generated **Path View** questions already provide useful benchmark-construction signal.

The paper framing of AgenQA separates the roles of Edge and Path views: Edge Views support step-level correctness control, while Path-Fold creates solver-facing Path View questions whose solver distributions reveal global difficulty and discrimination signal. This page therefore keeps only two tables: one for strong solvers on Path View questions, and one for the positive correlation between benchmark accuracy and model scale / expected capability within the Qwen family.

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

This table comes from an earlier set of `37` synthesis runs / `175` Path View questions. It shows that, within the same model family, benchmark accuracy is positively correlated with model scale / expected capability, suggesting that the tasks reflect solver capability differences rather than random noise.

| Solver | All Path View accuracy | Diagnostic-subset accuracy | Interpretation |
| --- | ---: | ---: | --- |
| qwen3-4b | 48.00% | 21.55% | lower-capacity baseline |
| qwen3-8b | 56.57% | 34.48% | mid-scale improvement |
| Qwen3-32B | 66.86% | 50.00% | larger-model improvement |
| All solvers | 59.54% | 38.97% | family-level aggregate |

Together, the two tables show that AgenQA's Path-Fold does more than create longer problem statements: it preserves a generator-side dependency state while producing solver-facing benchmark items with measurable difficulty and model-discrimination signal.
