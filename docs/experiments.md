# Redacted Experiment Summary

[中文](./experiments.zh.md)

> The numbers here are preliminary and redacted for hiring-context demonstration. They summarize internal validation signals without releasing raw benchmark items, source materials, prompts, or solver outputs.

## What the Evaluation Tries to Show

AgenQA is evaluated as a benchmark-synthesis framework. The central questions are:

1. Can the generated benchmark separate model capabilities rather than becoming trivially easy or uniformly unsolved?
2. Does the Edge/Path design reveal diagnostic information beyond final-answer accuracy?
3. Can online synthesis produce fresh benchmark instances while retaining quality-control signals?
4. Are the resulting artifacts reusable enough for benchmark construction and downstream data routing?

## Public Aggregate Signals

| Evidence type | Public summary | What it supports |
| --- | --- | --- |
| Validation scale | 37 synthesis runs / 175 generated QA candidates in an earlier validation phase | The system went beyond isolated demos |
| Batch run | 10-paper paper-seed batch; 9 successful tasks, 1 truncation failure | The pipeline can run across multiple sources with explicit failure accounting |
| Same-family gradient | Qwen-family scale checks showed clearer ability differences across model sizes | The benchmark has discriminative signal |
| SOTA comparison | Stronger SOTA models remained ahead in aggregate checks | The generated tasks are not random noise |
| Edge/Path gap | Edge views remain more locally solvable than folded Path views | Hidden intermediate dependencies add measurable difficulty |
| Depth-conditioned trend | Longer dependency paths create sharper drop-off for weaker solvers | Difficulty can be analyzed structurally, not only by final accuracy |

## Example Table Shape

The final public paper will use exact, audited values. For this showcase, the table below shows the intended reporting shape.

| Model group | Edge accuracy | Path accuracy | Edge-Path gap | Interpretation |
| --- | --- | --- | --- | --- |
| smaller same-family model | redacted | redacted | positive | local support helps, hidden path is harder |
| larger same-family model | redacted | redacted | positive but lower | stronger reconstruction ability |
| SOTA model group | redacted | redacted | positive | strongest aggregate performance |

## Quality Gates

Accepted chains are selected using harness-produced signals rather than raw generated text alone:

- Edge solver correctness and well-posedness checks;
- Path solver distribution and difficulty signal;
- multi-strong consensus;
- answer-contract and world-contract diagnostics;
- dependency-certificate integrity;
- revision history and final diagnostic comments.

## What Is Not Released Here

This page does not release:

- generated benchmark questions;
- source papers or source documents;
- raw solver responses;
- consensus JSON files;
- model API configuration;
- prompt snapshots.
