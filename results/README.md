# Aggregate Result Snapshots

[中文](./README.zh.md)

This directory makes the evaluation tables in
[`docs/experiments.md`](../docs/experiments.md) available as small,
machine-readable CSV snapshots.

| File | Content |
| --- | --- |
| `path_view_strong_solvers.csv` | Strong-solver counts and accuracy on all Path View evaluations and the diagnostic subset |
| `qwen_scale_consistency.csv` | Qwen-family aggregate accuracy by model scale |
| `training_transfer_2k.csv` | Early downstream transfer scores for the baseline and two trained variants |
| `manifest.json` | Scope, granularity, and release boundaries |

These are aggregate previews, not raw experiment dumps. They intentionally omit
generated questions, source papers, per-item judgments, model responses,
service metadata, and private run paths. The counts in the strong-solver table
are evaluation attempts and should not be read as unique-question counts; the
manifest separately records the `547` unique Path View questions in that
snapshot.

The training table is an early signal rather than a final statistical claim.
It does not include variance, confidence intervals, or paired tests.
