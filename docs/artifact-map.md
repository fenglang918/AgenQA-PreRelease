# Artifact Map

[中文](./artifact-map.zh.md)

This page explains what AgenQA records during a run and how those records map to the system design.

## Why Artifacts Matter

AgenQA is designed so that a generated item is not only a final QA pair. Each run records intermediate state, solver-facing views, validation signals, and feedback used for repair. This makes the synthesis process auditable and debuggable.

## Run Structure

```text
run/
  run_config.json                  # run-level configuration
  state.json                       # final chain state
  run_playback.md                  # human-readable run summary
  00_Summary/
    final_comment.json             # final diagnostic summary
  round_1/
    step_0_init/
      director_decision.json       # why the run initializes
      inputs/                      # source-derived inputs
      subruns/                     # skill-level intermediate outputs
    step_1_extend/
      director_decision.json       # extend/revise/finish decision
      edge_kqa.json                # local Edge view
      path_kqa.json                # folded Path view
      path_folded_question.json    # solver-facing Path question
      answer_contract_report.json  # answer-format diagnostics
      subruns/
        draft_chain.json
        format.json
        step_cert_builder.json
        path_fold.json
      solve/
        solver_outputs.jsonl
        consensus_summary_edge.json
        consensus_summary_path.json
        ambiguity_report.json
```

## Artifact-to-System Mapping

| Artifact family | System role | Explanation |
| --- | --- | --- |
| Director decisions | control layer | records why the harness extends, revises, or finishes |
| Edge KQA | local verification view | exposes local support for step-level solving |
| Path KQA | global challenge view | hides intermediate facts for end-to-end reconstruction |
| Dependency certificates | audit layer | records which facts support each new answer-equivalent fact |
| Contract reports | stabilization layer | detects answer-format and semantic instability |
| Consensus summaries | evaluator layer | aggregates multiple solver judgments |
| Run playback | observability layer | turns run artifacts into readable summaries |
