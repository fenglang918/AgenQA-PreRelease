# 产物地图

[English](./artifact-map.md)

本页解释 AgenQA 一次 run 会记录哪些信息，但不公开 raw artifacts。

## 为什么产物重要

AgenQA 生成的不是一个孤立 QA pair。每次 run 都会记录中间状态、solver-facing views、验证信号和用于修复的反馈。这样，整个 synthesis process 可以被审计、调试和复盘。

## Redacted Run Structure

```text
run/
  run_config.json                  # 脱敏配置快照
  state.json                       # 最终 chain state，本仓库不公开
  run_playback.md                  # 人类可读 run summary
  00_Prompts_Snapshot/             # public preview 不发布
  00_Summary/
    final_comment.json             # 最终诊断摘要
  round_1/
    step_0_init/
      director_decision.json       # 初始化原因
      inputs/                      # source-derived inputs，本仓库不公开
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
        solver_outputs.jsonl       # 本仓库不公开
        consensus_summary_edge.json
        consensus_summary_path.json
        ambiguity_report.json
```

## Artifact-to-System Mapping

| Artifact family | System role | Public explanation |
| --- | --- | --- |
| Director decisions | control layer | 记录 harness 为什么 extend、revise 或 finish |
| Edge KQA | local verification view | 暴露局部支撑，用于 step-level solving |
| Path KQA | global challenge view | 隐藏中间 facts，用于端到端重构 |
| Dependency certificates | audit layer | 记录每个新 answer-equivalent fact 由哪些 facts 支撑 |
| Contract reports | stabilization layer | 检测答案格式和语义不稳定 |
| Consensus summaries | evaluator layer | 聚合多个 solver judgments |
| Run playback | observability layer | 把内部 run artifacts 变成人类可读摘要 |

## Public Boundary

本仓库展示 artifact schema 和 workflow shape，不复制 raw data。Raw JSON outputs 可能包含 source-derived content、generated questions、solver responses 或 prompt traces，因此不进入这个 pre-release repo。
