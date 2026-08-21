"""Consensus signals schema description (code truth source).

This module defines the authoritative semantics for multi-strong solver
consensus signals written into `AgentState.solver_consensus`.
"""

from __future__ import annotations

from textwrap import dedent


SOLVER_CONSENSUS_DESCRIPTION_ZH = dedent(
    """\
    关于 `solver_consensus`（多 strong solver 共识信号）：

    - 位置：`solver_consensus.strong`。
    - 目的：用多个 strong solver 的“投票信号”辅助区分：
      - 当前题目是否 well-posed（条件充分、选项唯一、可判定）；
      - 链路产出的 `Answer`（proposed_answer）是否可能存在错误。

    字段说明（`solver_consensus.strong.*`）：
    - `mode`: `"none" | "always"`，表示共识机制的启用/触发策略（none=不启用）。
    - `proposed_answer`: 当前链路产出的答案文本（来自 Format/Revise 的 Answer 字段，原始字符串）。
    - `solvers`: strong solver 的逐个观测列表（便于诊断/回放），常见字段包括：
      - `solver_idx`: solver 序号（按 `solvers.strong` 列表顺序，从 0 开始）
      - `service_id` / `model`: 标识该 solver 的服务或模型
      - `status`: `"success" | "timeout" | "error" | "parse_error"`（以产物解析为准）
      - `answer` / `answer_normalized`: solver 的预测答案及其规范化形式（用于投票口径）
      - `question_well_posed`: solver 对题目是否 well-posed 的判断（若有）
      - `correctness_feedback` / `difficulty_feedback`: solver 反馈（若有）
    - `eligible_votes`: 用于“答案投票”的有效票数（通常要求 status=success 且 answer_normalized 可解析）。
    - `answer_consensus`: 若存在严格多数共识，则给出共识答案（规范化后）；否则为 null。
    - `consensus_strength`: 共识票数（无共识时为 0）。
    - `tie`: 是否无严格多数（无共识/票数不足均视为 tie）。
    - `tie_reason`: 无共识原因（例如 `"insufficient_votes"` / `"all different"` / `"2-way split"`）。
    - `wellposed_consensus`: 对 `question_well_posed` 的多数投票结果；若无有效票或无多数则为 null。
    - `differs_from_proposed`: 当 `answer_consensus` 与 `proposed_answer` 都可规范化时，表示两者是否不同；否则为 null。

    使用建议（面向 Director/算子，原则化而非机械阈值）：
    - 若 `wellposed_consensus=false`：优先怀疑题目设定或选项存在问题（条件不足/矛盾/不唯一），倾向 Revise 修题。
    - 若 `answer_consensus` 存在且 `differs_from_proposed=true` 且 `wellposed_consensus=true`：更可能是链路答案有误，倾向 Revise 修答案（同时结合 solver_feedback 与历史上下文）。
    - 若 `tie=true` 或 `eligible_votes` 很少：信号不确定，回退结合 medium 与逐个 strong solver 证据综合判断。
    """
)


SOLVER_CONSENSUS_DESCRIPTION_EN = dedent(
    """\
    About `solver_consensus` (multi-strong solver consensus signals):

    - Location: `solver_consensus.strong`.
    - Purpose: use votes from multiple strong solvers to help distinguish:
      - whether the question is well-posed (sufficient conditions, unique answer);
      - whether the chain-proposed `Answer` is likely wrong.

    Field semantics (`solver_consensus.strong.*`):
    - `mode`: `"none" | "always"`.
    - `proposed_answer`: the chain-proposed answer string (raw).
    - `solvers`: per-solver observations (for debugging/replay), including:
      `solver_idx`, `service_id`/`model`, `status`, `answer`/`answer_normalized`,
      `question_well_posed`, and optional feedback fields.
    - `eligible_votes`: number of eligible votes for answer voting.
    - `answer_consensus`: consensus answer when a strict majority exists; otherwise null.
    - `consensus_strength`: number of votes supporting the consensus answer.
    - `tie` / `tie_reason`: indicates no strict majority and why.
    - `wellposed_consensus`: majority vote for well-posedness; null if no majority.
    - `differs_from_proposed`: whether `answer_consensus` differs from `proposed_answer` (when comparable).

    Usage guidance (principle-based):
    - If `wellposed_consensus=false`, prioritize revising the question/spec.
    - If consensus exists and differs from proposed while well-posed, prioritize revising the proposed answer.
    - If signals are inconclusive (tie/insufficient votes), fall back to medium plus per-strong evidence.
    """
)


__all__ = ["SOLVER_CONSENSUS_DESCRIPTION_ZH", "SOLVER_CONSENSUS_DESCRIPTION_EN"]
