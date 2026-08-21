"""Solver feedback payload schema (code truth source).

Defines the authoritative description of the `solver_feedback` payload that gets
injected into Director/Diagnose prompts via the state JSON.
"""

from __future__ import annotations

from textwrap import dedent

SOLVER_FEEDBACK_PAYLOAD_DESCRIPTION_ZH = dedent(
    """\
    ## solver_feedback（注入 Director/Diagnose 的反馈摘要）

    `solver_feedback` 是“基于 multi-strong 结果构造的反馈摘要”，用于帮助 Director 判定是否需要 Revise（correctness / answer_contract / world_contract / reuse_hidden）。

    字段约定：
    - `from_step`: int，反馈来源的 step。
    - `mode`: "unanimous" | "contrast" | null。
      - `unanimous`：multi-strong 显式结果同向（全对或全错），摘要只保留 primary。
      - `contrast`：multi-strong 存在分歧，摘要保留“一对一错”的对照反馈。
    - `view`: "edge" | "path" | null，摘要优先来源视角。
    - `from_tier`: "strong" | "strong_multi" | "medium"，反馈来源 tier。
    - `question_well_posed`: bool | null，题目是否 well-posed（条件自洽、信息充分、可唯一求解）。
    - `correctness_feedback`: str | null，正确性/可解性反馈；若 `question_well_posed=false`，需指出问题点与修复建议。
    - `difficulty_feedback`: str | null，难度/加难建议；通常仅在题目 well-posed 且 solver 解对时提供。
    - `selected`: list[object] | null，被选中的 solver 条目元信息（solver_idx/correct/service_id/model/token_ratio）。
    """
)

SOLVER_FEEDBACK_PAYLOAD_DESCRIPTION_EN = dedent(
    """\
    ## solver_feedback (feedback summary injected into Director/Diagnose)

    `solver_feedback` is a compact summary derived from multi-strong results for the current step,
    used by Director to decide whether to Revise (correctness / answer_contract / world_contract / reuse_hidden).

    Field contract:
    - `from_step`: int, the source step.
    - `mode`: "unanimous" | "contrast" | null.
      - `unanimous`: explicit multi-strong outcomes are aligned (all correct or all incorrect); keep primary feedback only.
      - `contrast`: explicit multi-strong outcomes disagree; keep one-correct vs one-incorrect contrast feedback.
    - `view`: "edge" | "path" | null, preferred source view.
    - `from_tier`: "strong" | "strong_multi" | "medium", source tier.
    - `question_well_posed`: bool | null, whether the question is well-posed (consistent, sufficient, uniquely solvable).
    - `correctness_feedback`: str | null, well-posedness/correctness feedback; if `question_well_posed=false`, state concrete issues + fixes.
    - `difficulty_feedback`: str | null, difficulty/hardening suggestions; typically provided only when well-posed and solved correctly.
    - `selected`: list[object] | null, metadata of selected solver rows (solver_idx/correct/service_id/model/token_ratio).
    """
)
