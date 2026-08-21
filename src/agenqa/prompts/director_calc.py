"""Director prompt variants emphasizing computation-based difficulty judgments."""

from __future__ import annotations

from textwrap import dedent
import json
from typing import Any, Dict, List

from ._base import PromptSection, PromptTemplate
from .director import (
    DIRECTOR_ACTIONS_SECTION,
    DIRECTOR_ACTIONS_SECTION_EN,
    DIRECTOR_COGNITION_HEADER,
    DIRECTOR_COGNITION_HEADER_EN,
    DIRECTOR_INFO_HEADER,
    DIRECTOR_INFO_HEADER_EN,
    DIRECTOR_OUTPUT_SECTION,
    DIRECTOR_OUTPUT_SECTION_EN,
    DIRECTOR_ROLE_SECTION,
    DIRECTOR_ROLE_SECTION_EN,
    DIRECTOR_STATE_SECTION,
    DIRECTOR_STATE_SECTION_EN,
)
from .common import (
    COMMON_KNOWN_TREE,
    COMMON_KNOWN_TREE_EN,
    COMMON_EDGE_QA_VS_PATH,
    COMMON_EDGE_QA_VS_PATH_EN,
    COMMON_QUESTION_TYPE_CAPABILITIES,
    COMMON_QUESTION_TYPE_CAPABILITIES_EN,
    COMMON_SOLVER_SIGNALS_COGNITION,
    COMMON_SOLVER_SIGNALS_COGNITION_EN,
)

__all__ = [
    "DIRECTOR_TEMPLATE_CALC",
    "DIRECTOR_TEMPLATE_CALC_EN",
    "build_director_v1_body_calc",
]


DIRECTOR_DIFFICULTY_FOCUS_SECTION = PromptSection(
    text=dedent(
        """\
        ## 难度口径（计算 + 推导）

        - 提升难度优先通过“可复现的复杂推导/计算负担”（例如代数消元/积分/线代/级数/概率期望等）；但务必基于论文背景（episode_seed 的 anchor；可选 subject/keywords + premise_bank），并结合必要的垂类知识点提升难度，避免脱离论文主题出“泛化玩具题”。
        - 优先参考 `metrics.edge.kq_tokens_step_ratio_*` / `metrics.edge.completion_tokens_step_ratio_*`（以及对应的 `metrics.path.*`）的 step-to-step 趋势信号；`token_ratio` 是风格敏感的弱参考，不能作为绝对难度或跨模型比较依据。
        - 若结构化 solver 结果里的 `difficulty_feedback` 与对错信号冲突，优先相信 edge/path 的结构化 solver 结果与 well-posedness；difficulty_feedback 主要用于“如何加难/如何修订”的方向提示。
        - 硬规则补充：`edge strong` 负责 correctness/well-posedness；`path strong` 负责链路可达性与难度分层。若 `edge strong` 全对且 `path strong` 部分对（mixed），默认视为正向高区分度信号并倾向 Extend；仅在存在明确 Type1/Type2 或 path-fold 泄露/不自包含证据时才转为 Revise。
        - 口径补充：若 `solver_metrics.edge.strong` 整体可解、但 `solver_metrics.path.strong` 出现失败/分裂，这可能是预期信号（path 难解不等于结构问题）。不要仅据此推断结构问题或选择 `reuse_hidden`；只有在 path-fold 题面/notes 中能直接指出指针式引用/不自包含/泄露证据时，才允许选 `ReviseMode=reuse_hidden`。
        """
    )
)

DIRECTOR_DIFFICULTY_FOCUS_SECTION_EN = PromptSection(
    text=dedent(
        """\
        ## Difficulty criteria (computation + derivation)

        - Prioritize difficulty from reproducible, computation-heavy derivation (e.g., elimination/integrals/linear algebra/series/expectations); but stay grounded in the paper context (episode_seed anchor; optional subject/keywords + premise_bank) and use necessary domain knowledge accordingly—avoid drifting into generic toy problems unrelated to the paper.
        - Prefer the step-to-step trend signals `metrics.edge.kq_tokens_step_ratio_*` / `metrics.edge.completion_tokens_step_ratio_*` (and the corresponding `metrics.path.*`); `token_ratio` is style-sensitive and must not be treated as absolute difficulty or compared across models.
        - If `difficulty_feedback` inside the structured solver results conflicts with correctness/well-posedness signals, trust the structured edge/path solver results and well-posedness first; use difficulty_feedback mainly for actionable “how to make it harder / how to revise” directions.
        - Hard-rule addendum: `edge strong` is the authority for correctness/well-posedness, while `path strong` is for reachability and difficulty stratification. If `edge strong` is all-correct and `path strong` is mixed (partially-correct), treat this as a positive high-discrimination signal and prefer Extend by default; switch to Revise only with explicit Type1/Type2 evidence or explicit path-fold leakage/non-self-contained evidence.
        - Clarification: if `solver_metrics.edge.strong` looks solvable while `solver_metrics.path.strong` shows failures/splits, this can be expected (path being hard does not imply a structural issue). Do not infer structural problems or choose `reuse_hidden` from this gap alone; only choose `ReviseMode=reuse_hidden` if you can directly point to evidence in the path-fold prompt/notes (pointer reference / not self-contained / leakage).
        """
    )
)


DIRECTOR_TEMPLATE_CALC = PromptTemplate(
    name="director_v1_calc",
    sections=[
        DIRECTOR_ROLE_SECTION,
        DIRECTOR_COGNITION_HEADER,
        COMMON_EDGE_QA_VS_PATH,
        COMMON_SOLVER_SIGNALS_COGNITION,
        DIRECTOR_DIFFICULTY_FOCUS_SECTION,
        COMMON_KNOWN_TREE,
        COMMON_QUESTION_TYPE_CAPABILITIES,
        DIRECTOR_INFO_HEADER,
        DIRECTOR_STATE_SECTION,
        DIRECTOR_ACTIONS_SECTION,
        DIRECTOR_OUTPUT_SECTION,
    ],
)

DIRECTOR_TEMPLATE_CALC_EN = PromptTemplate(
    name="director_v1_calc_en",
    sections=[
        DIRECTOR_ROLE_SECTION_EN,
        DIRECTOR_COGNITION_HEADER_EN,
        COMMON_EDGE_QA_VS_PATH_EN,
        COMMON_SOLVER_SIGNALS_COGNITION_EN,
        DIRECTOR_DIFFICULTY_FOCUS_SECTION_EN,
        COMMON_KNOWN_TREE_EN,
        COMMON_QUESTION_TYPE_CAPABILITIES_EN,
        DIRECTOR_INFO_HEADER_EN,
        DIRECTOR_STATE_SECTION_EN,
        DIRECTOR_ACTIONS_SECTION_EN,
        DIRECTOR_OUTPUT_SECTION_EN,
    ],
)


def build_director_v1_body_calc(payload: Dict[str, Any], allowed_ops: List[str], *, lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    template = DIRECTOR_TEMPLATE_CALC_EN if lang_norm in {"en", "english"} else DIRECTOR_TEMPLATE_CALC
    ctx: Dict[str, Any] = {
        "state_json_pretty": json.dumps(payload, ensure_ascii=False, indent=2),
        "available_operations_json": json.dumps(allowed_ops, ensure_ascii=False),
    }
    return template.render_body(ctx)
