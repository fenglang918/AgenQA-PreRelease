"""Solver prompt variants emphasizing computation-derived difficulty signals."""

from __future__ import annotations

from textwrap import dedent

from ._base import PromptSection, PromptTemplate
from .solver import SOLVER_TEMPLATE, SOLVER_TEMPLATE_EN

__all__ = [
    "SOLVER_PROMPT_CALC",
    "SOLVER_TEMPLATE_CALC",
    "SOLVER_PROMPT_CALC_EN",
    "SOLVER_TEMPLATE_CALC_EN",
]


_SOL_DIFFICULTY_GUIDANCE = PromptSection(
    text=dedent(
        """\
        ## 难度反馈口径（重要）
        - 填写 `DifficultyFeedback` 时，优先依据“推导 + 计算负担”（符号运算/积分/线代/级数/概率等）；必要的垂类知识点可作为辅助难度来源，但不要把冷门名词/纯知识覆盖当作主要难度。
        - “加难建议”优先通过增加可复现的计算/推导约束实现；不要建议仅靠写更长文字来变难。
        - 输出必须简短、可消费：`DifficultyFeedback` 用一行文本（建议 <= 240 字），按如下固定格式写：
          `driver=<computation|reasoning|both>; comp=<一句>; reason=<一句>; suggest=<1-2条, 只增计算/推导>`
        """
    )
)

_SOL_DIFFICULTY_GUIDANCE_EN = PromptSection(
    text=dedent(
        """\
        ## Difficulty feedback guidance (important)
        - When filling `DifficultyFeedback`, prioritize derivation + computation burden (semantic manipulation / integrals / linear algebra / series / expectations); domain knowledge can contribute when necessary, but avoid relying mainly on niche jargon or trivia.
        - Harder suggestions should mainly add reproducible computation/derivation constraints; do not suggest “make it harder by writing longer text”.
        - Keep it compact and machine-consumable: write `DifficultyFeedback` as a single line (recommended <= 240 chars) using:
          `driver=<computation|reasoning|both>; comp=<one sentence>; reason=<one sentence>; suggest=<1-2 items, computation/derivation only>`
        """
    )
)


SOLVER_TEMPLATE_CALC = PromptTemplate(
    name="solver_v1_calc_mod",
    sections=[*SOLVER_TEMPLATE.sections[:-1], _SOL_DIFFICULTY_GUIDANCE, SOLVER_TEMPLATE.sections[-1]],
)
SOLVER_PROMPT_CALC = SOLVER_TEMPLATE_CALC.render_body({})


SOLVER_TEMPLATE_CALC_EN = PromptTemplate(
    name="solver_v1_calc_en",
    sections=[*SOLVER_TEMPLATE_EN.sections[:-1], _SOL_DIFFICULTY_GUIDANCE_EN, SOLVER_TEMPLATE_EN.sections[-1]],
)
SOLVER_PROMPT_CALC_EN = SOLVER_TEMPLATE_CALC_EN.render_body({})
