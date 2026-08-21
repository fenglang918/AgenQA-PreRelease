"""Python-style prompt sources for agent flow.

This module is now a thin re-export layer to keep backward compatibility:
role-specific prompt definitions live in dedicated submodules.
"""

from __future__ import annotations

from .qa_init import QA_INIT_V2, QAI_TEMPLATE, PAPER_BRIEF_PROMPT, QA_INIT_V2_EN, QAI_TEMPLATE_EN, PAPER_BRIEF_PROMPT_EN
from .solver import SOLVER_PROMPT, SOLVER_TEMPLATE, SOLVER_PROMPT_EN, SOLVER_TEMPLATE_EN
from .extend import (
    EXTEND_UPGRADE_V1,
    COMPRESS_HISTORY_PROMPT,
    COMPRESS_HISTORY_PROMPT_EN,
    PLAN_CRITIQUE_PROMPT,
    PLAN_CRITIQUE_PROMPT_EN,
    REFLECT_FUSE_PROMPT,
    REFLECT_FUSE_PROMPT_EN,
)

__all__ = [
    "QA_INIT_V2",
    "QAI_TEMPLATE",
    "QA_INIT_V2_EN",
    "QAI_TEMPLATE_EN",
    "PAPER_BRIEF_PROMPT",
    "PAPER_BRIEF_PROMPT_EN",
    "SOLVER_PROMPT",
    "SOLVER_TEMPLATE",
    "SOLVER_PROMPT_EN",
    "SOLVER_TEMPLATE_EN",
    "EXTEND_UPGRADE_V1",
    "COMPRESS_HISTORY_PROMPT",
    "COMPRESS_HISTORY_PROMPT_EN",
    "PLAN_CRITIQUE_PROMPT",
    "PLAN_CRITIQUE_PROMPT_EN",
    "REFLECT_FUSE_PROMPT",
    "REFLECT_FUSE_PROMPT_EN",
]
