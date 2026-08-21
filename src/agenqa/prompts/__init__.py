"""Python-style prompt package.

Role-specific prompts are implemented in dedicated modules (qa_init/solver/
extend/revise). Common entry points remain available via this package and the
legacy `agent_prompts` module.
"""

from __future__ import annotations

from .qa_init import QA_INIT_V2, QAI_TEMPLATE
from .solver import SOLVER_PROMPT, SOLVER_TEMPLATE
from .extend import (
    EXTEND_UPGRADE_V1,
    COMPRESS_HISTORY_PROMPT,
    PLAN_CRITIQUE_PROMPT,
    REFLECT_FUSE_PROMPT,
)
from .draft import (
    DRAFT_V1,
    DRAFT_REVISE_CORRECTNESS,
    DRAFT_REVISE_CORRECTNESS_TAGGED,
    DRAFT_REVISE_DIFFICULTY,
    DRAFT_REVISE_DIFFICULTY_TAGGED,
)
from .format import FORMAT_V1
from .diagnose import (
    DIAGNOSE_V1,
    DIAGNOSE_REVISE_CORRECTNESS,
    DIAGNOSE_REVISE_CORRECTNESS_TAGGED,
    DIAGNOSE_REVISE_DIFFICULTY,
    DIAGNOSE_REVISE_DIFFICULTY_TAGGED,
)
from .extract import EXTRACT_V1
from .director import DIRECTOR_TEMPLATE, build_director_v1_body

__all__ = [
    "QA_INIT_V2",
    "QAI_TEMPLATE",
    "SOLVER_PROMPT",
    "SOLVER_TEMPLATE",
    "EXTEND_UPGRADE_V1",
    "COMPRESS_HISTORY_PROMPT",
    "PLAN_CRITIQUE_PROMPT",
    "REFLECT_FUSE_PROMPT",
    "DRAFT_V1",
    "DRAFT_REVISE_CORRECTNESS",
    "DRAFT_REVISE_CORRECTNESS_TAGGED",
    "DRAFT_REVISE_DIFFICULTY",
    "DRAFT_REVISE_DIFFICULTY_TAGGED",
    "FORMAT_V1",
    "DIAGNOSE_V1",
    "DIAGNOSE_REVISE_CORRECTNESS",
    "DIAGNOSE_REVISE_CORRECTNESS_TAGGED",
    "DIAGNOSE_REVISE_DIFFICULTY",
    "DIAGNOSE_REVISE_DIFFICULTY_TAGGED",
    "EXTRACT_V1",
    "DIRECTOR_TEMPLATE",
    "build_director_v1_body",
]
