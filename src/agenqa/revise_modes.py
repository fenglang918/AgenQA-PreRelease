"""Centralized normalization for revise modes."""

from __future__ import annotations

from typing import Any

REVISE_MODE_CORRECTNESS = "correctness"
REVISE_MODE_WORLD_CONTRACT = "world_contract"
REVISE_MODE_ANSWER_CONTRACT = "answer_contract"
REVISE_MODE_REUSE_HIDDEN = "reuse_hidden"
REVISE_MODE_QUALITY = "quality"

ALLOWED_REVISE_MODES = (
    REVISE_MODE_CORRECTNESS,
    REVISE_MODE_WORLD_CONTRACT,
    REVISE_MODE_ANSWER_CONTRACT,
    REVISE_MODE_REUSE_HIDDEN,
    REVISE_MODE_QUALITY,
)

_CORRECTNESS_ALIASES = {
    REVISE_MODE_CORRECTNESS,
    "correct",
    "fix",
    "fix_answer",
}

_WORLD_CONTRACT_ALIASES = {
    REVISE_MODE_WORLD_CONTRACT,
    "worldcontract",
    "world-contract",
}

_ANSWER_CONTRACT_ALIASES = {
    REVISE_MODE_ANSWER_CONTRACT,
    "answercontract",
    "answer-contract",
}

_REUSE_HIDDEN_ALIASES = {
    REVISE_MODE_REUSE_HIDDEN,
    "reuse-hidden",
    "reuse",
    "progression",
    "hidden_reuse",
    # Generic alias (semantics: structure/chain health).
    "structure",
}

_QUALITY_ALIASES = {
    REVISE_MODE_QUALITY,
    # Legacy aliases kept for backward compatibility.
    "difficulty",
    "hardness",
    "complexity",
    # Common shorthand.
    "theme",
    "topic",
    "improve",
    "improvement",
    "enhance",
    "enhancement",
}


def normalize_revise_mode(raw: Any) -> str | None:
    if raw is None:
        return None
    mode = str(raw).strip().lower()
    if not mode:
        return None
    if mode in _CORRECTNESS_ALIASES:
        return REVISE_MODE_CORRECTNESS
    if mode in _WORLD_CONTRACT_ALIASES:
        return REVISE_MODE_WORLD_CONTRACT
    if mode in _ANSWER_CONTRACT_ALIASES:
        return REVISE_MODE_ANSWER_CONTRACT
    if mode in _REUSE_HIDDEN_ALIASES:
        return REVISE_MODE_REUSE_HIDDEN
    if mode in _QUALITY_ALIASES:
        return REVISE_MODE_QUALITY
    return None


def is_reuse_hidden(raw: Any) -> bool:
    return normalize_revise_mode(raw) == REVISE_MODE_REUSE_HIDDEN


def is_correctness(raw: Any) -> bool:
    return normalize_revise_mode(raw) == REVISE_MODE_CORRECTNESS


def is_world_contract(raw: Any) -> bool:
    return normalize_revise_mode(raw) == REVISE_MODE_WORLD_CONTRACT


def is_answer_contract(raw: Any) -> bool:
    return normalize_revise_mode(raw) == REVISE_MODE_ANSWER_CONTRACT


def is_quality(raw: Any) -> bool:
    return normalize_revise_mode(raw) == REVISE_MODE_QUALITY
