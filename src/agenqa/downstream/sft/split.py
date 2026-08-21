"""Episode-level split helpers for downstream SFT export."""

from __future__ import annotations

import hashlib


def assign_episode_split(
    episode_id: str,
    *,
    eval_ratio: float = 0.0,
    test_ratio: float = 0.0,
    seed: int = 17,
) -> str:
    if eval_ratio < 0 or test_ratio < 0 or eval_ratio + test_ratio >= 1:
        raise ValueError("eval_ratio and test_ratio must be non-negative and sum to less than 1.")

    if eval_ratio == 0 and test_ratio == 0:
        return "train"

    digest = hashlib.md5(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)

    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + eval_ratio:
        return "eval"
    return "train"
