"""Evaluation helpers for upstream/downstream audit workflows."""

from typing import Any

__all__ = ["run_path_hardcase_audit", "run_hard_question_cost_audit"]


def run_path_hardcase_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .path_hardcase_audit import run_path_hardcase_audit as _impl

    return _impl(*args, **kwargs)


def run_hard_question_cost_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .hard_question_cost_audit import run_hard_question_cost_audit as _impl

    return _impl(*args, **kwargs)
