"""Executable track error taxonomy (single source of truth).

This module defines stable, coarse-grained error types that can be used by:
- evaluators (code_solve / code_consensus) to classify failures deterministically
- Director / Memory to make decisions and diagnose trends

The taxonomy is intentionally conservative: it should be stable across models and
runtime environments, and should not rely on fragile heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


TYPE_SUCCESS = "success"
TYPE_WRONG_OUTPUT = "wrong_output"
TYPE_TIMEOUT = "timeout"
TYPE_PARSE_ERROR = "parse_error"
TYPE_DEPENDENCY_VIOLATION = "dependency_violation"
TYPE_CONTRACT_VIOLATION = "contract_violation"
TYPE_RUNTIME_ERROR = "runtime_error"
TYPE_SOLVER_FAILED = "solver_failed"
TYPE_GOLDEN_VERIFY_FAILED = "golden_verify_failed"
TYPE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExecutableErrorClassification:
    type: str
    recoverable: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "recoverable": bool(self.recoverable),
            "message": self.message,
        }


def classify_executable_error(error: Optional[str], *, correct: Optional[bool] = None) -> ExecutableErrorClassification:
    err = (error or "").strip()
    if correct is True and not err:
        return ExecutableErrorClassification(type=TYPE_SUCCESS, recoverable=False, message="")

    low = err.lower()
    if "golden_verify_failed" in low:
        return ExecutableErrorClassification(type=TYPE_GOLDEN_VERIFY_FAILED, recoverable=True, message=err)
    if "timeout" in low or "timed out" in low:
        return ExecutableErrorClassification(type=TYPE_TIMEOUT, recoverable=True, message=err)
    if "parse" in low and "json" in low:
        return ExecutableErrorClassification(type=TYPE_PARSE_ERROR, recoverable=True, message=err)
    if "moduleNotFoundError".lower() in low or "importerror" in low:
        return ExecutableErrorClassification(type=TYPE_DEPENDENCY_VIOLATION, recoverable=True, message=err)
    if "dependency" in low and ("not allowed" in low or "whitelist" in low or "violation" in low):
        return ExecutableErrorClassification(type=TYPE_DEPENDENCY_VIOLATION, recoverable=True, message=err)
    if "typeerror" in low or "nameerror" in low or "attributeerror" in low:
        return ExecutableErrorClassification(type=TYPE_CONTRACT_VIOLATION, recoverable=True, message=err)
    if "syntaxerror" in low or "indentationerror" in low:
        return ExecutableErrorClassification(type=TYPE_CONTRACT_VIOLATION, recoverable=True, message=err)
    if "solver_failed" in low:
        return ExecutableErrorClassification(type=TYPE_SOLVER_FAILED, recoverable=True, message=err)
    if err:
        return ExecutableErrorClassification(type=TYPE_RUNTIME_ERROR, recoverable=True, message=err)
    # No explicit error string, but failed correctness.
    return ExecutableErrorClassification(type=TYPE_WRONG_OUTPUT, recoverable=True, message="wrong_output")


def classify_row(row: Any) -> ExecutableErrorClassification:
    if not isinstance(row, dict):
        return ExecutableErrorClassification(type=TYPE_UNKNOWN, recoverable=True, message="missing_or_invalid_row")
    err = row.get("error")
    correct = row.get("correct")
    try:
        correct_bool = bool(correct) if correct is not None else None
    except Exception:
        correct_bool = None
    err_str = str(err) if isinstance(err, str) else (str(err) if err is not None else "")
    return classify_executable_error(err_str, correct=correct_bool)


__all__ = [
    "ExecutableErrorClassification",
    "classify_executable_error",
    "classify_row",
    "TYPE_SUCCESS",
    "TYPE_WRONG_OUTPUT",
    "TYPE_TIMEOUT",
    "TYPE_PARSE_ERROR",
    "TYPE_DEPENDENCY_VIOLATION",
    "TYPE_CONTRACT_VIOLATION",
    "TYPE_RUNTIME_ERROR",
    "TYPE_SOLVER_FAILED",
    "TYPE_GOLDEN_VERIFY_FAILED",
    "TYPE_UNKNOWN",
]
