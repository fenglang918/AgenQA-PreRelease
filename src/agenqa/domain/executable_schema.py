"""Executable track schema (single source of truth).

This module defines the minimal data structures for SciCode-like multi-step
executable problems. It is intentionally lightweight and dependency-free so it can
be imported by the core pipeline/state layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutableSubStep:
    """Definition of a single sub-step (unit-testable function/interface)."""

    step_number: str
    step_description: str
    function_header: str
    return_line: str
    step_background: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_description": self.step_description,
            "function_header": self.function_header,
            "return_line": self.return_line,
            "step_background": self.step_background,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExecutableSubStep":
        return ExecutableSubStep(
            step_number=str(data.get("step_number") or ""),
            step_description=str(data.get("step_description") or ""),
            function_header=str(data.get("function_header") or ""),
            return_line=str(data.get("return_line") or ""),
            step_background=(str(data.get("step_background")) if data.get("step_background") is not None else None),
        )


@dataclass
class ExecutableTestCase:
    """A single executable test snippet (assertions) for a sub-step."""

    test_id: str
    test_code: str
    derived_from_golden: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_code": self.test_code,
            "derived_from_golden": bool(self.derived_from_golden),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExecutableTestCase":
        return ExecutableTestCase(
            test_id=str(data.get("test_id") or ""),
            test_code=str(data.get("test_code") or ""),
            derived_from_golden=bool(data.get("derived_from_golden", True)),
        )


FIELD_EXECUTABLE_SUITABLE = "executable_suitable"
FIELD_NOTES = "notes"
FIELD_TASK_SKETCH = "task_sketch"
FIELD_INITIAL_SUB_STEPS = "initial_sub_steps"
FIELD_ESTIMATED_DIFFICULTY = "estimated_difficulty"

FIELD_STEP_NUMBER = "step_number"
FIELD_SUB_STEP = "sub_step"
FIELD_GOLDEN_STEP_CODE = "golden_step_code"
FIELD_DEPENDENCIES = "dependencies"
FIELD_REQUIRED_FACT_IDS = "required_fact_ids"
FIELD_PRIMARY_REQUIRED_FACT_ID = "primary_required_fact_id"
FIELD_REUSE_PLAN = "reuse_plan"

FIELD_TEST_INPUTS = "test_inputs"
FIELD_INPUTS_ARTIFACT_RELPATH = "inputs_artifact_relpath"
FIELD_INPUTS_SHA256 = "inputs_sha256"
FIELD_TEST_CASES_FOR_STEP = "test_cases_for_step"


@dataclass
class ExecutableExtractOutput:
    """Extract result for executable generation (step-wise pipeline)."""

    executable_suitable: bool
    notes: str
    task_sketch: str
    initial_sub_steps: List[ExecutableSubStep]
    estimated_difficulty: Optional[str] = None


@dataclass
class ExecutableDraftOutput:
    """Draft result for a single step (next sub-step only)."""

    step_number: str
    sub_step: ExecutableSubStep
    golden_step_code: str
    dependencies: str
    # Reuse contract (align semantic draft_chain semantics).
    required_fact_ids: List[str] = field(default_factory=list)
    primary_required_fact_id: Optional[str] = None
    reuse_plan: List[str] = field(default_factory=list)


@dataclass
class ExecutableTestDeriveOutput:
    """Test derivation result for a single step."""

    inputs: List[Dict[str, Any]]
    inputs_artifact_relpath: str
    inputs_sha256: str
    test_cases_for_step: List[ExecutableTestCase]


def executable_extract_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    notes_desc = "brief, non-CoT rationale" if use_en else "简短、非 CoT 的理由说明"
    steps_desc = "0-1 initial sub-steps only" if use_en else "仅允许 0-1 个起步 sub-step"
    difficulty_desc = "optional: easy/medium/hard" if use_en else "可选：easy/medium/hard"
    substep_schema = "\n".join(
        [
            ("ExecutableSubStep fields:" if use_en else "ExecutableSubStep 字段："),
            f"  - step_number: string",
            f"  - step_description: string",
            (
                "  - function_header: string  # include `def name(...):`; if `class Name:`, also include an indented `def __init__(...)` when the constructor args matter"
                if use_en
                else "  - function_header: string  # 至少包含 `def name(...):`；若为 `class Name:` 且构造参数重要，请同时给出缩进的 `def __init__(...)` 签名"
            ),
            (
                "  - return_line: string  # e.g. `return energy`"
                if use_en
                else "  - return_line: string  # 例如 `return energy`"
            ),
            f"  - step_background: string | null",
        ]
    )
    return "\n".join(
        [
            f"- {FIELD_EXECUTABLE_SUITABLE}: bool",
            f"- {FIELD_NOTES}: string  # {notes_desc}",
            f"- {FIELD_TASK_SKETCH}: string",
            f"- {FIELD_INITIAL_SUB_STEPS}: ExecutableSubStep[]  # {steps_desc}",
            f"- {FIELD_ESTIMATED_DIFFICULTY}: string | null  # {difficulty_desc}",
            "",
            substep_schema,
        ]
    )


def executable_draft_step_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    step_desc = "must be the next step number" if use_en else "必须是下一步的 step_number"
    code_desc = "code for this step only" if use_en else "只包含本步实现"
    required_desc = "fact ids reused from memory (step>=2)" if use_en else "从 memory 复用的 fact ids（step>=2）"
    reuse_desc = "brief per-fact reuse notes" if use_en else "逐条简述每个 fact 如何复用"
    substep_schema = "\n".join(
        [
            ("ExecutableSubStep fields:" if use_en else "ExecutableSubStep 字段："),
            f"  - step_number: string",
            f"  - step_description: string",
            (
                "  - function_header: string  # must include `def name(...):` or `class Name(...):`"
                if use_en
                else "  - function_header: string  # 必须包含 `def name(...):` 或 `class Name(...):`"
            ),
            (
                "  - return_line: string  # e.g. `return energy`"
                if use_en
                else "  - return_line: string  # 例如 `return energy`"
            ),
            f"  - step_background: string | null",
        ]
    )
    return "\n".join(
        [
            f"- {FIELD_STEP_NUMBER}: string  # {step_desc}",
            f"- {FIELD_SUB_STEP}: ExecutableSubStep",
            f"- {FIELD_GOLDEN_STEP_CODE}: string  # {code_desc}",
            f"- {FIELD_DEPENDENCIES}: string",
            f"- {FIELD_REQUIRED_FACT_IDS}: string[]  # {required_desc}",
            f"- {FIELD_PRIMARY_REQUIRED_FACT_ID}: string",
            f"- {FIELD_REUSE_PLAN}: string[]  # {reuse_desc}",
            "",
            substep_schema,
        ]
    )


def executable_test_inputs_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    inputs_desc = (
        "array of JSON-serializable cases; each case has args (list) and optional kwargs (object)"
        if use_en
        else "JSON 可序列化输入数组；每项包含 args(list) 与可选 kwargs(object)"
    )
    return "\n".join(
        [
            f"- {FIELD_TEST_INPUTS}: object[]  # {inputs_desc}",
        ]
    )


def executable_test_derive_output_schema_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    rel_desc = "relative path to inputs.jsonl" if use_en else "inputs.jsonl 的相对路径"
    sha_desc = "sha256 of inputs.jsonl" if use_en else "inputs.jsonl 的 sha256"
    return "\n".join(
        [
            f"- {FIELD_INPUTS_ARTIFACT_RELPATH}: string  # {rel_desc}",
            f"- {FIELD_INPUTS_SHA256}: string  # {sha_desc}",
            f"- {FIELD_TEST_CASES_FOR_STEP}: ExecutableTestCase[]",
        ]
    )


@dataclass
class ExecutableRecord:
    """Executable problem record (analogous to KQARecord).

    Notes:
    - `test_cases` is keyed by `step_number` (string), not by integer index.
    - `per_step_golden` may store ground-truth code (SciCode import) or a
      synthesized golden implementation (A: golden defines truth).
    """

    # Meta
    paper_id: Optional[str] = None
    problem_id: str = ""
    qa_idx: Optional[int] = None

    # Background / dependencies
    background: str = ""
    required_dependencies: str = ""

    # Multi-step spec
    sub_steps: List[ExecutableSubStep] = field(default_factory=list)

    # Tests (step_number -> tests)
    test_cases: Dict[str, List[ExecutableTestCase]] = field(default_factory=dict)
    # Inputs artifacts (step_number -> {inputs_artifact_relpath, inputs_sha256})
    inputs_artifacts: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Golden / reference implementations
    golden_code: Optional[str] = None
    per_step_golden: Dict[str, str] = field(default_factory=dict)

    # Metadata
    estimated_difficulty: Optional[str] = None
    subject: Optional[str] = None
    source: Optional[str] = None  # e.g., "scicode", "synthetic"

    # Path-Fold artifacts (optional): folded path spec for the tail step.
    # Keep the same field names as semantic KQARecord for a unified Director view.
    path_question_scaffolded: Optional[str] = None
    path_question_direct: Optional[str] = None
    path_fold_notes: Optional[str] = None

    @property
    def record_type(self) -> str:
        return "executable"

    @property
    def step(self) -> Optional[int]:
        return self.qa_idx

    @step.setter
    def step(self, value: Optional[int]) -> None:
        if value is None:
            self.qa_idx = None
            return
        try:
            self.qa_idx = int(value)
        except Exception:
            self.qa_idx = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "record_type": self.record_type,
            "paper_id": self.paper_id,
            "problem_id": self.problem_id,
            "qa_idx": self.qa_idx,
            "step": self.qa_idx,
            "background": self.background,
            "required_dependencies": self.required_dependencies,
            "sub_steps": [s.to_dict() for s in self.sub_steps],
            "test_cases": {
                step: [t.to_dict() for t in tests]
                for step, tests in (self.test_cases or {}).items()
                if isinstance(step, str)
            },
            "inputs_artifacts": dict(self.inputs_artifacts or {}),
            "golden_code": self.golden_code,
            "per_step_golden": dict(self.per_step_golden or {}),
            "estimated_difficulty": self.estimated_difficulty,
            "subject": self.subject,
            "source": self.source,
        }
        if isinstance(self.path_question_scaffolded, str) and self.path_question_scaffolded.strip():
            payload["path_question_scaffolded"] = self.path_question_scaffolded
        if isinstance(self.path_question_direct, str) and self.path_question_direct.strip():
            payload["path_question_direct"] = self.path_question_direct
        if isinstance(self.path_fold_notes, str) and self.path_fold_notes.strip():
            payload["path_fold_notes"] = self.path_fold_notes
        return payload

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExecutableRecord":
        sub_steps_raw = data.get("sub_steps") or []
        sub_steps: List[ExecutableSubStep] = []
        if isinstance(sub_steps_raw, list):
            for item in sub_steps_raw:
                if isinstance(item, dict):
                    sub_steps.append(ExecutableSubStep.from_dict(item))

        test_cases: Dict[str, List[ExecutableTestCase]] = {}
        test_cases_raw = data.get("test_cases") or {}
        if isinstance(test_cases_raw, dict):
            for step_num, tests_raw in test_cases_raw.items():
                if not isinstance(step_num, str):
                    continue
                tests: List[ExecutableTestCase] = []
                if isinstance(tests_raw, list):
                    for t in tests_raw:
                        if isinstance(t, dict):
                            tests.append(ExecutableTestCase.from_dict(t))
                test_cases[step_num] = tests

        inputs_artifacts: Dict[str, Dict[str, str]] = {}
        raw_inputs = data.get("inputs_artifacts") or {}
        if isinstance(raw_inputs, dict):
            for step_num, info in raw_inputs.items():
                if not isinstance(step_num, str) or not isinstance(info, dict):
                    continue
                relpath = info.get(FIELD_INPUTS_ARTIFACT_RELPATH)
                sha256 = info.get(FIELD_INPUTS_SHA256)
                if isinstance(relpath, str) and isinstance(sha256, str):
                    inputs_artifacts[step_num] = {
                        FIELD_INPUTS_ARTIFACT_RELPATH: relpath,
                        FIELD_INPUTS_SHA256: sha256,
                    }

        per_step_golden: Dict[str, str] = {}
        raw_psg = data.get("per_step_golden") or {}
        if isinstance(raw_psg, dict):
            for k, v in raw_psg.items():
                if isinstance(k, str) and isinstance(v, str):
                    per_step_golden[k] = v

        qa_idx_val = data.get("qa_idx")
        if qa_idx_val is None:
            qa_idx_val = data.get("step")  # backward compat
        try:
            qa_idx_norm = int(qa_idx_val) if qa_idx_val is not None else None
        except Exception:
            qa_idx_norm = None

        return ExecutableRecord(
            paper_id=(data.get("paper_id") if isinstance(data.get("paper_id"), str) else None),
            problem_id=str(data.get("problem_id") or ""),
            qa_idx=qa_idx_norm,
            background=str(data.get("background") or ""),
            required_dependencies=str(data.get("required_dependencies") or ""),
            sub_steps=sub_steps,
            test_cases=test_cases,
            inputs_artifacts=inputs_artifacts,
            golden_code=(data.get("golden_code") if isinstance(data.get("golden_code"), str) else None),
            per_step_golden=per_step_golden,
            estimated_difficulty=(data.get("estimated_difficulty") if isinstance(data.get("estimated_difficulty"), str) else None),
            subject=(data.get("subject") if isinstance(data.get("subject"), str) else None),
            source=(data.get("source") if isinstance(data.get("source"), str) else None),
            path_question_scaffolded=(
                data.get("path_question_scaffolded") if isinstance(data.get("path_question_scaffolded"), str) else None
            ),
            path_question_direct=(
                data.get("path_question_direct") if isinstance(data.get("path_question_direct"), str) else None
            ),
            path_fold_notes=(data.get("path_fold_notes") if isinstance(data.get("path_fold_notes"), str) else None),
        )


__all__ = [
    "ExecutableRecord",
    "ExecutableSubStep",
    "ExecutableTestCase",
    "ExecutableExtractOutput",
    "ExecutableDraftOutput",
    "ExecutableTestDeriveOutput",
    "FIELD_EXECUTABLE_SUITABLE",
    "FIELD_NOTES",
    "FIELD_TASK_SKETCH",
    "FIELD_INITIAL_SUB_STEPS",
    "FIELD_ESTIMATED_DIFFICULTY",
    "FIELD_STEP_NUMBER",
    "FIELD_SUB_STEP",
    "FIELD_GOLDEN_STEP_CODE",
    "FIELD_DEPENDENCIES",
    "FIELD_REQUIRED_FACT_IDS",
    "FIELD_PRIMARY_REQUIRED_FACT_ID",
    "FIELD_REUSE_PLAN",
    "FIELD_TEST_INPUTS",
    "FIELD_INPUTS_ARTIFACT_RELPATH",
    "FIELD_INPUTS_SHA256",
    "FIELD_TEST_CASES_FOR_STEP",
    "executable_extract_output_schema_text",
    "executable_draft_step_output_schema_text",
    "executable_test_inputs_schema_text",
    "executable_test_derive_output_schema_text",
]
