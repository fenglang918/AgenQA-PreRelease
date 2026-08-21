"""Canonical schema for AgenQA downstream SFT export."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Any, Dict, Literal


SampleType = Literal["edge", "path_direct"]


@dataclass(frozen=True)
class SFTSourcePaths:
    """Source files used to build one exported sample."""

    run_dir: str
    kqa_path: str
    draft_chain_path: str
    format_path: str
    answer_contract_report_path: str
    director_decision_path: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class CanonicalSFTSample:
    """Framework-agnostic downstream SFT sample."""

    sample_id: str
    episode_id: str
    split: str
    sample_type: SampleType
    step: int
    qa_idx: int
    question_type: str
    known_text: str
    question_text: str
    plan_text: str
    solution_text: str
    final_answer: str
    validation_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_paths: SFTSourcePaths | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.source_paths is not None:
            payload["source_paths"] = self.source_paths.to_dict()
        return payload

    def to_tuningfactory_alpaca(self) -> Dict[str, str]:
        """Render the sample to TuningFactory's Alpaca-style format."""

        instruction = "请基于给定的 Known 与 Question 作答。先输出 [Plan]，再输出 [Solution]，最后输出 [Final Answer]。"
        parts = []
        if self.known_text.strip():
            parts.append(f"Known:\n{self.known_text}")
        parts.append(f"Question:\n{self.question_text}")
        input_text = "\n\n".join(parts)
        output_text = (
            f"[Plan]\n{self.plan_text}\n\n"
            f"[Solution]\n{self.solution_text}\n\n"
            f"[Final Answer]\n{self.final_answer}"
        )
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        }
