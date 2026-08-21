"""End-to-end export pipeline for AgenQA downstream SFT data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from agenqa.domain.known_utils import format_known_for_solver

from .collector import StepSnapshot, discover_run_dirs, iter_step_snapshots
from .filters import Phase1FilterConfig, decide_edge, decide_path_direct
from .schema import CanonicalSFTSample, SFTSourcePaths
from .split import assign_episode_split
from .writers import write_canonical_jsonl, write_json, write_tuningfactory_json


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _preferred_known_payload(kqa: dict[str, Any]) -> Any:
    return kqa.get("known") or kqa.get("known_tree") or ""


def _preferred_question_text(kqa: dict[str, Any]) -> str:
    return _safe_text(kqa.get("question_for_solver") or kqa.get("question"))


def _kqa_metadata(kqa: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_text_source": (
            "question_for_solver"
            if _safe_text(kqa.get("question_for_solver"))
            else "question"
        ),
        "world_contract_text": _safe_text(kqa.get("world_contract_text")),
        "world_contract_present": bool(_safe_text(kqa.get("world_contract_text"))),
        "known_tree_present": bool(kqa.get("known_tree")),
    }


def _question_type_from_state(snapshot: StepSnapshot) -> str:
    for record in snapshot.state.iter_semantic_records():
        try:
            if int(getattr(record, "step", getattr(record, "qa_idx", 0)) or 0) == int(snapshot.artifacts.step):
                return _safe_text(getattr(record, "question_type", None))
        except Exception:
            continue
    return ""


def _build_source_paths(
    snapshot: StepSnapshot,
    *,
    kqa_path: Path,
    director_decision_path: Optional[Path],
) -> SFTSourcePaths:
    return SFTSourcePaths(
        run_dir=str(snapshot.artifacts.run_dir),
        kqa_path=str(kqa_path),
        draft_chain_path=str(snapshot.artifacts.draft_chain_path),
        format_path=str(snapshot.artifacts.format_path),
        answer_contract_report_path=(
            str(snapshot.artifacts.answer_contract_report_path)
            if snapshot.artifacts.answer_contract_report_path
            else ""
        ),
        director_decision_path=str(director_decision_path) if director_decision_path is not None else "",
    )


def _find_quality_decision_path(snapshot: StepSnapshot) -> Optional[Path]:
    preferred = snapshot.artifacts.step_dir.parent.parent / f"round_{snapshot.artifacts.round_idx + 1}" / "director" / "director_decision.json"
    if preferred.is_file():
        return preferred
    fallback = snapshot.artifacts.step_dir / "director_decision.json"
    if fallback.is_file():
        return fallback
    return None


def _common_metadata(snapshot: StepSnapshot, filter_metadata: dict[str, Any]) -> dict[str, Any]:
    report = snapshot.answer_contract_report or {}
    return {
        "run_id": snapshot.artifacts.run_id,
        "run_dir": str(snapshot.artifacts.run_dir),
        "round_idx": snapshot.artifacts.round_idx,
        "step_dir": str(snapshot.artifacts.step_dir),
        "answer_contract_error_count": int(report.get("error_count", 0) or 0),
        "answer_contract_warn_count": int(report.get("warn_count", 0) or 0),
        "filter": filter_metadata,
    }


def build_edge_sample(snapshot: StepSnapshot, *, split: str, filter_metadata: dict[str, Any]) -> CanonicalSFTSample:
    edge = snapshot.edge_kqa
    fmt = snapshot.format_output
    question_type = _safe_text(edge.get("question_type") or fmt.get("question_type") or _question_type_from_state(snapshot))
    director_decision_path = _find_quality_decision_path(snapshot)
    source_paths = _build_source_paths(
        snapshot,
        kqa_path=snapshot.artifacts.edge_kqa_path,
        director_decision_path=director_decision_path,
    )
    return CanonicalSFTSample(
        sample_id=f"{snapshot.artifacts.run_id}::edge::step_{snapshot.artifacts.step}",
        episode_id=snapshot.artifacts.run_id,
        split=split,
        sample_type="edge",
        step=int(snapshot.artifacts.step),
        qa_idx=int(edge.get("qa_idx") or edge.get("step") or snapshot.artifacts.step),
        question_type=question_type,
        known_text=format_known_for_solver(_preferred_known_payload(edge)),
        question_text=_preferred_question_text(edge),
        plan_text=_safe_text(snapshot.draft_chain.get("draft_solution_outline")),
        solution_text=_safe_text(fmt.get("Solution")),
        final_answer=_safe_text(fmt.get("Answer")),
        validation_passed=bool(fmt.get("validation_passed") is True),
        metadata={**_common_metadata(snapshot, filter_metadata), **_kqa_metadata(edge)},
        source_paths=source_paths,
    )


def build_path_direct_sample(snapshot: StepSnapshot, *, split: str, filter_metadata: dict[str, Any]) -> CanonicalSFTSample:
    path = snapshot.path_kqa or {}
    fmt = snapshot.format_output
    question_type = _safe_text(
        path.get("question_type")
        or fmt.get("question_type")
        or snapshot.edge_kqa.get("question_type")
        or _question_type_from_state(snapshot)
    )
    director_decision_path = _find_quality_decision_path(snapshot)
    source_paths = _build_source_paths(
        snapshot,
        kqa_path=snapshot.artifacts.path_kqa_path or snapshot.artifacts.edge_kqa_path,
        director_decision_path=director_decision_path,
    )
    return CanonicalSFTSample(
        sample_id=f"{snapshot.artifacts.run_id}::path_direct::step_{snapshot.artifacts.step}",
        episode_id=snapshot.artifacts.run_id,
        split=split,
        sample_type="path_direct",
        step=int(snapshot.artifacts.step),
        qa_idx=int(path.get("qa_idx_tail") or path.get("step_tail") or snapshot.artifacts.step),
        question_type=question_type,
        known_text=format_known_for_solver(_preferred_known_payload(path)),
        question_text=_preferred_question_text(path),
        plan_text=_safe_text(snapshot.draft_chain.get("draft_solution_outline")),
        solution_text=_safe_text(fmt.get("Solution")),
        final_answer=_safe_text(fmt.get("Answer")),
        validation_passed=bool(fmt.get("validation_passed") is True),
        metadata={**_common_metadata(snapshot, filter_metadata), **_kqa_metadata(path)},
        source_paths=source_paths,
    )


@dataclass(frozen=True)
class ExportReport:
    run_count: int
    sample_count: int
    edge_count: int
    path_count: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_count": self.run_count,
            "sample_count": self.sample_count,
            "edge_count": self.edge_count,
            "path_count": self.path_count,
            "output_dir": self.output_dir,
        }


def export_sft_dataset(
    input_paths: Iterable[Path],
    *,
    output_dir: Path,
    recursive: bool = False,
    include_edge: bool = True,
    include_path_direct: bool = True,
    eval_ratio: float = 0.0,
    test_ratio: float = 0.0,
    split_seed: int = 17,
    filter_config: Phase1FilterConfig | None = None,
) -> ExportReport:
    cfg = filter_config or Phase1FilterConfig()
    run_dirs = discover_run_dirs(input_paths, recursive=recursive)

    accepted_edge: list[CanonicalSFTSample] = []
    accepted_path: list[CanonicalSFTSample] = []
    rejected: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        for snapshot in iter_step_snapshots(run_dir):
            split = assign_episode_split(
                snapshot.artifacts.run_id,
                eval_ratio=eval_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
            )
            if include_edge:
                edge_decision = decide_edge(snapshot, cfg)
                if edge_decision.accepted:
                    accepted_edge.append(build_edge_sample(snapshot, split=split, filter_metadata=edge_decision.metadata))
                else:
                    rejected.append(
                        {
                            "sample_id": f"{snapshot.artifacts.run_id}::edge::step_{snapshot.artifacts.step}",
                            "sample_type": "edge",
                            "reasons": edge_decision.reasons,
                            "step_dir": str(snapshot.artifacts.step_dir),
                        }
                    )
            if include_path_direct:
                path_decision = decide_path_direct(snapshot, cfg)
                if path_decision.accepted:
                    accepted_path.append(build_path_direct_sample(snapshot, split=split, filter_metadata=path_decision.metadata))
                else:
                    rejected.append(
                        {
                            "sample_id": f"{snapshot.artifacts.run_id}::path_direct::step_{snapshot.artifacts.step}",
                            "sample_type": "path_direct",
                            "reasons": path_decision.reasons,
                            "step_dir": str(snapshot.artifacts.step_dir),
                        }
                    )

    all_samples = sorted(accepted_edge + accepted_path, key=lambda s: (s.episode_id, s.step, s.sample_type))
    by_split: dict[str, list[CanonicalSFTSample]] = {"train": [], "eval": [], "test": []}
    for sample in all_samples:
        by_split.setdefault(sample.split, []).append(sample)

    output_dir = output_dir.expanduser().resolve()
    canonical_dir = output_dir / "canonical"
    tuningfactory_dir = output_dir / "tuningfactory"
    manifests_dir = output_dir / "manifests"

    write_canonical_jsonl(canonical_dir / "all.jsonl", all_samples)
    write_tuningfactory_json(tuningfactory_dir / "all.json", all_samples)
    for split_name, split_samples in by_split.items():
        if not split_samples:
            continue
        write_canonical_jsonl(canonical_dir / f"{split_name}.jsonl", split_samples)
        write_tuningfactory_json(tuningfactory_dir / f"{split_name}.json", split_samples)

    write_json(
        manifests_dir / "summary.json",
        {
            "runs": [str(p) for p in run_dirs],
            "run_count": len(run_dirs),
            "sample_count": len(all_samples),
            "edge_count": len(accepted_edge),
            "path_count": len(accepted_path),
            "split_counts": {name: len(samples) for name, samples in by_split.items() if samples},
            "rejected_count": len(rejected),
        },
    )
    write_json(manifests_dir / "rejected.json", rejected)

    return ExportReport(
        run_count=len(run_dirs),
        sample_count=len(all_samples),
        edge_count=len(accepted_edge),
        path_count=len(accepted_path),
        output_dir=str(output_dir),
    )
