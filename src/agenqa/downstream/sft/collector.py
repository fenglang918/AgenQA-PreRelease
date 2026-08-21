"""Collect step-level SFT export candidates from AgenQA run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from agenqa.graph.state import AgentState


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(data).__name__}")
    return data


def _round_from_path(step_dir: Path) -> int:
    for parent in step_dir.parents:
        name = parent.name
        if name.startswith("round_"):
            try:
                return int(name.split("_", 1)[1])
            except Exception:
                return 0
    return 0


def _candidate_step_dirs(run_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    for edge_kqa_path in sorted(run_dir.rglob("edge_kqa.json")):
        step_dir = edge_kqa_path.parent.resolve()
        rel_parts = step_dir.relative_to(run_dir).parts
        if "subruns_raw" in rel_parts or "solve" in rel_parts or "_resume_archive" in rel_parts:
            continue
        seen.add(step_dir)
    return sorted(seen, key=lambda path: str(path.relative_to(run_dir)))


def _resolve_subrun_paths(step_dir: Path) -> tuple[Path | None, Path | None]:
    subruns_dir = step_dir / "subruns"
    candidates = [
        (subruns_dir / "01_draft_chain.json", subruns_dir / "02_format.json"),
        (subruns_dir / "02_draft_chain.json", subruns_dir / "03_format.json"),
    ]
    for draft_chain_path, format_path in candidates:
        if draft_chain_path.is_file() and format_path.is_file():
            return draft_chain_path, format_path
    return None, None


@dataclass(frozen=True)
class StepArtifacts:
    run_dir: Path
    run_id: str
    state_path: Path
    step_dir: Path
    round_idx: int
    step: int
    edge_kqa_path: Path
    path_kqa_path: Optional[Path]
    draft_chain_path: Path
    format_path: Path
    answer_contract_report_path: Optional[Path]


@dataclass
class StepSnapshot:
    artifacts: StepArtifacts
    state: AgentState
    edge_kqa: dict[str, Any]
    path_kqa: Optional[dict[str, Any]]
    draft_chain: dict[str, Any]
    format_output: dict[str, Any]
    answer_contract_report: Optional[dict[str, Any]]


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "state.json").is_file()


def discover_run_dirs(inputs: Iterable[Path], *, recursive: bool = False) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if is_run_dir(path):
            if path not in seen:
                run_dirs.append(path)
                seen.add(path)
            continue
        if not path.is_dir():
            continue
        if recursive:
            for state_path in sorted(path.rglob("state.json")):
                candidate = state_path.parent
                if is_run_dir(candidate) and candidate not in seen:
                    run_dirs.append(candidate)
                    seen.add(candidate)
    return run_dirs


def discover_step_artifacts(run_dir: Path) -> list[StepArtifacts]:
    run_dir = run_dir.expanduser().resolve()
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"state.json not found under run_dir={run_dir}")

    state = AgentState.load_from_file(state_path)
    run_id = str(getattr(state, "run_id", "") or run_dir.name)
    best_by_step: dict[int, StepArtifacts] = {}

    for step_dir in _candidate_step_dirs(run_dir):
        draft_chain_path, format_path = _resolve_subrun_paths(step_dir)
        edge_kqa_path = step_dir / "edge_kqa.json"
        path_kqa_path = step_dir / "path_kqa.json"
        answer_contract_report_path = step_dir / "answer_contract_report.json"

        if draft_chain_path is None or format_path is None or not edge_kqa_path.is_file():
            continue

        format_output = _load_json(format_path)
        edge_kqa = _load_json(edge_kqa_path)
        step = int(
            format_output.get("Step")
            or edge_kqa.get("step")
            or edge_kqa.get("qa_idx")
            or 0
        )
        round_idx = _round_from_path(step_dir)
        candidate = StepArtifacts(
            run_dir=run_dir,
            run_id=run_id,
            state_path=state_path,
            step_dir=step_dir,
            round_idx=round_idx,
            step=step,
            edge_kqa_path=edge_kqa_path,
            path_kqa_path=path_kqa_path if path_kqa_path.is_file() else None,
            draft_chain_path=draft_chain_path,
            format_path=format_path,
            answer_contract_report_path=(
                answer_contract_report_path if answer_contract_report_path.is_file() else None
            ),
        )
        prev = best_by_step.get(step)
        if prev is None:
            best_by_step[step] = candidate
            continue
        prev_key = (int(prev.round_idx), str(prev.step_dir))
        cur_key = (int(candidate.round_idx), str(candidate.step_dir))
        if cur_key >= prev_key:
            best_by_step[step] = candidate

    return [best_by_step[step] for step in sorted(best_by_step)]


def load_step_snapshot(artifacts: StepArtifacts, state: AgentState | None = None) -> StepSnapshot:
    loaded_state = state or AgentState.load_from_file(artifacts.state_path)
    path_kqa = _load_json(artifacts.path_kqa_path) if artifacts.path_kqa_path else None
    answer_contract_report = (
        _load_json(artifacts.answer_contract_report_path)
        if artifacts.answer_contract_report_path
        else None
    )
    return StepSnapshot(
        artifacts=artifacts,
        state=loaded_state,
        edge_kqa=_load_json(artifacts.edge_kqa_path),
        path_kqa=path_kqa,
        draft_chain=_load_json(artifacts.draft_chain_path),
        format_output=_load_json(artifacts.format_path),
        answer_contract_report=answer_contract_report,
    )


def iter_step_snapshots(run_dir: Path) -> Iterator[StepSnapshot]:
    state_path = run_dir / "state.json"
    state = AgentState.load_from_file(state_path)
    for artifacts in discover_step_artifacts(run_dir):
        yield load_step_snapshot(artifacts, state=state)
