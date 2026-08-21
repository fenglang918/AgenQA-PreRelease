"""Dev views: RunIndex / StepView / EpisodeView (read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json


def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl_first(path: Path) -> Dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not text:
        return None
    line = text.splitlines()[0]
    try:
        return json.loads(line)
    except Exception:
        return None


def _guess_step_from_dir(step_dir: Path) -> int | None:
    for name in ("edge_kqa.json", "edge_kqa.jsonl", "director_decision.json"):
        path = step_dir / name
        if not path.exists():
            continue
        payload = _read_jsonl_first(path) if name.endswith("jsonl") else _read_json(path)
        if not isinstance(payload, dict):
            continue
        val = payload.get("qa_idx") if payload.get("qa_idx") is not None else payload.get("step")
        try:
            if val is not None:
                return int(val)
        except Exception:
            continue
    return None


@dataclass
class StepEntry:
    qa_idx: int
    round_idx: int
    node: str
    step_dir: Path

    def role_outputs(self) -> Dict[str, Any]:
        subruns_dir = self.step_dir / "subruns"
        outputs: Dict[str, Any] = {}
        if not subruns_dir.exists():
            return outputs
        for item in sorted(subruns_dir.glob("*.json")):
            payload = _read_json(item)
            name = item.stem
            parts = name.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                name = parts[1]
            outputs[name] = payload if payload is not None else item.read_text(encoding="utf-8", errors="ignore")
        return outputs

    def artifacts(self) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        raw_dir = self.step_dir / "subruns_raw"
        if raw_dir.exists():
            paths["subruns_raw"] = raw_dir
        for name in (
            "edge_kqa.json",
            "edge_kqa.jsonl",
            "path_kqa.json",
            "path_kqa.jsonl",
            "director_decision.json",
        ):
            path = self.step_dir / name
            if path.exists():
                paths[name] = path
        solve_dir = self.step_dir / "solve"
        if solve_dir.exists():
            for item in sorted(solve_dir.glob("*.jsonl")):
                paths[f"solve/{item.name}"] = item
        return paths


@dataclass
class RunIndex:
    run_dir: Path
    entries: List[StepEntry]

    @classmethod
    def build(cls, run_dir: Path | str) -> "RunIndex":
        root = Path(run_dir)
        entries: List[StepEntry] = []
        for round_dir in sorted(root.glob("round_*")):
            if not round_dir.is_dir():
                continue
            try:
                round_idx = int(round_dir.name.replace("round_", ""))
            except Exception:
                continue
            for step_dir in sorted(round_dir.iterdir()):
                if not step_dir.is_dir():
                    continue
                if step_dir.name in {"00_Prompts_Snapshot", "00_Summary", "01_qa_init"}:
                    continue
                name = step_dir.name
                if name.startswith("step_"):
                    parts = name.split("_", 2)
                    if len(parts) >= 3 and parts[1].isdigit():
                        qa_idx = int(parts[1])
                        node = parts[2]
                        entries.append(StepEntry(qa_idx=qa_idx, round_idx=round_idx, node=node, step_dir=step_dir))
                        continue
                qa_idx = _guess_step_from_dir(step_dir)
                if qa_idx is None:
                    continue
                entries.append(StepEntry(qa_idx=qa_idx, round_idx=round_idx, node=name, step_dir=step_dir))
        return cls(run_dir=root, entries=entries)

    def steps(self) -> Dict[int, List[StepEntry]]:
        out: Dict[int, List[StepEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.qa_idx, []).append(entry)
        for entry_list in out.values():
            entry_list.sort(key=lambda e: (e.round_idx, e.node))
        return out


@dataclass
class StepView:
    qa_idx: int
    entries: List[StepEntry]

    @classmethod
    def from_index(cls, index: RunIndex, qa_idx: int) -> "StepView":
        entries = [e for e in index.entries if e.qa_idx == qa_idx]
        entries.sort(key=lambda e: (e.round_idx, e.node))
        return cls(qa_idx=qa_idx, entries=entries)


@dataclass
class EpisodeView:
    run_dir: Path
    state: Dict[str, Any]
    steps: Dict[int, StepView]

    @classmethod
    def build(cls, run_dir: Path | str) -> "EpisodeView":
        root = Path(run_dir)
        state_path = root / "state.json"
        state_payload = _read_json(state_path) if state_path.exists() else {}
        index = RunIndex.build(root)
        steps = {qa_idx: StepView.from_index(index, qa_idx) for qa_idx in index.steps().keys()}
        return cls(run_dir=root, state=state_payload or {}, steps=steps)


__all__ = ["RunIndex", "StepView", "EpisodeView", "StepEntry"]
