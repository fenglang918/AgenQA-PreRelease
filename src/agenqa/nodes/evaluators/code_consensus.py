"""Executable consensus node: aggregate multi-solver signals for executable track.

Unlike semantic consensus (which aggregates answer agreement), executable consensus
aggregates:
- correctness votes (pass/fail on tests)
- failure taxonomy (why it failed)

The output is written to disk and also summarized into KnownTree memory so the
Director can make long-term decisions using stable signals.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from infra.data.io import read_jsonl
from agenqa.domain.executable_error_taxonomy import ExecutableErrorClassification, classify_row
from agenqa.domain.known_tree import KnownTree
from agenqa.graph.output_manager import compute_step_dir
from agenqa.graph.state import AgentState
from agenqa.memory.store import save_state

logger = logging.getLogger(__name__)


_IDX_RE = re.compile(r"^(?P<prefix>solve(?:_path)?)_(?P<tier>[a-z]+)_(?P<idx>\d+)\.jsonl$")
_EVAL_CERT_KIND = "executable_eval_cert"


def _infer_op_name_for_code_solve(state: AgentState) -> str:
    op_raw = (state.last_decision.operation if state.last_decision else "extend")  # type: ignore[union-attr]
    op_lc = str(op_raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "revise" in op_lc:
        return "revise"
    if "extend" in op_lc or "init" in op_lc:
        return "extend"
    # Fallback to extend (executable track default).
    return "extend"


def _first_row(path: Path) -> Dict[str, Any] | None:
    try:
        for row in read_jsonl(path, schema=None, max_lines=1):
            return row if isinstance(row, dict) else None
    except Exception:
        return None
    return None


def _iter_solver_files(solve_dir: Path, *, target: str, tier: str) -> list[tuple[int, Path]]:
    # target: "edge"|"path"
    # tier: "strong"|"medium"
    prefix = "solve" if target == "edge" else "solve_path"
    files: dict[int, Path] = {}
    if not solve_dir.exists():
        return []

    # Prefer indexed.
    for path in solve_dir.glob(f"{prefix}_{tier}_*.jsonl"):
        m = re.fullmatch(rf"{re.escape(prefix)}_{re.escape(tier)}_(\d+)\.jsonl", path.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        files[idx] = path

    # Backward compatible primary (no suffix).
    legacy = solve_dir / f"{prefix}_{tier}.jsonl"
    if 0 not in files and legacy.exists():
        files[0] = legacy
    return sorted(files.items(), key=lambda x: x[0])


@dataclass
class VoteSummary:
    total: int
    correct: int
    error_counts_by_type: Dict[str, int]
    dominant_error_type: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "error_counts_by_type": dict(self.error_counts_by_type),
            "dominant_error_type": self.dominant_error_type,
        }


def _summarize_votes(rows: list[Dict[str, Any] | None]) -> VoteSummary:
    total = len(rows)
    correct = 0
    counts: Dict[str, int] = {}
    for row in rows:
        cls = classify_row(row)
        if cls.type == "success":
            correct += 1
        counts[cls.type] = counts.get(cls.type, 0) + 1
    dominant = None
    if counts:
        dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    return VoteSummary(total=total, correct=correct, error_counts_by_type=counts, dominant_error_type=dominant)


def _upsert_executable_eval_cert(mem: Dict[str, Any], *, step: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    mem = KnownTree.normalize_memory(mem)
    certs = mem.get("step_certs")
    if not isinstance(certs, list):
        certs = []
    out: list[Dict[str, Any]] = []
    replaced = False
    for item in certs:
        if not isinstance(item, dict):
            continue
        # Only replace eval cert entries for the same step; do not touch chain certs.
        if item.get("kind") == _EVAL_CERT_KIND and item.get("step") == step:
            if not replaced:
                out.append(payload)
                replaced = True
            continue
        out.append(item)
    if not replaced:
        out.append(payload)
    mem["step_certs"] = out
    return mem


def compute_code_consensus(agent_conf: Dict[str, Any], state: AgentState) -> AgentState:
    """Aggregate executable multi-solver signals and write consensus summary + memory cert."""
    try:
        step_idx = int(state.step or 0)
    except Exception:
        step_idx = 0
    try:
        round_idx = int(getattr(state, "rounds", 1) or 1)
    except Exception:
        round_idx = 1

    op_name = _infer_op_name_for_code_solve(state)
    step_dir = compute_step_dir(Path(state.artifacts_dir), op_name, step_idx, round_idx)
    solve_dir = step_dir / "solve"
    solve_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "step": step_idx,
        "round": round_idx,
        "op_name": op_name,
        "golden": None,
        "edge": {},
        "path": {},
    }

    try:
        golden_path = solve_dir / "solve_golden.jsonl"
        golden_row = _first_row(golden_path) if golden_path.exists() else None
        golden_cls = classify_row(golden_row)
        summary["golden"] = {
            "correct": (golden_row.get("correct") if isinstance(golden_row, dict) else None),
            "error_type": golden_cls.type,
            "recoverable": golden_cls.recoverable,
            "error": (golden_row.get("error") if isinstance(golden_row, dict) else None),
        }
    except Exception:
        summary["golden"] = None

    for target in ("edge", "path"):
        target_sum: Dict[str, Any] = {}
        for tier in ("strong", "medium"):
            files = _iter_solver_files(solve_dir, target=target, tier=tier)
            rows = [_first_row(p) for _, p in files]
            target_sum[tier] = _summarize_votes(rows).to_dict()
            # Include a compact per-solver list for debugging/diagnosis (no code content).
            votes: list[Dict[str, Any]] = []
            for idx, path in files:
                row = _first_row(path)
                cls: ExecutableErrorClassification = classify_row(row)
                err = (row.get("error") if isinstance(row, dict) else None)
                if isinstance(err, str) and len(err) > 400:
                    err = err[:400] + "…"
                votes.append(
                    {
                        "solver_idx": idx,
                        "tier": tier if idx == 0 else f"{tier}_{idx}",
                        "correct": (row.get("correct") if isinstance(row, dict) else None),
                        "error_type": cls.type,
                        "recoverable": cls.recoverable,
                        "error": err,
                        "model": (row.get("model") if isinstance(row, dict) else None),
                        "service_id": (row.get("service_id") if isinstance(row, dict) else None),
                    }
                )
            target_sum[f"{tier}_votes"] = votes
        summary[target] = target_sum

    # Persist to disk for inspection.
    try:
        (solve_dir / "executable_consensus_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Write a compact error classification for quick routing/triage.
    try:
        golden = summary.get("golden") if isinstance(summary.get("golden"), dict) else None
        golden_type = golden.get("error_type") if isinstance(golden, dict) else None
        golden_ok = golden.get("correct") is True if isinstance(golden, dict) else None
        strong_edge = ((summary.get("edge") or {}).get("strong") if isinstance(summary.get("edge"), dict) else None) or {}
        dominant = strong_edge.get("dominant_error_type") if isinstance(strong_edge, dict) else None
        recoverable = True
        if golden_ok is True and dominant == "success":
            recoverable = False
        payload = {
            "step": step_idx,
            "round": round_idx,
            "golden_ok": golden_ok,
            "golden_error_type": golden_type,
            "dominant_error_type(edge:strong)": dominant,
            "recoverable": recoverable,
        }
        (solve_dir / "error_classification.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Write a executable step cert into KnownTree memory (no golden code).
    try:
        mem = KnownTree.normalize_memory(getattr(state, "memory", None))
        cert_payload = {
            "kind": _EVAL_CERT_KIND,
            "step": step_idx,
            "round": round_idx,
            "op_name": op_name,
            "consensus": summary,
            "provenance": {
                "artifacts_dir": str(Path(state.artifacts_dir)),
                "step_dir": str(step_dir),
                "solve_dir": str(solve_dir),
            },
        }
        mem = _upsert_executable_eval_cert(mem, step=step_idx, payload=cert_payload)
        state.memory = mem
    except Exception:
        pass

    save_state(state)
    return state


__all__ = [
    "compute_code_consensus",
]
