"""Consensus node: aggregate signals from multiple strong solvers.

This node reads per-strong solver artifacts written by `solve_dual`:
  - edge: solve/solve_strong_0.jsonl, solve/solve_strong_1.jsonl, solve/solve_strong_2.jsonl, ...
  - path: solve/solve_path_strong_0.jsonl, solve/solve_path_strong_1.jsonl, solve/solve_path_strong_2.jsonl, ...

It then computes simple majority-vote consensus for:
  - answer agreement (normalized / LLM-clustered)
  - question_well_posed agreement

The results are written to:
  - solve/consensus_summary.json       (legacy edge alias; now includes `view: "edge"`)
  - solve/consensus_summary_edge.json
  - solve/consensus_summary_path.json
and the edge result is stored into:
  - state.solver_consensus.strong.*
"""

from __future__ import annotations

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from infra.data.io import read_jsonl
from agenqa.domain.contracts.answer_contract_bank import extract_answer_contract_context
from agenqa.domain.contracts.solver_contract_text import compose_solver_question
from agenqa.domain.known_tree import KnownTree
from agenqa.graph.output_manager import compute_step_dir
from agenqa.graph.state import AgentState, SolverConsensus, SolverVote, StrongSolverConsensus
from agenqa.memory.store import save_state
from agenqa.nodes.evaluators.expression_judge import load_expression_judge_generator, run_expression_equivalence_judge
from agenqa.nodes.evaluators.numeric_judge import run_numeric_equivalence_judge
from agenqa.nodes.utils import (
    allowed_question_types_for_step,
    infer_question_type_from_qa,
    normalize_question_type,
    select_question_type,
)

logger = logging.getLogger(__name__)


def _solver_visible_question_for_judge(
    *,
    state: AgentState,
    solver_rows: list[tuple[int, Dict[str, Any]]],
) -> str:
    latest = state.history[-1] if state.history else None
    question = (latest.question if latest else "") or ""
    world_contract_text = (latest.world_contract_text if latest else "") or ""

    for _idx, row in solver_rows:
        if not isinstance(row, dict):
            continue
        precomposed = row.get("question_for_solver")
        if isinstance(precomposed, str) and precomposed.strip():
            return precomposed.strip()
        question = question or (row.get("question") or "")
        world_contract_text = world_contract_text or (row.get("world_contract_text") or "")

    return compose_solver_question(str(question), str(world_contract_text))


def _consensus_block(agent_conf: Dict[str, Any]) -> Dict[str, Any]:
    block = agent_conf.get("consensus") or {}
    return block if isinstance(block, dict) else {}


def _consensus_mode(agent_conf: Dict[str, Any], strong_count: int) -> str:
    raw = _consensus_block(agent_conf).get("mode")
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val == "always":
            return "always"
        if val in {"none", "disabled", "off"}:
            return "none"
        raise ValueError(f"invalid consensus.mode: {raw!r}; expected 'always' or 'none'")
    # Default: if multiple strong solvers are configured, run all by default.
    # Consensus can be explicitly disabled via `mode = "none"` / `disabled`.
    return "always" if strong_count > 1 else "none"


def _answer_judge_mode(agent_conf: Dict[str, Any], qtype: str) -> str:
    """Select how to aggregate answer agreement.

    Config override:
      consensus.answer_judge: "llm" | "normalize"

    Defaults:
    - MCQ: normalize
    - Numeric: llm
    - Derivation: llm
    """
    if qtype == "MCQ":
        return "normalize"
    raw = _consensus_block(agent_conf).get("answer_judge")
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in {"llm", "normalize"}:
            return val
    return "llm"


def _min_success(agent_conf: Dict[str, Any]) -> int:
    raw = _consensus_block(agent_conf).get("min_success", 2)
    try:
        val = int(raw)
        return val if val >= 2 else 2
    except Exception:
        return 2


def _infer_op_name_for_solve(state: AgentState) -> str:
    op_raw = (state.last_decision.operation if state.last_decision else "extend")  # type: ignore[union-attr]
    op_lc = str(op_raw or "").lower()
    if "qa_init" in op_lc or "qainit" in op_lc:
        return "qa_init"
    if "init" in op_lc:
        return "extend"
    if "extend" in op_lc:
        return "extend"
    return op_lc.replace("-", "_") or "extend"


def _first_row(path: Path) -> Dict[str, Any] | None:
    try:
        for row in read_jsonl(path, schema=None, max_lines=1):
            return row if isinstance(row, dict) else None
    except Exception:
        return None
    return None


def _detect_status(row: Dict[str, Any] | None) -> str:
    if not row:
        return "request_failed"
    solver_status = row.get("solver_status")
    if isinstance(solver_status, str) and solver_status.strip():
        low = solver_status.strip().lower()
        if low in {"success", "request_failed", "parse_failed"}:
            return low
    err = row.get("error")
    if isinstance(err, str) and err.strip():
        answer_pred = (row.get("answer_pred") or row.get("solve") or "")
        if not (isinstance(answer_pred, str) and answer_pred.strip()):
            return "request_failed"
        return "parse_failed"
    # Tool-enabled solver rows may include a `tool` payload. If the solver claims tool usage
    # but the tool run failed (or produced no parsable value), treat the vote as invalid.
    tool = row.get("tool")
    if isinstance(tool, dict) and bool(tool.get("used")):
        code = tool.get("code")
        if not isinstance(code, str) or not code.strip():
            return "parse_failed"
        exec_payload = tool.get("exec")
        if isinstance(exec_payload, dict):
            ok = exec_payload.get("success")
            if ok is False:
                return "request_failed"
            if ok is True:
                # Execution succeeded but we still failed to parse a numeric value.
                if tool.get("value") is None:
                    return "parse_failed"
    return "success"


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract the content of the first \\boxed{...} block, balancing braces."""
    if not isinstance(text, str) or not text:
        return None
    start = text.find("\\boxed{")
    if start == -1:
        start = text.find("boxed{")
        if start == -1:
            return None
    i = text.find("{", start)
    if i == -1:
        return None
    depth = 1
    i += 1
    begin = i
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i]
        i += 1
    return None


def _normalize_mcq(ans: str) -> Optional[str]:
    if not isinstance(ans, str) or not ans.strip():
        return None
    core = _extract_boxed_content(ans) or ans
    s = core.strip()
    # Strip common wrappers
    while True:
        s2 = s.strip()
        if len(s2) >= 2 and ((s2[0], s2[-1]) in {("(", ")"), ("[", "]"), ("{", "}")}):
            s = s2[1:-1]
            continue
        s = s2
        break
    # Pattern: Option A / 选项 A / Choice A
    m = re.search(r"(?:option|choice|选项)\s*([A-E])", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Plain single letter
    s_up = s.strip().upper()
    if len(s_up) == 1 and s_up in {"A", "B", "C", "D", "E"}:
        return s_up
    return None


def _normalize_calc(ans: str, *, tol_decimals: int = 6) -> Optional[str]:
    if not isinstance(ans, str) or not ans.strip():
        return None
    core = _extract_boxed_content(ans) or ans
    s = core.strip()
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\%", "%")
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    # Try numeric canonicalization (only when the whole string is a number).
    try:
        raw = s[:-1] if s.endswith("%") else s
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", raw):
            val = float(raw)
            return f"{val:.{tol_decimals}f}"
    except Exception:
        pass
    return s


def _normalize_answer(ans: str, qtype: str) -> Optional[str]:
    if qtype == "MCQ":
        return _normalize_mcq(ans)
    return _normalize_calc(ans)


def _infer_question_type(agent_conf: Dict[str, Any], state: AgentState) -> str:
    try:
        step_idx = int(state.step or 0)
    except Exception:
        step_idx = 0

    try:
        allowed = allowed_question_types_for_step(agent_conf or {}, step_idx)
    except Exception:
        raise

    def _validate_or_raise(qt: str, *, source: str) -> str:
        if allowed and qt not in allowed:
            raise ValueError(
                f"[Consensus] question_type violates policy "
                f"(run_id={getattr(state, 'run_id', None)!r}, step={step_idx}, source={source}, qtype={qt!r}, allowed={allowed})"
            )
        return qt

    # 0) Prefer locked metadata from the latest KQARecord.
    try:
        cur = state.history[-1] if state.history else None
        if cur is not None:
            locked = normalize_question_type(getattr(cur, "question_type", None))
            if not locked:
                constraints = getattr(cur, "question_type_constraints", None)
                if isinstance(constraints, dict):
                    locked = normalize_question_type(constraints.get("locked_question_type"))
            if locked:
                return _validate_or_raise(locked, source="history.locked")
    except ValueError:
        raise
    except Exception:
        pass

    # 1) Prefer explicit director decision hint (if any).
    try:
        params = state.last_decision.params if state.last_decision else None  # type: ignore[assignment]
    except Exception:
        params = None
    hinted = normalize_question_type((params or {}).get("question_type")) if isinstance(params, dict) else None
    if hinted:
        return _validate_or_raise(hinted, source="director_hint")

    # 2) Fallback: infer from current QA content (revise is allowed to re-determine qtype).
    try:
        cur = state.history[-1] if state.history else None
        q = getattr(cur, "question", "") if cur is not None else ""
        a = getattr(cur, "answer", "") if cur is not None else ""
        inferred = infer_question_type_from_qa(q, a)
        if inferred:
            inferred_norm = normalize_question_type(inferred) or inferred
            return _validate_or_raise(str(inferred_norm), source="qa_heuristic")
    except ValueError:
        raise
    except Exception:
        pass

    # 3) Final fallback: previous behavior (default MCQ).
    return _validate_or_raise(select_question_type(state) or "MCQ", source="select_question_type")


def _iter_strong_solver_files(solve_dir: Path) -> list[tuple[int, Path]]:
    files: dict[int, Path] = {}
    if not solve_dir.exists():
        return []
    # Prefer indexed files.
    for path in solve_dir.glob("solve_strong_*.jsonl"):
        if path.name.endswith("_raw.jsonl"):
            continue
        m = re.fullmatch(r"solve_strong_(\d+)\.jsonl", path.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        files[idx] = path
    return sorted(files.items(), key=lambda x: x[0])


def _iter_path_strong_solver_files(solve_dir: Path) -> list[tuple[int, Path]]:
    """Iterate path-view strong solver artifacts: solve_path_strong(_i).jsonl."""
    files: dict[int, Path] = {}
    if not solve_dir.exists():
        return []
    for path in solve_dir.glob("solve_path_strong_*.jsonl"):
        if path.name.endswith("_raw.jsonl"):
            continue
        m = re.fullmatch(r"solve_path_strong_(\d+)\.jsonl", path.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        files[idx] = path
    return sorted(files.items(), key=lambda x: x[0])


def _collect_solver_votes(
    file_items: list[tuple[int, Path]],
) -> tuple[list[tuple[int, Dict[str, Any] | None]], list[SolverVote], dict[str, int]]:
    """Read solver rows and build vote objects from a solver file list."""
    solver_rows: list[tuple[int, Dict[str, Any] | None]] = []
    votes: list[SolverVote] = []
    wellposed_votes: dict[str, int] = {"true": 0, "false": 0}

    for idx, path in file_items:
        row = _first_row(path)
        solver_rows.append((idx, row))

        status = _detect_status(row)
        service_id = row.get("service_id") if isinstance(row, dict) else None
        model = row.get("model") if isinstance(row, dict) else None
        answer_raw = row.get("solve") if isinstance(row, dict) else None
        qwp = row.get("question_well_posed") if isinstance(row, dict) else None
        if qwp is not None and not isinstance(qwp, bool):
            try:
                qwp = bool(qwp)
            except Exception:
                qwp = None

        if status == "success" and isinstance(qwp, bool):
            wellposed_votes["true" if qwp else "false"] += 1

        votes.append(
            SolverVote(
                solver_idx=idx,
                service_id=service_id if isinstance(service_id, str) else None,
                model=model if isinstance(model, str) else None,
                status=status,
                answer=answer_raw if isinstance(answer_raw, str) else None,
                answer_normalized=None,
                question_well_posed=qwp if isinstance(qwp, bool) else None,
                correctness_feedback=(row.get("correctness_feedback") if isinstance(row, dict) else None),
                difficulty_feedback=(row.get("difficulty_feedback") if isinstance(row, dict) else None),
                judge_status=(
                    ((row.get("expression_judge") or {}).get(f"strong_{idx}", {}) if isinstance(row, dict) else {}).get("status")
                    if isinstance((row.get("expression_judge") if isinstance(row, dict) else None), dict)
                    else None
                ),
                judge_reason=(
                    (((row.get("expression_judge") or {}).get(f"strong_{idx}", {}) if isinstance(row, dict) else {}).get("skip_reason")
                     or ((row.get("expression_judge") or {}).get(f"strong_{idx}", {}) if isinstance(row, dict) else {}).get("failure_reason"))
                    if isinstance((row.get("expression_judge") if isinstance(row, dict) else None), dict)
                    else None
                ),
            )
        )

    return solver_rows, votes, wellposed_votes


def _write_path_multi_strong_ambiguity_report(
    *,
    solve_dir: Path,
    step_idx: int,
    round_idx: int,
    qtype: str,
) -> None:
    """Write a lightweight ambiguity report for Type1 detection (path view, multi-strong).

    This is intentionally deterministic and does NOT call any extra judge/LLM.
    """
    out_path = solve_dir / "ambiguity_report.json"
    files = _iter_path_strong_solver_files(solve_dir)
    if not files:
        return

    votes: list[dict[str, Any]] = []
    wellposed_votes: dict[str, int] = {"true": 0, "false": 0, "null": 0}
    evidence_files: list[str] = []

    def _truncate(val: Any, max_chars: int) -> str | None:
        if not isinstance(val, str):
            return None
        s = val.strip()
        if not s:
            return None
        if len(s) <= max_chars:
            return s
        return s[: max(0, max_chars - 1)] + "…"

    for idx, path in files:
        evidence_files.append(path.name)
        row = _first_row(path)
        status = _detect_status(row)
        service_id = row.get("service_id") if isinstance(row, dict) else None
        model = row.get("model") if isinstance(row, dict) else None
        ans_raw = row.get("solve") if isinstance(row, dict) else None
        qwp = row.get("question_well_posed") if isinstance(row, dict) else None
        if qwp is not None and not isinstance(qwp, bool):
            try:
                qwp = bool(qwp)
            except Exception:
                qwp = None

        if status == "success" and isinstance(qwp, bool):
            wellposed_votes["true" if qwp else "false"] += 1
        else:
            wellposed_votes["null"] += 1

        votes.append(
            {
                "idx": idx,
                "id": (service_id if isinstance(service_id, str) else None) or (model if isinstance(model, str) else None),
                "status": status,
                "answer_preview": _truncate(ans_raw, 220),
                "wellposed": qwp if isinstance(qwp, bool) else None,
            }
        )

    # Type1 should not be triggered by textual/signature divergence across equivalent answers.
    # Keep the trigger narrow and evidence-grounded: only explicit not-well-posed votes.
    type1_suspected = wellposed_votes.get("false", 0) > 0

    payload = {
        "step": int(step_idx),
        "round": int(round_idx),
        "view": "path",
        "tier": "strong",
        "question_type": str(qtype),
        "type1_suspected": bool(type1_suspected),
        "wellposed_votes": dict(wellposed_votes),
        "solvers": votes,
        "evidence_files": evidence_files,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_answer_key_for_vote(ans: str, qtype: str) -> Optional[str]:
    """Normalize a solver answer into a stable vote key.

    For non-MCQ answers (Derivation/Numeric), we cap long keys and append a short hash so that:
    - keys stay reasonably small for logs/prompts; and
    - different long expressions won't collide after truncation.
    """
    normalized = _normalize_answer(ans, qtype)
    if normalized is None:
        return None
    if qtype == "MCQ":
        return normalized

    # Derivation/Numeric: numeric keys stay numeric; non-numeric keys get a short hash.
    raw = normalized[:-1] if normalized.endswith("%") else normalized
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", raw):
        return normalized

    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    max_len = 160
    if len(normalized) > max_len:
        return f"{normalized[:max_len]}…#h={digest}"
    return f"{normalized}#h={digest}"


def _cluster_derivation_answers_with_llm_judge(
    *,
    generator: Dict[str, Any],
    known: str,
    question: str,
    answers: list[tuple[int, str]],
    answer_contract: Dict[str, Any] | None = None,
    lang: Optional[str] = None,
) -> Tuple[dict[int, str], dict[str, str]]:
    """Cluster Derivation answers by LLM-equivalence.

    Returns:
      - assignments: solver_idx -> vote_key (cluster key)
      - reps: vote_key -> representative raw answer string
    """
    eq_cache: dict[tuple[str, str], Optional[bool]] = {}
    clusters: list[dict[str, Any]] = []
    assignments: dict[int, str] = {}
    reps: dict[str, str] = {}

    for solver_idx, ans in answers:
        cand_key = _make_answer_key_for_vote(ans, "Derivation")
        if cand_key is None:
            continue

        placed = False
        for c in clusters:
            rep_key = c["key"]
            rep_ans = c["rep_answer"]

            if cand_key == rep_key:
                assignments[solver_idx] = rep_key
                placed = True
                break

            cache_key = (rep_key, cand_key)
            if cache_key in eq_cache:
                eq = eq_cache[cache_key]
            else:
                try:
                    eq, _reason = run_expression_equivalence_judge(
                        generator,
                        known=known,
                        question=question,
                        answer_ref=rep_ans,
                        answer_pred=ans,
                        answer_contract=answer_contract if isinstance(answer_contract, dict) else None,
                        lang=lang,
                    )
                except Exception:
                    eq = None
                eq_cache[cache_key] = eq
                eq_cache[(cand_key, rep_key)] = eq

            if eq is True:
                assignments[solver_idx] = rep_key
                placed = True
                break

        if not placed:
            clusters.append({"key": cand_key, "rep_answer": ans})
            assignments[solver_idx] = cand_key
            reps[cand_key] = ans

    for c in clusters:
        reps.setdefault(c["key"], c["rep_answer"])
    return assignments, reps


def _cluster_numeric_answers_with_llm_judge(
    *,
    generator: Dict[str, Any],
    known: str,
    question: str,
    answers: list[tuple[int, str]],
    lang: Optional[str] = None,
) -> Tuple[dict[int, str], dict[str, str]]:
    """Cluster Numeric answers by LLM-equivalence.

    Returns:
      - assignments: solver_idx -> vote_key (cluster key)
      - reps: vote_key -> representative raw answer string
    """
    eq_cache: dict[tuple[str, str], bool] = {}
    clusters: list[dict[str, Any]] = []
    assignments: dict[int, str] = {}
    reps: dict[str, str] = {}

    for solver_idx, ans in answers:
        cand_key = _make_answer_key_for_vote(ans, "Numeric")
        if cand_key is None:
            continue

        placed = False
        for c in clusters:
            rep_key = c["key"]
            rep_ans = c["rep_answer"]

            if cand_key == rep_key:
                assignments[solver_idx] = rep_key
                placed = True
                break

            cache_key = (rep_key, cand_key)
            if cache_key in eq_cache:
                eq = eq_cache[cache_key]
            else:
                eq, _reason = run_numeric_equivalence_judge(
                    generator,
                    known=known,
                    question=question,
                    answer_ref=rep_ans,
                    answer_pred=ans,
                    lang=lang,
                )
                eq_cache[cache_key] = eq
                eq_cache[(cand_key, rep_key)] = eq

            if eq is True:
                assignments[solver_idx] = rep_key
                placed = True
                break

        if not placed:
            clusters.append({"key": cand_key, "rep_answer": ans})
            assignments[solver_idx] = cand_key
            reps[cand_key] = ans

    for c in clusters:
        reps.setdefault(c["key"], c["rep_answer"])
    return assignments, reps


def _assign_answer_vote_keys(
    *,
    agent_conf: Dict[str, Any],
    state: AgentState,
    qtype: str,
    answer_judge_mode: str,
    agent_lang: Optional[str],
    solver_rows: list[tuple[int, Dict[str, Any] | None]],
    votes: list[SolverVote],
) -> None:
    """Populate `answer_normalized` for a vote set in place."""
    if qtype == "MCQ" or answer_judge_mode == "normalize":
        for v in votes:
            if v.status == "success" and isinstance(v.answer, str) and v.answer.strip():
                v.answer_normalized = _make_answer_key_for_vote(v.answer, qtype)
        return

    generator: Optional[Dict[str, Any]] = None
    try:
        generator = load_expression_judge_generator(agent_conf)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "consensus: answer_judge=llm requires expression_judge.generator to be configured"
        ) from exc
    if not generator:
        raise ValueError("consensus: answer_judge=llm requires expression_judge.generator to be configured")

    latest = state.history[-1] if state.history else None
    known = (latest.known if latest else "") or ""
    question = _solver_visible_question_for_judge(state=state, solver_rows=solver_rows)
    try:
        mem = KnownTree.normalize_memory(getattr(state, "memory", None))
        type2_ctx = extract_answer_contract_context(mem, step=int(state.step or 0))
    except Exception:
        type2_ctx = {}
    if not (known and question):
        for _idx, row in solver_rows:
            if not isinstance(row, dict):
                continue
            known = known or (row.get("known") or "")
            if known and question:
                break

    candidates: list[tuple[int, str]] = []
    for v in votes:
        if v.status == "success" and isinstance(v.answer, str) and v.answer.strip():
            candidates.append((v.solver_idx, v.answer))

    if qtype == "Numeric":
        assignments, _reps = _cluster_numeric_answers_with_llm_judge(
            generator=generator,
            known=str(known),
            question=str(question),
            answers=candidates,
            lang=agent_lang,
        )
    else:
        assignments, _reps = _cluster_derivation_answers_with_llm_judge(
            generator=generator,
            known=str(known),
            question=str(question),
            answers=candidates,
            answer_contract=type2_ctx if isinstance(type2_ctx, dict) else None,
            lang=agent_lang,
        )
    for v in votes:
        if v.solver_idx in assignments:
            v.answer_normalized = assignments[v.solver_idx]


def _build_consensus_summary_payload(
    *,
    votes: list[SolverVote],
    qtype: str,
    min_success: int,
    proposed_raw: Optional[str],
    answer_judge_mode: str,
    agent_conf: Dict[str, Any],
    state: AgentState,
    solver_rows: list[tuple[int, Dict[str, Any] | None]],
    agent_lang: Optional[str],
    view: str,
    step_idx: int,
    round_idx: int,
    problem_id: str,
) -> dict[str, Any]:
    """Build a serialized consensus payload from a vote set."""
    answer_votes: dict[str, int] = {}
    wellposed_votes: dict[str, int] = {"true": 0, "false": 0}
    status_counts: dict[str, int] = {
        "success": 0,
        "request_failed": 0,
        "parse_failed": 0,
        "judge_failed": 0,
        "judge_skipped": 0,
    }
    ineligible_reasons: list[dict[str, Any]] = []
    for v in votes:
        if isinstance(v.status, str):
            status_counts[v.status] = status_counts.get(v.status, 0) + 1
        if v.judge_status == "failed":
            status_counts["judge_failed"] += 1
        elif v.judge_status == "skipped":
            status_counts["judge_skipped"] += 1
        if isinstance(v.question_well_posed, bool):
            wellposed_votes["true" if v.question_well_posed else "false"] += 1
        if v.status == "success" and isinstance(v.answer_normalized, str) and v.answer_normalized:
            answer_votes[v.answer_normalized] = answer_votes.get(v.answer_normalized, 0) + 1
        else:
            if v.status != "success":
                reason = v.status
            elif not isinstance(v.answer, str) or not v.answer.strip():
                reason = "missing_answer_pred"
            else:
                reason = "missing_normalized_answer"
            ineligible_reasons.append(
                {
                    "idx": v.solver_idx,
                    "id": v.service_id or v.model,
                    "reason": reason,
                    **({"judge_status": v.judge_status} if isinstance(v.judge_status, str) and v.judge_status else {}),
                    **({"judge_reason": v.judge_reason} if isinstance(v.judge_reason, str) and v.judge_reason else {}),
                }
            )

    eligible_votes = sum(answer_votes.values())
    answer_consensus: Optional[str] = None
    consensus_strength = 0
    tie = False
    tie_reason: Optional[str] = None

    if eligible_votes < min_success:
        tie = True
        tie_reason = "insufficient_votes"
    else:
        best_key = None
        best_count = -1
        for k, v in answer_votes.items():
            if v > best_count:
                best_key = k
                best_count = v
        if best_key is not None and best_count > eligible_votes / 2:
            answer_consensus = best_key
            consensus_strength = best_count
            tie = False
        else:
            tie = True
            if len(answer_votes) == eligible_votes:
                tie_reason = "all different"
            elif len(answer_votes) == 2:
                tie_reason = "2-way split"
            else:
                tie_reason = "multi-way split"

    wellposed_consensus: Optional[bool] = None
    wp_eligible = wellposed_votes["true"] + wellposed_votes["false"]
    if wp_eligible >= min_success:
        if wellposed_votes["true"] > wp_eligible / 2:
            wellposed_consensus = True
        elif wellposed_votes["false"] > wp_eligible / 2:
            wellposed_consensus = False

    differs_from_proposed: Optional[bool] = None
    proposed_norm = _make_answer_key_for_vote(proposed_raw, qtype) if isinstance(proposed_raw, str) else None
    if answer_consensus is not None and proposed_raw is not None:
        if qtype == "Derivation" and answer_judge_mode == "llm":
            try:
                generator = load_expression_judge_generator(agent_conf)
            except Exception as exc:
                raise ValueError(
                    "consensus: answer_judge=llm requires expression_judge.generator to be configured"
                ) from exc
            if not generator:
                raise ValueError(
                    "consensus: answer_judge=llm requires expression_judge.generator to be configured"
                )
            rep_raw = None
            for v in votes:
                if v.answer_normalized == answer_consensus and isinstance(v.answer, str) and v.answer.strip():
                    rep_raw = v.answer
                    break
            latest = state.history[-1] if state.history else None
            known = (latest.known if latest else "") or ""
            question = _solver_visible_question_for_judge(state=state, solver_rows=solver_rows)
            if rep_raw and known and question:
                try:
                    try:
                        mem = KnownTree.normalize_memory(getattr(state, "memory", None))
                        type2_ctx = extract_answer_contract_context(mem, step=int(state.step or 0))
                    except Exception:
                        type2_ctx = {}
                    eq, _reason = run_expression_equivalence_judge(
                        generator,
                        known=str(known),
                        question=str(question),
                        answer_ref=str(proposed_raw),
                        answer_pred=str(rep_raw),
                        answer_contract=type2_ctx if isinstance(type2_ctx, dict) else None,
                        lang=agent_lang,
                    )
                    if eq is True:
                        differs_from_proposed = False
                    elif eq is False:
                        differs_from_proposed = True
                except Exception:
                    pass
        if differs_from_proposed is None and proposed_norm is not None:
            differs_from_proposed = answer_consensus != proposed_norm

    return {
        "view": view,
        "step": step_idx,
        "round": round_idx,
        "problem_id": problem_id,
        "question_type": qtype,
        "answer_judge": answer_judge_mode,
        "solvers": [
            {
                "idx": v.solver_idx,
                "id": v.service_id or v.model,
                "status": v.status,
                "answer": v.answer_normalized,
                "wellposed": v.question_well_posed,
                "judge_status": v.judge_status,
                "judge_reason": v.judge_reason,
            }
            for v in votes
        ],
        "status_counts": dict(status_counts),
        "ineligible_reasons": ineligible_reasons,
        "answer_votes": dict(answer_votes),
        "wellposed_votes": dict(wellposed_votes),
        "answer_consensus": answer_consensus,
        "wellposed_consensus": wellposed_consensus,
        "consensus_strength": consensus_strength,
        "eligible_votes": eligible_votes,
        "differs_from_proposed": differs_from_proposed,
        "proposed_answer": proposed_norm,
        "tie": tie,
        "tie_reason": tie_reason,
    }


def compute_consensus(agent_conf: Dict[str, Any], state: AgentState) -> AgentState:
    """Compute and store strong-solver consensus for the current (step, round)."""
    try:
        step_idx = int(state.step or 0)
    except Exception:
        step_idx = 0
    try:
        round_idx = int(getattr(state, "rounds", 1) or 1)
    except Exception:
        round_idx = 1

    # Determine expected strong solver count from config (best-effort)
    strong_conf_raw = (agent_conf.get("solvers") or {}).get("strong")
    strong_count = len(strong_conf_raw) if isinstance(strong_conf_raw, list) else (1 if isinstance(strong_conf_raw, dict) else 0)
    mode = _consensus_mode(agent_conf, strong_count)
    agent_lang = str(((agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}).get("lang") or "").lower().strip() or None

    proposed_raw: Optional[str]
    try:
        proposed_raw = state.history[-1].answer if state.history else None
    except Exception:
        proposed_raw = None

    if mode == "none":
        state.solver_consensus = SolverConsensus(strong=StrongSolverConsensus(mode="none", proposed_answer=proposed_raw))
        try:
            save_state(state)
        except Exception:
            pass
        return state

    op_name = _infer_op_name_for_solve(state)
    revise_mode = None
    if op_name == "revise":
        try:
            from agenqa.nodes.op_revise import _infer_revise_mode

            revise_mode = _infer_revise_mode(state)
        except Exception:
            revise_mode = None
    step_dir = compute_step_dir(Path(state.artifacts_dir), op_name, step_idx, round_idx, revise_mode=revise_mode)
    solve_dir = step_dir / "solve"

    qtype = _infer_question_type(agent_conf, state)
    answer_judge_mode = _answer_judge_mode(agent_conf, qtype)
    min_success = _min_success(agent_conf)

    edge_solver_rows, edge_votes, _edge_wp_votes = _collect_solver_votes(_iter_strong_solver_files(solve_dir))
    _assign_answer_vote_keys(
        agent_conf=agent_conf,
        state=state,
        qtype=qtype,
        answer_judge_mode=answer_judge_mode,
        agent_lang=agent_lang,
        solver_rows=edge_solver_rows,
        votes=edge_votes,
    )

    paper_id = None
    try:
        paper_id = state.history[-1].paper_id if state.history else None
    except Exception:
        paper_id = None
    if not paper_id:
        paper_id = getattr(state, "paper_id", None)
    problem_id = f"{paper_id}_step{step_idx}" if paper_id else f"unknown_step{step_idx}"
    edge_payload = _build_consensus_summary_payload(
        votes=edge_votes,
        qtype=qtype,
        min_success=min_success,
        proposed_raw=proposed_raw,
        answer_judge_mode=answer_judge_mode,
        agent_conf=agent_conf,
        state=state,
        solver_rows=edge_solver_rows,
        agent_lang=agent_lang,
        view="edge",
        step_idx=step_idx,
        round_idx=round_idx,
        problem_id=problem_id,
    )
    path_solver_rows, path_votes, _path_wp_votes = _collect_solver_votes(_iter_path_strong_solver_files(solve_dir))
    _assign_answer_vote_keys(
        agent_conf=agent_conf,
        state=state,
        qtype=qtype,
        answer_judge_mode=answer_judge_mode,
        agent_lang=agent_lang,
        solver_rows=path_solver_rows,
        votes=path_votes,
    )
    path_payload = _build_consensus_summary_payload(
        votes=path_votes,
        qtype=qtype,
        min_success=min_success,
        proposed_raw=proposed_raw,
        answer_judge_mode=answer_judge_mode,
        agent_conf=agent_conf,
        state=state,
        solver_rows=path_solver_rows,
        agent_lang=agent_lang,
        view="path",
        step_idx=step_idx,
        round_idx=round_idx,
        problem_id=problem_id,
    )

    state.solver_consensus = SolverConsensus(
        strong=StrongSolverConsensus(
            mode=mode,
            proposed_answer=proposed_raw,
            answer_consensus=edge_payload.get("answer_consensus"),
            wellposed_consensus=edge_payload.get("wellposed_consensus"),
            differs_from_proposed=edge_payload.get("differs_from_proposed"),
            consensus_strength=int(edge_payload.get("consensus_strength") or 0),
            eligible_votes=int(edge_payload.get("eligible_votes") or 0),
            tie=bool(edge_payload.get("tie")),
            tie_reason=edge_payload.get("tie_reason"),
            solvers=edge_votes,
        )
    )

    edge_payload["mode"] = mode
    path_payload["mode"] = mode

    # Write summary for debugging / replay
    try:
        solve_dir.mkdir(parents=True, exist_ok=True)
        (solve_dir / "consensus_summary.json").write_text(
            json.dumps(edge_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (solve_dir / "consensus_summary_edge.json").write_text(
            json.dumps(edge_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (solve_dir / "consensus_summary_path.json").write_text(
            json.dumps(path_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Write consensus_summary.json failed: %s", str(exc))

    # Write a lightweight ambiguity report for Type1 detection (path-view multi-strong).
    try:
        _write_path_multi_strong_ambiguity_report(
            solve_dir=solve_dir,
            step_idx=step_idx,
            round_idx=round_idx,
            qtype=qtype,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Write ambiguity_report.json failed: %s", str(exc))

    # Keep state snapshot consistent with on-disk artifacts.
    try:
        save_state(state)
    except Exception:
        pass

    logger.info(
        "Consensus[strong]: mode=%s eligible=%s answer=%s wellposed=%s tie=%s reason=%s",
        mode,
        edge_payload.get("eligible_votes"),
        edge_payload.get("answer_consensus"),
        edge_payload.get("wellposed_consensus"),
        edge_payload.get("tie"),
        edge_payload.get("tie_reason"),
    )
    return state


__all__ = ["compute_consensus"]
