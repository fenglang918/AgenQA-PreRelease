"""Shared utilities for node-level orchestration."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Dict, Optional

from agenqa.graph.state import AgentState

_IDEALAB_SESSION_HEADER = "x-idealab-session-id"


def _expand_env_str(val: Any) -> str:
    if not isinstance(val, str):
        return ""
    try:
        return os.path.expandvars(val)
    except Exception:
        return val


def _is_idealab_generator(generator: Dict[str, Any]) -> bool:
    if not isinstance(generator, dict):
        return False
    base = generator.get("api_base") or generator.get("base_url") or ""
    base = _expand_env_str(base).lower()
    return "idealab" in base


def _session_base_id(state: AgentState) -> str:
    """Return the run-scoped base session id (stable for the whole run)."""
    for key in ("IDEALAB_SESSION_ID", "SCICLONE_IDEALAB_SESSION_ID", "SCICLONE_SESSION_ID"):
        raw = os.environ.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    rid = getattr(state, "run_id", None)
    if isinstance(rid, str) and rid.strip():
        return f"run-{rid.strip()}"
    return "run-unknown"


def idealab_session_id_for_director(state: AgentState) -> str:
    """Director uses one session id from start to end."""
    return f"{_session_base_id(state)}:director"


def idealab_session_id_for_step_node(state: AgentState, node: str, step_idx: int) -> str:
    """Each step node gets its own session id (shared by all sub-calls in that node)."""
    node_norm = str(node or "node").strip().lower().replace(" ", "_")
    return f"{_session_base_id(state)}:{node_norm}:s{int(step_idx)}"


def with_idealab_session_id(generator: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Return a copy of generator config with x-idealab-session-id attached (Idealab only)."""
    if not isinstance(generator, dict) or not generator:
        return generator
    if not session_id:
        return generator
    if not _is_idealab_generator(generator):
        return generator

    gen = deepcopy(generator)
    client = gen.get("client")
    if not isinstance(client, dict):
        client = {}
    else:
        client = dict(client)
    extra = client.get("extra_headers")
    if not isinstance(extra, dict):
        extra = {}
    else:
        extra = dict(extra)
    extra[_IDEALAB_SESSION_HEADER] = session_id
    client["extra_headers"] = extra
    gen["client"] = client
    return gen


def normalize_question_type(val: Any) -> Optional[str]:
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if not v:
        return None
    if v in {"mcq", "choice", "single", "single_choice", "single-choice"}:
        return "MCQ"
    # Derivation: symbolic/analytic reasoning, not necessarily numeric.
    if v in {"derivation", "derive", "symbolic", "proof", "algebra", "formula"}:
        return "Derivation"
    # Numeric: numeric computation / estimation; typically needs explicit tolerance/precision in the question.
    if v in {"numeric", "calc", "calculation", "compute", "number", "estimate", "estimation", "approx", "approximation"}:
        return "Numeric"
    return None


def allowed_question_types_for_step(agent_conf: Dict[str, Any], step_idx: int) -> list[str]:
    """Compute effective allowed question types for a given semantic step (1-based).

    Semantics:
    1) Compute base_allowed from `agent.question_type_policy.no_mcq_from_step`:
       - If step >= threshold: base_allowed = {"Derivation", "Numeric"}
       - Else: base_allowed = {"MCQ", "Derivation", "Numeric"}
       - threshold default: 2; explicit null/off disables the policy.
    2) If `agent.question_type_policy.allowed_question_types` is set, intersect (preserving policy order).
    3) If the intersection is empty, fail-fast with ValueError.

    Notes:
    - For executable track, question types are not used; returns [].
    """

    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
    track = str(agent_block.get("track") or "").strip().lower() or "unified"
    if track == "executable":
        return []

    try:
        step_i = int(step_idx or 0)
    except Exception:
        step_i = 0

    policy = agent_block.get("question_type_policy")
    policy = policy if isinstance(policy, dict) else {}

    def _no_mcq_from_step() -> int | None:
        # Distinguish explicit null from missing key:
        # - missing key -> default 2
        # - key present with null/off -> disable
        if "no_mcq_from_step" not in policy:
            return 2
        raw = policy.get("no_mcq_from_step")
        if raw is None or raw is False:
            return None
        if isinstance(raw, (int, float)):
            try:
                v = int(raw)
                return v if v > 0 else None
            except Exception:
                return 2
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in {"off", "disable", "disabled", "none", "null"}:
                return None
            try:
                v = int(s)
                return v if v > 0 else None
            except Exception:
                return 2
        return 2

    threshold = _no_mcq_from_step()
    if threshold is not None and step_i >= threshold:
        base_allowed = ["Derivation", "Numeric"]
    else:
        base_allowed = ["MCQ", "Derivation", "Numeric"]

    raw_allowed = policy.get("allowed_question_types")
    if raw_allowed is None or raw_allowed is False:
        return base_allowed
    if isinstance(raw_allowed, str):
        s = raw_allowed.strip().lower()
        if s in {"off", "disable", "disabled", "none", "null"}:
            return base_allowed
        parts = [p for p in re.split(r"[,\s]+", raw_allowed.strip()) if p]
    elif isinstance(raw_allowed, (list, tuple)):
        parts = []
        for x in raw_allowed:
            if x is None:
                continue
            if isinstance(x, str):
                parts.extend([p for p in re.split(r"[,\s]+", x.strip()) if p])
            else:
                parts.append(str(x))
    else:
        raise ValueError(f"[CONFIG] invalid agent.question_type_policy.allowed_question_types type={type(raw_allowed)!r}")

    if not parts:
        raise ValueError("[CONFIG] agent.question_type_policy.allowed_question_types is empty")

    policy_allowed: list[str] = []
    seen: set[str] = set()
    for p in parts:
        qt = normalize_question_type(p)
        if not qt:
            raise ValueError(
                f"[CONFIG] invalid question type in allowed_question_types: {p!r} (expected MCQ/Derivation/Numeric)"
            )
        if qt in seen:
            continue
        seen.add(qt)
        policy_allowed.append(qt)

    base_set = set(base_allowed)
    effective = [qt for qt in policy_allowed if qt in base_set]
    if not effective:
        raise ValueError(
            f"[CONFIG] question_type_policy.allowed_question_types yields empty effective set "
            f"(step={step_i}, base_allowed={base_allowed}, policy_allowed={policy_allowed})"
        )
    return effective


_NUM_TOKEN_RE = re.compile(r"[-+]?(?:\\d+(?:\\.\\d+)?|\\.\\d+)(?:[eE][-+]?\\d+)?")


def infer_question_type_from_qa(question: Any, answer: Any = None) -> Optional[str]:
    """Heuristic question type inference from current QA text.

    This is used as a fallback when director decision does not provide an explicit question_type.
    It is intentionally conservative:
    - MCQ: detect typical option markers (A/B/C/D).
    - Numeric: detect tolerance/precision cues or numeric \\boxed{...}/scalar answers.
    - Otherwise: Derivation.
    """
    q = str(question or "")
    a = str(answer or "")
    if not q.strip():
        return None

    q_lc = q.lower()
    a_lc = a.lower()

    def _looks_like_mcq(text: str) -> bool:
        if all(x in text for x in ("A.", "B.", "C.", "D.")):
            return True
        # Multiline option markers
        if re.search(r"(?m)^\\s*A[\\.\\)\\、:：]", text) and re.search(r"(?m)^\\s*B[\\.\\)\\、:：]", text):
            return True
        if ("选项" in text) and ("A" in text) and ("B" in text):
            return True
        return False

    def _looks_like_numeric(q_text: str, a_text: str) -> bool:
        blob = f"{q_text}\\n{a_text}"
        blob_lc = blob.lower()
        # Tolerance / precision / rounding cues.
        for kw in (
            "abs_tol",
            "rel_tol",
            "tolerance",
            "within",
            "precision",
            "significant",
            "decimal",
            "approx",
            "approximately",
            "estimate",
            "compute",
            "calculate",
        ):
            if kw in blob_lc:
                return True
        for kw in ("误差", "容差", "保留", "小数", "有效数字", "近似", "估算", "计算", "±"):
            if kw in blob:
                return True

        # \\boxed{...} with a scalar-like payload.
        m = re.search(r"\\\\boxed\\{([^}]*)\\}", blob)
        boxed = (m.group(1).strip() if m else "")
        candidate = boxed or a_text.strip()
        if not candidate:
            return False
        if not re.search(r"\\d", candidate):
            return False
        # Accept "number [unit]" forms.
        # Examples: 3.91, -1.2e-3, 3.91 us, 3.91 μs, 50%
        cand = candidate.replace("−", "-").strip()
        m2 = _NUM_TOKEN_RE.match(cand)
        if not m2:
            return False
        rest = cand[m2.end() :].strip()
        if not rest:
            return True
        if rest == "%":
            return True
        # A small set of common unit-ish characters.
        if re.fullmatch(r"[a-zA-Zμµ°/\\^\\-_*·\\s]+", rest):
            return True
        return False

    if _looks_like_mcq(q):
        return "MCQ"
    if _looks_like_numeric(q, a):
        return "Numeric"
    return "Derivation"


def has_zam_cal_marker(val: Any) -> bool:
    try:
        s = str(val or "").strip().lower()
    except Exception:
        return False
    return "zam_cal" in s or "_calc" in s


def select_question_type(state: AgentState) -> str:
    """Extract question type hint from the last director decision."""
    try:
        params = getattr(state, "last_decision", None).params  # type: ignore[assignment]
    except Exception:
        params = None
    qtype = None
    if isinstance(params, dict):
        qtype = normalize_question_type(params.get("question_type"))
    return qtype or "MCQ"


def is_symbolic_only(agent_conf: Dict[str, Any]) -> bool:
    """Agent-level symbolic-only switch shared by extend/solve."""
    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
    val = agent_block.get("symbolic_only")
    if isinstance(val, bool):
        return val
    allow_numeric = agent_block.get("allow_numeric_values")
    if isinstance(allow_numeric, bool):
        return not allow_numeric
    return False


def is_symbolic_only_for_question_type(agent_conf: Dict[str, Any], question_type: Any) -> bool:
    """Question-type aware symbolic-only switch.

    Motivation:
    - Some experiments want Derivation to be "symbolic-only" (no numeric evaluation),
      while still allowing MCQ to carry concrete numbers and allowing Numeric to use tools.

    Priority:
    1) agent.symbolic_only == true (global)
    2) agent.allow_numeric_values == false (global)
    3) agent.symbolic_only_question_types includes the current question_type (per-qtype)
    """
    if is_symbolic_only(agent_conf):
        return True

    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
    raw = agent_block.get("symbolic_only_question_types")
    if raw is None:
        return False

    qtype = normalize_question_type(question_type)
    if not qtype:
        return False

    items: list[str] = []
    if isinstance(raw, str):
        parts = [p for p in raw.replace(",", " ").split() if p]
        items = parts
    elif isinstance(raw, (list, tuple)):
        for x in raw:
            if x is None:
                continue
            if isinstance(x, str):
                parts = [p for p in x.replace(",", " ").split() if p]
                items.extend(parts)
            else:
                items.append(str(x))
    else:
        return False

    for it in items:
        if normalize_question_type(it) == qtype:
            return True
    return False


def build_director_notes(state: AgentState, *, include_solver_feedback: bool = False) -> Optional[str]:
    """Summarize solver context and operator notes for downstream operators.

    Args:
        state: Agent state
        include_solver_feedback: If True, include solver_feedback natural-language text.
            Defaults to False to avoid polluting generation operators (e.g., Extend/Compress).
    """
    try:
        params = getattr(state, "last_decision", None).params  # type: ignore[assignment]
    except Exception:
        params = None
    if not isinstance(params, dict):
        return None

    notes_parts: list[str] = []
    note_val = (
        params.get("operator_notes")
        or params.get("notes")
        or params.get("note")
        or params.get("hint")
        or params.get("hints")
    )
    if isinstance(note_val, str) and note_val.strip():
        notes_parts.append(note_val.strip())

    solver_ctx = params.get("solver_context") if isinstance(params.get("solver_context"), dict) else {}
    solver_metrics = solver_ctx.get("solver_metrics") if isinstance(solver_ctx, dict) else {}

    if include_solver_feedback:
        emitted_feedback = False
        fb = solver_ctx.get("solver_feedback") if isinstance(solver_ctx, dict) else {}
        if isinstance(fb, dict):
            tier = fb.get("from_tier")
            step_src = fb.get("from_step")
            qwp = fb.get("question_well_posed")
            if qwp is not None:
                notes_parts.append(f"question_well_posed({tier}@step{step_src}): {qwp}")
                emitted_feedback = True
            cf_txt = fb.get("correctness_feedback")
            if isinstance(cf_txt, str) and cf_txt.strip():
                notes_parts.append(f"correctness_feedback({tier}@step{step_src}): {cf_txt.strip()}")
                emitted_feedback = True
            df_txt = fb.get("difficulty_feedback")
            if isinstance(df_txt, str) and df_txt.strip():
                notes_parts.append(f"difficulty_feedback({tier}): {df_txt.strip()}")
                emitted_feedback = True

        if (not emitted_feedback) and isinstance(solver_metrics, dict):
            for view_name in ("edge", "path"):
                view = solver_metrics.get(view_name) if isinstance(solver_metrics.get(view_name), dict) else {}
                strong_rows = view.get("strong") if isinstance(view, dict) else None
                if not isinstance(strong_rows, list):
                    continue
                for row in strong_rows:
                    if not isinstance(row, dict):
                        continue
                    tier = row.get("tier")
                    qwp = row.get("question_well_posed")
                    if qwp is not None:
                        notes_parts.append(f"question_well_posed({view_name}:{tier}): {qwp}")
                        emitted_feedback = True
                    cf_txt = row.get("correctness_feedback")
                    if isinstance(cf_txt, str) and cf_txt.strip():
                        notes_parts.append(f"correctness_feedback({view_name}:{tier}): {cf_txt.strip()}")
                        emitted_feedback = True
                    df_txt = row.get("difficulty_feedback")
                    if isinstance(df_txt, str) and df_txt.strip():
                        notes_parts.append(f"difficulty_feedback({view_name}:{tier}): {df_txt.strip()}")
                        emitted_feedback = True
                    if emitted_feedback:
                        break
                if emitted_feedback:
                    break

        # Type1 ambiguity (path-view multi-strong): keep it compact for prompts.
        try:
            a = solver_ctx.get("type1_ambiguity") if isinstance(solver_ctx, dict) else None
        except Exception:
            a = None
        if isinstance(a, dict) and a:
            suspected = a.get("type1_suspected")
            wp = a.get("wellposed_votes")
            ref = a.get("artifacts_ref")
            notes_parts.append(
                f"type1_ambiguity(path_multi_strong): suspected={suspected} wellposed_votes={wp} ref={ref}"
            )

        try:
            t2 = solver_ctx.get("type2_contract") if isinstance(solver_ctx, dict) else None
        except Exception:
            t2 = None
        if isinstance(t2, dict) and t2:
            has_errors = t2.get("has_errors")
            err_count = t2.get("error_count")
            warn_count = t2.get("warn_count")
            issue_err = t2.get("issue_types_error")
            issue_warn = t2.get("issue_types_warn")
            ref = t2.get("artifacts_ref")
            notes_parts.append(
                "type2_contract(answer_protocol): "
                f"has_errors={has_errors} error_count={err_count} warn_count={warn_count} "
                f"issue_types_error={issue_err} issue_types_warn={issue_warn} ref={ref}"
            )

    return " | ".join(notes_parts) if notes_parts else None


__all__ = [
    "normalize_question_type",
    "select_question_type",
    "is_symbolic_only",
    "is_symbolic_only_for_question_type",
    "build_director_notes",
    "idealab_session_id_for_director",
    "idealab_session_id_for_step_node",
    "with_idealab_session_id",
]
