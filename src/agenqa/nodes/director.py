"""Director 节点：根据历史与指标选择下一步操作（默认使用 py_style prompt）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from dataclasses import asdict
import json
import logging
import re
import time

from infra.llm.inference import resolve_inference
from infra.data.io import read_jsonl
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import snapshot_prompt_used, snapshot_rendered_prompt
from agenqa.graph.state import AgentState, Decision, SolverResult
from agenqa.graph.output_manager import compute_step_dir
from agenqa.domain.folded_question_schema import FIELD_QUESTION_TEXT, loads_folded_question
from agenqa.domain.known_tree import KnownTree
from agenqa.nodes.utils import (
    allowed_question_types_for_step,
    has_zam_cal_marker,
    idealab_session_id_for_director,
    with_idealab_session_id,
)
from infra.text.json_policy import clean_json_text
from agenqa.prompts.director import (
    build_director_v1_body,
    DIRECTOR_TEMPLATE,
    DIRECTOR_TEMPLATE_EN,
    DIRECTOR_TEMPLATE_EXECUTABLE,
    DIRECTOR_TEMPLATE_EXECUTABLE_EN,
)
from agenqa.prompts.director_calc import (
    DIRECTOR_TEMPLATE_CALC,
    DIRECTOR_TEMPLATE_CALC_EN,
    build_director_v1_body_calc,
)


logger = logging.getLogger(__name__)


def summarize_state(state: AgentState, agent_conf: Dict[str, Any] | None = None) -> Dict[str, Any]:
    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf, dict) else {}

    def _get_int_conf(key: str, default: int) -> int:
        try:
            val = agent_block.get(key)
            if val is None:
                return default
            return int(val)
        except Exception:
            return default

    draft_chain_window = _get_int_conf("draft_chain_window", 2)
    director_window = _get_int_conf("director_window", draft_chain_window)
    history_tail_size = _get_int_conf("director_history_tail", 5)
    path_window = _get_int_conf("director_path_window", 2)
    if path_window < 0:
        path_window = 0

    def _iter_solver_paths(step_dir: Path, filename: str):
        """Yield candidate paths for solver artifacts (prefer solve/ subdir)."""
        seen: set[Path] = set()
        for path in (step_dir / "solve" / filename, step_dir / filename):
            if path in seen:
                continue
            seen.add(path)
            yield path

    def _iter_step_dirs_for_step(step: int, artifacts_dir: Path):
        """Yield all candidate step dirs for a given step (new + legacy layout)."""
        # 新布局：artifacts_dir/round_*/step_{step}_*/（round 1）
        try:
            for round_dir in sorted(artifacts_dir.glob("round_*")):
                if not round_dir.is_dir():
                    continue
                # 查找 step_{step}_* 格式的目录（round 1）
                for step_dir in sorted(round_dir.glob(f"step_{step}_*")):
                    if step_dir.is_dir():
                        yield step_dir
                # 查找简化格式的目录（round >= 2）：从 edge_kqa.jsonl 读取 step 匹配
                exclude_names = {"00_Prompts_Snapshot", "00_Summary", "01_qa_init"}
                for item in sorted(round_dir.iterdir()):
                    if item.is_dir() and item.name not in exclude_names and not item.name.startswith("step_"):
                        kqa_path = item / "edge_kqa.jsonl"
                        if not kqa_path.exists():
                            continue
                        try:
                            kqa_rows = read_jsonl(kqa_path)
                            if kqa_rows and len(kqa_rows) > 0:
                                step_val = kqa_rows[0].get("step")
                                if step_val is not None and int(step_val) == step:
                                    yield item
                        except Exception:
                            pass
        except Exception:
            pass
        # 兼容旧布局：artifacts_dir/step_{step}_*
        try:
            for step_dir in sorted(artifacts_dir.glob(f"step_{step}_*")):
                if step_dir.is_dir():
                    yield step_dir
        except Exception:
            pass

    def _read_solver_result(step: int, artifacts_dir: Path, name: str) -> Dict[str, Any] | None:
        """读取指定 step 的求解结果（name 支持 solve_medium / solve_strong / solve_path_*）。

        返回最新 round 的结果（如果有多个匹配的结果）。
        """
        candidates = []
        try:
            for step_dir in _iter_step_dirs_for_step(step, artifacts_dir):
                # 从 step_dir 路径中提取 round 号（用于排序，选择最新的）
                round_num = None
                try:
                    # 尝试从路径中提取 round_* 部分
                    parts = step_dir.parts
                    for i, part in enumerate(parts):
                        if part.startswith("round_") and part[6:].isdigit():
                            round_num = int(part[6:])
                            break
                except Exception:
                    pass

                for path in _iter_solver_paths(step_dir, f"{name}.jsonl"):
                    if not path.exists():
                        continue
                    for row in read_jsonl(path):
                        try:
                            s_val = int(row.get("step"))
                        except Exception:
                            s_val = None
                        if s_val is not None and s_val != step:
                            continue
                        err = row.get("error")
                        if isinstance(err, str) and len(err) > 400:
                            err = err[:400] + "…"
                        extra: Dict[str, Any] = {}
                        if row.get("execution_time") is not None:
                            extra["execution_time"] = row.get("execution_time")
                        if row.get("eval_backend") is not None:
                            extra["eval_backend"] = row.get("eval_backend")
                        if row.get("code_source") is not None:
                            extra["code_source"] = row.get("code_source")
                        if isinstance(err, str) and err.strip():
                            extra["error"] = err
                        candidates.append((round_num or 0, {
                            "correct": row.get("correct"),
                            "token_ratio": row.get("token_ratio") or row.get("difficulty"),  # 向后兼容
                            "model": row.get("model"),
                            "service_id": row.get("service_id"),
                            "kq_tokens": (row.get("metrics") or {}).get("kq_tokens") if isinstance(row.get("metrics"), dict) else None,
                            "completion_tokens": (row.get("metrics") or {}).get("completion_tokens") if isinstance(row.get("metrics"), dict) else None,
                            # Track-specific extension payload (optional).
                            "extra": extra,
                        }))
                        break  # 每个文件只读取第一行
                    if candidates and candidates[-1][0] == round_num:
                        break  # 已找到该 step_dir 的结果
        except Exception:
            pass

        if not candidates:
            return None

        # 按 round_num 降序排序，返回最新的结果
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _trim_text(val: Any, *, max_chars: int) -> str | None:
        if not isinstance(val, str):
            return None
        txt = val.strip()
        if not txt:
            return None
        if len(txt) <= max_chars:
            return txt
        return txt[:max_chars] + " ...[truncated]"

    def _folded_question_preview(val: Any, *, max_chars: int) -> Any:
        """Return an LLM-friendly preview derived from canonical folded_question string."""
        if not (isinstance(val, str) and val.strip()):
            return None
        try:
            fq = loads_folded_question(val)
        except Exception:
            return None
        if fq.track in {"unified", "semantic"}:
            q = fq.payload.get(FIELD_QUESTION_TEXT)
            return _trim_text(q, max_chars=max_chars) if isinstance(q, str) else None
        if fq.track == "executable":
            payload = fq.payload if isinstance(fq.payload, dict) else {}
            sub_steps = payload.get("sub_steps")
            sub_preview: list[dict[str, Any]] = []
            if isinstance(sub_steps, list):
                for item in sub_steps[:6]:
                    if not isinstance(item, dict):
                        continue
                    sub_preview.append(
                        {
                            "step_number": item.get("step_number"),
                            "function_header": _trim_text(item.get("function_header"), max_chars=240),
                            "return_line": _trim_text(item.get("return_line"), max_chars=120),
                            "step_description": _trim_text(item.get("step_description"), max_chars=360),
                        }
                    )
            tests_map = payload.get("test_cases")
            tests_count: dict[str, int] = {}
            if isinstance(tests_map, dict):
                for k, v in tests_map.items():
                    if isinstance(k, str) and isinstance(v, list):
                        tests_count[k] = len(v)
            out: dict[str, Any] = {
                "sub_steps": sub_preview,
            }
            if tests_count:
                out["test_cases_count_by_step"] = tests_count
            return out
        return None

    tail = state.history[-max(1, history_tail_size) :] if state.history else []
    hist = [
        {
            "record_type": getattr(r, "record_type", None),
            "step": getattr(r, "step", getattr(r, "qa_idx", None)),
            "question": getattr(r, "question", ""),
            "answer": (getattr(r, "answer", "") or ""),
            "chain": getattr(r, "chain", None),
            "paper_id": getattr(r, "paper_id", None),
            # Path fold artifacts (optional; useful for diagnosing path failures).
            "path_question_direct": _folded_question_preview(getattr(r, "path_question_direct", None), max_chars=1600),
            "path_question_scaffolded": _folded_question_preview(getattr(r, "path_question_scaffolded", None), max_chars=1600),
            "path_fold_notes": _trim_text(getattr(r, "path_fold_notes", None), max_chars=800),
        }
        for r in tail
    ]

    try:
        step_idx = int(state.step or 0)
    except Exception:
        step_idx = 0

    # ---- Memory views for Director (semantic-rich; provenance stripped) ----
    mem = KnownTree.normalize_memory(getattr(state, "memory", None))
    chain_view = KnownTree.build_draft_chain_view(mem, step_idx, window=path_window)

    premise_bank = []
    for premise in KnownTree._filter_entries(chain_view.get("premise_bank", [])):
        pid = premise.get("id")
        if not isinstance(pid, str) or not pid.strip():
            continue
        premise_bank.append(KnownTree._strip_provenance(premise))

    fact_window = [KnownTree._strip_provenance(f) for f in KnownTree._filter_entries(chain_view.get("fact_window", []))]
    cert_window = [
        KnownTree._strip_provenance(c) for c in KnownTree._filter_entries(chain_view.get("step_certs_window", []))
    ]

    # NOTE:
    # - `path_start_memory` is the solver-visible path start view (P) for Path-Fold: premise_bank only (no episode_seed).
    # - `director_memory_window` is a director-only convenience view (includes episode_seed + a small window of recent facts/certs).
    path_start_memory = {
        "schema_version": chain_view.get("schema_version", mem.get("schema_version", KnownTree.SCHEMA_VERSION)),
        "premise_bank": premise_bank,
        "fact_bank": [],
        "step_certs": [],
    }
    director_memory_window = {
        "schema_version": chain_view.get("schema_version", mem.get("schema_version", KnownTree.SCHEMA_VERSION)),
        "episode_seed": chain_view.get("episode_seed", mem.get("episode_seed", {})),
        "premise_bank": premise_bank,
        "fact_window": fact_window,
        "step_certs_window": cert_window,
        "window": path_window,
    }

    def _find_fact(fid: str) -> Dict[str, Any] | None:
        if not isinstance(fid, str) or not fid.strip():
            return None
        for fact in KnownTree._filter_entries(mem.get("fact_bank", [])):
            if fact.get("id") == fid:
                return KnownTree._strip_provenance(fact)
        return None

    key_facts_tail: list[dict[str, Any]] = []
    if director_window < 0:
        director_window = 0
    for s in range(max(1, step_idx - director_window + 1), step_idx + 1):
        kid = KnownTree.key_fact_id_for_step(mem, s)
        if not kid:
            continue
        fact = _find_fact(kid) or {"id": kid}
        text = fact.get("statement")
        if text is None or (isinstance(text, str) and not text.strip()):
            text = fact.get("text")
        key_facts_tail.append(
            {
                "step": s,
                "key_fact_id": kid,
                "key_fact_text": "" if text is None else str(text),
                "key_fact": fact,
            }
        )

    expected_primary_fact_id = KnownTree.key_fact_id_for_step(mem, step_idx) if step_idx >= 1 else ""

    path_fold_result: Dict[str, Any] = {}
    try:
        if state.history:
            last = state.history[-1]
            q_direct = _folded_question_preview(getattr(last, "path_question_direct", None), max_chars=2200)
            q_scaf = _folded_question_preview(getattr(last, "path_question_scaffolded", None), max_chars=2200)
            notes = _trim_text(getattr(last, "path_fold_notes", None), max_chars=1200)
            if q_direct or q_scaf or notes:
                path_fold_result = {
                    "step_tail": int(getattr(last, "qa_idx", getattr(last, "step", 0)) or 0),
                    "variant_eval": "direct",
                    "question_direct": q_direct,
                    "question_scaffolded": q_scaf,
                    "fold_notes": notes,
                }
    except Exception:
        path_fold_result = {}

    def _extract_round_num(step_dir: Path) -> int:
        try:
            for part in step_dir.parts:
                if part.startswith("round_") and part[6:].isdigit():
                    return int(part[6:])
        except Exception:
            pass
        return 0

    def _safe_read_json(path: Path) -> Dict[str, Any] | None:
        try:
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _truncate_str(val: Any, max_chars: int) -> str:
        if val is None:
            return ""
        s = str(val)
        if len(s) <= max_chars:
            return s
        return s[: max(0, max_chars - 1)] + "…"

    def _build_task_graph_tail(artifacts_dir: Path) -> list[dict[str, Any]]:
        if task_graph_tail_window <= 0 or step_idx < 1:
            return []
        preview_limit = 6
        out: list[dict[str, Any]] = []
        for s in range(max(1, step_idx - task_graph_tail_window + 1), step_idx + 1):
            best_step_dir: Path | None = None
            best_key: tuple[int, str] = (-1, "")
            for step_dir in _iter_step_dirs_for_step(s, artifacts_dir):
                key = (_extract_round_num(step_dir), str(step_dir))
                if key > best_key:
                    best_key = key
                    best_step_dir = step_dir
            if best_step_dir is None:
                continue

            rel_step_dir = None
            try:
                rel_step_dir = str(best_step_dir.relative_to(artifacts_dir))
            except Exception:
                rel_step_dir = str(best_step_dir)

            edge: Dict[str, Any] = {
                "step": s,
                "from_step": s - 1,
                "expected_primary_fact_id": KnownTree.key_fact_id_for_step(mem, s) or "",
                "artifacts_ref": rel_step_dir,
            }

            chain_path = best_step_dir / "subruns_raw" / "draft_chain" / "parsed_output.json"
            chain_data = _safe_read_json(chain_path)
            if chain_data is not None:
                subtasks = chain_data.get("subtasks")
                subtasks_len = len(subtasks) if isinstance(subtasks, list) else 0
                subtasks_out = []
                if isinstance(subtasks, list):
                    for item in subtasks:
                        if not isinstance(item, dict):
                            continue
                        sid = item.get("id")
                        desc = item.get("description")
                        subtasks_out.append(
                            {
                                "id": "" if sid is None else str(sid),
                                "description": _truncate_str(desc, 220),
                            }
                        )
                deps = chain_data.get("dependencies")
                deps_keys = list(deps.keys()) if isinstance(deps, dict) else []
                edge["draft_chain"] = {
                    "primary_required_fact_id": chain_data.get("primary_required_fact_id"),
                    "required_fact_ids": chain_data.get("required_fact_ids"),
                    "subtasks_len": subtasks_len,
                    "subtasks": subtasks_out[:preview_limit],
                    "final_subtask_id": chain_data.get("final_subtask_id"),
                    "dependencies_keys": deps_keys[:preview_limit],
                }

            out.append(edge)
        return out

    last_decision_payload: Dict[str, Any] | None = None
    try:
        dec = getattr(state, "last_decision", None)
        if dec is not None and isinstance(getattr(dec, "operation", None), str):
            params = getattr(dec, "params", None)
            params_out: Dict[str, Any] = {}
            if isinstance(params, dict):
                for k in ("question_type", "revise_mode", "operator_notes"):
                    if k in params:
                        params_out[k] = params.get(k)
            last_decision_payload = {
                "operation": getattr(dec, "operation", ""),
                "reason": getattr(dec, "reason", ""),
                "params": params_out,
            }
    except Exception:
        last_decision_payload = None
    # 汇总本步 solver（edge/path × medium/strong）的结果概览
    # 优先从 state.solver_index 读取（单一事实源），如果没有则回退到磁盘扫描
    step_idx = int(state.step or 0)

    def _get_solver_result_from_index(step: int, target: str, tier: str) -> Dict[str, Any] | None:
        """从 state.solver_index 读取最新 round 的 solver 结果（使用 AgentState 的工具方法）。"""
        result = state.get_latest_solver(step, target, tier)
        if result is None:
            return None
        return {
            "correct": result.correct,
            "token_ratio": result.token_ratio,
            "model": result.model,
            "service_id": result.service_id,
            "kq_tokens": result.kq_tokens,
            "completion_tokens": result.completion_tokens,
            "extra": dict(getattr(result, "extra", None) or {}),
        }

    def _get_latest_solver_results_by_prefix(step: int, target: str, prefix: str) -> list[Dict[str, Any]]:
        """Return all latest-round solver results whose tier matches `prefix` / `prefix_*`.

        This is primarily used to expose multi-strong structured facts to Director
        without forcing an aggregate interpretation in the schema itself.
        """
        out: list[Dict[str, Any]] = []
        try:
            rounds_dict = state.solver_index.get(step) if isinstance(state.solver_index, dict) else None
            if not isinstance(rounds_dict, dict) or not rounds_dict:
                return out
            max_round = max(rounds_dict.keys())
            round_dict = rounds_dict.get(max_round) or {}
            target_dict = round_dict.get(target) if isinstance(round_dict, dict) else None
            if not isinstance(target_dict, dict):
                return out

            by_idx: dict[int, Dict[str, Any]] = {}

            def _solver_idx_for_tier(tier_name: str) -> int | None:
                tt = str(tier_name or "").strip()
                if tt == prefix:
                    return 0
                if tt.startswith(f"{prefix}_"):
                    try:
                        return int(tt.split("_", 1)[1])
                    except Exception:
                        return None
                return None

            for tier_name, result in target_dict.items():
                idx = _solver_idx_for_tier(str(tier_name))
                if idx is None or not isinstance(result, SolverResult):
                    continue
                row = {
                    "solver_idx": idx,
                    "tier": str(tier_name),
                    "correct": result.correct,
                    "token_ratio": result.token_ratio,
                    "model": result.model,
                    "service_id": result.service_id,
                    "kq_tokens": result.kq_tokens,
                    "completion_tokens": result.completion_tokens,
                    "question_well_posed": result.question_well_posed,
                    "correctness_feedback": result.correctness_feedback,
                    "difficulty_feedback": result.difficulty_feedback,
                    "extra": dict(getattr(result, "extra", None) or {}),
                }
                prev = by_idx.get(idx)
                if prev is None or str(tier_name) == prefix:
                    by_idx[idx] = row

            out = [by_idx[i] for i in sorted(by_idx.keys())]
        except Exception:
            return []
        return out

    # 优先从 index 读取，如果没有则回退到磁盘扫描
    solver_metrics: Dict[str, Any] = {
        "edge": {
            "medium": _get_solver_result_from_index(step_idx, "edge", "medium")
            or _read_solver_result(step_idx, Path(state.artifacts_dir), "solve_medium"),
            "strong": _get_latest_solver_results_by_prefix(step_idx, "edge", "strong"),
        },
        "path": {
            "medium": _get_solver_result_from_index(step_idx, "path", "medium")
            or _read_solver_result(step_idx, Path(state.artifacts_dir), "solve_path_medium"),
            "strong": _get_latest_solver_results_by_prefix(step_idx, "path", "strong"),
        },
    }

    def _safe_float(val: Any) -> float | None:
        try:
            if isinstance(val, bool):
                return None
            if not isinstance(val, (int, float)):
                return None
            out = float(val)
            if out != out:  # NaN
                return None
            return out
        except Exception:
            return None

    def _get_solver_result_any(step: int, target: str, tier: str) -> Dict[str, Any] | None:
        """从 index 优先取；缺失时回退扫盘（仅取第一行）。target: edge/path."""
        target_norm = str(target).strip().lower()
        tier_norm = str(tier).strip().lower()
        try:
            res = _get_solver_result_from_index(step, target_norm, tier_norm)
            if res:
                return res
        except Exception:
            pass

        name_map = {
            ("edge", "medium"): "solve_medium",
            ("edge", "strong"): "solve_strong",
            ("path", "medium"): "solve_path_medium",
            ("path", "strong"): "solve_path_strong",
        }
        name = name_map.get((target_norm, tier_norm))
        if not name:
            return None
        try:
            return _read_solver_result(step, Path(state.artifacts_dir), name)
        except Exception:
            return None

    def _step_ratio(step: int, target: str, tier: str, field: str) -> float:
        """step(t) / step(t-1) 的趋势比值；缺失时返回 1（弱参考，不作为硬阈值）。"""
        try:
            if step <= 0:
                return 1.0
            cur = _get_solver_result_any(step, target, tier) or {}
            prev = _get_solver_result_any(step - 1, target, tier) or {}
            cur_v = _safe_float(cur.get(field))
            prev_v = _safe_float(prev.get(field))
            if cur_v is None or prev_v is None or prev_v <= 0:
                return 1.0
            return float(cur_v) / float(prev_v)
        except Exception:
            return 1.0

    def _build_solver_feedback_compact() -> Dict[str, Any] | None:
        """Build a stable feedback summary for Director/Diagnose from multi-strong facts.

        Policy:
        - unanimous (all explicit correct or all explicit incorrect): keep only primary feedback;
        - mixed correctness: provide one correct + one incorrect contrast pair.
        """

        def _is_non_empty_text(v: Any) -> bool:
            return isinstance(v, str) and bool(v.strip())

        def _pick_primary(rows: list[Dict[str, Any]]) -> Dict[str, Any] | None:
            for row in rows:
                if int(row.get("solver_idx", -1)) == 0:
                    return row
            return rows[0] if rows else None

        def _pick_row(rows: list[Dict[str, Any]], want_correct: bool) -> Dict[str, Any] | None:
            for row in rows:
                if row.get("correct") is want_correct:
                    return row
            return None

        for view in ("path", "edge"):
            rows_any = ((solver_metrics.get(view) or {}).get("strong")) if isinstance(solver_metrics, dict) else None
            if not isinstance(rows_any, list):
                continue
            rows = [r for r in rows_any if isinstance(r, dict)]
            explicit = [r for r in rows if r.get("correct") in {True, False}]
            if not explicit:
                continue

            all_true = all(r.get("correct") is True for r in explicit)
            all_false = all(r.get("correct") is False for r in explicit)

            if all_true or all_false:
                primary = _pick_primary(explicit)
                if not isinstance(primary, dict):
                    continue
                return {
                    "mode": "unanimous",
                    "view": view,
                    "from_tier": str(primary.get("tier") or "strong"),
                    "from_step": step_idx,
                    "question_well_posed": primary.get("question_well_posed"),
                    "correctness_feedback": primary.get("correctness_feedback") if _is_non_empty_text(primary.get("correctness_feedback")) else None,
                    "difficulty_feedback": primary.get("difficulty_feedback") if _is_non_empty_text(primary.get("difficulty_feedback")) else None,
                    "selected": [
                        {
                            "solver_idx": primary.get("solver_idx"),
                            "correct": primary.get("correct"),
                            "service_id": primary.get("service_id"),
                            "model": primary.get("model"),
                            "token_ratio": primary.get("token_ratio"),
                        }
                    ],
                }

            ok_row = _pick_row(explicit, True)
            bad_row = _pick_row(explicit, False)
            if ok_row is None or bad_row is None:
                continue
            ok_fb = ok_row.get("correctness_feedback")
            bad_fb = bad_row.get("correctness_feedback")
            contrast_parts: list[str] = []
            if _is_non_empty_text(ok_fb):
                contrast_parts.append(f"[correct:{ok_row.get('service_id') or ok_row.get('model')}] {str(ok_fb).strip()}")
            if _is_non_empty_text(bad_fb):
                contrast_parts.append(f"[incorrect:{bad_row.get('service_id') or bad_row.get('model')}] {str(bad_fb).strip()}")
            return {
                "mode": "contrast",
                "view": view,
                "from_tier": "strong_multi",
                "from_step": step_idx,
                "question_well_posed": ok_row.get("question_well_posed"),
                "correctness_feedback": " | ".join(contrast_parts) if contrast_parts else None,
                "difficulty_feedback": ok_row.get("difficulty_feedback") if _is_non_empty_text(ok_row.get("difficulty_feedback")) else None,
                "selected": [
                    {
                        "solver_idx": ok_row.get("solver_idx"),
                        "correct": ok_row.get("correct"),
                        "service_id": ok_row.get("service_id"),
                        "model": ok_row.get("model"),
                        "token_ratio": ok_row.get("token_ratio"),
                    },
                    {
                        "solver_idx": bad_row.get("solver_idx"),
                        "correct": bad_row.get("correct"),
                        "service_id": bad_row.get("service_id"),
                        "model": bad_row.get("model"),
                        "token_ratio": bad_row.get("token_ratio"),
                    },
                ],
            }
        return None

    def _build_strong_summary(view: str) -> Dict[str, Any]:
        rows_any = ((solver_metrics.get(view) or {}).get("strong")) if isinstance(solver_metrics, dict) else None
        rows = [r for r in rows_any if isinstance(r, dict)] if isinstance(rows_any, list) else []
        explicit = [r for r in rows if r.get("correct") in {True, False}]
        correct_count = sum(1 for r in explicit if r.get("correct") is True)
        incorrect_count = sum(1 for r in explicit if r.get("correct") is False)
        total = len(rows)
        explicit_total = len(explicit)
        unknown_count = max(0, total - explicit_total)
        return {
            "total": total,
            "explicit_total": explicit_total,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "unknown_count": unknown_count,
            "all_correct": bool(explicit_total > 0 and incorrect_count == 0),
            "any_incorrect": bool(incorrect_count > 0),
        }

    # metrics 只暴露分组视图：edge / path，避免重复字段
    metrics: Dict[str, Any] = {}
    metrics["edge"] = {
        "correct_medium": state.metrics.correct_medium,
        "token_ratio_medium": state.metrics.token_ratio_medium,
        "strong_summary": _build_strong_summary("edge"),
        # 趋势信号（step(t) / step(t-1)）：缺失时=1（不作为硬阈值，仅作参考）
        "kq_tokens_step_ratio_medium": _step_ratio(step_idx, "edge", "medium", "kq_tokens"),
        "completion_tokens_step_ratio_medium": _step_ratio(step_idx, "edge", "medium", "completion_tokens"),
        "kq_tokens_step_ratio_strong": _step_ratio(step_idx, "edge", "strong", "kq_tokens"),
        "completion_tokens_step_ratio_strong": _step_ratio(step_idx, "edge", "strong", "completion_tokens"),
    }
    ht_med = (solver_metrics.get("path") or {}).get("medium")
    metrics["path"] = {
        "correct_medium": ht_med.get("correct") if isinstance(ht_med, dict) else None,
        "token_ratio_medium": (
            ht_med.get("token_ratio") or ht_med.get("difficulty") if isinstance(ht_med, dict) else None
        ),  # 向后兼容
        "strong_summary": _build_strong_summary("path"),
        # 趋势信号（step(t) / step(t-1)）：缺失时=1（不作为硬阈值，仅作参考）
        "kq_tokens_step_ratio_medium": _step_ratio(step_idx, "path", "medium", "kq_tokens"),
        "completion_tokens_step_ratio_medium": _step_ratio(step_idx, "path", "medium", "completion_tokens"),
        "kq_tokens_step_ratio_strong": _step_ratio(step_idx, "path", "strong", "kq_tokens"),
        "completion_tokens_step_ratio_strong": _step_ratio(step_idx, "path", "strong", "completion_tokens"),
    }

    solver_consensus_payload: Dict[str, Any] | None = None
    try:
        solver_consensus = getattr(state, "solver_consensus", None)
        if solver_consensus is not None:
            solver_consensus_payload = asdict(solver_consensus)
    except Exception:
        solver_consensus_payload = None

    # Type1 ambiguity summary (path-view multi-strong), written by the consensus evaluator.
    type1_ambiguity_payload: Dict[str, Any] | None = None
    try:
        artifacts_dir = Path(state.artifacts_dir)
        candidates: list[tuple[int, Dict[str, Any], Path]] = []
        for step_dir in _iter_step_dirs_for_step(step_idx, artifacts_dir):
            r = _extract_round_num(step_dir)
            for p in _iter_solver_paths(step_dir, "ambiguity_report.json"):
                data = _safe_read_json(p)
                if data is None:
                    continue
                candidates.append((r, data, p))
                break
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _r, data, p = candidates[0]
            try:
                ref = p.relative_to(state.artifacts_dir).as_posix()
            except Exception:
                ref = str(p)
            type1_ambiguity_payload = {
                "type1_suspected": data.get("type1_suspected"),
                "wellposed_votes": data.get("wellposed_votes"),
                "solvers": data.get("solvers"),
                "evidence_files": data.get("evidence_files"),
                "artifacts_ref": ref,
            }
    except Exception:
        type1_ambiguity_payload = None

    # Type2 answer-contract summary (step-scoped), written by extend/revise format nodes.
    type2_contract_payload: Dict[str, Any] | None = None
    try:
        artifacts_dir = Path(state.artifacts_dir)
        candidates2: list[tuple[int, Dict[str, Any], Path]] = []
        for step_dir in _iter_step_dirs_for_step(step_idx, artifacts_dir):
            r = _extract_round_num(step_dir)
            p = step_dir / "answer_contract_report.json"
            data = _safe_read_json(p)
            if data is None:
                continue
            candidates2.append((r, data, p))
        if candidates2:
            candidates2.sort(key=lambda x: x[0], reverse=True)
            _r, data2, p2 = candidates2[0]
            try:
                ref2 = p2.relative_to(state.artifacts_dir).as_posix()
            except Exception:
                ref2 = str(p2)
            err_count = data2.get("error_count")
            try:
                has_errors = int(err_count) > 0
            except Exception:
                has_errors = False
            type2_contract_payload = {
                "has_errors": has_errors,
                "error_count": data2.get("error_count"),
                "warn_count": data2.get("warn_count"),
                "issue_types_error": data2.get("issue_types_error"),
                "issue_types_warn": data2.get("issue_types_warn"),
                "answer_contract_ids": data2.get("answer_contract_ids"),
                "artifacts_ref": ref2,
            }
    except Exception:
        type2_contract_payload = None

    return {
        "step": state.step,
        "next_step": (step_idx + 1) if step_idx >= 0 else None,
        "max_steps": state.max_steps,
        "last_decision": last_decision_payload,
        "history_tail": hist,
        "path_fold_result": path_fold_result,
        "path_start_memory": path_start_memory,
        "director_memory_window": director_memory_window,
        "key_facts_tail": key_facts_tail,
        "expected_primary_fact_id": expected_primary_fact_id,
        # metrics 主视图：edge / path。
        "metrics": metrics,
        # 本步的 solver 概览（edge/path × medium/strong）
        "solver_metrics": solver_metrics,
        # Multi-strong derived compact feedback summary for Director/Diagnose.
        "solver_feedback": _build_solver_feedback_compact(),
        # Multi-strong 共识信号（如启用）
        "solver_consensus": solver_consensus_payload,
        # Type1 歧义（语义世界观不唯一）诊断摘要（path multi-strong）
        "type1_ambiguity": type1_ambiguity_payload,
        # Type2 作答协议/判对口径诊断摘要（answer_contract_report）
        "type2_contract": type2_contract_payload,
    }


def build_director_view(state: AgentState, agent_conf: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a unified Director view that supports unified/semantic and executable tracks.

    Design principles:
    - Reuse `summarize_state` for common solver/metrics signals (single truth source).
    - Provide a stable core envelope + track-specific context.
    """

    def _get_track() -> str:
        block = (agent_conf.get("agent") or {}) if isinstance(agent_conf, dict) else {}
        track = str(block.get("track") or "").strip().lower()
        if not track:
            return "unified"
        if track in {"unified", "semantic", "executable"}:
            return track
        raise ValueError(f"unsupported track={track!r} (expected 'unified'|'semantic'|'executable')")

    def _has_episode_seed() -> bool:
        memory = KnownTree.normalize_memory(getattr(state, "memory", None))
        seed = memory.get("episode_seed") or {}
        if not isinstance(seed, dict):
            return False
        anchor = seed.get("anchor")
        if isinstance(anchor, str) and anchor.strip():
            return True
        subject = seed.get("subject")
        if isinstance(subject, str) and subject.strip():
            return True
        keywords = seed.get("keywords")
        if isinstance(keywords, list) and any(str(k).strip() for k in keywords):
            return True
        return False

    def _has_executable_seed() -> bool:
        memory = KnownTree.normalize_memory(getattr(state, "memory", None))
        seed = memory.get("executable_seed")
        if not isinstance(seed, dict):
            return False
        pid = seed.get("problem_id")
        return isinstance(pid, str) and bool(pid.strip())

    def _compute_allowed_ops(track: str) -> list[str]:
        agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf, dict) else {}
        allowed_ops_conf = agent_block.get("allowed_operations")

        if track == "executable":
            has_seed = _has_executable_seed()
            has_hist = state.latest_executable_record() is not None
            if (not has_hist) and (not has_seed):
                allowed_ops = ["Init"]
            elif (not has_hist) and has_seed:
                allowed_ops = ["Extend"]
            else:
                if isinstance(allowed_ops_conf, (list, tuple)):
                    allowed_ops = [str(op) for op in allowed_ops_conf if str(op).strip()]
                else:
                    allowed_ops = ["Extend", "Revise", "Finish"]
                if not allowed_ops:
                    allowed_ops = ["Extend", "Finish"]
            return allowed_ops

        # semantic
        has_seed = _has_episode_seed()
        if not state.history and not has_seed:
            allowed_ops = ["Init"]
        elif not state.history and has_seed:
            allowed_ops = ["Extend"]
        else:
            if isinstance(allowed_ops_conf, (list, tuple)):
                allowed_ops = [str(op) for op in allowed_ops_conf if str(op).strip()]
            else:
                allowed_ops = ["Extend", "Revise", "Finish"]
            if not allowed_ops:
                allowed_ops = ["Extend", "Finish"]
        return allowed_ops

    def _build_executable_track_context(base: Dict[str, Any]) -> Dict[str, Any]:
        from agenqa.domain.executable_schema import ExecutableRecord, ExecutableTestCase

        mem = KnownTree.normalize_memory(getattr(state, "memory", None))
        seed = mem.get("executable_seed") if isinstance(mem, dict) else None
        seed_view: Dict[str, Any] = {}
        if isinstance(seed, dict):
            seed_view = {
                "source": seed.get("source"),
                "split": seed.get("split"),
                "problem_id": seed.get("problem_id"),
                "with_background": seed.get("with_background"),
                "hf_use_proxy": seed.get("hf_use_proxy"),
                "scicode_root": seed.get("scicode_root"),
            }

        def _executable_rec_view(rec: ExecutableRecord) -> Dict[str, Any]:
            tail = rec.sub_steps[-1] if rec.sub_steps else None
            tail_id = tail.step_number if tail else ""
            tests_tail: list[Dict[str, Any]] = []
            raw_tests = (rec.test_cases or {}).get(tail_id) or []
            for t in raw_tests:
                if isinstance(t, ExecutableTestCase):
                    tests_tail.append(t.to_dict())
                elif isinstance(t, dict):
                    tests_tail.append(dict(t))
            return {
                "record_type": rec.record_type,
                "step": rec.step,
                "problem_id": rec.problem_id,
                "background": rec.background,
                "required_dependencies": rec.required_dependencies,
                "sub_steps": [s.to_dict() for s in rec.sub_steps],
                "tail_step_number": tail_id,
                "test_cases_tail": tests_tail,
                "estimated_difficulty": rec.estimated_difficulty,
                "subject": rec.subject,
                "source": rec.source,
            }

        latest = state.latest_executable_record()
        executable_tail = _executable_rec_view(latest) if isinstance(latest, ExecutableRecord) else {}

        # Executable step certs (memory): stable, no-golden summaries for long-term decisions.
        step_certs = mem.get("step_certs") if isinstance(mem, dict) else None
        executable_step_certs: list[Dict[str, Any]] = []
        if isinstance(step_certs, list):
            for item in step_certs:
                if not isinstance(item, dict):
                    continue
                # Only keep executable-track certificates (chain + eval). Avoid legacy kind drift.
                if item.get("kind") not in {"executable_chain_cert", "executable_eval_cert"}:
                    continue
                executable_step_certs.append(item)
        # Keep only the tail window.
        executable_step_certs_tail = executable_step_certs[-3:] if len(executable_step_certs) > 3 else executable_step_certs

        history_tail_size = 5
        try:
            agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf, dict) else {}
            history_tail_size = int(agent_block.get("director_history_tail", history_tail_size))
        except Exception:
            history_tail_size = 5
        # Collect last N executable records (without golden code).
        executable_hist: list[Dict[str, Any]] = []
        if history_tail_size > 0:
            picked: list[ExecutableRecord] = []
            for r in reversed(getattr(state, "history", None) or []):
                if isinstance(r, ExecutableRecord):
                    picked.append(r)
                    if len(picked) >= history_tail_size:
                        break
            for rec in reversed(picked):
                executable_hist.append(_executable_rec_view(rec))

        return {
            "executable_seed": seed_view,
            "executable_tail": executable_tail,
            "executable_history_tail": executable_hist,
            "executable_step_certs_tail": executable_step_certs_tail,
            # Reuse the unified fold container if present (ExecutableRecord stores path_question_*).
            "path_fold_result": base.get("path_fold_result") or {},
        }

    base = summarize_state(state, agent_conf or {})
    track = _get_track()
    allowed_ops = _normalize_allowed_ops(_compute_allowed_ops(track))

    progress = {
        "step": base.get("step"),
        "next_step": base.get("next_step"),
        "max_steps": base.get("max_steps"),
    }

    if track == "executable":
        track_context = _build_executable_track_context(base)
    else:
        track_context = {
            "path_start_memory": base.get("path_start_memory"),
            "director_memory_window": base.get("director_memory_window"),
            "path_fold_result": base.get("path_fold_result"),
            "key_facts_tail": base.get("key_facts_tail"),
            "expected_primary_fact_id": base.get("expected_primary_fact_id"),
            "history_tail": base.get("history_tail"),
        }

    return {
        "track": track,
        "progress": progress,
        "last_decision": base.get("last_decision"),
        "metrics": base.get("metrics"),
        "solver_metrics": base.get("solver_metrics"),
        "solver_consensus": base.get("solver_consensus"),
        "type1_ambiguity": base.get("type1_ambiguity"),
        "type2_contract": base.get("type2_contract"),
        "track_context": track_context,
        "available_operations": allowed_ops,
        # Optional policy hints for Director (enforced in code).
        "available_question_types": _allowed_question_types_for_next_step(agent_conf or {}, state),
        "question_type_policy": _question_type_policy_view(agent_conf or {}, state),
    }


def _no_mcq_from_step(agent_conf: Dict[str, Any]) -> int | None:
    """Return the step index (1-based QA step) from which MCQ is disallowed.

    Semantics:
    - If next_step >= no_mcq_from_step: allowed types become {"Derivation", "Numeric"}.
    - None disables the policy.

    Default: 2 (i.e., from step 2 onward, only Derivation/Numeric).
    """
    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
    policy = agent_block.get("question_type_policy")
    policy = policy if isinstance(policy, dict) else {}
    # Distinguish explicit null from missing key:
    # - missing key -> default 2
    # - key present with null -> disable
    if "no_mcq_from_step" not in policy:
        return 2
    raw = policy.get("no_mcq_from_step")
    if raw is None:
        return None
    if raw is False:
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


def _allowed_question_types_for_next_step(agent_conf: Dict[str, Any], state: AgentState) -> list[str]:
    try:
        curr_step = int(state.step or 0)
    except Exception:
        curr_step = 0
    next_step = curr_step + 1
    return allowed_question_types_for_step(agent_conf, next_step)


def _question_type_policy_view(agent_conf: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
    track = str(agent_block.get("track") or "").strip().lower() or "unified"
    try:
        curr_step = int(state.step or 0)
    except Exception:
        curr_step = 0
    next_step = curr_step + 1
    threshold = _no_mcq_from_step(agent_conf)
    # Compute base/effective views for debugging.
    if track == "executable":
        base_allowed: list[str] = []
        effective_allowed: list[str] = []
        policy_allowed_view: list[str] | None = None
    else:
        if threshold is not None and next_step >= threshold:
            base_allowed = ["Derivation", "Numeric"]
        else:
            base_allowed = ["MCQ", "Derivation", "Numeric"]
        # Reuse the effective computation (may raise on misconfig).
        effective_allowed = _allowed_question_types_for_next_step(agent_conf, state)
        # Best-effort parse of policy list for display (keep None if unset/disabled).
        policy = agent_block.get("question_type_policy")
        policy = policy if isinstance(policy, dict) else {}
        raw_allowed = policy.get("allowed_question_types")
        if raw_allowed is None or raw_allowed is False:
            policy_allowed_view = None
        elif isinstance(raw_allowed, str) and raw_allowed.strip().lower() in {"off", "disable", "disabled", "none", "null"}:
            policy_allowed_view = None
        else:
            try:
                # Let _allowed_question_types_for_next_step do the heavy lifting; reconstruct by filtering base+synonyms.
                # We expose policy ordering if present; otherwise None.
                # NOTE: This is display-only and should not mutate behavior.
                if isinstance(raw_allowed, str):
                    parts = [p for p in re.split(r"[,\s]+", raw_allowed.strip()) if p]
                else:
                    parts = []
                    for x in raw_allowed or []:
                        if x is None:
                            continue
                        if isinstance(x, str):
                            parts.extend([p for p in re.split(r"[,\s]+", x.strip()) if p])
                        else:
                            parts.append(str(x))
                tmp: list[str] = []
                seen2: set[str] = set()
                for p in parts:
                    qt = _normalize_question_type(p)
                    if not qt:
                        continue
                    if qt in seen2:
                        continue
                    seen2.add(qt)
                    tmp.append(qt)
                policy_allowed_view = tmp or None
            except Exception:
                policy_allowed_view = None
    return {
        "track": track,
        "no_mcq_from_step": threshold,
        "applies_to_next_step": next_step,
        "allowed_question_types_for_next_step": _allowed_question_types_for_next_step(agent_conf, state),
        "base_allowed_question_types_for_next_step": base_allowed,
        "policy_allowed_question_types": policy_allowed_view,
        "effective_allowed_question_types_for_next_step": effective_allowed,
    }


def decide_next(agent_conf: Dict[str, Any], state: AgentState) -> Decision:
    """调用导演模型做决策。agent_conf['director'] 包含 generator 与 prompt_path。"""
    director_conf = agent_conf.get("director") or {}
    prompt_path = Path(director_conf.get("prompt_path") or "src/agenqa/prompts/director.prompt")
    use_calc = has_zam_cal_marker(prompt_path)
    generator = director_conf.get("generator") or {
        "service_type": "private_endpoint",
        "service_id": director_conf.get("service_id"),
    }
    generator = with_idealab_session_id(generator, idealab_session_id_for_director(state))
    max_attempts = max(1, int(director_conf.get("max_retries", 2) or 2))
    retry_delay = float(director_conf.get("retry_delay_seconds", 1.0) or 1.0)
    agent_lang = str((agent_conf.get("agent") or {}).get("lang") or "").lower() or None
    use_en = str(agent_lang or "").lower().strip() in {"en", "english"}
    # 可配置的可用动作列表：
    # - 对于链路首步（尚无 history 且无 episode_seed）：强制仅允许 Init；
    # - 之后按配置或默认 Extend/Revise/Finish。
    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
    allowed_ops_conf = agent_block.get("allowed_operations")

    track = str(agent_block.get("track") or "").strip().lower()
    if track and track not in {"unified", "semantic", "executable"}:
        raise ValueError(f"unsupported track={track!r} (expected 'unified'|'semantic'|'executable')")
    is_executable_track = track == "executable"

    def _has_episode_seed() -> bool:
        memory = KnownTree.normalize_memory(getattr(state, "memory", None))
        seed = memory.get("episode_seed") or {}
        if not isinstance(seed, dict):
            return False
        anchor = seed.get("anchor")
        if isinstance(anchor, str) and anchor.strip():
            return True
        subject = seed.get("subject")
        if isinstance(subject, str) and subject.strip():
            return True
        keywords = seed.get("keywords")
        if isinstance(keywords, list) and any(str(k).strip() for k in keywords):
            return True
        return False

    def _has_executable_seed() -> bool:
        memory = KnownTree.normalize_memory(getattr(state, "memory", None))
        seed = memory.get("executable_seed")
        if not isinstance(seed, dict):
            return False
        pid = seed.get("problem_id")
        return isinstance(pid, str) and bool(pid.strip())

    fallback_extend_op = "Extend"

    if is_executable_track:
        has_seed = _has_executable_seed()
        has_hist = any(True for _ in state.iter_executable_records())
        if (not has_hist) and (not has_seed):
            allowed_ops = ["Init"]
        elif (not has_hist) and has_seed:
            allowed_ops = ["Extend"]
        else:
            if isinstance(allowed_ops_conf, (list, tuple)):
                allowed_ops = [str(op) for op in allowed_ops_conf if str(op).strip()]
            else:
                allowed_ops = ["Extend", "Revise", "Finish"]
            if not allowed_ops:
                allowed_ops = ["Extend", "Finish"]
    else:
        has_seed = _has_episode_seed()
        if not state.history and not has_seed:
            # 首步：需要先构造 episode_seed
            allowed_ops = ["Init"]
        elif not state.history and has_seed:
            # episode_seed 已构造但还没有 QA：走 Extend 出第一道题
            allowed_ops = ["Extend"]
        else:
            if isinstance(allowed_ops_conf, (list, tuple)):
                allowed_ops = [str(op) for op in allowed_ops_conf if str(op).strip()]
            else:
                allowed_ops = ["Extend", "Revise", "Finish"]
            if not allowed_ops:
                allowed_ops = ["Extend", "Finish"]
    allowed_ops = _normalize_allowed_ops(allowed_ops)

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resolved = resolve_inference(generator)
            sess = resolved.session
            payload = build_director_view(state, agent_conf)
            try:
                view_ops = payload.get("available_operations")
                if isinstance(view_ops, list) and view_ops:
                    allowed_ops = _normalize_allowed_ops(list(view_ops))
            except Exception:
                pass

            # 在未达到步数上限前，通常不允许 Director 直接选择 Finish；
            # 仅在“path 强 solver 解错、edge 强 solver 解对”这一特例下允许提前收束。
            try:
                max_steps = int(agent_block.get("max_steps", state.max_steps))
            except Exception:
                max_steps = state.max_steps
            try:
                curr_step = int(state.step or 0)
            except Exception:
                curr_step = 0
            # 仅当当前 step 尚未接近 max_steps（即还可以继续扩展）时应用限制
            if curr_step < max_steps - 1:
                try:
                    solver_metrics = payload.get("solver_metrics") or {}
                    edge_strong = (solver_metrics.get("edge") or {}).get("strong")
                    path_strong = (solver_metrics.get("path") or {}).get("strong")
                    allow_early_finish = False
                    if isinstance(edge_strong, list) and isinstance(path_strong, list) and edge_strong and path_strong:
                        edge_all_correct = all(isinstance(x, dict) and x.get("correct") is True for x in edge_strong)
                        path_any_incorrect = any(isinstance(x, dict) and x.get("correct") is False for x in path_strong)
                        if edge_all_correct and path_any_incorrect:
                            allow_early_finish = True
                    if not allow_early_finish:
                        # 从可选动作中移除 Finish 相关选项，避免 Director 过早收束链路
                        allowed_ops = [op for op in allowed_ops if str(op).strip().lower() not in {"finish"}]
                        if not allowed_ops:
                            # 至少保留 Extend 作为安全回退
                            allowed_ops = [fallback_extend_op]
                except Exception:
                    # 若统计信息缺失，则保守起见同样避免提前 Finish
                    allowed_ops = [op for op in allowed_ops if str(op).strip().lower() not in {"finish"}] or [fallback_extend_op]

            # Keep the view payload consistent with the filtered allowed ops.
            try:
                payload["available_operations"] = list(allowed_ops)
            except Exception:
                pass

            if is_executable_track:
                user_prompt = build_director_v1_body(payload, allowed_ops, lang=agent_lang)
            else:
                user_prompt = (
                    build_director_v1_body_calc(payload, allowed_ops, lang=agent_lang)
                    if use_calc
                    else build_director_v1_body(payload, allowed_ops, lang=agent_lang)
                )
            # 将模板与渲染后的 Director prompt 落盘，便于回溯
            prompt_dir = None
            try:
                # Director 使用 Python 模块构建 prompt，而非 .prompt 文件
                # 获取模板的原始文本内容（未渲染的模板）
                track_norm = str(payload.get("track") or "").strip().lower()
                if track_norm == "executable":
                    template = DIRECTOR_TEMPLATE_EXECUTABLE_EN if use_en else DIRECTOR_TEMPLATE_EXECUTABLE
                else:
                    if use_calc:
                        template = DIRECTOR_TEMPLATE_CALC_EN if use_en else DIRECTOR_TEMPLATE_CALC
                    else:
                        template = DIRECTOR_TEMPLATE_EN if use_en else DIRECTOR_TEMPLATE
                template_text = "\n\n".join(section.text for section in template.sections)
                snapshot_prompt_used(
                    prompt_path,
                    state.artifacts_dir / "00_Prompts_Snapshot",
                    content=template_text,
                    name_prefix="prompt_used.director.",
                    logger=logger,
                )
                round_idx = state.current_round_index()
                step_idx = int(state.step or 0)
                # 使用统一的目录布局：round_r/step_step_director/
                director_step_dir = compute_step_dir(state.artifacts_dir, "director", step_idx, round_idx)
                prompt_dir = director_step_dir / "subruns_raw" / "director"
                snapshot_rendered_prompt(user_prompt, prompt_dir, logger=logger)
            except Exception:
                pass
            messages = build_messages_with_background(user_prompt, lang=agent_lang)
            resp = sess.chat(messages, **resolved.chat_args)
            text = sess.extract_text(resp, default="").strip()
            # 解析 JSON（模型策略统一处理清洗/兜底）
            cleaned_text = clean_json_text(
                text,
                generator=generator,
                task_name="director",
                lang=agent_lang or "zh",
                required_keys=[["Operation", "operation", "operator", "action"]],
                prompt_body=user_prompt,
                snapshot_dir=prompt_dir,
                allow_python=True,
            )
            data = json.loads(cleaned_text) if cleaned_text else None
            if not isinstance(data, dict):
                raise ValueError("Director output is not a JSON object")
            op_raw = str(data.get("Operation") or data.get("operator") or "Extend")
            op = _normalize_operation(op_raw, allowed_ops)
            # 如果操作被规范化（即 LLM 返回的操作不在 allowed_ops 中），需要检查 reason 和 operator_notes 是否匹配
            op_was_normalized = op_raw.strip().lower() != op.strip().lower()
            reason = str(data.get("Reason") or data.get("reason") or "")
            question_type = _normalize_question_type(data.get("QuestionType") or data.get("question_type"))
            # OperatorNotes（推荐字段）或 Macro/params（兼容旧格式）
            params_raw = (
                data.get("OperatorNotes")
                or data.get("operator_notes")
                or data.get("Macro")
                or data.get("params")
                or {}
            )
            # 允许 OperatorNotes 为字符串；统一包装为 dict
            if isinstance(params_raw, str):
                params = {"operator_notes": params_raw}
            elif isinstance(params_raw, dict):
                params = params_raw
            else:
                params = {}

            # 如果操作被规范化，且 reason 明显不匹配（包含被拒绝的操作名），则生成默认的 reason 和 operator_notes
            if op_was_normalized:
                op_raw_lower = op_raw.strip().lower()
                reason_lower = reason.lower()
                # 检查 reason 中是否包含被拒绝的操作名
                if op_raw_lower in reason_lower and op_raw_lower != op.strip().lower():
                    # 生成默认的 reason
                    if op.lower() == "extend":
                        if not state.history and has_seed:
                            reason = (
                                "episode_seed has already been initialized via Init; history is empty. "
                                "Generate the first question (step=1) based on the seed to start the multi-step reasoning chain."
                                if use_en
                                else "episode_seed 已通过 Init 初始化完成，当前 history 为空，需要基于 seed 生成第一道题目（step=1），开启多步推理链条。"
                            )
                            if isinstance(params, dict) and not params.get("operator_notes"):
                                params["operator_notes"] = (
                                    "Create the first question using the episode_seed (use the seed's anchoring fields; subject/keywords optional). Ensure it anchors the "
                                    "topic, sets up a clear reasoning path, and serves as a good starting point for a multi-step chain."
                                    if use_en
                                    else "基于 episode_seed（使用 seed 的锚定字段；subject/keywords 可选）生成第一道题目，确保题目能够充分锚定主题，为后续扩展题目奠定基础。题目应具有清晰的推理步骤，适合作为多步推理链的起点。"
                                )
                        else:
                            reason = (
                                "Continue by generating the next question to advance the multi-step reasoning chain."
                                if use_en
                                else "继续扩展生成下一道题目，推进多步推理链条。"
                            )
                    elif op.lower() == "knowninit":
                        reason = (
                            "At step=0 with empty history, initialize the Known structure (episode_seed) as the starting point "
                            "for the chain, so that subsequent questions have a stable foundation."
                            if use_en
                            else "当前处于 step=0，历史为空，需要先初始化 Known 结构（episode_seed）作为整条链路的起点，为后续题目生成奠定基础。"
                        )
                        if isinstance(params, dict) and not params.get("operator_notes"):
                            params["operator_notes"] = (
                                "Build a stable episode_seed from the input material (fields per contract; subject/keywords optional). "
                                "Keep it information-rich but not redundant, and make sure it can support generating multi-step reasoning problems."
                                if use_en
                                else "请从输入材料中构造一个稳定的 episode_seed（字段以 contract 为准；subject/keywords 可选），确保信息充分但不冗余，能够支撑多步推理题目的生成。"
                            )
                    else:
                        reason = f"Execute {op} to proceed with the problem-generation flow." if use_en else f"执行 {op} 操作以推进题目生成流程。"
                    logger.warning(
                        "Director 返回的操作 %s 不在允许列表中，已规范化为 %s，并生成了默认 reason 和 operator_notes",
                        op_raw,
                        op,
                    )
            # 若为 Revise 操作，允许通过 ReviseMode/revise_mode 明确区分修订模式
            if isinstance(params, dict) and str(op).lower() == "revise":
                try:
                    raw_mode = (
                        data.get("ReviseMode")
                        or data.get("revise_mode")
                        or (params.get("ReviseMode") if isinstance(params.get("ReviseMode"), str) else None)
                        or (params.get("revise_mode") if isinstance(params.get("revise_mode"), str) else None)
                    )
                    if raw_mode:
                        from agenqa.revise_modes import normalize_revise_mode

                        mode = normalize_revise_mode(raw_mode)
                        if mode:
                            params["revise_mode"] = mode
                except Exception:
                    pass

            # Enforce question-type policy for semantic track when Operation="Extend".
            if not is_executable_track and str(op).lower() == "extend" and isinstance(params, dict):
                allowed_qtypes = _allowed_question_types_for_next_step(agent_conf, state)

                if question_type is None:
                    # Missing QuestionType would later fall back to MCQ in select_question_type(),
                    # which may violate the policy; treat as invalid for retry/coercion.
                    if attempt < max_attempts - 1:
                        raise ValueError(f"Director missing QuestionType; allowed={allowed_qtypes}")
                    logger.error("Director missing QuestionType; coercing to %s (allowed=%s)", allowed_qtypes[0], allowed_qtypes)
                    question_type = allowed_qtypes[0]
                    params["question_type_policy"] = {
                        "violation": "missing_question_type",
                        "coerced_to": question_type,
                        "allowed": list(allowed_qtypes),
                    }
                elif question_type not in allowed_qtypes:
                    if attempt < max_attempts - 1:
                        raise ValueError(f"Director QuestionType={question_type!r} violates policy; allowed={allowed_qtypes}")
                    logger.error(
                        "Director QuestionType=%s violates policy; coercing to %s (allowed=%s)",
                        question_type,
                        allowed_qtypes[0],
                        allowed_qtypes,
                    )
                    coerced = allowed_qtypes[0]
                    params["question_type_policy"] = {
                        "violation": "disallowed_question_type",
                        "reported": question_type,
                        "coerced_to": coerced,
                        "allowed": list(allowed_qtypes),
                    }
                    question_type = coerced

                params.setdefault("question_type", question_type)
            elif question_type and isinstance(params, dict):
                params.setdefault("question_type", question_type)

            # 避免在连续 Revise 时浪费算力/污染轨迹：达到重试上限则直接终止 episode（fail-fast）。
            # 设计意图：连续多次仍选择 Revise，说明该题低质量或修复效率过低，应尽快放弃并重采样。
            try:
                agent_block2 = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
                raw_limit = agent_block2.get("max_consecutive_revise", agent_block2.get("revise_retry_limit", 1))
                max_consecutive_revise = int(raw_limit) if raw_limit is not None else 1
                if max_consecutive_revise < 1:
                    max_consecutive_revise = 1
            except Exception:
                max_consecutive_revise = 1

            try:
                m_ok = state.metrics.correct_medium
            except Exception:
                m_ok = None
            try:
                step_idx2 = int(getattr(state, "step", 0) or 0)
            except Exception:
                step_idx2 = 0
            try:
                edge_rows = state.get_latest_solver_results(step_idx2, "edge", "strong")
                edge_explicit = [res.correct for _tier, res in edge_rows if res.correct is not None]
                s_ok = True if edge_explicit and all(v is True for v in edge_explicit) else (False if any(v is False for v in edge_explicit) else None)
            except Exception:
                s_ok = None

            if str(op).lower() == "revise" and isinstance(params, dict):
                try:
                    prev_revise = int(getattr(state, "consecutive_revise", 0) or 0)
                except Exception:
                    prev_revise = 0
                # Pure cap policy: once consecutive revise attempts exceed the configured limit,
                # stop regardless of current solver correctness snapshot.
                if prev_revise >= max_consecutive_revise:
                    state.stop_reason = f"revise_retry_exhausted(max={max_consecutive_revise})"
                    op = "Finish"
                    reason = (
                        f"Stop episode: consecutive Revise reached max={max_consecutive_revise} and Director still selected Revise."
                    )
                    params.setdefault("stop_policy", {})
                    if isinstance(params.get("stop_policy"), dict):
                        params["stop_policy"].update(
                            {
                                "kind": "revise_retry_exhausted",
                                "max_consecutive_revise": max_consecutive_revise,
                                "observed_consecutive_revise": prev_revise,
                                "correct_medium": m_ok,
                                "strong_all_correct": s_ok,
                                "policy": "pure_revise_cap",
                            }
                        )
                    # Keep revise_mode for debugging context, but make the termination explicit.

                # Type2: avoid infinite answer_contract revise loops when contract errors persist.
                try:
                    raw_limit2 = agent_block2.get("max_consecutive_answer_contract_revise", 2)
                    max_consecutive_ac = int(raw_limit2) if raw_limit2 is not None else 2
                    if max_consecutive_ac < 1:
                        max_consecutive_ac = 1
                except Exception:
                    max_consecutive_ac = 2

                try:
                    prev_ac = int(getattr(state, "consecutive_answer_contract_revise", 0) or 0)
                except Exception:
                    prev_ac = 0

                try:
                    rm = str(params.get("revise_mode") or "").strip().lower()
                except Exception:
                    rm = ""

                t2 = None
                has_type2_errors = False
                try:
                    t2 = payload.get("type2_contract") if isinstance(payload, dict) else None
                    if isinstance(t2, dict):
                        has_type2_errors = bool(t2.get("has_errors") is True)
                except Exception:
                    has_type2_errors = False

                if rm == "answer_contract" and has_type2_errors and prev_ac >= max_consecutive_ac:
                    state.stop_reason = f"answer_contract_retry_exhausted(max={max_consecutive_ac})"
                    op = "Finish"
                    reason = (
                        f"Stop episode: consecutive Revise(answer_contract) reached max={max_consecutive_ac} but Type2 contract errors persisted."
                    )
                    params.setdefault("stop_policy", {})
                    if isinstance(params.get("stop_policy"), dict):
                        params["stop_policy"].update(
                            {
                                "kind": "answer_contract_retry_exhausted",
                                "max_consecutive_answer_contract_revise": max_consecutive_ac,
                                "observed_consecutive_answer_contract_revise": prev_ac,
                                "type2_contract_has_errors": has_type2_errors,
                                "type2_contract": (t2 if isinstance(t2, dict) else None),
                            }
                        )

            # Fail-fast: if the Director itself signals an unrecoverable foundation corruption, stop immediately.

            abort_text = "\n".join(
                [
                    str(reason or ""),
                    str(params.get("operator_notes") or "") if isinstance(params, dict) else "",
                ]
            ).lower()
            abort_markers = ("cannot continue", "terminate", "fatal", "must stop", "abort", "无法继续", "终止", "中止")
            corruption_markers = ("corrupted", "garbled", "malformed", "invalid", "乱码", "损坏", "不合法", "格式错误")
            foundation_markers = (
                "episode_seed",
                "seed_init",
                "known_0",
                "background_init",
                "background init",
                "background",
                "knowninit",
            )
            if any(m in abort_text for m in abort_markers) or (
                any(m in abort_text for m in corruption_markers) and any(m in abort_text for m in foundation_markers)
            ):
                logger.error("Director reported critical foundation issue; terminating: %s", reason)
                raise RuntimeError(f"Director critical error: {reason}")

            decision = Decision(operation=op, reason=reason, params=params)
            logger.info("Director 决策: %s | %s", op, reason)
            return decision
        except Exception as e:  # noqa: BLE001
            # Fail-fast for configuration errors (do not retry).
            if isinstance(e, ValueError) and str(e).startswith("[CONFIG]"):
                raise
            last_exc = e
            if attempt < max_attempts - 1:
                logger.warning("Director 调用失败，准备重试 (%s/%s): %s", attempt + 1, max_attempts, str(e))
                try:
                    time.sleep(retry_delay)
                except Exception:
                    pass
                continue
            logger.error("Director 调用失败，所有重试 (%s) 都已耗尽: %s", max_attempts, str(e))
            raise RuntimeError(f"Director failed after {max_attempts} attempts: {e}") from e


def _canonical_operation(op_raw: str) -> str:
    key = str(op_raw or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if not key:
        return ""
    if key in {"init", "knowninit", "executableinit", "qainit"}:
        return "Init"
    if "revise" in key:
        return "Revise"
    if "extend" in key:
        return "Extend"
    if key in {"finish", "stop"}:
        return "Finish"
    return str(op_raw).strip()


def _normalize_allowed_ops(allowed_ops: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for op in allowed_ops or []:
        canon = _canonical_operation(op)
        if not canon:
            continue
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
    return out


def _normalize_operation(op_raw: str, allowed_ops: list[str]) -> str:
    """Normalize requested operation against allowed_ops; fallback to first allowed."""
    allowed_ops = _normalize_allowed_ops(allowed_ops)
    if not allowed_ops:
        return "Extend"
    canon = _canonical_operation(op_raw)
    if not canon:
        return allowed_ops[0]
    for op in allowed_ops:
        if canon.strip().lower() == str(op).strip().lower():
            return op
    return allowed_ops[0] if allowed_ops else "Extend"


def _normalize_question_type(val: Any) -> str | None:
    """Normalize question type emitted by director into MCQ/Derivation/Numeric."""
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if not v:
        return None
    if v in {"mcq", "choice", "single", "single_choice", "single-choice"}:
        return "MCQ"
    if v in {"derivation", "derive", "symbolic", "proof", "algebra", "formula"}:
        return "Derivation"
    if v in {"numeric", "calc", "calculation", "compute", "number", "estimate", "estimation", "approx", "approximation"}:
        return "Numeric"
    return None
