from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from copy import deepcopy
import logging
import json

from infra.data.io import read_jsonl
from agenqa.graph.state import AgentState
from agenqa.memory.store import (
    dump_director_decision_for_step,
    dump_edge_kqa_for_step,
    dump_path_kqa_for_step,
)
from agenqa.graph.output_manager import OutputContext, compute_step_dir
from agenqa.domain.node_result import NodeResult
from agenqa.nodes.roles_nodes import revise_diagnose_node, revise_draft_node, revise_format_node
from agenqa.revise_modes import (
    REVISE_MODE_CORRECTNESS,
    REVISE_MODE_WORLD_CONTRACT,
    REVISE_MODE_ANSWER_CONTRACT,
    REVISE_MODE_REUSE_HIDDEN,
    normalize_revise_mode,
)

logger = logging.getLogger(__name__)

_HISTORY_POINTER_MARKERS: tuple[str, ...] = (
    "上一步",
    "上题",
    "上一题",
    "前一题",
    "前一步",
    "根据step",
    "根据 step",
    "step=",
    "step ",
    "as shown above",
    "use the previous",
    "previous step",
)


def _has_history_pointer(text: Any) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.lower()
    return any(m.lower() in t for m in _HISTORY_POINTER_MARKERS)


def _normalize_question_type(val: Any) -> str | None:
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


def _infer_question_type(agent_conf: Dict[str, Any], state: AgentState, step_idx: int) -> str | None:
    """Infer the question type for the current Revise target.

    Hard contract:
    - Revise must inherit and keep the *locked* question_type of the step being revised.
    - If a locked question_type exists but violates the policy for that step, fail-fast.
    - Heuristic inference is only a legacy fallback when the locked metadata is missing.
    """

    from agenqa.nodes.utils import allowed_question_types_for_step, infer_question_type_from_qa

    try:
        allowed = allowed_question_types_for_step(agent_conf or {}, int(step_idx))
    except Exception:
        # Config errors should surface early (fail-fast).
        raise

    def _validate_or_raise(qt: str, *, source: str) -> str:
        if allowed and qt not in allowed:
            raise ValueError(
                f"[Revise] locked question_type violates policy "
                f"(run_id={getattr(state, 'run_id', None)!r}, step={step_idx}, source={source}, locked={qt!r}, allowed={allowed})"
            )
        return qt

    # 0) Prefer locked metadata from the target KQARecord.
    try:
        target = state.history[-1] if state.history else None
        if target is not None:
            qt = _normalize_question_type(getattr(target, "question_type", None))
            if not qt:
                constraints = getattr(target, "question_type_constraints", None)
                if isinstance(constraints, dict):
                    qt = _normalize_question_type(constraints.get("locked_question_type"))
            if qt:
                return _validate_or_raise(qt, source="history.locked")
    except ValueError:
        raise
    except Exception:
        pass

    # 1) Explicit hint from current decision (if any).
    try:
        params = deepcopy(getattr(state, "last_decision", None).params) if getattr(state, "last_decision", None) else {}
        qt = _normalize_question_type(params.get("question_type"))
        if qt:
            return _validate_or_raise(qt, source="director_hint")
    except ValueError:
        raise
    except Exception:
        pass

    # 2) Legacy fallback: heuristic inference from current QA text.
    try:
        target = state.history[-1] if state.history else None
        q = getattr(target, "question", "") if target is not None else ""
        a = getattr(target, "answer", "") if target is not None else ""
        qt = infer_question_type_from_qa(q, a)
        if qt:
            qt_norm = _normalize_question_type(qt) or qt
            return _validate_or_raise(str(qt_norm), source="qa_heuristic")
    except ValueError:
        raise
    except Exception:
        pass

    if allowed:
        raise ValueError(
            f"[Revise] cannot infer question_type for revise target "
            f"(run_id={getattr(state, 'run_id', None)!r}, step={step_idx}, allowed={allowed})"
        )
    return None


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
                        rows = read_jsonl(kqa_path)
                        if rows and len(rows) > 0:
                            step_val = rows[0].get("step")
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


# 软限制，避免 solver 答案过长撑爆 Diagnose prompt
_MAX_SOLVER_ANSWER_CHARS = 1200
_MAX_SOLVER_REASONING_CHARS = 2000


def _read_solver_answer(step: int, artifacts_dir: Path, name: str) -> str | None:
    """读取指定 step 的 solver 作答文本（name: solve_strong / solve_medium / ...）。"""
    candidates = []
    try:
        for step_dir in _iter_step_dirs_for_step(step, artifacts_dir):
            round_num = None
            try:
                parts = step_dir.parts
                for part in parts:
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
                        s_val = row.get("step")
                        if s_val is not None and int(s_val) != step:
                            continue
                    except Exception:
                        continue
                    solve_txt = row.get("solve") or row.get("Answer") or row.get("answer")
                    if isinstance(solve_txt, str) and solve_txt.strip():
                        candidates.append((round_num or 0, solve_txt.strip()))
                    break  # 每个文件只读取第一行
                if candidates and candidates[-1][0] == round_num:
                    break  # 已找到该 step_dir 的结果
    except Exception:
        pass

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    ans = candidates[0][1]
    if len(ans) > _MAX_SOLVER_ANSWER_CHARS:
        return ans[:_MAX_SOLVER_ANSWER_CHARS] + " ...[truncated]"
    return ans


def _read_solver_reasoning(step: int, artifacts_dir: Path, name: str) -> str | None:
    """读取指定 step 的 solver 推导摘要（name: solve_strong / solve_medium / ...）。"""
    candidates = []
    try:
        for step_dir in _iter_step_dirs_for_step(step, artifacts_dir):
            round_num = None
            try:
                parts = step_dir.parts
                for part in parts:
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
                        s_val = row.get("step")
                        if s_val is not None and int(s_val) != step:
                            continue
                    except Exception:
                        continue
                    txt = row.get("solver_reasoning") or row.get("SolverReasoning")
                    if isinstance(txt, str) and txt.strip():
                        candidates.append((round_num or 0, txt.strip()))
                    break  # 每个文件只读取第一行
                if candidates and candidates[-1][0] == round_num:
                    break
    except Exception:
        pass

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    reasoning = candidates[0][1]
    if len(reasoning) > _MAX_SOLVER_REASONING_CHARS:
        return reasoning[:_MAX_SOLVER_REASONING_CHARS] + " ...[truncated]"
    return reasoning


def _build_director_notes(state: AgentState) -> str | None:
    """Assemble director_notes from last_decision (mirrors extend/compress behavior)."""
    director_notes: str | None = None
    try:
        params = deepcopy(getattr(state, "last_decision", None).params) if getattr(state, "last_decision", None) else {}
        notes_parts = []
        if isinstance(params, dict):
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
            fb = solver_ctx.get("solver_feedback") if isinstance(solver_ctx, dict) else {}
            if isinstance(fb, dict):
                tier = fb.get("from_tier")
                step_src = fb.get("from_step")
                # 优先使用新字段
                qwp = fb.get("question_well_posed")
                if qwp is not None:
                    notes_parts.append(f"question_well_posed({tier}@step{step_src}): {qwp}")
                cf_txt = fb.get("correctness_feedback")
                if isinstance(cf_txt, str) and cf_txt.strip():
                    notes_parts.append(f"correctness_feedback({tier}@step{step_src}): {cf_txt.strip()}")
                df_txt = fb.get("difficulty_feedback")
                if isinstance(df_txt, str) and df_txt.strip():
                    notes_parts.append(f"difficulty_feedback({tier}): {df_txt.strip()}")
        if notes_parts:
            director_notes = " | ".join(notes_parts)
    except Exception:
        director_notes = None
    return director_notes


def _build_solver_answers_text(state: AgentState) -> str | None:
    """提取最近一步 solver 的作答文本（strong/medium），供 Diagnose 对比参考。"""
    try:
        step_idx = int(state.step)
    except Exception:
        step_idx = max(len(state.history) - 1, 0) if state.history else 0

    artifacts_dir = Path(state.artifacts_dir)
    strong_ans = _read_solver_answer(step_idx, artifacts_dir, "solve_strong")
    medium_ans = _read_solver_answer(step_idx, artifacts_dir, "solve_medium")

    parts = []
    if strong_ans:
        parts.append(f"strong: {strong_ans}")
    if medium_ans:
        parts.append(f"medium: {medium_ans}")

    return "\n".join(parts) if parts else None


def _build_solver_reasoning_text(state: AgentState) -> str | None:
    """提取最近一步 solver 的推导摘要（strong/medium），供 Diagnose 分析推理路径。"""
    try:
        step_idx = int(state.step)
    except Exception:
        step_idx = max(len(state.history) - 1, 0) if state.history else 0

    artifacts_dir = Path(state.artifacts_dir)
    strong_txt = _read_solver_reasoning(step_idx, artifacts_dir, "solve_strong")
    medium_txt = _read_solver_reasoning(step_idx, artifacts_dir, "solve_medium")

    parts = []
    if strong_txt:
        parts.append(f"strong: {strong_txt}")
    if medium_txt:
        parts.append(f"medium: {medium_txt}")
    return "\n".join(parts) if parts else None


def _build_solver_feedback_text(state: AgentState) -> str | None:
    """从 last_decision 中提取与 solver 相关的反馈文本，供 Diagnose 使用。

    只使用新字段（correctness_feedback, difficulty_feedback, question_well_posed），
    根据 revise_mode 选择相应的反馈类型。
    """
    try:
        params = deepcopy(getattr(state, "last_decision", None).params) if getattr(state, "last_decision", None) else {}
    except Exception:
        params = {}
    if not isinstance(params, dict):
        return None
    solver_ctx = params.get("solver_context") if isinstance(params.get("solver_context"), dict) else {}
    if not isinstance(solver_ctx, dict):
        return None

    # 推断当前 revise_mode（用于选择反馈类型）
    revise_mode = _infer_revise_mode(state)
    parts = []

    fb = solver_ctx.get("solver_feedback")
    if isinstance(fb, dict):
        if revise_mode in {REVISE_MODE_CORRECTNESS, REVISE_MODE_WORLD_CONTRACT, REVISE_MODE_ANSWER_CONTRACT}:
            cf = fb.get("correctness_feedback")
            if isinstance(cf, str) and cf.strip():
                parts.append(cf.strip())
            qwp = fb.get("question_well_posed")
            if qwp is False:
                parts.append("[题目不 well-posed]")
        else:
            df = fb.get("difficulty_feedback")
            if isinstance(df, str) and df.strip():
                parts.append(f"[难度] {df.strip()}")

    if not parts:
        solver_metrics = solver_ctx.get("solver_metrics") if isinstance(solver_ctx.get("solver_metrics"), dict) else {}
        if isinstance(solver_metrics, dict):
            views = []
            if revise_mode in {REVISE_MODE_CORRECTNESS, REVISE_MODE_WORLD_CONTRACT, REVISE_MODE_ANSWER_CONTRACT}:
                views = ["edge", "path"]
            else:
                views = ["path", "edge"]
            for view_name in views:
                view = solver_metrics.get(view_name) if isinstance(solver_metrics.get(view_name), dict) else {}
                strong_rows = view.get("strong") if isinstance(view, dict) else None
                if not isinstance(strong_rows, list):
                    continue
                for row in strong_rows:
                    if not isinstance(row, dict):
                        continue
                    if revise_mode in {REVISE_MODE_CORRECTNESS, REVISE_MODE_WORLD_CONTRACT, REVISE_MODE_ANSWER_CONTRACT}:
                        cf = row.get("correctness_feedback")
                        if isinstance(cf, str) and cf.strip():
                            parts.append(cf.strip())
                        qwp = row.get("question_well_posed")
                        if qwp is False:
                            parts.append("[题目不 well-posed]")
                    else:
                        df = row.get("difficulty_feedback")
                        if isinstance(df, str) and df.strip():
                            parts.append(f"[难度] {df.strip()}")
                    if parts:
                        break
                if parts:
                    break

    return " | ".join(parts) if parts else None


def _infer_revise_mode(state: AgentState) -> str:
    """根据 Director 决策与当前求解状态推断 Revise 模式。

    优先顺从 Director：
    - 若 state.last_decision.params.revise_mode 显式给出，则使用该值（支持 correctness / world_contract / answer_contract / reuse_hidden / quality）；

    若 Director 未指定，则回退到简单启发式：
    - 若 edge multi-strong 结果中存在明确 incorrect / not-well-posed 信号，视为“正确性修订”（correctness）；
    - 否则默认视为“结构修订”（reuse_hidden）。若需要“质量修订”（quality），应由 Director 显式指定。
    """
    # 1) 优先读取 Director 显式指定的模式
    try:
        dec = getattr(state, "last_decision", None)
    except Exception:
        dec = None
    if dec is not None:
        try:
            params = dec.params or {}
        except Exception:
            params = {}
        if isinstance(params, dict):
            mode = normalize_revise_mode(params.get("revise_mode"))
            if mode:
                return mode

    # 2) 回退：根据 strong 在 edge QA 上的对错推断，以及“复用/隐藏”相关信号
    try:
        step_idx = int(getattr(state, "step", 0) or 0)
    except Exception:
        step_idx = 0
    try:
        edge_rows = state.get_latest_solver_results(step_idx, "edge", "strong")
        edge_explicit = [res.correct for _tier, res in edge_rows if res.correct is not None]
        edge_has_any_incorrect = bool(edge_explicit) and any(v is False for v in edge_explicit)
    except Exception:
        edge_has_any_incorrect = False
    if edge_has_any_incorrect:
        return REVISE_MODE_CORRECTNESS

    # 2.1) 若题面出现 history 指针语句，优先用 reuse_hidden 修复“隐藏”
    try:
        if state.history and _has_history_pointer(getattr(state.history[-1], "question", "")):
            return REVISE_MODE_REUSE_HIDDEN
    except Exception:
        pass

    # 2.2) 检查 solver 反馈中的 question_well_posed 标志：
    # - 若题面含指针语句导致不 well-posed → reuse_hidden
    # - 否则更可能是缺条件/矛盾等 → correctness
    try:
        params = dec.params or {} if dec is not None else {}
        if isinstance(params, dict):
            solver_ctx = params.get("solver_context") if isinstance(params.get("solver_context"), dict) else {}
            if isinstance(solver_ctx, dict):
                solver_metrics = solver_ctx.get("solver_metrics") if isinstance(solver_ctx.get("solver_metrics"), dict) else {}
                if isinstance(solver_metrics, dict):
                    for view_name in ("edge", "path"):
                        view = solver_metrics.get(view_name) if isinstance(solver_metrics.get(view_name), dict) else {}
                        strong_rows = view.get("strong") if isinstance(view, dict) else None
                        if not isinstance(strong_rows, list):
                            continue
                        if any(isinstance(row, dict) and row.get("question_well_posed") is False for row in strong_rows):
                            if state.history and _has_history_pointer(getattr(state.history[-1], "question", "")):
                                return REVISE_MODE_REUSE_HIDDEN
                            return REVISE_MODE_CORRECTNESS
    except Exception:
        pass

    # 2.3) 若 edge 与 path 都能解对，往往表示链路递进不足或存在泄露风险；
    # 默认用 reuse_hidden 做“复用上一步结论但在题面中隐藏”的结构修订（保持 Answer 等价）。
    try:
        edge_rows = state.get_latest_solver_results(step_idx, "edge", "strong")
        path_rows = state.get_latest_solver_results(step_idx, "path", "strong")
        edge_explicit = [res.correct for _tier, res in edge_rows if res.correct is not None]
        path_explicit = [res.correct for _tier, res in path_rows if res.correct is not None]
        if edge_explicit and path_explicit and all(v is True for v in edge_explicit) and all(v is True for v in path_explicit):
            return REVISE_MODE_REUSE_HIDDEN
    except Exception:
        pass

    return REVISE_MODE_REUSE_HIDDEN


def run_revise(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    """Revise operator: rewrite the last step's Question/Answer in-place."""
    if not state.history:
        # 没有历史时无法 revise，直接返回原状态（由 Director 控制首次必为 Extend）
        logger.warning("Revise 被调用但 history 为空，忽略本次操作。")
        return state

    ops_conf = (agent_conf.get("operators") or {})
    op_conf = (ops_conf.get("revise") or ops_conf.get("extend") or {})
    # Persist path_kqa variant preference onto state so OutputManager/store can dump consistently.
    try:
        agent_block2 = agent_conf.get("agent") or {}
        fold_variant = str(
            agent_block2.get("path_kqa_variant") or agent_block2.get("path_fold_variant") or "direct"
        ).strip().lower()
        if fold_variant not in {"direct", "scaffolded"}:
            fold_variant = "direct"
        state.path_kqa_variant = fold_variant
    except Exception:
        pass

    # 推断本次 Revise 的主要目标：正确性修订 vs 难度/结构修订
    revise_mode = _infer_revise_mode(state)

    # 目标 step：当前最后一条记录的 step；若解析失败则退回 len(history)-1
    try:
        step_idx = int(state.step)
    except Exception:
        step_idx = max(len(state.history) - 1, 0)

    # 为本次 Revise 准备输出目录，供后续 roles 路径（Diagnose/Draft/Format）与主 Revise 使用
    round_idx = state.current_round_index()
    ctx: OutputContext | None = None
    if output_manager:
        ctx = output_manager.begin("revise", step_idx, round_idx, revise_mode=revise_mode)
        step_dir = ctx.step_dir
        ctx.dump_director_decision(state, step_idx)
    else:
        step_dir = compute_step_dir(state.artifacts_dir, "revise", step_idx, round_idx, revise_mode=revise_mode)
        step_dir.mkdir(parents=True, exist_ok=True)
        # 记录本次由 Director 决策得到的操作信息
        dump_director_decision_for_step(state, step_dir, step_idx)

    # 在目录中保存 revise_mode 标记文件
    revise_mode_file = step_dir / "revise_mode.txt"
    revise_mode_file.write_text(revise_mode, encoding="utf-8")
    logger.info("Revise mode saved to %s: %s", revise_mode_file, revise_mode)

    # 执行 roles 路径：Diagnose → Draft → Format
    role_outputs: Dict[str, Any] = {}
    state, rev_ctx = revise_diagnose_node(agent_conf, state)
    rev_ctx = revise_draft_node(agent_conf, state, rev_ctx)
    state, role_outputs, step_dir, step_idx, round_idx = revise_format_node(agent_conf, state, rev_ctx)

    if not output_manager:
        dump_edge_kqa_for_step(state, step_dir)
        dump_path_kqa_for_step(state, step_dir, fold_variant=getattr(state, "path_kqa_variant", "direct") or "direct")

    logger.info("Revise 完成（roles 链路），已覆盖 step=%s 的 Question/Answer。", str(step_idx))

    if output_manager:
        return NodeResult(
            state=state,
            step_idx=step_idx,
            round_idx=round_idx,
            role_outputs=role_outputs,
            step_dir=step_dir,
        )
    return state
