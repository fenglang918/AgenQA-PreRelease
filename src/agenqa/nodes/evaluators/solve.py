"""Solve 节点：分别用中/强模型对最新 KQA 进行求解，更新指标。

本仓库的 semantic/unified pipeline 追求“通用性 + 稳健性”，因此 Solve 的正确性判分不依赖
脆弱的程序化字符串/数值对比（容易被 LaTeX 排版、等价变形、单位换算等噪声击穿）。

当启用 LLM judge 时（满足其一即可）：
- consensus.answer_judge="llm"
- 或 symbolic-only 模式对当前题型生效

Solve 会调用轻量 LLM judge 对 (GT answer, solver solve) 做等价判定，并覆盖 correct/token_ratio。
judge 失败默认 fail-fast（避免静默回退到程序化判分）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

from agenqa.graph.state import AgentState, SolverResult
from agenqa.domain.node_result import NodeResult, OutputSpec
from agenqa.graph.output_manager import OutputContext, compute_step_dir
from agenqa.memory.store import dump_latest_kqa_jsonl, save_state
from agenqa.skills.solving import SolverConfig, SolverRunner
from agenqa.skills.solver_tool import SolverToolConfig, SolverToolRunner
from infra.data.io import read_jsonl, write_jsonl
from agenqa.nodes.evaluators.expression_judge import load_expression_judge_generator, run_expression_equivalence_judge
from agenqa.nodes.evaluators.numeric_judge import run_numeric_equivalence_judge
from agenqa.domain.contracts.answer_contract_bank import (
    extract_answer_contract_context,
)
from agenqa.domain.contracts.solver_contract_text import compose_solver_question
from agenqa.domain.known_tree import KnownTree
from agenqa.nodes.utils import (
    allowed_question_types_for_step,
    infer_question_type_from_qa,
    has_zam_cal_marker,
    idealab_session_id_for_step_node,
    normalize_question_type,
    select_question_type,
    with_idealab_session_id,
)
from agenqa.prompts.agent_prompts import SOLVER_PROMPT, SOLVER_PROMPT_EN
from agenqa.prompts.solver_calc import SOLVER_PROMPT_CALC, SOLVER_PROMPT_CALC_EN
from agenqa.prompts.solver_tool import SOLVER_TOOL_PROMPT, SOLVER_TOOL_PROMPT_EN
from infra.prompt.prompt_tracker import snapshot_prompt_used

logger = logging.getLogger(__name__)


def _coerce_optional_bool(val: Any) -> Optional[bool]:
    return val if isinstance(val, bool) else None


def _read_first_correct_diff(path: Path) -> tuple[Optional[bool], Optional[float]]:
    try:
        for row in read_jsonl(path):
            correct = _coerce_optional_bool(row.get("correct")) if isinstance(row, dict) else None
            diff_val = (row.get("token_ratio") or row.get("difficulty")) if isinstance(row, dict) else None
            diff = diff_val if isinstance(diff_val, (int, float)) else None
            return correct, diff
    except Exception:
        pass
    return None, None


def _normalize_solver_status(row: Dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return "request_failed"
    raw = row.get("solver_status")
    if isinstance(raw, str) and raw.strip():
        val = raw.strip().lower()
        if val in {"success", "request_failed", "parse_failed"}:
            return val
    answer_pred = (row.get("answer_pred") or row.get("solve") or "")
    if isinstance(answer_pred, str) and answer_pred.strip():
        return "success"
    err = row.get("solver_failure_code") or row.get("error")
    if isinstance(err, str) and err.strip():
        return "request_failed"
    return "parse_failed"


def _set_expression_judge_payload(
    row: Dict[str, Any],
    *,
    tier: str,
    status: str,
    equivalent: Optional[str] = None,
    reason: Optional[str] = None,
    skip_reason: Optional[str] = None,
    failure_reason: Optional[str] = None,
) -> None:
    row.setdefault("expression_judge", {})
    payload: Dict[str, Any] = {"status": status}
    if equivalent is not None:
        payload["equivalent"] = equivalent
    if reason is not None:
        payload["reason"] = reason
    if skip_reason is not None:
        payload["skip_reason"] = skip_reason
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    row["expression_judge"][tier] = payload


def _solver_visible_question_for_judge(
    *,
    state: AgentState,
    row: Dict[str, Any] | None = None,
) -> str:
    latest = state.history[-1] if state.history else None
    row0 = row if isinstance(row, dict) else {}

    question = (
        row0.get("question")
        or (latest.question if latest else "")
        or ""
    )
    world_contract_text = (
        row0.get("world_contract_text")
        or (latest.world_contract_text if latest else "")
        or ""
    )
    precomposed = row0.get("question_for_solver")
    if isinstance(precomposed, str) and precomposed.strip():
        return precomposed.strip()
    return compose_solver_question(str(question), str(world_contract_text))


def _infer_question_type_for_solver(agent_conf: Dict[str, Any], state: AgentState) -> str:
    """Infer question type for solver execution.

    Prefer explicit director decision hint; otherwise infer from current QA text so that
    revise rounds can re-determine question type when the hint is missing.
    """
    try:
        step_idx = int(state.step or 0)
    except Exception:
        step_idx = 0

    try:
        allowed = allowed_question_types_for_step(agent_conf or {}, step_idx)
    except Exception:
        # Config errors should surface early (fail-fast).
        raise

    def _validate_or_raise(qt: str, *, source: str) -> str:
        if allowed and qt not in allowed:
            raise ValueError(
                f"[Solve] question_type violates policy "
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

    try:
        params = state.last_decision.params if state.last_decision else None  # type: ignore[assignment]
    except Exception:
        params = None
    hinted = normalize_question_type((params or {}).get("question_type")) if isinstance(params, dict) else None
    if hinted:
        return _validate_or_raise(hinted, source="director_hint")
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
    return _validate_or_raise(select_question_type(state) or "MCQ", source="select_question_type")


def _normalize_solver_tier_configs(agent_conf: Dict[str, Any], tier: str) -> list[Dict[str, Any]]:
    """Normalize solver tier configs into a list (backward compatible).

    Supports:
    - solvers.<tier>: { ... }      (legacy single config)
    - solvers.<tier>: [ { ... } ]  (multi solver configs)
    """
    solvers_block = agent_conf.get("solvers") or {}
    if not isinstance(solvers_block, dict):
        return []
    raw = solvers_block.get(tier)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict) and item]
    if isinstance(raw, dict) and raw:
        return [raw]
    return []


def _merge_generator_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge with deep merge for common nested keys (client/generation)."""
    merged: Dict[str, Any] = dict(base or {})
    for key, val in (override or {}).items():
        if key in {"client", "generation"} and isinstance(merged.get(key), dict) and isinstance(val, dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def _resolve_solver_generator(solve_conf: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a usable generator dict for SolverRunner.

    - If `generator` is provided and already contains api_base+model_name, use it.
    - If only `service_id` is provided, resolve it via llm_service `services.json`.
    """
    generator = solve_conf.get("generator")
    if isinstance(generator, dict) and generator:
        service_id = generator.get("service_id")
        has_api_base = bool(generator.get("api_base") or generator.get("base_url"))
        has_model = bool(generator.get("model_name"))
        if isinstance(service_id, str) and service_id.strip() and (not has_api_base or not has_model):
            service_cfg_path = Path(solve_conf.get("service_config") or DEFAULT_SERVICE_CONFIG)
            base = load_llm_service_full_config(
                service_cfg_path,
                service_id.strip(),
                explicit_model=solve_conf.get("service_model"),
                fallback_model=generator.get("model_name"),
            )
            return _merge_generator_configs(base, generator)
        return generator

    # Backward compatible: service_id at tier level
    service_id = solve_conf.get("service_id")
    if isinstance(service_id, str) and service_id.strip():
        service_cfg_path = Path(solve_conf.get("service_config") or DEFAULT_SERVICE_CONFIG)
        base = load_llm_service_full_config(
            service_cfg_path,
            service_id.strip(),
            explicit_model=solve_conf.get("service_model"),
        )
        # Allow overriding client/generation at tier level
        override: Dict[str, Any] = {}
        for k in ("client", "generation"):
            if isinstance(solve_conf.get(k), dict):
                override[k] = solve_conf.get(k)
        return _merge_generator_configs(base, override)

    # Fallback (may still fail later if api_base/model_name are missing)
    return {
        "service_type": "private_endpoint",
        "service_id": service_id,
    }


def _ensure_client_timeout(generator: Dict[str, Any], timeout_seconds: int | None) -> Dict[str, Any]:
    if not timeout_seconds:
        return generator
    if not isinstance(generator, dict):
        return generator
    gen = dict(generator)
    client = gen.get("client")
    if not isinstance(client, dict):
        client = {}
    if client.get("timeout") is None:
        client = dict(client)
        client["timeout"] = int(timeout_seconds)
    gen["client"] = client
    return gen


def _consensus_block(agent_conf: Dict[str, Any]) -> Dict[str, Any]:
    block = agent_conf.get("consensus") or {}
    return block if isinstance(block, dict) else {}


def _solver_timeout_seconds(agent_conf: Dict[str, Any]) -> int:
    """Default per-solver timeout (seconds). Only applied when generator.client.timeout is missing."""
    block = _consensus_block(agent_conf)
    raw = block.get("timeout_seconds")
    if isinstance(raw, (int, float)):
        try:
            val = int(raw)
            return val if val > 0 else 120
        except Exception:
            return 120
    return 120


def _is_symbolic_only(agent_conf: Dict[str, Any], *, question_type: str | None = None) -> bool:
    """判断是否启用“符号表达式 ONLY” 模式。

    支持两种配置方式（任一生效即可）：
    - agent.agent.symbolic_only = true
    - agent.agent.allow_numeric_values = false
    - agent.agent.symbolic_only_question_types: ["Derivation"]  # 仅对指定题型启用
    """
    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
    val = agent_block.get("symbolic_only")
    if isinstance(val, bool):
        # 约定：symbolic_only=true 强制开启；symbolic_only=false 不应否决其他开关（如 symbolic_only_question_types）。
        if val:
            return True
    allow_numeric = agent_block.get("allow_numeric_values")
    if isinstance(allow_numeric, bool):
        # allow_numeric_values=false 强制开启；true 不否决 symbolic_only_question_types。
        if not allow_numeric:
            return True
    raw = agent_block.get("symbolic_only_question_types")
    if raw is None:
        return False
    if not question_type:
        return False
    qtype = normalize_question_type(question_type)
    if not qtype:
        return False
    items: list[str] = []
    if isinstance(raw, str):
        items = [p for p in raw.replace(",", " ").split() if p]
    elif isinstance(raw, (list, tuple)):
        for x in raw:
            if x is None:
                continue
            if isinstance(x, str):
                items.extend([p for p in x.replace(",", " ").split() if p])
            else:
                items.append(str(x))
    else:
        return False
    for it in items:
        if normalize_question_type(it) == qtype:
            return True
    return False


def _apply_expression_judge(
    agent_conf: Dict[str, Any],
    state: AgentState,
    tier: str,
    solve_path: Path,
    numeric_ok: Optional[bool],
    numeric_diff: Optional[float],
    *,
    question_type: str | None,
) -> Tuple[Optional[bool], Optional[float]]:
    """在需要时调用 LLM judge，并在成功时覆盖 correct/token_ratio。

    触发条件（满足其一即可）：
    - symbolic-only 模式对当前题型生效（agent.symbolic_only / allow_numeric_values / symbolic_only_question_types）
    - consensus.answer_judge == "llm"（通用 pipeline 默认推荐）

    返回更新后的 (ok, token_ratio)。
    在 LLM judge 模式下：
    - judge 输出 equivalent=uncertain 会被视为判定失败：correct=False（不回退到数值判定）
    - 上游 solver 不可用时，显式标记 judge skipped，correct=None
    - judge 自身请求失败时，显式标记 judge failed，correct=None
    """
    qtype = normalize_question_type(question_type)
    if not qtype:
        return numeric_ok, numeric_diff

    cons_block = _consensus_block(agent_conf)
    answer_judge = str(cons_block.get("answer_judge") or "").strip().lower()
    use_llm_judge = False
    if _is_symbolic_only(agent_conf, question_type=question_type):
        use_llm_judge = True
    elif answer_judge == "llm":
        use_llm_judge = True
    if not use_llm_judge:
        return numeric_ok, numeric_diff

    rows = []
    try:
        rows = list(read_jsonl(solve_path, schema=None, max_lines=1))
    except Exception as exc:  # noqa: BLE001
        msg = f"expression_judge: failed to read solve jsonl: {exc}"
        logger.warning("expression_judge[%s]: failed to read solve jsonl: %s", tier, str(exc))
        return None, numeric_diff
    if not rows:
        logger.warning("expression_judge[%s]: empty solve jsonl", tier)
        return None, numeric_diff
    row = rows[0]

    # 优先从 solve 行中读取（尤其是 path 的 known/question 与 edge 不同），缺失时再回退到 AgentState。
    latest = state.history[-1] if state.history else None
    known = (row.get("known") or (latest.known if latest else "")) or ""
    question = _solver_visible_question_for_judge(state=state, row=row)
    answer_ref = (row.get("answer_ref") or row.get("answer") or (latest.answer if latest else "")) or ""
    answer_pred = (row.get("answer_pred") or row.get("solve") or "")
    solver_status = _normalize_solver_status(row)
    row.setdefault("answer_ref", answer_ref)
    row.setdefault("answer_pred", answer_pred)
    row.setdefault("solver_status", solver_status)

    if solver_status != "success":
        skip_reason = "solver_request_failed" if solver_status == "request_failed" else "solver_parse_failed"
        _set_expression_judge_payload(row, tier=tier, status="skipped", skip_reason=skip_reason)
        row["correct"] = None
        row["token_ratio"] = None
        try:
            write_jsonl([row], solve_path, schema=None, append=False)
        except Exception:
            pass
        logger.info("expression_judge[%s]: skipped due to upstream unavailable (%s)", tier, skip_reason)
        return None, numeric_diff

    if not (answer_ref and answer_pred):
        _set_expression_judge_payload(row, tier=tier, status="skipped", skip_reason="invalid_solver_row")
        row["correct"] = None
        row["token_ratio"] = None
        try:
            write_jsonl([row], solve_path, schema=None, append=False)
        except Exception:
            pass
        logger.warning("expression_judge[%s]: skipped due to invalid solver row (missing answer_ref/answer_pred)", tier)
        return None, numeric_diff

    try:
        generator = load_expression_judge_generator(agent_conf)
    except Exception as exc:  # noqa: BLE001
        msg = f"expression_judge: failed to load generator config: {exc}"
        _set_expression_judge_payload(row, tier=tier, status="failed", failure_reason=msg)
        row["correct"] = None
        row["token_ratio"] = None
        try:
            write_jsonl([row], solve_path, schema=None, append=False)
        except Exception:
            pass
        logger.warning("expression_judge[%s]: failed to load generator config: %s", tier, str(exc))
        return None, numeric_diff
    if not generator:
        msg = "expression_judge: no generator configured"
        _set_expression_judge_payload(row, tier=tier, status="failed", failure_reason=msg)
        row["correct"] = None
        row["token_ratio"] = None
        try:
            write_jsonl([row], solve_path, schema=None, append=False)
        except Exception:
            pass
        logger.warning("expression_judge[%s]: no generator configured", tier)
        return None, numeric_diff

    # 会话亲和：expression_judge 作为 solve 的子步骤，复用同一 step session id，保持路由与缓存一致性。
    op_raw = (state.last_decision.operation if state.last_decision else "extend")  # type: ignore[union-attr]
    op_lc = str(op_raw or "").lower()
    if "qa_init" in op_lc or "qainit" in op_lc:
        op_name = "qa_init"
    elif "init" in op_lc:
        op_name = "extend"
    elif "extend" in op_lc:
        op_name = "extend"
    elif "revise" in op_lc:
        op_name = "revise"
    else:
        op_name = op_lc.replace("-", "_") or "extend"
    try:
        step_idx = int(state.step)
    except Exception:
        step_idx = 0
    node_for_session = op_name
    if op_name == "revise":
        try:
            from agenqa.nodes.op_revise import _infer_revise_mode

            node_for_session = f"revise_{_infer_revise_mode(state)}"
        except Exception:
            node_for_session = "revise"
    generator = with_idealab_session_id(
        generator,
        idealab_session_id_for_step_node(state, node_for_session, step_idx),
    )

    try:
        mem = KnownTree.normalize_memory(getattr(state, "memory", None))
        type2_ctx = extract_answer_contract_context(mem, step=step_idx)
    except Exception:
        type2_ctx = {}

    try:
        agent_lang = str(((agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}).get("lang") or "").lower().strip() or None
        if qtype == "Numeric":
            eq, reason = run_numeric_equivalence_judge(
                generator,
                known=str(known),
                question=str(question),
                answer_ref=str(answer_ref),
                answer_pred=str(answer_pred),
                lang=agent_lang,
            )
        else:
            eq, reason = run_expression_equivalence_judge(
                generator,
                known=str(known),
                question=str(question),
                answer_ref=str(answer_ref),
                answer_pred=str(answer_pred),
                answer_contract=type2_ctx if isinstance(type2_ctx, dict) else None,
                lang=agent_lang,
            )
    except Exception as exc:  # noqa: BLE001
        msg = f"expression_judge: request/parse failed: {exc}"
        _set_expression_judge_payload(row, tier=tier, status="failed", failure_reason=msg)
        row["correct"] = None
        row["token_ratio"] = None
        try:
            write_jsonl([row], solve_path, schema=None, append=False)
        except Exception:
            pass
        logger.warning("expression_judge[%s]: failed: %s", tier, msg)
        return None, numeric_diff

    if eq is None:
        # LLM 无法给出等价性判断时，视为“判定失败且不依赖数值回退”（避免静默 fallback）。
        logger.info(
            "expression_judge[%s]: equivalent=uncertain, disable numeric fallback in symbolic-only mode (numeric_ok=%s)",
            tier,
            numeric_ok,
        )
        try:
            _set_expression_judge_payload(row, tier=tier, status="success", equivalent="uncertain", reason=reason)
            # 保留原始数值判定结果供诊断，但将最终 correct 统一视为 False
            row["correct_numeric"] = _coerce_optional_bool(row.get("correct"))
            row["correct"] = False
            row["token_ratio"] = None
            write_jsonl([row], solve_path, schema=None, append=False)
        except Exception:
            pass
        return False, None

    # 根据表达式 judge 结果覆盖 correct，并按 metrics 重新计算 token_ratio
    ok = bool(eq)
    metrics = row.get("metrics") or {}
    kq_tokens = metrics.get("kq_tokens")
    completion_tokens = metrics.get("completion_tokens")
    diff: Optional[float]
    try:
        if ok and isinstance(kq_tokens, (int, float)) and isinstance(completion_tokens, (int, float)):
            diff = float(completion_tokens) / max(1.0, float(kq_tokens))
        else:
            diff = None
    except Exception:
        diff = numeric_diff if ok else None

    # 回写 solve JSONL：保留数值判定结果用于诊断，但以 expression_judge 为主。
    try:
        _set_expression_judge_payload(
            row,
            tier=tier,
            status="success",
            equivalent="yes" if ok else "no",
            reason=reason,
        )
        row["correct_numeric"] = _coerce_optional_bool(row.get("correct"))
        row["correct"] = ok
        row["token_ratio"] = diff
        write_jsonl([row], solve_path, schema=None, append=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("expression_judge: 回写 solve 结果失败（不影响主流程）: %s", str(exc))

    logger.info("expression_judge[%s]: equivalent=%s diff=%s", tier, ok, diff)
    return ok, diff


def _solve_once(
    agent_conf: Dict[str, Any],
    state: AgentState,
    tier: str,
    solve_conf_override: Dict[str, Any] | None = None,
    *,
    enable_path: bool = True,
    timeout_seconds: int | None = None,
) -> Tuple[Optional[bool], float | None, Path, Path, Path | None]:
    solve_conf: Dict[str, Any]
    if solve_conf_override is not None:
        solve_conf = solve_conf_override
    else:
        raw_conf = (agent_conf.get("solvers") or {}).get(tier) or {}
        # Backward compatible: accept list configs, use the first entry.
        if isinstance(raw_conf, list):
            raw_conf = raw_conf[0] if raw_conf and isinstance(raw_conf[0], dict) else {}
        solve_conf = raw_conf if isinstance(raw_conf, dict) else {}

    prompt_path = Path(solve_conf.get("prompt_path") or "src/agenqa/prompts/solver.prompt")
    generator = _ensure_client_timeout(_resolve_solver_generator(solve_conf), timeout_seconds)
    agent_lang = str(((agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}).get("lang") or "").lower().strip() or None
    use_en = agent_lang in {"en", "english"}
    use_calc = has_zam_cal_marker(prompt_path)
    solver_prompt_text = (
        (SOLVER_PROMPT_CALC_EN if use_en else SOLVER_PROMPT_CALC)
        if use_calc
        else (SOLVER_PROMPT_EN if use_en else SOLVER_PROMPT)
    )
    use_py_prompt = True
    # 将解题产物写入当前 step 对应的目录（按最近一次决策的算子区分）
    # 注意：Init 只构造 episode_seed，实际 QA 是由后续 Extend 产出的，
    # 所以 init 应映射到 extend，保持 round 视图统一。
    op_raw = (state.last_decision.operation if state.last_decision else "extend")  # type: ignore[union-attr]
    op_lc = str(op_raw or "").lower()
    if "qa_init" in op_lc or "qainit" in op_lc:
        op_name = "qa_init"
    elif "init" in op_lc:
        # Init 后的第一道 QA 实际由 Extend 产出，映射到 extend
        op_name = "extend"
    elif "extend" in op_lc:
        op_name = "extend"
    elif "revise" in op_lc:
        op_name = "revise"
    else:
        op_name = op_lc.replace("-", "_") or "extend"
    try:
        step_idx = int(state.step)
    except Exception:
        step_idx = 0
    node_for_session = op_name
    if op_name == "revise":
        try:
            from agenqa.nodes.op_revise import _infer_revise_mode

            node_for_session = f"revise_{_infer_revise_mode(state)}"
        except Exception:
            node_for_session = "revise"
    generator = with_idealab_session_id(
        generator,
        idealab_session_id_for_step_node(state, node_for_session, step_idx),
    )
    round_idx = state.current_round_index()
    revise_mode = None
    if op_name == "revise":
        try:
            from agenqa.nodes.op_revise import _infer_revise_mode

            revise_mode = _infer_revise_mode(state)
        except Exception:
            revise_mode = None
    step_dir = compute_step_dir(state.artifacts_dir, op_name, step_idx, round_idx, revise_mode=revise_mode)
    step_dir.mkdir(parents=True, exist_ok=True)
    solve_dir = step_dir / "solve"
    solve_dir.mkdir(parents=True, exist_ok=True)

    # 1) 对 edge / path 这两个视角并发求解（同一 tier 内）
    def _run_edge() -> Tuple[Optional[bool], float | None, Path]:
        local_solver = SolverRunner(
            SolverConfig(
                prompt_path=prompt_path,
                prompt_text=solver_prompt_text,
                generator=generator,
                lang=agent_lang,
            )
        )
        kqa_path = dump_latest_kqa_jsonl(state, solve_dir / f"solve_{tier}_input.jsonl")
        out_path = solve_dir / f"solve_{tier}.jsonl"
        out_file = local_solver.run(kqa_path, out_path, append=False, concurrency=1)

        # 【新增】统一保存 Prompts 快照到 00_Prompts_Snapshot
        try:
            unified_prompt_dir = state.artifacts_dir / "00_Prompts_Snapshot"
            snapshot_prompt_used(
                local_solver.config.prompt_path,
                unified_prompt_dir,
                content=local_solver.prompt_template.template,
                logger=logger,
            )
        except Exception:
            pass

        ok_local: Optional[bool] = None
        diff_local: float | None = None
        try:
            ok_local, diff_local = _read_first_correct_diff(out_file)
        except Exception:
            ok_local, diff_local = None, None
        return ok_local, diff_local, out_file

    def _run_path() -> Path | None:
        if not enable_path:
            return None
        path_kqa: Path | None = None
        candidates: list[Path] = [
            step_dir / "path_kqa.jsonl",
            solve_dir / "path_kqa.jsonl",
        ]
        # Revise(correctness/reuse_hidden) 会把 path_kqa 快照落在 revise_{mode}/，
        # 但 solve 的 op_name 仍是 "revise"，导致在 round>=2 的布局下找不到输入文件。
        if op_name == "revise":
            try:
                from agenqa.nodes.op_revise import _infer_revise_mode

                revise_mode = _infer_revise_mode(state)
                alt_step_dir = compute_step_dir(
                    state.artifacts_dir,
                    "revise",
                    step_idx,
                    round_idx,
                    revise_mode=revise_mode,
                )
                candidates.extend(
                    [
                        alt_step_dir / "path_kqa.jsonl",
                        (alt_step_dir / "solve" / "path_kqa.jsonl"),
                    ]
                )
            except Exception:
                pass

        for cand in candidates:
            if cand.exists():
                path_kqa = cand
                break
        if path_kqa is None:
            return None
        try:
            if path_kqa.stat().st_size == 0:
                return None
        except Exception:
            pass
        try:
            local_solver = SolverRunner(
                SolverConfig(
                    prompt_path=prompt_path,
                    prompt_text=solver_prompt_text,
                    generator=generator,
                    lang=agent_lang,
                )
            )
            out_path = solve_dir / f"solve_path_{tier}.jsonl"
            local_solver.run(path_kqa, out_path, append=False, concurrency=1)
            return out_path
        except Exception:
            # path 求解失败不影响主链路
            return None

    ok: Optional[bool] = None
    diff: float | None = None
    out: Path = solve_dir / f"solve_{tier}.jsonl"
    path_out: Path | None = None

    if enable_path:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_edge = ex.submit(_run_edge)
            fut_path = ex.submit(_run_path)
            try:
                ok, diff, out = fut_edge.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Solve[%s]: edge KQA 求解失败，标记为 unavailable: %s", tier, str(exc))
                ok, diff = None, None
            # path 任务的异常已经在内部吞掉；这里只是确保完成
            try:
                path_out = fut_path.result()
            except Exception:
                pass
    else:
        try:
            ok, diff, out = _run_edge()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Solve[%s]: edge KQA 求解失败，标记为 unavailable: %s", tier, str(exc))
            ok, diff = None, None

    # 3) 若启用符号-only 模式，则追加一次基于 LLM 的表达式等价判定，并在成功时覆盖 ok/diff。
    qtype = _infer_question_type_for_solver(agent_conf, state)
    ok, diff = _apply_expression_judge(agent_conf, state, tier, out, ok, diff, question_type=qtype)

    # 4) path 在 symbolic_only 模式下也需要走同样的 judge 覆盖，
    #    否则 correct 会依赖 solver 自报或宽松的数值启发式，导致系统性偏高。
    if path_out and isinstance(path_out, Path) and path_out.exists():
        path_ok, path_diff = _read_first_correct_diff(path_out)
        _apply_expression_judge(agent_conf, state, tier, path_out, path_ok, path_diff, question_type=qtype)

    # 5) Dual evaluation views (no-contract vs with-contract) for auditability.
    # NOTE: This does NOT change judging behavior yet; it only produces separate artifacts.
    try:
        mem = KnownTree.normalize_memory(getattr(state, "memory", None))
        type2_ctx = extract_answer_contract_context(mem, step=step_idx)
    except Exception:
        type2_ctx = {}

    def _dump_dual(path_in: Path, *, prefix: str) -> None:
        try:
            row0 = None
            for r in read_jsonl(path_in, schema=None, max_lines=1):
                row0 = r if isinstance(r, dict) else None
                break
            if not isinstance(row0, dict):
                return
            out_no = solve_dir / f"{prefix}_{tier}_no_contract.jsonl"
            out_wc = solve_dir / f"{prefix}_{tier}_with_contract.jsonl"
            write_jsonl([row0], out_no, schema=None, append=False)
            row2 = dict(row0)
            row2["answer_contract"] = type2_ctx
            write_jsonl([row2], out_wc, schema=None, append=False)
        except Exception:
            return

    try:
        if out and isinstance(out, Path) and out.exists():
            _dump_dual(out, prefix="solve")
        if path_out and isinstance(path_out, Path) and path_out.exists():
            _dump_dual(path_out, prefix="solve_path")
    except Exception:
        pass

    return ok, diff, out, solve_dir, path_out


def _solve_tool_once(
    agent_conf: Dict[str, Any],
    state: AgentState,
    tier: str,
    solve_conf_override: Dict[str, Any] | None = None,
    *,
    enable_path: bool = True,
    timeout_seconds: int | None = None,
) -> Tuple[Optional[bool], float | None, Path, Path, Path | None]:
    """Run the tool-enabled solver tier (edge + optional path)."""
    solve_conf: Dict[str, Any]
    if solve_conf_override is not None:
        solve_conf = solve_conf_override
    else:
        raw_conf = (agent_conf.get("solvers") or {}).get(tier) or {}
        # Backward compatible: accept list conf but take first item.
        if isinstance(raw_conf, list):
            solve_conf = raw_conf[0] if raw_conf and isinstance(raw_conf[0], dict) else {}
        else:
            solve_conf = raw_conf if isinstance(raw_conf, dict) else {}

    # NOTE: tool-capable solver prompt is defined in `src/agenqa/prompts/solver_tool.py`.
    # `prompt_path` here is mainly used for prompt snapshot/debugging; for Numeric tool runs,
    # default to the solver_tool prompt unless the user explicitly points to a solver_tool variant.
    raw_prompt_path = solve_conf.get("prompt_path")
    if isinstance(raw_prompt_path, str) and raw_prompt_path.strip() and ("solver_tool" in raw_prompt_path):
        prompt_path = Path(raw_prompt_path.strip())
    else:
        prompt_path = Path("src/agenqa/prompts/solver_tool.py")
    generator = _ensure_client_timeout(_resolve_solver_generator(solve_conf), timeout_seconds)
    agent_lang = (
        str(((agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}).get("lang") or "")
        .lower()
        .strip()
        or None
    )
    use_en = agent_lang in {"en", "english"}
    solver_prompt_text = SOLVER_TOOL_PROMPT_EN if use_en else SOLVER_TOOL_PROMPT

    # 将解题产物写入当前 step 对应的目录（按最近一次决策的算子区分）
    op_raw = (state.last_decision.operation if state.last_decision else "extend")  # type: ignore[union-attr]
    op_lc = str(op_raw or "").lower()
    if "qa_init" in op_lc or "qainit" in op_lc:
        op_name = "qa_init"
    elif "init" in op_lc:
        op_name = "extend"
    elif "extend" in op_lc:
        op_name = "extend"
    elif "revise" in op_lc:
        op_name = "revise"
    else:
        op_name = op_lc.replace("-", "_") or "extend"
    try:
        step_idx = int(state.step)
    except Exception:
        step_idx = 0
    node_for_session = op_name
    if op_name == "revise":
        try:
            from agenqa.nodes.op_revise import _infer_revise_mode

            node_for_session = f"revise_{_infer_revise_mode(state)}"
        except Exception:
            node_for_session = "revise"
    generator = with_idealab_session_id(
        generator,
        idealab_session_id_for_step_node(state, node_for_session, step_idx),
    )
    round_idx = state.current_round_index()
    revise_mode = None
    if op_name == "revise":
        try:
            from agenqa.nodes.op_revise import _infer_revise_mode

            revise_mode = _infer_revise_mode(state)
        except Exception:
            revise_mode = None
    step_dir = compute_step_dir(state.artifacts_dir, op_name, step_idx, round_idx, revise_mode=revise_mode)
    step_dir.mkdir(parents=True, exist_ok=True)
    solve_dir = step_dir / "solve"
    solve_dir.mkdir(parents=True, exist_ok=True)

    tool_timeout = solve_conf.get("tool_timeout_seconds")
    tool_mem = solve_conf.get("tool_memory_limit_mb")
    tool_temp = solve_conf.get("tool_temp_dir")
    tool_python = solve_conf.get("tool_python_bin")

    def _run_edge() -> Tuple[Optional[bool], float | None, Path]:
        local_solver = SolverToolRunner(
            SolverToolConfig(
                prompt_path=prompt_path,
                prompt_text=solver_prompt_text,
                generator=generator,
                lang=agent_lang,
                timeout_seconds=float(tool_timeout) if isinstance(tool_timeout, (int, float)) else 10.0,
                memory_limit_mb=int(tool_mem) if isinstance(tool_mem, int) else 4096,
                temp_dir=str(tool_temp) if isinstance(tool_temp, str) and tool_temp.strip() else "/tmp",
                python_bin=str(tool_python) if isinstance(tool_python, str) and tool_python.strip() else "python",
            )
        )
        kqa_path = dump_latest_kqa_jsonl(state, solve_dir / f"solve_{tier}_input.jsonl")
        out_path = solve_dir / f"solve_{tier}.jsonl"
        out_file = local_solver.run(kqa_path, out_path, append=False)

        try:
            unified_prompt_dir = state.artifacts_dir / "00_Prompts_Snapshot"
            snapshot_prompt_used(
                prompt_path,
                unified_prompt_dir,
                content=local_solver.prompt_template.template,
                logger=logger,
            )
        except Exception:
            pass

        ok_local: Optional[bool] = None
        diff_local: float | None = None
        try:
            ok_local, diff_local = _read_first_correct_diff(out_file)
        except Exception:
            ok_local, diff_local = None, None
        return ok_local, diff_local, out_file

    def _run_path() -> Path | None:
        if not enable_path:
            return None
        path_kqa: Path | None = None
        candidates: list[Path] = [
            step_dir / "path_kqa.jsonl",
            solve_dir / "path_kqa.jsonl",
        ]
        if op_name == "revise":
            try:
                from agenqa.nodes.op_revise import _infer_revise_mode

                revise_mode = _infer_revise_mode(state)
                alt_step_dir = compute_step_dir(
                    state.artifacts_dir,
                    "revise",
                    step_idx,
                    round_idx,
                    revise_mode=revise_mode,
                )
                candidates.extend(
                    [
                        alt_step_dir / "path_kqa.jsonl",
                        (alt_step_dir / "solve" / "path_kqa.jsonl"),
                    ]
                )
            except Exception:
                pass
        for cand in candidates:
            if cand.exists():
                path_kqa = cand
                break
        if path_kqa is None:
            return None
        try:
            if path_kqa.stat().st_size == 0:
                return None
        except Exception:
            pass
        try:
            local_solver = SolverToolRunner(
                SolverToolConfig(
                    prompt_path=prompt_path,
                    prompt_text=solver_prompt_text,
                    generator=generator,
                    lang=agent_lang,
                    timeout_seconds=float(tool_timeout) if isinstance(tool_timeout, (int, float)) else 10.0,
                    memory_limit_mb=int(tool_mem) if isinstance(tool_mem, int) else 4096,
                    temp_dir=str(tool_temp) if isinstance(tool_temp, str) and tool_temp.strip() else "/tmp",
                    python_bin=str(tool_python) if isinstance(tool_python, str) and tool_python.strip() else "python",
                )
            )
            out_path = solve_dir / f"solve_path_{tier}.jsonl"
            local_solver.run(path_kqa, out_path, append=False)
            return out_path
        except Exception:
            return None

    ok: Optional[bool] = None
    diff: float | None = None
    out: Path = solve_dir / f"solve_{tier}.jsonl"
    path_out: Path | None = None

    if enable_path:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_edge = ex.submit(_run_edge)
            fut_path = ex.submit(_run_path)
            try:
                ok, diff, out = fut_edge.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("SolveTool[%s]: edge KQA 求解失败，标记为 unavailable: %s", tier, str(exc))
                ok, diff = None, None
            try:
                path_out = fut_path.result()
            except Exception:
                pass
    else:
        try:
            ok, diff, out = _run_edge()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SolveTool[%s]: edge KQA 求解失败，标记为 unavailable: %s", tier, str(exc))
            ok, diff = None, None

    # Dual evaluation views (no-contract vs with-contract) for auditability.
    try:
        mem = KnownTree.normalize_memory(getattr(state, "memory", None))
        type2_ctx = extract_answer_contract_context(mem, step=step_idx)
    except Exception:
        type2_ctx = {}

    def _dump_dual(path_in: Path, *, prefix: str) -> None:
        try:
            row0 = None
            for r in read_jsonl(path_in, schema=None, max_lines=1):
                row0 = r if isinstance(r, dict) else None
                break
            if not isinstance(row0, dict):
                return
            out_no = solve_dir / f"{prefix}_{tier}_no_contract.jsonl"
            out_wc = solve_dir / f"{prefix}_{tier}_with_contract.jsonl"
            write_jsonl([row0], out_no, schema=None, append=False)
            row2 = dict(row0)
            row2["answer_contract"] = type2_ctx
            write_jsonl([row2], out_wc, schema=None, append=False)
        except Exception:
            return

    try:
        if out and isinstance(out, Path) and out.exists():
            _dump_dual(out, prefix="solve")
        if path_out and isinstance(path_out, Path) and path_out.exists():
            _dump_dual(path_out, prefix="solve_path")
    except Exception:
        pass

    return ok, diff, out, solve_dir, path_out


def _read_solver_result_from_jsonl(path: Path | None, tier: str) -> SolverResult | None:
    """从 jsonl 文件读取完整的 solver 结果信息。

    统一的字段抽取和类型清洗逻辑，确保所有 solver 节点使用一致的数据结构。
    """
    if not path or not path.exists():
        return None
    try:
        for row in read_jsonl(path):
            extra: Dict[str, Any] = {}

            # Optional: capture tool usage signals when the row is produced by SolverToolRunner.
            tool = row.get("tool") if isinstance(row, dict) else None
            if isinstance(tool, dict):
                used = tool.get("used")
                if isinstance(used, bool):
                    extra["tool_used"] = used
                # keep a few diagnostic bits (small, no heavy payload)
                name = tool.get("name")
                if isinstance(name, str) and name.strip():
                    extra["tool_name"] = name.strip()
                exec_payload = tool.get("exec")
                if isinstance(exec_payload, dict):
                    ok = exec_payload.get("success")
                    if isinstance(ok, bool):
                        extra["tool_exec_success"] = ok
            solver_status = _normalize_solver_status(row if isinstance(row, dict) else None)
            if isinstance(solver_status, str) and solver_status.strip():
                extra["solver_status"] = solver_status.strip()
            solver_failure_code = row.get("solver_failure_code") if isinstance(row, dict) else None
            if isinstance(solver_failure_code, str) and solver_failure_code.strip():
                extra["solver_failure_code"] = solver_failure_code.strip()
            expr = row.get("expression_judge") if isinstance(row, dict) else None
            if isinstance(expr, dict):
                payload = expr.get(tier)
                if isinstance(payload, dict):
                    judge_status = payload.get("status")
                    if isinstance(judge_status, str) and judge_status.strip():
                        extra["judge_status"] = judge_status.strip()
                    judge_reason = payload.get("skip_reason") or payload.get("failure_reason")
                    if isinstance(judge_reason, str) and judge_reason.strip():
                        extra["judge_reason"] = judge_reason.strip()

            # 统一字段抽取和类型清洗
            correct = row.get("correct")
            if not isinstance(correct, bool):
                correct = None

            token_ratio = row.get("token_ratio") or row.get("difficulty")  # 向后兼容
            if token_ratio is not None and not isinstance(token_ratio, (int, float)):
                token_ratio = None

            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            kq_tokens = metrics.get("kq_tokens") if isinstance(metrics, dict) else None
            completion_tokens = metrics.get("completion_tokens") if isinstance(metrics, dict) else None
            try:
                kq_tokens = float(kq_tokens) if isinstance(kq_tokens, (int, float)) else None
            except Exception:
                kq_tokens = None
            try:
                completion_tokens = float(completion_tokens) if isinstance(completion_tokens, (int, float)) else None
            except Exception:
                completion_tokens = None

            model = row.get("model")
            if not isinstance(model, str):
                model = None

            service_id = row.get("service_id")
            if not isinstance(service_id, str):
                service_id = None

            question_well_posed = row.get("question_well_posed")
            if question_well_posed is not None and not isinstance(question_well_posed, bool):
                question_well_posed = bool(question_well_posed)

            correctness_feedback = row.get("correctness_feedback")
            if not isinstance(correctness_feedback, str):
                correctness_feedback = None

            difficulty_feedback = row.get("difficulty_feedback")
            if not isinstance(difficulty_feedback, str):
                difficulty_feedback = None

            question_feedback = row.get("question_feedback")
            if not isinstance(question_feedback, str):
                question_feedback = None

            harder_suggestion = row.get("harder_suggestion")
            if not isinstance(harder_suggestion, str):
                harder_suggestion = None

            # legacy 字段向新字段回填（便于后续逻辑统一只看 correctness/difficulty）
            if correctness_feedback is None and isinstance(question_feedback, str) and question_feedback.strip():
                correctness_feedback = question_feedback
            if difficulty_feedback is None and isinstance(harder_suggestion, str) and harder_suggestion.strip():
                difficulty_feedback = harder_suggestion

            return SolverResult(
                correct=correct,
                token_ratio=float(token_ratio) if token_ratio is not None else None,
                kq_tokens=kq_tokens,
                completion_tokens=completion_tokens,
                model=model,
                service_id=service_id,
                question_well_posed=question_well_posed,
                correctness_feedback=correctness_feedback,
                difficulty_feedback=difficulty_feedback,
                question_feedback=question_feedback,
                harder_suggestion=harder_suggestion,
                extra=extra,
            )
    except Exception:
        return None
    return None


def _update_solver_index_from_paths(
    state: AgentState,
    step_idx: int,
    round_idx: int,
    edge_solve_path: Path | None,
    path_solve_path: Path | None,
    tier: str,
) -> None:
    """统一的 helper 方法：从 jsonl 文件读取并更新 solver_index。

    所有 solver 节点（medium/strong/dual）都通过这个方法更新 solver_index，
    确保字段抽取、类型清洗的一致性。

    Args:
        state: AgentState 对象
        step_idx: QA 索引（step）
        round_idx: 轮次
        edge_solve_path: edge QA 的 solver 结果文件路径
        path_solve_path: path QA 的 solver 结果文件路径
        tier: "medium" 或 "strong"
    """
    try:
        # 更新 edge 结果
        edge_result = _read_solver_result_from_jsonl(edge_solve_path, tier)
        if edge_result:
            state.update_solver_index(step_idx, round_idx, "edge", tier, edge_result)
        # 更新 path 结果
        path_result = _read_solver_result_from_jsonl(path_solve_path, tier)
        if path_result:
            state.update_solver_index(step_idx, round_idx, "path", tier, path_result)
    except Exception:
        pass  # 更新失败不影响主流程


def solve_medium(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    qtype = _infer_question_type_for_solver(agent_conf, state)
    use_tool_prompt = (qtype == "Numeric") and (not _is_symbolic_only(agent_conf, question_type=qtype))
    if use_tool_prompt:
        ok, diff, path, solve_dir, ht_path = _solve_tool_once(
            agent_conf,
            state,
            "medium",
            enable_path=True,
            timeout_seconds=_solver_timeout_seconds(agent_conf),
        )
    else:
        ok, diff, path, solve_dir, ht_path = _solve_once(agent_conf, state, "medium", timeout_seconds=_solver_timeout_seconds(agent_conf))
    state.metrics.correct_medium = ok
    state.metrics.token_ratio_medium = diff
    logger.info("Solve[medium]: correct=%s diff=%s (%s)", ok, diff, str(path))

    # 更新 solver_index（使用统一的 helper 方法）
    try:
        step_idx = int(state.step or 0)
        round_idx = state.current_round_index()
        _update_solver_index_from_paths(state, step_idx, round_idx, path, ht_path, "medium")
    except Exception:
        pass  # 更新失败不影响主流程

    save_state(state)
    if output_manager:
        out_payload = path if path and path.exists() else None
        ht_payload = ht_path if ht_path and ht_path.exists() else None
        return NodeResult(
            state=state,
            step_idx=int(getattr(state, "step", 0) or 0),
            round_idx=state.current_round_index(),
            outputs={
                "solve_medium": out_payload,
                "solve_path_medium": ht_payload,
            },
            step_dir=solve_dir,
        )
    return state


def solve_strong(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    qtype = _infer_question_type_for_solver(agent_conf, state)
    use_tool_prompt = (qtype == "Numeric") and (not _is_symbolic_only(agent_conf, question_type=qtype))
    if use_tool_prompt:
        ok, diff, path, solve_dir, ht_path = _solve_tool_once(
            agent_conf,
            state,
            "strong",
            enable_path=True,
            timeout_seconds=_solver_timeout_seconds(agent_conf),
        )
    else:
        ok, diff, path, solve_dir, ht_path = _solve_once(agent_conf, state, "strong", timeout_seconds=_solver_timeout_seconds(agent_conf))
    logger.info("Solve[strong]: correct=%s diff=%s (%s)", ok, diff, str(path))
    # 每完成一次强求解，视为完成了一轮（director + operator + solve）
    try:
        prev_rounds = int(getattr(state, "rounds", 0) or 0)
    except Exception:
        prev_rounds = 0
    state.rounds = prev_rounds + 1

    # 更新 solver_index（使用统一的 helper 方法）
    try:
        step_idx = int(state.step or 0)
        round_idx = state.rounds  # 使用更新后的 rounds
        _update_solver_index_from_paths(state, step_idx, round_idx, path, ht_path, "strong")
    except Exception:
        pass  # 更新失败不影响主流程

    save_state(state)
    if output_manager:
        out_payload = path if path and path.exists() else None
        ht_payload = ht_path if ht_path and ht_path.exists() else None
        return NodeResult(
            state=state,
            step_idx=int(getattr(state, "step", 0) or 0),
            round_idx=state.rounds,
            outputs={
                "solve_strong": out_payload,
                "solve_path_strong": ht_payload,
            },
            step_dir=solve_dir,
        )
    return state


def solve_dual(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    """并行跑 medium + 所有 strong solvers。

    - 所有 strong solver 平权并行，`solvers.strong` 的配置顺序仅决定输出文件编号；
    - Numeric 题：medium/strong 使用同一套 solver_tool 协议（可选输出 ToolCode 并由系统执行验证）。
    """
    timeout_seconds = _solver_timeout_seconds(agent_conf)
    strong_confs = _normalize_solver_tier_configs(agent_conf, "strong")
    if not strong_confs:
        # Backward compatibility: accept legacy dict config.
        legacy = (agent_conf.get("solvers") or {}).get("strong")
        if isinstance(legacy, dict):
            strong_confs = [legacy]
    if not strong_confs:
        strong_confs = [{}]
    medium_confs = _normalize_solver_tier_configs(agent_conf, "medium")
    if not medium_confs:
        legacy = (agent_conf.get("solvers") or {}).get("medium")
        if isinstance(legacy, dict):
            medium_confs = [legacy]
    medium_conf = medium_confs[0] if medium_confs else {}

    qtype = _infer_question_type_for_solver(agent_conf, state)
    # Fail-fast: semantic/unified pipeline should not rely on programmatic judge.
    # Require LLM judge enabled either via consensus.answer_judge=llm or symbolic-only for this qtype.
    try:
        cons_block = _consensus_block(agent_conf)
        answer_judge = str(cons_block.get("answer_judge") or "").strip().lower()
    except Exception:
        answer_judge = ""
    if (answer_judge != "llm") and (not _is_symbolic_only(agent_conf, question_type=qtype)):
        raise ValueError(
            "Solve requires LLM judge for semantic pipeline. "
            "Set `consensus.answer_judge: llm` (recommended) or enable symbolic-only for this question type "
            "via `agent.symbolic_only` / `agent.allow_numeric_values: false` / `agent.symbolic_only_question_types`."
        )
    # NOTE: 取消独立 tool-tier：Numeric 时把“可选工具能力”并入 medium/strong（同一套 solver_tool 协议）。
    # 在符号-only 模式下不启用工具（由表达式等价 judge 负责判定）。
    use_tool_prompt = (qtype == "Numeric") and (not _is_symbolic_only(agent_conf, question_type=qtype))

    # 本轮 round_idx 在 solve 完成后才会写回 state.rounds；
    # 但所有求解产物必须落在同一个 round 目录下，所以不要提前更新 state.rounds。
    round_idx = state.current_round_index()

    results: Dict[str, Tuple[Optional[bool], Optional[float], Optional[Path], Optional[Path], Optional[Path]]] = {}

    def _safe_solve(
        label: str,
        conf: Dict[str, Any],
        *,
        enable_path: bool,
        use_tool_prompt_for_numeric: bool,
    ) -> Tuple[Optional[bool], Optional[float], Optional[Path], Optional[Path], Optional[Path]]:
        try:
            if use_tool_prompt_for_numeric:
                return _solve_tool_once(
                    agent_conf,
                    state,
                    label,
                    solve_conf_override=conf,
                    enable_path=enable_path,
                    timeout_seconds=timeout_seconds,
                )
            return _solve_once(
                agent_conf,
                state,
                label,
                solve_conf_override=conf,
                enable_path=enable_path,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Solve[%s] unavailable: solver_pipeline_failed: %s", label, str(exc))
            return None, None, None, None, None

    # 1) medium + all strongs 并行（Numeric 时两档都启用 solver_tool 协议）
    max_workers = max(1, len(strong_confs) + 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_safe_solve, "medium", medium_conf, enable_path=True, use_tool_prompt_for_numeric=use_tool_prompt): "medium",
        }
        for idx, conf in enumerate(strong_confs):
            label = f"strong_{idx}"
            future_map[
                executor.submit(
                    _safe_solve,
                    label,
                    conf,
                    enable_path=True,
                    use_tool_prompt_for_numeric=use_tool_prompt,
                )
            ] = label
        for fut in as_completed(future_map):
            tier = future_map[fut]
            try:
                results[tier] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Solve[%s] unavailable: solver_future_failed: %s", tier, str(exc))
                results[tier] = (None, None, None, None, None)

    m_ok, m_diff, m_path, m_solve_dir, m_ht_path = results.get("medium", (None, None, None, None, None))
    s_ok, s_diff, s_path, s_solve_dir, s_ht_path = results.get("strong_0", (None, None, None, None, None))

    state.metrics.correct_medium = m_ok
    state.metrics.token_ratio_medium = m_diff

    logger.info("Solve[medium]: correct=%s diff=%s (%s)", m_ok, m_diff, str(m_path))
    logger.info("Solve[strong_0]: correct=%s diff=%s (%s)", s_ok, s_diff, str(s_path))

    # 3) 回写 rounds（统一在本轮所有求解完成后）
    state.rounds = round_idx

    # 4) 更新 solver_index（使用统一 helper；strong tier 按 strong_{idx} 写入）
    try:
        step_idx = int(state.step or 0)
        _update_solver_index_from_paths(state, step_idx, round_idx, m_path, m_ht_path, "medium")
        for idx in range(len(strong_confs)):
            tier = f"strong_{idx}"
            _ok, _diff, edge_path, _solve_dir, path_out = results.get(tier, (False, None, None, None, None))
            _update_solver_index_from_paths(state, step_idx, round_idx, edge_path, path_out, tier)
    except Exception:
        pass  # 更新失败不影响主流程

    # 若当前 step 为 0，则将首题 QA‑Init 的求解结果在 01_qa_init 目录下也保留一份，便于浏览与后续处理。
    try:
        step_idx = int(state.step or 0)
    except Exception:
        step_idx = 0
    if step_idx == 0:
        try:
            qa_init_dir = state.artifacts_dir / "01_qa_init"
            if qa_init_dir.exists():
                src_dir = m_solve_dir or s_solve_dir
                if src_dir and src_dir.exists():
                    qa_init_solve_dir = qa_init_dir / "solve"
                    qa_init_solve_dir.mkdir(parents=True, exist_ok=True)
                    for pattern in [
                        "solve_medium*.jsonl",
                        "solve_path_medium*.jsonl",
                        "solve_strong*.jsonl",
                        "solve_path_strong*.jsonl",
                    ]:
                        for src in src_dir.glob(pattern):
                            if not src.is_file():
                                continue
                            dst = qa_init_solve_dir / src.name
                            try:
                                shutil.copy2(src, dst)
                            except Exception:
                                pass
        except Exception:
            # 目录组织问题不影响主流程
            pass
    save_state(state)
    if output_manager:
        use_dir = m_solve_dir or s_solve_dir
        m_payload = m_path if m_path and m_path.exists() else None
        s_payload = s_path if s_path and s_path.exists() else None
        m_ht_payload = m_ht_path if m_ht_path and isinstance(m_ht_path, Path) and m_ht_path.exists() else None
        s_ht_payload = s_ht_path if s_ht_path and isinstance(s_ht_path, Path) and s_ht_path.exists() else None
        return NodeResult(
            state=state,
            step_idx=int(getattr(state, "step", 0) or 0),
            round_idx=state.rounds,
            outputs={
                "solve_medium": m_payload,
                "solve_strong": s_payload,
                "solve_path_medium": m_ht_payload,
                "solve_path_strong": s_ht_payload,
            },
            step_dir=use_dir,
        )
    return state
