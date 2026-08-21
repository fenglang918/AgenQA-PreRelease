"""Episode runner for AgenQA graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import logging
import os
from uuid import uuid4

from agenqa.graph.builder import build_graph, require_langgraph
from agenqa.graph.state import AgentState
from agenqa.graph.output_manager import OutputManager
from agenqa.memory.store import save_state

logger = logging.getLogger(__name__)


def run_episode(agent_conf: Dict[str, Any], run_id: str, output_dir: Path, state: AgentState | None = None) -> AgentState:
    require_langgraph()
    from langgraph.errors import GraphRecursionError

    output_manager = OutputManager(output_dir)
    graph = build_graph(agent_conf, output_manager=output_manager)
    if state is None:
        agent_block = agent_conf.get("agent", {}) or {}
        fold_variant = str(agent_block.get("path_kqa_variant") or agent_block.get("path_fold_variant") or "").strip().lower()
        if fold_variant not in {"direct", "scaffolded"}:
            fold_variant = "direct"
        state = AgentState(
            run_id=run_id,
            artifacts_dir=output_dir,
            max_steps=int(agent_block.get("max_steps", 3)),
            path_kqa_variant=fold_variant,
        )
    else:
        # 覆盖 artifacts_dir 以防配置变动
        state.artifacts_dir = output_dir
        # 覆盖 max_steps（优先配置）
        try:
            state.max_steps = int(agent_conf.get("agent", {}).get("max_steps", state.max_steps))
        except Exception:
            pass
        # 覆盖 path_kqa variant（优先配置）
        try:
            agent_block = agent_conf.get("agent", {}) or {}
            fold_variant = str(agent_block.get("path_kqa_variant") or agent_block.get("path_fold_variant") or "").strip().lower()
            if fold_variant in {"direct", "scaffolded"}:
                state.path_kqa_variant = fold_variant
        except Exception:
            pass
    # 将初始状态快照
    save_state(state)
    # 执行图
    thread_id = os.environ.get("SCICLONE_THREAD_ID") or str(uuid4())
    # 为避免 LangGraph 默认递归上限过早触发，允许根据 max_steps/max_rounds 适度放宽 recursion_limit
    try:
        agent_cfg = agent_conf.get("agent") or {}
        max_steps = int(agent_cfg.get("max_steps", state.max_steps))
        max_rounds = int(agent_cfg.get("max_rounds", max_steps))
        # 一次 round 往往包含多个节点（director + operator + solve + consensus + judge + ...）。
        # 以 max_rounds/max_steps 估算一个保守上限，避免 LangGraph 默认 64 过早触发。
        computed = (max_rounds * 30) + (max_steps * 10) + 50
        recursion_limit_override = agent_cfg.get("recursion_limit", None)
        recursion_limit = int(recursion_limit_override if recursion_limit_override is not None else max(256, computed))
        logger.info(
            "LangGraph recursion_limit=%s (override=%s, computed=%s, max_rounds=%s, max_steps=%s)",
            recursion_limit,
            recursion_limit_override,
            computed,
            max_rounds,
            max_steps,
        )
    except Exception:
        recursion_limit = 64
        logger.warning("Failed to compute recursion_limit; fallback to %s", recursion_limit, exc_info=True)
    try:
        result = graph.invoke(
            state,
            config={"recursion_limit": recursion_limit, "configurable": {"thread_id": thread_id}},
        )
    except GraphRecursionError:
        state.stop_reason = f"graph_recursion_limit({recursion_limit})"
        save_state(state)
        logger.error("Graph recursion limit reached (limit=%s); stop_reason persisted.", recursion_limit)
        return state
    # LangGraph 可能返回原始 dict；将其规整为 AgentState
    if isinstance(result, AgentState):
        final_state = result
    else:
        # 合并回原始 state
        data = result if isinstance(result, dict) else {}
        state.update_from_mapping(data)
        final_state = state
    save_state(final_state)
    return final_state
