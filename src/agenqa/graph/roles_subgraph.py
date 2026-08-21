from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agenqa.graph.state import AgentState
from agenqa.nodes.roles_nodes import (
    episode_seed_builder_node,
    known_init_seed_node,
    extend_draft_node,
    extend_format_node,
    revise_diagnose_node,
    revise_draft_node,
    revise_format_node,
)

logger = logging.getLogger(__name__)


def run_extend_graph(agent_conf: Dict[str, Any], state: AgentState) -> tuple[AgentState, Dict[str, Any], Path, int, int]:
    """顺序执行 Extend Draft + Format（供算子或测试直接调用）。"""
    state, ext_ctx = extend_draft_node(agent_conf, state)
    state, role_outputs, step_dir, step_idx, round_idx = extend_format_node(agent_conf, state, ext_ctx)
    return state, role_outputs, step_dir, step_idx, round_idx


def run_revise_graph(agent_conf: Dict[str, Any], state: AgentState) -> tuple[AgentState, Dict[str, Any], Path, int, int]:
    """顺序执行 Revise Diagnose + Draft + Format。"""
    state, rev_ctx = revise_diagnose_node(agent_conf, state)
    rev_ctx = revise_draft_node(agent_conf, state, rev_ctx)
    state, role_outputs, step_dir, step_idx, round_idx = revise_format_node(agent_conf, state, rev_ctx)
    return state, role_outputs, step_dir, step_idx, round_idx


def run_known_init_graph(agent_conf: Dict[str, Any], state: AgentState) -> tuple[AgentState, Dict[str, Any], Path, int, int]:
    """顺序执行 Known-Init EpisodeSeedBuilder + SeedInit（仅生成 episode_seed，不产 QA）。"""
    qa_ctx = episode_seed_builder_node(agent_conf, state)
    state, role_outputs, step_dir, step_idx, round_idx = known_init_seed_node(agent_conf, state, qa_ctx)
    return state, role_outputs, step_dir, step_idx, round_idx


def run_known_init_and_first_qa_graph(agent_conf: Dict[str, Any], state: AgentState) -> tuple[AgentState, Dict[str, Any], Path, int, int]:
    """顺序执行 Known-Init + Extend（生成 episode_seed 后立即出第一道 QA）。"""
    qa_ctx = episode_seed_builder_node(agent_conf, state)
    state, known_init_role_outputs, step_dir, step_idx, round_idx = known_init_seed_node(agent_conf, state, qa_ctx)
    # 直接调用 extend 出第一道题
    state, extend_role_outputs, step_dir, step_idx, round_idx = run_extend_graph(agent_conf, state)
    # 合并所有角色的输出
    merged_role_outputs = {**(known_init_role_outputs or {}), **(extend_role_outputs or {})}
    return state, merged_role_outputs, step_dir, step_idx, round_idx


__all__ = ["run_extend_graph", "run_revise_graph", "run_known_init_graph", "run_known_init_and_first_qa_graph"]
