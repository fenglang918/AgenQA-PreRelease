"""Known-Init Operator: construct episode_seed only (no QA generation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import logging

from agenqa.graph.state import AgentState
from agenqa.graph.output_manager import OutputContext, compute_step_dir
from agenqa.graph.roles_subgraph import run_known_init_graph
from agenqa.domain.node_result import NodeResult
from agenqa.memory.store import dump_director_decision_for_step
from agenqa.domain.known_tree import KnownTree

logger = logging.getLogger(__name__)


def run_known_init(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    """Construct episode_seed via PaperBrief → EpisodeSeedBuilder → SeedInit.

    Only runs when there is no history and episode_seed is empty; otherwise returns state.
    """
    if state.history:
        logger.warning("run_known_init called with non-empty history; skip.")
        return state
    memory = KnownTree.normalize_memory(getattr(state, "memory", None))
    seed = memory.get("episode_seed") or {}
    has_seed = False
    if isinstance(seed, dict):
        anchor = seed.get("anchor")
        if isinstance(anchor, str) and anchor.strip():
            has_seed = True
        else:
            subject = seed.get("subject")
            keywords = seed.get("keywords")
            if isinstance(subject, str) and subject.strip():
                has_seed = True
            elif isinstance(keywords, list) and any(str(k).strip() for k in keywords):
                has_seed = True
    if has_seed:
        logger.warning("run_known_init called with existing episode_seed; skip.")
        return state

    step_idx = 0
    round_idx = state.current_round_index()
    if output_manager:
        ctx: OutputContext = output_manager.begin("init", step_idx, round_idx)
        step_dir = ctx.step_dir
        ctx.dump_director_decision(state, step_idx)
    else:
        step_dir = compute_step_dir(state.artifacts_dir, "init", step_idx, round_idx)
        step_dir.mkdir(parents=True, exist_ok=True)
        dump_director_decision_for_step(state, step_dir, step_idx)

    try:
        state, role_outputs, step_dir, step_idx, round_idx = run_known_init_graph(agent_conf, state)
    except Exception as exc:  # noqa: BLE001
        logger.error("Known-Init failed: %s", str(exc))
        raise

    if output_manager:
        return NodeResult(
            state=state,
            step_idx=step_idx,
            round_idx=round_idx,
            role_outputs=role_outputs,
            step_dir=step_dir,
        )
    return state


__all__ = ["run_known_init"]
