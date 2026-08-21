"""Judge 节点：融合中/强求解结果，决定是否继续或终止。"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from agenqa.graph.state import AgentState, Decision
from agenqa.memory.store import dump_director_decision_for_step, save_state


logger = logging.getLogger(__name__)


def _edge_strong_status(state: AgentState) -> tuple[bool, bool]:
    """Return (has_any_incorrect, all_correct) for latest edge strong results."""
    try:
        step_idx = int(getattr(state, "step", 0) or 0)
    except Exception:
        step_idx = 0
    rows = state.get_latest_solver_results(step_idx, "edge", "strong")
    explicit = [res.correct for _tier, res in rows if res.correct is not None]
    if not explicit:
        return False, False
    has_any_incorrect = any(v is False for v in explicit)
    all_correct = all(v is True for v in explicit)
    return has_any_incorrect, all_correct


def judge_node(agent_conf: Dict[str, Any], state: AgentState) -> AgentState:
    """Judge 节点：设置 stop_reason（会持久化到主 state）。"""
    max_steps = int(agent_conf.get("agent", {}).get("max_steps", state.max_steps))
    # 轮数上限：若未显式配置，则默认与 max_steps 一致
    max_rounds = int(agent_conf.get("agent", {}).get("max_rounds", max_steps))
    # 是否在 episode 结束时触发一次 Director 总结（默认关闭）
    enable_final_summary = bool((agent_conf.get("agent", {}) or {}).get("enable_final_summary", False))
    try:
        curr_rounds = int(getattr(state, "rounds", 0) or 0)
    except Exception:
        curr_rounds = 0
    if state.stop_reason:
        logger.info("Judge: stop due to reason=%s", state.stop_reason)
        return state
    # 简单策略：
    # - 已到达 max_rounds：终止
    # - 已到达 max_steps：终止
    # - 若强/中均错误且未到步数上限：交回 Director 继续决策
    # - 若中等正确但强错误：交回 Director 继续决策
    # - 否则根据强/中求解质量决定继续或终止
    def _run_final_director_summary(stop_tag: str) -> None:
        """在到达终止条件时运行一次 Director 做总结（不触发后续算子）。"""
        if not enable_final_summary:
            return
        try:
            from agenqa.nodes.director import decide_next

            decision: Decision = decide_next(agent_conf, state)
            state.last_decision = decision
            # 记录到独立目录，避免覆盖现有 step 产物
            summary_dir = Path(state.artifacts_dir) / f"final_summary_step_{state.step}"
            summary_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "stop_reason": stop_tag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision": {
                    "operation": decision.operation,
                    "reason": decision.reason,
                    "params": decision.params or {},
                },
                "step": state.step,
                "rounds": curr_rounds,
            }
            (summary_dir / "director_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                # 也保存一份标准化的 director_decision 便于回放（步号用当前 step）
                dump_director_decision_for_step(state, summary_dir, int(state.step or 0), filename="director_decision.json")
            except Exception:
                pass
            save_state(state)
        except Exception as exc:
            logger.warning("Final director summary failed: %s", str(exc))

    if curr_rounds >= max_rounds:
        state.stop_reason = f"reach_max_rounds({max_rounds})"
        logger.info("Judge: reach max rounds, finish")
        _run_final_director_summary(state.stop_reason)
        return state

    if (state.step or 0) >= (max_steps - 1):
        state.stop_reason = f"reach_max_steps({max_steps})"
        logger.info("Judge: reach max steps, finish")
        _run_final_director_summary(state.stop_reason)
        return state

    edge_has_any_incorrect, edge_all_correct = _edge_strong_status(state)
    both_wrong = (state.metrics.correct_medium is False) and edge_has_any_incorrect
    if both_wrong:
        # 在尚未达到步数上限前，不再强制终止，而是将“中/强均错”的信息交给 Director 决策后续操作。
        logger.info("Judge: both wrong but below max_steps, continue")
        return state

    # 若中等求解正确但强求解失败，也交回 Director 继续决策（不再硬终止）
    if state.metrics.correct_medium is True and edge_has_any_incorrect:
        logger.info("Judge: medium correct but strong wrong, continue")
        return state

    logger.info("Judge: continue")
    return state


def route_after_judge(state: AgentState) -> str:
    """纯路由函数：只读取 stop_reason。"""
    return "finish" if state.stop_reason else "continue"


def judge_and_route(agent_conf: Dict[str, Any], state: AgentState) -> str:
    """Deprecated：请使用 judge_node + route_after_judge（避免在路由函数内写 state）。"""
    state = judge_node(agent_conf, state)
    return route_after_judge(state)
