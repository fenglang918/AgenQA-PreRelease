"""LangGraph 图构建与执行入口。"""

from __future__ import annotations

from typing import Any, Dict

from agenqa.domain.node_result import NodeResult
from agenqa.graph.output_manager import OutputManager
from agenqa.graph.state import AgentState
from agenqa.graph.track_adapter import get_track_adapter
from agenqa.memory.store import save_state

# 标准化的操作 → 节点路由映射（统一生命周期；不保留历史兼容别名）
OPERATION_ROUTES = {
    "init": "init",
    "extend": "extend",
    "revise": "revise",
    "finish": "finish",
}


def _normalize_operation_key(raw: Any) -> str:
    key = str(raw or "").strip().lower()
    return key.replace("-", "").replace("_", "").replace(" ", "")


def require_langgraph():
    try:
        from langgraph.graph import StateGraph, END  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "未检测到 langgraph，请安装项目依赖：\n"
            "python -m pip install -e ."
        ) from e


def build_graph(agent_conf: Dict[str, Any], output_manager: OutputManager | None = None):
    # 延迟导入节点，避免加载时的循环引用
    from agenqa.nodes.director import decide_next, summarize_state
    from agenqa.nodes.evaluators.final_commenter import final_commenter_node
    from agenqa.nodes.judge import judge_node, route_after_judge
    from langgraph.graph import StateGraph, END

    sg = StateGraph(AgentState)
    adapter = get_track_adapter(agent_conf)

    def _handle_result(node_key: str, result: Any) -> AgentState:
        if isinstance(result, NodeResult):
            try:
                specs = adapter.specs_for(node_key)
                roles = adapter.roles_for(node_key)
                if output_manager:
                    output_manager.save_result(node_key, result, specs=specs or [], roles=roles or [])
            except Exception:
                # 输出失败不影响主流程
                pass
            return result.state
        return result

    # 包装节点函数以匹配 LangGraph 签名：fn(state) -> state
    def node_director(state: AgentState) -> AgentState:
        decision = decide_next(agent_conf, state)
        # Fail-fast: compress_history 已从主链路移除；若 Director 仍产出该操作，直接报错暴露问题。
        op_key = _normalize_operation_key(getattr(decision, "operation", ""))
        if "compress" in op_key or op_key in {"historyrefactor"}:
            raise RuntimeError(
                "Operation 'compress_history' has been removed from the pipeline. "
                "Please update Director prompt/config to stop producing compress/history-compress operations."
            )
        # 将 solver 相关的上下文附加到决策 params，便于算子与记录使用。
        # 这里保留 multi-strong 派生的 compact feedback（非单一 primary 口径），
        # 同时继续暴露 per-solver 结构化结果供下游精细消费。
        try:
            summary = summarize_state(state, agent_conf)
            solver_context = {
                "metrics": summary.get("metrics"),
                "solver_metrics": summary.get("solver_metrics"),
                "solver_feedback": summary.get("solver_feedback"),
                "type1_ambiguity": summary.get("type1_ambiguity"),
                "type2_contract": summary.get("type2_contract"),
            }
            if decision.params is None:
                decision.params = {}
            if isinstance(decision.params, dict):
                decision.params.setdefault("solver_context", solver_context)
        except Exception:
            pass
        state.last_decision = decision
        try:
            op_norm = str(getattr(decision, "operation", "") or "").strip().lower()
            if op_norm == "revise":
                state.consecutive_revise = int(getattr(state, "consecutive_revise", 0) or 0) + 1
            else:
                state.consecutive_revise = 0

            # Type2: bounded consecutive revise for answer_contract only.
            try:
                params = getattr(decision, "params", None)
                revise_mode = params.get("revise_mode") if isinstance(params, dict) else None
                revise_mode = str(revise_mode or "").strip().lower()
            except Exception:
                revise_mode = ""
            if op_norm == "revise" and revise_mode == "answer_contract":
                state.consecutive_answer_contract_revise = int(getattr(state, "consecutive_answer_contract_revise", 0) or 0) + 1
            else:
                state.consecutive_answer_contract_revise = 0
        except Exception:
            pass
        save_state(state)
        # Ensure the Director decision is always persisted to disk for this round.
        # For Finish decisions there may be no operator step dir to carry the snapshot,
        # so we write it under the canonical director directory: round_{r}/director/.
        try:
            from agenqa.graph.output_manager import compute_step_dir
            from agenqa.memory.store import dump_director_decision_for_step

            round_idx = state.current_round_index()
            step_idx = int(getattr(state, "step", 0) or 0)
            director_dir = compute_step_dir(state.artifacts_dir, "director", step_idx, round_idx)
            dump_director_decision_for_step(state, director_dir, step_idx)
        except Exception:
            pass
        return state

    def node_entry(state: AgentState) -> AgentState:
        """入口节点：允许做一次性状态初始化（不能放在 route_entry 里）。"""
        if adapter.needs_init(state):
            state.last_decision = adapter.build_init_decision()
            save_state(state)
        return state

    def node_init(state: AgentState) -> AgentState:
        res = adapter.run_init(agent_conf, state, output_manager=output_manager)
        state = _handle_result("init", res)
        # Init 完成后，如果是 round_1 且需要自动进入 Extend，创建 Extend 决策
        try:
            if adapter.should_auto_extend_after_init(state):
                state.last_decision = adapter.build_extend_decision_after_init()
                save_state(state)
        except Exception:
            pass
        return state

    def node_extend(state: AgentState) -> AgentState:
        res = adapter.run_extend(agent_conf, state, output_manager=output_manager)
        return _handle_result("extend", res)

    def node_revise(state: AgentState) -> AgentState:
        res = adapter.run_revise(agent_conf, state, output_manager=output_manager)
        return _handle_result("revise", res)

    def node_solve(state: AgentState) -> AgentState:
        res = adapter.run_solve(agent_conf, state, output_manager=output_manager)
        return _handle_result("solve", res)

    def node_consensus(state: AgentState) -> AgentState:
        res = adapter.run_consensus(agent_conf, state)
        return _handle_result("consensus", res)

    def node_judge(state: AgentState) -> AgentState:
        # NOTE: Do not mutate state in conditional-edge routing functions.
        state = judge_node(agent_conf, state)
        save_state(state)
        return state

    def node_final_commenter(state: AgentState) -> AgentState:
        return final_commenter_node(agent_conf, state)

    def route_after_director(state: AgentState) -> str:
        op = _normalize_operation_key(state.last_decision.operation if state.last_decision else "Extend")
        target = OPERATION_ROUTES.get(op)
        if target:
            return target
        raise RuntimeError(f"Unsupported operation: {getattr(state.last_decision, 'operation', None)!r} (op_key={op!r})")

    def route_after_judge_node(state: AgentState) -> str:
        return route_after_judge(state)

    def route_entry(state: AgentState) -> str:
        return "init" if adapter.needs_init(state) else "director"

    # 注册节点
    sg.add_node("entry", node_entry)
    sg.add_node("director", node_director)
    sg.add_node("init", node_init)
    sg.add_node("extend", node_extend)
    sg.add_node("revise", node_revise)
    sg.add_node("solve", node_solve)
    sg.add_node("consensus", node_consensus)
    sg.add_node("judge", node_judge)
    sg.add_node("final_commenter", node_final_commenter)

    sg.set_entry_point("entry")
    # 入口节点根据状态路由到 init 或 director
    sg.add_conditional_edges(
        "entry",
        route_entry,
        {
            "init": "init",
            "director": "director",
        },
    )
    # director 路由到 init / extend / revise 或 finish
    sg.add_conditional_edges(
        "director",
        route_after_director,
        {
            "init": "init",
            "extend": "extend",
            "revise": "revise",
            "finish": "final_commenter",
        },
    )
    # Init 完成后进入 Extend 产出第一道题 / 第一步
    sg.add_edge("init", "extend")
    # Extend/Revise 完成后进入 Solve
    sg.add_edge("extend", "solve")
    sg.add_edge("revise", "solve")
    # Solve -> Consensus -> Judge
    sg.add_edge("solve", "consensus")
    sg.add_edge("consensus", "judge")
    sg.add_conditional_edges("judge", route_after_judge_node, {"continue": "director", "finish": "final_commenter"})
    sg.add_edge("final_commenter", END)

    return sg.compile()


class GraphBuilder:
    """Thin wrapper around the current LangGraph builder."""

    def __init__(self, agent_conf: Dict[str, Any], output_manager: OutputManager | None = None) -> None:
        self.agent_conf = agent_conf
        self.output_manager = output_manager

    def build(self) -> Any:
        return build_graph(self.agent_conf, output_manager=self.output_manager)
