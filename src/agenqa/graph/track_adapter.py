"""Track-specific adapters for the unified graph lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

from agenqa.domain.known_tree import KnownTree
from agenqa.domain.node_result import OutputSpec
from agenqa.graph.state import AgentState, Decision


# ---- Defaults (per track, per unified node) ----

_COMMON_SOLVE_SPECS: list[OutputSpec] = [
    OutputSpec("solve_medium", "jsonl"),
    OutputSpec("solve_strong", "jsonl"),
    OutputSpec("solve_path_medium", "jsonl", required=False),
    OutputSpec("solve_path_strong", "jsonl", required=False),
]

DEFAULT_ROLES_BY_TRACK: dict[str, dict[str, list[str]]] = {
    "semantic": {
        "init": ["episode_seed_builder", "seed_init"],
        "extend": ["draft_chain", "format", "numeric_oracle", "step_cert_builder", "path_fold", "format_validation"],
        "revise": ["diagnose", "draft_chain", "format", "numeric_oracle", "step_cert_builder", "path_fold", "format_validation"],
    },
    "unified": {
        "init": ["episode_seed_builder", "seed_init"],
        "extend": ["draft_chain", "format", "numeric_oracle", "step_cert_builder", "path_fold", "format_validation"],
        "revise": ["diagnose", "draft_chain", "format", "numeric_oracle", "step_cert_builder", "path_fold", "format_validation"],
    },
    "executable": {
        # init may optionally run ExecutableExtract to materialize task_sketch into executable_seed
        "init": ["paper_brief", "executable_extract"],
        "extend": [
            "executable_extract",
            "executable_draft_step",
            "executable_step_cert_builder",
            "executable_test_inputs",
            "executable_path_fold",
        ],
        "revise": [
            "executable_diagnose",
            "executable_revise_step",
            "executable_step_cert_builder",
            "executable_test_inputs",
            "executable_path_fold",
        ],
    },
}

DEFAULT_SPECS_BY_TRACK: dict[str, dict[str, list[OutputSpec]]] = {
    "semantic": {
        "init": [],
        "extend": [OutputSpec("edge_kqa", "jsonl", required=False), OutputSpec("path_kqa", "jsonl", required=False)],
        "revise": [OutputSpec("edge_kqa", "jsonl", required=False), OutputSpec("path_kqa", "jsonl", required=False)],
        "solve": list(_COMMON_SOLVE_SPECS),
        "consensus": [],
        "judge": [],
    },
    "unified": {
        "init": [],
        "extend": [OutputSpec("edge_kqa", "jsonl", required=False), OutputSpec("path_kqa", "jsonl", required=False)],
        "revise": [OutputSpec("edge_kqa", "jsonl", required=False), OutputSpec("path_kqa", "jsonl", required=False)],
        "solve": list(_COMMON_SOLVE_SPECS),
        "consensus": [],
        "judge": [],
    },
    "executable": {
        "init": [],
        "extend": [
            OutputSpec("edge_executable", "jsonl", required=False),
            OutputSpec("path_executable", "jsonl", required=False),
        ],
        "revise": [
            OutputSpec("edge_executable", "jsonl", required=False),
            OutputSpec("path_executable", "jsonl", required=False),
        ],
        "solve": list(_COMMON_SOLVE_SPECS),
        "consensus": [],
        "judge": [],
    },
}


@dataclass(frozen=True)
class TrackAdapter:
    name: str
    roles_by_node: dict[str, list[str]]
    specs_by_node: dict[str, list[OutputSpec]]
    has_seed: Callable[[AgentState], bool]
    has_history: Callable[[AgentState], bool]
    run_init: Callable[[Dict[str, Any], AgentState, Any | None], AgentState | Any]
    run_extend: Callable[[Dict[str, Any], AgentState, Any | None], AgentState | Any]
    run_revise: Callable[[Dict[str, Any], AgentState, Any | None], AgentState | Any]
    run_solve: Callable[[Dict[str, Any], AgentState, Any | None], AgentState | Any]
    run_consensus: Callable[[Dict[str, Any], AgentState], AgentState]
    init_reason: str
    init_operator_notes: str
    extend_reason_after_init: str
    extend_operator_notes_after_init: str

    def needs_init(self, state: AgentState) -> bool:
        try:
            round_idx = state.current_round_index()
            is_first_round = round_idx == 1
        except Exception:
            return False
        if not is_first_round:
            return False
        return (not self.has_history(state)) and (not self.has_seed(state))

    def should_auto_extend_after_init(self, state: AgentState) -> bool:
        try:
            round_idx = state.current_round_index()
            is_first_round = round_idx == 1
        except Exception:
            return False
        if not is_first_round:
            return False
        return (not self.has_history(state)) and self.has_seed(state)

    def build_init_decision(self) -> Decision:
        return Decision(
            operation="Init",
            reason=self.init_reason,
            params={"operator_notes": self.init_operator_notes},
        )

    def build_extend_decision_after_init(self) -> Decision:
        return Decision(
            operation="Extend",
            reason=self.extend_reason_after_init,
            params={"operator_notes": self.extend_operator_notes_after_init},
        )

    def roles_for(self, node: str) -> Sequence[str]:
        return self.roles_by_node.get(node, [])

    def specs_for(self, node: str) -> Sequence[OutputSpec]:
        return self.specs_by_node.get(node, [])


def _has_semantic_seed(state: AgentState) -> bool:
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


def _has_executable_seed(state: AgentState) -> bool:
    memory = KnownTree.normalize_memory(getattr(state, "memory", None))
    seed = memory.get("executable_seed")
    if not isinstance(seed, dict):
        return False
    pid = seed.get("problem_id")
    return isinstance(pid, str) and bool(pid.strip())


def _has_semantic_history(state: AgentState) -> bool:
    try:
        return any(True for _ in state.iter_semantic_records())
    except Exception:
        return bool(getattr(state, "history", None))


def _has_executable_history(state: AgentState) -> bool:
    try:
        return state.latest_executable_record() is not None
    except Exception:
        return False


def _build_semantic_adapter() -> TrackAdapter:
    from agenqa.nodes.op_known_init import run_known_init
    from agenqa.nodes.op_extend import run_extend
    from agenqa.nodes.op_revise import run_revise
    from agenqa.nodes.evaluators.solve import solve_dual
    from agenqa.nodes.evaluators.consensus import compute_consensus

    return TrackAdapter(
        name="semantic",
        roles_by_node=DEFAULT_ROLES_BY_TRACK["semantic"],
        specs_by_node=DEFAULT_SPECS_BY_TRACK["semantic"],
        has_seed=_has_semantic_seed,
        has_history=_has_semantic_history,
        run_init=run_known_init,
        run_extend=run_extend,
        run_revise=run_revise,
        run_solve=solve_dual,
        run_consensus=compute_consensus,
        init_reason="round_1 固定流程：初始化 Known 结构（episode_seed）作为整条链路的起点。",
        init_operator_notes=(
            "请从输入材料中构造一个稳定的 episode_seed（字段以 contract 为准；建议提供可派生稳定话题锚定的信息），"
            "确保信息充分但不冗余，能够支撑多步推理题目的生成。"
        ),
        extend_reason_after_init=(
            "episode_seed 已通过 Init 初始化完成，当前 history 为空，需要基于 seed 生成第一道题目（step=1），开启多步推理链条。"
        ),
        extend_operator_notes_after_init=(
            "基于 episode_seed 生成第一道题目，确保题目能够充分利用背景锚定信息，"
            "为后续扩展题目奠定基础。题目应具有清晰的推理步骤，适合作为多步推理链的起点。"
        ),
    )


def _build_unified_adapter() -> TrackAdapter:
    from agenqa.nodes.op_known_init import run_known_init
    from agenqa.nodes.op_extend import run_extend
    from agenqa.nodes.op_revise import run_revise
    from agenqa.nodes.evaluators.solve import solve_dual
    from agenqa.nodes.evaluators.consensus import compute_consensus

    return TrackAdapter(
        name="unified",
        roles_by_node=DEFAULT_ROLES_BY_TRACK["unified"],
        specs_by_node=DEFAULT_SPECS_BY_TRACK["unified"],
        has_seed=_has_semantic_seed,
        has_history=_has_semantic_history,
        run_init=run_known_init,
        run_extend=run_extend,
        run_revise=run_revise,
        run_solve=solve_dual,
        run_consensus=compute_consensus,
        init_reason="round_1 固定流程：初始化 Known 结构（episode_seed）作为整条链路的起点。",
        init_operator_notes=(
            "请从输入材料中构造一个稳定的 episode_seed（字段以 contract 为准；建议提供可派生稳定话题锚定的信息），"
            "确保信息充分但不冗余，能够支撑多步推理题目的生成。"
        ),
        extend_reason_after_init=(
            "episode_seed 已通过 Init 初始化完成，当前 history 为空，需要基于 seed 生成第一道题目（step=1），开启多步推理链条。"
        ),
        extend_operator_notes_after_init=(
            "基于 episode_seed 生成第一道题目，确保题目能够充分利用背景锚定信息，"
            "为后续扩展题目奠定基础。题目应具有清晰的推理步骤，适合作为多步推理链的起点。"
        ),
    )


def _build_executable_adapter() -> TrackAdapter:
    from agenqa.nodes.op_executable_init import run_executable_init
    from agenqa.nodes.op_executable_extend import run_executable_extend
    from agenqa.nodes.op_executable_revise import run_executable_revise
    from agenqa.nodes.evaluators.code_solve import code_solve
    from agenqa.nodes.evaluators.code_consensus import compute_code_consensus

    return TrackAdapter(
        name="executable",
        roles_by_node=DEFAULT_ROLES_BY_TRACK["executable"],
        specs_by_node=DEFAULT_SPECS_BY_TRACK["executable"],
        has_seed=_has_executable_seed,
        has_history=_has_executable_history,
        run_init=run_executable_init,
        run_extend=run_executable_extend,
        run_revise=run_executable_revise,
        run_solve=code_solve,
        run_consensus=compute_code_consensus,
        init_reason="round_1 固定流程（executable）：初始化 executable_seed 作为整条 executable 链路的起点。",
        init_operator_notes="选择一个可复现的 executable problem seed（例如 SciCode problem_id），为后续 Extend 产出第一步子问题做准备。",
        extend_reason_after_init="executable_seed 已通过 Init 初始化完成，当前 executable_history 为空，需要基于 seed 生成第一步 executable 子问题（step=1）。",
        extend_operator_notes_after_init=(
            "基于 executable_seed 逐步生成第一步可执行的 executable sub-step（接口/目标）与 golden 代码，"
            "并确保后续可扩展为多步链条（单步递进、可测、确定性）。"
        ),
    )


_SEMANTIC_ADAPTER = _build_semantic_adapter()
_UNIFIED_ADAPTER = _build_unified_adapter()
_EXECUTABLE_ADAPTER = _build_executable_adapter()


def resolve_track(agent_conf: Dict[str, Any] | None) -> str:
    block = (agent_conf or {}).get("agent") if isinstance(agent_conf, dict) else None
    if not isinstance(block, dict):
        return "unified"
    track = str(block.get("track") or "").strip().lower()
    if not track:
        return "unified"
    if track in {"unified", "semantic", "executable"}:
        return track
    raise ValueError(f"unsupported track={track!r} (expected 'unified'|'semantic'|'executable')")


def get_track_adapter(agent_conf: Dict[str, Any] | None) -> TrackAdapter:
    track = resolve_track(agent_conf or {})
    if track == "executable":
        return _EXECUTABLE_ADAPTER
    if track == "semantic":
        return _SEMANTIC_ADAPTER
    return _UNIFIED_ADAPTER


__all__ = [
    "DEFAULT_ROLES_BY_TRACK",
    "DEFAULT_SPECS_BY_TRACK",
    "TrackAdapter",
    "get_track_adapter",
    "resolve_track",
]
