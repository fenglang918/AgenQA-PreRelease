"""Extend Operator：

- 基于最近一条 KQA 生成下一步；
- KnownInit 已在上游处理，本节点只负责 extend 链路（draft_chain → format → step_cert_builder）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import logging
from dataclasses import dataclass

from infra.data.io import read_jsonl
from agenqa.graph.state import AgentState
from agenqa.graph.output_manager import OutputContext, compute_step_dir
from infra.data.ids import generate_paper_id
from agenqa.domain.node_result import NodeResult
from agenqa.memory.store import dump_director_decision_for_step, dump_edge_kqa_for_step, dump_path_kqa_for_step
from agenqa.nodes.utils import build_director_notes, is_symbolic_only, select_question_type
from agenqa.graph.roles_subgraph import run_extend_graph


logger = logging.getLogger(__name__)


@dataclass
class ExtendNodeSettings:
    papers_path: Path
    prompt_path: Path
    draft_prompt_path: Path
    format_prompt_path: Path
    generator: Dict[str, Any]
    agent_lang: Optional[str]
    symbolic_only: bool
    max_background_step: Optional[int]


def _iter_one_paper(papers_path: Path) -> Iterable[Dict[str, Any]]:
    """Yield exactly one paper record for QA‑Init.

    Supports two input forms:
    - JSONL file: read the first JSON object as a paper record.
    - Plain text file (.txt): read entire file content into the `text` field.
    """
    suffix = papers_path.suffix.lower()
    if suffix == ".txt" or suffix == "":
        try:
            content = papers_path.read_text(encoding="utf-8")
        except Exception:
            content = papers_path.read_text(errors="ignore")
        yield {"paper_id": generate_paper_id({"text": content}), "text": content}
        return
    # Default: assume JSONL
    yield from (r for i, r in enumerate(read_jsonl(papers_path), start=1) if i <= 1)


def _resolve_settings(agent_conf: Dict[str, Any]) -> ExtendNodeSettings:
    data_conf = agent_conf.get("data") or {}
    papers_path = Path(data_conf.get("papers_path") or "data/input/papers/first_line.jsonl")

    op_conf = (agent_conf.get("operators") or {}).get("extend") or {}
    prompt_path = Path(op_conf.get("prompt_path") or "src/agenqa/prompts/extend_upgrade.prompt")
    draft_prompt_path = Path(op_conf.get("draft_prompt_path") or "src/agenqa/prompts/draft.prompt")
    format_prompt_path = Path(op_conf.get("format_prompt_path") or "src/agenqa/prompts/format.prompt")
    generator = op_conf.get("generator") or {
        "service_type": "private_endpoint",
        "service_id": op_conf.get("service_id"),
    }

    agent_lang = str((agent_conf.get("agent") or {}).get("lang") or "").lower() or None
    symbolic_only = is_symbolic_only(agent_conf)

    try:
        max_background_step = int(op_conf.get("max_background_step", 2))
    except Exception:
        max_background_step = op_conf.get("max_background_step", 2)

    return ExtendNodeSettings(
        papers_path=papers_path,
        prompt_path=prompt_path,
        draft_prompt_path=draft_prompt_path,
        format_prompt_path=format_prompt_path,
        generator=generator,
        agent_lang=agent_lang,
        symbolic_only=symbolic_only,
        max_background_step=max_background_step,
    )


def run_extend(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    """Extend：在已有 KQA 的基础上生成下一题。

    当前架构下，KnownInit 只负责初始化 episode_seed（字段由 contract 定义；anchor 可由结构化字段/subject/keywords 派生）。
    Extend 必须支持在 history 为空时生成第一题（step=1），作为链式推理的起点。
    """
    settings = _resolve_settings(agent_conf)
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

    # 后续：Extend‑Upgrade 基于最新 KQA
    before = len(state.history)

    # 本轮 step 索引：在已有历史基础上加一
    try:
        step_idx = int(state.step) + 1
    except Exception:
        step_idx = len(state.history)
    round_idx = state.current_round_index()
    ctx: OutputContext | None = None
    if output_manager:
        ctx = output_manager.begin("extend", step_idx, round_idx)
        step_dir = ctx.step_dir
        ctx.dump_director_decision(state, step_idx)
    else:
        step_dir = compute_step_dir(state.artifacts_dir, "extend", step_idx, round_idx)
        step_dir.mkdir(parents=True, exist_ok=True)
        # 记录本次由 Director 决策得到的操作信息
        dump_director_decision_for_step(state, step_dir, step_idx)

    # 执行 roles 路径：Draft → Format
    state, role_outputs, step_dir, step_idx, round_idx = run_extend_graph(agent_conf, state)
    appended = 1 if before < len(state.history) else 0
    # 生成本步的 edge_kqa 与 path_kqa 快照（若无 OutputManager，则直接落盘）
    if not output_manager:
        dump_edge_kqa_for_step(state, step_dir)
        dump_path_kqa_for_step(state, step_dir, fold_variant=getattr(state, "path_kqa_variant", "direct") or "direct")

    logger.info("Extend 完成（roles 模式），追加条数: %s", appended)
    if appended == 0 and before == len(state.history):
        state.stop_reason = "extend_no_output"
    if output_manager:
        return NodeResult(
            state=state,
            step_idx=step_idx,
            round_idx=round_idx,
            role_outputs=role_outputs,
            step_dir=step_dir,
        )
    return state
