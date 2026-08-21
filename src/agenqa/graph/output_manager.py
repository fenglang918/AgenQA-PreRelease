"""统一的节点输出管理器。

为各个节点提供声明式的输出保存能力，避免在节点内部散落重复的
目录创建与快照写入逻辑。节点只需返回 NodeResult，Runner 会根据
OutputSpec/roles 自动落盘。
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from agenqa.graph.state import AgentState
from agenqa.memory.store import (
    dump_director_decision_for_step,
    dump_edge_kqa_for_step,
    dump_path_kqa_for_step,
    dump_edge_executable_for_step,
    dump_path_executable_for_step,
)
from agenqa.domain.node_result import NodeResult, OutputSpec
from agenqa.revise_modes import ALLOWED_REVISE_MODES, normalize_revise_mode

logger = logging.getLogger(__name__)


def compute_step_dir(root: Path, node: str, qa_idx: int, round_idx: int | None, revise_mode: str | None = None) -> Path:
    """Return the canonical directory for a given (round, qa_idx, node).

    Layout (simplified):
        <root>/
          ├── round_1/
          │     ├── step_0_qa_init/      # round 1 有多个操作，保留 qa_idx 信息
          │     ├── step_0_director/
          │     └── step_1_extend/
          ├── round_2/
          │     ├── director/            # round >= 2 简化，直接使用 node 名
          │     └── extend/
          ├── round_3/
          │     ├── director/
          │     └── extend/
          ├── 00_Prompts_Snapshot/
          └── 00_Summary/

    This helper is used by OutputManager and node implementations to keep the
    on-disk organization consistent across the codebase.

    Args:
        revise_mode: 当 node 为 "revise" 时，用于区分不同修订子模式。
                     目录名将变为 "revise_correctness" / "revise_reuse_hidden" / "revise_quality"。

    Note: 为了向后兼容，目录名仍使用 step_ 前缀，但参数已改为 qa_idx。
    """
    ridx = round_idx or 1
    root = Path(root)

    # 方案 C：根据 revise_mode 调整目录名
    if node == "revise":
        mode = normalize_revise_mode(revise_mode)
        if mode in ALLOWED_REVISE_MODES:
            node = f"revise_{mode}"

    # round 1 有多个操作（qa_init + extend），需要保留 qa_idx 信息以区分
    # round >= 2 通常只有一个 operator，简化目录结构
    if ridx == 1:
        return root / f"round_{ridx}" / f"step_{qa_idx}_{node}"
    else:
        return root / f"round_{ridx}" / node


@dataclass
class OutputContext:
    """便捷上下文：封装一步的输出目录与常用写入方法。"""

    node: str
    qa_idx: int  # QA 索引（原 step_idx）
    round_idx: int
    step_dir: Path

    @property
    def step_idx(self) -> int:
        """向后兼容属性：返回 qa_idx"""
        return self.qa_idx

    def path(self, name: str) -> Path:
        return self.step_dir / name

    def dump_director_decision(self, state: AgentState, qa_idx_override: int | None = None) -> Path:
        try:
            return dump_director_decision_for_step(state, self.step_dir, qa_idx_override or self.qa_idx)
        except Exception:
            return self.step_dir / "director_decision.json"

    def dump_edge_kqa(self, state: AgentState, filename: str = "edge_kqa.jsonl") -> Path:
        try:
            return dump_edge_kqa_for_step(state, self.step_dir, filename=filename)
        except Exception:
            return self.step_dir / filename

    def dump_path_kqa(self, state: AgentState, filename: str = "path_kqa.jsonl") -> Path:
        try:
            return dump_path_kqa_for_step(state, self.step_dir, filename=filename)
        except Exception:
            return self.step_dir / filename

    def dump_edge_executable(self, state: AgentState, filename: str = "edge_executable.jsonl") -> Path:
        try:
            return dump_edge_executable_for_step(state, self.step_dir, filename=filename)
        except Exception:
            return self.step_dir / filename

    def dump_path_executable(self, state: AgentState, filename: str = "path_executable.jsonl") -> Path:
        try:
            return dump_path_executable_for_step(state, self.step_dir, filename=filename)
        except Exception:
            return self.step_dir / filename

    def save_role_output(self, role: str, payload: Any, index: int | None = None) -> Path:
        # TODO: 后续可以在不破坏历史 run 的前提下，考虑同时写入旧的 roles 目录以便过渡。
        roles_dir = self.step_dir / "subruns"
        roles_dir.mkdir(parents=True, exist_ok=True)
        idx_prefix = f"{index:02d}_" if index is not None else ""
        path = roles_dir / f"{idx_prefix}{role}.json"
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            text = str(payload)
        path.write_text(text, encoding="utf-8")
        return path


class OutputManager:
    """集中管理每个节点的输出与中间结果。"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    def build_step_dir(self, node: str, qa_idx: int, round_idx: int | None, revise_mode: str | None = None) -> Path:
        step_dir = compute_step_dir(self.run_dir, node, qa_idx, round_idx, revise_mode)
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def begin(self, node: str, qa_idx: int, round_idx: int | None, revise_mode: str | None = None) -> OutputContext:
        step_dir = self.build_step_dir(node, qa_idx, round_idx, revise_mode)
        ridx = round_idx or 1
        return OutputContext(node=node, qa_idx=qa_idx, round_idx=ridx, step_dir=step_dir)

    def save_result(
        self,
        node: str,
        result: NodeResult,
        specs: Sequence[OutputSpec] | None = None,
        roles: Sequence[str] | None = None,
    ) -> None:
        """根据节点声明与 NodeResult 自动落盘。"""
        qa_idx = result.step_idx if result.step_idx is not None else int(getattr(result.state, "qa_idx", 0) or 0)
        round_idx = result.round_idx if result.round_idx is not None else result.state.current_round_index()
        step_dir = Path(result.step_dir) if result.step_dir else self.build_step_dir(node, qa_idx, round_idx)

        # 1) 角色级中间结果
        self._save_roles(step_dir, result.role_outputs, roles)

        # 2) 声明式输出
        for spec in specs or []:
            payload = result.outputs.get(spec.name)
            if (
                spec.name
                not in {"edge_kqa", "path_kqa", "edge_executable", "path_executable"}
                and payload is None
                and not spec.required
            ):
                continue
            target_dir = step_dir
            # Solver 相关产物统一放入 solve 子目录，避免与其他节点文件混杂
            if spec.name.startswith("solve_") and step_dir.name != "solve":
                target_dir = step_dir / "solve"
            path = target_dir / f"{spec.name}.{spec.type}"
            if spec.name == "edge_kqa":
                self._save_edge_kqa(step_dir, result.state, payload, filename=path.name, expected_qa_idx=qa_idx)
                continue
            if spec.name == "path_kqa":
                self._save_path_kqa(step_dir, result.state, payload, filename=path.name, expected_qa_idx=qa_idx)
                continue
            if spec.name == "edge_executable":
                self._save_edge_executable(step_dir, result.state, payload, filename=path.name, expected_qa_idx=qa_idx)
                continue
            if spec.name == "path_executable":
                self._save_path_executable(step_dir, result.state, payload, filename=path.name, expected_qa_idx=qa_idx)
                continue
            self._write_payload(path, spec.type, payload)

    def _save_roles(self, step_dir: Path, role_outputs: Dict[str, Any], roles: Sequence[str] | None) -> None:
        if not role_outputs:
            return
        role_names: Iterable[str]
        if roles:
            role_names = roles
        else:
            role_names = role_outputs.keys()
        for idx, role in enumerate(role_names, start=1):
            if role not in role_outputs:
                continue
            payload = role_outputs.get(role)
            subruns_dir = step_dir / "subruns"
            subruns_dir.mkdir(parents=True, exist_ok=True)
            path = subruns_dir / f"{idx:02d}_{role}.json"
            self._write_payload(path, "json", payload)

    def _save_edge_kqa(self, step_dir: Path, state: AgentState, payload: Any, filename: str, expected_qa_idx: int | None) -> None:
        path = step_dir / filename
        if payload is None:
            try:
                last = state.history[-1] if state.history else None
                last_idx = int(getattr(last, "qa_idx", getattr(last, "step", -1)) or -1) if last else -1
                if expected_qa_idx is not None and last_idx != int(expected_qa_idx):
                    path.write_text("", encoding="utf-8")
                    return
            except Exception:
                pass
            try:
                dump_edge_kqa_for_step(state, step_dir, filename=filename)
                return
            except Exception:
                payload = {}
        self._write_payload(path, "jsonl", payload)

    def _save_path_kqa(self, step_dir: Path, state: AgentState, payload: Any, filename: str, expected_qa_idx: int | None) -> None:
        path = step_dir / filename
        if payload is None:
            try:
                last = state.history[-1] if state.history else None
                last_idx = int(getattr(last, "qa_idx", getattr(last, "step", -1)) or -1) if last else -1
                if expected_qa_idx is not None and last_idx != int(expected_qa_idx):
                    path.write_text("", encoding="utf-8")
                    return
            except Exception:
                pass
            try:
                dump_path_kqa_for_step(state, step_dir, filename=filename)
                return
            except Exception:
                payload = {}
        self._write_payload(path, "jsonl", payload)

    def _save_edge_executable(self, step_dir: Path, state: AgentState, payload: Any, filename: str, expected_qa_idx: int | None) -> None:
        path = step_dir / filename
        if payload is None:
            try:
                last = state.latest_executable_record()
                last_idx = int(getattr(last, "qa_idx", -1) or -1) if last else -1
                if expected_qa_idx is not None and last_idx != int(expected_qa_idx):
                    path.write_text("", encoding="utf-8")
                    return
            except Exception:
                pass
            try:
                dump_edge_executable_for_step(state, step_dir, filename=filename)
                return
            except Exception:
                payload = {}
        self._write_payload(path, "jsonl", payload)

    def _save_path_executable(self, step_dir: Path, state: AgentState, payload: Any, filename: str, expected_qa_idx: int | None) -> None:
        path = step_dir / filename
        if payload is None:
            try:
                last = state.latest_executable_record()
                last_idx = int(getattr(last, "qa_idx", -1) or -1) if last else -1
                if expected_qa_idx is not None and last_idx != int(expected_qa_idx):
                    path.write_text("", encoding="utf-8")
                    return
            except Exception:
                pass
            try:
                dump_path_executable_for_step(state, step_dir, filename=filename)
                return
            except Exception:
                payload = {}
        self._write_payload(path, "jsonl", payload)

    def _write_payload(self, path: Path, kind: str, payload: Any) -> None:
        if payload is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # 如果 payload 已经是文件，直接复制
        if isinstance(payload, Path) and payload.exists():
            try:
                if path.resolve() == payload.resolve():
                    return
                shutil.copy2(payload, path)
                return
            except Exception:
                pass
        if kind == "json":
            try:
                text = json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception:
                text = str(payload)
            path.write_text(text, encoding="utf-8")
            return
        if kind == "jsonl":
            self._write_jsonl(path, payload)
            return
        # 默认按文本写出
        path.write_text(str(payload), encoding="utf-8")

    def _write_jsonl(self, path: Path, payload: Any) -> None:
        if payload is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        if isinstance(payload, Path) and payload.exists():
            try:
                shutil.copy2(payload, path)
                return
            except Exception:
                pass
        elif isinstance(payload, dict):
            try:
                lines = [json.dumps(payload, ensure_ascii=False) + "\n"]
            except Exception:
                lines = [str(payload) + "\n"]
        elif isinstance(payload, Iterable):
            for item in payload:
                try:
                    lines.append(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception:
                    lines.append(str(item) + "\n")
        else:
            lines = [str(payload) + "\n"]
        path.write_text("".join(lines), encoding="utf-8")


__all__ = ["OutputManager", "OutputContext"]
