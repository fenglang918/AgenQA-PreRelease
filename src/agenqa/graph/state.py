"""Agent 状态与序列化模型（轻量，无外部依赖）。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
from copy import deepcopy
from agenqa.domain.known_tree import KnownTree
from agenqa.domain.executable_schema import ExecutableRecord


class KQARecord:
    """KQA 记录：表示一道题目及其上下文。"""

    def __init__(
        self,
        paper_id: Optional[str] = None,
        qa_idx: Optional[int] = None,
        step: Optional[int] = None,  # 向后兼容：如果提供了 step，会映射到 qa_idx
        known: str = "",
        question: str = "",
        answer: str = "",
        chain: Optional[str] = None,
        subject: Optional[str] = None,
        # Step-level constraints (optional): persisted so Revise/Solve/Consensus can inherit.
        question_type: Optional[str] = None,
        question_type_constraints: Optional[Dict[str, Any]] = None,
        world_contract_text: Optional[str] = None,
        # Path-Fold artifacts (optional)
        path_question_scaffolded: Optional[str] = None,
        path_question_direct: Optional[str] = None,
        path_fold_notes: Optional[str] = None,
    ):
        """初始化 KQARecord。

        Args:
            paper_id: 论文ID
            qa_idx: QA 索引（表示题目在 KQA 链中的序号）
            step: 向后兼容参数，如果提供了 step 且 qa_idx 为 None，则使用 step
            known: 已知条件
            question: 题目
            answer: 答案
            chain: 链式标识
            subject: 学科
        """
        self.paper_id = paper_id
        self.qa_idx = qa_idx if qa_idx is not None else step
        self.known = known
        self.question = question
        self.answer = answer
        self.chain = chain
        self.subject = subject
        self.question_type = question_type
        self.question_type_constraints = question_type_constraints
        self.world_contract_text = world_contract_text
        self.path_question_scaffolded = path_question_scaffolded
        self.path_question_direct = path_question_direct
        self.path_fold_notes = path_fold_notes

    @property
    def record_type(self) -> str:
        return "semantic"

    @property
    def step(self) -> Optional[int]:
        """向后兼容属性：返回 qa_idx"""
        return self.qa_idx

    @step.setter
    def step(self, value: Optional[int]) -> None:
        """向后兼容属性：设置 qa_idx"""
        self.qa_idx = value

    def __repr__(self) -> str:
        known_preview = str(self.known)[:50] if self.known is not None else ""
        return (
            f"KQARecord(paper_id={self.paper_id!r}, qa_idx={self.qa_idx}, "
            f"known={known_preview!r}..., question={self.question[:50]!r}..., "
            f"answer={self.answer[:50]!r}..., chain={self.chain!r}, subject={self.subject!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, KQARecord):
            return False
        return (
            self.paper_id == other.paper_id
            and self.qa_idx == other.qa_idx
            and self.known == other.known
            and self.question == other.question
            and self.answer == other.answer
            and self.chain == other.chain
            and self.subject == other.subject
            and getattr(self, "question_type", None) == getattr(other, "question_type", None)
            and getattr(self, "question_type_constraints", None) == getattr(other, "question_type_constraints", None)
            and getattr(self, "world_contract_text", None) == getattr(other, "world_contract_text", None)
            and getattr(self, "path_question_scaffolded", None) == getattr(other, "path_question_scaffolded", None)
            and getattr(self, "path_question_direct", None) == getattr(other, "path_question_direct", None)
            and getattr(self, "path_fold_notes", None) == getattr(other, "path_fold_notes", None)
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化"""
        known_val = self.known
        if isinstance(known_val, dict):
            known_val = KnownTree.to_json(known_val)
        elif known_val is None:
            known_val = ""
        result: Dict[str, Any] = {
            "record_type": self.record_type,
            "paper_id": self.paper_id,
            "qa_idx": self.qa_idx,
            "step": self.qa_idx,  # 向后兼容
            "known": known_val,
            "question": self.question,
            "answer": self.answer,
            "chain": self.chain,
        }
        if self.subject:
            result["subject"] = self.subject
        if isinstance(getattr(self, "question_type", None), str) and self.question_type.strip():
            result["question_type"] = self.question_type.strip()
        if isinstance(getattr(self, "question_type_constraints", None), dict) and self.question_type_constraints:
            result["question_type_constraints"] = self.question_type_constraints
        if isinstance(getattr(self, "world_contract_text", None), str) and self.world_contract_text.strip():
            result["world_contract_text"] = self.world_contract_text
        # Optional Path-Fold artifacts (kept out of the core KQA fields).
        if isinstance(getattr(self, "path_question_scaffolded", None), str) and self.path_question_scaffolded.strip():
            result["path_question_scaffolded"] = self.path_question_scaffolded
        if isinstance(getattr(self, "path_question_direct", None), str) and self.path_question_direct.strip():
            result["path_question_direct"] = self.path_question_direct
        if isinstance(getattr(self, "path_fold_notes", None), str) and self.path_fold_notes.strip():
            result["path_fold_notes"] = self.path_fold_notes
        return result


@dataclass
class SolveMetrics:
    correct_medium: Optional[bool] = None
    token_ratio_medium: Optional[float] = None


@dataclass
class SolverResult:
    """单个 solver 的完整结果信息。

    结构约定：
    - solver_index[step][round][target][tier] -> SolverResult
    - target ∈ {"edge", "path"}
    - tier ∈ {"tool", "medium", "strong"}（strong 也可能有 "strong_1/2/..." 的多 solver 变体）
    """
    correct: Optional[bool] = None
    token_ratio: Optional[float] = None
    # 拆分 token_ratio：保留输入/输出 token 的绝对量（趋势信号，避免只看比值）
    kq_tokens: Optional[float] = None
    completion_tokens: Optional[float] = None
    model: Optional[str] = None
    service_id: Optional[str] = None
    question_well_posed: Optional[bool] = None
    correctness_feedback: Optional[str] = None
    difficulty_feedback: Optional[str] = None
    question_feedback: Optional[str] = None
    harder_suggestion: Optional[str] = None
    # Track-specific extension fields (e.g., executable execution_time/error_type).
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverVote:
    """A single solver vote used by consensus aggregation (typically strong solvers)."""

    solver_idx: int
    service_id: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None  # success|request_failed|parse_failed
    answer: Optional[str] = None
    answer_normalized: Optional[str] = None
    question_well_posed: Optional[bool] = None
    correctness_feedback: Optional[str] = None
    difficulty_feedback: Optional[str] = None
    judge_status: Optional[str] = None  # success|skipped|failed
    judge_reason: Optional[str] = None


@dataclass
class StrongSolverConsensus:
    """Consensus signals derived from multiple strong solvers for the latest QA."""

    mode: str = "none"  # none|always
    proposed_answer: Optional[str] = None
    answer_consensus: Optional[str] = None
    wellposed_consensus: Optional[bool] = None
    differs_from_proposed: Optional[bool] = None
    consensus_strength: int = 0
    eligible_votes: int = 0
    tie: bool = False
    tie_reason: Optional[str] = None  # 2-way split|all different|insufficient_votes|...
    solvers: List[SolverVote] = field(default_factory=list)


@dataclass
class SolverConsensus:
    """Container for consensus signals (may expand beyond strong in the future)."""

    strong: StrongSolverConsensus = field(default_factory=StrongSolverConsensus)


@dataclass
class Decision:
    operation: str
    reason: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    run_id: str
    artifacts_dir: Path
    max_steps: int = 3
    qa_idx: int = 0  # 当前 QA 索引（原 step，表示当前题目在 KQA 链中的序号）
    # 已执行的"轮数"（每经过一次 director+operator+solve_medium+solve_strong 记为一轮）
    rounds: int = 0
    # 连续 Revise 次数（按 Director 决策统计，用于 fail-fast 终止死循环）
    consecutive_revise: int = 0
    # 连续 answer_contract Revise 次数（Type2 专用 fail-fast）
    consecutive_answer_contract_revise: int = 0

    @property
    def step(self) -> int:
        """向后兼容属性：返回 qa_idx"""
        return self.qa_idx

    @step.setter
    def step(self, value: int) -> None:
        """向后兼容属性：设置 qa_idx"""
        self.qa_idx = value
    history: List[KQARecord | ExecutableRecord] = field(default_factory=list)
    last_decision: Optional[Decision] = None
    metrics: SolveMetrics = field(default_factory=SolveMetrics)
    stop_reason: Optional[str] = None
    # solver_consensus: multi-solver consensus signals for Director/Judge (optional)
    solver_consensus: SolverConsensus = field(default_factory=SolverConsensus)
    # solver_index: 存储每个 step、每个 round 的完整 solver 结果
    # 结构: {step: {round: {"edge": {tier: SolverResult}, "path": {tier: SolverResult}}}}
    solver_index: Dict[int, Dict[int, Dict[str, Dict[str, SolverResult]]]] = field(default_factory=dict)
    # episode 级别的长期记忆（KnownTree v2）
    memory: Dict[str, Any] = field(default_factory=dict)
    paper_id: Optional[str] = None
    subject: Optional[str] = None
    # Path QA (Path-Fold) question variant selection: direct|scaffolded.
    # Used by output_manager/store when dumping path_kqa.jsonl without agent_conf context.
    path_kqa_variant: Optional[str] = None

    def iter_semantic_records(self):
        return (r for r in self.history if isinstance(r, KQARecord))

    def iter_executable_records(self):
        return (r for r in self.history if isinstance(r, ExecutableRecord))

    def latest_semantic_record(self) -> KQARecord | None:
        for r in reversed(self.history):
            if isinstance(r, KQARecord):
                return r
        return None

    def latest_executable_record(self) -> ExecutableRecord | None:
        for r in reversed(self.history):
            if isinstance(r, ExecutableRecord):
                return r
        return None

    def current_round_index(self) -> int:
        """Return the 1-based index of the *next* logical round.

        Internally `rounds` is incremented after each SolveDual/SolveStrong call.
        For operator nodes (Director→Extend/Revise/Compress) that run *before* the
        next solve, we want the upcoming round number, so we use `rounds + 1`.
        """
        try:
            prev = int(getattr(self, "rounds", 0) or 0)
        except Exception:
            prev = 0
        return max(prev + 1, 1)

    def update_solver_index(
        self,
        step: int,
        round: int,
        target: str,  # "edge" or "path"
        tier: str,  # "tool" | "medium" | "strong" | "strong_{i}"
        result: SolverResult,
    ) -> None:
        """更新 solver_index，记录指定 step/round/target/tier 的 solver 结果。

        结构约定：solver_index[step][round][target][tier] -> SolverResult
        - target ∈ {"edge", "path"}
        - tier ∈ {"tool", "medium", "strong"}（strong 也可能有多 solver 变体）

        Args:
            step: QA 索引（step）
            round: 轮次
            target: "edge" 或 "path"
            tier: "medium" 或 "strong"
            result: SolverResult 对象
        """
        norm_target = str(target or "").strip().lower().replace("-", "_")
        if norm_target not in {"edge", "path"}:
            raise ValueError(f"invalid solver_index target={target!r} (expected 'edge' or 'path')")
        target = norm_target
        if step not in self.solver_index:
            self.solver_index[step] = {}
        if round not in self.solver_index[step]:
            self.solver_index[step][round] = {}
        if target not in self.solver_index[step][round]:
            self.solver_index[step][round][target] = {}
        self.solver_index[step][round][target][tier] = result

    def get_latest_solver(
        self,
        step: int,
        target: str,  # "edge" or "path"
        tier: str,  # "tool" | "medium" | "strong" | "strong_{i}"
    ) -> SolverResult | None:
        """获取指定 step/target/tier 的最新 round 的 solver 结果。

        Args:
            step: QA 索引（step）
            target: "edge" 或 "path"
            tier: "tool" / "medium" / "strong" / "strong_{i}"

        Returns:
            最新 round 的 SolverResult，如果不存在则返回 None
        """
        norm_target = str(target or "").strip().lower().replace("-", "_")
        if norm_target not in {"edge", "path"}:
            raise ValueError(f"invalid solver_index target={target!r} (expected 'edge' or 'path')")
        target = norm_target
        try:
            if step not in self.solver_index:
                return None
            rounds_dict = self.solver_index[step]
            if not rounds_dict:
                return None
            # 找到最新的 round
            max_round = max(rounds_dict.keys())
            round_dict = rounds_dict[max_round]
            if target not in round_dict or tier not in round_dict[target]:
                return None
            return round_dict[target][tier]
        except Exception:
            return None

    def get_latest_solver_results(
        self,
        step: int,
        target: str,  # "edge" or "path"
        prefix: str,  # "strong" | "medium" | ...
    ) -> list[tuple[str, SolverResult]]:
        """获取指定 step/target 下、最新 round 中所有匹配 prefix 的 solver 结果。

        返回按 tier 顺序排序后的 `(tier_name, SolverResult)` 列表，例如：
        - `("strong", SolverResult(...))`
        - `("strong_1", SolverResult(...))`
        - `("strong_2", SolverResult(...))`
        """
        norm_target = str(target or "").strip().lower().replace("-", "_")
        pref = str(prefix or "").strip().lower()
        if norm_target not in {"edge", "path"} or not pref:
            return []

        try:
            rounds_dict = self.solver_index.get(step) if isinstance(self.solver_index, dict) else None
            if not isinstance(rounds_dict, dict) or not rounds_dict:
                return []
            max_round = max(rounds_dict.keys())
            round_dict = rounds_dict.get(max_round) or {}
            target_dict = round_dict.get(norm_target) if isinstance(round_dict, dict) else None
            if not isinstance(target_dict, dict):
                return []

            def _tier_key(t: str) -> tuple[int, int]:
                tt = str(t or "")
                if tt == pref:
                    return (0, 0)
                if tt.startswith(f"{pref}_"):
                    try:
                        return (0, int(tt.split("_", 1)[1]) + 1)
                    except Exception:
                        return (0, 999)
                return (1, 999)

            out: list[tuple[str, SolverResult]] = []
            for tier_name in sorted(target_dict.keys(), key=_tier_key):
                if tier_name != pref and not str(tier_name).startswith(f"{pref}_"):
                    continue
                result = target_dict.get(tier_name)
                if isinstance(result, SolverResult):
                    out.append((str(tier_name), result))
            return out
        except Exception:
            return []

    def get_latest_feedback(
        self,
        step: int,
    ) -> tuple[str, SolverResult] | None:
        """获取指定 step 的最新 round 的 solver 反馈（优先 strong，其次 medium）。

        只要存在任一反馈字段（包含新旧字段）即可返回，不要求 correct=True。
        """
        def _has_feedback(res: SolverResult) -> bool:
            try:
                if res.question_well_posed is not None:
                    return True
                for field in ("correctness_feedback", "difficulty_feedback", "question_feedback", "harder_suggestion"):
                    val = getattr(res, field, None)
                    if isinstance(val, str) and val.strip():
                        return True
                return False
            except Exception:
                return False

        # Prefer strong variants (strong, strong_1, strong_2, ...) then medium variants.
        try:
            rounds_dict = self.solver_index.get(step) if isinstance(self.solver_index, dict) else None
            if isinstance(rounds_dict, dict) and rounds_dict:
                max_round = max(rounds_dict.keys())
                round_dict = rounds_dict.get(max_round) or {}
                edge_dict = round_dict.get("edge") if isinstance(round_dict, dict) else None
                tiers = list(edge_dict.keys()) if isinstance(edge_dict, dict) else []

                def _tier_key(t: str) -> tuple[int, int]:
                    tt = str(t or "")
                    if tt == "strong":
                        return (0, 0)
                    if tt.startswith("strong_"):
                        try:
                            return (0, int(tt.split("_", 1)[1]) + 1)
                        except Exception:
                            return (0, 999)
                    if tt == "medium":
                        return (1, 0)
                    if tt.startswith("medium_"):
                        try:
                            return (1, int(tt.split("_", 1)[1]) + 1)
                        except Exception:
                            return (1, 999)
                    return (2, 999)

                for tier in sorted(tiers, key=_tier_key):
                    result = edge_dict.get(tier) if isinstance(edge_dict, dict) else None
                    if isinstance(result, SolverResult) and _has_feedback(result):
                        return (tier, result)
        except Exception:
            pass

        for tier in ("strong", "medium"):
            result = self.get_latest_solver(step, "edge", tier)
            if result and _has_feedback(result):
                return (tier, result)
        return None

    def to_json(self) -> str:
        def _default(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, KQARecord):
                return o.to_dict()
            if isinstance(o, SolverResult):
                return asdict(o)
            if hasattr(o, "__dict__"):
                d = asdict(o)
                # 为向后兼容，同时输出 step 和 qa_idx
                if isinstance(o, AgentState):
                    d["step"] = o.qa_idx
                return d
            return str(o)

        result = asdict(self)
        # 为向后兼容，同时输出 step 和 qa_idx
        result["step"] = self.qa_idx
        # 处理 history 中的 record（KQARecord / ExecutableRecord）
        try:
            result["history"] = [
                (rec.to_dict() if isinstance(rec, KQARecord) else rec.to_dict() if isinstance(rec, ExecutableRecord) else rec)
                for rec in (getattr(self, "history", None) or [])
            ]
        except Exception:
            pass
        return json.dumps(result, ensure_ascii=False, default=_default)

    def append_record(self, rec: KQARecord | ExecutableRecord) -> None:
        if isinstance(rec, KQARecord):
            # 规范化 known 视图：字典转 JSON 字符串
            if isinstance(rec.known, dict):
                rec.known = KnownTree.to_json(rec.known)
            elif rec.known is None:
                rec.known = ""
            else:
                rec.known = str(rec.known)
            if rec.qa_idx is None:
                # Semantic steps are 1-based; step=0 is reserved for seed/init.
                max_step = 0
                for r in self.iter_semantic_records():
                    try:
                        if r.qa_idx is not None:
                            max_step = max(max_step, int(r.qa_idx))
                    except Exception:
                        continue
                rec.qa_idx = max_step + 1
        elif isinstance(rec, ExecutableRecord):
            if rec.qa_idx is None:
                # Executable steps are 1-based; step=0 is reserved for seed/init.
                max_step = 0
                for r in self.iter_executable_records():
                    try:
                        if r.qa_idx is not None:
                            max_step = max(max_step, int(r.qa_idx))
                    except Exception:
                        continue
                rec.qa_idx = max_step + 1
        else:
            raise TypeError(f"Unsupported record type: {type(rec)}")

        self.history.append(rec)
        try:
            self.qa_idx = int(getattr(rec, "step", getattr(rec, "qa_idx", 0)) or 0)
        except Exception:
            self.qa_idx = int(self.qa_idx or 0)

    def append_history(self, rec: KQARecord) -> None:
        """Backward-compatible: append a semantic record."""
        self.append_record(rec)

    def append_executable_history(self, rec: ExecutableRecord) -> None:
        """Backward-compatible: append a executable record (deprecated)."""
        self.append_record(rec)

    # 用一组记录替换整个历史（用于 Compress‑History 等重构场景）
    def replace_history(self, records: List[KQARecord | ExecutableRecord]) -> None:
        self.history = list(records)
        if self.history:
            last = self.history[-1]
            try:
                last_step = int(getattr(last, "step", getattr(last, "qa_idx", 0)) or 0)
            except Exception:
                last_step = 0
            self.qa_idx = last_step
        else:
            self.qa_idx = 0

    @staticmethod
    def load_from_file(path: Path) -> "AgentState":
        """从 state.json 恢复 AgentState，用于断点续跑。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # 必要字段
        run_id = data.get("run_id") or "unknown"
        artifacts_dir = Path(data.get("artifacts_dir") or path.parent)
        max_steps = int(data.get("max_steps", 3))
        # 向后兼容：优先使用 qa_idx，如果没有则使用 step
        qa_idx = int(data.get("qa_idx") or data.get("step", 0))
        rounds = int(data.get("rounds", 0) or 0)
        consecutive_revise = int(data.get("consecutive_revise", 0) or 0)
        consecutive_answer_contract_revise = int(data.get("consecutive_answer_contract_revise", 0) or 0)
        stop_reason = data.get("stop_reason")
        # 恢复 metrics
        metrics_data = data.get("metrics") or {}
        metrics = SolveMetrics(
            correct_medium=metrics_data.get("correct_medium"),
            token_ratio_medium=metrics_data.get("token_ratio_medium") or metrics_data.get("difficulty_medium"),  # 向后兼容
        )
        # 恢复 history（支持 unified history: KQARecord/ExecutableRecord）
        hist_list: List[KQARecord | ExecutableRecord] = []

        def _is_executable_row(row: Dict[str, Any]) -> bool:
            rt = row.get("record_type")
            if isinstance(rt, str) and rt.strip().lower() == "executable":
                return True
            # Heuristic for legacy executable rows without record_type
            return any(k in row for k in ("problem_id", "sub_steps", "test_cases", "per_step_golden"))

        for r in data.get("history") or []:
            if not isinstance(r, dict):
                continue
            try:
                if _is_executable_row(r):
                    hist_list.append(ExecutableRecord.from_dict(r))
                    continue
                # Semantic row
                qa_idx_val = r.get("qa_idx") or r.get("step")
                hist_list.append(
                    KQARecord(
                        paper_id=r.get("paper_id"),
                        qa_idx=qa_idx_val,
                        known=r.get("known") or "",
                        question=r.get("question") or "",
                        answer=r.get("answer") or "",
                        chain=r.get("chain"),
                        subject=(r.get("subject") if isinstance(r.get("subject"), str) else None),
                        question_type=(r.get("question_type") if isinstance(r.get("question_type"), str) else None),
                        question_type_constraints=(
                            r.get("question_type_constraints") if isinstance(r.get("question_type_constraints"), dict) else None
                        ),
                        world_contract_text=(
                            r.get("world_contract_text") if isinstance(r.get("world_contract_text"), str) else None
                        ),
                        path_question_scaffolded=(
                            r.get("path_question_scaffolded") if isinstance(r.get("path_question_scaffolded"), str) else None
                        ),
                        path_question_direct=(
                            r.get("path_question_direct") if isinstance(r.get("path_question_direct"), str) else None
                        ),
                        path_fold_notes=(
                            r.get("path_fold_notes") if isinstance(r.get("path_fold_notes"), str) else None
                        ),
                    )
                )
            except Exception:
                logging.getLogger(__name__).warning("恢复 history 失败，忽略条目: %s", str(r)[:200])

        # 恢复 legacy executable_history（可选，合并入 unified history）
        for r in data.get("executable_history") or []:
            if not isinstance(r, dict):
                continue
            try:
                hist_list.append(ExecutableRecord.from_dict(r))
            except Exception:
                logging.getLogger(__name__).warning("恢复 executable_history 失败，忽略条目: %s", str(r)[:200])
        # 恢复 last_decision（可选）
        last_decision = None
        if data.get("last_decision"):
            try:
                ld = data["last_decision"]
                last_decision = Decision(
                    operation=ld.get("operation"),
                    reason=ld.get("reason", ""),
                    params=ld.get("params") or {},
                )
            except Exception:
                logging.getLogger(__name__).warning("恢复 last_decision 失败，忽略")
        # 恢复 solver_index（可选）
        solver_index: Dict[int, Dict[int, Dict[str, Dict[str, SolverResult]]]] = {}
        try:
            solver_index_data = data.get("solver_index") or {}
            for step_str, rounds_dict in solver_index_data.items():
                step_int = int(step_str)
                solver_index[step_int] = {}
                for round_str, targets_dict in rounds_dict.items():
                    round_int = int(round_str)
                    solver_index[step_int][round_int] = {}
                    for target_name, tiers_dict in targets_dict.items():
                        norm_target = str(target_name or "").strip().lower().replace("-", "_")
                        if norm_target not in {"edge", "path"}:
                            continue
                        if norm_target not in solver_index[step_int][round_int]:
                            solver_index[step_int][round_int][norm_target] = {}
                        for tier_name, result_dict in tiers_dict.items():
                            if isinstance(result_dict, dict):
                                solver_index[step_int][round_int][norm_target][tier_name] = SolverResult(
                                    correct=result_dict.get("correct"),
                                    token_ratio=result_dict.get("token_ratio"),
                                    kq_tokens=result_dict.get("kq_tokens"),
                                    completion_tokens=result_dict.get("completion_tokens"),
                                    model=result_dict.get("model"),
                                    service_id=result_dict.get("service_id"),
                                    question_well_posed=result_dict.get("question_well_posed"),
                                    correctness_feedback=result_dict.get("correctness_feedback"),
                                    difficulty_feedback=result_dict.get("difficulty_feedback"),
                                    question_feedback=result_dict.get("question_feedback"),
                                    harder_suggestion=result_dict.get("harder_suggestion"),
                                    extra=(result_dict.get("extra") if isinstance(result_dict.get("extra"), dict) else {}),
                                )
        except Exception:
            logging.getLogger(__name__).warning("恢复 solver_index 失败，忽略")
        # 恢复 solver_consensus（可选，向后兼容）
        solver_consensus = SolverConsensus()
        try:
            sc = data.get("solver_consensus") or {}
            if isinstance(sc, dict):
                strong = sc.get("strong") or {}
                if isinstance(strong, dict):
                    votes: List[SolverVote] = []
                    raw_votes = strong.get("solvers") or []
                    if isinstance(raw_votes, list):
                        for item in raw_votes:
                            if not isinstance(item, dict):
                                continue
                            try:
                                votes.append(
                                    SolverVote(
                                        solver_idx=int(item.get("solver_idx", 0) or 0),
                                        service_id=item.get("service_id"),
                                        model=item.get("model"),
                                        status=item.get("status"),
                                        answer=item.get("answer"),
                                        answer_normalized=item.get("answer_normalized"),
                                        question_well_posed=item.get("question_well_posed"),
                                        correctness_feedback=item.get("correctness_feedback"),
                                        difficulty_feedback=item.get("difficulty_feedback"),
                                    )
                                )
                            except Exception:
                                continue
                    solver_consensus = SolverConsensus(
                        strong=StrongSolverConsensus(
                            mode=str(strong.get("mode") or "off"),
                            proposed_answer=strong.get("proposed_answer"),
                            answer_consensus=strong.get("answer_consensus"),
                            wellposed_consensus=strong.get("wellposed_consensus"),
                            differs_from_proposed=strong.get("differs_from_proposed"),
                            consensus_strength=int(strong.get("consensus_strength", 0) or 0),
                            eligible_votes=int(strong.get("eligible_votes", 0) or 0),
                            tie=bool(strong.get("tie", False)),
                            tie_reason=strong.get("tie_reason"),
                            solvers=votes,
                        )
                    )
        except Exception:
            solver_consensus = SolverConsensus()
        memory = KnownTree.normalize_memory(data.get("memory"))
        state = AgentState(
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            max_steps=max_steps,
            qa_idx=qa_idx,
            rounds=rounds,
            consecutive_revise=consecutive_revise,
            consecutive_answer_contract_revise=consecutive_answer_contract_revise,
            history=hist_list,
            last_decision=last_decision,
            metrics=metrics,
            stop_reason=stop_reason,
            paper_id=data.get("paper_id"),
            subject=data.get("subject"),
            path_kqa_variant=(data.get("path_kqa_variant") if isinstance(data.get("path_kqa_variant"), str) else None),
            solver_consensus=solver_consensus,
            solver_index=solver_index,
            memory=memory,
        )
        # 确保 qa_idx 与 unified history 一致（取最后一条记录的 step）
        if state.history:
            try:
                last = state.history[-1]
                state.qa_idx = int(getattr(last, "step", getattr(last, "qa_idx", 0)) or 0)
            except Exception:
                pass
        return state

    # 从 LangGraph 可能返回的 dict 结构更新自身，避免状态丢失
    def update_from_mapping(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        try:
            if "run_id" in data and isinstance(data.get("run_id"), str):
                self.run_id = data["run_id"] or self.run_id
            if "artifacts_dir" in data and data.get("artifacts_dir"):
                try:
                    self.artifacts_dir = Path(data.get("artifacts_dir"))
                except Exception:
                    pass
            if "max_steps" in data and data.get("max_steps") is not None:
                try:
                    self.max_steps = int(data.get("max_steps"))
                except Exception:
                    pass
            # 向后兼容：优先使用 qa_idx，如果没有则使用 step
            if "qa_idx" in data and data.get("qa_idx") is not None:
                try:
                    self.qa_idx = int(data.get("qa_idx"))
                except Exception:
                    pass
            elif "step" in data and data.get("step") is not None:
                try:
                    self.qa_idx = int(data.get("step"))
                except Exception:
                    pass
            if "rounds" in data and data.get("rounds") is not None:
                try:
                    self.rounds = int(data.get("rounds"))
                except Exception:
                    pass
            if "consecutive_revise" in data and data.get("consecutive_revise") is not None:
                try:
                    self.consecutive_revise = int(data.get("consecutive_revise"))
                except Exception:
                    pass
            if "consecutive_answer_contract_revise" in data and data.get("consecutive_answer_contract_revise") is not None:
                try:
                    self.consecutive_answer_contract_revise = int(data.get("consecutive_answer_contract_revise"))
                except Exception:
                    pass
            if "stop_reason" in data and data.get("stop_reason") is not None:
                self.stop_reason = data.get("stop_reason")
            if "memory" in data:
                try:
                    self.memory = KnownTree.normalize_memory(data.get("memory"))
                except Exception:
                    pass
            if "paper_id" in data and data.get("paper_id") is not None:
                self.paper_id = data.get("paper_id")
            if "subject" in data and data.get("subject") is not None:
                self.subject = data.get("subject")
            if "path_kqa_variant" in data and isinstance(data.get("path_kqa_variant"), str):
                self.path_kqa_variant = data.get("path_kqa_variant") or self.path_kqa_variant
            metrics = data.get("metrics") or {}
            if isinstance(metrics, dict):
                if "correct_medium" in metrics:
                    self.metrics.correct_medium = metrics.get("correct_medium")
                if "token_ratio_medium" in metrics:
                    self.metrics.token_ratio_medium = metrics.get("token_ratio_medium")
                elif "difficulty_medium" in metrics:  # 向后兼容
                    self.metrics.token_ratio_medium = metrics.get("difficulty_medium")
            hist = data.get("history")
            if isinstance(hist, list):
                new_hist: List[KQARecord | ExecutableRecord] = []
                for r in hist:
                    if not isinstance(r, dict):
                        continue
                    try:
                        rt = r.get("record_type")
                        if isinstance(rt, str) and rt.strip().lower() == "executable":
                            new_hist.append(ExecutableRecord.from_dict(r))
                        elif any(k in r for k in ("problem_id", "sub_steps", "test_cases", "per_step_golden")):
                            new_hist.append(ExecutableRecord.from_dict(r))
                        else:
                            qa_idx_val = r.get("qa_idx") or r.get("step")
                            new_hist.append(
                                KQARecord(
                                    paper_id=r.get("paper_id"),
                                    qa_idx=qa_idx_val,
                                    known=r.get("known") or "",
                                    question=r.get("question") or "",
                                    answer=r.get("answer") or "",
                                    chain=r.get("chain"),
                                    subject=(r.get("subject") if isinstance(r.get("subject"), str) else None),
                                    world_contract_text=(
                                        r.get("world_contract_text") if isinstance(r.get("world_contract_text"), str) else None
                                    ),
                                    path_question_scaffolded=(
                                        r.get("path_question_scaffolded") if isinstance(r.get("path_question_scaffolded"), str) else None
                                    ),
                                    path_question_direct=(
                                        r.get("path_question_direct") if isinstance(r.get("path_question_direct"), str) else None
                                    ),
                                    path_fold_notes=(
                                        r.get("path_fold_notes") if isinstance(r.get("path_fold_notes"), str) else None
                                    ),
                                )
                            )
                    except Exception:
                        continue
                if new_hist:
                    self.history = new_hist
            # legacy executable_history: merge into unified history
            executable_hist = data.get("executable_history")
            if isinstance(executable_hist, list):
                for r in executable_hist:
                    if not isinstance(r, dict):
                        continue
                    try:
                        self.history.append(ExecutableRecord.from_dict(r))
                    except Exception:
                        continue
            ld = data.get("last_decision")
            if isinstance(ld, dict) and ld.get("operation"):
                try:
                    self.last_decision = Decision(
                        operation=ld.get("operation"),
                        reason=ld.get("reason", ""),
                        params=ld.get("params") or {},
                    )
                except Exception:
                    pass
            # 保持 qa_idx 与 unified history 一致（取最后一条记录的 step）
            if self.history:
                try:
                    last = self.history[-1]
                    self.qa_idx = int(getattr(last, "step", getattr(last, "qa_idx", 0)) or 0)
                except Exception:
                    pass
            # 更新 solver_index（可选）
            solver_index_data = data.get("solver_index")
            if isinstance(solver_index_data, dict):
                try:
                    for step_str, rounds_dict in solver_index_data.items():
                        step_int = int(step_str)
                        if step_int not in self.solver_index:
                            self.solver_index[step_int] = {}
                        for round_str, targets_dict in rounds_dict.items():
                            round_int = int(round_str)
                            if round_int not in self.solver_index[step_int]:
                                self.solver_index[step_int][round_int] = {}
                            for target_name, tiers_dict in targets_dict.items():
                                norm_target = str(target_name or "").strip().lower().replace("-", "_")
                                if norm_target not in {"edge", "path"}:
                                    continue
                                if norm_target not in self.solver_index[step_int][round_int]:
                                    self.solver_index[step_int][round_int][norm_target] = {}
                                for tier_name, result_dict in tiers_dict.items():
                                    if isinstance(result_dict, dict):
                                        self.solver_index[step_int][round_int][norm_target][tier_name] = SolverResult(
                                            correct=result_dict.get("correct"),
                                            token_ratio=result_dict.get("token_ratio"),
                                            kq_tokens=result_dict.get("kq_tokens"),
                                            completion_tokens=result_dict.get("completion_tokens"),
                                            model=result_dict.get("model"),
                                            service_id=result_dict.get("service_id"),
                                            question_well_posed=result_dict.get("question_well_posed"),
                                            correctness_feedback=result_dict.get("correctness_feedback"),
                                            difficulty_feedback=result_dict.get("difficulty_feedback"),
                                            question_feedback=result_dict.get("question_feedback"),
                                            harder_suggestion=result_dict.get("harder_suggestion"),
                                            extra=(result_dict.get("extra") if isinstance(result_dict.get("extra"), dict) else {}),
                                        )
                except Exception:
                    pass
            # 更新 solver_consensus（可选）
            sc = data.get("solver_consensus")
            if isinstance(sc, dict):
                strong = sc.get("strong") if isinstance(sc.get("strong"), dict) else {}
                if isinstance(strong, dict):
                    votes: List[SolverVote] = []
                    raw_votes = strong.get("solvers") or []
                    if isinstance(raw_votes, list):
                        for item in raw_votes:
                            if not isinstance(item, dict):
                                continue
                            try:
                                votes.append(
                                    SolverVote(
                                        solver_idx=int(item.get("solver_idx", 0) or 0),
                                        service_id=item.get("service_id"),
                                        model=item.get("model"),
                                        status=item.get("status"),
                                        answer=item.get("answer"),
                                        answer_normalized=item.get("answer_normalized"),
                                        question_well_posed=item.get("question_well_posed"),
                                        correctness_feedback=item.get("correctness_feedback"),
                                        difficulty_feedback=item.get("difficulty_feedback"),
                                    )
                                )
                            except Exception:
                                continue
                    self.solver_consensus = SolverConsensus(
                        strong=StrongSolverConsensus(
                            mode=str(strong.get("mode") or "off"),
                            proposed_answer=strong.get("proposed_answer"),
                            answer_consensus=strong.get("answer_consensus"),
                            wellposed_consensus=strong.get("wellposed_consensus"),
                            differs_from_proposed=strong.get("differs_from_proposed"),
                            consensus_strength=int(strong.get("consensus_strength", 0) or 0),
                            eligible_votes=int(strong.get("eligible_votes", 0) or 0),
                            tie=bool(strong.get("tie", False)),
                            tie_reason=strong.get("tie_reason"),
                            solvers=votes,
                        )
                    )
        except Exception:
            logging.getLogger(__name__).warning("update_from_mapping 失败，保持原状态")


__all__ = [
    "AgentState",
    "Decision",
    "KQARecord",
    "SolveMetrics",
    "SolverResult",
    "SolverVote",
    "StrongSolverConsensus",
    "SolverConsensus",
]
