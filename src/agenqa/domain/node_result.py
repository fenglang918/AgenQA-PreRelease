"""节点输出规格与结果容器。

独立模块，避免 nodes ↔ graph 循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from agenqa.graph.state import AgentState


@dataclass
class OutputSpec:
    """Declarative output schema used by operators."""

    name: str
    type: Literal["json", "jsonl", "txt"] = "json"
    required: bool = True
    is_intermediate: bool = False


@dataclass
class NodeResult:
    """Standardized node return payload."""

    state: "AgentState"
    step_idx: int | None = None
    round_idx: int | None = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    role_outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    step_dir: Any | None = None  # typically Path; kept loose to avoid hard dependency


__all__ = ["OutputSpec", "NodeResult"]
