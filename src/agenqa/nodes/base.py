"""Operator base interface for AgenQA agents.

Defines a minimal, pluggable interface so each operator can be
invoked uniformly by the episode-level graph.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, TYPE_CHECKING

# 从 domain 导入，避免循环依赖
from agenqa.domain.node_result import OutputSpec, NodeResult

if TYPE_CHECKING:
    from agenqa.graph.state import AgentState


class Operator:
    """Minimal operator interface.

    Implementations should be side-effect free except writing to
    the provided AgentState (history/metrics/snapshots/artifacts).
    Return the same state instance after mutation for LangGraph.
    """

    name: str = "operator"
    outputs: ClassVar[List[OutputSpec]] = []
    roles: ClassVar[List[str]] = []

    def run(self, agent_conf: Dict[str, Any], state: AgentState, **kwargs: Any) -> AgentState | NodeResult:  # noqa: D401
        """Execute the operator and return the state or NodeResult."""
        raise NotImplementedError


__all__ = ["Operator", "OutputSpec", "NodeResult"]
