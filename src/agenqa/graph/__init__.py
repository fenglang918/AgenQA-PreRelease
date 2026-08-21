"""Graph orchestration layer for AgenQA."""

from .state import AgentState, Decision, KQARecord, SolveMetrics

__all__ = [
    "AgentState",
    "Decision",
    "KQARecord",
    "SolveMetrics",
    "run_episode",
    "GraphBuilder",
    "build_graph",
    "OPERATION_ROUTES",
    "require_langgraph",
]


def run_episode(*args, **kwargs):  # type: ignore[no-untyped-def]
    # Lazy import to avoid circular imports at module import time.
    from .runner import run_episode as _run_episode

    return _run_episode(*args, **kwargs)


def build_graph(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .builder import build_graph as _build_graph

    return _build_graph(*args, **kwargs)


def require_langgraph(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .builder import require_langgraph as _require_langgraph

    return _require_langgraph(*args, **kwargs)


def __getattr__(name: str):  # noqa: D401
    """Lazy attribute loading for builder exports."""
    if name in {"GraphBuilder", "OPERATION_ROUTES"}:
        from . import builder as _builder

        return getattr(_builder, name)
    raise AttributeError(name)
