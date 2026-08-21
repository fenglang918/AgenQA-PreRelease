"""Domain helpers for AgenQA (Known trees, chain utilities, node result).

注意：schema 模块（draft_schema, solver_schema 等）可直接导入，
无需通过本 __init__.py，避免循环导入。
"""

from .known_tree import KnownTree, normalize_known, parse_known_to_dict
from .known_utils import normalize_known as normalize_known_raw, parse_known_to_dict as parse_known_to_dict_raw
from .node_result import NodeResult, OutputSpec


def __getattr__(name: str):
    """延迟导入 chain 模块，避免循环依赖。"""
    if name in (
        "collect_chain",
        "compose_known",
        "head_tail_view",
        "read_nodes",
        "verify_known_materialization",
    ):
        from . import chain
        return getattr(chain, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KnownTree",
    "normalize_known",
    "parse_known_to_dict",
    "normalize_known_raw",
    "parse_known_to_dict_raw",
    "collect_chain",
    "compose_known",
    "head_tail_view",
    "read_nodes",
    "verify_known_materialization",
    "NodeResult",
    "OutputSpec",
]
