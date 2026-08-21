"""Python-style prompts for shared Draft/Format/Diagnose/Extract roles.

历史上这些 Prompt 被集中放在一个 `roles.py` 中。为避免不同角色逻辑杂糅、便于维护，
现已拆分为多个专用模块（draft/format/diagnose/extract）。

本模块保留为向后兼容入口，仅做符号转发。
"""

from __future__ import annotations

from .draft import DRAFT_V1
from .format import FORMAT_V1
from .diagnose import DIAGNOSE_V1
from .extract import EXTRACT_V1

__all__ = [
    "DRAFT_V1",
    "FORMAT_V1",
    "DIAGNOSE_V1",
    "EXTRACT_V1",
]
