"""Prompt building helpers for agent nodes.

本模块已迁移到 src/agenqa/prompts/ 目录下。
Director prompt 相关代码已移至 src/agenqa/prompts/director.py。

此文件保留仅为向后兼容，建议直接使用 agenqa.prompts.director。
"""

from __future__ import annotations

# 向后兼容：从新位置导入
from agenqa.prompts.director import (
    DIRECTOR_ROLE_SECTION,
    DIRECTOR_STATE_SECTION,
    DIRECTOR_ACTIONS_SECTION,
    DIRECTOR_OUTPUT_SECTION,
    DIRECTOR_TEMPLATE,
    build_director_v1_body,
)

__all__ = [
    "DIRECTOR_ROLE_SECTION",
    "DIRECTOR_STATE_SECTION",
    "DIRECTOR_ACTIONS_SECTION",
    "DIRECTOR_OUTPUT_SECTION",
    "DIRECTOR_TEMPLATE",
    "build_director_v1_body",
]
