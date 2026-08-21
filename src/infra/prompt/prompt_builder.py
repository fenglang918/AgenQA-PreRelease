"""Prompt building utilities using LangChain ChatPromptTemplate.

This module centralizes how we wrap role-specific prompt bodies with a shared
AgenQA background (system message), so that:
- Background/语言要求在一个地方维护；
- 各角色仅关注各自的 Task/Input/Output 约定；
- LangGraph 节点统一接受 OpenAI 风格的 messages 列表。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


_PROMPT_BUILDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    _PROMPT_BUILDER_DIR.parent.parent
    if _PROMPT_BUILDER_DIR.parent.name == "src"
    else _PROMPT_BUILDER_DIR.parent
)
# md_style 目录已移除，这里保留占位路径仅为兼容旧配置；实际加载走 py_style。
BACKGROUND_DIR = REPO_ROOT / "prompts" / "md_style" / "background"
FRAGMENTS_DIR = REPO_ROOT / "prompts" / "md_style" / "fragments"


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_background_text(lang: str | None) -> str:
    """Load shared background prompt.

    Note: background.py has been removed. This function now returns an empty string.
    System-level information should be included in individual role prompts as needed.
    """
    return ""


def load_prompt_fragment(name: str, lang: str | None = None) -> str:
    """Load a reusable prompt fragment by name（统一使用 py_style 定义）。"""
    try:
        from agenqa.prompts.common import (
            COMMON_ANSWER_SCHEMA,
            COMMON_ANSWER_SCHEMA_EN,
            COMMON_EXTEND_CONSTRAINTS,
            COMMON_EXTEND_CONSTRAINTS_EN,
            COMMON_KNOWN_TREE,
            COMMON_KNOWN_TREE_EN,
            COMMON_QUESTION_TYPES,
            COMMON_QUESTION_TYPES_EN,
            COMMON_SOLUTION_SCHEMA,
            COMMON_SOLUTION_SCHEMA_EN,
        )
    except Exception:
        COMMON_ANSWER_SCHEMA = COMMON_EXTEND_CONSTRAINTS = COMMON_KNOWN_TREE = COMMON_QUESTION_TYPES = COMMON_SOLUTION_SCHEMA = None  # type: ignore[assignment]
        COMMON_ANSWER_SCHEMA_EN = COMMON_EXTEND_CONSTRAINTS_EN = COMMON_KNOWN_TREE_EN = COMMON_QUESTION_TYPES_EN = COMMON_SOLUTION_SCHEMA_EN = None  # type: ignore[assignment]

    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}

    if name == "answer_schema" and COMMON_ANSWER_SCHEMA is not None:
        if use_en and COMMON_ANSWER_SCHEMA_EN is not None:
            return COMMON_ANSWER_SCHEMA_EN.text
        return COMMON_ANSWER_SCHEMA.text
    if name == "extend_constraints" and COMMON_EXTEND_CONSTRAINTS is not None:
        if use_en and COMMON_EXTEND_CONSTRAINTS_EN is not None:
            return COMMON_EXTEND_CONSTRAINTS_EN.text
        return COMMON_EXTEND_CONSTRAINTS.text
    if name == "known_tree" and COMMON_KNOWN_TREE is not None:
        if use_en and COMMON_KNOWN_TREE_EN is not None:
            return COMMON_KNOWN_TREE_EN.text
        return COMMON_KNOWN_TREE.text
    if name == "question_types" and COMMON_QUESTION_TYPES is not None:
        if use_en and COMMON_QUESTION_TYPES_EN is not None:
            return COMMON_QUESTION_TYPES_EN.text
        return COMMON_QUESTION_TYPES.text
    if name in {"path_solution", "path_fold_solution"} and COMMON_SOLUTION_SCHEMA is not None:
        if use_en and COMMON_SOLUTION_SCHEMA_EN is not None:
            return COMMON_SOLUTION_SCHEMA_EN.text
        return COMMON_SOLUTION_SCHEMA.text

    # 若未来需要自定义文件片段，可按原有路径查找；当前缺省返回空字符串。
    candidates = []
    if lang_norm:
        candidates.append(FRAGMENTS_DIR / f"{name}_{lang_norm}.md")
    candidates.append(FRAGMENTS_DIR / f"{name}.md")
    for path in candidates:
        if path.exists():
            try:
                return _load_text(path)
            except Exception:
                continue
    return ""


def _lc_to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain messages to OpenAI-style dict messages."""
    result: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            # fallback to LC message type
            role = getattr(m, "type", "user")
        content = m.content
        if not isinstance(content, str):
            content = str(content)
        result.append({"role": role, "content": content})
    return result


def build_messages_with_background(body: str, *, lang: str | None) -> List[Dict[str, str]]:
    """Wrap a single user prompt body with shared background into chat messages.

    - `body` 是已经按需求渲染好的用户侧内容（可包含 JSON 示例等）；
    - `lang` 控制选择中文/英文背景与语言说明。
    """
    background = load_background_text(lang)
    message_defs = []
    if isinstance(background, str) and background.strip():
        message_defs.append(("system", background))
    message_defs.append(("user", "{body}"))
    prompt = ChatPromptTemplate.from_messages(message_defs)
    lc_messages = prompt.format_messages(body=body)
    return _lc_to_openai_messages(lc_messages)
