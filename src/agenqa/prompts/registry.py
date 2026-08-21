from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


# src/agenqa/prompts/registry.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PromptCatalog:
    """Declarative registry of prompts resolved from an agent config YAML.

    作用：
    - 以 agent 配置文件为真相源（如 config/agent_paper_path_v2.yaml），集中表达“当前这一版 AgenQA 实际使用哪些 prompt 文件”；
    - 便于代码与脚本引用，避免在多处硬编码路径常量；
    - 支持为不同 agent 配置构建不同的 PromptCatalog 实例（做版本/环境切换）。
    """

    # Global background（改为 py_style 背景）
    background_zh: Path
    background_en: Path

    # Core roles（从 agent 配置中解析）
    episode_seed_builder: Path
    director: Path
    extend_upgrade: Path
    solver_medium: Path
    solver_strong: Path

    @classmethod
    def from_agent_config(cls, config_path: Path) -> "PromptCatalog":
        """Create a PromptCatalog by reading an agent config YAML.

        仅解析与 prompt 路径相关的字段：
        - init.episode_seed.prompt_path
        - director.prompt_path
        - operators.extend.prompt_path
        - solvers.medium/strong.prompt_path
        """

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid agent config structure in {config_path}")

        def _to_path(p: str | None, default_rel: str) -> Path:
            val = p or default_rel
            path = Path(val)
            if not path.is_absolute():
                path = REPO_ROOT / path
            return path

        init_block: Dict[str, Any] = raw.get("init") or {}
        episode_seed_block: Dict[str, Any] = init_block.get("episode_seed") or {}
        director_block: Dict[str, Any] = raw.get("director") or {}
        ops_block: Dict[str, Any] = raw.get("operators") or {}
        solvers_block: Dict[str, Any] = raw.get("solvers") or {}

        episode_seed_path = _to_path(
            episode_seed_block.get("prompt_path"),
            "src/agenqa/prompts/episode_seed_builder.prompt",
        )
        director_path = _to_path(
            director_block.get("prompt_path"),
            "src/agenqa/prompts/director.prompt",
        )
        extend_block: Dict[str, Any] = ops_block.get("extend") or {}
        extend_path = _to_path(
            extend_block.get("prompt_path"),
            "src/agenqa/prompts/extend_upgrade.prompt",
        )

        medium_block: Dict[str, Any] = solvers_block.get("medium") or {}
        strong_block: Dict[str, Any] = solvers_block.get("strong") or {}
        solver_medium_path = _to_path(
            medium_block.get("prompt_path"),
            "src/agenqa/prompts/solver.prompt",
        )
        solver_strong_path = _to_path(
            strong_block.get("prompt_path"),
            "src/agenqa/prompts/solver.prompt",
        )

        return cls(
            background_zh=REPO_ROOT / "src" / "agenqa" / "prompts" / "background_init.py",
            background_en=REPO_ROOT / "src" / "agenqa" / "prompts" / "background_init.py",
            episode_seed_builder=episode_seed_path,
            director=director_path,
            extend_upgrade=extend_path,
            solver_medium=solver_medium_path,
            solver_strong=solver_strong_path,
        )


# 默认：基于当前“论文→path 多步出题 Agent v2”配置构建一份 Catalog。
DEFAULT_PROMPTS = PromptCatalog.from_agent_config(
    REPO_ROOT / "config" / "agent_paper_path_v2.yaml"
)
