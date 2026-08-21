"""Init 配置（best-practice first）。

本模块用于把“best practice 的 unified/paper-like init”从历史配置入口中解耦出来，
让 init 的输入与 generator/brief 行为在同一个配置块里自洽、可追踪、可校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class InitSourceConfig:
    """Init source config for unified init (KnownInit).

    Supported source types (best-practice):
    - paper: load a paper-like record from init.source.path
    - domain_seed_walk: generate a synthetic "material text" by walking a domain taxonomy
    """

    type: str  # "paper" | "domain_seed_walk"
    path: Optional[Path] = None
    pdf_extract: Dict[str, Any] = field(default_factory=dict)
    scipedia_pack: Dict[str, Any] = field(default_factory=dict)
    domain_seed_walk: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitEpisodeSeedConfig:
    """EpisodeSeedBuilder config (build episode_seed directly from paper text)."""

    enable: bool = True
    prompt_path: Optional[Path] = None
    generator: Optional[Dict[str, Any]] = None
    contract_path: Optional[Path] = None
    contract: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitConfig:
    source: InitSourceConfig
    episode_seed: InitEpisodeSeedConfig
    generator: Dict[str, Any]


def _as_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _require_str(val: Any, *, key: str) -> str:
    s = str(val or "").strip()
    if not s:
        raise ValueError(f"init config missing required field: {key}")
    return s


def assert_no_legacy_init_config(agent_conf: Dict[str, Any]) -> None:
    if not isinstance(agent_conf, dict):
        return
    qa_init = agent_conf.get("qa_init")
    if isinstance(qa_init, dict) and qa_init:
        raise ValueError("Legacy config 'qa_init' is no longer supported. Use top-level init.* instead.")
    ops = agent_conf.get("operators")
    if isinstance(ops, dict) and isinstance(ops.get("init"), dict) and ops.get("init"):
        raise ValueError("Legacy config 'operators.init' is no longer supported. Use top-level init.* instead.")
    data_conf = agent_conf.get("data")
    if isinstance(data_conf, dict) and data_conf.get("papers_path"):
        raise ValueError("Legacy config 'data.papers_path' is no longer supported. Use init.source.path instead.")




def normalize_paper_brief_mode(raw: Any, *, default: str = "subject-keywords") -> str:
    """Normalize init.paper_brief.version into a descriptive mode name.

    Canonical modes (by output field groups):
    - subject-keywords: extract minimal episode_seed fields (subject, keywords)
    - subject-keywords-brief: extract (subject, keywords, brief)
    - subject-keywords-skeleton: extract reasoning skeleton fields (V2 schema)

    Notes:
    - Legacy values v1/v2/v3 are rejected to enforce one-shot migration.
    - Earlier short names (seed_only/reasoning/brief) are also rejected.
    """
    s = str(raw or "").strip().lower()
    if not s:
        s = str(default or "subject-keywords").strip().lower() or "subject-keywords"

    # normalize separators for matching
    s_norm = s.replace("_", "-").replace(" ", "-")

    legacy_map = {
        "v1": "subject-keywords-brief",
        "v2": "subject-keywords-skeleton",
        "v3": "subject-keywords",
        "seed-only": "subject-keywords",
        "seed_only": "subject-keywords",
        "reasoning": "subject-keywords-skeleton",
        "brief": "subject-keywords-brief",
    }
    if s_norm in legacy_map:
        raise ValueError(
            f"Legacy init.paper_brief.version={s!r} is no longer supported; use {legacy_map[s_norm]!r} instead."
        )

    if s_norm in {"subject-keywords", "subject-keyword"}:
        return "subject-keywords"
    if s_norm in {"subject-keywords-brief", "subject-keywords-summary", "subject-keywords-compact"}:
        return "subject-keywords-brief"
    if s_norm in {
        "subject-keywords-skeleton",
        "subject-keywords-reasoning",
        "subject-keywords-reasoning-skeleton",
    }:
        return "subject-keywords-skeleton"

    raise ValueError(
        f"Unsupported init.paper_brief.version={s!r}. Allowed: 'subject-keywords', 'subject-keywords-brief', 'subject-keywords-skeleton'."
    )


def parse_init_config(agent_conf: Dict[str, Any]) -> InitConfig:
    """Parse best-practice init config.

    Scope:
    - unified/semantic track only (KnownInit)
    - paper-like source OR domain_seed_walk source
    - no cross-module generator fallback
    """
    assert_no_legacy_init_config(agent_conf)
    init_block = _as_dict(agent_conf.get("init"))
    if not init_block:
        raise ValueError("Missing top-level `init` config block (best-practice init requires it).")

    source_block = _as_dict(init_block.get("source"))
    source_type = str(source_block.get("type") or "paper").strip().lower() or "paper"
    source: InitSourceConfig
    if source_type in {"paper", "paper-like", "paperlike"}:
        path_raw = source_block.get("path")
        path_str = _require_str(path_raw, key="init.source.path")
        source = InitSourceConfig(
            type="paper",
            path=Path(path_str),
            pdf_extract=_as_dict(source_block.get("pdf_extract")),
            scipedia_pack=_as_dict(source_block.get("scipedia_pack")),
            domain_seed_walk={},
        )
    elif source_type in {"domain_seed_walk", "domain-seed-walk", "domain"}:
        dsw = _as_dict(source_block.get("domain_seed_walk"))
        root_domain = dsw.get("root_domain")
        if not isinstance(root_domain, str) or not root_domain.strip():
            raise ValueError("Missing init.source.domain_seed_walk.root_domain (required for domain_seed_walk).")
        if source_block.get("path") not in (None, "", {}):
            raise ValueError("init.source.path must be omitted when init.source.type=domain_seed_walk.")
        source = InitSourceConfig(
            type="domain_seed_walk",
            path=None,
            pdf_extract={},
            scipedia_pack={},
            domain_seed_walk=dsw,
        )
    else:
        raise ValueError(
            f"Unsupported init.source.type={source_type!r} (best-practice supports 'paper' or 'domain_seed_walk')."
        )

    gen = _as_dict(init_block.get("generator"))
    if not gen:
        raise ValueError("Missing init.generator (best-practice init does not fallback to other modules).")

    if isinstance(init_block.get("paper_brief"), dict) and (init_block.get("paper_brief") or {}):
        raise ValueError("init.paper_brief 已移除；请使用 init.episode_seed.contract(_path) 定义单一 contract。")

    es_block = _as_dict(init_block.get("episode_seed"))
    es_prompt_path = es_block.get("prompt_path")
    contract_path_raw = es_block.get("contract_path")
    contract_inline = es_block.get("contract")
    validation_block = _as_dict(es_block.get("validation"))

    legacy_keys = ("required_fields", "llm_output_fields", "seed_view_fields", "anchor_template", "keywords_target", "keywords_min", "keywords_max", "optional_fields")
    for key in legacy_keys:
        if key in es_block and es_block.get(key) not in (None, "", [], {}, ()):
            raise ValueError(f"init.episode_seed.{key} 已移除；请用 contract.output_schema.required 表达必需字段。")

    def _load_contract(path: Path) -> Dict[str, Any]:
        raw_text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            raise ValueError(f"Contract file {path} must be a mapping/dict.")
        if isinstance(data.get("contract"), dict):
            data = data["contract"]
        if not isinstance(data, dict):
            raise ValueError(f"Contract file {path} has invalid structure.")
        return data

    contract_path = None
    if isinstance(contract_path_raw, str) and contract_path_raw.strip():
        contract_path = Path(contract_path_raw.strip())
    elif isinstance(contract_path_raw, Path):
        contract_path = contract_path_raw

    if contract_path and contract_inline:
        raise ValueError("init.episode_seed.contract 与 contract_path 不可同时配置，请二选一。")

    contract: Dict[str, Any] = {}
    if contract_path:
        contract_load_path = contract_path
        if not contract_load_path.exists() and not contract_load_path.is_absolute():
            # Be robust to different working directories (repo root vs src/ etc.).
            # NOTE: we still fail-fast if the file cannot be found deterministically.
            repo_root = Path(__file__).resolve().parents[2]
            alt = repo_root / contract_load_path
            if alt.exists():
                contract_load_path = alt

        if not contract_load_path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            alt = repo_root / contract_path if not contract_path.is_absolute() else None
            extra = f" (also tried: {alt})" if alt is not None else ""
            raise ValueError(f"Contract file not found: {contract_path}{extra}")
        contract = _load_contract(contract_load_path)
    elif isinstance(contract_inline, dict) and contract_inline:
        contract = contract_inline

    if not contract:
        raise ValueError("init.episode_seed.contract 或 contract_path 必须提供。")

    instruction = contract.get("instruction")
    output_schema = contract.get("output_schema")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("contract.instruction 不能为空字符串。")
    if not isinstance(output_schema, dict) or not output_schema:
        raise ValueError("contract.output_schema 必须是非空 JSON Schema dict。")
    contract = dict(contract)
    contract["instruction"] = instruction.strip()
    contract["output_schema"] = output_schema

    episode_seed = InitEpisodeSeedConfig(
        enable=bool(es_block.get("enable", True)),
        prompt_path=Path(str(es_prompt_path).strip())
        if isinstance(es_prompt_path, str) and str(es_prompt_path).strip()
        else None,
        generator=es_block.get("generator") if isinstance(es_block.get("generator"), dict) and es_block.get("generator") else None,
        contract_path=contract_path,
        contract=contract,
        validation=validation_block,
    )

    return InitConfig(source=source, episode_seed=episode_seed, generator=gen)


__all__ = [
    "InitConfig",
    "InitSourceConfig",
    "InitEpisodeSeedConfig",
    "parse_init_config",
    "assert_no_legacy_init_config",
    "normalize_paper_brief_mode",
]
