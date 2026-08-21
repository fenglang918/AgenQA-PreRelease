"""Shared loader for llm_service service definitions (services.json).

Provides a single place to resolve base_url/api_key/model_name and defaults
into a generator config that `infra.llm.inference.resolve_inference` accepts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json

__all__ = [
    "normalize_api_base",
    "load_llm_service_full_config",
]


def normalize_api_base(raw_base: str) -> str:
    base = (raw_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("Private service API base URL is empty")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def load_llm_service_full_config(
    config_path: Path | str,
    service_id: str,
    *,
    explicit_model: Optional[str] = None,
    fallback_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a full generator config dict for a service_id in services.json.

    The return dict contains keys suitable for `create_llm_service_session`:
      - service_type: "private_endpoint"
      - service_id, api_base, alt_base_urls, api_key, model_name
      - generation: dict
      - client: dict
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"llm_service 配置文件不存在: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    services = data.get("services", [])
    target = next((svc for svc in services if svc.get("service_id") == service_id), None)
    if target is None:
        raise KeyError(f"在 {path} 中未找到 service_id={service_id}")

    base_url = target.get("base_url")
    if not base_url:
        raise ValueError(f"service_id={service_id} 缺少 base_url")
    normalized_base = normalize_api_base(base_url)
    alt_base_urls = target.get("alt_base_urls") or []

    api_key = target.get("api_key")

    meta = target.get("metadata") or {}
    model_name = explicit_model or meta.get("served_model_name") or fallback_model
    if not model_name:
        raise ValueError(
            "无法确定模型名称，请提供显式模型或在 metadata.served_model_name 中配置"
        )

    generation = meta.get("generation") or {}
    client = meta.get("client") or {}

    return {
        "service_type": "private_endpoint",
        "service_id": service_id,
        "api_base": normalized_base,
        "alt_base_urls": alt_base_urls,
        "api_channel": target.get("api_channel") or meta.get("api_channel"),
        "api_key": api_key,
        "model_name": model_name,
        "gateway_routing": meta.get("gateway_routing"),
        "generation": generation,
        "client": client,
    }
