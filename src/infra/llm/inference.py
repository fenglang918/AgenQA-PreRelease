"""Inference resolution helpers

Centralizes how we construct an LLMServiceSession and decide which request
arguments to pass to `chat()` so that:
- llm_service (services.json) remains the source of truth for best‑practice
  defaults; and
- YAML `generator.generation` keys act as explicit overrides when provided.

This file intentionally keeps a small surface and does not read services.json
by itself; higher layers (CLI) may already have merged a full generator block.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from .service_client import LLMServiceSession, create_llm_service_session


@dataclass
class ResolvedInference:
    """Resolved inference runtime.

    Attributes:
        session: Constructed LLM service session
        chat_args: Keyword args to forward to ApiClient.chat (model + overrides)
        provenance: Diagnostic info about defaults/overrides
    """

    session: LLMServiceSession
    chat_args: Dict[str, Any]
    provenance: Dict[str, Any]


def resolve_inference(generator: Dict[str, Any]) -> ResolvedInference:
    """Create a session and decide what to pass to chat().

    Policy:
    - Always set `model` to the session's model_name for clarity.
    - If `generator.generation` exists (non-empty mapping), pass those keys as
      explicit overrides; otherwise pass nothing and rely on llm_service defaults.
    """

    if not isinstance(generator, dict):
        raise ValueError("generator must be a mapping")

    session = create_llm_service_session(generator)

    # Decide chat args: model is always explicit; generation is opt-in override
    chat_args: Dict[str, Any] = {"model": session.model_name}
    gen_cfg = (generator.get("generation") or {})
    overrides_applied: List[str] = []
    if isinstance(gen_cfg, dict) and gen_cfg:
        for key, val in gen_cfg.items():
            if val is not None:
                chat_args[key] = val
                overrides_applied.append(key)

    provenance = {
        "service_id": session.service_id,
        "model_name": session.model_name,
        "overrides_applied": overrides_applied,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return ResolvedInference(session=session, chat_args=chat_args, provenance=provenance)
