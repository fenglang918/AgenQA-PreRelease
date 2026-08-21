"""Shared parallel helpers for threaded inference with LLMServiceSession.

Provides a thread-local session pool to avoid cross-thread client contention.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import threading

from .inference import resolve_inference
from .service_client import LLMServiceSession

__all__ = ["ThreadLocalSessionPool"]


class ThreadLocalSessionPool:
    """A tiny helper that constructs a per-thread LLM session from a generator config.

    Usage:
        pool = ThreadLocalSessionPool(generator_config)
        session, chat_args = pool.get()
    """

    def __init__(self, generator_config: Dict[str, Any]):
        self._generator_config = dict(generator_config)
        self._tls = threading.local()

    def get(self) -> Tuple[LLMServiceSession, Dict[str, Any]]:
        sess = getattr(self._tls, "session", None)
        chat_args = getattr(self._tls, "chat_args", None)
        if sess is None or chat_args is None:
            resolved = resolve_inference(self._generator_config)
            self._tls.session = resolved.session
            self._tls.chat_args = dict(resolved.chat_args)
            sess = self._tls.session
            chat_args = self._tls.chat_args
        return sess, chat_args
