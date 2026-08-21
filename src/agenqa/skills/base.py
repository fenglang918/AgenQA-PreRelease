"""Shared concurrency helpers for skill runners."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, Sequence

from infra.llm.parallel import ThreadLocalSessionPool


logger = logging.getLogger(__name__)


class LLMTruncatedError(ValueError):
    """Raised when the LLM output is truncated (finish_reason=length)."""


class BaseSkillRunner:
    """Provide a ThreadLocalSessionPool and a simple concurrent map helper."""

    def __init__(self, generator_config: Any) -> None:
        self._session_pool = ThreadLocalSessionPool(generator_config)

    @staticmethod
    def _check_finish_reason(response: Any, context: str) -> None:
        """Fail fast if the response is truncated (finish_reason=length)."""
        finish_reason: Any = None
        choices: Any = None

        if isinstance(response, dict):
            choices = response.get("choices")
        else:
            choices = getattr(response, "choices", None)

        if not choices or not isinstance(choices, Sequence):
            return

        first = choices[0] if len(choices) > 0 else None
        if isinstance(first, dict):
            finish_reason = first.get("finish_reason")
        else:
            finish_reason = getattr(first, "finish_reason", None)

        if isinstance(finish_reason, str) and finish_reason.lower() == "length":
            msg = f"{context}: output truncated (finish_reason=length)"
            logger.error(msg)
            raise LLMTruncatedError(msg)

    def _concurrent_map(self, records: Iterable[Any], worker: Callable[[Any], Any], max_workers: int) -> Iterator[Any]:
        """Run worker(record) concurrently and yield results as they complete."""
        max_workers = max(1, int(max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = set()
            for rec in records:
                futures.add(ex.submit(worker, rec))
                if len(futures) >= max_workers:
                    done = next(as_completed(futures))
                    futures.remove(done)
                    yield done.result()
            for done in as_completed(futures):
                yield done.result()


__all__ = ["BaseSkillRunner", "LLMTruncatedError"]
