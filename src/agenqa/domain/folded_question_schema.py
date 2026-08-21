"""Folded question payload schema (code truth source).

This module defines the canonical, typed JSON payload stored in:
- KQARecord.path_question_direct / path_question_scaffolded
- ExecutableRecord.path_question_direct / path_question_scaffolded

Design goals:
- Single discriminated union for both tracks (semantic/executable).
- Stable on-disk representation (JSON string) so state.json is self-contained.
- Downstream consumers (dump_path_*, evaluators) can parse deterministically.

Note: This schema is *not* the PathFold role output schema. PathFold roles may
still output raw strings; the pipeline wraps them into this structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


FoldTrack = Literal["unified", "semantic", "executable"]
FoldVariant = Literal["direct", "scaffolded"]

FOLDED_QUESTION_KIND = "folded_question"
FOLDED_QUESTION_SCHEMA_VERSION = 1

FIELD_KIND = "kind"
FIELD_SCHEMA_VERSION = "schema_version"
FIELD_TRACK = "track"
FIELD_VARIANT = "variant"
FIELD_PAYLOAD = "payload"

# Semantic payload fields
FIELD_QUESTION_TEXT = "question_text"


@dataclass(frozen=True)
class FoldedQuestion:
    track: FoldTrack
    variant: FoldVariant
    payload: Dict[str, Any]
    schema_version: int = FOLDED_QUESTION_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            FIELD_KIND: FOLDED_QUESTION_KIND,
            FIELD_SCHEMA_VERSION: int(self.schema_version),
            FIELD_TRACK: self.track,
            FIELD_VARIANT: self.variant,
            FIELD_PAYLOAD: dict(self.payload),
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        # Canonical on-disk representation lives inside `state.json` as a string, so we
        # prefer compact JSON by default. Human-readable snapshots should be written
        # as separate sidecar files (indent=2) derived from this canonical payload.
        if indent is None:
            return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def dumps_folded_question(
    *,
    track: FoldTrack,
    variant: FoldVariant,
    payload: Dict[str, Any],
    indent: Optional[int] = None,
) -> str:
    """Serialize a FoldedQuestion to JSON string."""
    return FoldedQuestion(track=track, variant=variant, payload=payload).to_json(indent=indent)


def _is_variant(v: Any) -> bool:
    return isinstance(v, str) and v.strip() in {"direct", "scaffolded"}


def loads_folded_question(text: Any) -> FoldedQuestion:
    """Parse a FoldedQuestion from JSON string (strict, no legacy fallback)."""
    if not (isinstance(text, str) and text.strip()):
        raise ValueError("folded_question text is empty")
    try:
        obj = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("folded_question is not valid JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("folded_question JSON must be an object")

    kind = obj.get(FIELD_KIND)
    if kind != FOLDED_QUESTION_KIND:
        raise ValueError(f"folded_question.kind mismatch: {kind!r}")
    schema_version = obj.get(FIELD_SCHEMA_VERSION)
    try:
        schema_version_i = int(schema_version)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("folded_question.schema_version must be int") from exc
    if schema_version_i != FOLDED_QUESTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported folded_question.schema_version={schema_version_i}")

    track = obj.get(FIELD_TRACK)
    if track not in {"unified", "semantic", "executable"}:
        raise ValueError(f"folded_question.track must be 'unified'|'semantic'|'executable', got {track!r}")

    variant = obj.get(FIELD_VARIANT)
    if not _is_variant(variant):
        raise ValueError(f"folded_question.variant must be 'direct'|'scaffolded', got {variant!r}")

    payload = obj.get(FIELD_PAYLOAD)
    if not isinstance(payload, dict):
        raise ValueError("folded_question.payload must be an object")

    return FoldedQuestion(
        track=track,  # type: ignore[arg-type]
        variant=variant,  # type: ignore[arg-type]
        payload=payload,
        schema_version=schema_version_i,
    )


def extract_semantic_question_text(text: Any, *, variant: Optional[FoldVariant] = None) -> str:
    fq = loads_folded_question(text)
    if fq.track not in {"unified", "semantic"}:
        raise ValueError(f"expected unified/semantic folded_question, got track={fq.track!r}")
    if variant is not None and fq.variant != variant:
        raise ValueError(f"expected variant={variant!r}, got {fq.variant!r}")
    q = fq.payload.get(FIELD_QUESTION_TEXT)
    if not (isinstance(q, str) and q.strip()):
        raise ValueError("semantic folded_question.payload.question_text missing")
    return q.strip()


def extract_executable_payload(text: Any, *, variant: Optional[FoldVariant] = None) -> Dict[str, Any]:
    fq = loads_folded_question(text)
    if fq.track != "executable":
        raise ValueError(f"expected executable folded_question, got track={fq.track!r}")
    if variant is not None and fq.variant != variant:
        raise ValueError(f"expected variant={variant!r}, got {fq.variant!r}")
    return fq.payload


__all__ = [
    "FoldTrack",
    "FoldVariant",
    "FoldedQuestion",
    "FOLDED_QUESTION_KIND",
    "FOLDED_QUESTION_SCHEMA_VERSION",
    "FIELD_QUESTION_TEXT",
    "dumps_folded_question",
    "loads_folded_question",
    "extract_semantic_question_text",
    "extract_executable_payload",
]
