"""Writers for canonical and framework-specific SFT export artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import CanonicalSFTSample


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_canonical_jsonl(path: Path, samples: list[CanonicalSFTSample]) -> None:
    write_jsonl(path, [sample.to_dict() for sample in samples])


def write_tuningfactory_json(path: Path, samples: list[CanonicalSFTSample]) -> None:
    write_json(path, [sample.to_tuningfactory_alpaca() for sample in samples])
