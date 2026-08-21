"""Helpers to normalize/parse/format Known structures."""

from __future__ import annotations

import json
import ast
from typing import Any, Dict, List


def _try_parse_known(val: Any) -> Any:
    """Best-effort parse of a Known string into Python objects."""
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def normalize_known(val: Any, *, max_depth: int = 3) -> str:
    """Canonicalize Known into a clean JSON string (ensure_ascii=False)."""
    parsed = val
    depth = 0
    while isinstance(parsed, str) and depth < max_depth:
        maybe = _try_parse_known(parsed)
        if maybe is None:
            break
        parsed = maybe
        depth += 1
    if isinstance(parsed, dict):
        try:
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
    return "" if parsed is None else str(parsed)


def parse_known_to_dict(val: Any, *, max_depth: int = 3) -> Dict[str, Any] | None:
    """Parse Known into dict if possible, tolerating nested repr strings."""
    parsed = val
    depth = 0
    while isinstance(parsed, str) and depth < max_depth:
        maybe = _try_parse_known(parsed)
        if maybe is None:
            break
        parsed = maybe
        depth += 1
    if isinstance(parsed, dict):
        return parsed
    return None


def format_known_for_solver(
    val: Any,
    *,
    include_seed_meta: bool = False,
    include_fact_bank: bool = True,
    include_step_certs: bool = True,
) -> str:
    """Format a Known payload into a solver-friendly text block.

    - Accepts either a JSON string, a dict, or arbitrary text.
    - Prefers KnownTree v2 fields when present.
    - Avoids leaking internal provenance/tool artifacts by using a strict allow-list.
    """

    def _as_clean_str(x: Any) -> str:
        if x is None:
            return ""
        s = x.strip() if isinstance(x, str) else str(x).strip()
        return s

    def _flatten_v1(known_obj: Any) -> str:
        if not isinstance(known_obj, dict):
            return _as_clean_str(known_obj)
        parts: List[str] = []
        k0 = known_obj.get("known_0")
        if isinstance(k0, str) and k0.strip():
            parts.append(f"Known_0: {k0.strip()}")
        bg = known_obj.get("background")
        if isinstance(bg, list):
            for idx, item in enumerate(bg):
                txt = _as_clean_str(item)
                if txt:
                    parts.append(f"Background_{idx}: {txt}")
        hist = known_obj.get("history")
        if isinstance(hist, list):
            for idx, item in enumerate(hist):
                if not isinstance(item, dict):
                    continue
                q = next((v for k, v in item.items() if isinstance(k, str) and k.startswith("question_")), None)
                a = next((v for k, v in item.items() if isinstance(k, str) and k.startswith("answer_")), None)
                q_txt = _as_clean_str(q)
                a_txt = _as_clean_str(a)
                if q_txt:
                    parts.append(f"Q_{idx}: {q_txt}")
                if a_txt:
                    parts.append(f"A_{idx}: {a_txt}")
        df = known_obj.get("derived_facts")
        if isinstance(df, list):
            for idx, item in enumerate(df):
                txt = _as_clean_str(item)
                if txt:
                    parts.append(f"DerivedFact_{idx}: {txt}")
        return "\n".join(parts)

    parsed: Any = val
    if isinstance(parsed, str):
        parsed_dict = parse_known_to_dict(parsed)
        if parsed_dict is None:
            return parsed.strip()
        parsed = parsed_dict

    if not isinstance(parsed, dict):
        return _as_clean_str(parsed)

    # Prefer v2 structure when it looks like KnownTree.
    if any(k in parsed for k in ("episode_seed", "premise_bank", "fact_bank", "step_certs")):
        parts: List[str] = []

        if include_seed_meta:
            seed = parsed.get("episode_seed")
            if isinstance(seed, dict):
                subject = _as_clean_str(seed.get("subject"))
                if subject:
                    parts.append(f"Subject: {subject}")
                keywords = seed.get("keywords")
                if isinstance(keywords, list):
                    kw = [str(x).strip() for x in keywords if str(x).strip()]
                    if kw:
                        parts.append("Keywords: " + ", ".join(kw))

        premise_bank = parsed.get("premise_bank")
        if isinstance(premise_bank, list):
            for idx, item in enumerate(premise_bank):
                if not isinstance(item, dict):
                    continue
                txt = _as_clean_str(item.get("text") or item.get("statement") or "")
                if txt:
                    parts.append(f"Premise_{idx}: {txt}")

        if include_fact_bank:
            fact_bank = parsed.get("fact_bank")
            if isinstance(fact_bank, list):
                for idx, item in enumerate(fact_bank):
                    if not isinstance(item, dict):
                        continue
                    txt = _as_clean_str(item.get("statement") or item.get("text") or "")
                    if txt:
                        parts.append(f"Fact_{idx}: {txt}")

        if include_step_certs:
            step_certs = parsed.get("step_certs")
            if isinstance(step_certs, list):
                for idx, item in enumerate(step_certs):
                    if not isinstance(item, dict):
                        continue
                    cert_text = _as_clean_str(item.get("cert_text") or item.get("text") or "")
                    step = item.get("step")
                    key_fact_id = item.get("key_fact_id")
                    meta: List[str] = []
                    if step is not None:
                        meta.append(f"step={step}")
                    kfi = _as_clean_str(key_fact_id)
                    if kfi:
                        meta.append(f"key={kfi}")
                    if cert_text:
                        meta.append(cert_text)
                    if meta:
                        parts.append(f"Cert_{idx}: " + " | ".join(meta))

        if parts:
            return "\n".join(parts)

    # Fallback: legacy known_0 structure.
    return _flatten_v1(parsed)


__all__ = ["normalize_known", "parse_known_to_dict", "format_known_for_solver"]
