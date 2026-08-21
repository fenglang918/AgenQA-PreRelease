"""Small utilities for extracting fenced code blocks from LLM outputs.

Why this exists:
- Many skills ask models to wrap JSON in ```json ... ``` fences.
- The JSON payload itself can legitimately contain the substring ``` (e.g., when
  a field embeds a Markdown code block). Naive `str.find("```")` extraction will
  truncate at the *first* occurrence and break JSON parsing.

These helpers only treat fences that appear as standalone lines as fence markers,
which avoids being confused by backticks inside JSON strings.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

_FENCE_LINE_RE = re.compile(r"^\s*```(?P<lang>[A-Za-z0-9_-]+)?\s*$")


def extract_fenced_blocks(text: str) -> List[Tuple[str, str]]:
    """Return all fenced blocks as (lang, content) tuples.

    - `lang` is lowercased ("" if not provided).
    - If a fence is opened but not closed, the remainder of the text is returned
      as the content for that block.
    """
    if not isinstance(text, str) or not text:
        return []

    lines = text.splitlines()
    blocks: List[Tuple[str, str]] = []
    in_block = False
    lang = ""
    buf: List[str] = []

    for line in lines:
        m = _FENCE_LINE_RE.match(line)
        if not in_block:
            if m:
                in_block = True
                lang = (m.group("lang") or "").strip().lower()
                buf = []
            continue

        # In block.
        if m:
            blocks.append((lang, "\n".join(buf)))
            in_block = False
            lang = ""
            buf = []
            continue

        buf.append(line)

    if in_block:
        blocks.append((lang, "\n".join(buf)))

    return blocks


def extract_preferred_fenced_block(
    text: str, *, preferred_langs: Sequence[str] = ("json",)
) -> Optional[str]:
    """Extract a fenced block, preferring a specific language label if present."""
    blocks = extract_fenced_blocks(text)
    if not blocks:
        return None

    prefs = [p.strip().lower() for p in preferred_langs if isinstance(p, str) and p.strip()]
    for pref in prefs:
        for lang, content in blocks:
            if lang == pref:
                return content.strip()

    # Fallback: first block.
    return blocks[0][1].strip()
