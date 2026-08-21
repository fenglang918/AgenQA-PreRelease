from __future__ import annotations

import hashlib
from typing import Any, Dict


def generate_paper_id(paper: Dict[str, Any]) -> str:
    """Generate a short deterministic ID from paper content.

    Uses first 200 characters of `text` field hashed by MD5, truncated to 12 hex chars.
    """
    text = (paper or {}).get('text', '')[:200]
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
