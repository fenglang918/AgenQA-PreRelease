"""Utilities for repairing near-JSON emitted by LLMs.

This project often requests strict JSON, but some models occasionally return:
- literal newlines/tabs inside JSON strings (invalid control characters)
- single backslashes for LaTeX commands (e.g. \tau, \frac) which get interpreted
  as JSON escapes (tab/formfeed) or invalid escapes

These helpers attempt a conservative, no-dependency sanitization so that the
output becomes valid JSON while preserving the original intended text as much
as possible.
"""

from __future__ import annotations

from typing import Optional


_HEX = set("0123456789abcdefABCDEF")


def sanitize_json_text(payload: str) -> str:
    """Best-effort sanitize JSON text so `json.loads()` is more likely to succeed.

    - Escapes literal control characters inside strings as \\n/\\t/\\r or \\uXXXX.
    - Escapes backslashes inside strings when they are not valid JSON escapes.
    - Treats common LaTeX sequences like `\\tau` / `\\frac` typed as `\tau` / `\frac`
      as literal backslashes (i.e. converts to `\\tau` / `\\frac` in JSON source).
    - Escapes stray double-quotes inside strings when they don't look like a string terminator.
    """
    if not isinstance(payload, str) or not payload:
        return payload

    out: list[str] = []
    in_str = False
    i = 0
    n = len(payload)

    while i < n:
        ch = payload[i]

        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue

        # In string.
        if ch == '"':
            # Heuristic: some models emit unescaped quotes inside JSON strings, e.g.
            #   "Solution": "DELFI 属于"无似然"（...）"
            # which breaks strict JSON. If the next non-whitespace character doesn't
            # look like a valid post-string token, treat this quote as literal.
            j = i + 1
            while j < n and payload[j] in {" ", "\t", "\r", "\n"}:
                j += 1
            if j < n and payload[j] not in {",", "}", "]", ":"}:
                out.append('\\"')
                i += 1
                continue
            out.append(ch)
            in_str = False
            i += 1
            continue

        if ch == "\\":
            nxt: Optional[str] = payload[i + 1] if i + 1 < n else None
            if nxt is None:
                out.append("\\\\")
                i += 1
                continue

            if nxt in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                # Preserve JSON escapes, except when it looks like a LaTeX command
                # that the model forgot to double-escape, e.g. \tau or \frac.
                if nxt in {"b", "f", "n", "r", "t"} and (i + 2) < n and payload[i + 2].isalpha():
                    out.append("\\\\")
                    out.append(nxt)
                    i += 2
                    continue
                out.append("\\")
                out.append(nxt)
                i += 2
                continue

            if nxt == "u":
                # Keep unicode escape if it is well-formed; otherwise treat as literal "\u...".
                if i + 5 < n and all(c in _HEX for c in payload[i + 2 : i + 6]):
                    out.append("\\u")
                    out.append(payload[i + 2 : i + 6])
                    i += 6
                    continue
                out.append("\\\\")
                out.append("u")
                i += 2
                continue

            # Unknown escape: escape the backslash itself.
            out.append("\\\\")
            out.append(nxt)
            i += 2
            continue

        code = ord(ch)
        if code < 0x20:
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(f"\\u{code:04x}")
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)
