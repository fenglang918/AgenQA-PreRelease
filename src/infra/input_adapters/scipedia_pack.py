from __future__ import annotations

"""SciPedia-style entry packing utilities (code)."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

H2_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
BLANKS_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ScipediaPackMeta:
    include_sections: List[str]
    selected_sections: List[Dict[str, Any]]
    has_h2: bool
    available_h2: List[str]
    lead_chars: int
    pack_chars: int


def _split_h2_sections(text: str) -> Tuple[str, Dict[str, str]]:
    matches = list(H2_RE.finditer(text))
    if not matches:
        return text.strip(), {}

    lead = text[: matches[0].start()].strip()
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections[name] = body
    return lead, sections


def _clean_text(text: str, *, strip_wiki_tokens: bool, normalize_whitespace: bool) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if strip_wiki_tokens:
        s = s.replace("{{", "").replace("}}", "")
    if normalize_whitespace:
        s = BLANKS_RE.sub("\n\n", s)
        s = "\n".join([ln.rstrip() for ln in s.split("\n")]).strip()
    return s.strip()


def build_scipedia_pack(
    *,
    title: str,
    text: str,
    include_sections: List[str],
    strip_wiki_tokens: bool = True,
    normalize_whitespace: bool = True,
    prepend_title: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Build a multi-section packed text for agent-run KnownInit.

    Returns: (pack_text, meta_dict)
    """
    clean = _clean_text(text, strip_wiki_tokens=strip_wiki_tokens, normalize_whitespace=normalize_whitespace)
    lead, sections = _split_h2_sections(clean)

    selected: List[Tuple[str, str]] = []
    for name in include_sections:
        if name == "Introduction":
            intro = "\n\n".join([p for p in [lead, sections.get("Introduction", "")] if p.strip()]).strip()
            if intro:
                selected.append(("Introduction", intro))
            continue
        body = sections.get(name, "").strip()
        if body:
            selected.append((name, body))

    if not selected:
        selected = [("FullText", clean)]

    header_lines: List[str] = []
    header_lines.append("[SourceType] SciPedia textbook-style entry")
    header_lines.append(f"[EntryTitle] {title}")
    header_lines.append("")
    if prepend_title:
        header_lines.append(f"Title: {title}")
        header_lines.append("")

    section_blocks: List[str] = []
    stats_sections: List[Dict[str, Any]] = []
    for name, body in selected:
        section_blocks.append(f'<SECTION name="{name}">\n{body}\n</SECTION>')
        stats_sections.append({"name": name, "orig_chars": len(body), "used_chars": len(body)})

    notes = [
        "",
        "[Notes]",
        "- Prefer “Key Takeaways” as canonical facts; use “Principles and Mechanisms” to derive relationships/symbols.",
        "- Do not invent symbols or assumptions not supported by the provided sections; define symbols before using them.",
        "",
    ]
    pack_text = "\n".join(header_lines + section_blocks + notes).strip() + "\n"

    meta = ScipediaPackMeta(
        include_sections=list(include_sections),
        selected_sections=stats_sections,
        has_h2=bool(sections),
        available_h2=list(sections.keys()),
        lead_chars=len(lead),
        pack_chars=len(pack_text),
    )
    return pack_text, meta.__dict__
