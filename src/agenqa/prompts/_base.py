"""Base prompt building classes (no external dependencies).

These classes are shared across py_style modules and must not import from
agenqa.* to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptSection:
    """A small, reusable fragment of a prompt body.

    text 内部可以使用 str.format(**ctx) 占位符；缺失字段时会回退为原文，
    避免因为上下文不完整导致运行时异常。
    """

    text: str
    # Optional structured metadata for safe prompt layout:
    # - title/title_level render the heading automatically, so authors don't have to hardcode `#`/`##`.
    # - kind is a semantic tag used for ordering/validation (e.g. "cognition", "info").
    title: Optional[str] = None
    title_level: Optional[int] = None
    kind: Optional[str] = None

    def render(self, ctx: Dict[str, Any]) -> str:
        try:
            return self.text.format(**ctx)
        except Exception:
            return self.text

    def render_block(self, ctx: Dict[str, Any]) -> str:
        body = self.render(ctx).rstrip()
        if not (isinstance(self.title, str) and self.title.strip()):
            return body

        level = self.title_level if isinstance(self.title_level, int) else 2
        if level < 1:
            level = 1
        if level > 6:
            level = 6

        try:
            title_txt = self.title.format(**ctx).strip()
        except Exception:
            title_txt = self.title.strip()
        title_txt = " ".join(title_txt.splitlines()).strip()
        heading = f"{'#' * level} {title_txt}".rstrip()

        if body:
            return f"{heading}\n\n{body}"
        return heading


@dataclass
class PromptTemplate:
    """Composable template made of ordered sections."""

    name: str
    sections: List[PromptSection]

    def validate_structure(self, *, required_kind_order: Optional[List[str]] = None) -> None:
        """Validate structured prompt layout.

        This is intentionally lightweight and only checks:
        - title_level range (when title exists)
        - (optional) required kind order monotonicity
        """
        # 1) Heading level sanity.
        for s in self.sections:
            if not (isinstance(s.title, str) and s.title.strip()):
                continue
            if s.title_level is None:
                continue
            if not isinstance(s.title_level, int) or not (1 <= s.title_level <= 6):
                raise ValueError(f"{self.name}: invalid title_level={s.title_level!r} for title={s.title!r}")

        # 2) Kind order sanity (if provided).
        if required_kind_order:
            kind_to_first_idx: Dict[str, int] = {}
            for idx, s in enumerate(self.sections):
                if isinstance(s.kind, str) and s.kind.strip() and s.kind not in kind_to_first_idx:
                    kind_to_first_idx[s.kind] = idx
            missing = [k for k in required_kind_order if k not in kind_to_first_idx]
            if missing:
                raise ValueError(f"{self.name}: missing required kinds: {missing}")
            idxs = [kind_to_first_idx[k] for k in required_kind_order]
            if idxs != sorted(idxs):
                raise ValueError(f"{self.name}: kind order violated: {required_kind_order} -> {idxs}")

    def render_body(self, ctx: Dict[str, Any]) -> str:
        """Render all sections into a single user-side prompt body."""
        parts: List[str] = []
        for section in self.sections:
            rendered = section.render_block(ctx).rstrip()
            if rendered:
                parts.append(rendered)
        # 使用空行分隔各逻辑段落，便于阅读与后续维护
        return "\n\n".join(parts)
