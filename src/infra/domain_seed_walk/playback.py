"""Generate playback markdown for a domain_seed_walk run directory.

Usage:
  python -m infra.domain_seed_walk.playback --run-dir data/domain_seed_walk/run_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _truncate(text: str, *, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 1] + "…", True


def _collect_level_dirs(run_dir: Path) -> List[Path]:
    return sorted([p for p in run_dir.glob("level_*") if p.is_dir()])


def _render_list(items: List[str]) -> List[str]:
    return [f"- {it}" for it in items]


def _render_candidates(candidates: List[str], chosen: str) -> List[str]:
    lines: List[str] = []
    for c in candidates:
        mark = " **(chosen)**" if c == chosen else ""
        lines.append(f"- {c}{mark}")
    return lines


def _candidate_name(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("name") or obj.get("subdomain") or "").strip()
    return str(obj or "").strip()


def _candidate_tags(obj: Any) -> List[str]:
    if not isinstance(obj, dict):
        return []
    tags = obj.get("context_tags") or obj.get("tags") or []
    if isinstance(tags, str) and tags.strip():
        return [x.strip() for x in tags.split(",") if x.strip()]
    if isinstance(tags, list):
        return [str(x).strip() for x in tags if str(x).strip()]
    return []


def _render_candidates_any(candidates: List[Any], chosen: str) -> List[str]:
    lines: List[str] = []
    for c in candidates:
        name = _candidate_name(c)
        if not name:
            continue
        tags = _candidate_tags(c)
        tags_part = f" (tags: {', '.join(tags)})" if tags else ""
        mark = " **(chosen)**" if name == chosen else ""
        lines.append(f"- {name}{tags_part}{mark}")
    return lines


def _safe_path(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except Exception:
        return str(path)


def build_playback(run_dir: Path, *, embed_preview: bool = True, preview_chars: int = 0) -> str:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found under {run_dir}")

    result = _read_json(result_path)
    lang = "en"
    if isinstance(result, dict):
        raw_lang = result.get("lang")
        if isinstance(raw_lang, str) and raw_lang.strip().lower() in {"zh", "en"}:
            lang = raw_lang.strip().lower()

    def t(en: str, zh: str) -> str:
        return zh if lang == "zh" else en

    path_trace = result.get("path_trace") if isinstance(result, dict) else None
    if not isinstance(path_trace, list):
        path_trace = []

    lines: List[str] = []
    lines.append(f"# {t('Domain Seed Walk Playback', 'Domain Seed Walk 回放')}")
    lines.append("")
    lines.append(f"- **{t('Run Dir', '运行目录')}**: `{run_dir}`")
    if isinstance(result, dict):
        ts = result.get("timestamp")
        if isinstance(ts, str) and ts.strip():
            lines.append(f"- **{t('Timestamp', '时间戳')}**: `{ts}`")
        lines.append(f"- **{t('Root Domain', '根域')}**: `{result.get('root_domain', 'N/A')}`")
        lines.append(f"- **{t('Depth', '深度')}**: `{result.get('depth', 'N/A')}`")
        lines.append(f"- **{t('Branching', '分支数')}**: `{result.get('branching', 'N/A')}`")
        lines.append(f"- **{t('Keywords/Leaf', '叶子关键词数')}**: `{len(result.get('problem_keywords') or [])}`")
        leaf = result.get("leaf_domain")
        if isinstance(leaf, str) and leaf.strip():
            lines.append(f"- **{t('Leaf Domain', '叶子域')}**: `{leaf}`")
        prov = result.get("provenance") if isinstance(result.get("provenance"), dict) else None
        if prov:
            model = prov.get("model_name")
            if isinstance(model, str) and model.strip():
                lines.append(f"- **{t('Model', '模型')}**: `{model}`")
            sid = prov.get("service_id")
            if isinstance(sid, str) and sid.strip():
                lines.append(f"- **{t('Service ID', '服务ID')}**: `{sid}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append(f"## {t('Path Trace', '路径轨迹')}")
    lines.append("")
    for entry in path_trace:
        if not isinstance(entry, dict):
            continue
        level = entry.get("level", "N/A")
        input_domain = entry.get("input_domain", "N/A")
        lines.append(f"### {t('Level', '层')} {level}")
        lines.append("")
        lines.append(f"- **{t('Input Domain', '输入域')}**: `{input_domain}`")
        candidates = entry.get("candidates") if isinstance(entry.get("candidates"), list) else []
        chosen = entry.get("chosen")
        if candidates:
            lines.append("")
            lines.append(t("Candidates:", "候选："))
            lines.extend(_render_candidates_any(candidates, str(chosen)))
        if chosen:
            lines.append("")
        lines.append("")

    keywords = result.get("problem_keywords") if isinstance(result, dict) else []
    if isinstance(keywords, list):
        lines.append(f"## {t('Leaf Keywords', '叶子关键词')}")
        lines.append("")
        lines.extend(_render_list([str(x) for x in keywords]))
        lines.append("")

    lines.append(f"## {t('Snapshots', '快照文件')}")
    lines.append("")
    lines.append("- `result.json`")
    for level_dir in _collect_level_dirs(run_dir):
        prompt = level_dir / "expand.prompt.txt"
        extracted = level_dir / "expand.extracted.txt"
        if prompt.exists():
            lines.append(f"- `{_safe_path(prompt, run_dir)}`")
        if extracted.exists():
            lines.append(f"- `{_safe_path(extracted, run_dir)}`")
        if embed_preview and extracted.exists():
            content = _read_text(extracted)
            if preview_chars and preview_chars > 0:
                preview, truncated = _truncate(content, max_chars=preview_chars)
            else:
                preview, truncated = content, False
            lines.append("")
            lines.append(f"**Preview {level_dir.name} / expand.extracted.txt**")
            lines.append("")
            lines.append("```text")
            lines.append(preview)
            lines.append("```")
            if truncated:
                lines.append(f"... (truncated; total chars={len(content)})")
            lines.append("")

    leaf_dir = run_dir / "leaf"
    if leaf_dir.exists():
        for name in ("keywords.prompt.txt", "keywords.extracted.txt"):
            p = leaf_dir / name
            if p.exists():
                lines.append(f"- `{_safe_path(p, run_dir)}`")
                if embed_preview and name.endswith("extracted.txt"):
                    content = _read_text(p)
                    if preview_chars and preview_chars > 0:
                        preview, truncated = _truncate(content, max_chars=preview_chars)
                    else:
                        preview, truncated = content, False
                    lines.append("")
                    lines.append("**Preview leaf / keywords.extracted.txt**")
                    lines.append("")
                    lines.append("```text")
                    lines.append(preview)
                    lines.append("```")
                    if truncated:
                        lines.append(f"... (truncated; total chars={len(content)})")
                    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate playback markdown for domain_seed_walk.")
    parser.add_argument("--run-dir", required=True, help="run directory under data/domain_seed_walk/run_*")
    parser.add_argument("--output", help="output markdown path (default: <run_dir>/playback.md)")
    parser.add_argument("--no-embed-preview", action="store_true", help="do not embed extracted previews")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=0,
        help="max chars for embedded previews; set to 0 to disable truncation (default: 0)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    output = Path(args.output) if args.output else (run_dir / "playback.md")
    content = build_playback(run_dir, embed_preview=not args.no_embed_preview, preview_chars=int(args.preview_chars))
    output.write_text(content, encoding="utf-8")
    print(f"Playback written to: {output}")


if __name__ == "__main__":
    main()
