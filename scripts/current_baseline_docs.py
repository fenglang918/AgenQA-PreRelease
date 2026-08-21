#!/usr/bin/env python3
"""
根据 experiments/upstream/generation/ 最新实验目录渲染与校验“当前实验基线”文档块。

Usage:
  python scripts/current_baseline_docs.py render
  python scripts/current_baseline_docs.py check
  python scripts/current_baseline_docs.py current
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


BLOCK_BEGIN = "<!-- BEGIN:CURRENT_BASELINE -->"
BLOCK_END = "<!-- END:CURRENT_BASELINE -->"
EXPERIMENTS_DIR = Path("experiments/upstream/generation")

BASELINE_ID_RE = re.compile(r"\b\d{8}(?:_\d{6})?__[a-z0-9_]+\b")
EXPERIMENT_DIR_RE = re.compile(r"^(?P<date>\d{8})(?:_(?P<time>\d{6}))?__(?P<slug>[a-z0-9_]+)$")

CURRENT_SYNCED_DOCS = (
    Path("README.md"),
    Path("experiments/upstream/generation/MAIN.md"),
    Path("docs/stable/code2doc/system_overview/system_overview.md"),
)

CURRENT_DOC_RULES = {
    Path("README.md"): {
        "forbid_any_baseline_ids_outside_block": True,
        "forbid_phrases": (),
    },
    Path("experiments/upstream/generation/MAIN.md"): {
        "forbid_any_baseline_ids_outside_block": False,
        "forbid_phrases": (
            "当前最佳实践",
            "latest / 推荐基线",
        ),
    },
    Path("docs/stable/code2doc/system_overview/system_overview.md"): {
        "forbid_any_baseline_ids_outside_block": True,
        "forbid_phrases": (),
    },
}

BASELINE_FIXED_DOCS = (
    Path("docs/stable/code2doc/07_operators.md"),
    Path("docs/stable/code2doc/system_overview/system_overview_reserved.md"),
    Path("docs/design/active/downstream/sft/supporting/agen_qa_产物字段速查_2026_03_11.md"),
    Path("docs/design/active/downstream/sft/agenqa_sft_pipeline_detailed_design_2026_03_14.md"),
)

BASELINE_FIXED_FORBIDDEN_PHRASES = (
    "当前实验基线",
    "当前 baseline",
    "当前 Baseline",
)
BASELINE_FIXED_HEADER_RE = re.compile(r"适用基线\s*\*?\*\s*[:：]")


@dataclass(frozen=True)
class ExperimentInfo:
    baseline_id: str
    date_token: str
    time_token: str
    dir_path: Path
    run_doc_path: Path
    config_path: Path
    title: str
    summary: str

    @property
    def date_display(self) -> str:
        return f"{self.date_token[:4]}-{self.date_token[4:6]}-{self.date_token[6:8]}"


def _relative_link(from_doc: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_doc.parent)


def find_managed_block(text: str) -> tuple[int, int]:
    begin = text.find(BLOCK_BEGIN)
    end = text.find(BLOCK_END)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError("缺少 CURRENT_BASELINE managed block 标记")
    end += len(BLOCK_END)
    return begin, end


def strip_managed_block(text: str) -> str:
    begin, end = find_managed_block(text)
    return text[:begin] + text[end:]


def parse_experiment_name(name: str) -> tuple[str, str] | None:
    match = EXPERIMENT_DIR_RE.match(name)
    if not match:
        return None
    return match.group("date"), match.group("time") or ""


def extract_run_title_summary(run_doc_path: Path) -> tuple[str, str]:
    if not run_doc_path.exists():
        return "", ""

    lines = run_doc_path.read_text(encoding="utf-8").splitlines()
    title = ""
    summary_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    in_blockquote = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_blockquote:
                break
            continue
        if stripped.startswith(">"):
            in_blockquote = True
            summary_lines.append(stripped.lstrip("> ").strip())
            continue
        if in_blockquote:
            break
        if stripped.startswith("#"):
            continue
        summary_lines.append(stripped)
        break

    summary = " ".join(part for part in summary_lines if part).strip()
    if summary.startswith("状态："):
        summary = summary[len("状态：") :].strip()
    return title, summary


def discover_experiments(repo_root: Path) -> list[ExperimentInfo]:
    base_dir = repo_root / EXPERIMENTS_DIR
    experiments: list[ExperimentInfo] = []
    for item in base_dir.iterdir():
        if not item.is_dir() or item.name == "_template":
            continue
        parsed = parse_experiment_name(item.name)
        if parsed is None:
            continue
        date_token, time_token = parsed
        run_doc_path = item / "run.md"
        config_path = item / "config.yaml"
        title, summary = extract_run_title_summary(run_doc_path)
        experiments.append(
            ExperimentInfo(
                baseline_id=item.name,
                date_token=date_token,
                time_token=time_token,
                dir_path=item,
                run_doc_path=run_doc_path,
                config_path=config_path,
                title=title or item.name,
                summary=summary,
            )
        )
    experiments.sort(key=lambda exp: (exp.date_token, exp.time_token, exp.baseline_id))
    return experiments


def get_current_baseline(repo_root: Path) -> tuple[ExperimentInfo, list[ExperimentInfo]]:
    experiments = discover_experiments(repo_root)
    if not experiments:
        raise ValueError("未找到任何符合命名规范的 experiments/upstream/generation 实验目录")
    current = experiments[-1]
    return current, list(reversed(experiments[-3:]))


def ensure_current_baseline_files(current: ExperimentInfo) -> None:
    if not current.run_doc_path.exists():
        raise ValueError(f"当前实验目录缺少 run.md: {current.run_doc_path}")
    if not current.config_path.exists():
        raise ValueError(f"当前实验目录缺少 config.yaml: {current.config_path}")


def render_current_block(repo_root: Path, doc_path: Path) -> str:
    current, recent = get_current_baseline(repo_root)
    ensure_current_baseline_files(current)

    run_link = _relative_link(doc_path, current.run_doc_path)
    config_link = _relative_link(doc_path, current.config_path)
    recent_lines = [
        f"> - [`{item.baseline_id}`]({_relative_link(doc_path, item.run_doc_path)})"
        for item in recent
    ]

    summary = current.summary or "见该实验目录下 run.md 的目标与状态说明。"
    block_lines = [
        BLOCK_BEGIN,
        "> 当前实验基线按 `experiments/upstream/generation/` 最新实验目录自动推导。",
        ">",
        f"> - **Baseline ID**：[`{current.baseline_id}`]({run_link})",
        f"> - **实验日期**：`{current.date_display}`",
        f"> - **Run Doc**：[run.md]({run_link})",
        f"> - **Config**：[config.yaml]({config_link})",
        f"> - **摘要**：{summary}",
        ">",
        "> 最近三次实验：",
        *recent_lines,
        BLOCK_END,
    ]
    return "\n".join(block_lines)


def replace_managed_block(text: str, rendered_block: str) -> str:
    begin, end = find_managed_block(text)
    updated = text[:begin] + rendered_block + text[end:]
    if not updated.endswith("\n"):
        updated += "\n"
    return updated


def render_docs(repo_root: Path) -> list[Path]:
    updated_files: list[Path] = []
    for rel_path in CURRENT_SYNCED_DOCS:
        doc_path = repo_root / rel_path
        text = doc_path.read_text(encoding="utf-8")
        rendered = render_current_block(repo_root, doc_path)
        updated = replace_managed_block(text, rendered)
        if updated != text:
            doc_path.write_text(updated, encoding="utf-8")
            updated_files.append(rel_path)
    return updated_files


def check_current_doc(repo_root: Path, rel_path: Path) -> None:
    doc_path = repo_root / rel_path
    text = doc_path.read_text(encoding="utf-8")
    rendered = render_current_block(repo_root, doc_path)
    expected = replace_managed_block(text, rendered)
    if expected != text:
        raise ValueError(f"{rel_path} 的 CURRENT_BASELINE block 未同步，请先执行 render")

    outside_block = strip_managed_block(text)
    rules = CURRENT_DOC_RULES[rel_path]

    for phrase in rules["forbid_phrases"]:
        if phrase in outside_block:
            raise ValueError(f"{rel_path} 在 managed block 外仍包含已废弃口径: {phrase}")

    if rules["forbid_any_baseline_ids_outside_block"] and BASELINE_ID_RE.search(outside_block):
        raise ValueError(f"{rel_path} 在 managed block 外仍硬编码 baseline id")


def check_baseline_fixed_doc(repo_root: Path, rel_path: Path) -> None:
    doc_path = repo_root / rel_path
    text = doc_path.read_text(encoding="utf-8")
    if BASELINE_FIXED_HEADER_RE.search(text) is None and "适用基线" not in text:
        raise ValueError(f"{rel_path} 缺少“适用基线：”头部说明")
    for phrase in BASELINE_FIXED_FORBIDDEN_PHRASES:
        if phrase in text:
            raise ValueError(f"{rel_path} 仍使用了需收口的措辞: {phrase}")


def check_repo(repo_root: Path) -> None:
    current, _recent = get_current_baseline(repo_root)
    ensure_current_baseline_files(current)
    for rel_path in CURRENT_SYNCED_DOCS:
        check_current_doc(repo_root, rel_path)
    for rel_path in BASELINE_FIXED_DOCS:
        check_baseline_fixed_doc(repo_root, rel_path)


def print_current(repo_root: Path) -> None:
    current, recent = get_current_baseline(repo_root)
    ensure_current_baseline_files(current)
    print(f"baseline_id={current.baseline_id}")
    print(f"date={current.date_display}")
    print(f"run_doc={current.run_doc_path.relative_to(repo_root)}")
    print(f"config={current.config_path.relative_to(repo_root)}")
    print(f"title={current.title}")
    print(f"summary={current.summary}")
    print("recent=" + ",".join(item.baseline_id for item in recent))


def main() -> int:
    parser = argparse.ArgumentParser(description="渲染/校验当前实验基线文档块")
    parser.add_argument(
        "command",
        choices=("render", "check", "current"),
        help="render: 更新 managed block；check: 校验；current: 打印当前基线信息",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="仓库根目录（默认：当前目录）",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    try:
        if args.command == "render":
            updated = render_docs(repo_root)
            if updated:
                for rel_path in updated:
                    print(f"[UPDATED] {rel_path}")
            else:
                print("[OK] managed block 已是最新")
        elif args.command == "check":
            check_repo(repo_root)
            print("[OK] 当前基线文档校验通过")
        else:
            print_current(repo_root)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
