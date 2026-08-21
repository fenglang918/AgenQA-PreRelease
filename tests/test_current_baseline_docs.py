from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_current_baseline_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "current_baseline_docs.py"
    spec = importlib.util.spec_from_file_location("current_baseline_docs", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_minimal_repo(tmp_path: Path) -> Path:
    repo = tmp_path
    write_file(
        repo / "README.md",
        "# Repo\n\n## 当前实验基线\n\n<!-- BEGIN:CURRENT_BASELINE -->\n> placeholder\n<!-- END:CURRENT_BASELINE -->\n",
    )
    write_file(
        repo / "experiments/upstream/generation/MAIN.md",
        "# MAIN\n\n<!-- BEGIN:CURRENT_BASELINE -->\n> placeholder\n<!-- END:CURRENT_BASELINE -->\n",
    )
    write_file(
        repo / "docs/stable/code2doc/system_overview/system_overview.md",
        "# Overview\n\n<!-- BEGIN:CURRENT_BASELINE -->\n> placeholder\n<!-- END:CURRENT_BASELINE -->\n",
    )
    write_file(
        repo / "docs/stable/code2doc/07_operators.md",
        "> 适用基线：`20260208__baseline`\n",
    )
    write_file(
        repo / "docs/stable/code2doc/system_overview/system_overview_reserved.md",
        "> 适用基线：`20260208__baseline`\n",
    )
    write_file(
        repo / "docs/design/active/downstream/sft/supporting/agen_qa_产物字段速查_2026_03_11.md",
        "> 适用基线：`20260208__baseline`\n",
    )
    write_file(
        repo / "docs/design/active/downstream/sft/agenqa_sft_pipeline_detailed_design_2026_03_14.md",
        "> 适用基线：`20260208__baseline`\n",
    )
    return repo


def make_experiment(repo: Path, name: str, *, with_run: bool = True, with_config: bool = True) -> None:
    exp_dir = repo / "experiments/upstream/generation" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    if with_run:
        write_file(
            exp_dir / "run.md",
            f"# {name}\n\n> 状态：{name} summary\n",
        )
    if with_config:
        write_file(exp_dir / "config.yaml", "name: test\n")


def test_current_baseline_prefers_latest_named_dir_and_ignores_template(tmp_path: Path) -> None:
    module = load_current_baseline_module()
    repo = make_minimal_repo(tmp_path)
    make_experiment(repo, "20260315__paper_seed_skeleton")
    make_experiment(repo, "20260315_120000__later_same_day")
    make_experiment(repo, "20260314__older")
    (repo / "experiments/upstream/generation/_template").mkdir(parents=True)
    (repo / "experiments/upstream/generation/not_an_experiment").mkdir(parents=True)

    current, recent = module.get_current_baseline(repo)

    assert current.baseline_id == "20260315_120000__later_same_day"
    assert [item.baseline_id for item in recent] == [
        "20260315_120000__later_same_day",
        "20260315__paper_seed_skeleton",
        "20260314__older",
    ]


def test_render_is_idempotent_and_writes_current_block(tmp_path: Path) -> None:
    module = load_current_baseline_module()
    repo = make_minimal_repo(tmp_path)
    make_experiment(repo, "20260313__domain_seed_multistrong_cleanup")
    make_experiment(repo, "20260315__paper_seed_skeleton")

    updated = module.render_docs(repo)
    assert len(updated) == 3

    first_readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "20260315__paper_seed_skeleton" in first_readme
    assert "最近三次实验" in first_readme

    updated_second = module.render_docs(repo)
    second_readme = (repo / "README.md").read_text(encoding="utf-8")
    assert updated_second == []
    assert second_readme == first_readme


def test_check_fails_when_latest_baseline_missing_config(tmp_path: Path) -> None:
    module = load_current_baseline_module()
    repo = make_minimal_repo(tmp_path)
    make_experiment(repo, "20260313__domain_seed_multistrong_cleanup")
    make_experiment(repo, "20260315__paper_seed_skeleton", with_config=False)

    with pytest.raises(ValueError, match="缺少 config.yaml"):
        module.check_repo(repo)


def test_check_fails_when_current_doc_contains_hardcoded_baseline_outside_block(tmp_path: Path) -> None:
    module = load_current_baseline_module()
    repo = make_minimal_repo(tmp_path)
    make_experiment(repo, "20260315__paper_seed_skeleton")
    module.render_docs(repo)

    readme_path = repo / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8") + "\n旧基线：`20260313__domain_seed_multistrong_cleanup`\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="README.md 在 managed block 外仍硬编码 baseline id"):
        module.check_repo(repo)


def test_check_fails_for_baseline_fixed_doc_using_current_baseline_wording(tmp_path: Path) -> None:
    module = load_current_baseline_module()
    repo = make_minimal_repo(tmp_path)
    make_experiment(repo, "20260315__paper_seed_skeleton")
    module.render_docs(repo)

    fixed_doc = repo / "docs/stable/code2doc/07_operators.md"
    fixed_doc.write_text(
        "> 适用基线：`20260208__baseline`\n\n当前 baseline 很重要。\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="07_operators.md 仍使用了需收口的措辞"):
        module.check_repo(repo)
