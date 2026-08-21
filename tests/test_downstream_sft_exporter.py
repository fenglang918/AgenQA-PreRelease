from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenqa.downstream.sft.pipeline import export_sft_dataset


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_export_reference_run_to_canonical_and_tuningfactory(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "data" / "output" / "reference_run"
    if not run_dir.is_dir():
        pytest.skip("private reference_run fixture is not distributed in the public repository")
    output_dir = tmp_path / "sft_export"

    report = export_sft_dataset([run_dir], output_dir=output_dir)

    assert report.run_count == 1
    assert report.sample_count > 0
    assert report.edge_count > 0
    assert report.path_count > 0

    canonical_rows = _read_jsonl(output_dir / "canonical" / "all.jsonl")
    tuningfactory_rows = _read_json(output_dir / "tuningfactory" / "all.json")
    summary = _read_json(output_dir / "manifests" / "summary.json")

    assert len(canonical_rows) == report.sample_count
    assert len(tuningfactory_rows) == report.sample_count
    assert summary["sample_count"] == report.sample_count

    sample_types = {row["sample_type"] for row in canonical_rows}
    assert "edge" in sample_types
    assert "path_direct" in sample_types

    for row in canonical_rows:
        assert row["plan_text"].strip()
        assert row["solution_text"].strip()
        assert row["final_answer"].strip()
        assert row["source_paths"]["kqa_path"]
        assert "question_text_source" in row["metadata"]
        assert "world_contract_present" in row["metadata"]
        assert "known_tree_present" in row["metadata"]

    for row in canonical_rows:
        if row["sample_type"] == "path_direct":
            assert int(row["step"]) >= 2

    first_tf_row = tuningfactory_rows[0]
    assert first_tf_row["instruction"].strip()
    assert "Question:" in first_tf_row["input"]
    assert "[Plan]" in first_tf_row["output"]
    assert "[Solution]" in first_tf_row["output"]
    assert "[Final Answer]" in first_tf_row["output"]
