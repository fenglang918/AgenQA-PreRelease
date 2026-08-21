from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_strong_solver_percentages_match_counts() -> None:
    for row in _rows("path_view_strong_solvers.csv"):
        all_pct = 100 * int(row["all_correct"]) / int(row["all_total"])
        diagnostic_pct = 100 * int(row["diagnostic_correct"]) / int(row["diagnostic_total"])
        assert round(all_pct, 2) == float(row["all_accuracy_pct"])
        assert round(diagnostic_pct, 2) == float(row["diagnostic_accuracy_pct"])


def test_training_transfer_matches_published_deltas() -> None:
    rows = {row["model"]: row for row in _rows("training_transfer_2k.csv")}
    baseline = rows["Qwen3-4B-Instruct baseline"]
    gspo = rows["instruct-gspo"]
    grpo = rows["instruct-grpo"]
    assert round(float(gspo["aim24"]) - float(baseline["aim24"]), 2) == 2.36
    assert round(float(grpo["hmmt_feb"]) - float(baseline["hmmt_feb"]), 2) == 1.87
    assert round(float(gspo["scibench"]) - float(baseline["scibench"]), 2) == 1.05


def test_manifest_references_existing_snapshot_files() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_type"] == "aggregate_preview"
    for snapshot in manifest["snapshots"]:
        assert (RESULTS / snapshot["file"]).is_file()
        assert snapshot["raw_outputs_included"] is False
