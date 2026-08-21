from __future__ import annotations

from typing import Any, Dict, Optional

from agenqa.domain.contracts.world_contract import merge_world_contract, normalize_world_contract


def _get_choice(wc: Dict[str, Any], level: str, axis: str) -> Optional[Any]:
    secs = wc.get("sections")
    if not isinstance(secs, list):
        return None
    for sec in secs:
        if not isinstance(sec, dict):
            continue
        if sec.get("level") != level:
            continue
        pts = sec.get("points")
        if not isinstance(pts, list):
            return None
        for pt in pts:
            if isinstance(pt, dict) and pt.get("axis") == axis:
                return pt.get("choice")
    return None


def _count_axis(wc: Dict[str, Any], level: str, axis: str) -> int:
    secs = wc.get("sections")
    if not isinstance(secs, list):
        return 0
    n = 0
    for sec in secs:
        if not isinstance(sec, dict):
            continue
        if sec.get("level") != level:
            continue
        pts = sec.get("points")
        if not isinstance(pts, list):
            continue
        for pt in pts:
            if isinstance(pt, dict) and pt.get("axis") == axis:
                n += 1
    return n


def test_merge_upsert_preserve_other_points() -> None:
    old = normalize_world_contract(
        {
            "sections": [
                {"level": "L1", "points": [{"axis": "paradigm_id", "choice": "p1"}]},
                {"level": "L3", "points": [{"axis": "send_increments", "choice": False}]},
            ]
        }
    )
    new = {"sections": [{"level": "L3", "points": [{"axis": "repair_writeback", "choice": True}]}]}
    merged = merge_world_contract(old, new, role="extend_world_contract", step=3, round=9, raw_ref="x")
    assert _get_choice(merged, "L1", "paradigm_id") == "p1"
    assert _get_choice(merged, "L3", "send_increments") is False
    assert _get_choice(merged, "L3", "repair_writeback") is True
    assert _count_axis(merged, "L3", "send_increments") == 1
    assert _count_axis(merged, "L3", "repair_writeback") == 1


def test_merge_upsert_overwrite_same_axis_records_changelog() -> None:
    old = {"sections": [{"level": "L1", "points": [{"axis": "paradigm_id", "choice": "p1"}]}, {"level": "L3", "points": [{"axis": "send_increments", "choice": False}]}]}
    new = {"sections": [{"level": "L3", "points": [{"axis": "send_increments", "choice": True}]}]}
    merged = merge_world_contract(old, new, role="revise_world_contract", step=3, round=9, raw_ref="y")
    assert _get_choice(merged, "L3", "send_increments") is True
    extra = merged.get("extra_internal") or {}
    assert isinstance(extra, dict)
    changelog = extra.get("changelog")
    assert isinstance(changelog, list) and changelog
    ops = changelog[-1].get("ops")
    assert isinstance(ops, list) and ops
    assert any(op.get("level") == "L3" and op.get("axis") == "send_increments" and op.get("old") is False and op.get("new") is True for op in ops)


def test_merge_l1_change_clears_l2_l3() -> None:
    old = {
        "sections": [
            {"level": "L1", "points": [{"axis": "paradigm_id", "choice": "p1"}]},
            {"level": "L2", "points": [{"axis": "default_policy", "choice": "foo"}]},
            {"level": "L3", "points": [{"axis": "send_increments", "choice": False}]},
        ]
    }
    new = {"sections": [{"level": "L1", "points": [{"axis": "paradigm_id", "choice": "p2"}]}]}
    merged = merge_world_contract(old, new, role="extend_world_contract", step=1, round=1, raw_ref="z")
    assert _get_choice(merged, "L1", "paradigm_id") == "p2"
    assert _get_choice(merged, "L2", "default_policy") is None
    assert _get_choice(merged, "L3", "send_increments") is None
    assert merged.get("status") == "defaulted"
