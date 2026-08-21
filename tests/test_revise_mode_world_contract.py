from __future__ import annotations


def test_normalize_revise_mode_world_contract() -> None:
    from agenqa.revise_modes import normalize_revise_mode

    assert normalize_revise_mode("world_contract") == "world_contract"
    assert normalize_revise_mode("world-contract") == "world_contract"
    assert normalize_revise_mode("worldcontract") == "world_contract"


def test_normalize_revise_mode_no_semantics_compat() -> None:
    from agenqa.revise_modes import normalize_revise_mode

    # Clean rename: no alias/fallback kept for the old name.
    assert normalize_revise_mode("semantics") is None
    assert normalize_revise_mode("semantic") is None
    assert normalize_revise_mode("clarify_semantics") is None
