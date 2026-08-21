from __future__ import annotations


def test_normalize_revise_mode_answer_contract() -> None:
    from agenqa.revise_modes import normalize_revise_mode

    assert normalize_revise_mode("answer_contract") == "answer_contract"
    assert normalize_revise_mode("answer-contract") == "answer_contract"
    assert normalize_revise_mode("answercontract") == "answer_contract"


def test_normalize_revise_mode_no_acb_alias() -> None:
    from agenqa.revise_modes import normalize_revise_mode

    # Clean mode name: avoid broad aliases that can collide with other concepts.
    assert normalize_revise_mode("acb") is None
    assert normalize_revise_mode("contract") is None
