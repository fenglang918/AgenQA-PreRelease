from agenqa.prompts.format import (
    FORMAT_V1,
    FORMAT_V1_EN,
    FORMAT_V1_TAGGED,
    FORMAT_V1_TAGGED_EN,
)


def test_format_prompt_zh_requires_derivation_disambiguation() -> None:
    assert "最终 solver-visible 交付" in FORMAT_V1
    assert "必须主动消歧" in FORMAT_V1
    assert "参数顺序" in FORMAT_V1
    assert "边界严格性" in FORMAT_V1
    assert "validation_passed=false" in FORMAT_V1


def test_format_prompt_en_requires_derivation_disambiguation() -> None:
    assert "final solver-visible deliverable" in FORMAT_V1_EN
    assert "must resolve notation ambiguity" in FORMAT_V1_EN
    assert "parameter order" in FORMAT_V1_EN
    assert "boundary strictness" in FORMAT_V1_EN
    assert "validation_passed=false" in FORMAT_V1_EN


def test_tagged_format_prompts_keep_derivation_checks() -> None:
    assert "[validation_passed] 必须为 false" in FORMAT_V1_TAGGED
    assert "[validation_passed] must be false" in FORMAT_V1_TAGGED_EN
