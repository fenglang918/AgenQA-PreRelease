from __future__ import annotations

from agenqa.domain.contracts.answer_contract_builder_schema import (
    FIELD_ANSWER_SEMANTICS,
    FIELD_ANSWER_STYLE,
    FIELD_SUPPORT_WITNESS,
)
from agenqa.skills.answer_contract_builder import AnswerContractBuilderRunner


def test_answer_contract_builder_parse_output_accepts_v2_payload() -> None:
    runner = AnswerContractBuilderRunner.__new__(AnswerContractBuilderRunner)
    text = """```json
{
  "answer_style": {
    "boxed": true,
    "form": "single_expression",
    "rendering_notes": ["final answer only"]
  },
  "answer_semantics": {
    "answer_object": "symbolic_expr",
    "acceptance_mode": "exact",
    "branch_policy": {
      "allow_branches": false,
      "require_complete_enumeration": false
    },
    "allowed_symbols": ["x", "y"],
    "required_qualifiers": ["closed_form"],
    "equivalence_rules": ["algebraic_rewrite_ok"]
  },
  "support_witness": [
    {
      "type": "equivalence_cue",
      "statement": "Judge may accept algebraically equivalent rewrites."
    }
  ]
}
```"""

    out = AnswerContractBuilderRunner._parse_output(runner, text)

    assert out.answer_style == {
        "boxed": True,
        "form": "single_expression",
        "rendering_notes": ["final answer only"],
    }
    assert out.answer_semantics == {
        "answer_object": "symbolic_expr",
        "acceptance_mode": "exact",
        "branch_policy": {
            "allow_branches": False,
            "require_complete_enumeration": False,
        },
        "allowed_symbols": ["x", "y"],
        "required_qualifiers": ["closed_form"],
        "equivalence_rules": ["algebraic_rewrite_ok"],
    }
    assert out.support_witness == [
        {
            "type": "equivalence_cue",
            "statement": "Judge may accept algebraically equivalent rewrites.",
        }
    ]

    payload = {
        FIELD_ANSWER_STYLE: out.answer_style,
        FIELD_ANSWER_SEMANTICS: out.answer_semantics,
        FIELD_SUPPORT_WITNESS: out.support_witness,
    }
    assert payload[FIELD_ANSWER_STYLE]["boxed"] is True
