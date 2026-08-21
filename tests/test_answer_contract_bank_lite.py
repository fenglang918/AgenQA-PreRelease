from __future__ import annotations

from agenqa.domain.contracts.answer_contract_bank import (
    build_answer_contract_validation_background,
    build_answer_output_spec_prompt_section,
    extract_answer_contract_context,
    extract_answer_output_spec_context,
    make_default_answer_contracts,
    persist_answer_contracts,
    validate_answer_contracts,
)
from agenqa.domain.known_tree import KnownTree
from agenqa.domain.contracts.world_contract import empty_world_contract


def test_make_default_contracts_numeric_allows_approx_and_validator_flags_missing_regime_order() -> None:
    ids, contracts = make_default_answer_contracts(
        step=2,
        question_type="Numeric",
        question="You may approximate the final value if needed.",
        answer="\\boxed{1.0}",
        abs_tol=1e-3,
        rel_tol=None,
        sig_figs=None,
        unit="",
    )
    assert len(ids) == 2
    assert len(contracts) == 2

    err, warn = validate_answer_contracts(contracts)
    assert "approx_missing_regime_order" in err
    assert isinstance(warn, list)
    # Template metadata should be present for auditability.
    assert all(isinstance(c, dict) and isinstance(c.get("template"), dict) for c in contracts)


def test_validator_reads_type2_requirements_from_world_contract_l4() -> None:
    wc = empty_world_contract()
    # Tighten numeric requirement to abs_tol only; relax approx requirements.
    for sec in wc.get("sections", []):
        if sec.get("level") == "L4":
            for pt in sec.get("points") or []:
                if pt.get("axis") == "type2.numeric_requires_any_of":
                    pt["choice"] = ["abs_tol"]
                if pt.get("axis") == "type2.approx_requires":
                    pt["choice"] = []

    ids, contracts = make_default_answer_contracts(
        step=4,
        question_type="Numeric",
        question="You may approximate the final value if needed.",
        answer="\\boxed{1.0}",
        abs_tol=None,
        rel_tol=None,
        sig_figs=3,
        unit="",
    )
    err, _warn = validate_answer_contracts(contracts, world_contract=wc)
    assert "numeric_missing_tolerance" in err
    assert "approx_missing_regime_order" not in err

    wc2 = empty_world_contract()
    for sec in wc2.get("sections", []):
        if sec.get("level") == "L4":
            for pt in sec.get("points") or []:
                if pt.get("axis") == "type2.numeric_requires_any_of":
                    pt["choice"] = ["sig_figs"]
                if pt.get("axis") == "type2.approx_requires":
                    pt["choice"] = []

    err2, _warn2 = validate_answer_contracts(contracts, world_contract=wc2)
    assert "numeric_missing_tolerance" not in err2
    assert "approx_missing_regime_order" not in err2


def test_persist_contracts_and_hide_answer_contract_cert_from_solver_views() -> None:
    mem = KnownTree.empty_memory()
    ids, contracts = make_default_answer_contracts(
        step=1,
        question_type="Derivation",
        question="Derive the expression for X.",
        answer="\\boxed{x}",
    )
    err, warn = validate_answer_contracts(contracts)
    mem = persist_answer_contracts(
        mem,
        step=1,
        where="extend_format",
        answer_contract_ids=ids,
        answer_contracts=contracts,
        issue_types_error=err,
        issue_types_warn=warn,
        raw_ref="round_1/step_1_extend/subruns_raw/format/",
    )

    mem2 = KnownTree.normalize_memory(mem)
    assert isinstance(mem2.get("answer_contract_bank"), list)
    assert any(isinstance(x, dict) and x.get("id") in ids for x in mem2["answer_contract_bank"])
    assert isinstance(mem2.get("answer_contract_validation_errors"), list)
    assert isinstance(mem2.get("answer_contract_validation_candidates"), dict)

    # Must be present in raw memory step_certs (internal storage).
    step_certs = mem2.get("step_certs")
    assert isinstance(step_certs, list)
    assert any(isinstance(x, dict) and x.get("kind") == "answer_contract_cert" for x in step_certs)

    # Must be hidden from solver-facing views (redaction).
    edge_view = KnownTree.build_edge_solver_view(mem2, step=2)
    step_certs_window = edge_view.get("step_certs_window") or []
    assert isinstance(step_certs_window, list)
    assert all((not isinstance(x, dict)) or x.get("kind") != "answer_contract_cert" for x in step_certs_window)


def test_validation_background_includes_error_summary() -> None:
    mem = KnownTree.empty_memory()
    ids, contracts = make_default_answer_contracts(
        step=3,
        question_type="Numeric",
        question="You may approximate.",
        answer="\\boxed{1}",
    )
    err, warn = validate_answer_contracts(contracts)
    mem = persist_answer_contracts(
        mem,
        step=3,
        where="extend_format",
        answer_contract_ids=ids,
        answer_contracts=contracts,
        issue_types_error=err,
        issue_types_warn=warn,
        raw_ref="x",
    )
    bg = build_answer_contract_validation_background(mem, step=3, lang="en")
    assert "Answer Contract Validation" in bg
    assert "approx_missing_regime_order" in bg


def test_validation_background_not_stale_after_fix() -> None:
    mem = KnownTree.empty_memory()
    # 1) First persist: approx allowed -> creates approx contract without regime/order -> error.
    ids1, contracts1 = make_default_answer_contracts(
        step=7,
        question_type="Numeric",
        question="You may approximate.",
        answer="\\boxed{1}",
        abs_tol=1e-3,
        rel_tol=None,
        sig_figs=None,
        unit="",
    )
    err1, warn1 = validate_answer_contracts(contracts1)
    assert "approx_missing_regime_order" in err1
    mem = persist_answer_contracts(
        mem,
        step=7,
        where="extend_format",
        answer_contract_ids=ids1,
        answer_contracts=contracts1,
        issue_types_error=err1,
        issue_types_warn=warn1,
        raw_ref="x1",
    )
    bg1 = build_answer_contract_validation_background(mem, step=7, lang="en")
    assert "approx_missing_regime_order" in bg1

    # 2) Second persist: no approx allowed -> exact-only -> should be clean.
    ids2, contracts2 = make_default_answer_contracts(
        step=7,
        question_type="Numeric",
        question="Compute the value.",
        answer="\\boxed{1}",
        abs_tol=1e-3,
        rel_tol=None,
        sig_figs=None,
        unit="",
    )
    err2, warn2 = validate_answer_contracts(contracts2)
    assert err2 == []
    assert warn2 == []
    mem = persist_answer_contracts(
        mem,
        step=7,
        where="revise_format",
        answer_contract_ids=ids2,
        answer_contracts=contracts2,
        issue_types_error=err2,
        issue_types_warn=warn2,
        raw_ref="x2",
    )
    bg2 = build_answer_contract_validation_background(mem, step=7, lang="en")
    assert "approx_missing_regime_order" not in bg2


def test_extract_answer_contract_context_returns_policy_ids_and_contract_summaries() -> None:
    mem = KnownTree.empty_memory()
    ids, contracts = make_default_answer_contracts(
        step=1,
        question_type="Numeric",
        question="Compute x.",
        answer="\\boxed{1}",
        abs_tol=1e-3,
        rel_tol=None,
        sig_figs=None,
        unit="",
    )
    err, warn = validate_answer_contracts(contracts, world_contract=mem.get("world_contract"))
    mem = persist_answer_contracts(
        mem,
        step=1,
        where="extend_format",
        answer_contract_ids=ids,
        answer_contracts=contracts,
        issue_types_error=err,
        issue_types_warn=warn,
        raw_ref="x",
    )
    ctx = extract_answer_contract_context(mem, step=1)
    assert ctx.get("step") == 1
    assert isinstance(ctx.get("type2_policy"), dict)
    assert ctx.get("answer_contract_ids") == ids
    ac = ctx.get("answer_contracts")
    assert isinstance(ac, list) and ac
    assert "template_id" in (ac[0] or {})


def test_make_default_contracts_derivation_builds_answer_form_and_allowed_symbols() -> None:
    ids, contracts = make_default_answer_contracts(
        step=2,
        question_type="Derivation",
        question=(
            "Derive a single expression in terms of $P_{\\text{param}}$, "
            "$P_{\\text{context}}$, and $T_{\\text{thresh}}$."
        ),
        answer="\\boxed{x}",
    )

    assert len(ids) == 1
    assert len(contracts) == 1
    answer_style = (contracts[0] or {}).get("answer_style") or {}
    answer_semantics = (contracts[0] or {}).get("answer_semantics") or {}
    assert answer_style.get("boxed") is True
    assert answer_style.get("form") == "single_expression"
    assert answer_semantics.get("allowed_symbols") == [
        "P_{\\text{param}}",
        "P_{\\text{context}}",
        "T_{\\text{thresh}}",
    ]
    assert answer_semantics.get("answer_object") == "symbolic_expr"
    assert answer_semantics.get("acceptance_mode") == "exact"
    assert answer_semantics.get("branch_policy") == {
        "allow_branches": False,
        "require_complete_enumeration": False,
    }


def test_extract_answer_output_spec_context_hides_internal_fields() -> None:
    mem = KnownTree.empty_memory()
    ids, contracts = make_default_answer_contracts(
        step=1,
        question_type="Numeric",
        question="Compute x.",
        answer="\\boxed{1}",
        abs_tol=1e-3,
        rel_tol=None,
        sig_figs=None,
        unit="m",
    )
    err, warn = validate_answer_contracts(contracts, world_contract=mem.get("world_contract"))
    mem = persist_answer_contracts(
        mem,
        step=1,
        where="extend_format",
        answer_contract_ids=ids,
        answer_contracts=contracts,
        issue_types_error=err,
        issue_types_warn=warn,
        raw_ref="x",
    )
    ctx = extract_answer_output_spec_context(mem, step=1)
    assert ctx.get("step") == 1
    specs = ctx.get("answer_output_specs")
    assert isinstance(specs, list) and specs
    spec0 = specs[0] or {}
    assert spec0.get("question_type") == "Numeric"
    assert "mode" in spec0
    assert "answer_shape" in spec0
    assert "numeric" in spec0
    assert "id" not in spec0
    assert "judge" not in spec0
    assert "template_id" not in spec0
    assert "source_step" not in spec0


def test_extract_answer_output_spec_context_includes_derivation_v2_public_fields() -> None:
    mem = KnownTree.empty_memory()
    ids, contracts = make_default_answer_contracts(
        step=2,
        question_type="Derivation",
        question="Derive the explicit update equation in terms of $m_i$ and $s_i(t^-)$.",
        answer="\\boxed{s_i(t)=\\mathrm{mem}(m_i, s_i(t^-))}",
        answer_contract_payload={
            "answer_style": {
                "boxed": True,
                "form": "single_expression",
            },
            "answer_semantics": {
                "answer_object": "symbolic_expr",
                "acceptance_mode": "exact",
                "branch_policy": {
                    "allow_branches": True,
                    "require_complete_enumeration": False,
                },
                "allowed_symbols": ["m_i", "s_i(t^-)"],
                "required_qualifiers": ["branch_description", "closed_form"],
                "equivalence_rules": ["algebraic_rewrite_ok", "branch_collapse_not_ok"],
            },
            "support_witness": [
                {
                    "type": "boundary",
                    "statement": "Preserve the non-strict branch description in the final claim.",
                }
            ],
        },
    )
    err, warn = validate_answer_contracts(contracts, world_contract=mem.get("world_contract"))
    mem = persist_answer_contracts(
        mem,
        step=2,
        where="extend_format",
        answer_contract_ids=ids,
        answer_contracts=contracts,
        issue_types_error=err,
        issue_types_warn=warn,
        raw_ref="x",
    )
    ctx = extract_answer_output_spec_context(mem, step=2)
    specs = ctx.get("answer_output_specs")
    assert isinstance(specs, list) and specs
    spec0 = specs[0] or {}
    assert spec0.get("answer_style") == {
        "boxed": True,
        "form": "single_expression",
    }
    assert spec0.get("answer_semantics") == {
        "answer_object": "symbolic_expr",
        "acceptance_mode": "exact",
        "branch_policy": {
            "allow_branches": True,
            "require_complete_enumeration": False,
        },
        "allowed_symbols": ["m_i", "s_i(t^-)"],
        "required_qualifiers": ["branch_description", "closed_form"],
    }
    assert "support_witness" not in spec0
    assert "equivalence_rules" not in (spec0.get("answer_semantics") or {})
    assert "id" not in spec0
    assert "judge" not in spec0


def test_build_answer_output_spec_prompt_section_contains_visible_requirements_only() -> None:
    mem = KnownTree.empty_memory()
    ids, contracts = make_default_answer_contracts(
        step=2,
        question_type="Derivation",
        question="Derive a single expression in terms of $x$ and $y$.",
        answer="\\boxed{x}",
        answer_contract_payload={
            "answer_style": {"boxed": True, "form": "single_expression"},
            "answer_semantics": {
                "answer_object": "symbolic_expr",
                "acceptance_mode": "exact",
                "allowed_symbols": ["x", "y"],
                "required_qualifiers": ["closed_form"],
                "equivalence_rules": ["algebraic_rewrite_ok"],
            },
            "support_witness": [
                {
                    "type": "equivalence_cue",
                    "statement": "Judge may accept algebraically equivalent rewrites.",
                }
            ],
        },
    )
    err, warn = validate_answer_contracts(contracts)
    mem = persist_answer_contracts(
        mem,
        step=2,
        where="extend_format",
        answer_contract_ids=ids,
        answer_contracts=contracts,
        issue_types_error=err,
        issue_types_warn=warn,
        raw_ref="x",
    )
    bg = build_answer_output_spec_prompt_section(mem, step=2, lang="zh")
    assert "补充答案要求" in bg
    assert "answer_style" in bg
    assert "answer_semantics" in bg
    assert "allowed_symbols" in bg
    assert "closed_form" in bg
    assert "support_witness" not in bg
    assert "equivalence_rules" not in bg
    assert "template_id" not in bg
    assert "source_step" not in bg


def test_extract_answer_contract_context_includes_internal_derivation_v2_fields() -> None:
    mem = KnownTree.empty_memory()
    ids, contracts = make_default_answer_contracts(
        step=5,
        question_type="Derivation",
        question="Derive a single expression in terms of $x$ and $y$.",
        answer="\\boxed{x+y}",
        answer_contract_payload={
            "answer_style": {"boxed": True, "form": "single_expression"},
            "answer_semantics": {
                "answer_object": "symbolic_expr",
                "acceptance_mode": "exact",
                "allowed_symbols": ["x", "y"],
                "required_qualifiers": ["closed_form"],
                "equivalence_rules": ["algebraic_rewrite_ok", "branch_collapse_not_ok"],
            },
            "support_witness": [
                {
                    "type": "equivalence_cue",
                    "statement": "Do not collapse branch-specific qualifiers.",
                }
            ],
        },
    )
    err, warn = validate_answer_contracts(contracts)
    mem = persist_answer_contracts(
        mem,
        step=5,
        where="extend_format",
        answer_contract_ids=ids,
        answer_contracts=contracts,
        issue_types_error=err,
        issue_types_warn=warn,
        raw_ref="x",
    )
    ctx = extract_answer_contract_context(mem, step=5)
    contracts_ctx = ctx.get("answer_contracts")
    assert isinstance(contracts_ctx, list) and contracts_ctx
    contract0 = contracts_ctx[0] or {}
    assert contract0.get("answer_style") == {
        "boxed": True,
        "form": "single_expression",
    }
    assert (contract0.get("answer_semantics") or {}).get("equivalence_rules") == [
        "algebraic_rewrite_ok",
        "branch_collapse_not_ok",
    ]
    assert contract0.get("support_witness") == [
        {
            "type": "equivalence_cue",
            "statement": "Do not collapse branch-specific qualifiers.",
        }
    ]


def test_make_default_contracts_accept_legacy_derivation_spec_as_transition_input() -> None:
    ids, contracts = make_default_answer_contracts(
        step=6,
        question_type="Derivation",
        question="Derive a single expression in terms of $m_i$ and $s_i(t^-)$.",
        answer="\\boxed{\\mathrm{mem}(m_i, s_i(t^-))}",
        derivation_spec={
            "parameter_order": [
                {
                    "symbol": "\\mathrm{mem}",
                    "args": ["message", "previous_state"],
                    "required_in_question": True,
                }
            ],
            "boundary_strictness": [
                {
                    "phrase": "at least",
                    "operator": "\\ge",
                    "required_in_question": True,
                }
            ],
        },
    )
    assert len(ids) == 1
    assert len(contracts) == 1
    support_witness = (contracts[0] or {}).get("support_witness") or []
    assert support_witness == [
        {
            "type": "signature",
            "statement": 'parameter_order: {"args":["message","previous_state"],"required_in_question":true,"symbol":"\\\\mathrm{mem}"}',
        },
        {
            "type": "boundary",
            "statement": 'boundary_strictness: {"operator":"\\\\ge","phrase":"at least","required_in_question":true}',
        },
    ]
