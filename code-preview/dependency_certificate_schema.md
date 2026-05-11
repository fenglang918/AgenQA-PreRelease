# Dependency Certificate Schema

## What This Shows

AgenQA does not only store final QA pairs. Each accepted step records what premises and facts it used, what new facts it produced, and which produced fact is answer-equivalent.

That record is the dependency certificate.

## Interface

```json
{
  "premise_delta": [
    {
      "id": "p_i",
      "kind": "definition | assumption | condition",
      "text": "new durable premise",
      "source_step": "i",
      "provenance": "where it came from"
    }
  ],
  "fact_delta": [
    {
      "id": "f_i",
      "kind": "key_fact | intermediate_fact",
      "statement": "new reusable conclusion",
      "tags": ["answer_equivalent"],
      "source_step": "i",
      "provenance": "question / solution / answer"
    }
  ],
  "step_cert": {
    "step": "i",
    "uses_premise_ids": ["p_0", "p_1"],
    "uses_fact_ids": ["f_{i-1}"],
    "produces_fact_ids": ["f_i"],
    "key_fact_id": "f_i",
    "cert_text": "short explanation of the dependency"
  },
  "key_fact_id": "f_i"
}
```

## Contract Excerpt

```text
Memory write policy:
- write stable semantic anchors, not every derivation detail;
- premise_delta is for true premises only;
- derived conclusions must go to fact_delta;
- key_fact_id must point to a fact equivalent to the step answer.

Reference constraints:
- uses_premise_ids may reference previous premises or newly added premises;
- uses_fact_ids may reference previous facts or newly added facts;
- produces_fact_ids must come from fact_delta;
- new IDs must not collide with existing memory IDs.
```

## Answer Contract Layer

For derivation-style questions, AgenQA also extracts an answer contract:

```json
{
  "answer_style": {
    "boxed": true,
    "form": "single_expression",
    "rendering_notes": ["final answer only"]
  },
  "answer_semantics": {
    "answer_object": "symbolic_expr",
    "acceptance_mode": "exact",
    "allowed_symbols": ["x", "y"],
    "required_qualifiers": ["closed_form"],
    "equivalence_rules": ["algebraic_rewrite_ok"]
  },
  "support_witness": []
}
```

This separates task semantics from answer acceptance. The system can stabilize judging without silently changing the problem.

## Invariants

- The key fact must be answer-equivalent.
- Premises and facts must not be mixed.
- Reusable memory should be conservative and stable.
- Later steps should be able to cite facts by ID rather than by copying text.

## Failure Handling

If the answer object, accepted symbols, branch policy, or equivalence rule is ambiguous, the Director can trigger:

```text
Revise(answer_contract)
```

If the task semantics themselves are underspecified, it routes to:

```text
Revise(world_contract)
```

## Artifact

Typical artifacts:

```text
step_cert_builder.json
answer_contract_report.json
state.json
```

These artifacts are the bridge between algorithmic design and engineering debuggability: each generated item carries an auditable dependency record.
