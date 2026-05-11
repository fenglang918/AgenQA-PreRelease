# Path-Fold Visibility Contract

## What This Shows

Path-Fold turns a locally verifiable chain into a harder solver-facing problem by changing visibility, not by inventing arbitrary difficulty.

The same Chain-of-KQA state can produce:

- **Edge View**: local support is visible, so the step is easier to verify.
- **Path View**: intermediate facts and certificates are hidden, so the solver must reconstruct the dependency path.

## Interface

```json
{
  "input": {
    "step": "tail step",
    "question_type": "MCQ | Derivation | Numeric",
    "premise_bank_json": "primitive definitions and conditions",
    "history_json": {
      "recent_steps": "recent question-answer sequence",
      "older_summary": "compressed earlier context"
    }
  },
  "output": {
    "question_scaffolded": "path question with intermediate sub-goals",
    "question_direct": "final ask only, no hints",
    "fold_notes": "short folding notes"
  }
}
```

## Visibility Contract Excerpt

```text
Path-Fold must:
- use the tail step Answer as the source of truth;
- keep scaffolded and direct versions answer-equivalent;
- preserve the same required output set;
- hide intermediate key facts and step certificates;
- avoid internal pointers such as step/history/fact_bank/premise_bank;
- avoid pasting any previous Answer as a given;
- make the direct version ask only the final target with no hint text.
```

## Edge vs Path

| View | Visible to solver | Used for |
| --- | --- | --- |
| Edge View | local premises and support for one transition | correctness and well-posedness control |
| Path View | only initial premises and final target | difficulty and model-discrimination signal |

The Edge/Path gap is intentional. If Edge solvers succeed while Path solvers split, the item can be both locally valid and globally discriminative.

## Invariants

- Answer equivalence: folded questions target the same answer as the tail step.
- Visibility separation: hidden facts and certificates do not leak into the Path View.
- Path preservation: the folded question remains solvable by reconstructing the dependency route.
- Output alignment: direct and scaffolded versions ask for the same final object.

## Failure Handling

Path-Fold failures route to targeted revision:

| Failure | Route |
| --- | --- |
| internal pointer appears in question | `Revise(reuse_hidden)` |
| previous answer is pasted as a premise | `Revise(reuse_hidden)` |
| direct version asks for fewer outputs | `Revise(answer_contract)` |
| question becomes underspecified | `Revise(world_contract)` |

## Artifact

Typical artifacts:

```text
path_fold.json
edge_kqa.json
path_kqa.json
consensus_summary_edge.json
consensus_summary_path.json
```

This is the core benchmark-construction move: AgenQA preserves local verification while creating harder end-to-end solver tasks through controlled visibility.
