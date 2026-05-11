# Director Decision Contract

## What This Shows

The Director is the control layer. It does not write the next question directly. It reads the current Chain-of-KQA state, solver feedback, ambiguity signals, contract reports, and repair history, then chooses the next operation.

## Interface

```json
{
  "input": {
    "progress": "step / next_step / max_steps",
    "available_operations": ["Extend", "Revise", "Finish"],
    "solver_metrics": {
      "edge": "local correctness / well-posedness signal",
      "path": "end-to-end difficulty / reachability signal"
    },
    "reports": ["type1_ambiguity", "answer_contract", "path_fold_notes"],
    "history_tail": "recent KQA steps and answers"
  },
  "output": {
    "Operation": "Extend | Revise | Finish",
    "QuestionType": "MCQ | Derivation | Numeric",
    "ReviseMode": "correctness | world_contract | answer_contract | reuse_hidden | quality | ''",
    "Reason": "short evidence-based decision rationale"
  }
}
```

## Decision Policy Excerpt

```text
Use Edge signals for local correctness:
- if Edge solvers agree the step is correct and well-posed, the current transition is likely usable.

Use Path signals for difficulty:
- if Edge is correct but Path solvers split, treat this as a positive discrimination signal.
- prefer Extend unless there is explicit evidence of ambiguity, leakage, or answer-contract instability.

Choose Revise only with evidence:
- correctness: Edge says incorrect or not well-posed.
- world_contract: the semantic world is under-specified.
- answer_contract: answer format or equivalence policy is unstable.
- reuse_hidden: Path-Fold leaks internal pointers or hidden answers.
- quality: the item is correct but too easy, repetitive, or low-discrimination.
```

## State Mutation

The Director itself does not mutate the chain. It emits a typed operation decision that routes control to an operator:

```text
d_r = Director(S_r)
S_{r+1} = Operator(d_r, S_r)
```

## Invariants

- `Operation` must come from the enabled operation set.
- `Extend` must specify the next `QuestionType`.
- `Revise` must carry an explicit `ReviseMode`.
- Decisions should be grounded in artifacts and signals, not surface-level wording.

## Failure Handling

The Director routes failures into targeted repair:

| Signal | Route |
| --- | --- |
| local incorrectness | `Revise(correctness)` |
| semantic ambiguity | `Revise(world_contract)` |
| unstable answer equivalence | `Revise(answer_contract)` |
| folded prompt leaks hidden state | `Revise(reuse_hidden)` |
| correct but weak item | `Revise(quality)` |

## Artifact

Typical artifact:

```text
director_decision.json
```

This makes the control decision replayable: after a run fails, the system can inspect not only the generated question, but also why the controller chose that operation.
