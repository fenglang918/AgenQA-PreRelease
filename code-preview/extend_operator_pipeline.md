# Extend Operator Pipeline

## What This Shows

Extend is the constructive operator. It appends one new verifiable KQA transition to the current Chain-of-KQA state.

This is where open-ended question generation becomes a typed state mutation.

## Interface

```json
{
  "input": {
    "current_step": "i - 1",
    "known_state": "current premises, history, and derived facts",
    "previous_question": "Question_{i-1}",
    "previous_answer": "Answer_{i-1}",
    "question_type": "MCQ | Derivation | Numeric",
    "director_notes": "optional difficulty or repair guidance"
  },
  "output": {
    "Step": "i",
    "Question": "solver-facing local question",
    "Solution": "structured derivation steps",
    "Answer": "machine-checkable final answer",
    "NewBackground": "optional true premises or conditions",
    "DerivedFacts": "optional reusable facts"
  }
}
```

## Pipeline

```text
draft -> format -> certify -> fold
```

| Stage | Responsibility |
| --- | --- |
| `draft` | propose the next reasoning transition |
| `format` | normalize the question, solution, and answer |
| `certify` | build dependency certificate and key fact |
| `fold` | construct solver-facing Edge/Path views |

## State Edit

Extend appends one transition:

```text
s_i = (K_i, q_i, a_i, c_i)
```

where:

- `K_i` is the visible support state before step `i`;
- `q_i` is the local question;
- `a_i` is the answer;
- `c_i` is the dependency certificate.

The new key fact extends the dependency spine:

```text
K_0 -> f_1 -> ... -> f_{i-1} -> f_i
```

## Contract Excerpt

```text
The new step must:
- add one answer-equivalent fact rather than create an unrelated question;
- explicitly depend on the immediately previous step;
- keep NewBackground for true premises only, not derived answers;
- avoid copying a previous answer into the new question;
- preserve the paper/source setting rather than drifting to a generic math exercise.
```

## Invariants

- Dependency continuity: removing the previous step should make the new step underspecified or harder to solve.
- Answer separability: the new question must not reveal the answer it is supposed to test.
- Local verifiability: the step must be checkable through Edge View.
- Foldability: the step must later support Path View construction.

## Failure Handling

If Extend produces a transition with leakage, weak dependency, ambiguity, or contract instability, the Director routes the state into `revise`. The important engineering detail is that `revise` can reuse the same synthesis core while changing the state edit from append to repair.

```text
extend: append transition
revise: repair transition
```

## Artifact

Typical artifacts:

```text
draft_chain.json
format.json
step_cert_builder.json
path_fold.json
edge_kqa.json
path_kqa.json
```

Together these artifacts make the operator output inspectable at every stage, not only at the final question.
