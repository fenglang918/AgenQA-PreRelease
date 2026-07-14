# Architecture

[中文](./architecture.zh.md)

## Core Idea

AgenQA is a synthesis harness for challenging reasoning QA. Its central object is not a single generated question, but a **Chain-of-KQA**: a growing dependency chain where each step binds known context, a question, an answer-equivalent fact, and a dependency certificate.

The system keeps two views of the same chain:

- **Edge View**: exposes the local support needed to verify one dependency transition.
- **Path View**: hides intermediate facts through Path-Fold so the solver must reconstruct the dependency path end to end.

This separates two goals that are usually coupled in synthetic QA generation: step-level correctness and global difficulty.

## System Loop

The synthesis process is organized as a controlled state-transition loop:

```text
Current Chain-of-KQA State
        |
        v
Director: choose init / extend / revise / finish
        |
        v
Operator: edit the chain state
        |
        v
Edge / Path view construction
        |
        v
Evaluator: solvers + consensus + judge signals
        |
        v
Feedback to Director and revision planning
```

## Director

The Director is the control layer. It does not write final questions directly. It reads a compact state view that includes chain progress, recent history, available operations, solver outcomes, ambiguity signals, contract reports, and repair history. It then chooses whether to initialize, extend, revise, or finish.

## Operators

Operators are state editors over the Chain-of-KQA object.

| Operator | Role | State edit |
| --- | --- | --- |
| `init` | initialize source-grounded state | create the starting context |
| `extend` | add a new dependency step | append a KQA transition |
| `revise` | repair a problematic step | overwrite or repair the current transition |
| `finish` | route accepted chain | export benchmark-facing artifacts |

Extension and revision reuse the same synthesis core: draft a candidate step, normalize it into QA form, build a dependency certificate, then run Path-Fold. They differ in adapter logic and state-edit semantics.

## Evaluator

The Evaluator probes generated Edge and Path views using solver responses, multi-solver consensus, and judge signals. Feedback is written back into the online state so that later Director decisions can extend, revise, or stop with context.

This is why AgenQA is not a post-hoc filtering pipeline. Verification is part of generation.

## Why This Architecture Scales

The harness scales because the control interface remains stable while the chain grows:

- longer chains still use the same append / repair / evaluate / route lifecycle;
- different domains can replace source grounding, operator adapters, view constructors, or solver ensembles;
- benchmark, SFT, and RL-style exports can be routed from the same accepted Chain-of-KQA snapshots.

![Step-Verifiable Chain Growth and Path-Fold](../paper-preview/figures/figure1_chain_growth_path_fold.png)

![Agentic Operators over a Chain-of-KQA State](../paper-preview/figures/figure3_agentic_operators_chain_state.png)
