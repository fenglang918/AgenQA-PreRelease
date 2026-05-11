# 2. Background and Motivation

[中文](./02_background_motivation.zh.md)

> Pre-release excerpt. This section is adapted from a work-in-progress manuscript and is not a final preprint.

## Synthetic Reasoning Data Background

Synthetic data for scientific and complex reasoning is not merely about generating more questions. Scientific post-training data work such as MegaScience shows that answer reliability, contamination control, length, domain coverage, and data mixture all affect whether downstream models learn useful reasoning capabilities. Data Darwinism further frames high-value data as a product of staged processing: raw text or simple QA must be transformed into objects that are more learnable, verifiable, and environment-like.

From this perspective, challenging reasoning QA synthesis is not a scale problem alone. It is a question of how to construct data objects that can carry complex reasoning while remaining checkable, filterable, and reusable.

## Synthesis Paradigms

Existing synthetic reasoning-data methods can be grouped by their generation mechanisms:

- seed or evolution-based methods rewrite or increase the complexity of existing problems;
- corpus or concept extraction methods extract and recombine problems from texts, knowledge points, or design logic;
- composition-based methods connect existing solvable items or prompts into longer tasks;
- bottom-up verifiable-subtask methods construct difficult problems from checkable subtasks;
- agentic benchmark pipelines decompose benchmark creation into planning, generation, verification, and evaluation;
- agentic proposing methods formulate problem synthesis as a sequential decision process.

Together, these paradigms show that high-value reasoning-data synthesis has moved from one-shot text generation toward controlled construction.

## The Correctness--Difficulty Coupling

Existing paradigms often couple the way difficulty is increased with the way correctness is maintained.

In direct final-QA generation or evolution, difficulty is commonly pursued through more complex problem statements. But this creates a dual failure mode. A strong model asked to generate hard questions may still produce items inside its own solving distribution, so apparent difficulty does not reliably translate into solver difficulty. Pushing the same process toward greater apparent difficulty can also introduce implicit assumptions, underspecified problem statements, unstable answers, or unsolvable instances.

Composition-based methods address this by amplifying long-horizon structure. R-HORIZON and Composition-RL show that composition and long dependency chains can create evaluation difficulty or restore learning signal. However, when composition happens after individual items are generated, dependency failures are difficult to locate and repair during synthesis.

Bottom-up methods such as CHASE are closer to our target because they bring verifiable subtasks into difficult-problem construction. Yet the key question for AgenQA is not only whether subtasks exist, but whether intermediate dependencies are maintained as a unified state that can keep growing, be checked step by step, and later be transformed into a hard solver-facing problem.

Across these paradigms, correctness is often checked after the fact or within isolated steps, while difficulty is increased through surface complexity, longer chains, or latent policy search. What is missing is a persistent intermediate object that lets a system grow reasoning dependencies while keeping each growth step step-verifiable.

## Dependency Paths as Control Objects

This motivates a different control object for hard-QA synthesis: the dependency path that connects the initial known context to the final answer.

AgenQA does not treat complex reasoning structure only as latent problem quality, post-hoc composition structure, or a byproduct of iterative generation. It materializes the structure as a persistent path state. Making this path explicit separates two roles that are otherwise entangled: individual growth steps can remain small enough to verify, while the accumulated path can encode a difficult multi-step dependency.

This also explains why progressive construction alone is not enough. If all intermediate facts are exposed, the task becomes scaffolded problem solving. If the final hard question is generated without an intermediate dependency state, correctness and repair become opaque.

Path-Fold is motivated by this gap: the generator should retain the dependency path for checking and repair, while the solver-facing problem should fold or hide intermediate conclusions so that difficulty comes from reconstructing a reliable path rather than from unverifiable one-shot complexity.

## Design Requirements

These observations yield six design requirements:

1. **Step-level verifiability.** Each generation step must be checkable in isolation.
2. **Progressive dependency growth.** The chain should grow dependencies rather than degenerate into loosely related questions.
3. **Persistent dependency state.** The latent reasoning space must be materialized as an auditable and repairable state.
4. **Visibility separation.** Intermediate facts available to the generator must not be leaked to the Path solver.
5. **Folded answer equivalence.** Folding must preserve the final answer and the intended dependency spine.
6. **Online evaluation and repair.** Quality control should occur during synthesis rather than only as post-hoc filtering.

The next section turns these requirements into AgenQA's formalization and system design.
