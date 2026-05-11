# 3. Method: The AgenQA Framework

[中文](./03_method.zh.md)

> Pre-release excerpt. This section is adapted from a work-in-progress manuscript and is not a final preprint.

AgenQA treats challenging reasoning-QA synthesis not as one-shot final-question generation, but as progressive construction of a step-verifiable reasoning chain followed by Path-Fold, which hides intermediate reasoning steps to form a globally challenging question.

![Step-Verifiable Chain Growth and Path-Fold](./figures/figure1_chain_growth_path_fold.png)

## Progressive Chain-of-KQA

A conventional QA item can be written as:

```text
x = (K, q, a)
```

where `K` is the known context given to the solver, `q` is the question, and `a` is the target answer. This representation describes a final problem, but it does not constrain how the problem is synthesized.

AgenQA instead makes generation progressive. Each step extends the current known/support state by one locally checkable answer-equivalent fact. A step is a KQA transition:

```text
s_t = (K_t, q_t, a_t, c_t)
```

Here:

- `K_t` is the visible support state before step `t`;
- `q_t` is the local question;
- `a_t` is the answer-equivalent conclusion produced by the step;
- `c_t` is a dependency certificate.

The certificate records the knowledge consumed and produced by the transition:

```text
c_t = (U_t, Delta F_t, f_t)
```

where `U_t` is the set of consumed premises or previously derived facts, `Delta F_t` is the set of newly produced facts, and `f_t` is the key fact whose content is answer-equivalent to `a_t`.

The chain forms a dependency spine:

```text
K_0 -> f_1 -> f_2 -> ... -> f_T
```

This prevents the generated object from degenerating into an unordered pool of related facts. Each accepted chain has both locally auditable transitions and a global dependency direction.

## Path-Fold and Solver-Facing Views

AgenQA exposes the same underlying chain through two solver-facing views.

For a local step, the **Edge View** exposes the support context needed to solve the local question. Edge success is therefore a proxy for step-level correctness and well-posedness.

The **Path View** constructs the chain-level challenge. Path-Fold keeps primitive premises, definitions, and conditions available to the solver, while hiding intermediate key facts and step certificates. The solver sees only the folded final question and must recover the latent dependency path from the visible context to the final answer.

This is the main difficulty-amplification mechanism. The hard item is not produced by asking a model to directly invent a difficult final question. It is produced by first growing a chain of locally checkable transitions, then folding away the intermediate conclusions.

A valid fold must satisfy:

- **Answer equivalence**: the folded question has the same target answer as the tail step.
- **Visibility separation**: intermediate key facts, step history, and internal pointers are not leaked as solver-visible givens.
- **Path preservation**: the folded question remains solvable only by reconstructing the dependency path or an equivalent reasoning path.

## Agentic State-Transition Harness

![Agentic Operators over a Chain-of-KQA State](./figures/figure2_agentic_operators_chain_state.png)

The synthesis process is a controlled state-transition loop over the progressive chain and its solver-facing views.

```text
d_r = D(S_r)
S_{r+1} = O_{d_r}(S_r)
z_{r+1} = E(Pi(S_{r+1}))
```

where:

- `S_r` is the synthesis state at round `r`;
- `D` is the Director controller;
- `d_r` is the selected operation;
- `O` is the corresponding Operator;
- `Pi` constructs Edge/Path projections;
- `E` returns solver, consensus, and acceptance signals.

### Director

The Director controls the current synthesis instance by selecting from the available operator set. It reads progress limits, recent chain history, memory windows, available operations, question-type policy, Edge/Path solver outcomes, consensus summaries, ambiguity and contract reports, and repair history.

### Operator

Operators are structured generation procedures over the Chain-of-KQA state. Initialization constructs the starting anchor. Extension appends a new KQA transition. Revision repairs the latest problematic transition while preserving its intended role in the chain.

In the semantic track, extension and revision share the same core role sequence:

```text
draft -> format -> certify -> fold
```

They differ in adapter logic and state-edit semantics: extension appends a new step, while revision overwrites or repairs a problematic step.

### Evaluator

The Evaluator probes generated Edge and Path views. Solvers answer both views, consensus aggregates multiple strong-solver judgments, and acceptance logic decides whether the current chain should be extended, revised, accepted, or terminated.

Evaluator feedback is not a post-hoc filter. It is part of the online state that conditions later Director decisions.

## Scalability and Extensibility

The harness is scalable because increasing chain length or task difficulty does not require a new final-question generator. Each round applies the same state-transition interface to the same Chain-of-KQA object: extension appends a transition, revision repairs a problematic transition, evaluation reads Edge/Path projections, and routing exports accepted states.

The harness is extensible because the formal object is separated from domain-specific implementations. Source grounding, operator adapters, view constructors, solver ensembles, and acceptance policies can be replaced while preserving the Chain-of-KQA, Edge View, and Path View interfaces.

## Failure-Mode Controls

AgenQA introduces concrete controls for recurring failure modes in synthetic reasoning-QA generation:

- **Grounding** anchors a synthesis instance in papers, technical reports, domain material, or other traceable sources.
- **State persistence** maintains premises and derived facts across steps.
- **Dependency certificates** record which prior premises or facts support each new answer-equivalent fact.
- **Contract-based stabilization** reduces semantic ambiguity and answer-format instability.
- **Evaluator-guided repair** uses solver feedback and judge signals to perform targeted local repair.

## Output Routing

Once a chain is accepted, AgenQA can route the same object to multiple downstream uses:

- benchmark construction, where Path questions serve as challenging items and Edge outcomes serve as correctness gates;
- supervised fine-tuning, where accepted snapshots can be exported as Edge QA, Path-direct QA, or mixtures;
- reinforcement-learning-style training, where Path questions can serve as terminal tasks and hidden dependency structure can support process-level signals.

In the current project stage, the primary public validation focuses on benchmark construction.
