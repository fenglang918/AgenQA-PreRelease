# 1. Introduction

[中文](./01_introduction.zh.md)

## Challenging Reasoning QA as a Data Substrate

High-quality data for challenging reasoning question answering (QA) is a shared substrate for both evaluating and improving large language models. It can be instantiated as benchmarks, supervised fine-tuning examples, reinforcement-learning tasks, or curriculum data for post-training.

As models gain stronger instruction-following and longer inference-time reasoning capabilities, flat question-answer data provides diminishing diagnostic and training value when the reasoning structure behind each answer is missing. A final answer can indicate whether a model succeeded, but rarely explains whether the success came from robust multi-step reasoning, shallow pattern matching, lucky guessing, or exposure to a similar instance during training.

At the same time, manually constructing fresh and challenging reasoning questions is slow, expensive, and difficult to scale. This motivates synthetic QA data, automatic benchmark construction, and agent-assisted problem generation.

## The Correctness--Difficulty Tension

Synthesizing high-quality data for challenging reasoning QA is not simply a matter of asking a strong model to generate harder questions. The central difficulty is a tension between producing questions that are genuinely hard and keeping their construction checkable.

One-shot generation often fails on the difficulty side first. A strong model asked to write a hard question tends to produce items that remain within its own solving distribution, and therefore may not challenge comparable solvers. Pushing the same one-shot process toward apparent difficulty can introduce implicit assumptions, underspecified statements, unstable answers, or unverifiable answer equivalence.

Conversely, if a problem is decomposed into small and fully exposed steps, each local step becomes easier to verify but may be too easy to challenge strong solvers.

## The AgenQA Object: Local Edges and Folded Paths

AgenQA organizes challenging reasoning QA synthesis around an explicit separation between step-level correctness control and global difficulty amplification.

It converts reasoning-QA synthesis from one-shot final-question generation into progressive construction of a step-verifiable **Chain-of-KQA**, followed by **Path-Fold**, which hides intermediate reasoning steps and turns the chain into a challenging final question.

We use **Edge** and **Path** in the graph sense:

- an **Edge** denotes a single local dependency transition;
- a **Path** denotes the multi-step dependency route composed from such transitions.

Each KQA transition binds a local known state, a question, an answer-equivalent key fact, and a dependency certificate.

In an **Edge View**, a solver receives the local context needed to derive one next fact, making the transition easier to verify and audit. In a **Path View**, Path-Fold hides the intermediate derivation steps, forcing the solver to reconstruct the route from the initial premises to the final answer.

Thus, individual transitions can remain within the verification capacity of strong models, while the folded path can become globally challenging.

## Agentic Synthesis Harness

This formulation changes the role of agents in QA generation. AgenQA does not treat an agent as a black-box question writer. It uses agents to operate over an explicit reasoning state.

The synthesis process is implemented as a controlled state-transition loop:

- a **Director** reads the current chain state, available operations, and evaluator feedback;
- **Operators** edit the Chain-of-KQA state through initialization, extension, or revision;
- **Edge/Path views** expose the state to solvers under different visibility conditions;
- an **Evaluator** probes the views through solvers and consensus, returning acceptance and repair signals.

In this loop, quality control is not a post-hoc filter applied after generation. It is part of the generation process itself.

## Downstream Routing and Validation

AgenQA is designed to support multiple downstream uses of the same accepted chains: benchmark construction, supervised fine-tuning, and reinforcement-learning-style training. These uses do not start from raw generated questions alone. They start from accepted Chain-of-KQA snapshots together with harness-produced selection signals, including Edge/Path solver outcomes, consensus summaries, contracts, dependency certificates, and path-integrity checks.

In the current paper framing, the primary empirical validation focuses on benchmark construction. Edge-side solver signals serve as correctness and well-posedness gates, while Path-side solver distributions provide difficulty and discrimination signals.

## Contributions

1. **Edge/Path-grounded Chain-of-KQA formalism.** AgenQA turns the correctness--difficulty tension into a step-to-global design principle: Edge views support step-level correctness control, while Path-Fold amplifies global difficulty over the same generated chain.
2. **Scalable and extensible agentic synthesis harness.** AgenQA treats QA generation as a controlled state-transition process over explicit knowledge dependencies, organized as a Director--Operator--Evaluator loop.
3. **Benchmark-construction validation.** AgenQA is evaluated as a benchmark-construction framework, focusing on whether harness-produced signals enable reliable filtering, slicing, and routing of accepted chains.
