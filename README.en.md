# AgenQA-PreRelease

**Project showcase for AgenQA: agentic data synthesis and benchmark construction for scientific reasoning QA.**

[中文](./README.md) · [Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf) · [Prompt Files](./prompts/) · [Evaluation Results](./docs/experiments.md) · [Sample Questions](./docs/examples.md) · [Architecture](./docs/architecture.md)

## Paper Preview

The main showcase document is now a LaTeX-rendered PDF:

**[Open AgenQA Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf)**

The PDF follows a paper-style structure: title, abstract, introduction, background and motivation, method, two evaluation tables, three representative sample questions, conclusion, and references.

## Project Snapshot

AgenQA studies how to synthesize challenging scientific reasoning QA data without losing verifiability. Instead of asking a model to write a hard final question in one shot, AgenQA first grows a **step-verifiable Chain-of-KQA**, then uses **Path-Fold** to hide intermediate facts and create a harder end-to-end **Path View** while keeping local **Edge Views** available for verification.

The current showcase focuses on **benchmark construction and model diagnosis**. Evaluation snapshots show that AgenQA-generated Path View questions provide difficulty and model-discrimination signal.

## Supporting Pages

- [Paper Markdown Preview](./paper-preview/README.en.md)
- [Prompt Files](./prompts/)
- [Architecture Notes](./docs/architecture.md)
- [Evaluation Results](./docs/experiments.md)
- [Representative Sample Questions](./docs/examples.md)
- [Artifact Map](./docs/artifact-map.md)

## My Role

I led AgenQA as an academic collaboration project from November 2025 to March 2026. My work covered:

- **Research abstraction**: framed difficult QA synthesis as step-verifiable dependency-chain growth plus Path-Fold, with Edge/Path views separating correctness control from difficulty amplification.
- **System design**: designed the Director--Operator--Evaluator loop, including extend/revise operations, solver feedback, consensus, answer/world contracts, and replayable artifacts.
- **Evaluation loop**: organized the benchmark-validation story around model gradients, Edge/Path gaps, depth-conditioned behavior, and quality filtering.
- **Team coordination**: coordinated multiple RA sub-directions and translated upstream data-synthesis needs into reusable benchmark, vision, coding, and training-data tracks.

## Technical Ownership

| Layer | What AgenQA Builds | My contribution |
| --- | --- | --- |
| Method | Chain-of-KQA, Edge/Path views, Path-Fold | Formalized the step-to-global synthesis object and paper framing |
| Agent loop | Director, Operators, Evaluator, revise loop | Designed the controlled state-transition lifecycle |
| Verification | multi-strong solvers, consensus, contracts | Connected solver feedback to acceptance and repair decisions |
| Artifacts | replayable run directories and state snapshots | Made intermediate products auditable for debugging and evaluation |
| Evaluation | model gradients, Edge/Path gaps, quality gates | Organized the benchmark validation narrative and aggregate reporting |
