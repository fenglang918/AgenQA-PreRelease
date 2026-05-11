# AgenQA-PreRelease

**Pre-release project showcase for AgenQA: agentic data synthesis and benchmark construction for scientific reasoning QA.**

[中文说明](./README.md) · [Paper Preview](./paper-preview/README.en.md) · [Architecture](./docs/architecture.md) · [Evaluation Results](./docs/experiments.md) · [Artifact Map](./docs/artifact-map.md)

> This repository is a public hiring-context showcase. It explains the research problem, method, system design, and evaluation snapshots behind AgenQA. The source code, full prompts, source materials, and private collaboration artifacts are not public in this pre-release repository.

## Project Snapshot

AgenQA studies how to synthesize challenging scientific reasoning QA data without losing verifiability. Instead of asking a model to write a hard final question in one shot, AgenQA first grows a **step-verifiable Chain-of-KQA**, then uses **Path-Fold** to hide intermediate facts and create a harder end-to-end **Path View** while keeping local **Edge Views** available for verification.

The resulting object can support benchmark construction, supervised fine-tuning exports, and reinforcement-learning-style process signals. In this pre-release showcase, the primary validated use case is **benchmark construction and model diagnosis**.

## My Role

I led AgenQA as an academic collaboration project from November 2025 to March 2026. My work covered:

- **Research abstraction**: framed difficult QA synthesis as step-verifiable dependency-chain growth plus Path-Fold, with Edge/Path views separating correctness control from difficulty amplification.
- **System design**: designed the Director--Operator--Evaluator loop, including extend/revise operations, solver feedback, consensus, answer/world contracts, and replayable artifacts.
- **Evaluation loop**: organized the benchmark-validation story around model gradients, Edge/Path gaps, depth-conditioned behavior, and quality filtering.
- **Team coordination**: coordinated multiple RA sub-directions and translated upstream data-synthesis needs into reusable benchmark, vision, coding, and training-data tracks.

## What Is Public

This repository includes:

- a GitHub-readable preview of the current paper draft's first three substantive sections;
- public-facing architecture notes for the AgenQA synthesis harness;
- evaluation snapshots and model comparison tables;
- a safe artifact map showing what a run produces without exposing raw questions or source data;
- selected method figures used as visual explanation references.

## What Is Not Public

This repository does **not** include:

- implementation source code;
- raw generated benchmark items or solver outputs;
- full prompts or prompt snapshots;
- source papers, source documents, or private data;
- collaborator governance notes, internal project management records, or private decision logs.

## Paper Preview

The pre-release paper preview currently covers:

1. [Introduction](./paper-preview/01_introduction.md)
2. [Background and Motivation](./paper-preview/02_background_motivation.md)
3. [Method: The AgenQA Framework](./paper-preview/03_method.md)

These pages are adapted from a work-in-progress manuscript. They are intended to explain the technical idea, not to serve as a final citable preprint.

## Technical Ownership at a Glance

| Layer | What AgenQA Builds | My contribution |
| --- | --- | --- |
| Method | Chain-of-KQA, Edge/Path views, Path-Fold | Formalized the step-to-global synthesis object and public paper framing |
| Agent loop | Director, Operators, Evaluator, revise loop | Designed the controlled state-transition lifecycle |
| Verification | multi-strong solvers, consensus, contracts | Connected solver feedback to acceptance and repair decisions |
| Artifacts | replayable run directories and state snapshots | Made intermediate products auditable for debugging and evaluation |
| Evaluation | model gradients, Edge/Path gaps, quality gates | Organized the benchmark validation narrative and aggregate reporting |

## Preliminary Evidence

AgenQA has gone through multiple synthesis/evaluation rounds. The public evidence here illustrates difficulty and model-discrimination signal on solver-facing Path View questions; it should not be read as the paper's final experiment table:

- SOTA solvers are evaluated under a stronger problem-specification setting over 96 synthesis runs / 547 solver-facing Path View questions;
- this SOTA evaluation reaches `84.18%` overall accuracy and `51.77%` diagnostic-subset accuracy;
- the Qwen-family evaluation covers 37 synthesis runs / 175 Path View questions;
- the Qwen family shows a clear scale gradient: `qwen3-4b` at `48.00%`, `qwen3-8b` at `56.57%`, and `Qwen3-32B` at `66.86%`.

See [Experiments](./docs/experiments.md) for the public table form and caveats.

## Repository Status

This repo is a pre-release showcase. The code and benchmark release plan depends on paper-review, collaboration, and data-safety constraints.
