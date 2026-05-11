# AgenQA-PreRelease

**AgenQA 项目展示：面向科学推理 QA 的 agentic data synthesis 与 benchmark construction。**

[English](./README.en.md) · [Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf) · [评测结果](./docs/experiments.zh.md) · [样例题](./docs/examples.zh.md) · [系统架构](./docs/architecture.zh.md)

## Paper Preview

主展示文档已整理为 LaTeX PDF：

**[打开 AgenQA Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf)**

PDF 按论文结构组织：

1. Title
2. Abstract
3. Introduction
4. Background and Motivation
5. Method: The AgenQA Framework
6. Evaluation Snapshot：两个实验表格
7. Representative Path View Samples：三道题目展示
8. Conclusion
9. References

## Project Snapshot

AgenQA 研究的问题是：如何合成具有挑战性的 scientific reasoning QA 数据，同时不失去可验证性。它不是让模型一次性生成一个困难最终问题，而是先构造一条 **step-verifiable Chain-of-KQA**，再通过 **Path-Fold** 隐藏中间 facts，形成更难的端到端 **Path View**；同时保留局部 **Edge Views** 用于单步验证。

当前展示重点是 **benchmark construction 与模型诊断**。阶段性评测显示，AgenQA 生成的 Path View 题目具有 difficulty 与 model-discrimination signal。

## Supporting Pages

- [论文 Markdown 预览](./paper-preview/)
- [系统架构说明](./docs/architecture.zh.md)
- [阶段性评测结果](./docs/experiments.zh.md)
- [代表性样例题](./docs/examples.zh.md)
- [产物地图](./docs/artifact-map.zh.md)

## My Role

我在 2025 年 11 月至 2026 年 3 月作为项目负责人推进 AgenQA，负责：

- **研究抽象**：将困难 QA 合成建模为 step-verifiable dependency-chain growth + Path-Fold，用 Edge/Path views 分离 correctness control 与 difficulty amplification。
- **系统设计**：设计 Director--Operator--Evaluator loop，包括 extend/revise 操作、solver feedback、consensus、answer/world contracts 和 replayable artifacts。
- **评测闭环**：围绕 model gradient、Edge/Path gap、depth-conditioned behavior 和质量过滤组织 benchmark validation。
- **协作推进**：协调多个 RA 子方向，将上游数据合成需求拆解到 benchmark、vision、coding、training-data 等可复用方向。

## Technical Ownership

| Layer | AgenQA 做什么 | 我的贡献 |
| --- | --- | --- |
| Method | Chain-of-KQA、Edge/Path views、Path-Fold | 形式化 step-to-global synthesis object 并打磨论文表达 |
| Agent loop | Director、Operators、Evaluator、revise loop | 设计受控状态转移生命周期 |
| Verification | multi-strong solvers、consensus、contracts | 将 solver feedback 接入 acceptance 与 repair |
| Artifacts | 可回放 run directories 和 state snapshots | 让中间产物可审计、可调试、可评测 |
| Evaluation | model gradients、Edge/Path gaps、quality gates | 组织 benchmark validation 叙事和 aggregate reporting |
