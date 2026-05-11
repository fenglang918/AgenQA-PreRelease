# AgenQA-PreRelease

**AgenQA 的预发布展示仓库：面向科学推理 QA 的 agentic data synthesis 与 benchmark construction。**

[English](./README.en.md) · [论文预览](./paper-preview/) · [系统架构](./docs/architecture.zh.md) · [实验摘要](./docs/experiments.zh.md) · [产物地图](./docs/artifact-map.zh.md)

> 这是一个面向投简历和项目展示的 public repo。它公开 AgenQA 的研究问题、方法动机、系统设计和脱敏实验信号。源码、原始题目、完整 prompt、真实 source materials 和私有协作材料暂不公开。

## Project Snapshot

AgenQA 研究的问题是：如何合成具有挑战性的 scientific reasoning QA 数据，同时不失去可验证性。它不是让模型一次性生成一个困难最终问题，而是先构造一条 **step-verifiable Chain-of-KQA**，再通过 **Path-Fold** 隐藏中间 facts，形成更难的端到端 **Path View**；同时保留局部 **Edge Views** 用于单步验证。

这个对象可以服务 benchmark construction、SFT 数据导出和 RL-style process signals。在这个预发布展示仓库中，主要公开验证对象是 **benchmark construction 与模型诊断**。

## My Role

我在 2025 年底到 2026 年初作为项目负责人推进 AgenQA，负责：

- **研究抽象**：将困难 QA 合成建模为 step-verifiable dependency-chain growth + Path-Fold，用 Edge/Path views 分离 correctness control 与 difficulty amplification。
- **系统设计**：设计 Director--Operator--Evaluator loop，包括 extend/revise 操作、solver feedback、consensus、answer/world contracts 和 replayable artifacts。
- **评测闭环**：围绕 model gradient、Edge/Path gap、depth-conditioned behavior 和质量过滤组织 benchmark validation。
- **协作推进**：协调多个 RA 子方向，将上游数据合成需求拆解到 benchmark、vision、coding、training-data 等可复用方向。

## 公开内容

本仓库包括：

- 当前论文前三个主体部分的 GitHub-readable preview；
- 面向外部读者的 AgenQA synthesis harness 架构说明；
- 脱敏后的 aggregate 实验摘要和表格；
- 不暴露原始题目/源码的 run artifact map；
- 方法图和系统图。

## 不公开内容

本仓库不包含：

- 实现源码；
- 原始生成题目或 solver raw outputs；
- 完整 prompts 或 prompt snapshots；
- source papers、source documents 或私有数据；
- 协作治理、人员判断、项目管理和私有决策材料。

## 论文预览

当前公开的预发布论文部分包括：

1. [Introduction](./paper-preview/01_introduction.zh.md)
2. [Background and Motivation](./paper-preview/02_background_motivation.zh.md)
3. [Method: The AgenQA Framework](./paper-preview/03_method.zh.md)

这些内容来自进行中的论文草稿，用于解释技术思想，不是最终可引用 preprint。

## 技术 ownership 概览

| Layer | AgenQA 做什么 | 我的贡献 |
| --- | --- | --- |
| Method | Chain-of-KQA、Edge/Path views、Path-Fold | 形式化 step-to-global synthesis object 并打磨论文表达 |
| Agent loop | Director、Operators、Evaluator、revise loop | 设计受控状态转移生命周期 |
| Verification | multi-strong solvers、consensus、contracts | 将 solver feedback 接入 acceptance 与 repair |
| Artifacts | 可回放 run directories 和 state snapshots | 让中间产物可审计、可调试、可评测 |
| Evaluation | model gradients、Edge/Path gaps、quality gates | 组织 benchmark validation 叙事和 aggregate reporting |

## Preliminary Evidence

内部项目已经完成多轮 synthesis / evaluation。这里公开的证据均为 aggregate / redacted：

- 早期验证阶段包含 37 次 synthesis runs / 175 个 generated QA candidates；
- 后续 paper-seed 5-strong 配置下完成 10-paper batch run，其中 9/10 成功，1 个 truncation failure；
- Qwen-family scale-gradient checks 和 SOTA comparison 用于初步 discriminativity evidence；
- Edge/Path analysis 用于区分 local step solvability 与 hidden-chain reconstruction difficulty。

详见 [实验摘要](./docs/experiments.zh.md)。

## Repository Status

这是一个 pre-release showcase。代码和 benchmark 的公开计划取决于论文评审、协作边界和数据安全约束。
