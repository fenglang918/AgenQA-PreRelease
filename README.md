# AgenQA-PreRelease

**AgenQA 项目展示：面向科学推理 QA 的 agentic data synthesis 与 benchmark construction。**

[English](./README.en.md) · [论文预览](./paper-preview/) · [系统架构](./docs/architecture.zh.md) · [评测结果](./docs/experiments.zh.md) · [样例题](./docs/examples.zh.md) · [产物地图](./docs/artifact-map.zh.md)

## Project Snapshot

AgenQA 研究的问题是：如何合成具有挑战性的 scientific reasoning QA 数据，同时不失去可验证性。它不是让模型一次性生成一个困难最终问题，而是先构造一条 **step-verifiable Chain-of-KQA**，再通过 **Path-Fold** 隐藏中间 facts，形成更难的端到端 **Path View**；同时保留局部 **Edge Views** 用于单步验证。

这个对象可以服务 benchmark construction、SFT 数据导出和 RL-style process signals。当前展示重点是 **benchmark construction 与模型诊断**。

## My Role

我在 2025 年 11 月至 2026 年 3 月作为项目负责人推进 AgenQA，负责：

- **研究抽象**：将困难 QA 合成建模为 step-verifiable dependency-chain growth + Path-Fold，用 Edge/Path views 分离 correctness control 与 difficulty amplification。
- **系统设计**：设计 Director--Operator--Evaluator loop，包括 extend/revise 操作、solver feedback、consensus、answer/world contracts 和 replayable artifacts。
- **评测闭环**：围绕 model gradient、Edge/Path gap、depth-conditioned behavior 和质量过滤组织 benchmark validation。
- **协作推进**：协调多个 RA 子方向，将上游数据合成需求拆解到 benchmark、vision、coding、training-data 等可复用方向。

## 内容导航

本仓库包括：

- 当前论文前三个主体部分的 GitHub-readable preview；
- 面向外部读者的 AgenQA synthesis harness 架构说明；
- 阶段性评测结果和模型对比表；
- 三道 representative Path View sample questions；
- run artifact map；
- 方法图和系统图。

## 论文预览

当前论文预览包括：

1. [Introduction](./paper-preview/01_introduction.zh.md)
2. [Background and Motivation](./paper-preview/02_background_motivation.zh.md)
3. [Method: The AgenQA Framework](./paper-preview/03_method.zh.md)

这些内容用于解释 AgenQA 的问题定义、研究动机和方法设计。

## 技术 ownership 概览

| Layer | AgenQA 做什么 | 我的贡献 |
| --- | --- | --- |
| Method | Chain-of-KQA、Edge/Path views、Path-Fold | 形式化 step-to-global synthesis object 并打磨论文表达 |
| Agent loop | Director、Operators、Evaluator、revise loop | 设计受控状态转移生命周期 |
| Verification | multi-strong solvers、consensus、contracts | 将 solver feedback 接入 acceptance 与 repair |
| Artifacts | 可回放 run directories 和 state snapshots | 让中间产物可审计、可调试、可评测 |
| Evaluation | model gradients、Edge/Path gaps、quality gates | 组织 benchmark validation 叙事和 aggregate reporting |

## Preliminary Evidence

AgenQA 已经完成多轮 synthesis / evaluation。当前评测信号显示 Path View 题目具有 difficulty 与 model-discrimination signal：

- SOTA solvers 在更明确的问题规约版本上评测了 96 次 synthesis runs / 547 道 solver-facing Path View questions；
- 这组 SOTA 评测的整体准确率为 `84.18%`，诊断子集准确率为 `51.77%`；
- Qwen-family 评测覆盖 37 次 synthesis runs / 175 道 Path View questions；
- Qwen 系列内呈现清晰梯度：`qwen3-4b` 为 `48.00%`，`qwen3-8b` 为 `56.57%`，`Qwen3-32B` 为 `66.86%`。

详见 [评测结果](./docs/experiments.zh.md)。
