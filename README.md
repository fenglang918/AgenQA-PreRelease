# AgenQA-PreRelease

**面向 AI research / RA 投递场景的 AgenQA 项目预览：agentic data synthesis for scientific reasoning QA benchmark construction。**

[Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf) · [实验展示](./docs/experiments.zh.md) · [样例题 PDF](./docs/examples.pdf) · [系统架构](./docs/architecture.zh.md) · [Prompt 文件](./prompts/) · [English](./README.en.md)

## 30 秒版

AgenQA 研究的问题是：如何合成真正有挑战性的 scientific reasoning QA 数据，同时让生成过程仍然可检查、可修复、可评测。

- **核心方法**：先构造一条 step-verifiable **Chain-of-KQA**，再通过 **Path-Fold** 隐藏中间 facts，形成更难的 solver-facing **Path View**。
- **验证视角**：同一条 dependency chain 同时投影为局部 **Edge View** 和全局 **Path View**，分离 correctness control 与 difficulty amplification。
- **系统设计**：用 **Director--Operator--Evaluator loop** 组织 init / extend / revise / finish，让 generation、verification 和 repair 发生在同一显式 reasoning state 上。
- **阶段信号**：Path View 题目在 SOTA solver 评测中整体准确率为 **84.18%**，诊断子集准确率为 **51.77%**；Qwen-family 梯度从 **48.00%** 提升到 **66.86%**。

## 推荐阅读路径

**首选阅读：** [AgenQA Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf)

PDF 已整理为论文式展示文档，包含标题、摘要、Motivation and Core Idea、AgenQA Framework、Experimental Showcase、Conclusion 和 References。它是这个 repo 最完整、最顺眼的阅读入口。

| 想快速看什么 | 入口 |
| --- | --- |
| 论文式项目说明 | [Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf) |
| 两个核心实验表 | [docs/experiments.zh.md](./docs/experiments.zh.md) |
| 三道代表性样例题 | [docs/examples.pdf](./docs/examples.pdf) |
| 系统与方法架构 | [docs/architecture.zh.md](./docs/architecture.zh.md) |
| Prompt 角色边界 | [prompts/](./prompts/) |

## Core Idea

![AgenQA overview](./paper-preview/figures/figure1_chain_growth_path_fold.png)

AgenQA 不把困难题生成当作一次性 final-question generation，而是把它拆成一个可控制的 synthesis object：

1. 在生成侧维护可审计的 dependency chain；
2. 在每一步保留 local certificate，用 Edge View 检查 transition 是否 well-posed；
3. 在 solver-facing 侧通过 Path-Fold 隐藏中间依赖，让模型必须恢复完整 reasoning path；
4. 用 evaluator feedback 驱动 revise，而不是只在生成后做 post-hoc filtering。

## Technical Highlights

- **Step-Verifiable Dependency Chains**：每一步都绑定可见前提、局部问题、答案等价事实与 dependency certificate。
- **Edge / Path Views**：Edge View 服务局部验证，Path View 服务全局挑战。
- **Path-Fold**：折叠中间 facts，同时保留 answer equivalence 和 path integrity。
- **Agentic State-Transition Harness**：Director 选择操作，Operators 修改 chain state，Evaluator 读取 solver / consensus / contract signals。
- **Scalable and Extensible Design**：extend、revise、evaluation、routing 共用同一 state interface，方便扩展 chain length、solver ensemble 和 domain adapters。

## My Role

我在 **2025 年 11 月至 2026 年 3 月** 作为项目负责人推进 AgenQA，主要负责：

| 方向 | 工作内容 |
| --- | --- |
| 研究抽象 | 将困难 QA 合成建模为 step-verifiable dependency-chain growth + Path-Fold |
| 系统设计 | 设计 Director--Operator--Evaluator loop，以及 extend / revise / evaluator feedback 的状态转移生命周期 |
| 评测闭环 | 组织 SOTA solver comparison、Qwen scale gradient、Edge/Path gap 和质量过滤分析 |
| 论文表达 | 将内部系统语言改写为 public paper / hiring showcase 可读的 research framing |
| 协作推进 | 协调多个 RA 子方向，将上游数据合成需求拆解到 benchmark、vision、coding、training-data 等可复用方向 |

## Repository Map

```text
paper-preview/   论文式项目预览；推荐优先阅读 PDF
docs/            架构说明、实验结果、样例题和 artifact map
prompts/         公开的角色级 prompt excerpts
```
