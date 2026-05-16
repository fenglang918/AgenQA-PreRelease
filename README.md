# AgenQA-PreRelease

**AgenQA 研究项目预览：agentic data synthesis for scientific reasoning QA benchmark construction。**

[Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf) · [实验展示](./docs/experiments.zh.md) · [样例题 PDF](./docs/examples.pdf) · [系统架构](./docs/architecture.zh.md) · [Prompt 文件](./prompts/) · [English](./README.en.md)

## 30 秒版

AgenQA 研究的问题是：如何合成真正有挑战性的 scientific reasoning QA 数据，同时让生成过程仍然可检查、可修复、可评测。

![AgenQA overview](./paper-preview/figures/figure1_chain_growth_path_fold.png)

- **核心方法**：先构造一条 step-verifiable **Chain-of-KQA**，再通过 **Path-Fold** 隐藏中间 facts，形成更难的 solver-facing **Path View**。
- **验证视角**：同一条 dependency chain 同时投影为局部 **Edge View** 和全局 **Path View**，分离 correctness control 与 difficulty amplification。
- **系统设计**：用 **Director--Operator--Evaluator loop** 组织 init / extend / revise / finish，让 generation、verification 和 repair 发生在同一显式 reasoning state 上。
- **阶段信号**：Path View 题目在 SOTA solver 评测中整体准确率为 **84.18%**，诊断子集准确率为 **51.77%**；在 Qwen-family 内，同一 benchmark 的 accuracy 与模型规模/理论能力呈正相关（4B 为 **48.00%**，32B 为 **66.86%**）。

## 推荐阅读路径

**首选阅读：** [AgenQA Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf)

PDF 已整理为论文式展示文档，包含标题、摘要、Motivation and Core Idea、Contributions、AgenQA Framework、Experimental Showcase、Conclusion 和 References。它是这个 repo 最完整、最顺眼的阅读入口。

| 想快速看什么 | 入口 |
| --- | --- |
| 论文式项目说明 | [Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf) |
| 两个核心实验表 | [docs/experiments.zh.md](./docs/experiments.zh.md) |
| 三道代表性样例题 | [docs/examples.pdf](./docs/examples.pdf) |
| 系统与方法架构 | [docs/architecture.zh.md](./docs/architecture.zh.md) |
| Prompt 角色边界 | [prompts/](./prompts/) |

## 核心贡献

1. **Edge/Path-grounded Chain-of-KQA formalism**：将 correctness--difficulty tension 转化为 step-to-global design principle：Edge View 支持 step-level correctness control，Path-Fold 在同一条 generated chain 上放大全局 difficulty。
2. **Scalable and extensible agentic synthesis harness**：将 QA generation 视为显式知识依赖上的受控 state-transition process；Director--Operator--Evaluator loop 通过 grounding、state persistence、dependency auditability、contract-based stabilization 和 evaluator-guided repair 等显式 controls 支持可扩展 synthesis，并保持 Operator layer modular。
3. **Pre-release benchmark evidence with SOTA and Qwen-family evaluations**：当前 public preview 展示两组 benchmark-construction signals：SOTA solver 评测覆盖 `96` 次 synthesis runs / `547` 道 Path View questions，整体准确率为 **84.18%**，诊断子集准确率为 **51.77%**，说明 Path View 在强模型区间仍保留难度和区分区域；Qwen-family 梯度评测覆盖 `37` 次 runs / `175` 道 Path View questions，同一 benchmark 上 accuracy 从 **48.00%**（4B）上升到 **66.86%**（32B），与模型尺寸/预期能力一致，可作为 benchmark 合理性的 scale-consistency sanity check，并低成本补充人工逐题审查。

## Core Idea

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
| 评测闭环 | 组织 SOTA solver comparison、Qwen-family scale-correlation analysis、Edge/Path gap 和质量过滤分析 |
| 论文表达 | 将内部系统语言改写为 public-facing paper / research framing |
| 协作推进 | 协调多个协作方向，将上游数据合成需求拆解到 benchmark、vision、coding、training-data 等可复用方向 |

## Repository Map

```text
paper-preview/   论文式项目预览；推荐优先阅读 PDF
docs/            架构说明、实验结果、样例题和 artifact map
prompts/         公开的角色级 prompt excerpts
```
