# 2. Background and Motivation

[English](./02_background_motivation.md)

> 预发布节选。本节改写自进行中的论文草稿，不是最终 preprint。

## Synthetic Reasoning Data Background

面向科学与复杂推理的 synthetic data 并不只是“生成更多题目”。MegaScience 等科学后训练数据工作表明，数据的答案可靠性、去污染、长度控制、领域覆盖和配方都会影响下游模型是否真正学到推理能力。Data Darwinism 进一步从数据加工层级提醒我们，高价值数据需要从原始文本或简单问答转化为更可学习、更可验证、更接近任务环境的对象。

由此看，challenging reasoning QA synthesis 的问题不是规模本身，而是如何构造一种既能承载复杂推理，又能被检查、筛选和复用的数据对象。

## Synthesis Paradigms

已有 synthetic reasoning-data 方法可以按生成机制分为几类：

- seed 或 evolution-based methods 从已有题目出发改写或提升复杂度；
- corpus 或 concept extraction methods 从语料、知识点或设计逻辑中抽取并重组问题；
- composition-based methods 将已有可解题目或 prompts 组合成更长任务；
- bottom-up verifiable-subtask methods 从可检查的子任务构造困难题；
- agentic benchmark pipelines 将 benchmark creation 拆成 planning、generation、verification 和 evaluation；
- agentic proposing methods 将 problem synthesis 组织为 sequential decision process。

这些范式共同说明，合成高价值推理数据已经从一次性文本生成，转向受控构造过程。

## The Correctness--Difficulty Coupling

已有范式往往把“如何提高 difficulty”和“如何维持 correctness”耦合在一起。

Direct final-QA generation 或 evolution 通常通过更复杂的题面来追求难度，但这会带来双重失败模式：强模型被要求生成难题时，仍可能产出落在自身解题分布内的问题，所以表面难度不一定转化为 solver difficulty；如果继续提高表面难度，又容易引入隐含条件、题面不自足、答案不稳定或不可解实例。

Composition-based methods 试图通过放大 long-horizon structure 来缓解这一点。R-HORIZON 和 Composition-RL 都说明 composition / long dependency chains 可以制造评测难度或恢复训练信号。但如果 composition 发生在 individual items 生成之后，dependency failures 很难在 synthesis 过程中被定位和修复。

CHASE 这类 bottom-up methods 更接近我们的目标，因为它把 verifiable subtasks 纳入 difficult-problem construction。但对 AgenQA 来说，关键还不只是“有子任务”，而是 intermediate dependencies 能否被维护为一个统一 state：它要能持续生长、能 step by step 地检查，并且之后能转化为 hard solver-facing problem。

总体看，correctness 往往是在事后或孤立步骤中被检查，而 difficulty 则通过表面复杂化、更长链条或 latent policy search 来提高。这里缺失的是一个 persistent intermediate object，使系统可以在每一步保持 step-level verifiability 的同时持续增长 reasoning dependencies。

## Dependency Paths as Control Objects

因此，hard-QA synthesis 需要一个不同的 control object：从初始 known context 连接到最终答案的 dependency path。

AgenQA 不把复杂推理结构只当作 latent problem quality、post-hoc composition structure 或 iterative generation 的副产品，而是将其 materialized 为 persistent path state。显式化这条 path 可以把原本纠缠在一起的两个目标分开：individual growth steps 可以保持足够小，从而便于验证；累积起来的 path 仍然可以承载困难的多步依赖。

这种分离也解释了为什么 progressive construction 本身还不够。如果所有中间 facts 都暴露给 solver，任务会退化成脚手架式解题；如果直接生成 final hard question 而不保留中间依赖状态，正确性和修复又会变得不透明。

Path-Fold 的动机正来自这个缺口：生成侧应保留 dependency path 用于检查和修复，而 solver-facing problem 应折叠或隐藏中间结论，使难度来自对一条可靠路径的重构，而不是不可验证的一次性复杂化。

## Design Requirements

由此，一个合成方法需要满足六项设计要求：

1. **Step-level verifiability.** 每一步生成都必须能被独立检查。
2. **Progressive dependency growth.** 链条应持续增长依赖，而不是退化为若干松散相关的问题。
3. **Persistent dependency state.** 隐含推理空间必须被 materialized 为可审计、可修复的 state。
4. **Visibility separation.** 生成侧可见的中间 facts 不能直接泄漏给 Path solver。
5. **Folded answer equivalence.** Folding 必须保持最终答案与预期 dependency spine。
6. **Online evaluation and repair.** 质量控制应发生在 synthesis 过程中，而不是只做 post-hoc filtering。

下一节将这些要求具体化为 AgenQA 的形式化定义与系统设计。
