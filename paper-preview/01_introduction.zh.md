# 1. Introduction

[English](./01_introduction.md)

## Challenging Reasoning QA as a Data Substrate

面向 challenging reasoning QA 的高质量数据不只服务于 benchmark，它同时是评估和改进大语言模型的共同数据基底。它可以被实例化为 benchmark、supervised fine-tuning examples、reinforcement-learning tasks，或者 post-training 中的 curriculum data。

随着模型具备更强的指令遵循能力和更长的 inference-time reasoning 能力，如果每条答案背后的推理结构缺失，扁平 question-answer data 的诊断和训练价值都会下降。最终答案可以告诉我们模型是否答对，却很难解释成功来自稳健的多步推理、浅层模式匹配、幸运猜测，还是训练阶段已经见过相似实例。

与此同时，人工构造新鲜且具有挑战性的 reasoning questions 速度慢、成本高，也难以规模化。这推动了 synthetic QA data、自动 benchmark 构建以及 agent-assisted problem generation 等方向的发展。

## The Correctness--Difficulty Tension

合成高质量 challenging reasoning QA 并不只是让强模型生成更难的问题。核心困难在于：既要得到真正困难的问题，又要让构造过程保持可检查。

One-shot generation 往往首先失败在 difficulty 这一侧。强模型被要求出难题时，常常会生成仍落在自身解题分布内的问题，因此未必能挑战同级别 solver。继续用同一个 one-shot 过程制造表面难度时，又容易引入隐含假设、题面不完整、答案不稳定或答案等价性不可验证等问题。

反过来，如果题目被拆成小而完全暴露的步骤，每个局部步骤会更容易验证，但也可能过于简单，无法挑战强 solver。

## The AgenQA Object: Local Edges and Folded Paths

AgenQA 围绕 step-level correctness control 与 global difficulty amplification 的显式分离来组织 challenging reasoning QA synthesis。

它将 reasoning-QA synthesis 从一次性生成 final question，转化为先 progressive construction 一条 step-verifiable **Chain-of-KQA**，再通过 **Path-Fold** 隐藏中间推导步骤，把这条 chain 转化为 challenging final question。

这里的 **Edge** 和 **Path** 采用 graph sense：

- **Edge** 表示一个局部 dependency transition；
- **Path** 表示由这些 transitions 组成的多步 dependency route。

每个 KQA transition 都绑定 local known state、question、answer-equivalent key fact 和 dependency certificate。

在 **Edge View** 中，solver 会获得推出下一个 fact 所需的局部上下文，因此该 transition 更容易被验证和审计。在 **Path View** 中，Path-Fold 会隐藏中间推导步骤，迫使 solver 从初始 premises 到最终答案重构整条 route。

因此，局部 transitions 可以保持在强模型可验证的范围内，而折叠后的 path 可以变成全局困难问题。

## Agentic Synthesis Harness

这种形式化改变了 agent 在 QA generation 中的角色。AgenQA 并不把 agent 当成黑箱式出题者，而是让 agent 在显式 reasoning state 上操作。

整个 synthesis process 被实现为一个受控 state-transition loop：

- **Director** 读取当前 chain state、可选 operations 和 evaluator feedback；
- **Operators** 通过 initialization、extension 或 revision 修改 Chain-of-KQA state；
- **Edge/Path views** 在不同 visibility 条件下把状态暴露给 solvers；
- **Evaluator** 通过 solvers 和 consensus 检查这些 views，并返回 acceptance 与 repair signals。

在这个循环中，quality control 不是生成之后附加的 post-hoc filter，而是 generation process 本身的一部分。

## Downstream Routing and Validation

AgenQA 旨在让同一批 accepted chains 支持多种下游用途：benchmark construction、supervised fine-tuning 和 reinforcement-learning-style training。这些用途不是从 raw generated questions 开始，而是从 accepted Chain-of-KQA snapshots 及其 harness-produced selection signals 开始，包括 Edge/Path solver outcomes、consensus summaries、contracts、dependency certificates 和 path-integrity checks。

在当前论文框架中，primary empirical validation 聚焦 benchmark construction。Edge-side solver signals 作为 correctness 与 well-posedness gates，Path-side solver distributions 提供 difficulty 与 discrimination signals。

## Contributions

1. **Edge/Path-grounded Chain-of-KQA formalism.** AgenQA 将 correctness--difficulty tension 转化为 step-to-global design principle：Edge views 支持 step-level correctness control，Path-Fold 在同一条 generated chain 上放大全局 difficulty。
2. **Scalable and extensible agentic synthesis harness.** AgenQA 将 QA generation 视为显式知识依赖上的受控 state-transition process，以 Director--Operator--Evaluator loop 组织。
3. **Benchmark-construction validation.** AgenQA 作为 benchmark-construction framework 进行评估，重点验证 harness-produced signals 是否支持 accepted chains 的 filtering、slicing 与 routing。
