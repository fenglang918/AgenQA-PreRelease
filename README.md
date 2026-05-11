# AgenQA-PreRelease

**AgenQA 项目展示：面向科学推理 QA 的 agentic data synthesis 与 benchmark construction。**

[English](./README.en.md) · [论文预览](./paper-preview/) · [系统架构](./docs/architecture.zh.md) · [评测结果](./docs/experiments.zh.md) · [样例题](./docs/examples.zh.md) · [产物地图](./docs/artifact-map.zh.md)

## Introduction

面向 challenging reasoning QA 的高质量数据不只服务于 benchmark，它同时是评估和改进大语言模型的共同数据基底。它可以被实例化为 benchmark、supervised fine-tuning examples、reinforcement-learning tasks，或者 post-training 中的 curriculum data。

随着模型具备更强的指令遵循能力和更长的 inference-time reasoning 能力，如果每条答案背后的推理结构缺失，扁平 question-answer data 的诊断和训练价值都会下降。最终答案可以告诉我们模型是否答对，却很难解释成功来自稳健的多步推理、浅层模式匹配、幸运猜测，还是训练阶段已经见过相似实例。

与此同时，人工构造新鲜且具有挑战性的 reasoning questions 速度慢、成本高，也难以规模化。这推动了 synthetic QA data、自动 benchmark 构建以及 agent-assisted problem generation 等方向的发展。

### The Correctness--Difficulty Tension

合成高质量 challenging reasoning QA 并不只是让强模型生成更难的问题。核心困难在于：既要得到真正困难的问题，又要让构造过程保持可检查。

One-shot generation 往往首先失败在 difficulty 这一侧。强模型被要求出难题时，常常会生成仍落在自身解题分布内的问题，因此未必能挑战同级别 solver。继续用同一个 one-shot 过程制造表面难度时，又容易引入隐含假设、题面不完整、答案不稳定或答案等价性不可验证等问题。

反过来，如果题目被拆成小而完全暴露的步骤，每个局部步骤会更容易验证，但也可能过于简单，无法挑战强 solver。

### The AgenQA Object: Local Edges and Folded Paths

AgenQA 围绕 step-level correctness control 与 global difficulty amplification 的显式分离来组织 challenging reasoning QA synthesis。

它将 reasoning-QA synthesis 从一次性生成 final question，转化为先 progressive construction 一条 step-verifiable **Chain-of-KQA**，再通过 **Path-Fold** 隐藏中间推导步骤，把这条 chain 转化为 challenging final question。

这里的 **Edge** 和 **Path** 采用 graph sense：

- **Edge** 表示一个局部 dependency transition；
- **Path** 表示由这些 transitions 组成的多步 dependency route。

每个 KQA transition 都绑定 local known state、question、answer-equivalent key fact 和 dependency certificate。

在 **Edge View** 中，solver 会获得推出下一个 fact 所需的局部上下文，因此该 transition 更容易被验证和审计。在 **Path View** 中，Path-Fold 会隐藏中间推导步骤，迫使 solver 从初始 premises 到最终答案重构整条 route。

因此，局部 transitions 可以保持在强模型可验证的范围内，而折叠后的 path 可以变成全局困难问题。

### Agentic Synthesis Harness

这种形式化改变了 agent 在 QA generation 中的角色。AgenQA 并不把 agent 当成黑箱式出题者，而是让 agent 在显式 reasoning state 上操作。

整个 synthesis process 被实现为一个受控 state-transition loop：

- **Director** 读取当前 chain state、可选 operations 和 evaluator feedback；
- **Operators** 通过 initialization、extension 或 revision 修改 Chain-of-KQA state；
- **Edge/Path views** 在不同 visibility 条件下把状态暴露给 solvers；
- **Evaluator** 通过 solvers 和 consensus 检查这些 views，并返回 acceptance 与 repair signals。

在这个循环中，quality control 不是生成之后附加的 post-hoc filter，而是 generation process 本身的一部分。

## Method: The AgenQA Framework

AgenQA 不把 challenging reasoning-QA synthesis 视为一次性生成最终难题，而是先构造一条 step-verifiable reasoning chain，再通过 Path-Fold 隐藏中间推理步骤，形成全局更困难的问题。

![Step-Verifiable Chain Growth and Path-Fold](./paper-preview/figures/figure1_chain_growth_path_fold.png)

### Progressive Chain-of-KQA

普通 QA 可以写成：

```text
x = (K, q, a)
```

其中 `K` 是给 solver 的已知条件，`q` 是问题，`a` 是目标答案。这个表示只描述最终题目，却没有说明题目是怎样被合成出来的。

AgenQA 把生成过程改成 progressive generation。每一步只在当前已知状态上扩展一个局部可检查、与答案等价的新 fact。一个步骤是一个 KQA transition：

```text
s_t = (K_t, q_t, a_t, c_t)
```

其中：

- `K_t` 是第 `t` 步前可见的支撑信息；
- `q_t` 是该步局部问题；
- `a_t` 是该步得到的答案等价结论；
- `c_t` 是 dependency certificate。

Certificate 记录这一答案从哪里来：

```text
c_t = (U_t, Delta F_t, f_t)
```

其中 `U_t` 是该步使用的前提或先前事实，`Delta F_t` 是该步新产生的事实，`f_t` 是与 `a_t` 等价的关键事实。

这些关键事实形成一条 dependency spine：

```text
K_0 -> f_1 -> f_2 -> ... -> f_T
```

这可以避免生成结果退化成一组松散相关的问题。每条 accepted chain 同时具有 locally auditable transitions 和 global dependency direction。

### Path-Fold and Solver-Facing Views

AgenQA 将同一条 underlying chain 暴露为两种 solver-facing views。

**Edge View** 暴露求解当前局部问题所需的支撑信息。Edge success 因此可以作为 step-level correctness 和 well-posedness 的代理信号。

**Path View** 构造 chain-level challenge。Path-Fold 保留原始前提、定义和必要条件，但隐藏中间 key facts 和 step certificates。Solver 只能看到折叠后的最终问题，需要从可见上下文恢复通向最终答案的 latent dependency path。

这是 AgenQA 放大难度的关键机制。难题不是让模型直接凭空发明出来的，而是先长出一条局部可检查的 transition chain，再折叠隐藏其中间结论。

一个有效 fold 至少满足：

- **Answer equivalence**：折叠后的问题仍然以尾部步骤的答案为目标答案。
- **Visibility separation**：中间 key facts、step history 和内部指针不能泄露给 solver。
- **Path preservation**：题目仍然可以求解，但需要重构 dependency path 或等价推理路径。

### Agentic State-Transition Harness

![Agentic Operators over a Chain-of-KQA State](./paper-preview/figures/figure2_agentic_operators_chain_state.png)

整个 synthesis process 是作用在 progressive chain 及其 solver-facing views 上的 controlled state-transition loop。

```text
d_r = D(S_r)
S_{r+1} = O_{d_r}(S_r)
z_{r+1} = E(Pi(S_{r+1}))
```

其中：

- `S_r` 是第 `r` 轮 synthesis state；
- `D` 是 Director controller；
- `d_r` 是被选择的 operation；
- `O` 是对应 Operator；
- `Pi` 构造 Edge/Path projections；
- `E` 返回 solver、consensus 和 acceptance signals。

**Director** 负责从 available operator set 中选择下一步。它读取 progress limits、recent chain history、memory windows、available operations、question-type policy、Edge/Path solver outcomes、consensus summaries、ambiguity / contract reports 和 repair history。

**Operators** 是作用在 Chain-of-KQA state 上的 structured generation procedures。Initialization 构造 starting anchor。Extension 追加新的 KQA transition。Revision 修复最新的问题 transition，同时保持该步骤在 chain 中的 intended role。

在 semantic track 中，extension 和 revision 共享同一条核心 role sequence：

```text
draft -> format -> certify -> fold
```

二者区别在于 adapter logic 和 state-edit semantics：extension 追加新步骤，revision 覆盖或修复有问题的步骤。

**Evaluator** 检查生成出的 Edge 和 Path views。Solvers 同时回答两种 views，consensus 聚合多个 strong-solver judgments，acceptance logic 决定当前 chain 是继续 extend、revise、accept 还是 terminate。

Evaluator feedback 不是 post-hoc filter，而是 online state 的一部分，会影响后续 Director decisions。

### Scalability and Extensibility

这个 harness 可扩展，是因为增加 chain length 或 task difficulty 不需要重新设计一个 final-question generator。每一轮都应用同一个 state-transition interface：extension 追加 transition，revision 修复问题 transition，evaluation 读取 Edge/Path projections，routing 导出 accepted states。

这个 harness 可扩展，也是因为 formal object 与 domain-specific implementations 分离。Source grounding、operator adapters、view constructors、solver ensembles 和 acceptance policies 都可以替换，同时保留 Chain-of-KQA、Edge View 和 Path View 这套接口。

### Output Routing

一旦 chain 被接受，AgenQA 可以把同一对象路由到多种下游用途：

- benchmark construction：Path questions 作为 challenging items，Edge outcomes 作为 correctness gates；
- supervised fine-tuning：accepted snapshots 可以导出为 Edge QA、Path-direct QA 或 mixtures；
- reinforcement-learning-style training：Path questions 可以作为 terminal tasks，隐藏的 dependency structure 可以支持 process-level signals。

在当前项目阶段，主要验证聚焦 benchmark construction。

## Preliminary Evidence

AgenQA 已经完成多轮 synthesis / evaluation。当前评测信号显示 Path View 题目具有 difficulty 与 model-discrimination signal：

- SOTA solvers 在更明确的问题规约版本上评测了 96 次 synthesis runs / 547 道 solver-facing Path View questions；
- 这组 SOTA 评测的整体准确率为 `84.18%`，诊断子集准确率为 `51.77%`；
- Qwen-family 评测覆盖 37 次 synthesis runs / 175 道 Path View questions；
- Qwen 系列内呈现清晰梯度：`qwen3-4b` 为 `48.00%`，`qwen3-8b` 为 `56.57%`，`Qwen3-32B` 为 `66.86%`。

详见 [评测结果](./docs/experiments.zh.md) 和 [代表性样例题 PDF](./docs/examples.pdf)。

## More Documents

- [Background and Motivation](./paper-preview/02_background_motivation.zh.md)
- [Architecture Notes](./docs/architecture.zh.md)
- [Artifact Map](./docs/artifact-map.zh.md)
- [Full Paper Preview Directory](./paper-preview/)

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
