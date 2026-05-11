# 3. Method: The AgenQA Framework

[English](./03_method.md)

> 预发布节选。本节改写自进行中的论文草稿，不是最终 preprint。

AgenQA 不把 challenging reasoning-QA synthesis 视为一次性生成最终难题，而是先构造一条 step-verifiable reasoning chain，再通过 Path-Fold 隐藏中间推理步骤，形成全局更困难的问题。

![Step-Verifiable Chain Growth and Path-Fold](./figures/figure1_chain_growth_path_fold.png)

## Progressive Chain-of-KQA

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

## Path-Fold and Solver-Facing Views

AgenQA 将同一条 underlying chain 暴露为两种 solver-facing views。

**Edge View** 暴露求解当前局部问题所需的支撑信息。Edge success 因此可以作为 step-level correctness 和 well-posedness 的代理信号。

**Path View** 构造 chain-level challenge。Path-Fold 保留原始前提、定义和必要条件，但隐藏中间 key facts 和 step certificates。Solver 只能看到折叠后的最终问题，需要从可见上下文恢复通向最终答案的 latent dependency path。

这是 AgenQA 放大难度的关键机制。难题不是让模型直接凭空发明出来的，而是先长出一条局部可检查的 transition chain，再折叠隐藏其中间结论。

一个有效 fold 至少满足：

- **Answer equivalence**：折叠后的问题仍然以尾部步骤的答案为目标答案。
- **Visibility separation**：中间 key facts、step history 和内部指针不能泄露给 solver。
- **Path preservation**：题目仍然可以求解，但需要重构 dependency path 或等价推理路径。

## Agentic State-Transition Harness

![Agentic Operators over a Chain-of-KQA State](./figures/figure2_agentic_operators_chain_state.png)

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

### Director

Director 负责从 available operator set 中选择下一步。它读取 progress limits、recent chain history、memory windows、available operations、question-type policy、Edge/Path solver outcomes、consensus summaries、ambiguity / contract reports 和 repair history。

### Operator

Operators 是作用在 Chain-of-KQA state 上的 structured generation procedures。Initialization 构造 starting anchor。Extension 追加新的 KQA transition。Revision 修复最新的问题 transition，同时保持该步骤在 chain 中的 intended role。

在 semantic track 中，extension 和 revision 共享同一条核心 role sequence：

```text
draft -> format -> certify -> fold
```

二者区别在于 adapter logic 和 state-edit semantics：extension 追加新步骤，revision 覆盖或修复有问题的步骤。

### Evaluator

Evaluator 检查生成出的 Edge 和 Path views。Solvers 同时回答两种 views，consensus 聚合多个 strong-solver judgments，acceptance logic 决定当前 chain 是继续 extend、revise、accept 还是 terminate。

Evaluator feedback 不是 post-hoc filter，而是 online state 的一部分，会影响后续 Director decisions。

## Scalability and Extensibility

这个 harness 可扩展，是因为增加 chain length 或 task difficulty 不需要重新设计一个 final-question generator。每一轮都应用同一个 state-transition interface：extension 追加 transition，revision 修复问题 transition，evaluation 读取 Edge/Path projections，routing 导出 accepted states。

这个 harness 可扩展，也是因为 formal object 与 domain-specific implementations 分离。Source grounding、operator adapters、view constructors、solver ensembles 和 acceptance policies 都可以替换，同时保留 Chain-of-KQA、Edge View 和 Path View 这套接口。

## Failure-Mode Controls

AgenQA 引入了一组控制机制来处理 synthetic reasoning-QA generation 中反复出现的失败模式：

- **Grounding** 将 synthesis instance 锚定到论文、技术报告、领域材料或其他可追踪来源；
- **State persistence** 在步骤之间维护 premises 和 derived facts；
- **Dependency certificates** 记录哪些 prior premises 或 facts 支撑每个新的 answer-equivalent fact；
- **Contract-based stabilization** 降低语义歧义和答案格式不稳定；
- **Evaluator-guided repair** 使用 solver feedback 和 judge signals 做 targeted local repair。

## Output Routing

一旦 chain 被接受，AgenQA 可以把同一对象路由到多种下游用途：

- benchmark construction：Path questions 作为 challenging items，Edge outcomes 作为 correctness gates；
- supervised fine-tuning：accepted snapshots 可以导出为 Edge QA、Path-direct QA 或 mixtures；
- reinforcement-learning-style training：Path questions 可以作为 terminal tasks，隐藏的 dependency structure 可以支持 process-level signals。

在当前项目阶段，主要公开验证聚焦 benchmark construction。
