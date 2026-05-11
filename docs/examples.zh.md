# 代表性样例题

[English](./examples.md)

> 本页展示三道 selected **Path View** sample questions，用于说明 AgenQA 生成题目的形态与难度来源。它们不是完整 benchmark release，也不附带 source papers、raw solver responses、prompt snapshots 或 run artifacts。

这些样例都属于 solver-facing **Path View**：题面只给出最终求解所需的可见条件，中间 dependency path 被折叠，解题者需要自行恢复关键推理步骤。为保证 GitHub 预览稳定，长公式以 LaTeX snippet 展示。

## Sample 1. FedAvg 收敛界中的学习率选择

**Subject**: Federated and Distributed Learning

**Path View focus**: 从 FedAvg 的 smoothness、non-IID 梯度异质性和通信轮数条件出发，恢复收敛上界中的学习率优化步骤。

**Solver-facing question**

给定 FedAvg 设定：

- 全局目标函数为 `F(w) = sum_k (n_k / n) F_k(w)`。
- 每个客户端从 `w^t` 出发，以常数学习率 `eta > 0` 执行 `E > 1` 步本地 SGD，再由服务器按权重平均聚合。
- 客户端数据 non-IID，梯度异质性由加权 RMS 量 `G` 表示。
- 全局损失满足 `L`-smoothness。
- 一共进行 `T` 轮全客户端参与通信，`E, G, L, T` 均为正数。

要求求出使时间平均平方梯度范数上界最小的学习率 `eta*`，并给出优化后的最紧上界。最终答案只能使用 `F(w^0)-F(w*)`、`L`、`E`、`G`、`T`。

**Reference answer**

```latex
eta* =
sqrt( 2[F(w^0)-F(w*)] / (L E^2 G^2 T) )

(1/T) sum_{t=0}^{T-1} ||grad F(w^t)||^2
<=
2 sqrt( 2 L G^2 [F(w^0)-F(w*)] / T )
```

**Why this is a useful sample**

这道题展示了 Path View 的基本形态：题面不直接给出中间收敛界的单步优化式，而要求 solver 从 FedAvg 设定中恢复出学习率选择与优化后上界。

## Sample 2. Pointer-Generator Coverage Loss 的梯度路径

**Subject**: Natural Language Processing / Abstractive Summarization

**Path View focus**: 在 pointer-generator + coverage loss 结构中，恢复编码器隐藏状态 `h_i` 到总损失的两条梯度路径。

**Solver-facing question**

给定软注意力 encoder-decoder：

- attention score: `e_{t,i} = score(s_t, h_i)`
- attention weight: `a_{t,i} = exp(e_{t,i}) / sum_j exp(e_{t,j})`
- context vector: `c_t = sum_i a_{t,i} h_i`
- total loss: `L_total = L_NLL + lambda L_cov`
- coverage state: `cov_{t,i} = sum_{t' < t} a_{t',i}`
- coverage loss: `L_cov = sum_t sum_i min(a_{t,i}, cov_{t,i})`

要求推导 `partial L_total / partial h_i` 的单一闭式符号表达。由于 `score` 函数未固定，`partial e_{t,i} / partial h_i` 保持为未展开形式。答案应同时捕捉：

- `h_i -> e_{t,i} -> a_{t,*}` 的 score path；
- `h_i -> c_t -> L_NLL` 的 context path。

**Reference answer**

```latex
partial L_total / partial h_i
=
sum_t [
  a_{t,i} ( g_{t,i} - sum_j a_{t,j} g_{t,j} )
  * partial e_{t,i} / partial h_i
  +
  a_{t,i} * partial L_NLL / partial c_t
]
```

**Why this is a useful sample**

这道题体现了 Path-Fold 后的 chain-level challenge：solver 需要把 softmax Jacobian、coverage dependency 和 context-vector path 合并，而不是只回答一个局部梯度公式。

## Sample 3. TRADES 鲁棒训练目标的外层与内层条件

**Subject**: Machine Learning Security / Adversarial Robustness

**Path View focus**: 从 TRADES-style robust objective 出发，同时恢复外层 envelope-theorem gradient 与内层 PGD / KKT 条件。

**Solver-facing question**

给定参数化深度神经网络 `f_theta`，可行扰动集合为闭 `L_p` 球：

```latex
{ delta : ||delta||_p <= epsilon }
```

TRADES-style 目标为：

```latex
min_theta E_{(x,y)~mu} [
  L(f_theta(x), y)
  +
  beta * max_{||delta||_p <= epsilon}
    KL( f_theta(x) || f_theta(x + delta) )
]
```

其中 `KL(f_theta(x) || f_theta(x+delta))` 以 clean input distribution 为第一参数，以 perturbed input distribution 为第二参数。`delta*(theta)` 表示内层 KL 最大化问题的解。

要求合并推导：

1. `nabla_theta max_delta KL(...)` 的 envelope-theorem 形式，以及 `theta*` 处外层驻点条件；
2. 内层关于 `delta` 的归一化 PGD 更新规则；
3. 收敛时的 KKT 条件，包括梯度驻点、互补松弛和可行性。

**Reference answer**

```latex
nabla_theta max_{||delta||_p <= epsilon}
  KL(f_theta(x) || f_theta(x + delta))
=
nabla_theta KL(
  f_theta(x) || f_theta(x + delta*(theta))
) |_{delta = delta*(theta)}

nabla_theta E_{(x,y)~mu}[L(f_{theta*}(x), y)]
+
beta nabla_theta E_{(x,y)~mu}[
  KL(f_{theta*}(x) || f_{theta*}(x + delta*(theta*)))
]
= 0

delta_{t+1}
=
Proj_{||.||_p <= epsilon}(
  delta_t
  +
  alpha *
  nabla_delta KL(f_theta(x) || f_theta(x + delta_t))
  /
  ||nabla_delta KL(f_theta(x) || f_theta(x + delta_t))||_p
)

nabla_delta KL(f_theta(x) || f_theta(x + delta*))
=
lambda nabla_delta ||delta*||_p,
lambda >= 0

lambda (||delta*||_p - epsilon) = 0,
||delta*||_p <= epsilon
```

**Why this is a useful sample**

这道题展示了 AgenQA 题目可以覆盖较复杂的 optimization / robustness reasoning：solver 需要同时处理外层参数优化、内层 adversarial maximization 和 constrained optimality conditions。
