# 代表性样例题

[English](./examples.md)

> 本页展示三道 selected Path View sample questions，用于说明 AgenQA 生成题目的形态与难度来源。它们不是完整 benchmark release，也不附带 source papers、raw solver responses、prompt snapshots 或 run artifacts。

这些样例都属于 solver-facing **Path View**：题面只给出最终求解所需的可见条件，中间 dependency path 被折叠，解题者需要自行恢复关键推理步骤。

## Sample 1. FedAvg 收敛界中的学习率选择

**Subject**: Federated and Distributed Learning

**Question**

考虑如下设定下的 Federated Averaging（FedAvg）算法：

- 全局目标函数为 $F(w) = \sum_k \frac{n_k}{n} F_k(w)$，其中 $F_k(w)$ 是客户端 $k$ 的局部经验损失，$n_k = |D_k|$，且 $n = \sum_k n_k$。
- 每个被选中的客户端都从当前全局模型 $w^t$ 初始化，在其局部损失 $F_k$ 上以常数学习率 $\eta > 0$ 执行 $E > 1$ 步本地 SGD，然后把得到的参数增量发送给服务器，由服务器按权重平均聚合。
- 各客户端本地数据分布不同，因此局部梯度最小点与全局最小点不同。
- 加权 RMS 梯度异质性定义为

$$
G =
\left(
\sum_k \frac{n_k}{n}
\|\nabla F_k(w) - \nabla F(w)\|^2
\right)^{1/2}
\geq 0.
$$

- 全局损失满足 $L$-smoothness：

$$
F(w') \leq F(w) + \langle \nabla F(w), w' - w\rangle
+ \frac{L}{2}\|w' - w\|^2,
\quad \forall w,w'.
$$

- 一共进行 $T$ 轮全客户端参与的通信。量 $E$、$G$、$L$、$T$ 都是严格为正的常数，$w^\*$ 表示 $F$ 的一个全局最小点。

请确定能够使

$$
\frac{1}{T}\sum_{t=0}^{T-1}\|\nabla F(w^t)\|^2
$$

的最紧收敛上界达到最小的学习率，并计算对应的优化后上界。

请给出两个分别加框的答案：先写 $\eta^\*$，再写优化后的上界；每个答案都只能用 $F(w^0)-F(w^\*)$、$L$、$E$、$G$、$T$ 表示。

**Reference Answer**

$$
\boxed{
\eta^\* =
\sqrt{
\frac{2[F(w^0)-F(w^\*)]}
{L E^2 G^2 T}
}
}
$$

$$
\boxed{
\frac{1}{T}\sum_{t=0}^{T-1}\|\nabla F(w^t)\|^2
\leq
2\sqrt{
\frac{2 L G^2 [F(w^0)-F(w^\*)]}{T}
}
}
$$

## Sample 2. Pointer-Generator Coverage Loss 的梯度路径

**Subject**: Natural Language Processing / Abstractive Summarization

**Question**

在一个软注意力编码器-解码器模型中，在解码步 $t$ 处，注意力分数对每个源位置 $i$ 计算为

$$
e_{t,i} = \mathrm{score}(s_t, h_i),
$$

再经 softmax 归一化得到

$$
a_{t,i} =
\frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})},
$$

上下文向量定义为

$$
c_t = \sum_i a_{t,i}h_i.
$$

在 pointer-generator 模型中，每一步还计算生成概率

$$
p_{\mathrm{gen}} =
\sigma(w_c^\top c_t + w_s^\top s_t + w_x^\top x_t + b_{\mathrm{gen}}).
$$

组合训练损失为

$$
L_{\mathrm{total}} = L_{\mathrm{NLL}} + \lambda L_{\mathrm{cov}},
$$

其中

$$
\mathrm{cov}_{t,i} = \sum_{t' < t} a_{t',i},
\quad
\mathrm{cov}_{0,i}=0,
$$

且

$$
L_{\mathrm{cov}} =
\sum_t \sum_i \min(a_{t,i}, \mathrm{cov}_{t,i}).
$$

编码器隐藏状态 $h_i$ 是一个向量，$\lambda \geq 0$ 是固定标量，$\partial e_{t,i}/\partial h_i$ 表示标量 score 关于 $h_i$ 的向量值 Jacobian。由于 score 函数未固定，所以保持为未展开形式。

请推导 $\partial L_{\mathrm{total}} / \partial h_i$ 的单一闭式符号表达，并将答案写成一个 $\boxed{\cdots}$ 形式的单个求和公式。允许使用的符号包括 $a_{t,i}$、$a_{t,j}$、$\mathrm{cov}_{t,i}$、$\mathrm{cov}_{t,j}$、$a_{t'',i}$、$a_{t'',j}$、$\mathrm{cov}_{t'',i}$、$\mathrm{cov}_{t'',j}$、$g_{t,i}$、$g_{t,j}$、$\partial e_{t,i}/\partial h_i$、$\partial L_{\mathrm{NLL}}/\partial a_{t,j}$、$\partial L_{\mathrm{NLL}}/\partial c_t$、$t$、$t''$、$i$、$j$、$\lambda$ 和 $\mathbf{1}[\cdot]$，其中 $g_{t,j}$ 可作为 $\partial L_{\mathrm{total}}/\partial a_{t,j}$ 的简写。

**Reference Answer**

$$
\boxed{
\frac{\partial L_{\text{total}}}{\partial h_i}
=
\sum_t
\left[
a_{t,i}
\left(
g_{t,i}
-
\sum_j a_{t,j}g_{t,j}
\right)
\cdot
\frac{\partial e_{t,i}}{\partial h_i}
+
a_{t,i}
\cdot
\frac{\partial L_{\text{NLL}}}{\partial c_t}
\right]
}
$$

## Sample 3. TRADES 鲁棒训练目标的外层与内层条件

**Subject**: Machine Learning Security / Adversarial Robustness

**Question**

考虑一个参数化深度神经网络 $f_\theta$，其设定如下：可行扰动集合是闭 $L_p$ 球

$$
\{\delta : \|\delta\|_p \leq \varepsilon\},
$$

其中 $p \geq 1$ 且 $\varepsilon > 0$。$L(f_\theta(x+\delta), y)$ 是扰动输入 $x+\delta$ 相对于真实标签 $y$ 的标量损失。数据分布为 $(x,y)\sim\mu$。$\mathrm{KL}(f_\theta(x)\|f_\theta(x+\delta))$ 表示 KL 散度，其中干净输入分布 $f_\theta(x)$ 是第一参数，扰动输入分布 $f_\theta(x+\delta)$ 是第二参数。$\beta\geq0$ 是固定标量。$\Pi_{\|\cdot\|_p\leq\varepsilon}(v)$ 表示把 $v$ 投影到半径为 $\varepsilon$ 的闭 $L_p$ 球上的欧氏投影。$\alpha>0$ 是步长。$\lambda\geq0$ 是拉格朗日乘子。$\delta^\*(\theta)$ 表示对固定的 $(\theta,x)$，在约束 $\|\delta\|_p\leq\varepsilon$ 下使 $\mathrm{KL}(f_\theta(x)\|f_\theta(x+\delta))$ 达到最大值的解。

从如下目标出发：

$$
\min_{\theta}
\mathbb{E}_{(x,y)\sim\mu}
\left[
L(f_{\theta}(x), y)
+
\beta
\cdot
\max_{\|\delta\|_p\leq\varepsilon}
\mathrm{KL}
\left(
f_{\theta}(x)
\,\big\|\,
f_{\theta}(x+\delta)
\right)
\right],
$$

将以下内容合并推导为一个单独加框的表达式：

1. $\nabla_\theta\max_{\|\delta\|_p\leq\varepsilon}\mathrm{KL}(f_\theta(x)\|f_\theta(x+\delta))$ 的包络定理形式，以及由此得到的 $\theta^\*$ 处外层驻点条件，并要求该驻点条件显式写成恰好两个梯度项相加等于零；
2. 内层关于 $\delta$ 的最大化问题对应的归一化 PGD 更新规则，以及收敛时的固定点 KKT 条件，包括梯度驻点、互补松弛和可行性。

请把完整答案写成一个单独的 $\boxed{\cdots}$ 表达式，并且只使用 $\{f_{\theta}, L, x, y, \delta, \varepsilon, p, \lambda, \theta, \mu, \beta, \mathrm{KL}, \alpha\}$ 这些符号。

**Reference Answer**

$$
\boxed{
\begin{aligned}
&\nabla_{\theta}
\max_{\|\delta\|_p\leq\varepsilon}
\mathrm{KL}
\left(
f_{\theta}(x)
\big\|
f_{\theta}(x+\delta)
\right)
=
\nabla_{\theta}
\mathrm{KL}
\left(
f_{\theta}(x)
\big\|
f_{\theta}(x+\delta^\*(\theta))
\right)
\Big|_{\delta=\delta^\*(\theta)},\\
&\nabla_{\theta}
\mathbb{E}_{(x,y)\sim\mu}
\left[
L(f_{\theta^\*}(x),y)
\right]
+
\beta
\nabla_{\theta}
\mathbb{E}_{(x,y)\sim\mu}
\left[
\mathrm{KL}
\left(
f_{\theta^\*}(x)
\big\|
f_{\theta^\*}(x+\delta^\*(\theta^\*))
\right)
\right]
=0,\\
&\delta_{t+1}
=
\Pi_{\|\cdot\|_p\leq\varepsilon}
\left(
\delta_t
+
\alpha
\frac{
\nabla_{\delta}
\mathrm{KL}
\left(
f_{\theta}(x)
\big\|
f_{\theta}(x+\delta_t)
\right)
}{
\left\|
\nabla_{\delta}
\mathrm{KL}
\left(
f_{\theta}(x)
\big\|
f_{\theta}(x+\delta_t)
\right)
\right\|_p
}
\right),\\
&\nabla_{\delta}
\mathrm{KL}
\left(
f_{\theta}(x)
\big\|
f_{\theta}(x+\delta^\*)
\right)
=
\lambda
\nabla_{\delta}\|\delta^\*\|_p,\quad
\lambda\geq 0,\\
&\lambda(\|\delta^\*\|_p-\varepsilon)=0,
\quad
\|\delta^\*\|_p\leq\varepsilon.
\end{aligned}
}
$$
