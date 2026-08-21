# AgenQA-PreRelease

**AgenQA 研究项目预览：agentic data synthesis for scientific reasoning QA benchmark construction。**

[Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf) · [完整代码运行](./docs/quickstart.zh.md) · [示例论文 PDF](./examples/papers/layered_thermal_transport_demo.pdf) · [机器可读结果](./results/README.zh.md) · [English](./README.en.md)

## 30 秒版

AgenQA 研究的问题是：如何合成真正有挑战性的 scientific reasoning QA 数据，同时让生成过程仍然可检查、可修复、可评测。

![AgenQA overview](./paper-preview/figures/figure1_chain_growth_path_fold.png)

- **核心方法**：先构造一条 step-verifiable **Chain-of-KQA**，再通过 **Path-Fold** 隐藏中间 facts，形成更难的 solver-facing **Path View**。
- **验证视角**：同一条 dependency chain 同时投影为局部 **Edge View** 和全局 **Path View**，分离 correctness control 与 difficulty amplification。
- **系统设计**：用 **Director--Operator--Evaluator loop** 组织 init / extend / revise / finish，让 generation、verification 和 repair 发生在同一显式 reasoning state 上。
- **阶段信号**：Path View 题目在 SOTA solver 评测中整体准确率为 **84.18%**，诊断子集准确率为 **51.77%**；在 Qwen-family 内，同一 benchmark 的 accuracy 与模型规模/理论能力呈正相关（4B 为 **48.00%**，32B 为 **66.86%**）；早期下游训练中，约 **2K** 条 AgenQA 数据让 Qwen3-4B-Instruct 在 AIM24 上从 **59.62** 提升到 **61.98**，同时 GPQA 系列基本持平。

**互补 worked example：** 下面的构造型三步数学例子进一步展示 Premises 与 Intermediate Memory 如何增长、错误 candidate answer 如何被局部 revise，以及同一 candidate prefix 如何进入 Step/Folded 双视图评测。它用于解释机制，不是 actual AgenQA run、gold case 或难度证据。

![AgenQA candidate-prefix control worked example](./paper-preview/figures/figure2_candidate_prefix_dual_view.png)

## 推荐阅读路径

**首选阅读：** [AgenQA Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf)

PDF 已整理为论文式展示文档，包含标题、摘要、Motivation and Core Idea、Contributions、AgenQA Framework、Experimental Showcase、Conclusion 和 References。它是这个 repo 最完整、最顺眼的阅读入口。

| 想快速看什么 | 入口 |
| --- | --- |
| 论文式项目说明 | [Paper Preview PDF](./paper-preview/agenqa_paper_preview.zh.pdf) |
| 核心实验表 | [docs/experiments.zh.md](./docs/experiments.zh.md) |
| 三道代表性样例题 | [docs/examples.pdf](./docs/examples.pdf) |
| 系统与方法架构 | [docs/architecture.zh.md](./docs/architecture.zh.md) |
| Prompt 角色边界 | [prompts/](./prompts/) |
| 完整 runtime 与一键运行 | [docs/quickstart.zh.md](./docs/quickstart.zh.md) |
| 可直接输入的示例论文 | [examples/papers/layered_thermal_transport_demo.pdf](./examples/papers/layered_thermal_transport_demo.pdf) |
| 源代码快照与公网适配差异 | [docs/source-provenance.md](./docs/source-provenance.md) |
| 聚合结果 CSV 与 manifest | [results/](./results/) |

## 核心贡献

1. **Edge/Path-grounded Chain-of-KQA formalism**：将 correctness--difficulty tension 转化为 step-to-global design principle：Edge View 支持 step-level correctness control，Path-Fold 在同一条 generated chain 上放大全局 difficulty。
2. **Scalable and extensible agentic synthesis harness**：将 QA generation 视为显式知识依赖上的受控 state-transition process；Director--Operator--Evaluator loop 通过 grounding、state persistence、dependency auditability、contract-based stabilization 和 evaluator-guided repair 等显式 controls 支持可扩展 synthesis，并保持 Operator layer modular。
3. **Pre-release benchmark and training-utility evidence**：当前 public preview 展示两类 evidence。Benchmark-construction signals 包括：SOTA solver 评测覆盖 `96` 次 synthesis runs / `547` 道 Path View questions，整体准确率为 **84.18%**，诊断子集准确率为 **51.77%**，说明 Path View 在强模型区间仍保留难度和区分区域；Qwen-family 梯度评测覆盖 `37` 次 runs / `175` 道 Path View questions，同一 benchmark 上 accuracy 从 **48.00%**（4B）上升到 **66.86%**（32B），可作为 benchmark 合理性的 scale-consistency sanity check。Early downstream training signal 显示：约 `2,000` 条 AgenQA 数据训练后的 Qwen3-4B-Instruct variants 在 AIM24、HMMT-FEB 和 SciBench 上出现正向迁移，GPQA / GPQA-Diamond 基本持平。

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

## 完整代码与运行

- `src/agenqa/` 包含完整的 domain / prompts / skills / nodes / graph / memory / evaluation / downstream runtime。
- `src/infra/` 包含实际使用的 HTTP inference client、PDF loader、prompt tracking、artifact I/O、playback 和可选 code verifier。
- `config/agent_openai.yaml` 提供只需 `OPENAI_API_KEY` 的 portable config；`agenqa-demo` 会直接运行完整 graph。
- `tests/` 包含原 runtime 回归测试与新增的公网 API/PDF portability 测试。
- `results/` 将当前三张公开实验表转为 CSV + manifest，不包含原始题目、逐题回复或私有 run 路径。

快速开始：

```bash
python -m pip install -e ".[test]"
export OPENAI_API_KEY="your-key"
agenqa-demo
```

完整命令、输出结构和自定义 PDF 用法见 [运行指南](./docs/quickstart.zh.md)。代码以 [Apache-2.0](./LICENSE) 许可证发布。

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
src/agenqa/      完整 AgenQA runtime
src/infra/       API / PDF / artifact / playback 基础设施
config/          可携式 API 配置
examples/papers/ 可直接运行的合成论文 PDF
tests/           runtime 回归与 portability 验证
results/         公开表格的 CSV 与 release manifest
```
