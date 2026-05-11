# 脱敏实验摘要

[English](./experiments.md)

> 本页数字均为 preliminary / redacted，用于投简历场景下说明项目真实性和评测信号。这里不公开原始题目、source materials、prompts 或 solver outputs。

## 评测想回答什么

AgenQA 在当前阶段主要被评估为 benchmark-synthesis framework。核心问题包括：

1. 生成出的 benchmark 能否区分模型能力，而不是过于简单或所有模型都解不出？
2. Edge/Path design 是否提供了超越 final-answer accuracy 的诊断信息？
3. 在线 synthesis 是否能产生 fresh benchmark instances，同时保留质量控制信号？
4. 产物是否足够可复用，能支持 benchmark construction 和后续数据路由？

## Public Aggregate Signals

| Evidence type | Public summary | 支撑什么 |
| --- | --- | --- |
| Validation scale | 早期验证阶段包含 37 次 synthesis runs / 175 个 generated QA candidates | 系统不是 isolated demo |
| Batch run | 10-paper paper-seed batch；9 个任务成功，1 个 truncation failure | pipeline 可跨 source 运行，并显式记录失败 |
| Same-family gradient | Qwen-family scale checks 显示不同模型规模之间有能力差异 | benchmark 具有 discriminative signal |
| SOTA comparison | SOTA 组在 aggregate checks 中保持领先 | 任务不是随机噪声 |
| Edge/Path gap | Edge views 通常比 folded Path views 更容易局部求解 | 隐藏中间依赖会增加难度 |
| Depth-conditioned trend | 更长依赖路径会让弱 solver 更早 drop off | 难度可以结构化分析，而不只是看最终准确率 |

## Example Table Shape

最终论文会使用精确审计后的数值。这里展示的是 public reporting shape。

| Model group | Edge accuracy | Path accuracy | Edge-Path gap | Interpretation |
| --- | --- | --- | --- | --- |
| smaller same-family model | redacted | redacted | positive | local support helps, hidden path is harder |
| larger same-family model | redacted | redacted | positive but lower | stronger reconstruction ability |
| SOTA model group | redacted | redacted | positive | strongest aggregate performance |

## Quality Gates

Accepted chains 不是从 raw generated text 直接筛出来的，而是依赖 harness-produced signals：

- Edge solver correctness 与 well-posedness checks；
- Path solver distribution 与 difficulty signal；
- multi-strong consensus；
- answer-contract 与 world-contract diagnostics；
- dependency-certificate integrity；
- revision history 与 final diagnostic comments。

## 本页不公开什么

本页不公开：

- generated benchmark questions；
- source papers 或 source documents；
- raw solver responses；
- consensus JSON files；
- model API configuration；
- prompt snapshots。
