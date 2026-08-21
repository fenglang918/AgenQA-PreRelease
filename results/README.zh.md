# 聚合结果快照

[English](./README.md)

本目录将 [`docs/experiments.zh.md`](../docs/experiments.zh.md) 中的三组公开表格转为小体积、可机器读取的 CSV 快照。

| 文件 | 内容 |
| --- | --- |
| `path_view_strong_solvers.csv` | 强 solver 在全部 Path View 评测和诊断子集上的计数与准确率 |
| `qwen_scale_consistency.csv` | Qwen-family 按模型规模聚合的准确率 |
| `training_transfer_2k.csv` | baseline 与两个训练 variant 的早期下游迁移分数 |
| `manifest.json` | 每组 snapshot 的范围、粒度与公开边界 |

这些文件是聚合预览，不是 raw experiment dump。它们刻意不包含生成题目、源论文、逐题判定、模型回复、服务元数据和私有 run 路径。强 solver 表中的分母是 evaluation attempts，不等于唯一题目数；`manifest.json` 单独记录了该 snapshot 中的 `547` 道唯一 Path View questions。

训练表只是早期信号，尚不包含方差、置信区间或 paired test，不应解读为最终统计结论。
