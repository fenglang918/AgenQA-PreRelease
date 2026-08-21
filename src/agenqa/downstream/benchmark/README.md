# benchmark

本目录用于放 `AgenQA` **已采纳的 benchmark 集成层**。

这里的定位不是保存外部 benchmark 真源，而是保存：

- `AgenQA` 当前决定采纳哪些 benchmark
- 这些 benchmark 在本仓库内的统一接入代码
- 参数映射、结果解析、公共 schema 等真正被使用的实现

边界约定：

- 外部 benchmark 原仓与 vendored 代码：放 `external/`
- `AgenQA` 内部已采纳的 benchmark 集成代码：放这里
- 具体实验入口、配置、run 记录、结果汇总：放 `experiments/downstream/`

当前先保留为最小骨架目录；后续如开始正式接入，可逐步加入：

- benchmark registry
- adapter / runner wrapper
- result parser
- shared schema

一句话说：

`external/` 回答“benchmark 原来是什么”，这里回答“`AgenQA` 怎么真正使用它”。
