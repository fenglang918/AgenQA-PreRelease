# AgenQA Graph 架构设计

## 概述

`agenqa/graph` 模块是 AgenQA 系统的图编排层，负责构建和执行基于 LangGraph 的工作流。该模块将节点层的各个组件组织成有向图，实现题目生成链路的自动化执行。

## 架构层次

```
┌─────────────────────────────────────────────────────────┐
│              Application Layer (调用层)                  │
│              (run_episode, 外部脚本)                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Graph Layer (图编排层)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Runner     │  │   Builder    │  │ State/Output │ │
│  │  (执行器)     │  │  (构建器)     │  │  (状态管理)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              LangGraph Framework                        │
│              (StateGraph, 状态流转)                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Nodes Layer (节点层)                        │
│              (director, operators, evaluators)          │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 状态管理 (`state.py`)

#### 1.1 数据模型

**KQARecord**：题目记录
```python
@dataclass
class KQARecord:
    paper_id: Optional[str]      # 论文ID
    step: Optional[int]          # 步骤索引
    known: str                   # 已知条件
    question: str                # 题目
    answer: str                  # 答案
    chain: Optional[str]         # 链式标识（如 "k0,q1,a1"）
    subject: Optional[str]       # 学科
```

**SolveMetrics**：求解指标
```python
@dataclass
class SolveMetrics:
    correct_medium: Optional[bool]      # 中等模型是否正确
    token_ratio_medium: Optional[float]  # 中等模型 token 比例（输出 tokens / 输入 tokens）
    # 语义主链路不再使用单一 strong 标量；改由
    # solver_metrics.edge/path.strong[] + metrics.edge/path.strong_summary 表达。
```

**Decision**：决策记录
```python
@dataclass
class Decision:
    operation: str               # 操作类型（Extend/Revise/Finish等）
    reason: str                  # 决策理由
    params: Dict[str, Any]       # 操作参数
```

**AgentState**：Agent 状态（核心状态对象）
```python
@dataclass
class AgentState:
    run_id: str                  # 运行ID
    artifacts_dir: Path          # 产物目录
    max_steps: int = 3           # 最大步数
    step: int = 0                # 当前步数
    rounds: int = 0              # 已执行轮数
    history: List[KQARecord]     # 历史记录
    last_decision: Optional[Decision]  # 上次决策
    metrics: SolveMetrics        # 求解指标
    stop_reason: Optional[str]   # 停止原因
    roles_cache: Dict[str, Any]  # 角色缓存（临时上下文）
```

#### 1.2 状态操作方法

**历史管理**：
- `append_history(rec)`: 追加历史记录，自动规范化 known 并更新 step
- `replace_history(records)`: 替换整个历史（用于 Compress-History）

**快照与持久化**：
- `save_snapshot()`: 保存状态到 `artifacts_dir/state.json`
- `to_json()`: 序列化为 JSON
- `load_from_file(path)`: 从文件恢复状态（支持断点续跑）
- `update_from_mapping(data)`: 从字典更新状态（兼容 LangGraph 返回）

**产物导出**：
- `dump_latest_kqa_jsonl(path)`: 导出最新 KQA 为 JSONL
- `dump_edge_kqa_for_step(step_dir)`: 导出指定步骤的 edge_kqa
- `dump_path_kqa_for_step(step_dir)`: 导出 path_kqa（Path-Fold）
- `dump_director_decision_for_step(step_dir)`: 导出 Director 决策

**辅助方法**：
- `current_round_index()`: 返回下一轮的索引（1-based）

### 2. 图构建器 (`builder.py`)

#### 2.1 操作路由映射

```python
OPERATION_ROUTES = {
    "init": "init",
    "extend": "extend",
    "revise": "revise",
    "compress_history": "compress_history",
    "finish": "finish",
}
```

将 Director 决策的操作类型映射到对应的图节点入口。

#### 2.2 图结构

**节点注册**：

1. **控制节点**：
   - `director`: 决策节点

2. **Known-Init 子链**（2个节点）：
   - `episode_seed_builder`: 单次 LLM + contract 生成 `episode_seed`
   - `seed_init`: 写入 `memory.episode_seed`

3. **Extend 子链**（2个节点）：
   - `extend_draft`: 扩展草稿
   - `extend_format`: 格式化

4. **Revise 子链**（3个节点）：
   - `revise_diagnose`: 诊断
   - `revise_draft`: 修订草稿
   - `revise_format`: 格式化

5. **评估节点**：
   - `solve_medium`: 中等模型求解（保留，供调试）
   - `solve_strong`: 强模型求解（保留，供调试）
   - `solve`: 并行求解（实际使用；medium + all strong）

6. **其他操作符**：
   - `compress_history`: 历史压缩

**边连接**：

```
director (起点)
    ↓ [条件路由]
    ├─→ episode_seed_builder → seed_init
    ├─→ extend_draft → extend_format
    ├─→ revise_diagnose → revise_draft → revise_format
    ├─→ compress_history
    └─→ END (finish)

所有操作符完成 → solve
    ↓ [条件路由]
    ├─→ consensus → judge → director (continue)
    └─→ consensus → judge → END (finish)
```

#### 2.3 节点包装

每个节点函数被包装为 LangGraph 兼容的签名：`fn(state: AgentState) -> AgentState`

**关键包装逻辑**：
- `_handle_result()`: 处理节点返回的 `NodeResult`，调用 `OutputManager` 保存产物
- 将 `NodeResult` 转换为 `AgentState` 返回

**Director 节点特殊处理**：
- 汇总求解上下文（`metrics`、`solver_metrics`、`solver_feedback`、Type1/Type2 摘要）并附加到决策 params
- 保存决策到 `state.last_decision`

#### 2.4 路由函数

**route_after_director**：
- 根据 `state.last_decision.operation` 查找 `OPERATION_ROUTES`
- 返回目标节点名称

**route_after_judge_node**：
- 由 `judge` 节点先写入 `stop_reason`，再由纯路由函数判断是否继续
- 返回 `"continue"` 或 `"finish"`

### 3. 执行器 (`runner.py`)

#### 3.1 run_episode 函数

**功能**：执行一个完整的 episode（题目生成链路）

**流程**：
1. 检查 LangGraph 依赖
2. 创建 `OutputManager`
3. 构建图（`build_graph`）
4. 初始化或恢复状态
5. 保存初始快照
6. 执行图（`graph.invoke`）
7. 处理返回结果（兼容 dict/AgentState）
8. 保存最终快照

**关键参数**：
- `agent_conf`: Agent 配置
- `run_id`: 运行ID
- `output_dir`: 输出目录
- `state`: 可选，用于断点续跑

**递归限制**：
- 根据 `max_steps` 和 `max_rounds` 动态计算 `recursion_limit`
- 默认公式：`max(64, base_limit * 6)`
- 避免 LangGraph 默认递归上限过早触发

**线程ID**：
- 从环境变量 `SCICLONE_THREAD_ID` 读取，或生成 UUID
- 用于 LangGraph 的线程隔离

### 4. 输出管理器 (`output_manager.py`)

#### 4.1 OutputContext

**功能**：封装一步的输出上下文，提供便捷的写入方法

**属性**：
- `node`: 节点名称
- `step_idx`: 步骤索引
- `round_idx`: 轮次索引
- `step_dir`: 步骤目录

**方法**：
- `path(name)`: 获取产物路径
- `dump_director_decision(state)`: 导出 Director 决策
- `dump_edge_kqa(state)`: 导出 edge_kqa
- `dump_path_kqa(state)`: 导出 path_kqa（Path-Fold）
- `save_role_output(role, payload)`: 保存角色输出到 `subruns/`

#### 4.2 OutputManager

**功能**：集中管理所有节点的输出与中间结果

**核心方法**：

1. **begin(node, step_idx, round_idx)**：
   - 创建步骤目录：`step_{step_idx}_round{round_idx}_{node}`
   - 返回 `OutputContext`

2. **save_result(node, result, specs, roles)**：
   - 根据 `NodeResult` 和 `OutputSpec` 自动落盘
   - 保存角色级中间结果到 `subruns/`
   - 保存声明式输出（edge_kqa、path_kqa、solve_*.jsonl 等）

**产物组织**：

```
step_{step_idx}_round{round_idx}_{node}/
    ├── director_decision.json      # Director 决策
    ├── edge_kqa.jsonl              # 最新 KQA
    ├── path_kqa.jsonl              # Path-Fold KQA（direct）
    ├── subruns/                    # 角色级中间结果
    │   ├── 01_episode_seed_builder.json
    │   ├── 02_seed_init.json
    │   ├── 03_draft.json
    │   └── 04_format.json
    └── solve/                      # 求解产物（如适用）
        ├── solve_medium.jsonl
        ├── solve_strong_0.jsonl
        ├── solve_path_medium.jsonl
        └── solve_path_strong_0.jsonl
```

**输出类型支持**：
- `json`: JSON 格式（带缩进）
- `jsonl`: JSON Lines 格式（每行一个 JSON 对象）
- 文件复制：如果 payload 是 Path 且存在，直接复制

### 5. 角色子图 (`roles_subgraph.py`)

**功能**：提供独立的角色链执行函数，供算子或测试直接调用

**函数**：
- `run_extend_graph()`: 执行 Extend 子链（Draft → Format）
- `run_revise_graph()`: 执行 Revise 子链（Diagnose → Draft → Format）
- `run_known_init_graph()`: 执行 Known-Init 子链（EpisodeSeedBuilder → SeedInit）

**返回值**：
- `(state, role_outputs, step_dir, step_idx, round_idx)`

**用途**：
- 算子内部调用（避免重复实现）
- 单元测试
- 独立调试

## 执行流程

### 完整 Episode 流程

```
1. 初始化
   ├─ 创建 OutputManager
   ├─ 构建图（build_graph）
   └─ 初始化/恢复 AgentState

2. 执行循环
   ├─ Director 决策
   │   ├─ 汇总状态（summarize_state）
   │   ├─ 调用 LLM 决策（decide_next）
   │   └─ 更新 last_decision
   │
   ├─ 路由到操作符
   │   ├─ Init: EpisodeSeedBuilder → SeedInit
   │   ├─ Extend: Draft → Format
   │   ├─ Revise: Diagnose → Draft → Format
   │   └─ CompressHistory: 压缩历史
   │
   ├─ 求解评估
   │   └─ SolveBoth: 并行中/强模型求解
   │
   └─ Judge 判断
       ├─ continue → 返回 Director（循环）
       └─ finish → 结束

3. 保存结果
   ├─ 保存最终状态快照
   └─ 所有产物已由 OutputManager 自动保存
```

### 状态流转

```
AgentState (初始)
    ↓
[Director] → 更新 last_decision
    ↓
[路由] → 根据 operation 选择子链
    ↓
[操作符子链] → 更新 history、step、roles_cache
    ↓
[Solve] → 更新 metrics (correct_medium, strong_summary, ...)
    ↓
[Judge] → 可能设置 stop_reason
    ↓
[路由] → continue/finish
    ↓
AgentState (更新后) → [循环或结束]
```

## 设计特点

### 1. 声明式输出

- 节点通过 `NodeResult` 声明输出
- `OutputManager` 根据 `OutputSpec` 自动落盘
- 避免节点内部散落文件操作逻辑

### 2. 状态持久化

- 每步自动保存快照（`state.json`）
- 支持断点续跑（`load_from_file`）
- 兼容 LangGraph 的 dict 返回

### 3. 产物组织

- 按步骤和轮次组织目录
- 角色级中间结果统一保存到 `subruns/`
- 求解产物统一保存到 `solve/`

### 4. 可扩展性

- 操作路由映射支持别名扩展
- 节点注册机制便于添加新节点
- 子图函数支持独立调用

### 5. 错误处理

- 输出失败不影响主流程
- 递归限制动态调整
- 状态更新容错处理

## 与 Nodes 层的关系

### 调用关系

```
Graph Layer (builder.py)
    ↓ 调用
Nodes Layer
    ├─ director.decide_next()
    ├─ operators.run_*()
    ├─ evaluators.solve_*()
    └─ roles_nodes.*_node()
```

### 数据流

```
AgentState (Graph)
    ↓ 传递
Nodes (读取/修改)
    ↓ 返回
AgentState (Graph) / NodeResult (Graph 转换为 AgentState)
```

### 产物流

```
Nodes (生成产物)
    ↓ 通过 NodeResult
OutputManager (自动保存)
    ↓ 写入
artifacts_dir/round_{r}/step_{step}_{op}/
```

## 扩展指南

### 添加新操作符节点

1. 在 `nodes/operators/` 实现操作符
2. 在 `OPERATION_ROUTES` 添加路由映射
3. 在 `builder.py` 注册节点和边

### 修改图结构

1. 修改 `build_graph()` 中的节点注册
2. 调整 `add_edge()` 和 `add_conditional_edges()`
3. 更新路由函数

### 自定义输出格式

1. 在 `OutputSpec` 中添加新类型
2. 在 `OutputManager._write_payload()` 中实现写入逻辑

### 断点续跑

1. 使用 `AgentState.load_from_file()` 恢复状态
2. 传入 `run_episode()` 的 `state` 参数
3. 确保 `artifacts_dir` 一致

## 相关文件

- **节点层**: `agenqa/nodes/` (director, operators, evaluators, roles_nodes)
- **领域模型**: `agenqa/domain/` (node_result, OutputSpec)
- **技能层**: `agenqa/skills/` (drafting, formatting, solving, etc.)
- **LangGraph**: 外部依赖，提供图执行框架
