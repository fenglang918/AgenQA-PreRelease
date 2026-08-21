# AgenQA Nodes 架构设计

## 概述

`agenqa/nodes` 模块是 AgenQA 系统的核心节点层，负责实现题目生成链路中的各种操作节点。该模块采用分层架构设计，提供了统一的接口和可插拔的组件机制。

## 架构层次

```
┌─────────────────────────────────────────────────────────┐
│                    Graph Layer (LangGraph)              │
│              (builder.py, runner.py, state.py)          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Nodes Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Director   │  │    Judge     │  │  Evaluators  │ │
│  │  (决策节点)   │  │  (判断节点)   │  │  (评估节点)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Operators (操作符)                   │  │
│  │  Extend | Revise | Init | ReflectFuse | ...     │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Roles (角色节点)                     │  │
│  │  Draft | Format | Diagnose | Extract | ...      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Base Layer                           │
│              (base.py: Operator 接口)                   │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 基础接口层 (`base.py`)

定义了所有操作符的统一接口：

```python
class Operator:
    """最小化操作符接口"""
    name: str = "operator"              # 操作符名称
    outputs: ClassVar[List[OutputSpec]] = []  # 输出规格
    roles: ClassVar[List[str]] = []     # 使用的角色列表

    def run(self, agent_conf, state, **kwargs) -> AgentState | NodeResult:
        """执行操作并返回状态或结果"""
```

**设计原则**：
- 最小化接口，便于扩展
- 副作用仅作用于 `AgentState`（历史/指标/快照/产物）
- 返回状态实例以支持 LangGraph 状态传递

### 2. 控制节点

#### 2.1 Director 节点 (`director.py`)

**职责**：根据历史状态和指标，决策下一步操作（Extend/Revise/Finish/Init）

**核心函数**：
- `summarize_state(state)`: 汇总当前状态（历史、指标、求解反馈）
- `decide_next(agent_conf, state)`: 调用导演模型进行决策

**决策逻辑**：
- 首步（无历史）：强制选择 `Init`
- 后续步骤：根据配置允许的操作列表（默认：Extend/Revise/Finish）
- 防止过早终止：在未达到 `max_steps` 前，通常不允许直接 Finish
- 防止死循环：若上一轮是 Revise 且中/强均错，强制降级为 Extend

**输出**：`Decision` 对象，包含：
- `operation`: 操作类型（Extend/Revise/Finish/Init）
- `reason`: 决策理由
- `params`: 操作参数（如 `question_type`、`revise_mode`、`operator_notes`）

#### 2.2 Judge 节点 (`judge.py`)

**职责**：根据求解结果判断是否继续或终止

**核心函数**：
- `judge_node(agent_conf, state)`: 纯节点函数，负责写入 `state.stop_reason`
- `route_after_judge(state)`: 纯路由函数（只读 `stop_reason`），返回 `"continue"` 或 `"finish"`
- `judge_and_route(agent_conf, state)`: 兼容入口（Deprecated），内部组合调用上述两者

**判断策略**：
- 达到 `max_rounds` 或 `max_steps`：终止
- 中/强求解均错误：继续（交由 Director 决策）
- 中等正确但强错误：继续（交由 Director 决策）
- 其他情况：继续

**可选功能**：
- `enable_final_summary`: 在终止时触发 Director 总结

### 3. 操作符层 (`operators/`)

操作符是执行具体业务逻辑的组件，每个操作符实现 `Operator` 接口。

#### 3.1 ExtendOperator (`operators/extend.py`)

**功能**：扩展生成新题目

**流程**：
1. Draft 角色：生成题目草稿
2. Format 角色：格式化为标准 KQA 格式

**输出**：
- `edge_kqa.jsonl`: 最新题目（单条）
- `path_kqa.jsonl`: Path-Fold 题面（用于评估）

**实现**：`run_extend()` 函数（`op_extend.py`）

#### 3.2 ReviseOperator (`operators/revise.py`)

**功能**：修订已有题目

**流程**：
1. Diagnose 角色：诊断题目问题
2. Draft 角色：生成修订后的题目草稿
3. Format 角色：格式化为标准 KQA 格式

**修订模式**：
- `correctness`: 修正答案错误
- `difficulty`: 调整题目难度

**输出**：同 ExtendOperator

**实现**：`run_revise()` 函数（`op_revise.py`）

#### 3.3 Init（Known-Init）

**功能**：从论文初始化 episode_seed（不直接生成题目）

**流程**：
1. EpisodeSeedBuilder：单次 LLM + contract 输出 episode_seed
2. SeedInit：写入 `memory.episode_seed`

#### 3.4 其他操作符

- `ReflectFuseOperator`: 反思融合
- `PlanCritiqueOperator`: 计划批判
- `CompressHistoryOperator`: 历史压缩

#### 3.5 操作符注册表 (`operators/registry.py`)

```python
OP_REG: Dict[str, Operator] = {
    "extend": ExtendOperator(),
    "revise": ReviseOperator(),
    "reflect_fuse": ReflectFuseOperator(),
    "plan_critique": PlanCritiqueOperator(),
    "compress_history": CompressHistoryOperator(),
}
```

### 4. 角色节点层 (`roles_nodes.py`)

角色节点是细粒度的执行单元，每个操作符由多个角色节点组成。

#### 4.1 Known-Init 角色链（兼容旧名 QA-Init）

```
episode_seed_builder → seed_init
```

- `episode_seed_builder_node`: 从 paper text 直接按 contract 构造 episode_seed
- `known_init_seed_node`: 写入 `memory.episode_seed`（KnownTree）

#### 4.2 Extend 角色链

```
extend_draft → extend_format
```

- `extend_draft_node`: 基于历史生成新题目草稿
- `extend_format_node`: 格式化为 KQA

#### 4.3 Revise 角色链

```
revise_diagnose → revise_draft → revise_format
```

- `revise_diagnose_node`: 诊断当前题目问题
- `revise_draft_node`: 生成修订后的题目草稿
- `revise_format_node`: 格式化为 KQA

#### 4.4 角色基础功能 (`roles_base.py`)

提供共享的角色执行函数：

- `run_draft_base()`: 统一的 Draft 角色执行
- `run_format_base()`: 统一的 Format 角色执行

### 5. 评估节点层 (`evaluators/`)

#### 5.1 Solve 节点 (`evaluators/solve.py`)

**功能**：使用中/强模型对题目进行求解，更新指标

**核心函数**：
- `solve_medium()`: 使用中等模型求解
- `solve_strong()`: 使用强模型求解
- `solve_dual()`: 并行执行中/强求解

**评估模式**：

1. **数值模式**（默认）：
   - 直接使用 grow/solver 中的数值/文本判定逻辑
   - 更新 `AgentState.metrics.correct_{medium,strong}`

2. **符号表达式模式**（`agent.symbolic_only=True`）：
   - 额外调用轻量 LLM 作为表达式等价 judge
   - 对 `(answer, solve)` 进行符号等价性判断
   - judge 结果覆盖 `correct_{medium,strong}`

3. **Numeric 的可选工具能力（并入 medium/strong）**：
   - 当 `question_type=Numeric` 且未开启 symbolic-only 时，`medium/strong` solver 使用同一套 `solver_tool` 输出协议（可选输出 `ToolCode` 并由系统执行验证）。
   - tool 工件只写入 solve 侧 JSONL 行内用于诊断，不写入 premise_bank，也不泄露到 Path 视角。

**输出**：
- `solve_medium.jsonl`: 中等模型求解结果
- `solve_strong_0.jsonl`: strong[0] 的求解结果
- `solve_path_medium.jsonl`: path 题面的中等模型求解
- `solve_path_strong_0.jsonl`: path 题面的 strong[0] 求解
  - 当启用 multi-strong 时，会生成 `solve_strong_0/1/2/...jsonl` 与 `solve_path_strong_0/1/2/...jsonl`
  - 共识摘要会同时落盘为 `consensus_summary_edge.json` 与 `consensus_summary_path.json`；兼容旧路径的 `consensus_summary.json` 明确表示 edge 视角
  - Numeric 时：`solve_{medium,strong}.jsonl` 的行内会包含 `tool.*`（是否使用工具、代码与执行结果等）

### 6. 工具函数 (`utils.py`)

提供节点层共享的工具函数：

- `normalize_question_type()`: 标准化题目类型（MCQ/Derivation/Numeric）
- `select_question_type()`: 从 Director 决策中提取题目类型
- `is_symbolic_only()`: 判断是否启用符号表达式模式
- `build_director_notes()`: 构建 Director 注释（包含求解状态、反馈等）

## 执行流程

### 完整链路

```
Director (决策)
    ↓
[路由到操作符]
    ↓
Init: EpisodeSeedBuilder → SeedInit
Extend: Draft → Format
Revise: Diagnose → Draft → Format
    ↓
Solve (并行：中/强模型)
    ↓
Judge (判断)
    ↓
[继续] → Director (循环)
[终止] → END
```

### 状态流转

1. **Director 决策**：
   - 汇总历史状态（`summarize_state`）
   - 调用 LLM 进行决策（`decide_next`）
   - 更新 `state.last_decision`

2. **操作符执行**：
   - 根据 `last_decision.operation` 路由到对应操作符
   - 执行角色链（Draft/Format/Diagnose）
   - 更新 `state.history`（添加 KQA 记录）
   - 保存产物到 `state.artifacts_dir`

3. **求解评估**：
   - 并行调用中/强模型求解
   - 更新 `state.metrics`（correct/token_ratio）
   - 保存求解结果到 artifacts

4. **判断路由**：
   - 根据指标判断是否继续
   - 返回 `"continue"` 或 `"finish"`

## 设计特点

### 1. 分层解耦

- **控制层**（Director/Judge）：决策与路由
- **操作层**（Operators）：业务逻辑封装
- **角色层**（Roles）：细粒度执行单元
- **评估层**（Evaluators）：质量评估

### 2. 可插拔性

- 统一的 `Operator` 接口
- 操作符注册表机制
- 角色节点可独立替换

### 3. 状态管理

- 所有状态集中在 `AgentState`
- 副作用仅作用于状态对象
- 支持快照与回放

### 4. 产物管理

- 统一的 `artifacts_dir` 结构
- 标准化的 JSONL 输出格式
- 支持 `OutputManager` 管理输出

### 5. 错误处理

- 操作符级别的重试机制
- 决策失败时的回退策略
- 防止死循环的保护机制

## 扩展指南

### 添加新操作符

1. 在 `operators/` 目录创建新文件
2. 实现 `Operator` 接口
3. 在 `operators/registry.py` 中注册
4. 在 `graph/builder.py` 中添加路由

### 添加新角色

1. 在 `roles_nodes.py` 中实现角色节点函数
2. 在 `roles_base.py` 中添加共享逻辑（如需要）
3. 在操作符中调用角色节点

### 修改决策逻辑

1. 修改 `director.py` 中的 `decide_next()` 函数
2. 调整 `summarize_state()` 以提供更多上下文
3. 更新 prompt 模板（`src/agenqa/prompts/director.prompt`）

## 相关文件

- **图构建**: `agenqa/graph/builder.py`
- **状态定义**: `agenqa/graph/state.py`
- **技能层**: `agenqa/skills/` (drafting, formatting, solving, etc.)
- **领域模型**: `agenqa/domain/` (node_result, role_context, etc.)
- **Prompt 模板**: `src/agenqa/prompts/`
