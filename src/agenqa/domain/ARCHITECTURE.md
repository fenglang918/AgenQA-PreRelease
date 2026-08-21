# AgenQA Domain 架构设计

## 概述

`agenqa/domain` 模块是 AgenQA 系统的领域模型层，定义了核心数据结构、Schema 规范和领域逻辑。该模块提供统一的数据模型和工具函数，确保整个系统对领域概念的理解一致。

## 架构层次

```
┌─────────────────────────────────────────────────────────┐
│              Domain Layer (领域模型层)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Known Tree  │  │   Schemas    │  │   Contexts   │ │
│  │  (知识树)     │  │  (输出规范)   │  │  (角色上下文) │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Node Result  │  │ Chain Utils  │  │   KQA Model  │ │
│  │  (节点结果)   │  │  (链式工具)   │  │  (题目模型)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Application Layers                         │
│              (nodes, graph, skills)                     │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Known Tree 管理 (`known_tree.py`, `known_utils.py`)

#### 1.1 Known 结构定义

Known 是系统统一维护的"知识树"对象，用于管理题目生成链路中的上下文信息。

**核心字段**：

- `known_0`: 整条链共享的初始背景，仅由首轮 QA-Init 决定，不在后续步骤重写
- `history`: 按 step 顺序累积的问答快照（`question_i` / `answer_i`），供 full-history 场景回顾与推理
- `background`: 各步通过 "NewBackground" 引入并由系统合并的新增设定或假设，可在后续题干中作为"已知条件"引用
- `derived_facts`: 可选的衍生结论聚合区，用于记录模型在上一轮推理得到的可复用事实

**字段可见性约定**（仅限解题侧可见性，出题/Director 仍可读全量 Known）：

- `known_0`: `visible_to_path = true`（path 解题时始终可见的基础背景）
- `background`: `visible_to_path = true`（由 NewBackground 累积的新增设定）
- `history`: `visible_to_path = false`（path 场景下不可直接暴露）
- `derived_facts`: `visible_to_path = false`（仅供内部推理与出题参考）

#### 1.2 KnownTree 类

**核心方法**：

1. **parse(val, max_depth=3)**:
   - 解析 Known 字符串/对象为字典
   - 支持嵌套字符串的递归解析（最多 max_depth 层）

2. **normalize(val, max_depth=3)**:
   - 规范化 Known 为干净的 JSON 字符串
   - 确保输出格式一致

3. **merge(prev_known, prev_step, prev_q, prev_a, llm_known, llm_new_background, llm_derived_facts, ...)**:
   - 合并当前步骤的信号到 Known 树
   - 合并规则：
     - `known_0` 保持首次值，不被后续覆盖
     - `history` 仅追加上一轮的 `question_i/answer_i`
     - `background` / `derived_facts` 以列表形式追加
     - 任何非列表的 background/derived_facts 会被标准化为空列表

4. **merge_background(...)**:
   - 仅合并新背景的便捷方法

5. **history_brief(val)**:
   - 渲染轻量级历史摘要，用于 prompts
   - 格式：`[Step i] Q: ...\nA: ...`

6. **to_json(tree)**:
   - 将 Known 树序列化为 JSON 字符串

#### 1.3 Known Utils (`known_utils.py`)

**核心函数**：

- `normalize_known(val, max_depth=3)`: 规范化 Known 为 JSON 字符串
- `parse_known_to_dict(val, max_depth=3)`: 解析 Known 为字典

**解析策略**：
- 优先尝试 JSON 解析
- 失败则尝试 `ast.literal_eval`
- 支持嵌套字符串的递归解析（最多 max_depth 层）

### 2. Schema 定义模块

Schema 模块定义了各角色输出的字段规范，采用"单一来源"（Single Source of Truth）原则，确保字段名在整个系统中保持一致。

#### 2.1 Draft Schema (`draft_schema.py`)

**字段定义**：

```python
FIELD_DRAFT_QUESTION = "draft_question"          # 题目草稿
FIELD_DRAFT_SOLUTION = "draft_solution"          # 解答草稿
FIELD_DRAFT_ANSWER = "draft_answer"              # 答案草稿
FIELD_DRAFT_BACKGROUND = "draft_background"      # Draft 新增设定候选（列表；后续会筛选为 NewBackground）
FIELD_REUSED_CONCLUSIONS = "reused_conclusions"  # 复用结论（列表）
FIELD_GROUNDING_CHECK = "grounding_check"        # 基础检查
```

**核心函数**：
- `draft_output_schema_text()`: 生成 Draft 输出格式的 prompt 描述
- `draft_output_to_dict(draft_out)`: 将 `DraftOutput` 转换为 dict

#### 2.2 Format Schema (`format_schema.py`)

**字段定义**：

```python
FIELD_STEP = "Step"                              # 步骤索引
FIELD_QUESTION = "Question"                      # 题目
FIELD_SOLUTION = "Solution"                      # 解答
FIELD_ANSWER = "Answer"                          # 答案
FIELD_NEW_BACKGROUND = "NewBackground"           # 新背景（字符串或列表）
FIELD_VALIDATION_PASSED = "validation_passed"    # 验证是否通过
FIELD_VALIDATION_ERRORS = "validation_errors"    # 验证错误列表
```

**核心函数**：
- `format_output_schema_text()`: 生成 Format 输出格式的 prompt 描述
- `format_output_to_dict(out)`: 将 `FormatOutput` 转换为 dict

#### 2.3 Extract Schema (`extract_schema.py`)

**字段定义**：

```python
FIELD_EXAM_POINTS = "exam_points"                    # 考点列表
FIELD_RECOMMENDED_FIRST_POINT = "recommended_first_point"  # 推荐的首个考点
FIELD_CHAIN_POTENTIAL = "chain_potential"            # 链式潜力
```

**核心函数**：
- `extract_output_schema_text()`: 生成 Extract 输出格式的 prompt 描述
- `extract_output_to_dict(out)`: 将 `ExtractOutput` 转换为 dict

#### 2.4 Diagnose Schema (`diagnose_schema.py`)

**字段定义**：

```python
FIELD_ISSUES = "issues"                    # 问题列表
FIELD_FIX_SUGGESTIONS = "fix_suggestions"  # 修复建议列表
FIELD_DIAGNOSIS = "diagnosis"              # 诊断总结
```

**核心函数**：
- `diagnose_output_schema_text()`: 生成 Diagnose 输出格式的 prompt 描述
- `diagnose_output_to_dict(out)`: 将 `DiagnoseOutput` 转换为 dict

#### 2.5 Solver Schema (`solver_schema.py`)

**字段定义**：

```python
FIELD_ANSWER = "Answer"                    # 答案（可能包含 \\boxed{}）
FIELD_FEEDBACK = "Feedback"                # 反馈
FIELD_HARDER_SUGGESTION = "HarderSuggestion"  # 如何变难的建议
FIELD_KEY_CONCLUSION = "KeyConclusion"     # 关键结论（可选）
```

**核心函数**：
- `solver_output_schema_text()`: 生成 Solver 输出格式的 prompt 描述
- `solver_output_base()`: 返回带空字段的基础 dict，用于初始化/合并

#### 2.5.1 SolverTool Schema (`solver_tool_schema.py`)

用于 **tool-enabled solver**（通常是 Numeric 的难度梯度 tier）：允许 solver 输出一段可执行代码作为内部可验证工件。

**字段定义**：

```python
ToolUsed   # 是否使用工具/代码
ToolName   # 工具名（统一为 python_executor）
ToolCode   # Python 代码（仅 stdlib，禁止网络/文件 IO；stdout 输出一行 {"value": <number>}）
ToolNotes  # 简短说明（非 CoT，可为空）
```

**核心函数**：
- `solver_tool_output_schema_text()`: 生成 SolverTool 输出字段的 prompt 描述

#### 2.6 QA-Init Schema (`qa_init_schema.py`)

**字段定义**：

```python
FIELD_STEP = "Step"                        # 步骤索引（=0）
FIELD_SUBJECT = "Subject"                  # 学科
FIELD_KNOWN = "Known"                      # Known 对象
FIELD_QUESTION = "Question"                # 题目
FIELD_SOLUTION = "Solution"                # 解答
FIELD_ANSWER = "Answer"                    # 答案
```

**Known 子字段**：
- `KNOWN_FIELD_KNOWN0 = "known_0"`: 初始背景
- `KNOWN_FIELD_HISTORY = "history"`: 历史记录

**核心函数**：
- `qa_init_output_schema_text()`: 生成 QA-Init 输出格式的 prompt 描述

### 3. 节点结果容器 (`node_result.py`)

#### 3.1 OutputSpec

**功能**：声明式输出规格，用于定义节点的输出要求

```python
@dataclass
class OutputSpec:
    name: str                              # 输出名称
    type: Literal["json", "jsonl", "txt"] # 输出类型
    required: bool = True                  # 是否必需
    is_intermediate: bool = False          # 是否为中间产物
```

**用途**：
- 节点声明其输出规格
- `OutputManager` 根据规格自动保存产物

#### 3.2 NodeResult

**功能**：标准化的节点返回载荷

```python
@dataclass
class NodeResult:
    state: "AgentState"                    # 更新后的状态
    step_idx: int | None = None            # 步骤索引
    round_idx: int | None = None           # 轮次索引
    outputs: Dict[str, Any] = {}           # 声明式输出（按 OutputSpec）
    role_outputs: Dict[str, Any] = {}      # 角色级中间结果
    artifacts: Dict[str, Any] = {}         # 其他产物
    step_dir: Any | None = None            # 步骤目录（通常为 Path）
```

**用途**：
- 节点返回统一的结果格式
- `OutputManager` 根据 `outputs` 和 `role_outputs` 自动保存
- 避免节点内部散落文件操作逻辑

### 4. 角色上下文 (`role_context.py`)

#### 4.1 DraftContext

**功能**：Draft 角色的共享上下文

```python
@dataclass
class DraftContext:
    mode: str                              # qa_init / extend / revise
    prev_step: int                         # 上一步索引
    target_step: int                       # 目标步骤索引
    known_0: str                           # 初始背景
    history_brief: str                     # 历史摘要
    director_notes: Optional[str] = None   # Director 注释
    question_type: Optional[str] = None    # 题目类型（MCQ/Derivation/Numeric）
    diagnose_summary: Optional[str] = None # 诊断摘要（仅 revise 模式）
```

#### 4.2 FormatContext

**功能**：Format 角色的共享上下文

```python
@dataclass
class FormatContext:
    mode: str                              # qa_init / extend / revise
    prev_step: int                         # 上一步索引
    step: int                              # 当前步骤索引
    known_0: str                           # 初始背景
    history_brief: str                     # 历史摘要
    question_type: Optional[str] = None    # 题目类型（MCQ/Derivation/Numeric）
```

**用途**：
- 统一角色节点的输入参数
- 便于在 `roles_base.py` 中实现共享逻辑

### 5. 链式工具 (`chain.py`)

**功能**：链式相关的工具函数（从 `skills/chain_utils` 重新导出）

**核心函数**：
- `collect_chain()`: 收集链式信息
- `compose_known()`: 组合 Known
- `head_tail_view()`: 生成 head-tail 视图
- `read_nodes()`: 读取节点
- `verify_known_materialization()`: 验证 Known 物化

**数据模型**：
- `KQANode`: KQA 节点模型（从 `skills/chain_utils` 导入）

### 6. KQA 模型 (`kqa.py`)

**功能**：KQA 相关的领域对象重新导出

**导出内容**：
- `KQARecord`: 从 `graph.state` 导入
- `KQANode`: 从 `skills/chain_utils` 导入

## 设计原则

### 1. 单一来源（Single Source of Truth）

- Schema 模块定义字段名常量
- 所有解析、输出、prompt 生成都引用这些常量
- 避免字段名散落造成不一致

### 2. 声明式输出

- 节点通过 `OutputSpec` 声明输出
- `NodeResult` 统一返回格式
- `OutputManager` 自动处理保存

### 3. 领域模型隔离

- Domain 模块独立于其他层
- 避免循环依赖（使用 TYPE_CHECKING 和延迟导入）
- 提供清晰的领域概念抽象

### 4. 容错处理

- Known 解析支持多种格式（JSON、Python 字面量、嵌套字符串）
- 合并操作容错处理（缺失字段、类型不匹配）
- 规范化操作保证输出一致性

### 5. 可见性控制

- 明确定义字段的可见性规则
- 支持 head-tail 场景的可见性过滤
- 便于实现不同场景下的上下文管理

## 使用模式

### 1. Known Tree 使用

```python
from agenqa.domain import KnownTree

# 解析 Known
tree = KnownTree.parse(known_str)

# 合并新步骤
merged = KnownTree.merge(
    prev_known=prev_known,
    prev_step=prev_step,
    prev_q=prev_q,
    prev_a=prev_a,
    llm_known=llm_known,
    llm_new_background=new_bg,
    llm_derived_facts=derived,
)

# 规范化输出
normalized = KnownTree.normalize(merged)

# 生成历史摘要
brief = KnownTree.history_brief(merged)
```

### 2. Schema 使用

```python
from agenqa.domain.draft_schema import (
    FIELD_DRAFT_QUESTION,
    draft_output_to_dict,
    draft_output_schema_text,
)

# 在 prompt 中使用
prompt = f"输出格式：\n{draft_output_schema_text()}"

# 解析 JSON
question = json_data[FIELD_DRAFT_QUESTION]

# 转换为 dict
output_dict = draft_output_to_dict(draft_output)
```

### 3. NodeResult 使用

```python
from agenqa.domain import NodeResult, OutputSpec

# 节点返回结果
result = NodeResult(
    state=updated_state,
    step_idx=step_idx,
    round_idx=round_idx,
    outputs={
        "edge_kqa": kqa_dict,
        "path_kqa": path_dict,
    },
    role_outputs={
        "draft": draft_dict,
        "format": format_dict,
    },
    step_dir=step_dir,
)

# 声明输出规格
specs = [
    OutputSpec("edge_kqa", "jsonl", required=False),
    OutputSpec("path_kqa", "jsonl", required=False),
]
```

### 4. 角色上下文使用

```python
from agenqa.domain.role_context import DraftContext

# 构建上下文
ctx = DraftContext(
    mode="extend",
    prev_step=prev_step,
    target_step=target_step,
    known_0=known_0,
    history_brief=history_brief,
    director_notes=director_notes,
    question_type="MCQ",
)

# 传递给角色函数
output = run_draft_base(ctx, generator=generator, ...)
```

## 扩展指南

### 添加新 Schema

1. 创建新的 schema 文件（如 `new_role_schema.py`）
2. 定义字段常量（`FIELD_*`）
3. 实现 `*_output_schema_text()` 函数
4. 实现 `*_output_to_dict()` 函数（如适用）
5. 在相关 prompt 和解析代码中使用这些常量

### 扩展 Known Tree

1. 在 `KNOWN_TREE_DESCRIPTION` 中更新文档
2. 在 `KNOWN_TREE_FIELD_VISIBILITY` 中添加可见性规则
3. 在 `KnownTree.merge()` 中添加合并逻辑
4. 更新 `_build_base()` 以支持新字段

### 添加新上下文类型

1. 在 `role_context.py` 中定义新的 `@dataclass`
2. 在相关角色函数中使用该上下文
3. 确保字段与角色需求一致

## 相关文件

- **节点层**: `agenqa/nodes/` (使用 domain 模型)
- **图层**: `agenqa/graph/` (使用 domain 模型)
- **技能层**: `agenqa/skills/` (定义技能输出，domain 定义 Schema)
- **状态管理**: `agenqa/graph/state.py` (KQARecord, AgentState)
