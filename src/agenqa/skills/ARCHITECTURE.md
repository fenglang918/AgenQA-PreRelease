# AgenQA Skills 架构设计

## 概述

`agenqa/skills` 模块是 AgenQA 系统的技能层，实现了题目生成链路中的各种 LLM 调用技能。每个技能封装了特定的任务逻辑，包括 prompt 构建、LLM 调用、输出解析等，提供了统一的接口和可复用的组件。

## 架构层次

```
┌─────────────────────────────────────────────────────────┐
│              Skills Layer (技能层)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Base Runner  │  │ Core Skills  │  │   Utilities  │ │
│  │  (基础类)     │  │  (核心技能)   │  │  (工具函数)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Infrastructure Layer                       │
│              (inference, prompt_builder, etc.)         │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 基础类 (`base.py`)

#### 1.1 BaseSkillRunner

**功能**：提供共享的并发支持和会话管理

```python
class BaseSkillRunner:
    def __init__(self, generator_config: Any):
        self._session_pool = ThreadLocalSessionPool(generator_config)

    def _concurrent_map(self, records, worker, max_workers):
        """并发执行 worker(record) 并流式返回结果"""
```

**特性**：
- 线程本地会话池（`ThreadLocalSessionPool`）
- 并发映射工具（`_concurrent_map`）
- 支持流式返回结果

### 2. 核心技能

所有技能遵循统一的模式：`Config` → `Input` → `Runner` → `Output`

#### 2.1 Draft 技能 (`drafting.py`)

**功能**：在 Known_0 + History 基础上构思下一题草稿

**组件**：
- `DraftConfig`: 配置（generator, prompt_path, lang, mode, protocol）
- `DraftInput`: 输入（known_0, history_brief, director_notes, prev_step, question_type）
- `DraftOutput`: 输出（draft_question, draft_solution, draft_answer, draft_background, reused_conclusions, grounding_check）
- `DraftRunner`: 执行器

**核心方法**：
- `_build_prompt(draft_in)`: 构建 prompt（使用 Template 替换）
- `_parse_output(text)`: 解析输出（支持 JSON 和 tagged 两种协议）
- `_parse_tagged(text)`: 解析带字段标记的纯文本格式
- `run_one(draft_in, snapshot_dir)`: 执行单次调用

**输出协议**：
- `json`: 标准 JSON 格式（默认）
- `tagged`: 带字段标记的纯文本格式（`[field]...[/field]`）

#### 2.2 Format 技能 (`formatting.py`)

**功能**：在 Draft 草稿基础上整理正式题目并自检

**组件**：
- `FormatConfig`: 配置
- `FormatInput`: 输入（draft_json, known_0, history_brief, prev_step, step, question_type）
- `FormatOutput`: 输出（step, question, solution, answer, new_background, validation_passed, validation_errors）
- `FormatRunner`: 执行器

**特性**：
- 接收 Draft 输出作为输入
- 进行格式化和验证
- 生成最终的 KQA 记录

#### 2.3 Solving 技能 (`solving.py`)

**功能**：对 KQA 记录进行求解（看不到标准答案）

**组件**：
- `SolverConfig`: 配置（prompt_path, prompt_text, generator, service_config_path, service_id）
- `SolverRunner`: 执行器（继承 `BaseSkillRunner`）

**核心功能**：
- 基于 Known + Question 求解
- 计算答题正确性（数值/文本判定）
- 估算题目难度（基于 tokens 比例）
- 支持符号表达式等价判断（可选）

**输出字段**：
- `Answer`: 答案（可能包含 `\boxed{}`）
- `Feedback`: 反馈
- `HarderSuggestion`: 如何变难的建议
- `KeyConclusion`: 关键结论（可选）

**并发支持**：
- 支持批量并发求解
- 支持增量写入和断点续跑

#### 2.4 Extract 技能 (`extracting.py`)

**功能**：从论文摘要中提炼考点（QA-Init 前置模块）

**组件**：
- `ExtractConfig`: 配置
- `ExtractInput`: 输入（paper_brief_json, paper_brief_text）
- `ExtractOutput`: 输出（exam_points, recommended_first_point, chain_potential）
- `ExtractRunner`: 执行器

**用途**：
- 分析论文内容
- 提取可出题的考点
- 推荐首个考点
- 评估链式潜力

#### 2.5 Diagnose 技能 (`diagnosing.py`)

**功能**：针对单道题做诊断分析（Revise 前置模块）

**组件**：
- `DiagnoseConfig`: 配置
- `DiagnoseInput`: 输入（known_0, question, answer, solver_feedback, director_notes, solver_answers, background）
- `DiagnoseOutput`: 输出（issues, fix_suggestions, diagnosis）
- `DiagnoseRunner`: 执行器

**用途**：
- 识别题目问题
- 提供修复建议
- 生成诊断总结

#### 2.6 Init（Known-Init，非 Skill）

**说明**：旧版 `qa_init` Skill 已移除；当前 init 逻辑由 `agenqa/nodes/op_known_init.py` 与 `roles_nodes.py` 负责，
在 `agent-run` 中完成：
- EpisodeSeedBuilder（单次 LLM，按 contract 输出 `episode_seed`）
- JSON Schema 校验（严格，失败可选重试）
- 写入 `memory.episode_seed`（至少包含 `anchor`，其余字段由 contract 决定）

#### 2.7 PaperBrief 技能 (`paper_brief.py`, legacy)

**状态**：已从主链路移除，仅保留给历史/可选 pipeline 使用。

#### 2.8 ExtendUpgrade 技能 (`extend_upgrade.py`)

**功能**：使用 extend-upgrade prompt 迭代扩展 K/Q/A 记录

**组件**：
- `ExtendUpgradeConfig`: 配置（generator, prompt_path, lang, normalize_known_tree, director_notes, question_type, max_background_step, symbolic_only）
- `ExtendUpgradeRunner`: 执行器（继承 `BaseSkillRunner`）

**特性**：
- 支持多种模式（extend, compress_history, plan_critique, reflect_fuse）
- 自动合并 Known 树
- 支持符号表达式 only 模式
- 支持并发执行

#### 2.9 HeadTail 组合器 (`head_tail.py`)

**功能**：从多步迭代链中抽取首尾，生成 head-tail KQA

**组件**：
- `HeadTailConfig`: 配置
- `HeadTailComposer`: 组合器

**用途**：
- 抽掉中间过程
- 输出仅包含头部已知（K_0）与尾部题问/答案（Q_n/A_n）
- 形成需要多步推理才能解答的题目（如 k0,q5,a5）

### 3. 工具函数 (`chain_utils.py`)

#### 3.1 KQANode 数据模型

```python
@dataclass
class KQANode:
    paper_id: str
    step: int
    known: str
    question: str
    answer: str
    subject: Optional[str] = None
    chain: Optional[str] = None
    prev_step: Optional[int] = None
    known_derivation: Optional[Dict[str, Any]] = None
    components: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    service_id: Optional[str] = None
    timestamp: Optional[str] = None
```

#### 3.2 核心函数

**read_nodes(path)**:
- 从 JSONL 或连续 JSON 对象读取节点
- 返回 `List[KQANode]`

**collect_chain(nodes)**:
- 按 `paper_id` 分组节点
- 按 `step` 升序排序
- 返回 `Dict[str, List[KQANode]]`

**compose_known(chain, step, separator)**:
- 组合 Known_i 为 K0 + concat(A_0..A_{i-1})
- 返回组合后的 Known 字符串

**verify_known_materialization(chain, node)**:
- 验证物化的 `node.known` 是否包含所有前序答案
- 使用空格不敏感的匹配
- 返回 `(bool, Dict[str, Any])`

**head_tail_view(known_dict, visible_fields)**:
- 生成 head-tail 可见性视图
- 根据字段可见性规则过滤

## 设计模式

### 1. 统一的技能接口

所有技能遵循相同的模式：

```python
# 1. 配置
@dataclass
class SkillConfig:
    generator: Dict[str, Any]
    prompt_path: Path
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    protocol: Optional[str] = None  # json / tagged

# 2. 输入
@dataclass
class SkillInput:
    # 字段根据技能而定
    ...

# 3. 输出
@dataclass
class SkillOutput:
    # 字段根据技能而定
    ...

# 4. 执行器
class SkillRunner:
    def __init__(self, config: SkillConfig):
        # 初始化会话、模板等
        ...

    def _build_prompt(self, input: SkillInput) -> str:
        # 构建 prompt
        ...

    def _parse_output(self, text: str) -> Optional[SkillOutput]:
        # 解析输出（支持 JSON 和 tagged）
        ...

    def run_one(self, input: SkillInput, snapshot_dir=None) -> Optional[SkillOutput]:
        # 执行单次调用
        ...
```

### 2. 输出协议支持

**JSON 协议**（默认）：
- 标准 JSON 格式
- 支持代码块提取（` ```json ... ``` `）
- 支持 thinking 片段过滤（`</think>`）

**Tagged 协议**：
- 带字段标记的纯文本格式
- 格式：`[field]...[/field]`
- 支持列表字段（`- item` 格式）

**解析策略**：
1. 优先尝试 JSON 解析
2. 失败则尝试 tagged 解析（如果启用）
3. 容错处理（缺失字段、类型转换）

### 3. 快照支持

所有技能支持快照保存（用于调试和排查）：

```python
snapshot_dir/
    ├── prompt.txt              # 构建的 prompt
    ├── response.json           # 原始响应（JSON）
    ├── response.txt            # 原始响应（文本，如果 JSON 失败）
    ├── response_text.txt       # 提取的文本
    └── parsed.json             # 解析后的输出
```

### 4. Prompt 管理

**模板系统**：
- 使用 Python `Template` 类进行变量替换
- 支持 `$variable` 和 `${variable}` 语法
- 使用 `safe_substitute` 允许未定义占位符

**Prompt 来源**：
1. 代码内嵌（`SCICLONE_USE_CODE_PROMPTS=1`）
2. 文件路径（`prompt_path`）
3. 直接文本（`prompt_text`）

**Prompt 片段注入**：
- 支持加载通用片段（如 `answer_schema`, `question_types`）
- 通过 `load_prompt_fragment()` 注入

### 5. 并发执行

**BaseSkillRunner 提供**：
- 线程本地会话池
- `_concurrent_map()` 方法
- 流式返回结果

**使用示例**：
```python
class SolverRunner(BaseSkillRunner):
    def _run_concurrent(self, kqa_path, output_path, append, concurrency):
        def worker(record):
            # 处理单条记录
            ...

        for result in self._concurrent_map(records, worker, concurrency):
            # 处理结果
            ...
```

## 使用模式

### 1. 单次调用

```python
from agenqa.skills.drafting import DraftConfig, DraftInput, DraftRunner

# 配置
config = DraftConfig(
    generator={"service_type": "private_endpoint", "service_id": "..."},
    prompt_path=Path("src/agenqa/prompts/draft.prompt"),
    lang="zh",
    mode="extend",
)

# 创建执行器
runner = DraftRunner(config)

# 准备输入
draft_in = DraftInput(
    known_0="...",
    history_brief="...",
    director_notes="...",
    prev_step=0,
    question_type="MCQ",
)

# 执行
output = runner.run_one(draft_in, snapshot_dir=Path("snapshots/"))
```

### 2. 批量处理

```python
from agenqa.skills.solving import SolverConfig, SolverRunner

# 配置
config = SolverConfig(
    prompt_path=Path("src/agenqa/prompts/solver.prompt"),
    service_id="remote:qwen3-30b-a3b-thinking",
)

# 创建执行器
runner = SolverRunner(config)

# 批量求解
output_path = runner.run(
    kqa_path=Path("input.jsonl"),
    output_path=Path("output.jsonl"),
    append=False,
    concurrency=4,  # 并发数
)
```

### 3. 链式调用

```python
# Draft → Format
from agenqa.skills.drafting import DraftRunner, DraftConfig, DraftInput
from agenqa.skills.formatting import FormatRunner, FormatConfig, FormatInput

draft_runner = DraftRunner(draft_config)
draft_out = draft_runner.run_one(draft_in)

format_runner = FormatRunner(format_config)
format_in = FormatInput(
    draft_json=json.dumps(draft_output_to_dict(draft_out)),
    known_0=known_0,
    history_brief=history_brief,
    prev_step=prev_step,
    step=step,
)
format_out = format_runner.run_one(format_in)
```

## 扩展指南

### 添加新技能

1. **创建技能文件**（如 `new_skill.py`）：
   ```python
   from agenqa.skills.base import BaseSkillRunner
   from dataclasses import dataclass

   @dataclass
   class NewSkillConfig:
       generator: Dict[str, Any]
       prompt_path: Path
       ...

   @dataclass
   class NewSkillInput:
       ...

   @dataclass
   class NewSkillOutput:
       ...

   class NewSkillRunner(BaseSkillRunner):
       def __init__(self, config: NewSkillConfig):
           super().__init__(config.generator)
           ...

       def _build_prompt(self, input: NewSkillInput) -> str:
           ...

       def _parse_output(self, text: str) -> Optional[NewSkillOutput]:
           ...

       def run_one(self, input: NewSkillInput, snapshot_dir=None):
           ...
   ```

2. **在 `__init__.py` 中导出**：
   ```python
   from .new_skill import NewSkillConfig, NewSkillInput, NewSkillOutput, NewSkillRunner
   ```

3. **创建对应的 Schema**（在 `domain/` 中）：
   - 定义字段常量
   - 实现 `*_output_schema_text()`
   - 实现 `*_output_to_dict()`

### 修改输出协议

1. 在 `_parse_output()` 中添加新协议支持
2. 实现对应的 `_parse_*()` 方法
3. 更新 `Config` 的 `protocol` 字段文档

### 添加并发支持

1. 继承 `BaseSkillRunner`
2. 实现 `_run_concurrent()` 方法
3. 在 `run()` 中根据 `concurrency` 参数选择执行路径

## 相关文件

- **领域模型**: `agenqa/domain/` (Schema 定义)
- **节点层**: `agenqa/nodes/` (使用 skills)
- **基础设施**: `infra/` (inference, prompt_builder, service_client)
- **Prompt 模板**: `src/agenqa/prompts/`
