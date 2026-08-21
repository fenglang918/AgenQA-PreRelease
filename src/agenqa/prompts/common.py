"""Common sections shared by multiple agent roles."""

from textwrap import dedent, indent
from ._base import PromptSection
from agenqa.domain.known_tree import KNOWN_TREE_DESCRIPTION, KNOWN_TREE_DESCRIPTION_EN
from agenqa.domain.contracts.world_contract import (
    WORLD_CONTRACT_MODEL_TEXT_EN,
    WORLD_CONTRACT_MODEL_TEXT_ZH,
    WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_EN,
    WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_ZH,
)
from agenqa.domain.contracts.answer_contract_bank import (
    ANSWER_CONTRACT_MODEL_TEXT_EN,
    ANSWER_CONTRACT_MODEL_TEXT_ZH,
)
from agenqa.domain.metrics_schema import (
    METRICS_AND_SOLVERS_DESCRIPTION_ZH,
    METRICS_AND_SOLVERS_DESCRIPTION_EN,
    EDGE_QA_VS_PATH_DESCRIPTION_ZH,
    EDGE_QA_VS_PATH_DESCRIPTION_EN,
)
from agenqa.domain.consensus_schema import SOLVER_CONSENSUS_DESCRIPTION_ZH, SOLVER_CONSENSUS_DESCRIPTION_EN
from agenqa.domain.executable_schema import (
    executable_extract_output_schema_text,
    executable_draft_step_output_schema_text,
    executable_test_inputs_schema_text,
)
from agenqa.domain.draft_schema import (
    FIELD_DRAFT_QUESTION_EXPLICIT,
    FIELD_DRAFT_QUESTION,
    FIELD_DRAFT_SOLUTION,
    FIELD_DRAFT_ANSWER,
    FIELD_DRAFT_BACKGROUND,
    FIELD_REUSED_CONCLUSIONS,
    FIELD_REUSED_REFS,
    FIELD_GROUNDING_CHECK,
)

# 通用答案格式规范
COMMON_ANSWER_SCHEMA = PromptSection(
    text=dedent(
        """\
        - 所有题型：最终答案只给结论本身，用 LaTeX `\\boxed{…}` 包裹，不附加额外句子或解释。
        - `MCQ`：`Answer` 中仅写唯一正确选项的字母（如 `\\boxed{A}`），最终 solver-visible 交付必须明确“仅给出选项字母”。
        - `Derivation`：给出唯一可判定的解析表达式/符号结论（如 `\\boxed{2k}` 或 `\\boxed{R(\\gamma)=2/\\gamma}`）；最终 solver-visible 交付必须明确允许的符号集合（例如 `in terms of ...`），不得引入新符号。
        - `Numeric`：需要"建模 + 代码执行"才能可靠求解的定量计算题；最终 solver-visible 交付（`Question + World Contract`）必须显式声明误差/精度/单位口径（例如绝对误差/相对误差阈值，或"保留 N 位有效数字"）。
        """
    )
)

# Common answer format rules (EN)
COMMON_ANSWER_SCHEMA_EN = PromptSection(
    text=dedent(
        """\
        - For all question types: output **only** the final result, wrapped in LaTeX `\\boxed{...}`. Do not add extra sentences or explanations.
        - `MCQ`: the `Answer` must be the **single correct option letter only** (e.g., `\\boxed{A}`), and the final solver-visible deliverable must say “answer with the option letter only”.
        - `Derivation`: output a uniquely checkable symbolic/expression answer (e.g., `\\boxed{2k}` or `\\boxed{R(\\gamma)=2/\\gamma}`); the final solver-visible deliverable must explicitly constrain the allowed symbols (e.g., via an `in terms of ...` list) and must not introduce new symbols.
        - `Numeric`: requires "modeling + code execution" to reliably solve; the final solver-visible deliverable (`Question + World Contract`) must explicitly state the tolerance/precision and unit conventions (e.g., absolute/relative error thresholds or "N significant figures").
        """
    )
)

# 通用题型定义
COMMON_QUESTION_TYPES = PromptSection(
    text=dedent(
        """\
        - `QuestionType=MCQ`：单选题，题干需明确“仅给出选项字母”；至少 4 个互斥选项（格式如“选项：A. …；B. …；C. …；D. …”），保证有且仅有一个严格正确；若涉及数值，需在题干或选项说明近似口径。
        - `QuestionType=Derivation`：推导/符号推理题，强调多步推演得到唯一可判定的解析表达式；最终 solver-visible 交付需给出必要条件与近似，并明确允许出现的符号集合（不要引入新符号）。可以带选项也可以开放式，但不要把 Derivation 强行做成缺乏推理的选择题。
        - `QuestionType=Numeric`：需要"建模 + 代码执行"才能可靠求解的定量计算题。需整合多个科学/数学概念建立模型，或需要数值方法（积分/优化/迭代/采样）求解；最终 solver-visible 交付必须给出必要条件、单位与近似口径，并**显式声明容差/精度规则**（绝对/相对误差或有效数字；通常由 Numeric-oracle 工具链生成并注入到独立 contract 文本）。避免简单套公式代数值。
        """
    )
)

# Common question type definitions (EN)
COMMON_QUESTION_TYPES_EN = PromptSection(
    text=dedent(
        """\
        - `QuestionType=MCQ`: single-choice multiple choice. The question must explicitly say “answer with the option letter only”; provide at least 4 mutually exclusive options (format like “Options: A. ...; B. ...; C. ...; D. ...”), with **exactly one** strictly correct. If numeric, state approximation/rounding conventions in the question/options.
        - `QuestionType=Derivation`: derivation/symbolic reasoning problem. Emphasize multi-step reasoning to reach a uniquely checkable symbolic expression; the final solver-visible deliverable must provide necessary conditions/approximations and explicitly constrain the allowed symbols (do not introduce new symbols). It may include options, but do not turn Derivation into a trivial MCQ with no reasoning.
        - `QuestionType=Numeric`: requires "modeling + code execution" to reliably solve. Should integrate multiple science/math concepts to build a model, or require numerical methods (integration/optimization/iteration/sampling); the final solver-visible deliverable must provide necessary conditions, units, and approximation conventions, and must **explicitly state** tolerance/precision rules (absolute/relative error or significant figures; typically injected by the Numeric-oracle toolchain into separate contract text). Avoid simple plug-and-chug.
        """
    )
)

# 给 Director 的“题型能力”说明：强调“能实现什么”，而不是把生成/格式细则塞给 Director。
COMMON_QUESTION_TYPE_CAPABILITIES = PromptSection(
    text=dedent(
        """\
        ## 题型能力（给 Director 的最小充分语义）

        选择 QuestionType 本质上是在选择“下一题的出题能力/工具链”：

        - `MCQ`：概念辨析/口径对齐/快速分叉；适合用干净的选项设计拉开区分度（但不适合作为长期唯一题型）。
        - `Derivation`：符号推理/公式推导/解析式变形；目标是得到可机判的解析表达式或符号结论，强调推理结构本身。
        - `Numeric`：定量计算题，需要建立科学/数学模型并通过代码执行求解。系统可用内部工具链（Numeric-oracle / solver_tool）来"钉死"数值真值与误差口径，从而避免纯心算错误污染链路。
        """
    )
)

COMMON_QUESTION_TYPE_CAPABILITIES_EN = PromptSection(
    text=dedent(
        """\
        ## QuestionType capabilities (minimal semantics for Director)

        Choosing QuestionType is choosing the problem type *and* its toolchain:

        - `MCQ`: concept disambiguation / convention alignment / quick branching; good for discrimination via clean options (but should not be the only long-term type).
        - `Derivation`: symbolic reasoning / derivation / analytic manipulation; the goal is a uniquely checkable symbolic expression, emphasizing the reasoning structure.
        - `Numeric`: quantitative computation problem requiring modeling and code execution to solve reliably. The system may use internal toolchains (Numeric-oracle / solver_tool) to lock down the numeric ground truth and tolerance conventions, avoiding mental-arithmetic errors.
        """
    )
)

# 通用 Solution 规范 (Path-Fold compatible)
COMMON_SOLUTION_SCHEMA = PromptSection(
    text=dedent(
        """\
        - Solution 用 S1/S2/S3… 的自然语言步骤展示推理链条，必须能从 `premise_bank + Question_i` 推到与 Answer_i 一致的结论（solver 不可见 episode_seed）。
        - 当 i≥1 时，推理中要显式重建或引用前序步骤的关键结论（通常是上一步 key_fact），再结合本轮的新设定完成求解，避免将旧结论当成黑箱常量直接套用。
        - 步骤需清晰、可复现，突出关键判据/计算链路，而非冗长叙事；保持 head–tail 场景下也能让解题者自行推导出必要的中间结论。
        """
    )
)

# Common solution rules (Path-Fold compatible, EN)
COMMON_SOLUTION_SCHEMA_EN = PromptSection(
    text=dedent(
        """\
        - Write Solution as natural-language steps S1/S2/S3...; it must be reproducible from `premise_bank + Question_i` to a conclusion consistent with Answer_i (episode_seed is not solver-visible).
        - For i≥1, explicitly reconstruct or re-derive the key conclusion from the previous step (typically the prior key_fact), then combine it with the current setup; do not treat old conclusions as black-box constants.
        - Keep steps clear and checkable (criteria / equations / scaling), avoid long storytelling; ensure a head-tail solver can derive the necessary intermediate results by themselves.
        """
    )
)

# Known 结构说明 (用于输入解释)
COMMON_KNOWN_TREE = PromptSection(
    text=KNOWN_TREE_DESCRIPTION
)

# Known tree description (EN)
COMMON_KNOWN_TREE_EN = PromptSection(
    text=KNOWN_TREE_DESCRIPTION_EN
)

# World contract model description (shared prompt block)
COMMON_WORLD_CONTRACT_MODEL = PromptSection(
    text=WORLD_CONTRACT_MODEL_TEXT_ZH
)

COMMON_WORLD_CONTRACT_MODEL_EN = PromptSection(
    text=WORLD_CONTRACT_MODEL_TEXT_EN
)

# World contract prompt guidance (shared prompt block)
COMMON_WORLD_CONTRACT_GUIDANCE = PromptSection(
    text=WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_ZH
)

COMMON_WORLD_CONTRACT_GUIDANCE_EN = PromptSection(
    text=WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_EN
)

# Answer contract model description (shared prompt block)
COMMON_ANSWER_CONTRACT_MODEL = PromptSection(
    text=ANSWER_CONTRACT_MODEL_TEXT_ZH
)

COMMON_ANSWER_CONTRACT_MODEL_EN = PromptSection(
    text=ANSWER_CONTRACT_MODEL_TEXT_EN
)

# Metrics 与 Solver 信号说明 (用于 Director 等需要理解 metrics 的角色)
COMMON_METRICS_DESCRIPTION = PromptSection(
    text=METRICS_AND_SOLVERS_DESCRIPTION_ZH
)

# Metrics and solver signals description (EN)
COMMON_METRICS_DESCRIPTION_EN = PromptSection(
    text=METRICS_AND_SOLVERS_DESCRIPTION_EN
)

# Edge QA vs Path QA 概念说明（用于 Director/Format/Draft 等需要理解“同一目标、不同视角”的角色）
COMMON_EDGE_QA_VS_PATH = PromptSection(
    text=EDGE_QA_VS_PATH_DESCRIPTION_ZH
)

# Edge QA vs Path QA (EN)
COMMON_EDGE_QA_VS_PATH_EN = PromptSection(
    text=EDGE_QA_VS_PATH_DESCRIPTION_EN
)

# Multi-Strong Solver 共识信号说明
COMMON_SOLVER_CONSENSUS_DESCRIPTION = PromptSection(
    text=SOLVER_CONSENSUS_DESCRIPTION_ZH
)

# Multi-strong solver consensus signals (EN)
COMMON_SOLVER_CONSENSUS_DESCRIPTION_EN = PromptSection(
    text=SOLVER_CONSENSUS_DESCRIPTION_EN
)

# 给 Director 的 solver 信号“认知”说明：避免重复解释所有 schema 字段，只提供决策所需的最小充分语义。
COMMON_SOLVER_SIGNALS_COGNITION = PromptSection(
    text=dedent(
        """\
        ## Solver 信号（给 Director 的最小充分语义）

        你会在 state_json 中看到这些信号来源：`metrics` / `solver_metrics` / `solver_consensus`。
        它们用于回答三个问题：题目是否可解且自洽、答案是否可信、下一步是 Extend 还是 Revise/Finish。

        - `solver_metrics.edge/path.strong`：多 strong solver 的结构化结果列表；优先看这组分布，而不是把 strong 压成单个结论。每个条目可携带 `question_well_posed` / `correctness_feedback` / `difficulty_feedback`。
        - `metrics.edge.correct_medium` + `metrics.edge.strong_summary`：edge 视角下 medium 与 multi-strong 聚合信号；用于判断“当前题本身是否自洽可解”。
        - `metrics.path.correct_medium` + `metrics.path.strong_summary`：path 折叠视角下 medium 与 multi-strong 聚合信号；用于判断“从起点 P 端到端是否可达”与折叠题是否过难/不自洽。
        - `solver_consensus.strong`：多 strong 的投票信号；在 strong 解不稳定时用于辅助判断：
          - `wellposed_consensus=false` 常提示题目设定/选项/条件有问题；
          - `answer_consensus` 与 `proposed_answer` 明显不一致时，优先怀疑链路答案有误（尤其在 wellposed_consensus=true 时）。
        - `tool tier`：对 Numeric 题，tool solver 可输出可执行代码用于数值求值/验证；若 tool 正确但 medium/strong 错，通常表示题目数值推演负担很重、心算易错（可作为“高难 Numeric”或“需要澄清口径/单位”的信号）。

        使用原则（避免机械阈值）：
        - correctness / well-posedness 优先于 difficulty；
        - token_ratio 与 step_ratio 仅作同一次 run 内的弱趋势参考，不作绝对难度与跨模型比较依据。
        """
    )
)

COMMON_SOLVER_SIGNALS_COGNITION_EN = PromptSection(
    text=dedent(
        """\
        ## Solver signals (minimal semantics for Director)

        In the state JSON you may see: `metrics` / `solver_metrics` / `solver_consensus`.
        They answer three questions: is the problem well-posed/solvable, is the proposed answer reliable, and should you Extend vs Revise/Finish.

        - `solver_metrics.edge/path.strong`: a structured list of multi-strong results; prefer inspecting this distribution rather than compressing strong into a single conclusion. Each item may carry `question_well_posed` / `correctness_feedback` / `difficulty_feedback`.
        - `metrics.edge.correct_medium` + `metrics.edge.strong_summary`: medium and multi-strong aggregate signals under edge view; indicates whether the current step is locally solvable/sound.
        - `metrics.path.correct_medium` + `metrics.path.strong_summary`: medium and multi-strong aggregate signals under path (folded) view; indicates end-to-end reachability from the start P and whether the fold prompt is too hard or not well-posed.
        - `solver_consensus.strong`: multi-strong voting; useful when strong is unstable:
          - `wellposed_consensus=false` often suggests issues in conditions/options/well-posedness.
          - if `answer_consensus` differs from `proposed_answer` while well-posed, the pipeline answer is likely wrong.
        - `tool tier`: for Numeric, the tool solver may emit executable code to compute/verify values; if tool is correct but medium/strong fail, it often means the task is numerically heavy or error-prone without a calculator.

        Principles (avoid mechanical thresholds):
        - correctness / well-posedness first; difficulty second;
        - token_ratio/step_ratio are weak within-run trend signals only (not absolute difficulty; not comparable across models).
        """
    )
)

# ---- Executable (C1) schema snippets ----

COMMON_EXECUTABLE_EXTRACT_SCHEMA = PromptSection(
    text=executable_extract_output_schema_text()
)

COMMON_EXECUTABLE_EXTRACT_SCHEMA_EN = PromptSection(
    text=executable_extract_output_schema_text("en")
)

COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA = PromptSection(
    text=executable_draft_step_output_schema_text()
)

COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA_EN = PromptSection(
    text=executable_draft_step_output_schema_text("en")
)

COMMON_EXECUTABLE_TEST_INPUTS_SCHEMA = PromptSection(
    text=executable_test_inputs_schema_text()
)

COMMON_EXECUTABLE_TEST_INPUTS_SCHEMA_EN = PromptSection(
    text=executable_test_inputs_schema_text("en")
)

# Extend/Revise 通用约束 (NewBackground & Reuse History)
COMMON_EXTEND_CONSTRAINTS = PromptSection(
    text=dedent(
        """\
        # Extend/Revise 通用约束

        **字段语义**：
        - `premise_bank`：新增前提，由 step_cert_builder 写入。
        - `fact_bank` / `step_certs`：可复用结论与证书，由 step_cert_builder 写入。
        - 复用关系通过 `draft_chain.required_fact_ids` 等结构化字段表达，不写入 Question。

        **设计约束**：
        - 保持场景一致（锚定 episode_seed 的话题/机制；同时保证题目可仅凭 `premise_bank + Question` 求解）。
        """
    )
)

# Extend/Revise common constraints (EN)
COMMON_EXTEND_CONSTRAINTS_EN = PromptSection(
    text=dedent(
        """\
        # Extend/Revise Common Constraints

        **Field semantics**:
        - `premise_bank`: new premises, written by step_cert_builder.
        - `fact_bank` / `step_certs`: reusable facts and certificates, written by step_cert_builder.
        - Reuse is expressed via structured fields (e.g., `draft_chain.required_fact_ids`), not in Question.

        **Design constraint**:
        - Stay anchored to the episode_seed topic/mechanism, while ensuring the problem is solvable from `premise_bank + Question` alone.
        """
    )
)

# 通用输出风格
COMMON_STYLE_SECTION = PromptSection(
    text=dedent(
        """\
        - 建模信息，明确 Given / Goal / Constraints，要求跨概念、跨步骤的多步推理。
        - Google-proof：避免可通过检索直接背诵的事实问答，强调专业理解与推理。
        - 保证唯一、可机判答案；即便能直接算出数值，也优先通过选择题表达（将数值或结论放入互斥选项中）。
        - 一定要避免只是简单的计算堆砌，避免简单的考察一些代入和公式的套用。
        """
    )
)

# Common style guidelines (EN)
COMMON_STYLE_SECTION_EN = PromptSection(
    text=dedent(
        """\
        - Model it clearly: Given / Goal / Constraints; require multi-step reasoning across concepts/steps.
        - Google-proof: avoid trivia that can be answered by memorization or search; emphasize professional understanding and reasoning.
        - Ensure a unique, machine-checkable answer; even if a numeric value is computable, prefer expressing it via MCQ by embedding mutually exclusive options.
        - Avoid rote plug-and-chug; avoid pure computation stacking without conceptual reasoning.
        """
    )
)

# 通用 JSON 输出字段定义 (Key list & Basic constraints)
COMMON_JSON_FIELDS_DESC = PromptSection(
    text=dedent(
        """\
        - 仅输出一个合法 JSON 字符串（不得包含额外解释或第二个 JSON）。
        - 顶层键通常包括：Step, Question, Solution, Answer，以及可选的 validation_*。
        - Step 必须与当前轮次一致。
        - Answer 必须使用 LaTeX \\boxed{…} 包裹。
        """
    )
)

# Common JSON output field rules (EN)
COMMON_JSON_FIELDS_DESC_EN = PromptSection(
    text=dedent(
        """\
        - Output **one** valid JSON object only (no extra explanations and no second JSON).
        - Top-level keys typically include: Step, Question, Solution, Answer, and optional validation_* fields.
        - Step must match the current step index.
        - Answer must be wrapped in LaTeX `\\boxed{...}`.
        """
    )
)

# ---- Reuse / Assumptions rules (shared across roles/modes) ----

COMMON_REUSED_CONCLUSIONS_RULES = PromptSection(
    text=dedent(
        f"""\
        ### 双版本 Question 协议

        **输出字段含义**：
        - `{FIELD_DRAFT_QUESTION_EXPLICIT}`：显式版，可引用前序结论，用于验证逻辑链。
        - `{FIELD_DRAFT_QUESTION}`：隐藏版，给 solver 的实际题面。
        - `{FIELD_REUSED_CONCLUSIONS}`：记录复用了什么（内部日志）。

        **设计约束**：隐藏版对应 Now QA 的 solver 输入，可依赖 `fact_bank/step_certs(<t)` 复用前序结论；
        但不要出现内部实现指针（如 history/fact_bank/step_certs/premise_bank/known_0），也不要通过复制粘贴前序答案来伪造“题干自足”。
        """
    )
)

# Reuse previous conclusions (EN)
COMMON_REUSED_CONCLUSIONS_RULES_EN = PromptSection(
    text=dedent(
        f"""\
        ### Dual-Question Protocol

        **Output field semantics**:
        - `{FIELD_DRAFT_QUESTION_EXPLICIT}`: explicit version; may reference prior conclusions; for verification.
        - `{FIELD_DRAFT_QUESTION}`: hidden version; actual solver input.
        - `{FIELD_REUSED_CONCLUSIONS}`: internal log of what is reused.

        **Design constraint**: the hidden version is the Now QA solver input and may rely on `fact_bank/step_certs(<t)` for reuse;
        do not mention internal pointers such as history/fact_bank/step_certs/premise_bank/known_0, and do not fake “self-containedness” by copy-pasting prior answers into the question body.
        """
    )
)


COMMON_REUSED_REFS_RULES = PromptSection(
    text=dedent(
        f"""\
        ### 结构化复用引用（`{FIELD_REUSED_REFS}`）

        **协议约束**（系统校验用）：
        - 类型：`object[]`，每个元素含 `source_step: int`（必填）。
        - 必须包含对上一步（step=i-1）的引用。
        - 若上一步是 MCQ，需额外填 `mcq_choice`。
        """
    )
)

# Structured reuse references (EN)
COMMON_REUSED_REFS_RULES_EN = PromptSection(
    text=dedent(
        f"""\
        ### Structured Reuse References (`{FIELD_REUSED_REFS}`)

        **Protocol constraint** (system validation):
        - Type: `object[]`, each element has `source_step: int` (required).
        - Must include a reference to the previous step (step=i-1).
        - If previous step is MCQ, additionally fill `mcq_choice`.
        """
    )
)

COMMON_FORMAT_DUAL_QUESTION_USAGE = PromptSection(
    text=dedent(
        """\
- **⚠️ Input Source（输入来源）**：
  * `draft_question_explicit`：仅供理解，**禁止用于输出**。
  * `draft_question`：你的 **唯一内容来源**。
- **⚠️ Transformation Rule（转换规则）**：
  * 允许：结构化重写（重排、统一格式）。
  * 禁止：引入新信息（从 `draft_solution`/`draft_question_explicit`/常识 添加条件/数值/公式）。
  * 缺结构判 fail：不要编造缺失的结构（如 MCQ 缺选项），设 `validation_passed=false`。
        """
    )
)

COMMON_FORMAT_DUAL_QUESTION_USAGE_EN = PromptSection(
    text=dedent(
        """\
- **⚠️ Input Source**:
  * `draft_question_explicit`: for understanding only, **DO NOT use for output**.
  * `draft_question`: your **SOLE content source**.
- **⚠️ Transformation Rule**:
  * Allowed: Structural rewriting (reorder, unify format).
  * Forbidden: Introduce new info (add conditions/values/formulas from `draft_solution`/`draft_question_explicit`/knowledge).
  * Missing structure = fail: Do not invent missing structure (e.g., MCQ without options); set `validation_passed=false`.
        """
    )
)


COMMON_DRAFT_BACKGROUND_RULES = PromptSection(
    text=dedent(
        f"""\
        ### Draft 新增设定（`{FIELD_DRAFT_BACKGROUND}`）

        **字段含义**：新增的前提条件（假设/参数/实验条件），供后续步骤复用。

        **设计约束**：
        - 默认输出空数组 `[]`。
        - 只放"前提"，不放"推导结论"（结论属于 Solution，不属于 Background）。
        - 必须与既有 Background 自洽。
        """
    )
)

# Draft background candidates / NewBackground rules (EN)
COMMON_DRAFT_BACKGROUND_RULES_EN = PromptSection(
    text=dedent(
        f"""\
        ### Draft Background Candidates (`{FIELD_DRAFT_BACKGROUND}`)

        **Field semantics**: new premises (assumptions/parameters/conditions) for later steps to reuse.

        **Design constraint**:
        - Default: output empty `[]`.
        - Only premises, not derived conclusions (conclusions belong in Solution, not Background).
        - Must stay consistent with existing Background.
        """
    )
)


COMMON_FIRST_STEP_REUSE_RULES = PromptSection(
    text=dedent(
        f"""\
        ### 首题模式（history 为空）

        - `{FIELD_REUSED_CONCLUSIONS}`：空或"首题，无前序结论"。
        - `{FIELD_REUSED_REFS}`：空数组 `[]`。
        """
    )
)

# First-step special rules (EN)
COMMON_FIRST_STEP_REUSE_RULES_EN = PromptSection(
    text=dedent(
        f"""\
        ### First-step Mode (empty history)

        - `{FIELD_REUSED_CONCLUSIONS}`: empty or "first step; no prior conclusions".
        - `{FIELD_REUSED_REFS}`: empty array `[]`.
        """
    )
)


# ---- Draft prompt building blocks (shared) ----

COMMON_DRAFT_ROLE_DESC = PromptSection(
    text=dedent(
        f"""\
        - **题面双版本**（重要！）：
          - `{FIELD_DRAFT_QUESTION_EXPLICIT}`：显式版本，可以写出对前序结论的引用（用于验证逻辑链）；
          - `{FIELD_DRAFT_QUESTION}`：隐藏版本，不透露前序结论（实际输出给 solver）；
        - 解题思路与关键推理步骤（Solution 草稿）；
        - 预期答案结论（Answer 草稿）；
        - 需要新增的假设/设定（{FIELD_DRAFT_BACKGROUND}）；
        - 计划复用的前序结论（{FIELD_REUSED_CONCLUSIONS}）；
        - 结构化复用引用（{FIELD_REUSED_REFS}，用于系统侧稳定对齐）；
        - 对场景锚定的自检说明（{FIELD_GROUNDING_CHECK}）。

        ---
        """
    )
)

COMMON_DRAFT_ROLE_DESC_EN = PromptSection(
    text=dedent(
        f"""\
        - **question in TWO versions** (important!):
          - `{FIELD_DRAFT_QUESTION_EXPLICIT}`: explicit version that may reference prior conclusions (for verification);
          - `{FIELD_DRAFT_QUESTION}`: hidden version without any prior result leakage (actual solver input);
        - solution outline with key reasoning steps ({FIELD_DRAFT_SOLUTION})
        - expected final answer ({FIELD_DRAFT_ANSWER})
        - new premises to introduce ({FIELD_DRAFT_BACKGROUND})
        - which prior conclusions the step conceptually builds on ({FIELD_REUSED_CONCLUSIONS})
        - structured reuse anchors for the system ({FIELD_REUSED_REFS})
        - a short grounding self-check ({FIELD_GROUNDING_CHECK})
        """
    )
)

COMMON_DRAFT_INPUT_DESC = PromptSection(
    text=dedent(
        """\
        ## 输入说明

        系统会把本轮可见的信息填入占位符，你会看到一段类似（示意）的内容：

        Step_{prev} = $prev_step

        [Known (结构化 JSON)]
        $known_full

        **Known JSON 结构说明**：
        - `known_0`：整条链共享的初始物理背景（字符串）
        - `history`：按 step 顺序累积的问答记录（列表），每个元素包含 `question_i` 和 `answer_i` 字段
        - `background`：各步通过 `new_background` 累积的物理设定、模型假设、参数约束等（字符串列表）
        - `derived_facts`：可选的衍生结论聚合区（字符串列表）

        **请仔细阅读 Known JSON 中的各字段内容**，特别是：
        - 从 `history` 字段了解前序步骤的问答内容，用于复用前序结论和避免答案泄露
        - 从 `background` 字段了解已建立的物理模型、数学形式、关键假设，确保新增设定与其自洽

        [Known_0（快速引用）]
        $known_0

        [Director Notes]
        $director_notes
        """
    )
)

COMMON_DRAFT_INPUT_DESC_EN = PromptSection(
    text=dedent(
        """\
        ## Inputs

        Step_{prev} = $prev_step

        [Known (structured JSON)]
        $known_full

        [Known_0 (quick reference)]
        $known_0

        [Director Notes]
        $director_notes
        """
    )
)

COMMON_DRAFT_QUESTION_TYPE_SECTION = PromptSection(
    text=dedent(
        f"""\
        ### 题型与答案格式约定（供 Draft 参考）
{COMMON_QUESTION_TYPES.text}
{COMMON_ANSWER_SCHEMA.text}
        """
    )
)

COMMON_DRAFT_QUESTION_TYPE_SECTION_EN = PromptSection(
    text=dedent(
        f"""\
        ### Question types / answer format
{COMMON_QUESTION_TYPES_EN.text}
{COMMON_ANSWER_SCHEMA_EN.text}
        """
    )
)

COMMON_DRAFT_GROUNDING_CHECK = PromptSection(
    text=dedent(
        f"""\
        ---

        ## 自检（`{FIELD_GROUNDING_CHECK}`）

        **字段含义**：简述本题与 `known_0` 的关联，确保题目扎根于科学背景。
        """
    )
)

COMMON_DRAFT_GROUNDING_CHECK_EN = PromptSection(
    text=dedent(
        f"""\
        ---

        ## Grounding self-check (`{FIELD_GROUNDING_CHECK}`)

        **Field semantics**: briefly describe how the problem relates to `known_0`, ensuring scientific grounding.
        """
    )
)
