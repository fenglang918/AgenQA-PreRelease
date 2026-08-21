"""Extend/Compress/Plan-Critique/Reflect-Fuse prompts (Python style)."""

from __future__ import annotations
from textwrap import dedent

from ._base import PromptSection, PromptTemplate
from .common import (
    COMMON_KNOWN_TREE,
    COMMON_ANSWER_SCHEMA,
    COMMON_QUESTION_TYPES,
    COMMON_SOLUTION_SCHEMA,
    COMMON_STYLE_SECTION,
    COMMON_JSON_FIELDS_DESC,
)

__all__ = [
    "EXTEND_UPGRADE_V1",
    "COMPRESS_HISTORY_PROMPT",
    "COMPRESS_HISTORY_PROMPT_EN",
    "PLAN_CRITIQUE_PROMPT",
    "PLAN_CRITIQUE_PROMPT_EN",
    "REFLECT_FUSE_PROMPT",
    "REFLECT_FUSE_PROMPT_EN",
]

# === Extend-Upgrade Sections ===

_EXT_ROLE = PromptSection(
    text=dedent(
        """\
        # Extend‑Upgrade（扩展算子说明）

        你是多步难题链条中的“生长算子”，负责在上一轮 K/Q/A 的基础上，生成下一轮的已知条件、题面和答案。
        """
    )
)

_EXT_INPUT = PromptSection(
    text=dedent(
        """\
        ## 输入说明（你能看到什么）

        系统会把“上一轮的 K/Q/A 记录”填入本提示中的占位符：
        {
          Step:  $step,
          Known: $known,
          Question: $question,
          Answer: $answer
        }

        含义说明：
        - `Step` / `$step`：上一轮的步数 i‑1（整数）；
        - `Known` / `$known`：上一轮系统维护后的已知条件对象；
        - `Question` / `$question`：上一轮的题面文本；
        - `Answer` / `$answer`：上一轮的最终答案。
        """
    )
)

_EXT_TASK_INTRO = PromptSection(
    text=dedent(
        """\
        你可以把这一段理解为“当前链条最后一轮的完整摘要”，本轮需要在此基础上生成下一轮 i 的题目。
        令本轮步数 i = $step + 1。

        ## 任务
        从 (i‑1) 轮扩展出第 i 轮，重点在于构建严密的逻辑链条：

        1. 系统侧会自动维护 Known（仅陈述“已学事实”）：
           - 程序会保持 `"known_0"` 不变；
           - 程序会在 `Known.history` 末尾追加上一轮的问答；
           - 你可以将输入中的 `Known_$step` 视为只读上下文，**通常不需要在输出中返回新的 Known**。
        """
    )
)

_EXT_LOGIC_RULES = PromptSection(
    text=dedent(
        """\
        2. 基于（上一轮 Known + 当前问答），设计新的第 i 步题目时，需遵守以下原则：
          - **新引入背景（NewBackground）**：本轮新增的假设、参数设定或实验条件，应当写入 `"NewBackground"` 字段。
            - **NewBackground 的职责边界**：
              * ✅ 允许：新增的物理模型假设、参数约束、实验条件、理论框架设定等"前提条件"
              * ❌ 禁止：本题要推导的结论、答案中的数值或表达式、可以直接从题干推出的结果
              * ❌ 禁止：前序步骤已推导出的结论（这些应通过 `history` 复用，而非重复写入 `NewBackground`）
            - **硬性约束**：通常仅在 i=1、2 时允许适度引入（每步 0–2 条）；当 i≥3 时，原则上**禁止**再引入新的 NewBackground。
          - **复用前序结论（Reuse History，强制要求）**：
            - **核心约束**：本轮**必须**利用**上一步（step=i-1）的 Answer 或 Solution 中已证实的结论**作为推理基础。这是强制要求，不能跳过上一步直接引用更早的结论。
            - 可以额外复用更早步骤的结论，但上一步的结论是必须的，这确保了链条的连续性和递进性。
            - **注意**：Now QA（edge QA）下 solver 可见 `history/fact_bank/step_certs`，因此题面允许把前序结论作为已知输入或显式引用（例如“使用上一步的结论…”）。
              但更推荐用结构化复用引用（reused_refs/required_fact_ids）+ 简短文字锚点来表达依赖，避免大段复述前序答案造成冗余或不一致。

            **⚠️ 避免复述/复制前序答案（防止递进坍塌）**：
            - 尽量不要在本轮 Question 题干中直接复制前序 Answer 的**选项字母、数值结果、结论性陈述**（除非它确实是本轮的明确给定输入）；
            - 即使换一种表述方式（如把"答案是 B"改写为"合理的解释是：σ 越小时..."），只要实质内容与前序 Answer 相同，通常都会导致冗余与递进坍塌，应避免；
            - **负面示例（严禁）**：
              - ❌ "在上一轮分析中，合理的解释是：当 σ 很小时，高阶模 n≥1 中会出现一簇数值较小且彼此接近的本征值..." ← 复述了 Q0 选项 B 的内容
              - ❌ "由前一轮可知 Γ_3(σ_2)/Γ_0 = 4" ← 直接暴露了前序数值答案
              - ❌ "根据上一步的结论，小本征值群在 σ 增大时会消失或分裂..." ← 复述了前序答案的核心结论
            - **正面示例（允许）**：
              - ✅ "在前述模型的基础上，现进一步假设..." ← 不透露前序答案内容
              - ✅ "利用已给的洛伦兹型响应表达式..." ← 引用 background 中的公式，而非 Answer
              - ✅ "假设在 σ=σ_2 时峰高衰减为 H_0/4，求 Γ_3(σ_2)/Γ_0" ← 把结论作为待求目标

          - **允许的"数学/物理中间节点"**：
            - 必须显式承接此前链条的结论或参数；
            - Question_i 中应保留论文场景或前几步构建的物理/工程背景，不得跳到一个无关学科；
            - 若删除 Known.known_0 后变成一道通用题，说明已脱离场景，需重写。
          - **逻辑叠加**：所有题目必须在 `known_0 + 已有 background + 前序结论` 上纵向加深推理。
        """
    )
)

_EXT_FIELDS_REQ = PromptSection(
    text=dedent(
        """\
        在此基础上，你需要构造新的 Question_i，满足：
        - 解题者利用此前各步的 Answer/Solution，加上本轮的 NewBackground 和题干条件应能解出唯一答案；
        - **强制要求**：Question_i 的推理链条必须在 Solution 中**显式复用上一步（step=i-1）的 Answer 或关键中间结论**。这是必须的，不能跳过。可以额外复用更早步骤的结论，但上一步的结论是强制要求。
        - 题面需使用自然语言，明确 Given / Goal / 约束条件。

        - 题型约定（按本轮 QuestionType 执行）：
            - 本轮 QuestionType = `$question_type`。
            - 题型简要提示：$question_type_guidance
        """
    )
)

_EXT_FIELD_DETAILS = PromptSection(
    text=dedent(
        """\
        3. 详细字段要求：

        - Question:
        """
    )
)

# 组合具体字段要求 (复用 Common)
# 注意：这里我们手动拼接一些特定前缀，或者直接把 Common Section 放进 Template list
# 为了排版好看，我们用一个 Section 把它们串起来，或者在 Template 里依次列出

_EXT_DIRECTOR_ADVICE = PromptSection(
    text=dedent(
        """\
        若导演（Director）通过 `director_notes` 给出希望关注的角度或提升难度的思路，请将其中 1–2 条合理要点融入本轮 Question_i 的设定；在此基础上，若你已有“如何变难”的直觉，也可以结合自己的判断适度提升复杂度，前提是题面自洽、可解。
        """
    )
)

_EXT_SELF_CHECK = PromptSection(
    text=dedent(
        """\
        ## 输出前自检（必须执行）

        在输出 JSON 之前，请逐条检查以下约束，若任一项不通过则必须修改后再输出：

        1. **答案泄露检查**：逐一对比本轮 Question 与 history 中每个 `answer_i` 的内容：
           - 若 Question 中出现了任何 `answer_i` 的**选项字母**（如"选项 B 是正确的"）→ 必须删除
           - 若 Question 中出现了任何 `answer_i` 的**数值结果**（如"Γ_3/Γ_0 = 4"）→ 必须改为待求目标或删除
           - 若 Question 中**复述了某个选项的核心结论**（即使没提选项字母）→ 必须重写，避免泄露

        2. **结论复述检查**：检查 Question 中是否有以下模式：
           - "在上一轮分析中，合理的解释是..." → 禁止
           - "前一轮已证明/已知/可推得..." + 具体结论 → 禁止
           - "根据上一步的结论，..." + 具体结论 → 禁止

           这些结论应当由解题者在 Solution 中重新推导，而不是在 Question 中直接给出。

        3. **正确的引用方式**：Question 中只能引用：
           - `known_0` 中的背景设定
           - `background` 中的假设/参数定义
           - 本轮新增的 `NewBackground`
           - **不能**引用 history 中的 answer 内容

        4. **链条递进检查（重要，强制复用上一步）**：确保本题真正在前序题目基础上递进，而不是简单重复：
           - 检查点1（强制要求）：本题的 Solution 是否**必须依赖上一步（step=i-1）的结论**？如果删除上一步的结论，本题是否仍能独立求解？若仍能求解，说明没有真正递进，必须重写。**上一步的结论是强制要求，不能跳过。**
           - 检查点2：本题与上一步题目的**核心推理模式**是否过于相似？是否只是换了参数或问法，但本质上还是同一类问题？若是，必须引入新的推理维度。
           - 检查点3：本题是否引入了**新的推理维度**（如新的物理机制、新的数学工具、新的约束条件、新的综合视角），而不仅仅是重复上一步的推理路径？
           - 若发现只是简单重复或换汤不换药，必须重写题目，确保真正递进。
        """
    )
)

_EXT_OUTPUT_FORMAT = PromptSection(
    text=dedent(
        """\
        ## 输出格式（严格 JSON，用 ```json 代码块包裹）

        仅输出一个合法 JSON 对象，必须用 ```json 代码块包裹，顶层键要求如下：
        - `"Step"`：整数，必须等于 i（即 `$step + 1`）；
        - `"Question"`：题面文本（可包含公式），不包含答案本身；题末可简要提醒"最终答案用 \\\\boxed{} 包裹"；
        - `"Solution"`：字符串，按 "S1: …; S2: …; …" 给出多步解法；步骤必须足以从 `known_0 + 既有背景 + 本轮 NewBackground` 推导出与 Answer_i 完全一致的结论，并在推理中**必须显式复用上一步（step=i-1）的 Answer 或关键中间结论**（这是强制要求，不能跳过）。可以额外复用更早步骤的结论。
        - `"Answer"`：仅包含最终答案，遵守上方的统一答案格式；
        - `"NewBackground"`：可选字符串或字符串列表，描述本轮新增的、不能从 `known_0` 推出的背景假设/参数设定；
        - **不要返回 `"Known"`，由系统统一维护 Known 结构。**

        示例结构（仅示意）：

        ```json
        {
          "Step": $next_step,
          "Question": "...题面文本...",
          "Solution": "S1: ...; S2: ...",
          "Answer": "\\\\boxed{...}",
          "NewBackground": ["...可选新增背景..."]
        }
        ```
        """
    )
)

_EXT_TEMPLATE = PromptTemplate(
    name="extend_upgrade_v1_mod",
    sections=[
        _EXT_ROLE,
        _EXT_INPUT,
        COMMON_KNOWN_TREE,  # 插入通用的 Known 结构说明
        _EXT_TASK_INTRO,
        _EXT_LOGIC_RULES,
        _EXT_FIELDS_REQ,
        COMMON_QUESTION_TYPES, # 插入通用的题型说明

        PromptSection(text="4. 写出结构化解法 Solution（**必填**）："),
        COMMON_SOLUTION_SCHEMA, # 插入通用的 Solution 规范

        PromptSection(text="5. 给出 Answer_i（统一答案格式）："),
        COMMON_ANSWER_SCHEMA,   # 插入通用的 Answer 规范

        PromptSection(text="## 风格与约束"),
        COMMON_STYLE_SECTION,   # 插入通用的风格约束
        _EXT_DIRECTOR_ADVICE,

        _EXT_SELF_CHECK,        # 插入自检步骤
        _EXT_OUTPUT_FORMAT,
    ]
)

EXTEND_UPGRADE_V1 = _EXT_TEMPLATE.render_body({})


# Compress / Plan / Reflect 暂时保持原样（Monolithic String），后续可按相同模式拆分
COMPRESS_HISTORY_PROMPT = dedent(
    """\
    # Input
    {
      Step: $step,
      Known_$step: $known,
      Question_$step: $question,
      Answer_$step: $answer
    }

    # Task · Compress/Refactor History（压缩/重构历史，相当于“带历史的再 QA‑Init”）
    - 令 i = $step + 1。
    - 目标：在保持 head–tail 可复现的前提下，利用已有的 `Known` 与多轮问答历史，整理出**一段新的背景 Known 文本**，并基于该背景设计下一步的高质量单选题。若导演（Director）通过 `director_notes` 给出偏好/提醒，请优先结合这些要点；在此基础上，可结合你自己的“如何变难”直觉适度提升难度，前提是题面自洽、可解。
    - 行动：
      1) 压缩：阅读当前 `Known` 与过往 Question/Answer，**至少参考 2–3 条历史问答中的关键结论**（例如某个区间判据、模式排序或临界条件），用它们来选择和组织真正决定物理情景的参数与关系；但请尽量避免直接把“某模式属于哪个区间”等具体答题结论逐字写入新的 `Known`，而是保留足够的推理空间，让这些结论可以在新题中通过多步推导重新获得。新的 `Known` 应主要陈述模型设定、无量纲参数定义及其物理意义，而不是列出已经算好的最终判断。
      2) 重构：在这段新的背景 Known 基础上，提出新的 Question_i（**必须为单选题，选项用 A/B/C/D 表示**），要求解题过程**需要综合利用上述被参考的历史结论背后的物理关系**（例如同一组无量纲参数、区间边界、线性标度），再叠加新条件进行多步推理，而不是简单重复原有问题；并给出唯一可机判的 Answer_i（结论用 \\\\boxed{…}）与 HeadTailSolution（S1/S2/…）。如有合理的 `harder_suggestion`，应在不损害题面自洽性的前提下尽量吸收其中 1–2 条（例如增加对不同参数组合/实验情形/截断策略的比较），使题目在综合性和推理深度上显著高于前几步。
    - Known 更新（替换模式）：
      - 输出中的 `Known` 字段是**必填**：表示压缩/重构后的背景描述，应为能够独立配合 Question_i 使用的完整上下文。
      - **保持 `Known` 为一段（或少段）纯文本**，避免嵌套 JSON、转义大括号等；不要逐字复制 `$known` 的原始 JSON，而是用自然语言重述关键设定（长度以能支撑新题推理为限）。
      - 不要在 Question 中提到诸如 “known_0”、“history” 等内部字段名，只以物理/数学语境自然表述背景。
      - 过往问答如何被压缩进 history 由下游程序处理，不需要在输出中显式给出 `history` 结构。

    # Style & Constraints
    - 第一性原理与终极目标：题面需体现关键推理路径，避免平庸直代；通过合理设问与信息组织，使解题需要多步、可复现且有方法脚印，但不强制具体数量或标签。
    - 复现守护：仅凭 `known_0 + 新题面` 可独立求解；口径一致、单位一致、守恒账本清晰。
    - Question 中明确答案格式（此处固定为单选题 A/B/C/D）；Answer 使用 LaTeX `\\\\boxed{…}`，仅给出一个字母（如 `\\\\boxed{A}`）。
    - HeadTailSolution：用 “S1: …; S2: …” 简要列出求解步骤，能从 `known_0 + Question` 推到 Answer，避免空缺。对于 Compress‑History 产生的题目，解法中应**显式体现对历史问答关键结论的复用**：例如在某一步先根据背景重新得到一个无量纲判据或区间边界（对应早期某问的结论），再在后续步骤中叠加新条件做比较或排序，而不是只做一次性代入。

    # Output（严格 JSON，用 ```json 代码块包裹；注意：本算子会替换历史，而非追加）
    - 只输出一段 JSON，必须用 ```json 代码块包裹。
    - 顶层字段集合必须**严格**为：`Step`、`Known`、`Question`、`Answer`、`HeadTailSolution`，不允许添加任何其他字段。
    - `Known` 必须是压缩/重构后的背景纯文本，不要再嵌套 JSON/转义的字符串。

    ```json
    {"Step": $next_step,
     "Known": "...压缩后的背景纯文本...",
     "Question": "...Question_i 文本（单选题 A/B/C/D）...",
     "Answer": "...\\\\boxed{A}...",
     "HeadTailSolution": "S1: ...; S2: ...; S3: ..."}
    ```
    """
)


COMPRESS_HISTORY_PROMPT_EN = dedent(
    """\
    # Input
    {
      Step: $step,
      Known_$step: $known,
      Question_$step: $question,
      Answer_$step: $answer
    }

    # Task · Compress/Refactor History ("history-aware re-init")
    - Let i = $step + 1.
    - Goal: while preserving head-tail reproducibility, use the existing Known and multi-step QA history to produce:
      1) a **new compressed background** (the `Known` field in the output), and
      2) a **new high-quality single-choice MCQ** for step i (Question/Answer/HeadTailSolution).
    - If Director provides preferences in `director_notes`, follow them.
    - You may increase difficulty modestly, but only if the problem remains self-consistent and solvable.

    Key requirements:
    1) Compress: read the existing Known + past QA. Reference at least 2–3 key ideas/conclusions implicitly, but do **not** copy the literal previous answers into the new Known.
       The new Known should mostly state model setup, definitions, and constraints—leave enough reasoning space so conclusions are not "given away".
    2) Refactor: based on the new Known, propose a new Question_i (must be single-choice MCQ with options A/B/C/D) that requires multi-step reasoning.
       Provide a unique machine-checkable Answer_i (\\boxed{A/B/C/D}) and a brief HeadTailSolution (S1/S2/...).

    Known update (replacement semantics):
    - The output `Known` field is required. It must be a standalone background text usable with the new Question.
    - Keep `Known` as plain text (one or a few paragraphs). Do not embed JSON and do not escape braces.
    - Do not mention internal field names like "known_0" or "history" in the Question; write in natural scientific language.
    - Downstream code will rebuild history; you do not need to output `history` structure.

    Style & constraints:
    - The Question must explicitly say: "Answer with the option letter only (A/B/C/D)".
    - Answer must be exactly one letter wrapped in LaTeX: \\boxed{A}.
    - HeadTailSolution must be sufficient to derive the answer from `Known + Question` (no history pointers).
    - All output text must be in English.

    # Output (strict JSON in a ```json code block; this operator replaces history)
    - Output one JSON only, wrapped in ```json.
    - Top-level keys must be exactly: Step, Known, Question, Answer, HeadTailSolution (no extra keys).

    ```json
    {"Step": $next_step,
     "Known": "...compressed background plain text...",
     "Question": "...MCQ with options A/B/C/D (answer with letter only)...",
     "Answer": "\\\\boxed{A}",
     "HeadTailSolution": "S1: ...; S2: ...; S3: ..."}
    ```
    """
)


PLAN_CRITIQUE_PROMPT = dedent(
    """\
    # Input
    {
      Step: $step,
      Known_$step: $known,
      Question_$step: $question,
      Answer_$step: $answer
    }

    # Task · Plan‑Critique（规划—批判扩展）
    - 令 i = $step + 1。
    - 站在“宏观策略/多线索整合”的角度，给出新的 Question_i（可优先使用选择题形式），要求能仅凭 known_0 + 题面独立求解。
    - 对上一轮方案做“批判性检查”，在 Question 的设问中嵌入易错点（如：口径未统一/边界取错/单位不一致/忽略依赖）。
    - 更新 Known：保持 known_0 不变，将上一轮 QA 以标准格式追加到 history 末尾。
    - 给出唯一、可机判的 Answer_i（最终结论必须 \\\\boxed{…}）。

    # Style & Constraints
    - 若为选择题：在 Question 中明确“仅给出选项字母（如 A/B/C/D）”。
    - 题面应包含至少两个“未直给中间量”的推理步骤；显式要求关键方法脚印（如：分层/统一口径/不等式/DP 状态）。
    - 闭卷自洽；Answer 使用 LaTeX `\\\\boxed{…}`。

    # Output（严格 JSON，用 ```json 代码块包裹）

    ```json
    {"Step": $next_step,
     "Known": {"known_0": "...", "history": [{"question_0": "...", "answer_0": "..."}, ...]},
     "Question": "...Question_i...",
     "Answer": "...\\\\boxed{...}..."}
    ```
    """
)


PLAN_CRITIQUE_PROMPT_EN = dedent(
    """\
    # Input
    {
      Step: $step,
      Known_$step: $known,
      Question_$step: $question,
      Answer_$step: $answer
    }

    # Task · Plan-Critique (strategy-aware extend)
    - Let i = $step + 1.
    - Propose a new Question_i that is solvable using only known_0 + the question statement (no "see above"/history pointers).
    - Embed common failure modes as traps (e.g., inconsistent conventions, wrong boundary, unit mismatch).
    - Update Known: keep known_0 unchanged and append the previous QA into history (downstream may normalize).
    - Provide a unique machine-checkable Answer_i wrapped in \\boxed{...}.
    - All output text must be in English.

    # Output (strict JSON in ```json)

    ```json
    {"Step": $next_step,
     "Known": {"known_0": "...", "history": [{"question_0": "...", "answer_0": "..."}, ...]},
     "Question": "...Question_i...",
     "Answer": "...\\\\boxed{...}..."}
    ```
    """
)


REFLECT_FUSE_PROMPT = dedent(
    """\
    # Input
    {
      Step: $step,
      Known_$step: $known,
      Question_$step: $question,
      Answer_$step: $answer
    }

    # Task · Reflect‑Fuse（反思与融合扩展）
    - 令 i = $step + 1。
    - 先对上一轮 Question/Answer 进行“方法反思”，指出可替代或互补的方法线索（如：分层加权/口径归一化/不等式边界/构图容量/先验-似然 等），并“融合”成一种新的解题路径。
    - 更新 Known：
      - 保持 "known_0" 不变；
      - 将上一轮问答以标准格式追加到 history 末尾：
        append {"question_$step": $question, "answer_$step": $answer}
    - 基于反思-融合后的方法，提出新的 Question_i，并给出唯一、可机判的 Answer_i（最终结论必须 \\\\boxed{…}）。

    # Style & Constraints
    - 多样性：尽量采用与上一轮“方法家族”不同的视角；在 Question 中显式体现关键操作脚印（如“统一口径/定义权重/KKT/边界分析/全概率式”等）。
    - 闭卷自洽：不得引用“见图/上文”；给出必要常数/单位。
    - 结果格式在 Question 中明确；Answer 使用 LaTeX `\\\\boxed{…}`。

    # Output（严格 JSON，用 ```json 代码块包裹）

    ```json
    {"Step": $next_step,
     "Known": {"known_0": "...", "history": [{"question_0": "...", "answer_0": "..."}, ...]},
     "Question": "...Question_i...",
     "Answer": "...\\\\boxed{...}..."}
    ```
    """
)


REFLECT_FUSE_PROMPT_EN = dedent(
    """\
    # Input
    {
      Step: $step,
      Known_$step: $known,
      Question_$step: $question,
      Answer_$step: $answer
    }

    # Task · Reflect-Fuse (method reflection + fusion)
    - Let i = $step + 1.
    - Reflect on the previous Question/Answer and identify alternative/complementary solution methods.
    - Fuse them into a new reasoning route.
    - Update Known:
      - keep "known_0" unchanged;
      - append the previous QA into history (downstream may normalize).
    - Propose a new Question_i based on the fused method and provide a unique machine-checkable Answer_i (\\boxed{...}).
    - All output text must be in English.

    # Output (strict JSON in ```json)

    ```json
    {"Step": $next_step,
     "Known": {"known_0": "...", "history": [{"question_0": "...", "answer_0": "..."}, ...]},
     "Question": "...Question_i...",
     "Answer": "...\\\\boxed{...}..."}
    ```
    """
)
