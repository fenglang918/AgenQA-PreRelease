"""Brief/Extract V2 reasoning prompts (Python style; single source of truth).

These prompts are used by KnownInit (PaperBrief v2 + Extract v2) in roles_nodes.py.
"""

from __future__ import annotations

from textwrap import dedent

__all__ = [
    "PAPER_BRIEF_V2_REASONING",
    "PAPER_BRIEF_V2_REASONING_EN",
    "EXTRACT_V2_REASONING",
    "EXTRACT_V2_REASONING_EN",
]


PAPER_BRIEF_V2_REASONING = dedent(
    """\
    # PaperBrief‑V2（Reasoning‑First, Justification‑Explicit）示例 Prompt

    你扮演“科学骨架抽取者”。目标是把论文重建为**可用于链式难题生成**的知识骨架，并显式区分：
    - **Evidence**：论文“发现了什么”（事实层）
    - **Justification**：论文“为什么认为 evidence 能支撑某个假设/主张”（论证层）

    请牢记：**以输入论文文本为唯一信息源**。只允许抽象/归纳，不要编造未出现的事实、数值或推导；若论证缺口无法由文本闭合，请用 `gap` 标注不确定性。

    ---

    ## 概念层级（Brief‑V2 Schema）

    请严格按以下层级抽取，避免混淆或结论泄露：

    1.  **Background（背景）**
        *   主题、研究对象、任务与动机（“我们在讨论什么/为什么关心”）。
        *   不包含工具箱细节，更不包含结论。

    2.  **Premises / Dependencies（前提/依赖）**
        *   论文默认可用、无需本文论证的工具箱：定义/符号、已知定理、公认模型/方程、方法部件等。
        *   判别：问“这个点需要本文证明/验证吗？”→ 不需要 → Premise。

    3.  **Assumptions（假设） + Conditions（适用边界）**
        *   论文提出并依赖本文论证/证据支撑的关键假设/近似，并给出其适用边界与失效情形。
        *   判别：问“这是论文自己提出并试图验证/支撑的吗？”→ 是 → Assumption。

    4.  **Evidence（证据/发现）**
        *   实验/仿真/推导产物中“观察到的事实/趋势/对比结果”（事实输入，不解释为什么）。
        *   轻量列举即可，避免装置/数据点堆叠。

    5.  **Justifications（论证/解释）**
        *   解释层：为什么 evidence 支撑某个 assumption/claim（机理、判据、标度关系、推导要点、对照逻辑等）。
        *   以 “evidence → supports → because” 原子条目给出；不够闭合则填 `gap`，不要硬编。

    6.  **Reasoning Summary（推理脉络摘要）**
        *   极简串联：Premises + Assumptions/Conditions + Evidence + Justifications → Conclusion。
        *   这是理解性摘要，不是可直接拆题的模板。

    7.  **Conclusion（核心结论/Core Claim）**
        *   原子化、独立可读的核心知识陈述（避免把答案细节泄露到 Background/Premises）。

    ---

    ## 输入

    [Paper Text]
    {paper}

    ---

    ## 任务

    1.  **学科锚定**：识别论文所属学科及二级方向（如“物理‑低温物理”）。
    2.  **分层抽取**：提取 `background`, `premises`, `assumptions(with condition/breaks_when)`, `evidence`, `justifications`。
    3.  **生成推理脉络摘要 (`reasoning_summary`)**：紧凑串联逻辑流向，并标注 gap/不确定性（如有）。
    4.  **提炼核心结论 (`conclusion`)**：输出为原子化、独立可读的 `core_claim`。

    ---

    ## 输出格式

    仅输出一个合法 JSON，必须用 ```json 代码块包裹，不得包含额外解释文字。

    ```json
    {
      "subject": "<学科-二级方向>",
      "keywords": ["<关键词/符号/方法 1>", "<关键词/符号/方法 2>", "<...>"],
      "background": "<研究背景、问题定位与动机>",
      "premises": [
        "<不需本文论证的工具/定义/定理/模型 1>",
        "<不需本文论证的工具/定义/定理/模型 2>"
      ],
      "assumptions": [
        {
          "assumption": "<论文提出并依赖本文支撑的假设/近似>",
          "condition": "<该假设的适用边界/Regime（可为 N/A）>",
          "breaks_when": "<何时失效/会出现何种偏差（可为 N/A）>"
        }
      ],
      "evidence": [
        "<关键 evidence / 发现 1 (可为 N/A)>"
      ],
      "justifications": [
        {
          "evidence": "<引用或摘要某条 evidence>",
          "supports": "<它支撑的 assumption/claim>",
          "because": "<为什么能支撑：机理/判据/标度/推导要点>",
          "gap": "<缺口/不确定性（若无填 N/A）>"
        }
      ],
      "reasoning_summary": "<理解性推理脉络摘要：Premises + Assumptions/Conditions + Evidence + Justifications → Conclusion>",
      "conclusion": "<原子化的核心结论/Core Claim>"
    }
    ```

    约束提醒：
    - **分层严格**：不要把 Assumption/Condition 混入 Premises；不要把 Evidence 当作 Justification；不要在 Background/Premises 中提前泄露 Conclusion。
    - **真实性**：严格基于文本；缺口用 `gap` 或 "N/A" 标注，不要编造。
    - **关键词数量**：`keywords` 必须是非空数组，**最多 10 个**，按重要性排序；若超过 10 个，请只保留最关键的 10 个（避免关键词膨胀污染后续上下文）。
    - **关键词选择**：优先覆盖论文的主领域与主任务（核心对象/观测量/模型参数/方法框架/主要实验或观测配置及关键系统效应）；避免把后验验证/统计检验/评估指标等细枝末节写进 keywords，除非它们是论文的核心贡献之一。
    """
)


PAPER_BRIEF_V2_REASONING_EN = dedent(
    """\
    # PaperBrief‑V2 (Reasoning‑First, Justification‑Explicit) — English Prompt

    You are a **Scientific Skeleton Extractor**. Your goal is to reconstruct the paper into a **knowledge skeleton** suitable for **multi-step reasoning problem synthesis**, and explicitly separate:

    - **Evidence**: what the paper *finds/observes* (facts/results)
    - **Justification**: why the paper believes the evidence supports a claim (reasoning/argument)

    Important: treat the **input paper text as the only source of truth**. You may abstract/summarize, but **must not fabricate** facts, numbers, or derivations that do not appear in the text. If a reasoning gap cannot be closed from the text, mark it with `gap`.

    ---

    ## Concept Hierarchy (Brief‑V2 Schema)

    Extract strictly following the hierarchy below; do not mix layers or leak conclusions early:

    1. **Background**
       - Topic, object of study, task, and motivation (what is being discussed / why it matters).
       - Do not include toolbox details, and do not include conclusions.

    2. **Premises / Dependencies**
       - Toolbox the paper treats as given: definitions, symbols, known theorems, standard models/equations, method components.
       - Test: “Does this require proof/validation in the paper?” → if **no**, it is a premise.

    3. **Assumptions + Conditions (validity regime)**
       - Key assumptions/approximations proposed and relied upon by the paper, with validity boundaries and failure modes.
       - Test: “Is this proposed by the paper and supported by evidence/argument?” → if **yes**, it is an assumption.

    4. **Evidence**
       - Observable facts/trends/comparisons from experiments/simulations/derivations (facts only; no “because”).
       - Keep it compact; avoid dumping apparatus details or raw data points.

    5. **Justifications**
       - Explanation layer: why evidence supports a claim/assumption (mechanism, criteria, scaling, key derivation steps, contrast logic, etc.).
       - Provide as atomic items: `evidence → supports → because`; if not closed, fill `gap` (do not invent).

    6. **Reasoning Summary**
       - Minimal narrative: Premises + Assumptions/Conditions + Evidence + Justifications → Conclusion.
       - This is for understanding, not a direct problem template.

    7. **Conclusion (Core Claim)**
       - Atomic, standalone core knowledge statement (avoid leaking it into Background/Premises).

    ---

    ## Input

    [Paper Text]
    {paper}

    ---

    ## Tasks

    1. **Subject anchoring**: identify the paper’s subject area and sub-area (e.g., “Physics — Low-temperature physics / phonon transport”).
    2. **Layered extraction**: extract `background`, `premises`, `assumptions(with condition/breaks_when)`, `evidence`, `justifications`.
    3. **Reasoning summary**: produce `reasoning_summary` and mark gaps/uncertainties if any.
    4. **Core conclusion**: produce an atomic `conclusion` (core claim).

    ---

    ## Output Format

    Output a single valid JSON object, wrapped in a ```json code block, with no extra text.

    ```json
    {
      "subject": "<subject area — sub-area>",
      "keywords": ["<keyword/symbol/method 1>", "<keyword/symbol/method 2>", "<...>"],
      "background": "<background / motivation / problem framing>",
      "premises": [
        "<premise / given tool / definition / standard model 1>",
        "<premise / given tool / definition / standard model 2>"
      ],
      "assumptions": [
        {
          "assumption": "<assumption/approximation proposed and supported by the paper>",
          "condition": "<validity regime (or N/A)>",
          "breaks_when": "<failure mode / when it breaks (or N/A)>"
        }
      ],
      "evidence": [
        "<key evidence / finding 1 (or N/A)>"
      ],
      "justifications": [
        {
          "evidence": "<evidence snippet>",
          "supports": "<the assumption/claim it supports>",
          "because": "<why it supports: mechanism/criterion/scaling/key derivation>",
          "gap": "<uncertainty if any (or N/A)>"
        }
      ],
      "reasoning_summary": "<Premises + Assumptions/Conditions + Evidence + Justifications → Conclusion>",
      "conclusion": "<atomic core claim>"
    }
    ```

    Constraints:
    - **Strict layering**: do not put Assumptions/Conditions into Premises; do not treat Evidence as Justification; do not leak Conclusion in Background/Premises.
    - **Truthfulness**: strictly grounded in the paper; mark gaps with `gap` or N/A; do not fabricate.
    - **Keyword count**: `keywords` must be a non-empty array with **at most 10 items**, ordered by importance; if you have more, keep only the top 10 (avoid keyword bloat that pollutes downstream context).
    - **Keyword selection**: prioritize the paper’s main domain + task (core objects/observables/model parameters/method framework/major experiment or observation setup and key systematics); avoid minor validation/tests/metrics unless they are a core contribution.
    """
)


EXTRACT_V2_REASONING = dedent(
    """\
    # Extract‑V2（Reasoning‑Chain Centric）示例 Prompt

    你扮演"考点策划者"的角色。你不直接出题或给答案，只负责从 Brief‑V2 的分层骨架中抽出**可递进出题的考点链（potential problem chain）**。

    核心原则：**Justification + ReasoningStep 二者皆要**。
    - **ReasoningStep** 构成链的骨架（"怎么推"）
    - **Justification** 是链的血肉（"为什么每步成立"）

    ### 风格转化提示（实验风格 → 推导风格）

    论文的 Justification 往往是**解释型**（为什么观察到的现象能支撑假设，方向：Evidence → Claim），
    而题目需要的往往是**推导型**（给定条件，推出结论，方向：Premises → Conclusion）。

    对于每个考点，请判断并标注 `exam_style`：
    - `explanation`：可直接作为"解释型"题目（判断/分析：为什么 A 支撑 B？）
    - `derivation`：需要转化为"推导型"题目（计算/推导：给定 X，求 Y）

    若为 `derivation`，请在 `transform_hint` 中说明如何翻转方向（例如：把"观察到 X 说明 Y"翻转为"给定 Y 的条件，求 X"）。

    ---

    请遵循递进逻辑：先建立工具箱与符号（Premise），再明确假设与边界（Assumption/Condition），再以 **Justification + ReasoningStep 作为主链节点**，最后用 Boundary 做护栏。必要时可把结论改写成"待求目标"（Goal），而不是背诵点。

    ---

    ## 输入

    [Paper Brief JSON]
    $paper_brief_json

    [Paper Brief Text]
    $paper_brief_text

    ---

    ## 任务：从 Brief‑V2 提炼考点链

    1.  **围绕 Justification + ReasoningStep 构建主链**：提取 2–5 个考点，并为每个考点标注 `layer`：
        *   `Premise`：基础定义/符号体系/已知定理或标准模型（对应 `premises/background`）。
        *   `Assumption`：论文提出的关键假设本身（对应 `assumptions[].assumption`）。
        *   `Condition`：假设的适用边界/Regime（对应 `assumptions[].condition`）。
        *   `Justification`：核心论证节点：为什么 evidence 支撑某个 assumption/claim（对应 `justifications[]`）。
        *   `ReasoningStep`：可拆解的推导步/判据构造/无量纲化/标度关系（可来自 `justifications[].because` 或 `reasoning_summary`）。
        *   `Boundary`：假设失效后的诊断/修正方向（对应 `assumptions[].breaks_when` 或 `justifications[].gap`）。
        *   `Goal`（可选）：把 `conclusion` 改写为“待求目标/可检验命题”，不要把结论当作已知背诵点。

    2.  **构建 exam_points**：每个考点需包含：
        *   `id`：简短编号（如 P1/A1/C1/J1/R1/B1/G1），便于在 chain_potential 中引用（建议填写）。
        *   `layer`：上述层级标签之一。
        *   `point`：考点内容（原子化、可直接出题；写成"要判断/要推出/要诊断什么"，不要写答案）。
        *   `exam_style`：`explanation`（解释型：为什么 A 支撑 B）或 `derivation`（推导型：给定 X 求 Y）。
        *   `transform_hint`：若 `exam_style` 为 `derivation`，说明如何从论文的解释型翻转为推导型（若为 `explanation` 可填 `N/A`）。
        *   `assessment`：考察重点（如 Recall/Derivation/Analysis/Diagnosis）。
        *   `style`：适合的题型（如 MCQ/Calculation/Judgment/Fill-in）。

    3.  **chain_potential**：
        *   用一段话描述考点间的层级递进依赖，并尽量用 `id` 引用。
        *   **严禁**复述结论的具体答案/数值/选项字母；只描述依赖关系与可考察的推理接口。

    ---

    ## 输出格式

    仅输出一个合法 JSON，必须用 ```json 代码块包裹：

    ```json
    {
      "exam_points": [
        {
          "id": "P1",
          "layer": "Premise",
          "point": "<考点描述>",
          "exam_style": "explanation",
          "transform_hint": "N/A",
          "assessment": "<考察能力>",
          "style": "MCQ/Fill-in"
        }
      ],
      "chain_potential": "<描述层级递进依赖，不泄露结论答案>"
    }
    ```

    约束提醒：
    - **层级标签必填**：每个考点必须归类到 `Premise/Assumption/Condition/Justification/ReasoningStep/Boundary/(Goal)` 之一。
    - **风格转化必填**：每个考点必须标注 `exam_style`（explanation/derivation）；若为 derivation，`transform_hint` 必须说明翻转方向。
    - **不越界**：Extract 只负责"圈地与规划"，不负责"答题"。不要在 point 里给出最终数值/最终表达式/选项字母；也不要把 conclusion 当成背诵点。
    - **关注缺口**：若 Brief 标注了 `gap` 或 breaks_when，请转化为 `Boundary/Condition` 类考点，并在 chain_potential 中说明它影响哪一步。
    """
)


EXTRACT_V2_REASONING_EN = dedent(
    """\
    # Extract‑V2 (Reasoning‑Chain Centric) — English Prompt

    You are an **Exam-Point Planner**. You do **not** write questions or answers. You extract a **progressive chain of exam points** (a potential problem chain) from the Brief‑V2 layered skeleton.

    Core principle: both **Justification** and **ReasoningStep** must exist.

    - **ReasoningStep**: the chain’s skeleton (how to derive)
    - **Justification**: the chain’s flesh (why each step holds)

    ### Style conversion hint (experimental → derivational)

    Paper justifications are often **explanatory** (Evidence → Claim: “why X supports Y”),
    while exam problems often want **derivational** direction (Premises → Conclusion: “given Y, derive X”).

    For each exam point, set `exam_style`:

    - `explanation`: can be asked as “why does A support B?”
    - `derivation`: should be converted into “given X, derive Y”

    If `exam_style = derivation`, write `transform_hint` explaining how to flip the direction.

    ---

    Follow a progressive structure: build the toolbox and symbols (Premise), then assumptions and regimes (Assumption/Condition), then connect via **Justification + ReasoningStep** as the main chain, and use Boundary as guardrails. When helpful, rewrite the conclusion as a **Goal** (a testable target), not a memorized statement.

    ---

    ## Input

    [Paper Brief JSON]
    $paper_brief_json

    [Paper Brief Text]
    $paper_brief_text

    ---

    ## Task: Extract an exam-point chain from Brief‑V2

    1. **Build the main chain around Justification + ReasoningStep**: extract 2–5 exam points, and label each with `layer`:
       - `Premise`: definitions/symbols/known theorems/standard models (from `premises/background`).
       - `Assumption`: key assumption itself (from `assumptions[].assumption`).
       - `Condition`: the validity regime (from `assumptions[].condition`).
       - `Justification`: core argument node (from `justifications[]`).
       - `ReasoningStep`: a decomposable derivation step / criterion construction / nondimensionalization / scaling law (from `justifications[].because` or `reasoning_summary`).
       - `Boundary`: failure diagnostics / correction directions (from `assumptions[].breaks_when` or `justifications[].gap`).
       - `Goal` (optional): rewrite `conclusion` into a target statement (do not treat it as given).

    2. **Build `exam_points`**: each item must include:
       - `id`: short identifier (e.g., P1/A1/C1/J1/R1/B1/G1).
       - `layer`: one of the labels above.
       - `point`: atomic exam point phrased as “what to derive/judge/diagnose”, not the answer.
       - `exam_style`: `explanation` or `derivation`.
       - `transform_hint`: for `derivation`, how to flip the direction; else `N/A`.
       - `assessment`: skill type (Recall/Derivation/Analysis/Diagnosis).
       - `style`: suitable question format (MCQ/Calculation/Judgment/Fill-in).

    3. **`chain_potential`**:
       - Describe the dependency / progression in one paragraph, referencing `id`s when possible.
       - Do **not** leak concrete answers, numbers, or MCQ letters; describe only interfaces and dependencies.

    ---

    ## Output Format

    Output a single valid JSON object, wrapped in a ```json code block:

    ```json
    {
      "exam_points": [
        {
          "id": "P1",
          "layer": "Premise",
          "point": "<exam point description>",
          "exam_style": "explanation",
          "transform_hint": "N/A",
          "assessment": "<skill>",
          "style": "MCQ/Fill-in"
        }
      ],
      "chain_potential": "<one-paragraph dependency/progression description (reference ids), no answer leakage>"
    }
    ```
    """
)
