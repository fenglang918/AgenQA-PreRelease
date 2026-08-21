"""SciPedia-specific prompt variants (Python-controlled).

These prompts are used when `data.scipedia_pack.enable=true` and Brief/Extract v2 are enabled.
"""

from __future__ import annotations

from textwrap import dedent

__all__ = [
    "PAPER_BRIEF_V2_REASONING_SCIPEDIA",
    "PAPER_BRIEF_V2_REASONING_SCIPEDIA_EN",
    "EXTRACT_V2_REASONING_SCIPEDIA",
    "EXTRACT_V2_REASONING_SCIPEDIA_EN",
]


PAPER_BRIEF_V2_REASONING_SCIPEDIA_EN = dedent(
    """\
    # PaperBrief‑V2 (SciPedia Pack Aware) — English Prompt

    You are a **Scientific Skeleton Extractor**. Your goal is to reconstruct the input into a **knowledge skeleton** suitable for **multi-step reasoning problem synthesis**, and explicitly separate:

    - **Evidence**: what the text *states/claims/observes* (facts/results)
    - **Justification**: why the text believes the evidence supports a claim (reasoning/argument)

    Important: treat the **input text as the only source of truth**. You may abstract/summarize, but **must not fabricate** facts, numbers, or derivations that do not appear in the text. If a reasoning gap cannot be closed from the text, mark it with `gap`.

    ---

    ## Special note: SciPedia multi-section packs

    Sometimes the input is a **textbook-style SciPedia entry** packed as:
    - `<SECTION name="Key Takeaways"> ... </SECTION>`
    - `<SECTION name="Principles and Mechanisms"> ... </SECTION>`
    - `<SECTION name="Introduction"> ... </SECTION>`

    When you see these tags:
    - Treat **Key Takeaways** as canonical facts/definitions.
    - Use **Principles and Mechanisms** to extract derivational relationships, symbols, and mechanisms.
    - Do not overweight narrative/application sections unless they define key variables or regimes.

    ---

    ## Concept Hierarchy (Brief‑V2 Schema)

    Extract strictly following the hierarchy below; do not mix layers or leak conclusions early:

    1. **Background**
       - Topic, object of study, task, and motivation (what is being discussed / why it matters).
       - Do not include toolbox details, and do not include conclusions.

    2. **Premises / Dependencies**
       - Toolbox the text treats as given: definitions, symbols, known theorems, standard models/equations, method components.
       - Test: “Does this require proof/validation in the text?” → if **no**, it is a premise.

    3. **Assumptions + Conditions (validity regime)**
       - Key assumptions/approximations relied upon by the text, with validity boundaries and failure modes.
       - Test: “Is this proposed/claimed and supported by evidence/argument?” → if **yes**, it is an assumption.

    4. **Evidence**
       - Observable facts/trends/comparisons from explanations/derivations (facts only; no “because”).
       - Keep it compact; avoid dumping irrelevant details.

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

    [Text]
    {paper}

    ---

    ## Tasks

    1. **Subject anchoring**: identify the subject area and sub-area (e.g., “Mathematics — Differential Geometry / Riemannian geometry”).
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
      "background": "<background / motivation / framing>",
      "premises": [
        "<premise / definition / standard model 1>",
        "<premise / definition / standard model 2>"
      ],
      "assumptions": [
        {
          "assumption": "<assumption/approximation>",
          "condition": "<validity regime (or N/A)>",
          "breaks_when": "<failure mode (or N/A)>"
        }
      ],
      "evidence": [
        "<key evidence / finding 1 (or N/A)>"
      ],
      "justifications": [
        {
          "evidence": "<evidence>",
          "supports": "<claim/assumption supported>",
          "because": "<mechanism/criterion/scaling/key step>",
          "gap": "<gap/uncertainty (or N/A)>"
        }
      ],
      "reasoning_summary": "<minimal reasoning: Premises + Assumptions/Conditions + Evidence + Justifications → Conclusion>",
      "conclusion": "<atomic core claim>"
    }
    ```

    Constraints:
    - **Strict layering**: do not mix layers; do not leak Conclusion into Background/Premises.
    - **Faithfulness**: strictly grounded in the text; use `gap` or `N/A` instead of fabrication.
    - **Keyword count**: `keywords` must be a non-empty array with **at most 10 items**, ordered by importance; if you have more, keep only the top 10 (avoid keyword bloat that pollutes downstream context).
    - **Keyword selection**: prioritize the paper’s main domain + task (core objects/observables/model parameters/method framework/major experiment or observation setup and key systematics); avoid minor validation/tests/metrics unless they are a core contribution.
    """
)


EXTRACT_V2_REASONING_SCIPEDIA_EN = dedent(
    """\
    # Extract‑V2 (SciPedia Pack Aware) — English Prompt

    You are an **Exam-Point Planner**. You do **not** write questions or answers. You extract a **progressive chain of exam points** (a potential problem chain) from the Brief‑V2 layered skeleton.

    Core principle: both **Justification** and **ReasoningStep** must exist.

    ---

    ## Special note: SciPedia packs

    The upstream text may come from a **SciPedia multi-section pack** (textbook-style).
    Prefer exam points that are grounded in:
    - **Key Takeaways** (canonical definitions/facts)
    - **Principles and Mechanisms** (derivable relationships/mechanisms/symbols/regimes)

    Avoid over-weighting narrative/application prose unless it introduces key variables, regimes, or constraints.

    ---

    ## Input

    [Paper Brief JSON]
    $paper_brief_json

    [Paper Brief Text]
    $paper_brief_text

    ---

    ## Task: Extract an exam-point chain from Brief‑V2

    1. **Build the main chain around Justification + ReasoningStep**: extract 2–5 exam points, and label each with `layer`:
       - `Premise`: definitions/symbols/known theorems/standard models.
       - `Assumption`: a key assumption itself.
       - `Condition`: the validity regime.
       - `Justification`: core argument node (evidence supports claim because ...).
       - `ReasoningStep`: a decomposable derivation step / criterion construction / scaling law.
       - `Boundary`: failure diagnostics / correction directions.
       - `Goal` (optional): rewrite the conclusion into a testable target.

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


PAPER_BRIEF_V2_REASONING_SCIPEDIA = dedent(
    """\
    # PaperBrief‑V2（SciPedia Pack Aware）示例 Prompt

    你扮演“科学骨架提取器”。你的目标是把输入重建为**适合多步推理题合成的知识骨架**，并显式区分：

    - **Evidence**：文本明确陈述/声称/总结的事实（不写“因为”）
    - **Justification**：文本为何认为这些事实能支撑某个结论/主张（机制/论证/推导要点）

    重要：把**输入文本视为唯一真相源**。允许抽象/总结，但**不得凭空编造**文本中不存在的事实、数值或推导。如果论证链条无法从文本闭合，用 `gap` 标注不确定性。

    ---

    ## 特别说明：SciPedia 多段 Pack 输入

    有时输入是“教材条目（SciPedia）”的多段汇总，形如：
    - `<SECTION name="Key Takeaways"> ... </SECTION>`
    - `<SECTION name="Principles and Mechanisms"> ... </SECTION>`
    - `<SECTION name="Introduction"> ... </SECTION>`

    当你看到这些标签时：
    - 把 **Key Takeaways** 当作权威事实/定义/结论来源；
    - 把 **Principles and Mechanisms** 当作机制与可推导关系（符号、方程、条件）的主要来源；
    - 不要被叙述型/应用型段落带偏，除非它定义了关键变量或适用 regime。

    ---

    ## 概念层级（Brief‑V2 Schema）

    必须严格按下述层级抽取；不要混层、不要提前泄露结论：

    1. **Background**：主题、对象、任务、动机（讲“是什么/为什么重要”），不写工具箱细节与结论。
    2. **Premises / Dependencies**：被当作已知的工具箱（定义、符号、定理、标准模型/方程、方法组件）。
    3. **Assumptions + Conditions**：关键假设/近似、适用条件与失效模式。
    4. **Evidence**：事实/趋势/对比（只写事实，不写“因为”）。
    5. **Justifications**：论证层：evidence 为何支持某个 claim/assumption（机制、判据、标度关系、关键推导步等）；不能闭合就填 `gap`。
    6. **Reasoning Summary**：最短推理叙述：Premises + Assumptions/Conditions + Evidence + Justifications → Conclusion。
    7. **Conclusion**：原子化核心结论（不要泄露到 Background/Premises）。

    ---

    ## 输入

    [文本]
    {paper}

    ---

    ## 任务

    1. **Subject anchoring**：识别学科与二级方向（如“数学—微分几何/黎曼几何”）。
    2. **分层抽取**：输出 `background/premises/assumptions/evidence/justifications`。
    3. **推理总结**：输出 `reasoning_summary`，并标注 gaps。
    4. **核心结论**：输出原子化 `conclusion`。

    ---

    ## 输出格式

    仅输出一个合法 JSON，必须用 ```json 代码块包裹，且不附加额外文本。

    ```json
    {
      "subject": "<学科—二级方向>",
      "keywords": ["<关键词/符号/方法 1>", "<关键词/符号/方法 2>", "<...>"],
      "background": "<背景/动机/问题框架>",
      "premises": ["<前提1>", "<前提2>"],
      "assumptions": [
        {"assumption": "<假设>", "condition": "<适用条件或 N/A>", "breaks_when": "<失效模式或 N/A>"}
      ],
      "evidence": ["<证据/事实1 或 N/A>"],
      "justifications": [
        {"evidence": "<证据>", "supports": "<支持的主张/假设>", "because": "<原因/机制/关键步>", "gap": "<N/A 或 gap>"}
      ],
      "reasoning_summary": "<最短推理：前提+假设+证据+论证→结论>",
      "conclusion": "<原子化核心结论>"
    }
    ```

    约束：
    - **严格分层**：不要混层，不要提前泄露结论。
    - **忠实性**：只基于文本；无法闭合就用 `gap` 或 `N/A`。
    - **关键词数量**：`keywords` 必须是非空数组，**最多 10 个**，按重要性排序；若超过 10 个，请只保留最关键的 10 个（避免关键词膨胀污染后续上下文）。
    - **关键词选择**：优先覆盖论文的主领域与主任务（核心对象/观测量/模型参数/方法框架/主要实验或观测配置及关键系统效应）；避免把后验验证/统计检验/评估指标等细枝末节写进 keywords，除非它们是论文的核心贡献之一。
    """
)


EXTRACT_V2_REASONING_SCIPEDIA = dedent(
    """\
    # Extract‑V2（SciPedia Pack Aware）示例 Prompt

    你扮演“考点策划者”。你不直接出题或给答案，只负责从 Brief‑V2 的分层骨架中抽出**可递进出题的考点链**。

    核心原则：**Justification + ReasoningStep 二者皆要**。

    ---

    ## 特别说明：SciPedia Pack

    上游输入可能来自“教材条目（SciPedia）”的多段汇总（textbook-style）。
    请优先围绕以下来源抽取“可考点”：
    - **Key Takeaways**：权威事实/定义/关键结论
    - **Principles and Mechanisms**：机制、可推导关系、符号体系、适用 regime

    不要被叙述型/应用型段落带偏，除非它引入关键变量/条件/约束。

    ---

    ## 输入

    [Paper Brief JSON]
    $paper_brief_json

    [Paper Brief Text]
    $paper_brief_text

    ---

    ## 任务：从 Brief‑V2 提炼考点链

    1) 围绕 **Justification + ReasoningStep** 构建主链，提取 2–5 个考点并标注 `layer`：
    - `Premise` / `Assumption` / `Condition` / `Justification` / `ReasoningStep` / `Boundary` / `Goal(可选)`

    2) `exam_points` 每项必须包含：
    - `id` / `layer` / `point` / `exam_style` / `transform_hint` / `assessment` / `style`

    3) `chain_potential`：
    - 用一段话描述依赖递进关系，尽量引用 `id`；
    - 严禁泄露答案、数值或选项字母。

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
    """
)
