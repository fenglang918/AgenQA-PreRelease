"""EpisodeSeedBuilder prompt: build contract-defined episode_seed from paper text.

Design:
- Input is the raw paper text (title/abstract/body concatenated by caller).
- A per-experiment Contract is injected: instruction + JSON Schema.
"""

from __future__ import annotations

from textwrap import dedent

__all__ = [
    "EPISODE_SEED_BUILDER_PROMPT",
    "EPISODE_SEED_BUILDER_PROMPT_EN",
    # V1 aliases for vendoring convention
    "EPISODE_SEED_BUILDER_V1",
    "EPISODE_SEED_BUILDER_V1_EN",
]


EPISODE_SEED_BUILDER_PROMPT = dedent(
    """\
    # EpisodeSeedBuilder（从 PaperBrief 生成 episode_seed）

    你将收到论文文本（title/abstract/body 拼接）以及一个 Contract（instruction + JSON Schema）。

    你的目标是：构造一个 **可复用、稳定、信息密度高但不冗余** 的 `episode_seed`，用于后续多步推理题生成的“话题锚定”。

    ## episode_seed 的作用（请按这个意图去写）

    把 episode_seed 理解为“话题契约 / 任务锚点”，它会被后续多个角色反复读取（如 Director / Extend / Diagnose 等），主要用于：
    - **防跑题**：保证整个链条始终围绕同一篇 paper 的主题/对象/任务纵向加深，而不是漂到通用题；
    - **统一语境**：明确“研究对象是什么、核心量/符号是什么、在什么 regime/假设下讨论”，让题面能持续复用同一套语义；
    - **可扩展**：足够具体以支持后续出更难、更深的多步推理题，但又不把论文结论/答案泄露到锚点里。

    episode_seed **不是**论文摘要/实验复述/结论清单；它也不是用来“解释为什么”的推理过程。

    ## 输入

    [Paper Text]
    {paper_text}

    [Contract Instruction]
    {contract_instruction}

    [Output Schema JSON]（你必须严格遵守）
    {output_schema_json}

    ## 输出（仅允许的输出）

    - 仅输出一个合法 JSON，必须用 ```json 代码块包裹，不得输出任何额外文本。
    - 输出结构必须 **严格符合** Output Schema JSON（推荐设置 `additionalProperties: false`）。

    ### 字段要求

    - 所有字段、是否必填、类型约束，均以 Output Schema JSON 为准。
    - `subject` / `keywords` 不具有特殊性：是否出现与类型限制由 schema 决定。
    """
)


EPISODE_SEED_BUILDER_PROMPT_EN = dedent(
    """\
    # EpisodeSeedBuilder (build episode_seed from PaperBrief)

    You will receive the paper text (title/abstract/body concatenated by caller) plus a Contract (instruction + JSON Schema).

    Goal: construct a reusable, stable, information-dense but non-redundant `episode_seed`
    that anchors the topic for downstream multi-step reasoning question generation.

    ## What episode_seed is for (write with this intent)

    Think of episode_seed as a "topic contract / task anchor" that will be read repeatedly by multiple roles
    (e.g., Director / Extend / Diagnose). It is mainly used to:
    - Prevent topic drift: keep the entire chain grounded in the same paper's topic/objects/task (not generic problems);
    - Unify the semantic context: clarify the object of study, core quantities/symbols, and the regime/assumptions;
    - Enable vertical deepening: support generating harder multi-step reasoning questions without leaking conclusions/answers.

    episode_seed is NOT a paper abstract, experimental narration, or a list of final conclusions.

    ## Input

    [Paper Text]
    {paper_text}

    [Contract Instruction]
    {contract_instruction}

    [Output Schema JSON] (must be strictly followed)
    {output_schema_json}

    ## Output (the only allowed output)

    - Output exactly one valid JSON wrapped in a ```json code block. No extra text.
    - The output structure must **strictly conform** to the Output Schema JSON (recommend `additionalProperties: false`).

    ### Field requirements

    - All fields, requiredness, and types are defined by the Output Schema JSON.
    - `subject` / `keywords` are not special: governed entirely by the schema.
    """
)


# Vendoring convention: role module exports <ROLE>_V1(_EN) text.
EPISODE_SEED_BUILDER_V1 = EPISODE_SEED_BUILDER_PROMPT
EPISODE_SEED_BUILDER_V1_EN = EPISODE_SEED_BUILDER_PROMPT_EN
