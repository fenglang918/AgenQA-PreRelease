"""Diagnose role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent

from agenqa.domain.diagnose_schema import (
    diagnose_output_schema_text,
    FIELD_ISSUES,
    FIELD_FIX_SUGGESTIONS,
    FIELD_DIAGNOSIS,
)

__all__ = [
    "DIAGNOSE_V1",
    "DIAGNOSE_V1_TAGGED",
    "DIAGNOSE_REVISE_CORRECTNESS",
    "DIAGNOSE_REVISE_CORRECTNESS_TAGGED",
    "DIAGNOSE_REVISE_WORLD_CONTRACT",
    "DIAGNOSE_REVISE_WORLD_CONTRACT_TAGGED",
    "DIAGNOSE_REVISE_ANSWER_CONTRACT",
    "DIAGNOSE_REVISE_ANSWER_CONTRACT_TAGGED",
    "DIAGNOSE_REVISE_DIFFICULTY",
    "DIAGNOSE_REVISE_DIFFICULTY_TAGGED",
    "DIAGNOSE_REVISE_REUSE_HIDDEN",
    "DIAGNOSE_REVISE_REUSE_HIDDEN_TAGGED",
    "DIAGNOSE_V1_EN",
    "DIAGNOSE_V1_TAGGED_EN",
    "DIAGNOSE_REVISE_CORRECTNESS_EN",
    "DIAGNOSE_REVISE_CORRECTNESS_TAGGED_EN",
    "DIAGNOSE_REVISE_WORLD_CONTRACT_EN",
    "DIAGNOSE_REVISE_WORLD_CONTRACT_TAGGED_EN",
    "DIAGNOSE_REVISE_ANSWER_CONTRACT_EN",
    "DIAGNOSE_REVISE_ANSWER_CONTRACT_TAGGED_EN",
    "DIAGNOSE_REVISE_DIFFICULTY_EN",
    "DIAGNOSE_REVISE_DIFFICULTY_TAGGED_EN",
    "DIAGNOSE_REVISE_REUSE_HIDDEN_EN",
    "DIAGNOSE_REVISE_REUSE_HIDDEN_TAGGED_EN",
    "get_diagnose_prompt",
]


# 基础部分（不包含任务描述）
BASE_BODY_NO_TASK = dedent(
    f"""\
    # Diagnose（题目诊断算子说明，Revise 专用）

    你扮演"题目体检医生"的角色，仅负责分析问题，而不直接重写题目。

    ---

    ## 输入说明

    [KnownTree JSON]
    $known_0

    说明：在 Revise Diagnose 场景下，上述 JSON 通常是一个“DiagnoseView”（用于对齐 extend 的 draft_chain 视角）：
    - episode_seed / premise_bank：话题锚定与前提集合（premise_bank 为 < step 的条目）。
    - fact_window / step_certs_window：最近 window 步（< step）的可复用结论与推理证书。
    - current_step：当前待修订 step 的旧条目（premise_delta / fact_delta / step_cert），用于定位问题根因。
    - 该视图已剥离 provenance 等元数据，请不要依赖 provenance/raw_ref。

    [Premise Summary (optional)]
    $background

    [Current Question]
    $question

    [Current Answer]
    $answer

    [Solver Feedback]
    $solver_feedback

    [Solver Answers]
    $solver_answers

    [Solver Reasoning]
    $solver_reasoning

    [Director Notes]
    $director_notes

    ---
    """
)

# English base (no task)
BASE_BODY_NO_TASK_EN = dedent(
    """\
    # Diagnose (Problem Diagnosis; Revise only)

    You are the "problem doctor". Your job is to analyze issues, **not** to rewrite the problem directly.

    All output text must be in English.

    ---

    ## Inputs

    [KnownTree JSON]
    $known_0

    Note: In Revise Diagnose, the JSON above is typically a "DiagnoseView" aligned with the draft_chain view:
    - episode_seed / premise_bank: topic anchoring and premises (premise_bank contains only < step entries).
    - fact_window / step_certs_window: reusable facts and step certificates from the recent window steps (< step).
    - current_step: existing entries of the step being revised (premise_delta / fact_delta / step_cert) for root-cause analysis.
    - Provenance metadata is stripped; do not rely on provenance/raw_ref.

    [Premise Summary (optional)]
    $background

    [Current Question]
    $question

    [Current Answer]
    $answer

    [Solver Feedback]
    $solver_feedback

    [Solver Answers]
    $solver_answers

    [Solver Reasoning]
    $solver_reasoning

    [Director Notes]
    $director_notes

    ---
    """
)

# 通用任务描述（用于 DIAGNOSE_V1）
TASK_GENERAL = dedent(
    """\

    ## 任务：诊断当前题目的主要问题

    重点包括但不限于：
    1. 正确性问题（条件矛盾、答案错误、推导缺失等）；
    2. 可解性与信息充分性（是否缺少关键条件、单位/口径不明确等）；
    3. 场景与考点匹配（是否仍然锚定 episode_seed 的 anchor（及可选主题/关键词），是否与 premise_bank 自洽）；
    4. 表达与难度（是否存在歧义表述、难度明显不合理等）。

    注意：`[KnownTree JSON]` 在 Revise Diagnose 中通常为 DiagnoseView：包含 episode_seed / premise_bank / fact_window / step_certs_window / current_step。
    - premise_bank 为 < step 的前提集合；fact_window/step_certs_window 为最近 window 步（< step）的前序结论与证书。
    - current_step 提供当前待修订 step 的旧条目（premise_delta/fact_delta/step_cert），用于定位本步问题。
    - 若 `[Premise Summary]` 为空请以 JSON 为准。
"""
)

# General task (EN)
TASK_GENERAL_EN = dedent(
    """\

    ## Task: diagnose the main issues of the current problem

    Focus includes but is not limited to:
    1) correctness issues (contradictions, wrong answer, missing derivation, etc.);
    2) solvability and information sufficiency (missing conditions, unclear conventions/units, etc.);
    3) grounding and consistency (anchored in episode_seed; consistent with premise_bank);
    4) expression and difficulty (ambiguity, unreasonable difficulty, etc.).

    Note: In Revise Diagnose, [KnownTree JSON] is typically a DiagnoseView containing:
    - episode_seed / premise_bank / fact_window / step_certs_window / current_step.
    - premise_bank contains < step premises; fact_window/step_certs_window are prior window-step items (< step).
    - current_step provides existing entries for the step being revised (premise_delta/fact_delta/step_cert).
    If [Premise Summary] is empty, rely on the JSON.
    """
)


# Correctness 模式特定的任务描述
TASK_CORRECTNESS = dedent(
    """\

    ## 任务：诊断当前题目的正确性问题

    你当前处于 **Correctness 修订模式**，需要专注于判断这道题本身在**逻辑自洽性与正确性**上的问题，而不是重新出题。

    请在综合 `[KnownTree JSON]`、`[Premise Summary]`、`[Current Question]`、`[Current Answer]`（标准解）、`[Solver Feedback]`、`[Solver Answers]` 的基础上，自主完成分析，重点关注但不限于：
    - 题干与前提（premise_bank）、基本物理/数学规律之间是否自洽，是否存在明显的条件矛盾或信息缺失；
    - 标准答案是否与合理推导一致（包括量纲/单位/符号/等价表达等），以及是否可以从题干条件出发重建出一条合理推理路径；
    - 推理过程中是否隐含不合理或未声明的关键假设；
    - Solver 的反馈和作答中暴露出的错误模式，可作为辅助证据，但不要机械照抄。

    **诊断输出要求**：
    - 明确指出主要错误类型（如条件矛盾/答案错误/推导缺失/信息不足等）；
    - 说明问题大致出现在题干、解答过程还是最终答案；
    - 提出可操作的修复思路或建议（不要求给出完整新题面）。
"""
)

# Correctness mode (EN)
TASK_CORRECTNESS_EN = dedent(
    """\

    ## Task: diagnose correctness / well-posedness issues

    You are in **Correctness revise mode**. Focus on logical consistency and correctness, not on making the question harder.

    Using [KnownTree JSON], [Premise Summary], [Current Question], [Current Answer] (reference), and solver signals, analyze issues such as:
    - contradictions or missing information;
    - whether the reference answer is consistent with a reasonable derivation (dimensions/units/notation/equivalent forms);
    - hidden critical assumptions that are not stated;
    - solver failure modes as supporting evidence (do not copy mechanically).

    Output requirements:
    - clearly name the main issue types (contradiction / missing info / wrong answer / non-unique / severe ambiguity, etc.);
    - indicate whether the issue is in the question, reasoning path, or the final answer;
    - propose actionable fix suggestions (no need to fully rewrite).
    """
)


# World-Contract 模式特定的任务描述（Type1：语义世界观歧义）
TASK_WORLD_CONTRACT = dedent(
    """\

    ## 任务：诊断当前题目的语义世界观歧义（Type 1）

    你当前处于 **World-Contract 修订模式**。你的目标不是“算对”，而是判断这道题是否存在**语义世界观不唯一**的问题，
    并给出“如何把语义钉死”的可执行建议。

    关注点（L1-first）：
    - L1（Paradigm Choice）是否未选边：题面是否混用了多个合法范式/流派的术语或规则（例如 process clock vs version vector）。
    - L3（Question-Specific Rules）是否未钉死：题面是否缺少关键的题目特定规则，导致多套自洽解释。
    - L2（范式内默认）只在 L1 scoped 下成立：如果 L1 不明确，不要用“跨范式默认”强行补齐。
    - 术语一致性：若你选择了某个 L1 范式，题面中的术语/定义是否与之冲突（应当作为“冲突”问题指出）。

    **诊断输出要求**：
    - 明确指出主要歧义属于哪一层（优先判定 L1 或 L3 underdetermined），以及典型的分歧轴；
    - 给出可操作的修复建议：如何补充/修正独立 `world_contract` 字段来钉死 L1/L3；下游会把 `Question + World Contract` 一起交给 solver，不要默认要求把整块重复塞回题面；
    - 若语义钉死会影响现有参考答案（GT），请明确指出“需要同步更新 Answer/GT”的原因与范围（不必给出完整新答案）。
    """
)

# World-Contract mode (EN)
TASK_WORLD_CONTRACT_EN = dedent(
    """\

    ## Task: diagnose semantic world-view ambiguity (Type 1)

    You are in **World-Contract revise mode**. Your goal is NOT to re-solve the math, but to determine whether the question
    has **non-unique semantic world assumptions** (multiple self-consistent interpretations), and propose actionable fixes
    to make the semantics unique.

    Focus (L1-first):
    - L1 (Paradigm Choice) underdetermined: mixed terminology/rules across paradigms (e.g., process clock vs version vector).
    - L3 (Question-Specific Rules) underdetermined: missing question-specific rules that cause multiple interpretations.
    - L2 (paradigm defaults) are only valid after L1 is chosen; do not apply cross-paradigm defaults.
    - Terminology consistency: if you pick an L1 paradigm, check for conflicts in terminology/definitions and report them.

    Output requirements:
    - explicitly state which layer is underdetermined (prefer L1 or L3) and the key ambiguity axes;
    - propose concrete fixes: update the separate `world_contract` field to pin down L1/L3; downstream will deliver `Question + World Contract` together to the solver, so do not default to duplicating the whole block inside the question body;
    - if pinning semantics invalidates the current reference answer (GT), state that Answer/GT must be updated and why (no need to fully rewrite).
    """
)


# Answer-Contract 模式特定的任务描述（Type2：作答协议/格式口径歧义）
TASK_ANSWER_CONTRACT = dedent(
    """\

    ## 任务：诊断当前题目的作答协议/判对口径歧义（Type 2）

    你当前处于 **Answer-Contract 修订模式**。你的目标不是改变题目的语义世界观（Type1/L1-L3），
    而是让这道题的**作答要求与判对口径**足够明确、可审计、可稳定评测。

    重点关注但不限于：
    - exact vs approx：题面是否明确允许近似？若允许，是否明确近似口径（如允许的近似层级/范围）；
    - numeric：容差（abs_tol/rel_tol）、有效数字、单位/量纲与单位策略是否明确；
    - derivation/symbolic：等价类与允许的等价变换范围是否明确（例如是否允许代数等价、分段/反三角等）；
    - branch：若存在多分支/多解，题面是否声明“允许分支”，以及是否要求给出分支描述/覆盖范围；
    - MCQ：是否保证唯一正确选项；是否存在题面或参考答案导致多选项合理的情况；
    - 输出协议：Answer 的格式是否唯一可解析（例如必须 `\\boxed{...}`；或 JSON schema 明确等）。

    **诊断输出要求**：
    - 明确指出缺失/冲突的“判对口径”要点（Type2），并说明风险（会导致 judge/评测漂移）；
    - 给出可操作的修复建议：如何补充/修正独立 contract 的 L4 / answer-output spec；若 Question 已经足够表达题意，不要再把完整 “Answer Requirements” 块重复抄回题面；
    - 若你认为参考答案（GT）需要同步调整，请明确说明原因（通常是“口径澄清后旧 GT 不再可判/不再匹配”），但不要重写 Type1 语义。
    """
)

# Answer-Contract mode (EN)
TASK_ANSWER_CONTRACT_EN = dedent(
    """\

    ## Task: diagnose answer-protocol / judging-contract ambiguity (Type 2)

    You are in **Answer-Contract revise mode**. Your goal is NOT to change the semantic world view (Type1/L1-L3),
    but to make the **answer requirements and judging contract** explicit, auditable, and stable for evaluation.

    Focus includes:
    - exact vs approx: whether approximation is allowed; if yes, what approximation regime/requirements apply;
    - numeric: tolerances (abs/rel), sig figs, units and unit policy;
    - symbolic/derivation: what equivalences are allowed (e.g., algebraic equivalence scope, piecewise, inverse trig);
    - branches: whether multiple branches/solutions are allowed and whether branch descriptions are required;
    - MCQ: uniqueness of the correct option (avoid multiple plausible options);
    - output protocol: make the final Answer uniquely parseable (e.g., required `\\boxed{...}` or explicit JSON schema).

    Output requirements:
    - explicitly list missing/conflicting contract items and why they cause judge/eval drift;
    - propose minimal, concrete fixes in the separate L4 / answer-output contract layer; if the question body already states the problem clearly, do not ask to duplicate the full “Answer Requirements / Answer Contract” block there;
    - if the reference answer (GT) must change due to the clarified contract, state why (do not rewrite Type1 semantics).
    """
)


# Difficulty 模式特定的任务描述
TASK_DIFFICULTY = dedent(
    """\

    ## 任务：诊断当前题目的难度与结构问题

    你当前处于 **Difficulty 修订模式**，需要专注于判断这道题在**难度与结构**上是否存在“太平/太机械/缺乏区分度”等问题，而不是改写结论本身。

    请在综合 `[KnownTree JSON]`、`[Premise Summary]`、`[Current Question]`、`[Current Answer]`（标准解）、`[Solver Feedback]`、`[Solver Answers]` 以及 `[Director Notes]` 中的 solver 状态信息的基础上，自主完成分析，重点关注但不限于：
    - 当前题目是否几乎不用推理、主要依赖记忆或直接代公式，缺乏必要的推理台阶和结构；
    - strong / medium 等 solver 的表现是否说明题目过易、过难或缺乏区分度（例如都轻松做对，或者表现高度一致）；
    - 是否存在“机械变体”（仅改参数/表述，本质结构与已有题目高度重合），没有提供新的思考空间；
    - 在保持结论不变的前提下，是否可以通过增加中间推理步骤、引入更合理的约束或更有辨识度的选项来提升质量。
    - 若 `[Director Notes]` 明确要求换方向/避免某类变体，请优先将其视为质量问题的一部分：给出可行的改写方向；若不建议遵循，也要说明理由与权衡。

    **诊断输出要求**：
    - 明确指出主要的难度/结构问题类型（如太容易、太机械、缺乏区分度等）；
    - 说明当前难度与期望难度的大致差距与原因；
    - 给出可操作的加难/重构方向（不要求给出完整新题面，但需要说明调整思路）。
"""
)

TASK_REUSE_HIDDEN = dedent(
    """\

    ## 任务：诊断“复用与隐藏”（reuse_hidden）问题

    你当前处于 **reuse_hidden 修订模式**。目标是确保：从 `premise_bank` 推导到本步答案时，合理/标准的推导路径 **必须依赖** 上一步的 key_fact（或其等价中间结果）作为必要推理台阶，但题面中 **不得显式引用或泄露** 该结论（head-tail 视角下 fact_bank/step_certs 不可见；episode_seed 也不对 solver 可见）。

    请在综合 `[KnownTree JSON]`、`[Premise Summary]`、`[Current Question]`、`[Current Answer]`（标准解）、`[Solver Feedback]`、`[Solver Answers]` 以及 `[Director Notes]` 的基础上，检查以下 **必须避免** 的问题（任一出现都会降低 head-tail 题目的推理深度与难度）：

    1. **显式历史引用**：出现“上一步/如前所述/根据 step/见前文”等指针式表达，暴露了本应由 solver 自己推导的中间步骤或推理路径线索；
    2. **链路递进不足**：从 `premise_bank` 推导到答案的过程中，可以绕过上一步的 key_fact（不需要将其作为推理的必经中间步骤），说明链条未真正延伸；
    3. **结论泄露**：把上一步的答案/结论直接写入题干、选项或已知条件（而不是让 solver 通过推理重新得出）。

    **输出要求**：
    - 明确指出存在哪些问题（若无问题则说明未发现 reuse_hidden 问题）；
    - 若有问题，给出可操作的修复方向：如何移除显式引用或泄露，同时保持题目仍需通过相同推理路径（复用上一步关键结论作为中间台阶）才能求解；避免建议“把上一步结论写成已知条件/前提”这类会导致泄露的做法。
"""
)

# Difficulty mode (EN)
TASK_DIFFICULTY_EN = dedent(
    """\

    ## Task: diagnose difficulty / structure issues

    You are in **Difficulty revise mode**. Focus on whether the problem is too easy / too mechanical / not discriminative, not on changing the conclusion.

    Using [KnownTree JSON], [Premise Summary], [Current Question], [Current Answer] (reference), solver signals, and [Director Notes], analyze:
    - whether it is trivial, plug-and-chug, or memory-based;
    - whether solver performance indicates too easy / too hard / low discrimination;
    - whether it is a mechanical variant of earlier steps;
    - how to improve quality by adding reasoning steps, better constraints, or better distractors (MCQ) while keeping the conclusion equivalent.
    - if [Director Notes] explicitly asks to change direction / avoid a type of variant, treat it as a primary quality signal; propose feasible directions, or explain why it is not advisable.

    Output requirements:
    - name the main difficulty/structure issue types (too easy, too mechanical, low discrimination, etc.);
    - explain the gap vs desired difficulty and why;
    - provide actionable directions (no need to fully rewrite).
    """
)

TASK_REUSE_HIDDEN_EN = dedent(
    """\

    ## Task: diagnose reuse+hide issues (reuse_hidden)

    You are in **reuse_hidden revise mode**. The goal is to ensure: a reasonable/canonical derivation of this step’s answer from `premise_bank` **must depend on** the previous step’s key_fact (or an equivalent intermediate result) as a necessary intermediate step, but the question text **must not explicitly reference or leak** that conclusion (fact_bank/step_certs are hidden in head-tail view; episode_seed is also not solver-visible).

    Using [KnownTree JSON], [Premise Summary], [Current Question], [Current Answer] (reference), solver signals, and [Director Notes], check the following **issues to avoid** (any of them will reduce reasoning depth/difficulty):

    1. **Explicit history references**: phrases like "previous step", "as shown above", "from step X" that expose intermediate steps or reasoning-path hints the solver should derive independently;
    2. **Weak progression**: the derivation from `premise_bank` to the answer can bypass the previous key_fact (does not require it as a necessary intermediate step), indicating the chain is not truly extending;
    3. **Conclusion leakage**: the previous answer/conclusion is directly stated in the question/options/givens (rather than requiring re-derivation).

    Output requirements:
    - name which issues exist (or state if none);
    - if issues exist, provide actionable repair directions: remove explicit references/leakage while keeping the problem solvable only through the same canonical reasoning path (reusing the prior conclusion as an intermediate step); avoid suggesting to restate the prior conclusion as new givens/premises.
    """
)


# 输出格式部分（JSON）
_OUTPUT_FORMAT_JSON = dedent(
    f"""\

    ---

    ## 输出格式

    仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段严格为：
{diagnose_output_schema_text()}
    """
)

# 输出格式部分（Tagged）
_OUTPUT_FORMAT_TAGGED = dedent(
    f"""\

    ---

    ## 输出格式（带字段标记的纯文本）

    当被明确要求使用"tagged 协议"时，请不要输出 JSON。
    仅输出一段纯文本，使用以下字段标签包裹内容（标签独占一行）：

    [{FIELD_ISSUES}]
    - 问题1
    - 问题2
    [/{FIELD_ISSUES}]

    [{FIELD_FIX_SUGGESTIONS}]
    - 修复建议1
    - 修复建议2
    [/{FIELD_FIX_SUGGESTIONS}]

    [{FIELD_DIAGNOSIS}]
    （总体诊断总结）
    [/{FIELD_DIAGNOSIS}]

    字段名必须严格使用上述英文标识，便于解析。
    """
)

# Output format (Tagged, EN)
_OUTPUT_FORMAT_TAGGED_EN = dedent(
    f"""\

    ---

    ## Output format (tagged plain text)

    When explicitly required to use the "tagged" protocol, do not output JSON.
    Output a plain text block with the following field tags (tag on its own line):

    [{FIELD_ISSUES}]
    - issue 1
    - issue 2
    [/{FIELD_ISSUES}]

    [{FIELD_FIX_SUGGESTIONS}]
    - fix suggestion 1
    - fix suggestion 2
    [/{FIELD_FIX_SUGGESTIONS}]

    [{FIELD_DIAGNOSIS}]
    (overall diagnosis summary)
    [/{FIELD_DIAGNOSIS}]

    Field names must match exactly for parsing.
    """
)


def _get_output_format(protocol: str | None = None) -> str:
    """根据协议返回对应的输出格式部分。"""
    if protocol and protocol.strip().lower() == "tagged":
        return _OUTPUT_FORMAT_TAGGED
    return _OUTPUT_FORMAT_JSON


def _get_output_format_en(protocol: str | None = None) -> str:
    if protocol and protocol.strip().lower() == "tagged":
        return _OUTPUT_FORMAT_TAGGED_EN
    return _OUTPUT_FORMAT_JSON


# 通用版本（包含通用任务描述）
DIAGNOSE_V1 = dedent(BASE_BODY_NO_TASK + TASK_GENERAL + _OUTPUT_FORMAT_JSON)
DIAGNOSE_V1_TAGGED = dedent(BASE_BODY_NO_TASK + TASK_GENERAL + _OUTPUT_FORMAT_TAGGED)

DIAGNOSE_V1_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_GENERAL_EN + _OUTPUT_FORMAT_JSON)
DIAGNOSE_V1_TAGGED_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_GENERAL_EN + _OUTPUT_FORMAT_TAGGED_EN)


# Correctness 模式的 Diagnose Prompt（基础版本，默认 JSON）
DIAGNOSE_REVISE_CORRECTNESS = dedent(BASE_BODY_NO_TASK + TASK_CORRECTNESS + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_CORRECTNESS_TAGGED = dedent(BASE_BODY_NO_TASK + TASK_CORRECTNESS + _OUTPUT_FORMAT_TAGGED)

DIAGNOSE_REVISE_CORRECTNESS_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_CORRECTNESS_EN + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_CORRECTNESS_TAGGED_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_CORRECTNESS_EN + _OUTPUT_FORMAT_TAGGED_EN)

# World-Contract 模式的 Diagnose Prompt（基础版本，默认 JSON）
DIAGNOSE_REVISE_WORLD_CONTRACT = dedent(BASE_BODY_NO_TASK + TASK_WORLD_CONTRACT + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_WORLD_CONTRACT_TAGGED = dedent(BASE_BODY_NO_TASK + TASK_WORLD_CONTRACT + _OUTPUT_FORMAT_TAGGED)

DIAGNOSE_REVISE_WORLD_CONTRACT_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_WORLD_CONTRACT_EN + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_WORLD_CONTRACT_TAGGED_EN = dedent(
    BASE_BODY_NO_TASK_EN + TASK_WORLD_CONTRACT_EN + _OUTPUT_FORMAT_TAGGED_EN
)

# Answer-Contract 模式的 Diagnose Prompt（基础版本，默认 JSON）
DIAGNOSE_REVISE_ANSWER_CONTRACT = dedent(BASE_BODY_NO_TASK + TASK_ANSWER_CONTRACT + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_ANSWER_CONTRACT_TAGGED = dedent(BASE_BODY_NO_TASK + TASK_ANSWER_CONTRACT + _OUTPUT_FORMAT_TAGGED)

DIAGNOSE_REVISE_ANSWER_CONTRACT_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_ANSWER_CONTRACT_EN + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_ANSWER_CONTRACT_TAGGED_EN = dedent(
    BASE_BODY_NO_TASK_EN + TASK_ANSWER_CONTRACT_EN + _OUTPUT_FORMAT_TAGGED_EN
)

# Difficulty 模式的 Diagnose Prompt（基础版本，默认 JSON）
DIAGNOSE_REVISE_DIFFICULTY = dedent(BASE_BODY_NO_TASK + TASK_DIFFICULTY + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_DIFFICULTY_TAGGED = dedent(BASE_BODY_NO_TASK + TASK_DIFFICULTY + _OUTPUT_FORMAT_TAGGED)

DIAGNOSE_REVISE_DIFFICULTY_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_DIFFICULTY_EN + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_DIFFICULTY_TAGGED_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_DIFFICULTY_EN + _OUTPUT_FORMAT_TAGGED_EN)

# reuse_hidden 模式的 Diagnose Prompt（基础版本，默认 JSON）
DIAGNOSE_REVISE_REUSE_HIDDEN = dedent(BASE_BODY_NO_TASK + TASK_REUSE_HIDDEN + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_REUSE_HIDDEN_TAGGED = dedent(BASE_BODY_NO_TASK + TASK_REUSE_HIDDEN + _OUTPUT_FORMAT_TAGGED)

DIAGNOSE_REVISE_REUSE_HIDDEN_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_REUSE_HIDDEN_EN + _OUTPUT_FORMAT_JSON)
DIAGNOSE_REVISE_REUSE_HIDDEN_TAGGED_EN = dedent(BASE_BODY_NO_TASK_EN + TASK_REUSE_HIDDEN_EN + _OUTPUT_FORMAT_TAGGED_EN)


def get_diagnose_prompt(revise_mode: str | None = None, protocol: str | None = None) -> str:
    """根据 revise_mode 和 protocol 动态获取对应的 Diagnose prompt。

    Args:
        revise_mode: "correctness" | "world_contract" | "answer_contract" | "reuse_hidden" | None（使用通用版本；兼容：difficulty -> reuse_hidden）
        protocol: "tagged" | "json" | None（默认 JSON）

    Returns:
        完整的 prompt 文本
    """
    output_format = _get_output_format(protocol)

    mode = str(revise_mode or "").strip().lower()
    if mode in {"correct", "correctness", "fix", "fix_answer"}:
        return dedent(BASE_BODY_NO_TASK + TASK_CORRECTNESS + output_format)
    if mode in {"world_contract", "worldcontract", "world-contract"}:
        return dedent(BASE_BODY_NO_TASK + TASK_WORLD_CONTRACT + output_format)
    if mode in {"answer_contract", "answercontract", "answer-contract"}:
        return dedent(BASE_BODY_NO_TASK + TASK_ANSWER_CONTRACT + output_format)
    if mode in {"reuse_hidden", "reuse-hidden", "reuse", "progression", "hidden_reuse", "difficulty", "hardness", "complexity"}:
        return dedent(BASE_BODY_NO_TASK + TASK_REUSE_HIDDEN + output_format)
    else:
        # 通用版本
        return dedent(BASE_BODY_NO_TASK + TASK_GENERAL + output_format)
