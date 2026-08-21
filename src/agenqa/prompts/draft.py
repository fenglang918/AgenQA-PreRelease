"""Draft role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent, indent

from agenqa.domain.draft_schema import (
    draft_output_schema_text,
    FIELD_DRAFT_QUESTION_EXPLICIT,
    FIELD_DRAFT_QUESTION,
    FIELD_DRAFT_SOLUTION,
    FIELD_DRAFT_ANSWER,
    FIELD_GROUNDING_CHECK,
    FIELD_DRAFT_BACKGROUND,
    FIELD_REUSED_CONCLUSIONS,
    FIELD_REUSED_REFS,
)
from .common import (
    COMMON_ANSWER_SCHEMA,
    COMMON_ANSWER_SCHEMA_EN,
    COMMON_QUESTION_TYPES,
    COMMON_QUESTION_TYPES_EN,
    COMMON_DRAFT_GROUNDING_CHECK,
    COMMON_DRAFT_GROUNDING_CHECK_EN,
    COMMON_DRAFT_INPUT_DESC,
    COMMON_DRAFT_INPUT_DESC_EN,
    COMMON_DRAFT_QUESTION_TYPE_SECTION,
    COMMON_DRAFT_QUESTION_TYPE_SECTION_EN,
    COMMON_DRAFT_ROLE_DESC,
    COMMON_DRAFT_ROLE_DESC_EN,
    COMMON_FIRST_STEP_REUSE_RULES,
    COMMON_FIRST_STEP_REUSE_RULES_EN,
    COMMON_EDGE_QA_VS_PATH,
    COMMON_EDGE_QA_VS_PATH_EN,
    COMMON_DRAFT_BACKGROUND_RULES,
    COMMON_DRAFT_BACKGROUND_RULES_EN,
    COMMON_REUSED_CONCLUSIONS_RULES,
    COMMON_REUSED_CONCLUSIONS_RULES_EN,
    COMMON_REUSED_REFS_RULES,
    COMMON_REUSED_REFS_RULES_EN,
)

__all__ = [
    "DRAFT_V1",
    "DRAFT_V1_TAGGED",
    "DRAFT_V1_FIRST_STEP",
    "DRAFT_V1_TAGGED_FIRST_STEP",
    "DRAFT_REVISE_CORRECTNESS",
    "DRAFT_REVISE_CORRECTNESS_TAGGED",
    "DRAFT_REVISE_DIFFICULTY",
    "DRAFT_REVISE_DIFFICULTY_TAGGED",
    "DRAFT_REVISE_REUSE_HIDDEN",
    "DRAFT_REVISE_REUSE_HIDDEN_TAGGED",
    "DRAFT_V1_EN",
    "DRAFT_V1_TAGGED_EN",
    "DRAFT_V1_FIRST_STEP_EN",
    "DRAFT_V1_TAGGED_FIRST_STEP_EN",
    "DRAFT_REVISE_CORRECTNESS_EN",
    "DRAFT_REVISE_CORRECTNESS_TAGGED_EN",
    "DRAFT_REVISE_DIFFICULTY_EN",
    "DRAFT_REVISE_DIFFICULTY_TAGGED_EN",
    "DRAFT_REVISE_REUSE_HIDDEN_EN",
    "DRAFT_REVISE_REUSE_HIDDEN_TAGGED_EN",
    "get_draft_revise_prompt",
]


BASE_BODY = dedent(
    f"""\
    # Draft（出题草稿撰写者）

    ## Role
    构思下一道题的核心内容。

    ---

{COMMON_DRAFT_INPUT_DESC.text}

    本轮目标步数：i = $prev_step + 1，建议题型：$question_type

    ---

    ## Output 字段含义

{COMMON_DRAFT_ROLE_DESC.text}

    ---

    ## 必要 Context

    ### 领域概念
{indent(COMMON_EDGE_QA_VS_PATH.text, "    ")}

    ### 协议约定
{indent(COMMON_REUSED_CONCLUSIONS_RULES.text, "    ")}
{indent(COMMON_REUSED_REFS_RULES.text, "    ")}
{indent(COMMON_DRAFT_BACKGROUND_RULES.text, "    ")}

    ### 设计约束
    - 题目扎根于 `known_0` 的科学背景，新增设定与既有 `background` 自洽。
    - 本题与上一步在推理模式上有明显区别（引入新维度，而非仅换参数）。

{COMMON_DRAFT_QUESTION_TYPE_SECTION.text}
{COMMON_DRAFT_GROUNDING_CHECK.text}
"""
)


DRAFT_V1 = dedent(
    BASE_BODY
    + f"""

---

## 输出格式

仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段严格为：
{dedent(draft_output_schema_text())}
"""
)


DRAFT_V1_TAGGED = dedent(
    BASE_BODY
    + f"""

---

## 输出格式（带字段标记的纯文本）

当被明确要求使用"tagged 协议"时，请不要输出 JSON。
仅输出一段纯文本，使用以下字段标签包裹内容（标签独占一行）：

[{FIELD_DRAFT_QUESTION_EXPLICIT}]
（显式版本题面：可以写出对前序结论的引用）
[{"/"+FIELD_DRAFT_QUESTION_EXPLICIT}]

[{FIELD_DRAFT_QUESTION}]
（隐藏版本题面：不透露前序结论，实际输出给 solver）
[{"/"+FIELD_DRAFT_QUESTION}]

[{FIELD_DRAFT_SOLUTION}]
（解题思路草稿，S1/S2/...）
[{"/"+FIELD_DRAFT_SOLUTION}]

[{FIELD_DRAFT_ANSWER}]
（答案草稿）
[{"/"+FIELD_DRAFT_ANSWER}]

[{FIELD_DRAFT_BACKGROUND}]
- 新增设定 1
- 新增设定 2
[{"/"+FIELD_DRAFT_BACKGROUND}]

[{FIELD_REUSED_CONCLUSIONS}]
- 计划复用的前序结论 1
- 计划复用的前序结论 2
[{"/"+FIELD_REUSED_CONCLUSIONS}]

[{FIELD_REUSED_REFS}]
[{{"source_step": $prev_step, "mcq_choice": "A"}}]
[{"/"+FIELD_REUSED_REFS}]

[{FIELD_GROUNDING_CHECK}]
（对场景锚定的自检说明）
[{"/"+FIELD_GROUNDING_CHECK}]

其中字段名必须严格使用上述英文标识，便于解析。
"""
)

# 首题模式专用 prompt（prev_step = 0 或 history 为空）
FIRST_STEP_SPECIAL_RULES = dedent(
    f"""\

---

## 首题模式（prev_step = 0）

**设计约束**：
- 答案需组合至少两个独立关系推导得出，不能直接抄 `known_0` 某一行。
- 允许适度引入 `{FIELD_DRAFT_BACKGROUND}`。

{COMMON_FIRST_STEP_REUSE_RULES.text}
"""
)

# 首题模式：替换复用前序结论部分
BASE_BODY_FIRST_STEP = dedent(
    f"""\
    # Draft（首题模式）

    ## Role
    构思首题的核心内容。当前为首题模式，`history` 为空。

    ---

    ## Input
    - `known_0`：整条链共享的初始科学背景
    - `background`：已累积的物理设定（首题时通常为空或仅含初始设定）
    - `director_notes`：Director 的指引

    本轮目标步数：i = $prev_step + 1，建议题型：$question_type

    ---

    ## Output 字段含义
    - `{FIELD_DRAFT_QUESTION}`：题面草稿
    - `{FIELD_DRAFT_SOLUTION}`：解题思路（S1/S2/…）
    - `{FIELD_DRAFT_ANSWER}`：预期答案
    - `{FIELD_DRAFT_BACKGROUND}`：新增前提（默认空）
    - `{FIELD_REUSED_CONCLUSIONS}`：首题无前序结论
    - `{FIELD_REUSED_REFS}`：首题为空数组 `[]`
    - `{FIELD_GROUNDING_CHECK}`：场景锚定自检

    ---

    ## 必要 Context

{indent(COMMON_DRAFT_BACKGROUND_RULES.text, "    ")}

    ### 设计约束
    - 题目扎根于 `known_0` 的科学背景。
    - 新增设定与既有 `background` 自洽。

{COMMON_DRAFT_QUESTION_TYPE_SECTION.text}
{COMMON_DRAFT_GROUNDING_CHECK.text}
"""
)


DRAFT_V1_FIRST_STEP = dedent(
    BASE_BODY_FIRST_STEP
    + FIRST_STEP_SPECIAL_RULES
    + f"""

---

## 输出格式

仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段严格为：
{dedent(draft_output_schema_text())}
"""
)


DRAFT_V1_TAGGED_FIRST_STEP = dedent(
    BASE_BODY_FIRST_STEP
    + FIRST_STEP_SPECIAL_RULES
    + f"""

---

## 输出格式（带字段标记的纯文本）

当被明确要求使用"tagged 协议"时，请不要输出 JSON。
仅输出一段纯文本，使用以下字段标签包裹内容（标签独占一行）：

[{FIELD_DRAFT_QUESTION}]
（题面草稿）
[{"/"+FIELD_DRAFT_QUESTION}]

[{FIELD_DRAFT_SOLUTION}]
（解题思路草稿，S1/S2/...）
[{"/"+FIELD_DRAFT_SOLUTION}]

[{FIELD_DRAFT_ANSWER}]
（答案草稿）
[{"/"+FIELD_DRAFT_ANSWER}]

[{FIELD_DRAFT_BACKGROUND}]
- 新增设定 1
- 新增设定 2
[{"/"+FIELD_DRAFT_BACKGROUND}]

[{FIELD_REUSED_CONCLUSIONS}]
- 本题为首题，无前序结论可复用
[{"/"+FIELD_REUSED_CONCLUSIONS}]

[{FIELD_REUSED_REFS}]
[]
[{"/"+FIELD_REUSED_REFS}]

[{FIELD_GROUNDING_CHECK}]
（对场景锚定的自检说明）
[{"/"+FIELD_GROUNDING_CHECK}]

其中字段名必须严格使用上述英文标识，便于解析。
"""
)


# === Revise 模式专用的基础 Body ===
BASE_BODY_REVISE = dedent(
    f"""\
    # Draft（Revise 模式）

    ## Role
    根据诊断结果修订当前题目。

    ---

    ## Input
    - `known_full`：当前 Known JSON
    - `director_notes`：诊断结果与修订指引

    当前处于 Revise 模式，修订 step=i=$prev_step，上一步 step=i-1。

    ---

    ## Output 字段含义
{COMMON_DRAFT_ROLE_DESC.text}

    ---

    ## 协议约定
{indent(COMMON_REUSED_CONCLUSIONS_RULES.text, "    ")}
{indent(COMMON_REUSED_REFS_RULES.text, "    ")}

    ---
    """
)


# Correctness 模式的修订任务说明
REVISE_TASK_CORRECTNESS = dedent(
    f"""\

    ## Correctness 修订模式

    **目标**：修正正确性问题（条件矛盾/答案错误/推导缺失），不刻意提升难度。

    **设计约束**：
    - 可修改 Question/Answer 使逻辑自洽。
    - 新增设定与既有 `background` 自洽。
    - 保持场景与核心考点不变。

{indent(COMMON_DRAFT_BACKGROUND_RULES.text, "    ")}
{COMMON_DRAFT_QUESTION_TYPE_SECTION.text}
{COMMON_DRAFT_GROUNDING_CHECK.text}
"""
)


# Difficulty 模式的修订任务说明
REVISE_TASK_DIFFICULTY = dedent(
    f"""\

    ## Difficulty 修订模式

    **目标**：在保持 Answer 等价的前提下调节难度/结构。

    **设计约束**：
    - Answer 必须与原 Answer 等价。
    - 可增加推理步骤、引入新中间量、改善 MCQ 干扰项。
    - 新增设定与既有 `background` 自洽。

{indent(COMMON_DRAFT_BACKGROUND_RULES.text, "    ")}
{COMMON_DRAFT_QUESTION_TYPE_SECTION.text}
{COMMON_DRAFT_GROUNDING_CHECK.text}
"""
)

REVISE_TASK_REUSE_HIDDEN = dedent(
    f"""\

    ## reuse_hidden 修订模式（复用与隐藏）

    **目标**：确保本步在 Now QA（edge QA）视角下，合理/标准解法 **必须经过** 上一步的关键结论（或其等价中间结果）这一推理台阶；保持 Answer 与原 Answer 等价（默认不变）。

    **重要澄清**：当前系统的 Now Head-Tail QA 使用“路径折叠题面（Q_fold）”评测；因此本模式不需要、也不应该强行把本步题面改写成“删除 history 后仍可解”的 head-tail 自足题。

    **设计约束**：
    - `draft_question_explicit`：可显式写出对上一步结论的引用，用于核对“确实复用了什么”。
    - `draft_question`（隐藏版，给 solver）：对应 Now QA 的 solver 输入，可依赖 `fact_bank/step_certs(<t)` 复用前序结论；允许用自然语言显式表达依赖关系（如“使用上一步的结论…”），但不要出现内部实现指针（history/fact_bank/step_certs/premise_bank/known_0）。
    - 避免用复制粘贴前序答案来“补齐题干自足”（容易冗余或不一致，也会让链路递进坍塌）；更推荐通过结构化复用引用（`{FIELD_REUSED_REFS}`）+ 题面中轻量说明来表达依赖。
    - `{FIELD_REUSED_REFS}` 必须至少包含 `source_step=i-1`（上一步为 MCQ 时带 `mcq_choice` 并与上一步 Answer 一致）。

{indent(COMMON_DRAFT_BACKGROUND_RULES.text, "    ")}
{COMMON_DRAFT_QUESTION_TYPE_SECTION.text}
{COMMON_DRAFT_GROUNDING_CHECK.text}
"""
)


# 输出格式部分（JSON，用于 Revise 模式）
_DRAFT_REVISE_OUTPUT_FORMAT_JSON = f"""

---

## 输出格式

仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段严格为：
{dedent(draft_output_schema_text())}
"""

# 输出格式部分（Tagged，用于 Revise 模式）
_DRAFT_REVISE_OUTPUT_FORMAT_TAGGED = f"""

---

## 输出格式（带字段标记的纯文本）

当被明确要求使用"tagged 协议"时，请不要输出 JSON。
仅输出一段纯文本，使用以下字段标签包裹内容（标签独占一行）：

[{FIELD_DRAFT_QUESTION_EXPLICIT}]
（显式版本题面：可以写出对前序结论的引用）
[{"/"+FIELD_DRAFT_QUESTION_EXPLICIT}]

[{FIELD_DRAFT_QUESTION}]
（隐藏版本题面：不透露前序结论，实际输出给 solver）
[{"/"+FIELD_DRAFT_QUESTION}]

[{FIELD_DRAFT_SOLUTION}]
（解题思路草稿，S1/S2/...）
[{"/"+FIELD_DRAFT_SOLUTION}]

[{FIELD_DRAFT_ANSWER}]
（答案草稿）
[{"/"+FIELD_DRAFT_ANSWER}]

[{FIELD_DRAFT_BACKGROUND}]
- 新增设定 1
- 新增设定 2
[{"/"+FIELD_DRAFT_BACKGROUND}]

[{FIELD_REUSED_CONCLUSIONS}]
- 计划复用的前序结论 1
- 计划复用的前序结论 2
[{"/"+FIELD_REUSED_CONCLUSIONS}]

（重要：Revise 模式下当前 step=i=$prev_step，因此 `{FIELD_REUSED_REFS}` 必须至少包含 `source_step=i-1` 的引用；若上一步为 MCQ，则 `mcq_choice` 必须与上一步 Answer 的字母一致。）
[{FIELD_REUSED_REFS}]
[{{"source_step": 0, "mcq_choice": "A"}}]
[{"/"+FIELD_REUSED_REFS}]

[{FIELD_GROUNDING_CHECK}]
（对场景锚定的自检说明）
[{"/"+FIELD_GROUNDING_CHECK}]

其中字段名必须严格使用上述英文标识，便于解析。
"""


def _get_draft_revise_output_format(protocol: str | None = None) -> str:
    """根据协议返回对应的输出格式部分（用于 Revise 模式）。"""
    if protocol and protocol.strip().lower() == "tagged":
        return _DRAFT_REVISE_OUTPUT_FORMAT_TAGGED
    return _DRAFT_REVISE_OUTPUT_FORMAT_JSON


# Correctness 模式的 Draft Prompt（基础版本，默认 JSON）
DRAFT_REVISE_CORRECTNESS = dedent(BASE_BODY_REVISE + REVISE_TASK_CORRECTNESS + _DRAFT_REVISE_OUTPUT_FORMAT_JSON)
DRAFT_REVISE_CORRECTNESS_TAGGED = dedent(BASE_BODY_REVISE + REVISE_TASK_CORRECTNESS + _DRAFT_REVISE_OUTPUT_FORMAT_TAGGED)

# Difficulty 模式的 Draft Prompt（基础版本，默认 JSON）
DRAFT_REVISE_DIFFICULTY = dedent(BASE_BODY_REVISE + REVISE_TASK_DIFFICULTY + _DRAFT_REVISE_OUTPUT_FORMAT_JSON)
DRAFT_REVISE_DIFFICULTY_TAGGED = dedent(BASE_BODY_REVISE + REVISE_TASK_DIFFICULTY + _DRAFT_REVISE_OUTPUT_FORMAT_TAGGED)

# reuse_hidden 模式的 Draft Prompt（基础版本，默认 JSON）
DRAFT_REVISE_REUSE_HIDDEN = dedent(BASE_BODY_REVISE + REVISE_TASK_REUSE_HIDDEN + _DRAFT_REVISE_OUTPUT_FORMAT_JSON)
DRAFT_REVISE_REUSE_HIDDEN_TAGGED = dedent(BASE_BODY_REVISE + REVISE_TASK_REUSE_HIDDEN + _DRAFT_REVISE_OUTPUT_FORMAT_TAGGED)


def get_draft_revise_prompt(revise_mode: str, protocol: str | None = None) -> str:
    """根据 revise_mode 和 protocol 动态获取对应的 Draft Revise prompt。

    Args:
        revise_mode: "correctness" | "reuse_hidden"（兼容：difficulty -> reuse_hidden）
        protocol: "tagged" | "json" | None（默认 JSON）

    Returns:
        完整的 prompt 文本
    """
    output_format = _get_draft_revise_output_format(protocol)

    mode = str(revise_mode or "").strip().lower()
    if mode == "correctness":
        return dedent(BASE_BODY_REVISE + REVISE_TASK_CORRECTNESS + output_format)
    if mode in {"reuse_hidden", "reuse-hidden", "reuse", "progression", "hidden_reuse", "difficulty", "hardness", "complexity"}:
        return dedent(BASE_BODY_REVISE + REVISE_TASK_REUSE_HIDDEN + output_format)
    raise ValueError(f"Invalid revise_mode: {revise_mode}, expected 'correctness' or 'reuse_hidden'")


# =========================
# English variants (lang=en)
# =========================

BASE_BODY_EN = dedent(
    f"""\
    # Draft (Problem Ideation; shared skeleton for Extend/Revise/QA‑Init)

    You are the "draft writer". Given the current background and history context, draft the next-step problem content:
{COMMON_DRAFT_ROLE_DESC_EN.text}

    All output text must be in English.

    ---

{COMMON_DRAFT_INPUT_DESC_EN.text}

    Target step is i = $prev_step + 1.

    ---

    ## Task: draft the next-step problem

    1) Grounding
       - The problem must be rooted in the scientific/engineering context of `known_0` (physics/chemistry/biology/materials, etc.).
       - Mathematical tools are allowed, but must serve the original scientific problem.
       - If `known_0` were removed, the problem should not degrade into an unrelated pure-math exercise.

    2) Respect accumulated Background
       - Read `background` in Known carefully; do not contradict established assumptions/model family.
       - If a change is truly needed, explicitly state and justify it in `{FIELD_DRAFT_BACKGROUND}`.

	    3) Edge QA vs Path QA (critical)
{indent(COMMON_EDGE_QA_VS_PATH_EN.text, "    ")}

	    4) Reuse / progression (internal metadata; do not leak)
	       - Hidden `draft_question` must be head-tail self-contained: no history-pointer phrases and no prior-step answer leakage; encode reuse only via `reused_conclusions` / `reused_refs`.
{indent(COMMON_REUSED_CONCLUSIONS_RULES_EN.text, "    ")}
{indent(COMMON_REUSED_REFS_RULES_EN.text, "    ")}

	    5) Draft background candidates (prevent leakage)
{indent(COMMON_DRAFT_BACKGROUND_RULES_EN.text, "    ")}

	    6) Question type and answer format
	       - Suggested QuestionType: $question_type (empty means MCQ by default).
{COMMON_QUESTION_TYPES_EN.text}
{COMMON_ANSWER_SCHEMA_EN.text}
{COMMON_DRAFT_GROUNDING_CHECK_EN.text}
"""
)


DRAFT_V1_EN = dedent(
    BASE_BODY_EN
    + f"""\n\n---\n\n## Output format\n\nOutput one JSON object only, wrapped in a ```json code block. Top-level keys must be exactly:\n{dedent(draft_output_schema_text(lang="en"))}\n"""
)


DRAFT_V1_TAGGED_EN = dedent(
    BASE_BODY_EN
    + f"""\n\n---\n\n## Output format (tagged plain text)\n\nWhen explicitly required to use the \"tagged\" protocol, do not output JSON.\nOutput a plain text block with the following field tags (tag on its own line):\n\n[{FIELD_DRAFT_QUESTION_EXPLICIT}]\n(explicit version: may reference prior conclusions, for verification)\n[{"/"+FIELD_DRAFT_QUESTION_EXPLICIT}]\n\n[{FIELD_DRAFT_QUESTION}]\n(hidden version: no prior result leakage, actual solver input)\n[{"/"+FIELD_DRAFT_QUESTION}]\n\n[{FIELD_DRAFT_SOLUTION}]\n(draft solution outline; S1/S2/...)\n[{"/"+FIELD_DRAFT_SOLUTION}]\n\n[{FIELD_DRAFT_ANSWER}]\n(draft answer)\n[{"/"+FIELD_DRAFT_ANSWER}]\n\n[{FIELD_DRAFT_BACKGROUND}]\n- new premise 1\n- new premise 2\n[{"/"+FIELD_DRAFT_BACKGROUND}]\n\n[{FIELD_REUSED_CONCLUSIONS}]\n- reused prior conclusion 1\n- reused prior conclusion 2\n[{"/"+FIELD_REUSED_CONCLUSIONS}]\n\n[{FIELD_REUSED_REFS}]\n[{{\"source_step\": $prev_step, \"mcq_choice\": \"A\"}}]\n[{"/"+FIELD_REUSED_REFS}]\n\n[{FIELD_GROUNDING_CHECK}]\n(grounding self-check)\n[{"/"+FIELD_GROUNDING_CHECK}]\n\nField names must match exactly for parsing.\n"""
)


FIRST_STEP_SPECIAL_RULES_EN = dedent(
    f"""\

---

## First-step special constraints (prev_step = 0 or history is empty)

**Important: this is the first step; there is no prior QA to reuse.**

- **Core constraint**: do NOT turn a single sentence from `known_0` or `background` into a trivial MCQ/Derivation/Numeric.
- Must satisfy:
  1) the problem must combine **at least two independent** definitions/equations/relations to reach the answer;
  2) the answer must not be directly copyable from a single line of `known_0`/`background`; it must require a short derivation (substitution, limits, scaling comparison, conservation, etc.);
  3) if you find the answer is just a restatement of a background sentence, discard and redesign.

{COMMON_FIRST_STEP_REUSE_RULES_EN.text}
"""
)


BASE_BODY_FIRST_STEP_EN = dedent(
    f"""\
    # Draft (First step; no prior history)

    You are the "draft writer". Draft the first problem after KnownInit. There is no prior QA in history.

    All output text must be in English.

    ---

{COMMON_DRAFT_INPUT_DESC_EN.text}

    Target step is i = $prev_step + 1.

    ---

    ## Task

    - The problem must be grounded in `known_0`, but must not be a trivial rewrite of a single background sentence.
    - Introduce only necessary premises in `{FIELD_DRAFT_BACKGROUND}`; do not leak the answer.

{indent(COMMON_DRAFT_BACKGROUND_RULES_EN.text, "    ")}
{COMMON_QUESTION_TYPES_EN.text}
{COMMON_ANSWER_SCHEMA_EN.text}
"""
)


DRAFT_V1_FIRST_STEP_EN = dedent(
    BASE_BODY_FIRST_STEP_EN
    + FIRST_STEP_SPECIAL_RULES_EN
    + f"""\n\n---\n\n## Output format\n\nOutput one JSON object only, wrapped in a ```json code block. Top-level keys must be exactly:\n{dedent(draft_output_schema_text(lang="en"))}\n"""
)


DRAFT_V1_TAGGED_FIRST_STEP_EN = dedent(
    BASE_BODY_FIRST_STEP_EN
    + FIRST_STEP_SPECIAL_RULES_EN
    + f"""\n\n---\n\n## Output format (tagged plain text)\n\nWhen explicitly required to use the \"tagged\" protocol, do not output JSON.\nOutput a plain text block with the following field tags (tag on its own line):\n\n[{FIELD_DRAFT_QUESTION}]\n(draft question)\n[{"/"+FIELD_DRAFT_QUESTION}]\n\n[{FIELD_DRAFT_SOLUTION}]\n(draft solution outline; S1/S2/...)\n[{"/"+FIELD_DRAFT_SOLUTION}]\n\n[{FIELD_DRAFT_ANSWER}]\n(draft answer)\n[{"/"+FIELD_DRAFT_ANSWER}]\n\n[{FIELD_DRAFT_BACKGROUND}]\n- new premise 1\n- new premise 2\n[{"/"+FIELD_DRAFT_BACKGROUND}]\n\n[{FIELD_REUSED_CONCLUSIONS}]\n(first step; no prior conclusions)\n[{"/"+FIELD_REUSED_CONCLUSIONS}]\n\n[{FIELD_REUSED_REFS}]\n[]\n[{"/"+FIELD_REUSED_REFS}]\n\n[{FIELD_GROUNDING_CHECK}]\n(grounding self-check)\n[{"/"+FIELD_GROUNDING_CHECK}]\n\nField names must match exactly for parsing.\n"""
)


BASE_BODY_REVISE_EN = dedent(
    f"""\
    # Draft (Revise; produce a corrected/improved draft)

    You are revising the current step i=$prev_step based on diagnostic notes. Your job is to produce an improved Draft:
{COMMON_DRAFT_ROLE_DESC_EN.text}

    All output text must be in English.

    ---

    ## Inputs

    Step = $prev_step

    [Known (structured JSON)]
    $known_full

    [Known_0 (quick reference)]
    $known_0

    [Director Notes / Diagnose Summary]
    $director_notes

    ---

	    ## Global constraints

	    - Keep the scientific grounding in `known_0` and stay consistent with the accumulated `background`.
	    - The hidden `draft_question` must be path-fold self-contained: no history-pointer phrases and no prior-step answer leakage; encode reuse only via `reused_conclusions` / `reused_refs`.
{COMMON_EDGE_QA_VS_PATH_EN.text}

{indent(COMMON_DRAFT_BACKGROUND_RULES_EN.text, "    ")}

{COMMON_DRAFT_QUESTION_TYPE_SECTION_EN.text}
	    """
)


_DRAFT_REVISE_OUTPUT_FORMAT_JSON_EN = f"""\n\n---\n\n## Output format\n\nOutput one JSON object only, wrapped in a ```json code block. Top-level keys must be exactly:\n{dedent(draft_output_schema_text(lang="en"))}\n"""


_DRAFT_REVISE_OUTPUT_FORMAT_TAGGED_EN = f"""\n\n---\n\n## Output format (tagged plain text)\n\nWhen explicitly required to use the "tagged" protocol, do not output JSON.\nOutput a plain text block with the following field tags (tag on its own line):\n\n[{FIELD_DRAFT_QUESTION_EXPLICIT}]\n(explicit version: may reference prior conclusions, for verification)\n[{"/"+FIELD_DRAFT_QUESTION_EXPLICIT}]\n\n[{FIELD_DRAFT_QUESTION}]\n(hidden version: no prior result leakage, actual solver input)\n[{"/"+FIELD_DRAFT_QUESTION}]\n\n[{FIELD_DRAFT_SOLUTION}]\n(draft solution outline; S1/S2/...)\n[{"/"+FIELD_DRAFT_SOLUTION}]\n\n[{FIELD_DRAFT_ANSWER}]\n(draft answer)\n[{"/"+FIELD_DRAFT_ANSWER}]\n\n[{FIELD_DRAFT_BACKGROUND}]\n- new premise 1\n- new premise 2\n[{"/"+FIELD_DRAFT_BACKGROUND}]\n\n[{FIELD_REUSED_CONCLUSIONS}]\n- reused prior conclusion 1\n- reused prior conclusion 2\n[{"/"+FIELD_REUSED_CONCLUSIONS}]\n\n[{FIELD_REUSED_REFS}]\n[{{\"source_step\": 0, \"mcq_choice\": \"A\"}}]\n[{"/"+FIELD_REUSED_REFS}]\n\n[{FIELD_GROUNDING_CHECK}]\n(grounding self-check)\n[{"/"+FIELD_GROUNDING_CHECK}]\n\nField names must match exactly for parsing.\n"""


REVISE_TASK_CORRECTNESS_EN = dedent(
    """\

    ## Task: revise for correctness (keep the intent)

    You are in **Correctness revise mode**. Focus on fixing logical consistency / missing conditions / wrong answer issues. Do not try to make the problem harder by default.

    - Ensure the problem is well-posed (consistent givens, sufficient information, unambiguous conventions).
    - If you must introduce additional premises to make it solvable, put them into `draft_background` (premises only; no derived conclusions).
    - Ensure the resulting `draft_solution` is a complete, checkable reasoning outline, and `draft_answer` matches it.
    """
)


REVISE_TASK_DIFFICULTY_EN = dedent(
    """\

    ## Task: revise for difficulty/structure (keep the conclusion equivalent)

    You are in **Difficulty revise mode**. You must keep the conclusion/answer **equivalent** to the original step’s intent, while improving difficulty/structure.

    - Default: do NOT change the conclusion; only allow equivalent reformulations.
    - Improve quality by adding a reasoning step, introducing a more discriminative constraint, or improving MCQ distractors.
    - Explain what was changed in `draft_background` / `reused_conclusions` (as internal notes), but do not leak them in the Question.
    """
)

REVISE_TASK_REUSE_HIDDEN_EN = dedent(
    """\

    ## Task: revise for reuse+hide (keep the Answer equivalent)

    You are in **reuse_hidden revise mode**. The goal is: a reasonable/canonical derivation of this step’s Answer from `known_0 + background` must go through the previous step’s key conclusion (or an equivalent intermediate result), but the solver-visible question must not explicitly reference or leak that conclusion (no "previous step"/"as above" pointers). Keep the Answer equivalent (default unchanged).

    - `draft_question_explicit`: may explicitly reference the reused prior conclusion for verification.
    - `draft_question`: must be head-tail self-contained; avoid history-pointer phrases; do not state (or smuggle as new givens/definitions/`draft_background`) the prior conclusion or its equivalent; instead structure the task/constraints so that the canonical solution naturally and unavoidably re-derives an equivalent intermediate result before reaching the Answer.
    - Anti-pattern to avoid: making the hidden question “self-contained” by directly stating the reused prior conclusion (e.g., “after some algebra/integration, one finds …”) when that statement matches a prior step’s key_fact. That is still leakage and usually collapses difficulty. Instead, give only pre-reuse premises and phrase the needed intermediate result as something the solver must derive within this same question.
    - `reused_refs` must include source_step=i-1 (and for MCQ, mcq_choice matches the previous answer).
    """
)


DRAFT_REVISE_CORRECTNESS_EN = dedent(BASE_BODY_REVISE_EN + REVISE_TASK_CORRECTNESS_EN + _DRAFT_REVISE_OUTPUT_FORMAT_JSON_EN)
DRAFT_REVISE_CORRECTNESS_TAGGED_EN = dedent(BASE_BODY_REVISE_EN + REVISE_TASK_CORRECTNESS_EN + _DRAFT_REVISE_OUTPUT_FORMAT_TAGGED_EN)

DRAFT_REVISE_DIFFICULTY_EN = dedent(BASE_BODY_REVISE_EN + REVISE_TASK_DIFFICULTY_EN + _DRAFT_REVISE_OUTPUT_FORMAT_JSON_EN)
DRAFT_REVISE_DIFFICULTY_TAGGED_EN = dedent(BASE_BODY_REVISE_EN + REVISE_TASK_DIFFICULTY_EN + _DRAFT_REVISE_OUTPUT_FORMAT_TAGGED_EN)

DRAFT_REVISE_REUSE_HIDDEN_EN = dedent(BASE_BODY_REVISE_EN + REVISE_TASK_REUSE_HIDDEN_EN + _DRAFT_REVISE_OUTPUT_FORMAT_JSON_EN)
DRAFT_REVISE_REUSE_HIDDEN_TAGGED_EN = dedent(BASE_BODY_REVISE_EN + REVISE_TASK_REUSE_HIDDEN_EN + _DRAFT_REVISE_OUTPUT_FORMAT_TAGGED_EN)
