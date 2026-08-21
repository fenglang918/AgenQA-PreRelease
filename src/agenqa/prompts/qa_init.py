"""QA-Init prompt (Python style)."""

from __future__ import annotations
from textwrap import dedent

from ._base import PromptSection, PromptTemplate
from .common import (
    COMMON_KNOWN_TREE,
    COMMON_ANSWER_SCHEMA,
    COMMON_ANSWER_SCHEMA_EN,
    COMMON_QUESTION_TYPES,
    COMMON_QUESTION_TYPES_EN,
    COMMON_SOLUTION_SCHEMA,
    COMMON_SOLUTION_SCHEMA_EN,
    COMMON_STYLE_SECTION,
    COMMON_STYLE_SECTION_EN,
)

__all__ = [
    "QA_INIT_V2",
    "QAI_TEMPLATE",
    "QA_INIT_V2_EN",
    "QAI_TEMPLATE_EN",
    "PAPER_BRIEF_PROMPT",
    "PAPER_BRIEF_PROMPT_EN",
]

# 论文内容独立 section，{paper} 占位符仅在此处出现一次
_QAI_PAPER_INPUT = PromptSection(
    text=dedent(
        """\
        ## 论文摘要

        {paper}
        """
    )
)

_QAI_TASK = PromptSection(
    text=dedent(
        """\
        ## 任务目标

        根据下方提供的论文仅生成一道题。输出题干、已知条件、提问与答案，保证唯一、可机判。
        """
    )
)

# QA Init 特有的 Output Schema (Step=0, history=[])
_QAI_OUTPUT_SCHEMA = PromptSection(
    text=dedent(
        """\
        ## 仅允许的输出

        - 仅输出一个合法 JSON，必须用 ```json 代码块包裹，不得包含额外解释或第二个 JSON。
        - 顶层键且仅有：Step, Subject, Known, Question, Solution, Answer。
        - Step 必须为 0。
        - Subject 为字符串，表示论文所属学科及二级方向（如"数学-概率与统计"）。
        - Known 为对象：{"known_0": "...纯文本...", "history": []}，其中 history 首轮必须为空数组。
        - Solution 为结构化多步求解说明（详见"Solution 规范"）。
        - Answer 必须使用 LaTeX \\boxed{…} 包裹。
        """
    )
)

_QAI_PROCESS = PromptSection(
    text=dedent(
        """\
        ## 出题过程（不出现在输出中）

        - 先识别论文所属学科及相关二级方向（如：数学-概率与统计 / 物理-光学 / 生物-分子与细胞 等）。
        - 找出该论文主要解决的难题与所用方法。
        - 围绕"学科-难题-方法"构建题目，**首题必须紧贴下方论文的核心内容**，不得忽略论文本身而单独生成与论文无关的通用数学/概率/逻辑题。
        - Question 必须**显式使用 Known.known_0 中出现的至少 1 个核心符号、量或概念**，并在题面中以相同或等价记号出现；仅在论文本身是纯数学论文时，才允许完全抽象到纯数学情形。
        - 如需在首题中引入"辅助的数学推理"（例如简单计算或概率子问题），必须嵌入论文的物理/工程场景中；题目整体语境仍应是论文描述的研究对象，而不是脱离场景的独立题库题。
        """
    )
)

_QAI_FIELDS_HEADER = PromptSection(
    text=dedent(
        """\
        ## 字段规范

        - Subject：
          - 使用“学科-二级方向”的表述，如 "数学-概率与统计"；需与题目设定一致且可唯一确定。
          - 若论文本身属于某一自然科学/工程领域（如物理、化学、材料、计算机等），Subject 应优先反映该学科，例如 "物理-低温物理与声学"；**仅当原论文本身是一篇纯数学论文时，才使用“数学-…” 作为首题 Subject**。
        - Known.known_0：
          - 给出完整、闭卷自洽的场景与相关假设、必要常数或近似；为构成多步推理的综合性难题做铺垫；不得依赖“见图/表/本文/如下”等外部指代。
        - Question：
          - 本题强制采用 **MCQ (单选题)** 模式。
        """
    )
)

_QAI_FIELDS_QUESTION_SUFFIX = PromptSection(
    text=dedent(
        """\
          - 务必严谨设计选项，避免“以上皆是/以上皆非”等含糊选项。
        """
    )
)

_QAI_SELF_CHECK = PromptSection(
    text=dedent(
        """\
        ## 自检清单

        - 是否已按照要求生成**选择题**：至少 4 个互斥选项，题干含“仅给出选项字母”提示，并在 Question 中给出完整选项列表。
        - 若选项涉及数值/百分比，题面是否说明近似与小数位口径，且最终 Answer 仅给出选项字母。
        - Known.history 是否为 []；顶层键是否仅为 Step/Subject/Known/Question/Answer；Step 是否为 0。
        - Answer 是否使用 \\boxed{…}；若为百分比，是否已转为小数形式（去掉 %）。
        - Subject 是否为“学科-二级方向”风格的非空字符串。
        - 是否避免外部指代（见图/表/本文/如下）并给出必要常数与近似。
        - Solution 是否为规范化多步求解（S1/S2/…），至少包含 2 个清晰步骤，逻辑连贯且可复核。
        - 检查 Solution 最后一行的结论与 Answer 中 \\boxed{…} 的内容是否完全一致；若不一致，以 Solution 的结论为准重写 Answer。
        - 若"删除 Known.known_0 中与论文相关的名词/符号后，当前 Question 仍然可以作为一题完全独立的通用题存在"，则说明题目已脱离论文背景，需要重写使其重新锚定论文的场景与量。

        ## 结构示例

        ```json
        {"Step": 0, "Subject": "学科-二级方向", "Known": {"known_0": "完整场景与假设...", "history": []}, "Question": "题干文本（含选项 A/B/C/D...）", "Solution": "S1: ...; S2: ...", "Answer": "\\\\boxed{A}"}
        ```
        """
    )
)

# 组合 Template
QAI_TEMPLATE = PromptTemplate(
    name="qa_init_v2_mod",
    sections=[
        _QAI_TASK,
        _QAI_OUTPUT_SCHEMA,
        _QAI_PROCESS,

        # Fields Breakdown
        _QAI_FIELDS_HEADER,
        COMMON_QUESTION_TYPES, # Insert Question Types
        _QAI_FIELDS_QUESTION_SUFFIX,

        PromptSection(text="- Solution（结构化解法，针对当前 head–tail 场景）："),
        COMMON_SOLUTION_SCHEMA, # Insert Solution Schema

        PromptSection(text="- Answer："),
        COMMON_ANSWER_SCHEMA,   # Insert Answer Schema

        PromptSection(text="## 出题风格"),
        COMMON_STYLE_SECTION,   # Use Common Style

        _QAI_SELF_CHECK,
        _QAI_PAPER_INPUT,  # 论文内容放在最后，{paper} 仅此一处
    ],
)

QA_INIT_V2 = QAI_TEMPLATE.render_body({})

# -------------------------
# English variants (lang=en)
# -------------------------

_QAI_PAPER_INPUT_EN = PromptSection(
    text=dedent(
        """\
        ## Paper

        {paper}
        """
    )
)

_QAI_TASK_EN = PromptSection(
    text=dedent(
        """\
        ## Goal

        Based on the paper content below, generate exactly **one** problem. Output the problem statement and a uniquely checkable answer.
        All output text must be in English.
        """
    )
)

_QAI_OUTPUT_SCHEMA_EN = PromptSection(
    text=dedent(
        """\
        ## Output contract (only allowed output)

        - Output one valid JSON only, wrapped in a ```json code block. No extra text and no second JSON.
        - Top-level keys must be exactly: Step, Subject, Known, Question, Solution, Answer.
        - Step must be 0.
        - Subject is a string like "Physics-Fluid Mechanics", "Mathematics-Probability and Statistics".
        - Known is an object: {"known_0": "...plain text...", "history": []}. For the first problem, history must be an empty array.
        - Solution must be a structured multi-step explanation (see "Solution rules").
        - Answer must be wrapped in LaTeX \\boxed{...}.
        """
    )
)

_QAI_PROCESS_EN = PromptSection(
    text=dedent(
        """\
        ## Internal process (do not include in output)

        - Identify the paper's discipline and sub-area.
        - Identify the core problem and method.
        - Build the first question tightly grounded in the paper's content; do not ignore the paper and write a generic unrelated question.
        - The Question must explicitly use at least one key symbol/quantity/concept that appears in Known.known_0 (or in the paper).
        - Any auxiliary math must remain embedded in the paper’s scientific/engineering scenario; do not drift into pure-math trivia.
        """
    )
)

_QAI_FIELDS_HEADER_EN = PromptSection(
    text=dedent(
        """\
        ## Field rules

        - Subject:
          - Use a "Discipline-Subarea" string consistent with the problem.
        - Known.known_0:
          - Provide a complete closed-book setup: scenario + assumptions + required constants/approximations; avoid external pointers like "see figure/table".
        - Question:
          - This first problem should preferably be MCQ (single choice), with at least 4 mutually exclusive options.
        """
    )
)

_QAI_FIELDS_QUESTION_SUFFIX_EN = PromptSection(
    text=dedent(
        """\
          - Avoid vague options such as "all of the above / none of the above".
        """
    )
)

_QAI_SELF_CHECK_EN = PromptSection(
    text=dedent(
        """\
        ## Self-check

        - If MCQ: at least 4 mutually exclusive options; the Question says "answer with the option letter only"; Answer is a single letter in \\boxed{...}.
        - Known.history is [] and top-level keys are exactly Step/Subject/Known/Question/Solution/Answer; Step is 0.
        - No external pointers ("see figure/table/above/below").
        - Solution uses S1/S2/... with at least 2 clear steps; the final conclusion matches Answer exactly.
        - If removing paper-grounded nouns/symbols still leaves a generic standalone problem, the question is not grounded enough and must be rewritten.

        ## Example

        ```json
        {"Step": 0, "Subject": "Discipline-Subarea", "Known": {"known_0": "setup...", "history": []}, "Question": "Question text (Options A/B/C/D...)", "Solution": "S1: ...; S2: ...", "Answer": "\\\\boxed{A}"}
        ```
        """
    )
)

QAI_TEMPLATE_EN = PromptTemplate(
    name="qa_init_v2_en",
    sections=[
        _QAI_TASK_EN,
        _QAI_OUTPUT_SCHEMA_EN,
        _QAI_PROCESS_EN,
        _QAI_FIELDS_HEADER_EN,
        COMMON_QUESTION_TYPES_EN,
        _QAI_FIELDS_QUESTION_SUFFIX_EN,
        PromptSection(text="- Solution (structured steps; head-tail compatible):"),
        COMMON_SOLUTION_SCHEMA_EN,
        PromptSection(text="- Answer:"),
        COMMON_ANSWER_SCHEMA_EN,
        PromptSection(text="## Style"),
        COMMON_STYLE_SECTION_EN,
        _QAI_SELF_CHECK_EN,
        _QAI_PAPER_INPUT_EN,
    ],
)

QA_INIT_V2_EN = QAI_TEMPLATE_EN.render_body({})


# 论文摘要压缩 Prompt（Python style，对应原 paper_brief.md）
PAPER_BRIEF_PROMPT = dedent(
    """\
    你是一名技术写作者，请基于输入的论文文本，输出一个高度压缩的“出题草图”，用于后续生成考试题，而非复述实验报告。

    请只输出一个合法 JSON，对应键含义如下：

    - "subject"：论文最可能的学科与二级方向，格式如 "物理-流体力学"、"数学-概率与统计"。
    - "keywords"：3–6 个高价值关键词或核心变量，尽量用通用符号或术语（如 Re、λ、能量预算、判据/阈值、方程名等）。
    - "brief"：一段简洁的中文摘要，聚焦核心物理/数学问题、主要假设、关键量与关系，避免实验装置/数据表述，避免流水账。

    约束：
    - 不要给出题目或问句，不要写选项。
    - 摒弃冗余的实验流程、测量细节、数据点；保留可复用的定义、判据、标度、方程或模型关系。
    - 若输入噪声较大，请宁可模糊些也不要编造细节。

    输入占位符：
    {paper}
    """
)


PAPER_BRIEF_PROMPT_EN = dedent(
    """\
    You are a technical writer. Based on the input paper text, output a highly compressed "question-generation brief" for building exam-style problems (not an experimental report).

    Output one valid JSON only. Keys:

    - "subject": the most likely discipline and sub-area, e.g., "Physics-Fluid Mechanics", "Mathematics-Probability and Statistics".
    - "keywords": 3–6 high-value keywords or core variables (use common symbols/terms where possible: Re, λ, energy budget, thresholds/criteria, equation names, etc.).
    - "brief": a concise English brief focusing on the core scientific/mathematical problem, main assumptions, key quantities and relationships. Avoid apparatus details and narrative fluff.

    Constraints:
    - Do not output questions or options.
    - Remove redundant experimental procedures/measurement details/data points; keep reusable definitions, criteria, scaling relations, equations, or model relationships.
    - If the input is noisy, prefer being vague rather than inventing details.

    Input placeholder:
    {paper}
    """
)
