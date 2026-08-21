"""Revise prompts (Python style)."""

from __future__ import annotations
from textwrap import dedent

from ._base import PromptSection, PromptTemplate
from .common import (
    COMMON_KNOWN_TREE,
    COMMON_EXTEND_CONSTRAINTS,
    COMMON_QUESTION_TYPES,
    COMMON_SOLUTION_SCHEMA,
    COMMON_ANSWER_SCHEMA,
)

__all__ = [
    "REVISE_V1",
    "REVISE_CORRECTNESS_V1",
    "REVISE_DIFFICULTY_V1",
]

# === Shared Sections ===

_REV_INPUT = PromptSection(
    text=dedent(
        """\
        系统会在本提示中填入以下信息：
        {
          Step: i,
          Known: { ... 当前已知条件对象 ... },
          Question: "... 上一版 Question_i 文本 ...",
          Answer: "... 上一版 Answer_i 文本 ...",
          director_notes: "...（可选）来自 Director/solver 的反馈与改进建议..."
        }

        """
    )
)

# === Revise Standard ===

_REV_STD_ROLE = PromptSection(
    text=dedent(
        """\
        # Revise Operator（重写当前 Step 的题目）

        你是“Revise”算子，负责在**保留当前链路主题与难度级别**的前提下，重写最近一步的题目与答案，用于修正表达问题或细节错误。
        """
    )
)

_REV_STD_TASK = PromptSection(
    text=dedent(
        """\
        你的任务（可以理解为“对上一轮 Extend 的质量复查+修订”）：
        - 保持 `Step=i` 不变，对同一步题目进行“修订版”设计，而不是新增第 i+1 步；
        - 保持当前场景与核心考点不变，但可以：
          - 修正错误或不一致的条件；
          - 改善逻辑结构与表述清晰度；
          - 适度调整难度，使其更合理但不过度膨胀；
        - 在新题中补全使其**自洽可解**所需的全部条件，避免出现互相矛盾或无法满足的参数设定；
        - 若 `director_notes` 提供了具体改进方向（如“统一符号”“保证选项中存在唯一正确答案”“修复参数关系矛盾”），尽量采纳。
        """
    )
)

# === Revise Correctness ===

_REV_CORR_ROLE = PromptSection(
    text=dedent(
        """\
        # Revise-Correctness Operator（修正题目正确性）

        你是“Revise-Correctness”算子，负责在**保持当前链路场景与大致难度不变**的前提下，修正最近一步题目的“正确性问题”，包括但不限于：
        - 题干条件不充分或自相矛盾；
        - 选项中不存在唯一正确答案；
        - Answer 给出的结论与已知物理/数学关系不一致；
        - 解法中隐含使用了题干未声明的假设。
        """
    )
)

_REV_CORR_TASK = PromptSection(
    text=dedent(
        """\
        你的主要任务可以理解为“对上一轮 Extend 的**正确性**做复查+修订”：

        - **保持 Step=i 不变**，对同一步题目进行修订，而不是新增第 i+1 步；
        - **保持当前场景与核心考点不变**，但可以：
          - 修正错误或不一致的条件定义、符号约定或维度关系；
          - 调整 Question 文本，使其与 Answer/Solution 中隐含的假设完全一致；
          - 修改 Answer，使其与正确推导结果一致；
          - 适度补充必要的条件，以保证“给定 known_0 + background 后，题目有唯一可解答案”；
        - **不要刻意提升难度**：本算子优先解决“题目对不对”的问题，而不是“题目难不难”的问题；只有在修正错误时不可避免地改变难度时，才允许轻微调整难度级别；
        - 若 director_notes 中明确指出了错误类型或期望修正方向，请优先采纳这些修正意见。
        """
    )
)

# === Revise Difficulty ===

_REV_DIFF_ROLE = PromptSection(
    text=dedent(
        """\
        # Revise-Difficulty Operator（调节题目难度与结构）

        你是“Revise-Difficulty”算子，负责在**保证当前题目逻辑正确、答案不变或等价**的前提下，调节最近一步题目的“难度与结构”，使其更符合目标难度区间（例如：head–tail 对 strong 可解，对 medium 具有挑战）。
        """
    )
)

_REV_DIFF_TASK = PromptSection(
    text=dedent(
        """\
        你的主要任务可以理解为“在保持**结论正确**的前提下，对上一轮 Extend 的**难度曲线**做精修”：

        - **保持 Step=i 不变**，对同一步题目进行重写，而不是新增第 i+1 步；
        - **默认保持 Answer 的数学/物理结论不变**，允许做等价重写，但不得随意更改结论本身；只有在 director_notes 明确指出原结论有错误时，才交由“Correctness”模式处理；
        - 通过以下方式适度提升或精细化难度（根据 director_notes 指示选择合适组合）：
          - 将原本直接给出的事实/公式，改为需要在题目中自行推导的中间结论；
          - 引入更精细的参数依赖（例如从简单幂律变成需要区分不同极限区间的分段表达式）；
          - 将“单步代入计算”改为“多步推理+最后一步代入”；
          - 调整题型结构（例如从直接问数值改为问标度关系、极限比较、对数图线斜率等）。
        - 避免引入全新的、与当前链路主题无关的情景；本算子主要在“原结论正确但题目太平/太机械”的前提下工作。
        """
    )
)

# === Shared Constraints & Output ===

_REV_OUTPUT_FORMAT = PromptSection(
    text=dedent(
        """\
        ## 输出格式（严格 JSON，用 ```json 代码块包裹）

        仅输出一个合法 JSON 对象，必须用 ```json 代码块包裹。顶层键必须为：

        - `"Step"`：整数，应等于输入中的 i；
        - `"Question"`：修订后的题面文本（已修复定义/条件/设定错误）；
        - `"Solution"`：字符串，按 "S1: …; S2: …; …" 给出多步解法；步骤必须足以从 `known_0 + 既有 background + 修订后的 Question_i` 推导出与 Answer_i 完全一致的结论，并在推理中显式复用至少一个前面步骤的 Answer 或关键中间结论；
        - `"Answer"`：修订后的最终答案文本（含 `\\boxed{…}`，遵守统一答案格式）；
        - `"NewBackground"`：可选字符串或字符串列表；仅当修正/调整过程中**确实需要新增此前完全没有出现过的背景设定**时使用；
        - **不要返回 `"Known"`，系统统一维护 Known 结构。**

        示例结构（仅示意）：

        ```json
        {
          "Step": 2,
          "Question": "... 修订后的 Question_2 文本 ...",
          "Solution": "S1: ...; S2: ...",
          "Answer": "\\\\boxed{A}",
          "NewBackground": ["... 可选 ..."]
        }
        ```
        """
    )
)

# === Templates Assembly ===

_REVISE_STD_TEMPLATE = PromptTemplate(
    name="revise_v1_mod",
    sections=[
        _REV_STD_ROLE,
        _REV_INPUT,
        COMMON_KNOWN_TREE,
        _REV_STD_TASK,
        COMMON_EXTEND_CONSTRAINTS,

        PromptSection(text="题型说明：\n默认保持与原题相同的题型，除非原题型明显导致歧义或不可判定。若需调整，请遵守："),
        COMMON_QUESTION_TYPES,

        PromptSection(text="答案要求（统一格式）："),
        COMMON_ANSWER_SCHEMA,

        PromptSection(text="解法书写（Solution）："),
        COMMON_SOLUTION_SCHEMA,

        _REV_OUTPUT_FORMAT,
    ]
)

_REVISE_CORR_TEMPLATE = PromptTemplate(
    name="revise_correctness_v1_mod",
    sections=[
        _REV_CORR_ROLE,
        _REV_INPUT,
        COMMON_KNOWN_TREE,
        _REV_CORR_TASK,
        COMMON_EXTEND_CONSTRAINTS,

        PromptSection(text="题型说明：\n默认保持与原题相同的题型。若确需调整题型，请在 director_notes 中有明确指示时才进行。"),
        COMMON_QUESTION_TYPES,

        PromptSection(text="答案要求（统一格式）："),
        COMMON_ANSWER_SCHEMA,

        PromptSection(text="解法书写（Solution）："),
        COMMON_SOLUTION_SCHEMA,

        _REV_OUTPUT_FORMAT,
    ]
)

_REVISE_DIFF_TEMPLATE = PromptTemplate(
    name="revise_difficulty_v1_mod",
    sections=[
        _REV_DIFF_ROLE,
        _REV_INPUT,
        COMMON_KNOWN_TREE,
        _REV_DIFF_TASK,
        COMMON_EXTEND_CONSTRAINTS,

        PromptSection(text="题型说明：\n默认保持与原题相同的题型。若原题为选择题，可增强干扰项区分度；若为计算题，可增加推理步骤。"),
        COMMON_QUESTION_TYPES,

        PromptSection(text="答案要求（统一格式）："),
        COMMON_ANSWER_SCHEMA,

        PromptSection(text="解法书写（Solution）："),
        COMMON_SOLUTION_SCHEMA,

        _REV_OUTPUT_FORMAT,
    ]
)

REVISE_V1 = _REVISE_STD_TEMPLATE.render_body({})
REVISE_CORRECTNESS_V1 = _REVISE_CORR_TEMPLATE.render_body({})
REVISE_DIFFICULTY_V1 = _REVISE_DIFF_TEMPLATE.render_body({})
