"""Extract role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent

from agenqa.domain.extract_schema import extract_output_schema_text

__all__ = ["EXTRACT_V1", "EXTRACT_V1_EN"]


EXTRACT_V1 = dedent(
    f"""\
    # Extract（考点提取算子说明，KnownInit 专用）

    你扮演"考点策划者"的角色，负责从论文摘要/精简背景中提炼可出题的考点。

    你不负责直接出题或给答案，只做"看论文 → 想考什么"的规划。
    具体选择哪个考点作为首题，由后续的 Extend 算子决定。

    ---

    ## 输入说明

    [Paper Brief JSON]
    $paper_brief_json

    [Paper Brief Text]
    $paper_brief_text

    ---

    ## 任务：从论文中提炼可考点

    1. 识别 2–5 个适合作为题干基础的考点（关键科学量、无量纲参数、核心模型/方程、重要近似等）。
    2. 对每个考点简要说明可以考察哪些知识/能力，以及适合的题型风格。
    3. 说明这些考点之间的逻辑关系，以及如何形成链式出题的潜力。

    ---

    ## 输出格式

    仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段严格为：
{extract_output_schema_text()}
    """
)


EXTRACT_V1_EN = dedent(
    f"""\
    # Extract (Exam-Point Planning; KnownInit only)

    You are the "exam-point planner". From the paper brief / condensed background, extract exam points that can be turned into high-quality problems.

    You do NOT directly write problems or answers here. This role only plans: "read the paper → decide what can be tested".
    The first-step selection will be handled by the downstream generation operator.

    All output text must be in English.

    ---

    ## Input

    [Paper Brief JSON]
    $paper_brief_json

    [Paper Brief Text]
    $paper_brief_text

    ---

    ## Task

    1. Identify 2–5 exam points suitable as the backbone of problem statements (key quantities, dimensionless parameters, core models/equations, important approximations, etc.).
    2. For each exam point, briefly describe what knowledge/skills it tests and what question style fits.
    3. Explain the logical relationships among these points and how they can form a multi-step chain.

    ---

    ## Output Format

    Output one JSON object only, wrapped in a ```json code block. The top-level fields must be exactly:
{extract_output_schema_text(lang="en")}
    """
)
