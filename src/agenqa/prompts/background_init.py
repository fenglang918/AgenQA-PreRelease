"""BackgroundInit role prompt (Python style): 合成 episode 级背景 known_0。"""

from __future__ import annotations

from textwrap import dedent

__all__ = ["BACKGROUND_INIT_V1", "BACKGROUND_INIT_V1_EN"]


BACKGROUND_INIT_V1 = dedent(
    """\
    # BackgroundInit（背景合成算子）

    你扮演"背景构建者"的角色，负责将论文摘要与考点规划信息合成为一段**干净、精炼、适合作为题干背景**的描述。

    这段背景将作为后续所有题目的共享上下文（known_0），因此需要：
    1. **自洽完整**：包含解题所需的核心设定（物理量、符号约定、模型假设等），读者无需回溯原论文；
    2. **精炼无冗余**：去除论文摘要中的研究动机、文献综述、实验细节等与出题无关的内容；
    3. **符号规范**：明确定义所有将在题目中使用的符号及其物理意义；
    4. **覆盖考点**：确保所有考点所需的前置知识都在背景中有所交代。

    ---

    ## 输入

    [Subject]
    $subject

    [Keywords]
    $keywords

    [Paper Brief]
    $brief_text

    [Exam Points]
    $exam_points

    [Chain Potential]
    $chain_potential

    ---

    ## 输出格式

    仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段严格为：

    ```json
    {
      "known_0": "<合成的背景描述，一段或多段文本，包含核心设定与符号定义>",
      "symbol_definitions": ["<符号1>: <定义>", "<符号2>: <定义>", ...],
      "core_assumptions": ["<假设1>", "<假设2>", ...]
    }
    ```

    - `known_0`：精炼的背景描述，适合直接作为题干的"已知条件"部分；
    - `symbol_definitions`：背景中涉及的关键符号及其定义（可选，若背景已清晰定义可省略）；
    - `core_assumptions`：背景中隐含的核心假设（可选，帮助后续出题时保持一致性）。
    """
)


BACKGROUND_INIT_V1_EN = dedent(
    """\
    # BackgroundInit (Episode Background Synthesis)

    You are the "background builder". Your job is to synthesize the paper brief and the exam-point plan into a **clean, concise, question-ready** episode background description.

    This background becomes the shared context `known_0` for all subsequent problems in the chain. It must therefore be:
    1. **Self-consistent & sufficient**: include the core setup needed for solving (variables, symbol conventions, model assumptions, etc.), without requiring the reader to go back to the paper.
    2. **Concise, no clutter**: remove motivation, literature review, experimental apparatus details, and other non-essential content.
    3. **Symbol disciplined**: clearly define all symbols that may appear in later questions.
    4. **Exam-point coverage**: ensure prerequisites for the planned exam points are present in the background.

    All output text must be in English.

    ---

    ## Input

    [Subject]
    $subject

    [Keywords]
    $keywords

    [Paper Brief]
    $brief_text

    [Exam Points]
    $exam_points

    [Chain Potential]
    $chain_potential

    ---

    ## Output Format

    Output one JSON object only, wrapped in a ```json code block. The top-level fields must be exactly:

    ```json
    {
      "known_0": "<synthesized background; one or more paragraphs; core setup + symbol definitions>",
      "symbol_definitions": ["<symbol1>: <definition>", "<symbol2>: <definition>", ...],
      "core_assumptions": ["<assumption1>", "<assumption2>", ...]
    }
    ```

    - `known_0`: concise background, suitable to paste into a problem statement as the "Given" section.
    - `symbol_definitions`: key symbols and definitions (optional; can be omitted if already unambiguous in `known_0`).
    - `core_assumptions`: implicit core assumptions (optional; helps later steps stay consistent).
    """
)
