"""PaperBrief v3: seed-only extractor (subject + keywords).

Used as a lightweight PaperBrief variant; downstream EpisodeSeedBuilder may
further normalize/compress it into episode_seed.
"""

from __future__ import annotations

from textwrap import dedent

__all__ = [
    "PAPER_BRIEF_V3_SEED",
    "PAPER_BRIEF_V3_SEED_EN",
]


PAPER_BRIEF_V3_SEED = dedent(
    """\
    # PaperBrief-V3（Seed Only）

    你扮演“主题锚定器”。你的目标是：仅从输入文本中提取用于构造 episode_seed 的两项候选信息：
    - `subject`：学科-二级方向（与论文主题一致）
    - `keywords`：能代表论文核心对象/方法/关键符号/术语的关键词列表

    输入文本是唯一信息源；允许抽象/归纳，但不要编造不存在的事实、结论或数值。

    ## 输入

    [Paper Text]
    {paper}

    ## 输出（仅允许的输出）

    - 仅输出一个合法 JSON，必须用 ```json 代码块包裹，不得输出任何额外文本。
    - 顶层键且仅有：`subject`, `keywords`。
    - `keywords` 必须是非空数组（建议 5-10 个，**最多 10 个**），元素为字符串；按重要性排序，尽量包含论文中出现的关键术语/符号/方法名。
    - `keywords` 需优先覆盖论文的主领域与主任务（核心对象/观测量/模型参数/方法框架/主要实验或观测配置及关键系统效应）；避免把后验验证/统计检验/评估指标等细枝末节写进 keywords，除非它们是论文的核心贡献之一。

    ```json
    {
      "subject": "<学科-二级方向>",
      "keywords": ["<kw1>", "<kw2>", "<kw3>"]
    }
    ```
    """
)


PAPER_BRIEF_V3_SEED_EN = dedent(
    """\
    # PaperBrief-V3 (Seed Only)

    Role: topic anchor extractor. Goal: extract only the minimal episode_seed fields from the input text:
    - `subject`: a "Discipline-Subarea" string consistent with the paper
    - `keywords`: a list of keywords capturing core objects/methods/symbols/terms from the paper

    The paper text is the only source of truth. You may summarize/abstract, but do not invent facts or results.

    ## Input

    [Paper Text]
    {paper}

    ## Output (the only allowed output)

    - Output exactly one valid JSON wrapped in a ```json code block. No extra text.
    - Top-level keys must be exactly: `subject`, `keywords`.
    - `keywords` must be a non-empty array (suggested 5-10 items, **at most 10**), each item is a string, ordered by importance.
    - `keywords` should prioritize the paper’s main domain + task (core objects/observables/model parameters/method framework/major experiment or observation setup and key systematics); avoid minor validation/tests/metrics unless they are a core contribution.

    ```json
    {
      "subject": "<Discipline-Subarea>",
      "keywords": ["<kw1>", "<kw2>", "<kw3>"]
    }
    ```
    """
)
