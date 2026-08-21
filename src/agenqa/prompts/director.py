"""Director prompt (Python style).

Director 负责根据当前链路状态，从可用动作中选择下一步操作。
"""

from __future__ import annotations

from typing import Any, Dict, List
from textwrap import dedent
import json

from ._base import PromptSection, PromptTemplate
from .common import (
    COMMON_KNOWN_TREE,
    COMMON_KNOWN_TREE_EN,
    COMMON_EDGE_QA_VS_PATH,
    COMMON_EDGE_QA_VS_PATH_EN,
    COMMON_SOLVER_SIGNALS_COGNITION,
    COMMON_SOLVER_SIGNALS_COGNITION_EN,
)

__all__ = [
    "DIRECTOR_TEMPLATE",
    "DIRECTOR_TEMPLATE_EN",
    "DIRECTOR_TEMPLATE_EXECUTABLE",
    "DIRECTOR_TEMPLATE_EXECUTABLE_EN",
    "build_director_v1_body",
    "DIRECTOR_ROLE_SECTION",
    "DIRECTOR_ROLE_SECTION_EN",
    "DIRECTOR_STATE_SECTION",
    "DIRECTOR_STATE_SECTION_EN",
    "DIRECTOR_COGNITION_HEADER",
    "DIRECTOR_COGNITION_HEADER_EN",
    "DIRECTOR_INFO_HEADER",
    "DIRECTOR_INFO_HEADER_EN",
    "DIRECTOR_TRACK_SEMANTIC_SECTION",
    "DIRECTOR_TRACK_SEMANTIC_SECTION_EN",
    "DIRECTOR_TRACK_EXECUTABLE_SECTION",
    "DIRECTOR_TRACK_EXECUTABLE_SECTION_EN",
    "DIRECTOR_ACTIONS_SECTION",
    "DIRECTOR_ACTIONS_SECTION_EN",
    "DIRECTOR_OUTPUT_SECTION",
    "DIRECTOR_OUTPUT_SECTION_EN",
]

# === Director v1 prompt sections ===

DIRECTOR_ROLE_SECTION = PromptSection(
    text=dedent(
        """\
        你是题目生成流程中的"导演（Director）"。

        你的任务：根据当前链路状态，从当前启用的动作中选择下一步；
        若选择 "Extend"，需要同时指定下一题的题型（QuestionType）。
        """
    )
)

DIRECTOR_COGNITION_HEADER = PromptSection(
    title="认知（先读）",
    title_level=1,
    kind="cognition",
    text=dedent(
        """\
        以下是做决策所需的概念与约束；请先阅读，再查看后面的“信息（本轮输入）”。
        """
    ),
)

DIRECTOR_COGNITION_HEADER_EN = PromptSection(
    title="Cognition (read first)",
    title_level=1,
    kind="cognition",
    text=dedent(
        """\
        Below are the concepts and constraints needed for decision-making. Read this first, then check the “Information (inputs)” section.
        """
    ),
)

DIRECTOR_INFO_HEADER = PromptSection(
    title="信息（本轮输入）",
    title_level=1,
    kind="info",
    text="",
)

DIRECTOR_INFO_HEADER_EN = PromptSection(
    title="Information (inputs)",
    title_level=1,
    kind="info",
    text="",
)

DIRECTOR_ROLE_SECTION_EN = PromptSection(
    text=dedent(
        """\
        You are the "Director" of the problem-generation workflow.

        Your task: based on the current chain state, choose the next operation from the enabled actions.
        If you choose "Extend", you must also specify the next question type (QuestionType).
        All output text must be in English.
        """
    )
)

DIRECTOR_STATE_SECTION = PromptSection(
    text=dedent(
        """\
        下面是当前链路状态（统一 Director View JSON）：
        [STATE_JSON]
        {state_json_pretty}

        该 JSON 是系统提供给你的“统一视图”，主要字段包括：
        - progress：step / next_step / max_steps
        - metrics / solver_metrics / solver_consensus：edge 与 path 两视角的求解信号与反馈
        - type1_ambiguity：Type1（语义世界观不唯一）诊断摘要（基于 path multi-strong；见 artifacts_ref）
        - track_context：该模式下的上下文（见下方“模式说明”）
        - available_operations：当前允许的 Operation 列表（系统也会单独重复提供一次）
        """
    )
)

DIRECTOR_STATE_SECTION_EN = PromptSection(
    text=dedent(
        """\
        Below is the current chain state (unified Director View JSON):
        [STATE_JSON]
        {state_json_pretty}

        This JSON is a unified Director View provided by the system. Key fields:
        - progress: step / next_step / max_steps
        - metrics / solver_metrics / solver_consensus: edge vs path solver signals and feedback
        - type1_ambiguity: Type1 semantic ambiguity summary (path multi-strong; see artifacts_ref)
        - track_context: mode-specific context (see the Mode Section below)
        - available_operations: allowed operations (also provided separately)
        """
    )
)

DIRECTOR_TRACK_SEMANTIC_SECTION = PromptSection(
    text=dedent(
        """\
        ## Mode Section: Unified

        track_context 通常包含：
        - path_start_memory：Path-Fold 的 solver 起点 P（仅 premise_bank；不含 episode_seed）
        - director_memory_window：窗口化上下文（用于诊断与决策）
        - history_tail：最近几步题面与答案
        - path_fold_result：当前 tail 的 path-fold 题面（direct/scaffolded）与 fold_notes
        - key_facts_tail / expected_primary_fact_id：链式递进锚点（用于泄露/坍塌检测）

        ### 决策映射（约束 + 证据优先）
        命中高优先级约束时优先触发：

        - **硬规则：edge/path 信号分工**
          - `edge strong` 用于判断“题目/答案是否正确（correctness/well-posedness）”。
          - `path strong` 用于判断“链路可达性与难度分层（是否过难、是否有区分度）”。
          - 当出现 `edge strong` 全对且 `path strong` 部分对（即既有 correct 也有 incorrect）时，应默认视为“高区分度且可用”的正向信号，**优先选择 Extend**；
            仅当存在明确的 Type1/Type2 证据或 path-fold 泄露/不自包含证据时，才允许改判为 Revise。

        - 若发现 **Path-Fold 题面存在指针式引用**（`path_fold_result` 的 direct/scaffolded 中出现“上一步/根据 step/previous step/as shown above”等）
          或 **答案泄露 / 递进坍塌**（题干显式包含本题 Answer，或把 key_facts_tail 的完整表达式直接作为“已知条件”），则：
          - Operation="Revise"
          - ReviseMode="reuse_hidden"
          - 口径补充：若 `solver_metrics.edge.strong` 整体显示 edge 视角可解，而 `solver_metrics.path.strong` 出现失败/分裂，这可能只是预期信号（edge 自洽可解，而 path 因隐藏链条/推理负担更难解）。这种差异本身不足以推断结构问题，更不能单独作为选择 `reuse_hidden` 的依据。
          - 选择 `reuse_hidden` 时，必须能从 `path_fold_result.question_direct/question_scaffolded/fold_notes` 中直接指出证据（指针式引用/不自包含/泄露或坍塌）；否则不要选 `reuse_hidden`。

        - 若发现 **Type1 语义世界观歧义**（`type1_ambiguity.type1_suspected==true`，当前以 path multi-strong 的 well-posed 反例为主证据），优先考虑：
          - Operation="Revise"
          - ReviseMode="world_contract"
          - 该信号应结合 `type1_ambiguity.wellposed_votes` 与 `type1_ambiguity.solvers` 证据解释，不应仅凭文本表面差异触发。

        - 若发现 **Type2 作答协议/判对口径歧义**（例如 exact/approx 未声明、容差/单位不明确、分支/等价类口径缺失、MCQ 唯一性不足、Answer 输出协议不唯一可解析等），则：
          - Operation="Revise"
          - ReviseMode="answer_contract"
          - 参考信号：若 `type2_contract.has_errors==true`（来自 `answer_contract_report.json`），应当优先触发该路径。

        - 若题目 **可解且正确**，但存在 **质量问题**（太简单/重复/仅换词换参/区分度低），则：
          - Operation="Revise"
          - ReviseMode="quality"

        - 若发现 **正确性 / 良定义问题**（例如 `solver_metrics.edge.strong` 中存在明确的 incorrect / not-well-posed 结果，或其中的 `correctness_feedback` 指出矛盾/缺条件/答案错/歧义严重，或 question_well_posed==false），则：
          - Operation="Revise"
          - ReviseMode="correctness"

        - 只有在以上问题都不命中时，才允许选择 Extend 或 Finish。
        - 当 Operation="Extend" 时必须给出 QuestionType：
          - 若输入 JSON 中包含 `available_question_types`，则必须从该列表中选择；
          - 否则在 MCQ/Derivation/Numeric 中选择。
        """
    )
)

DIRECTOR_TRACK_SEMANTIC_SECTION_EN = PromptSection(
    text=dedent(
        """\
        ## Mode Section: Unified

        track_context typically contains:
        - path_start_memory: Path-Fold solver start P (premise_bank only; no episode_seed)
        - director_memory_window: windowed context for diagnosis/decisions
        - history_tail: recent question/answer texts
        - path_fold_result: folded path prompts (direct/scaffolded) + fold_notes
        - key_facts_tail / expected_primary_fact_id: progression anchors (for leakage/collapse checks)

        ### Decision mapping (constraints with evidence-first policy)
        When high-priority constraints match, they should be preferred:

        - **Hard rule: edge/path signal responsibility split**
          - Use `edge strong` to judge correctness/well-posedness of the step answer itself.
          - Use `path strong` to judge end-to-end reachability and difficulty stratification (too hard vs discriminative).
          - If `edge strong` is all-correct while `path strong` is partially-correct (mixed correct+incorrect), treat this as a positive high-discrimination signal by default and **prefer Extend**;
            only switch to Revise when there is explicit Type1/Type2 evidence or explicit path-fold leakage / non-self-contained evidence.

        - If you detect **pointer references in Path-Fold** (in `path_fold_result` direct/scaffolded)
          or **answer leakage / progression collapse**, then:
          - Operation="Revise"
          - ReviseMode="reuse_hidden"
          - Clarification: if `solver_metrics.edge.strong` looks locally solvable/sound while `solver_metrics.path.strong` shows failures/splits, this can be expected (path is harder due to hidden chain or reasoning burden). This gap alone is not evidence of a structural issue and must not be the sole reason to choose `reuse_hidden`.
          - If you choose `reuse_hidden`, you must point to direct evidence in `path_fold_result.question_direct/question_scaffolded/fold_notes` (pointer reference / not self-contained / leakage or collapse). Otherwise, do not choose `reuse_hidden`.

        - If you detect **Type1 semantic world-view ambiguity** (`type1_ambiguity.type1_suspected==true`; currently driven primarily by path multi-strong not-well-posed evidence), prefer:
          - Operation="Revise"
          - ReviseMode="world_contract"
          - This signal must be justified with `type1_ambiguity.wellposed_votes` and `type1_ambiguity.solvers`; do not trigger it from superficial text-form differences alone.

        - If you detect **Type2 answer-protocol / judging-contract ambiguity** (e.g., exact/approx not stated, missing tolerances/units, missing branch/equivalence policy, MCQ non-uniqueness, or non-parseable Answer protocol), then:
          - Operation="Revise"
          - ReviseMode="answer_contract"
          - Signal: if `type2_contract.has_errors==true` (from `answer_contract_report.json`), prefer this path.

        - If the problem is solvable/correct but has **quality issues** (too easy/repetitive/low discrimination), then:
          - Operation="Revise"
          - ReviseMode="quality"

        - If you detect **correctness / well-posedness issues** (for example, `solver_metrics.edge.strong` contains explicit incorrect / not-well-posed results, or its `correctness_feedback` points to contradictions/missing conditions/wrong answers/severe ambiguity, or question_well_posed==false), then:
          - Operation="Revise"
          - ReviseMode="correctness"

        - Only if none of the above issues match, you may choose Extend or Finish.
        - If Operation="Extend", you must provide QuestionType:
          - If the input JSON includes `available_question_types`, you must pick from that list.
          - Otherwise, pick from MCQ/Derivation/Numeric.
        """
    )
)

DIRECTOR_QUESTION_TYPE_SECTION = PromptSection(
    text=dedent(
        """\
        ## 题型能力与本质区别

        选择 QuestionType 本质上是在选择"考察什么能力"：

        | 题型 | 本质 | 考察能力 | LLM 表现 | 工具配合 |
        |------|------|----------|----------|----------|
        | MCQ | 识别/选择 | 概念辨析/口径对齐 | 擅长 | 无 |
        | Derivation | 符号推理 | 公式推导/变形/证明 | 擅长 | 无（LLM-Judge 可选） |
        | Numeric | 建模 + 计算 | 整合概念建模 + 数值方法求解 | 易出错 | oracle_code + solver_tool |

        **关键区分**：
        - MCQ：答案在选项中，考察"选对"
        - Derivation：答案是符号表达式，考察"推对"
        - Numeric：答案是数值，考察"建模对 + 算对"（需整合多个科学/数学概念，或需数值方法如积分/优化/迭代）
        """
    )
)

DIRECTOR_QUESTION_TYPE_SECTION_EN = PromptSection(
    text=dedent(
        """\
        ## QuestionType capabilities and the key distinctions

        Choosing QuestionType is essentially choosing "what capability to test":

        | Type | Essence | What it tests | LLM performance | Tooling |
        |------|---------|---------------|-----------------|---------|
        | MCQ | identify/choose | concept disambiguation / convention alignment | strong | none |
        | Derivation | symbolic reasoning | derivation / transformation / proof | strong | none (LLM-judge optional) |
        | Numeric | modeling + computation | modeling integration + numerical methods | error-prone | oracle_code + solver_tool |

        **Key distinctions**:
        - MCQ: the answer is among options; it tests "choose correctly"
        - Derivation: the answer is a symbolic expression; it tests "derive correctly"
        - Numeric: the answer is a number; it tests "model correctly + compute correctly" (requires integrating concepts or using numerical methods such as integration/optimization/iteration)
        """
    )
)

DIRECTOR_TRACK_EXECUTABLE_SECTION = PromptSection(
    text=dedent(
        """\
        ## Mode Section: Executable

        当 track == "executable" 时，track_context 通常包含：
        - executable_seed：executable episode 的起点信息（如 problem_id/split/source；不包含 golden 代码）
        - executable_tail：当前 tail step 的 spec（接口/描述/tests 等；不包含 golden 代码）
        - executable_history_tail：最近若干步的 executable spec 历史（用于判断重复/泄露/链路退化）
        - executable_step_certs_tail：来自 Memory 的稳定摘要（multi-solver 共识/错误分类等；不包含 golden 代码）
        - path_fold_result：当前 tail 的 path-fold spec（direct/scaffolded）与 fold_notes（自包含性检查）

        ### 决策映射（硬约束）
        命中任一条即触发：

        - 若发现 path-fold spec **不自包含** 或存在 **指针式引用**（例如“上一题/step X/previous step/as shown above”等），则：
          - Operation="Revise"
          - ReviseMode 可填 "reuse_hidden"（也可留空）

        - 若发现 **正确性 / 良定义问题**（例如执行失败，或 `solver_metrics.edge.strong` 中出现明确的 incorrect / error / not-well-posed 结果，或其中的反馈指向 error/traceback），则：
          - Operation="Revise"
          - ReviseMode 可填 "correctness"（也可留空）

        - 若当前 step 可解且正确，但你判断存在质量问题（重复/太平/区分度低），则：
          - Operation="Revise"
          - ReviseMode 可填 "quality"（也可留空）

        - 若 multi-solver 共识显示“失败模式集中”（例如大多数 solver 因同类错误失败：依赖越界/契约不一致/超时），优先选择 Revise，并在 operator_notes 中指出主要失败类型与修复方向。

        - 只有在以上问题都不命中时，才允许选择 Extend 或 Finish。

        说明：
        - Executable track 不需要 QuestionType；请将 QuestionType 置为空字符串。
        """
    )
)

DIRECTOR_TRACK_EXECUTABLE_SECTION_EN = PromptSection(
    text=dedent(
        """\
        ## Mode Section: Executable

        When track == "executable", track_context typically contains:
        - executable_seed: executable episode seed info (e.g., problem_id/split/source; NO golden code)
        - executable_tail: the current tail step spec (interfaces/descriptions/tests; NO golden code)
        - executable_history_tail: recent executable spec history (for repetition/leakage/degeneration checks)
        - executable_step_certs_tail: stable Memory summaries (multi-solver consensus / error taxonomy; NO golden code)
        - path_fold_result: folded path spec (direct/scaffolded) + fold_notes (self-containedness checks)

        ### Decision mapping (hard constraints)
        If any rule matches, it triggers:

        - If the path-fold spec is **not self-contained** or contains **pointer references** (e.g., “previous step”, “step X”, “as shown above”), then:
          - Operation="Revise"
          - ReviseMode may be "reuse_hidden" (or empty)

        - If you detect **correctness / well-posedness issues** (e.g., execution failure, or `solver_metrics.edge.strong` contains explicit incorrect / error / not-well-posed results, or its feedback points to error/traceback), then:
          - Operation="Revise"
          - ReviseMode may be "correctness" (or empty)

        - If it is correct/solvable but has **quality issues** (repetitive/too easy/low discrimination), then:
          - Operation="Revise"
          - ReviseMode may be "quality" (or empty)

        - If multi-solver consensus shows a concentrated dominant failure mode (e.g., most solvers fail with the same taxonomy: dependency violation / contract mismatch / timeout), prefer Revise and mention the dominant failure mode + repair direction in operator_notes.

        - Only if none of the above issues match, you may choose Extend or Finish.

        Notes:
        - Executable track does not use QuestionType; set QuestionType to an empty string.
        """
    )
)

DIRECTOR_ACTIONS_SECTION = PromptSection(
    text=dedent(
        """\
        可用动作列表由系统动态提供，形如 JSON 数组：
        [AVAILABLE_OPERATIONS]
        {available_operations_json}

        Operation 字段必须从上述列表中选择；若列表为空，应当回退为 "Extend"。
        """
    )
)

DIRECTOR_ACTIONS_SECTION_EN = PromptSection(
    text=dedent(
        """\
        The available actions are provided dynamically as a JSON array:
        [AVAILABLE_OPERATIONS]
        {available_operations_json}

        The Operation field must be chosen from the list above; if the list is empty, fall back to "Extend".
        """
    )
)

DIRECTOR_OUTPUT_SECTION = PromptSection(
    text=dedent(
        """\
        请仅输出一个严格的 JSON 对象，不要使用 Markdown 代码块或额外解释文字，
        字段约定如下：
        {{
          "Operation": "从 [AVAILABLE_OPERATIONS] 中选择一个动作（如 Extend/Revise/Finish）",
          "ReviseMode": "correctness" | "world_contract" | "answer_contract" | "reuse_hidden" | "quality" | "",
          "QuestionType": "MCQ" | "Derivation" | "Numeric" | "",
          "Reason": "一句话解释你为何选择这个动作，推荐 50–120 字",
          "OperatorNotes": "可选：给下一步算子的简短提示，可为空字符串"
        }}
        - 当 Operation 为 "Extend" 时，必须给出 QuestionType；
        - 当 Operation 为 "Revise" 时，必须给出 ReviseMode（"correctness" / "world_contract" / "answer_contract" / "reuse_hidden" / "quality"）；
        """
    )
)

DIRECTOR_OUTPUT_SECTION_EN = PromptSection(
    text=dedent(
        """\
        Output one strict JSON object only (no Markdown code blocks; no extra explanation).
        Field contract:
        {
          "Operation": "one action from [AVAILABLE_OPERATIONS] (e.g., Extend/Revise/Finish)",
          "ReviseMode": "correctness" | "world_contract" | "answer_contract" | "reuse_hidden" | "quality" | "",
          "QuestionType": "MCQ" | "Derivation" | "Numeric" | "",
          "Reason": "one sentence explaining your choice (recommended 50–120 chars)",
          "OperatorNotes": "optional short hint for the next operator; may be empty"
        }
        - If Operation == "Extend", you must provide QuestionType.
        - If Operation == "Revise", you must provide ReviseMode ("correctness" / "world_contract" / "answer_contract" / "reuse_hidden" / "quality").
        """
    )
)

DIRECTOR_OUTPUT_SECTION_EXECUTABLE = PromptSection(
    text=dedent(
        """\
        请仅输出一个严格的 JSON 对象，不要使用 Markdown 代码块或额外解释文字，
        字段约定如下：
        {{
          "Operation": "从 [AVAILABLE_OPERATIONS] 中选择一个动作（如 Extend/Revise/Finish）",
          "ReviseMode": "correctness" | "world_contract" | "answer_contract" | "reuse_hidden" | "quality" | "",
          "QuestionType": "",
          "Reason": "一句话解释你为何选择这个动作，推荐 50–120 字",
          "OperatorNotes": "可选：给下一步算子的简短提示，可为空字符串"
        }}
        - 当 Operation 为 "Revise" 时，ReviseMode 可留空（也可填 "correctness"/"world_contract"/"answer_contract"/"reuse_hidden"/"quality"）；
        - Executable 模式不使用 QuestionType：请将 QuestionType 固定为 ""。
        """
    )
)

DIRECTOR_OUTPUT_SECTION_EXECUTABLE_EN = PromptSection(
    text=dedent(
        """\
        Output one strict JSON object only (no Markdown code blocks; no extra explanation).
        Field contract:
        {
          "Operation": "one action from [AVAILABLE_OPERATIONS] (e.g., Extend/Revise/Finish)",
          "ReviseMode": "correctness" | "world_contract" | "answer_contract" | "reuse_hidden" | "quality" | "",
          "QuestionType": "",
          "Reason": "one sentence explaining your choice (recommended 50–120 chars)",
          "OperatorNotes": "optional short hint for the next operator; may be empty"
        }
        - If Operation == "Revise", ReviseMode may be empty (or one of "correctness"/"world_contract"/"answer_contract"/"reuse_hidden"/"quality").
        - Executable mode does not use QuestionType: set QuestionType to "".
        """
    )
)

DIRECTOR_TEMPLATE = PromptTemplate(
    name="director_v1",
    sections=[
        DIRECTOR_ROLE_SECTION,
        DIRECTOR_COGNITION_HEADER,
        DIRECTOR_TRACK_SEMANTIC_SECTION,
        DIRECTOR_QUESTION_TYPE_SECTION,
        COMMON_EDGE_QA_VS_PATH,  # Edge QA vs Path QA（同一目标、不同视角）
        COMMON_SOLVER_SIGNALS_COGNITION,  # Solver 信号的最小充分语义（给 Director）
        COMMON_KNOWN_TREE,  # 复用统一的 Known 结构说明与可见性约定
        DIRECTOR_INFO_HEADER,
        DIRECTOR_STATE_SECTION,
        DIRECTOR_ACTIONS_SECTION,
        DIRECTOR_OUTPUT_SECTION,
    ],
)

DIRECTOR_TEMPLATE_EN = PromptTemplate(
    name="director_v1_en",
    sections=[
        DIRECTOR_ROLE_SECTION_EN,
        DIRECTOR_COGNITION_HEADER_EN,
        DIRECTOR_TRACK_SEMANTIC_SECTION_EN,
        DIRECTOR_QUESTION_TYPE_SECTION_EN,
        COMMON_EDGE_QA_VS_PATH_EN,
        COMMON_SOLVER_SIGNALS_COGNITION_EN,
        COMMON_KNOWN_TREE_EN,
        DIRECTOR_INFO_HEADER_EN,
        DIRECTOR_STATE_SECTION_EN,
        DIRECTOR_ACTIONS_SECTION_EN,
        DIRECTOR_OUTPUT_SECTION_EN,
    ],
)


DIRECTOR_TEMPLATE_EXECUTABLE = PromptTemplate(
    name="director_v1_executable",
    sections=[
        DIRECTOR_ROLE_SECTION,
        DIRECTOR_COGNITION_HEADER,
        DIRECTOR_TRACK_EXECUTABLE_SECTION,
        COMMON_EDGE_QA_VS_PATH,
        COMMON_SOLVER_SIGNALS_COGNITION,
        DIRECTOR_INFO_HEADER,
        DIRECTOR_STATE_SECTION,
        DIRECTOR_ACTIONS_SECTION,
        DIRECTOR_OUTPUT_SECTION_EXECUTABLE,
    ],
)

DIRECTOR_TEMPLATE_EXECUTABLE_EN = PromptTemplate(
    name="director_v1_executable_en",
    sections=[
        DIRECTOR_ROLE_SECTION_EN,
        DIRECTOR_COGNITION_HEADER_EN,
        DIRECTOR_TRACK_EXECUTABLE_SECTION_EN,
        COMMON_EDGE_QA_VS_PATH_EN,
        COMMON_SOLVER_SIGNALS_COGNITION_EN,
        DIRECTOR_INFO_HEADER_EN,
        DIRECTOR_STATE_SECTION_EN,
        DIRECTOR_ACTIONS_SECTION_EN,
        DIRECTOR_OUTPUT_SECTION_EXECUTABLE_EN,
    ],
)


def build_director_v1_body(
    payload: Dict[str, Any],
    allowed_ops: List[str] | None = None,
    *,
    lang: str | None = None,
) -> str:
    """Render Director v1 user prompt body from structured payload.

    - payload 对应 build_director_view(state) 的结果（或 summarize_state 的旧结构）；
    - allowed_ops 为当前允许的 Operation 列表；若未提供则尝试从 payload.available_operations 读取。
    """
    lang_norm = (lang or "").lower().strip()
    use_en = lang_norm in {"en", "english"}
    track = str(payload.get("track") or "").strip().lower()
    if track in {"executable", "code"}:
        template = DIRECTOR_TEMPLATE_EXECUTABLE_EN if use_en else DIRECTOR_TEMPLATE_EXECUTABLE
    else:
        template = DIRECTOR_TEMPLATE_EN if use_en else DIRECTOR_TEMPLATE
    ops = allowed_ops if isinstance(allowed_ops, list) else (payload.get("available_operations") or [])
    ctx: Dict[str, Any] = {
        "state_json_pretty": json.dumps(payload, ensure_ascii=False, indent=2),
        "available_operations_json": json.dumps(ops, ensure_ascii=False),
    }
    return template.render_body(ctx)


# 向后兼容：保留旧变量名，指向同一模板实例（不再导出到 __all__）
DIRECTOR_V1_TEMPLATE = DIRECTOR_TEMPLATE
