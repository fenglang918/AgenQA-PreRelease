"""ExecutableStepCertBuilder specification text (single source of truth).

This module defines the executable-track-specific semantics for producing
premise_delta / fact_delta / step_cert / key_fact_id in a way that aligns
with the semantic pipeline's KnownTree v2 contract.
"""

from __future__ import annotations

from textwrap import dedent


EXECUTABLE_CHAIN_CERT_GUIDE_ZH = dedent(
    """\
    # Executable 链式证书（ExecutableStepCertBuilder）语义约定

    目标：让 executable track 的多步任务，复用 semantic pipeline 的“显式记忆 + 链式递进”机制：
    - premise_delta / fact_delta / step_cert / key_fact_id
    - 写入 KnownTree v2（premise_bank / fact_bank / step_certs）
    - 通过 key_fact_id(step=t) → expected_primary_fact_id(step=t+1) 形成可复用锚点链

    ## 1) Executable 的 premise_delta（可见于 edge/path）
    premise_delta 应写入“长期有效、解题必须知道、且不会泄露解法”的前提/约定（contract premises），例如：
    - 输入语义：坐标系（fractional/cartesian）、单位、shape、dtype、广播规则
    - 几何/晶格约定：latvec 的行/列约定；坐标到笛卡尔的变换口径
    - 截断/口径：cutoff_radius 的定义；是否枚举晶格向量；是否最小镜像
    - 数值容差：tol、对称性不变量的判定口径（仅作为约定，不写具体答案）

    ## 2) Executable 的 fact_delta（仅 edge 可见；用于链式复用）
    fact_delta 应写入“可复用的语义锚点”，尤其是本步新增接口的契约锚点：
    - key_fact（必须有）：描述本步函数/类“应该做什么”的稳定语义锚点（contract fact）
    - 可选：少量可复用的性质/不变量（property facts），但不要把 golden 代码或测试期望写进去

    ## 3) step_cert（结构化链式证书）
    step_cert 用于显式声明：
    - uses_premise_ids：本步引用了哪些 premises（来自 memory_json.premise_bank 或 premise_delta）
    - uses_fact_ids：本步复用了哪些 prior facts（来自 memory_json.fact_bank；step>=2 时应包含 expected_primary_fact_id）
    - produces_fact_ids：本步新增哪些 facts（必须来自 fact_delta）
    - key_fact_id：必须指向 produces_fact_ids 中的一个 fact（且属于 fact_delta）

    ## 4) ID 约定（强建议）
    - 建议将本步新增 ID 统一前缀为 `c{step}.`（例如 `c2.contract.cross_term`），以避免与 semantic ID 冲突。
    - ID 必须稳定、可读、无前后空白，不得与已有 memory ID 冲突。
    """
)


EXECUTABLE_CHAIN_CERT_GUIDE_EN = dedent(
    """\
    # Executable Chain Certificate (ExecutableStepCertBuilder): Semantics

    Goal: align executable track with the semantic pipeline's explicit memory and chain constraints:
    - premise_delta / fact_delta / step_cert / key_fact_id
    - written into KnownTree v2 (premise_bank / fact_bank / step_certs)
    - key_fact_id(step=t) becomes expected_primary_fact_id(step=t+1) to form a reusable anchor chain

    1) premise_delta (visible to both edge/path)
       Store durable, solver-needed, non-solution-leaking contract premises, e.g.:
       - input semantics: coordinate frame, units, shapes, dtype/broadcast rules
       - lattice conventions: latvec row/col convention; frac->cart mapping
       - cutoff/summation convention: enumeration vs minimum image, cutoff definition
       - numeric tolerance conventions (without leaking expected answers)

    2) fact_delta (edge-only; for chain reuse)
       Store reusable semantic anchors, especially the tail interface contract:
       - key_fact (required): a stable contract fact describing what the step's function/class must do
       - optional: a few reusable property facts; do NOT include golden code or expected values

    3) step_cert (structured chain certificate)
       - uses_premise_ids: premises referenced (from memory_json.premise_bank or premise_delta)
       - uses_fact_ids: prior facts reused (should include expected_primary_fact_id for step>=2)
       - produces_fact_ids: must come from fact_delta IDs
       - key_fact_id: must point to a fact_delta entry and be included in produces_fact_ids

    4) ID convention (strongly recommended)
       Prefix new IDs with `c{step}.` (e.g. `c2.contract.cross_term`) to avoid collisions with semantic IDs.
    """
)


def executable_chain_cert_guide_text(lang: str | None = None) -> str:
    use_en = str(lang or "").strip().lower() in {"en", "english"}
    return EXECUTABLE_CHAIN_CERT_GUIDE_EN if use_en else EXECUTABLE_CHAIN_CERT_GUIDE_ZH


__all__ = [
    "EXECUTABLE_CHAIN_CERT_GUIDE_ZH",
    "EXECUTABLE_CHAIN_CERT_GUIDE_EN",
    "executable_chain_cert_guide_text",
]
