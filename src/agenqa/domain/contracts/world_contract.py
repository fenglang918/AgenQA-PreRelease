"""World Contract (Type1 semantic world-view contract).

This module defines the canonical structured representation of `state.memory.world_contract`
and provides deterministic normalization + merge (upsert) utilities.

Design goals:
- Layered sections (L1-L4) with points (axis/choice/note).
- Deterministic upsert-by-(level, axis) merge.
- L1-first dependency: if L1 paradigm changes, clear inherited L2/L3 points to avoid
  cross-paradigm default leakage.
- Code-derived `status`; do not trust LLM-generated `status`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from textwrap import dedent

WORLD_CONTRACT_SCHEMA_VERSION = 1
WORLD_CONTRACT_LEVELS: Tuple[str, ...] = ("L1", "L2", "L3", "L4")


WORLD_CONTRACT_MODEL_TEXT_ZH = dedent(
    """\
    ## World Contract（Type1）工作模型

    - 这里的 **Type1** 指的是“题目语义世界观不唯一”的问题：
      - 也就是同一道题可能存在两套以上彼此自洽、但会导向不同答案的语义解释。
      - 例如：某个函数到底是一元还是二元、某个参数顺序是否有语义、某个边界条件到底是 `>` 还是 `>=`。
    - `world_contract` 的作用，就是把这类 Type1 语义前提结构化固定下来。
    - 因此，`world_contract` 是题目语义真源模型；下游会把它渲染成独立的 solver-visible block，并与 Question 拼接使用。
    - 它回答的是：这道题到底在什么语义世界里成立。

    ### 层级（L1-L4）

    - `L1`：范式选择（paradigm choice）
      - 这题采用哪种语义范式/世界观。
      - 若 L1 不同，后续默认语义也可能完全不同。

    - `L2`：该范式下的默认语义（defaults under the chosen paradigm）
      - 只有在 `L1` 已选定后才成立。
      - 表示“在该范式里，通常默认如何理解”的规则。
      - 一般不需要全部写进题面；只有当某个默认值会造成真实分叉时，才需要显式提升处理。

    - `L3`：题目特定、必须钉死的语义规则
      - 用于消除这道题里特有的分叉点。
      - 例如函数签名、参数顺序、边界严格性、对象解释、题目特定 branch 规则。
      - 若不写清，solver 和 judge 可能实际上在解不同的题。

    - `L4`：solver-visible 的作答/输出约束层
      - 这里承载公开给 solver 的作答/输出要求。
      - 结构化的 judge-only 细节仍可保留在内部 answer contract 机制中；但下游会把其公开切片投影到 L4。

    ### 判断规则

    - 若改动某条规则会让题目本身变成另一道题，它属于 `world_contract`。
    - 原始 `world_contract` 仍以题目语义层为主；但 solver-facing 的最终 contract 文本可在 L4 合并公开的作答/输出约束。
    """
).rstrip()


WORLD_CONTRACT_MODEL_TEXT_EN = dedent(
    """\
    ## World Contract (Type1) working model

    - **Type1** means the semantic world of the task is not uniquely fixed:
      - the same question text may allow two or more internally consistent interpretations that lead to different answers.
      - for example: whether a function is unary or binary, whether argument order has semantics, or whether a boundary is `>` vs `>=`.
    - The role of `world_contract` is to structurally pin down those Type1 semantic assumptions.
    - Therefore, `world_contract` is the source-of-truth model for task semantics; downstream renders it as a separate solver-visible block and concatenates it with the Question.
    - It answers: under which semantic world does this question hold?

    ### Levels (L1-L4)

    - `L1`: paradigm choice
      - which semantic paradigm / world the task adopts.
      - if L1 changes, downstream defaults may also change.

    - `L2`: defaults under the chosen paradigm
      - these defaults only make sense after `L1` is fixed.
      - they represent the usual interpretation inside that paradigm.
      - they usually do not all need to be written into the question; only defaults that would otherwise create a real fork should be made explicit.

    - `L3`: question-specific semantic rules that must be pinned down
      - this level removes task-specific fork points.
      - examples include function signatures, argument order, boundary strictness, object interpretation, or task-specific branch rules.
      - if these are not made explicit, the solver and the judge may effectively solve different problems.

    - `L4`: solver-visible answer/output constraint layer
      - this level carries the public answer/output requirements visible to solvers.
      - judge-only structured details may still live in the internal answer-contract mechanism, while its public slice is projected into L4 downstream.

    ### Decision rule

    - if changing a rule would make the problem itself become a different problem, that rule belongs to `world_contract`
    - the raw `world_contract` still primarily defines task semantics, but the final solver-facing contract text may merge public answer/output constraints into L4
    """
).rstrip()


WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_ZH = dedent(
    """\
    - 【World Contract（Type1）】这里处理的是“题目语义世界观不唯一”的问题，而不是答案格式问题。
      - 你的目标是让 solver 与 judge 解的是同一道题，而不是各自按不同默认语义猜题意。
      - 你必须输出结构化 `world_contract` 对象，使用分层 `sections + points` 表示。
      - 层级使用规则：
        - `L1`：范式选择；axis 固定为 `paradigm_id`
        - `L2`：选定范式后的默认语义；只有在需要说明范式内默认口径时才写
        - `L3`：题目特定、必须钉死的语义规则，例如函数签名、参数顺序、边界严格性、对象解释
      - `L4`：承载 solver-visible 的作答/输出要求；judge-only 细节仍保留在内部 answer contract 机制
      - 若存在任何可能导致多套语义自洽解释的分叉点（尤其是 L1/L3），必须在结构化 `world_contract` 中钉死；下游会把它渲染成独立的 solver-visible block，而不是要求你把整块重复塞回 Question。
      - 不要把 provenance、validator severity、contract id 等内部 judge 配置写进 solver-facing 的 `world_contract` 内容。
    """
).rstrip()


WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_EN = dedent(
    """\
    - [World Contract (Type1)] This handles the problem that the task semantic world is not uniquely fixed, not answer-format issues.
      - Your goal is to make sure the solver and the judge are solving the same problem, instead of guessing different default semantics.
      - You must output a structured `world_contract` object using layered `sections + points`.
      - Level usage:
        - `L1`: paradigm choice; axis must be `paradigm_id`
        - `L2`: defaults under the chosen paradigm; include only when the paradigm-level default needs to be stated
        - `L3`: question-specific semantic rules that must be pinned down, such as function signatures, argument order, boundary strictness, or object interpretation
      - `L4`: carry solver-visible answer/output requirements; judge-only structured details still live in the internal answer-contract mechanism
      - If there is any plausible semantic fork point (especially in L1/L3), you must pin it down in the structured `world_contract`; downstream will render it as a separate solver-visible block rather than asking you to duplicate it inside the Question.
      - Do not put provenance, validator severity, contract ids, or other judge-only internals into solver-facing `world_contract` content.
    """
).rstrip()


def world_contract_model_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").strip().lower()
    return WORLD_CONTRACT_MODEL_TEXT_EN if lang_norm in {"en", "english"} else WORLD_CONTRACT_MODEL_TEXT_ZH


def world_contract_prompt_guidance_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").strip().lower()
    return WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_EN if lang_norm in {"en", "english"} else WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_ZH


def empty_world_contract() -> Dict[str, Any]:
    # L4 keeps the public solver-visible Type2/output-policy layer. Raw step-specific answer
    # contracts still live in ACB; downstream may project their public slice into L4 text.
    return {
        "schema_version": WORLD_CONTRACT_SCHEMA_VERSION,
        "status": "underdetermined",
        "sections": [
            {"level": "L1", "points": []},
            {"level": "L2", "points": []},
            {"level": "L3", "points": []},
            {
                "level": "L4",
                "points": [
                    {"axis": "type2.contract_source", "choice": "answer_contract_bank"},
                    {"axis": "type2.default_mode", "choice": "exact"},
                    {"axis": "type2.approx_requires", "choice": ["regime", "order"]},
                    # Numeric judging needs at least one tolerance signal. Use any-of policy
                    # to keep it flexible (abs/rel/sigfigs).
                    {"axis": "type2.numeric_requires_any_of", "choice": ["abs_tol", "rel_tol", "sig_figs"]},
                    # Unit policy: ignore|required. Default ignore for v0 to avoid over-blocking.
                    {"axis": "type2.numeric_unit_policy", "choice": "ignore"},
                    {"axis": "type2.branch_requires", "choice": ["branch_description"]},
                    # Validator severity map (override defaults). Values: error|warn|ignore.
                    {
                        "axis": "type2.issue_severity",
                        "choice": {
                            "contract_not_dict": "error",
                            "missing_contract_id": "error",
                            "invalid_contract_id_prefix": "error",
                            "duplicate_contract_id": "error",
                            "numeric_missing_tolerance": "error",
                            "numeric_missing_unit": "warn",
                            "approx_missing_regime_order": "error",
                            "branch_requires_description": "warn",
                        },
                    },
                ],
            },
        ],
        "extra_internal": {"provenance": [], "changelog": []},
    }


def _normalize_level(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().upper()
    else:
        s = str(val).strip().upper()
    if not s:
        return None
    # Accept "1".."4" as L1..L4 (common drift).
    if s in {"1", "2", "3", "4"}:
        s = f"L{s}"
    return s if s in WORLD_CONTRACT_LEVELS else None


def _normalize_axis(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def _derive_status(paradigm_id: str, l3_points: List[Dict[str, Any]]) -> str:
    if not paradigm_id:
        return "underdetermined"
    return "fixed" if l3_points else "defaulted"


def _sections_to_level_map(sections: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Convert sections -> {level -> {axis -> point}}.

    Later duplicates override earlier ones (deterministic).
    """
    level_map: Dict[str, Dict[str, Dict[str, Any]]] = {lvl: {} for lvl in WORLD_CONTRACT_LEVELS}
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        lvl = _normalize_level(sec.get("level"))
        if not lvl:
            continue
        pts = sec.get("points")
        if not isinstance(pts, list):
            continue
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            axis = _normalize_axis(pt.get("axis"))
            if not axis:
                continue
            if lvl == "L1" and axis != "paradigm_id":
                continue
            if "choice" not in pt:
                continue
            entry: Dict[str, Any] = {"axis": axis, "choice": pt.get("choice")}
            note = pt.get("note")
            if isinstance(note, str) and note.strip():
                entry["note"] = note.strip()
            # Canonicalize L1 choice to a stripped string.
            if lvl == "L1" and axis == "paradigm_id":
                choice = entry.get("choice")
                if choice is None:
                    entry["choice"] = ""
                elif isinstance(choice, str):
                    entry["choice"] = choice.strip()
                else:
                    entry["choice"] = str(choice).strip()
            level_map[lvl][axis] = entry
    return level_map


def _level_map_to_sections(level_map: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for lvl in WORLD_CONTRACT_LEVELS:
        pts = list((level_map.get(lvl) or {}).values())
        pts.sort(key=lambda p: p.get("axis") or "")
        sections.append({"level": lvl, "points": pts})
    return sections


def get_paradigm_id(world_contract: Dict[str, Any]) -> str:
    try:
        sections = world_contract.get("sections")
    except Exception:
        sections = None
    if not isinstance(sections, list):
        return ""
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        if _normalize_level(sec.get("level")) != "L1":
            continue
        pts = sec.get("points")
        if not isinstance(pts, list):
            return ""
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            if _normalize_axis(pt.get("axis")) != "paradigm_id":
                continue
            choice = pt.get("choice")
            if choice is None:
                return ""
            if isinstance(choice, str):
                return choice.strip()
            return str(choice).strip()
    return ""


def normalize_world_contract(raw: Any) -> Dict[str, Any]:
    """Normalize raw world_contract into canonical v2 object.

    Breaking change: legacy v0 shape (status/paradigm_id/overrides) is not supported.
    """
    if raw is None or not isinstance(raw, dict):
        return empty_world_contract()

    # Detect legacy v0 shape and fail fast (clean refactor: no fallback/alias).
    if "sections" not in raw and ("paradigm_id" in raw or "overrides" in raw):
        raise ValueError("legacy world_contract shape detected (paradigm_id/overrides); v2 requires sections+points")

    # Accept both list-sections and dict-sections ({L1: [...], ...}) to be robust.
    sections_in = raw.get("sections")
    sections_norm: List[Dict[str, Any]] = []
    if isinstance(sections_in, dict):
        for k, v in sections_in.items():
            lvl = _normalize_level(k)
            if not lvl or not isinstance(v, list):
                continue
            sections_norm.append({"level": lvl, "points": v})
    elif isinstance(sections_in, list):
        sections_norm = [s for s in sections_in if isinstance(s, dict)]
    else:
        sections_norm = []

    level_map = _sections_to_level_map(sections_norm)
    sections_out = _level_map_to_sections(level_map)

    # Preserve extra_internal (forward-compat) but normalize provenance/changelog containers.
    extra_in = raw.get("extra_internal")
    extra_out: Dict[str, Any] = dict(extra_in) if isinstance(extra_in, dict) else {}

    prov = extra_out.get("provenance")
    if isinstance(prov, dict):
        prov = [prov]
    if not isinstance(prov, list):
        prov = []
    extra_out["provenance"] = prov

    ch = extra_out.get("changelog")
    if isinstance(ch, dict):
        ch = [ch]
    if not isinstance(ch, list):
        ch = []
    extra_out["changelog"] = ch

    paradigm_id = ""
    l3_pts: List[Dict[str, Any]] = []
    for sec in sections_out:
        if sec.get("level") == "L1":
            pts = sec.get("points") or []
            if pts and isinstance(pts, list):
                for pt in pts:
                    if isinstance(pt, dict) and pt.get("axis") == "paradigm_id":
                        choice = pt.get("choice")
                        paradigm_id = choice.strip() if isinstance(choice, str) else (str(choice).strip() if choice is not None else "")
                        break
        if sec.get("level") == "L3":
            pts = sec.get("points")
            l3_pts = pts if isinstance(pts, list) else []

    return {
        "schema_version": WORLD_CONTRACT_SCHEMA_VERSION,
        "status": _derive_status(paradigm_id, l3_pts),
        "sections": sections_out,
        "extra_internal": extra_out,
    }


def merge_world_contract(
    old: Any,
    new: Any,
    *,
    role: str,
    step: int,
    round: int,
    raw_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministically merge world_contract by upserting points by (level, axis)."""
    base = normalize_world_contract(old)
    update = normalize_world_contract(new)

    base_map = _sections_to_level_map(base.get("sections") if isinstance(base.get("sections"), list) else [])
    upd_map = _sections_to_level_map(update.get("sections") if isinstance(update.get("sections"), list) else [])

    old_pid = get_paradigm_id(base)
    new_pid = get_paradigm_id(update)

    ops: List[Dict[str, Any]] = []

    # L1-first dependency rule: if L1 changes (and new is non-empty), clear inherited L2/L3.
    if new_pid and new_pid != old_pid:
        for lvl in ("L2", "L3"):
            for pt in (base_map.get(lvl) or {}).values():
                ops.append({"level": lvl, "axis": pt.get("axis", ""), "old": pt.get("choice"), "new": None})
        base_map["L2"] = {}
        base_map["L3"] = {}

    # Upsert updates.
    for lvl in WORLD_CONTRACT_LEVELS:
        for axis, new_pt in (upd_map.get(lvl) or {}).items():
            if not axis:
                continue
            old_pt = (base_map.get(lvl) or {}).get(axis)
            old_choice = old_pt.get("choice") if isinstance(old_pt, dict) else None
            new_choice = new_pt.get("choice")
            base_map[lvl][axis] = dict(new_pt)
            if old_pt is None or old_choice != new_choice:
                ops.append({"level": lvl, "axis": axis, "old": old_choice, "new": new_choice})

    merged_sections = _level_map_to_sections(base_map)
    merged = {
        "schema_version": WORLD_CONTRACT_SCHEMA_VERSION,
        "sections": merged_sections,
        "extra_internal": dict(base.get("extra_internal") or {}),
    }

    # Deterministically derive status.
    pid = get_paradigm_id({"sections": merged_sections})
    l3_points: List[Dict[str, Any]] = []
    for sec in merged_sections:
        if sec.get("level") == "L3":
            l3_points = sec.get("points") if isinstance(sec.get("points"), list) else []
            break
    merged["status"] = _derive_status(pid, l3_points)

    # Append provenance + changelog.
    extra = merged["extra_internal"]
    prov = extra.get("provenance")
    if not isinstance(prov, list):
        prov = []
    else:
        prov = list(prov)
    prov.append(
        {
            "role": role,
            "step": int(step),
            "round": int(round),
            "raw_ref": raw_ref or "",
        }
    )
    extra["provenance"] = prov

    if ops:
        ch = extra.get("changelog")
        if not isinstance(ch, list):
            ch = []
        else:
            ch = list(ch)
        ch.append({"step": int(step), "round": int(round), "role": role, "ops": ops})
        extra["changelog"] = ch

    merged["extra_internal"] = extra
    return merged


__all__ = [
    "WORLD_CONTRACT_SCHEMA_VERSION",
    "WORLD_CONTRACT_LEVELS",
    "WORLD_CONTRACT_MODEL_TEXT_ZH",
    "WORLD_CONTRACT_MODEL_TEXT_EN",
    "WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_ZH",
    "WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_EN",
    "empty_world_contract",
    "get_paradigm_id",
    "normalize_world_contract",
    "merge_world_contract",
    "world_contract_model_text",
    "world_contract_prompt_guidance_text",
]
