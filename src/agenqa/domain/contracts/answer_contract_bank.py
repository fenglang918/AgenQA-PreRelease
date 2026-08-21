"""Answer Contract Bank (ACB-lite) for Type2 judging / output-protocol governance.

Design goals (borrowed from the ACB design, adapted to this repo):
- Step-scoped raw contracts (keep judge-facing details in ACB; project only the public
  solver-visible slice into world_contract/L4 text downstream).
- Deterministic generation (do not rely on LLM to emit the contract).
- Deterministic validation (fail-fast signals, plus bounded debug artifacts).
- OR semantics via step_certs entry: answer_contract_ids=[...].
"""

from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ACB_ID_PREFIX = "ac_"

# Keep debug artifacts bounded to avoid bloating state.json.
MAX_VALIDATION_ERRORS = 50
MAX_VALIDATION_CANDIDATE_STEPS = 50

_INLINE_MATH_TOKEN_RE = re.compile(r"\$(.+?)\$|\\\((.+?)\\\)")
_IN_TERMS_OF_RE = re.compile(r"in terms of(?P<body>.*?)(?:[\.;\n]|$)", re.IGNORECASE | re.DOTALL)


ANSWER_CONTRACT_MODEL_TEXT_ZH = dedent(
    """\
    ## Answer Contract（Type2）工作模型

    - 这里的 Type2 指的是：在题目语义已经固定之后，什么 final answer 算合格。
    - `answer_contract` 不是题目语义真源，也不是 judge 的完整实现。
    - 它回答的是：在这道已经定义好的题里，什么 final answer 算可接受。

    ### 与 World Contract 的边界

    - `world_contract` 定义题目语义层：
      - 范式
      - 函数/对象定义
      - 参数顺序语义
      - 边界规则本身
    - `answer_contract` 定义答案接受层：
      - final answer 的写法要求
      - final answer 的接受边界
      - judge 在高歧义场景下需要的局部 witness 规格
    - 在当前架构里：
      - 原始 answer contract 结构仍保留在内部 ACB；
      - 对 solver 公开的 output-spec 切片会被投影到 `world_contract` 的 L4 文本层。

    - 判断规则：
      - 如果改动一条规则会让题目本身变成另一道题，它属于 `world_contract`
      - 如果题目不变，只是 final answer 的接受边界变化了，它属于 `answer_contract`

    ### 完整模型（概念上）

    - `answer_style`
      - final answer 怎么写
    - `answer_semantics`
      - 什么 final answer 算可接受
    - `support_witness_spec`
      - judge 在高歧义场景下可能需要什么局部 witness 规格

    ### 当前实现约束

    - 当前仓库里的 builder / schema 仍处于过渡状态。
    - Derivation 正在从旧的 `derivation_spec` 过渡到 `answer_style / answer_semantics / support_witness` 三层 payload。
    """
).rstrip()


ANSWER_CONTRACT_MODEL_TEXT_EN = dedent(
    """\
    ## Answer Contract (Type2) working model

    - Type2 means: after task semantics are already fixed, what counts as an acceptable final answer.
    - `answer_contract` is not the source-of-truth for task semantics, and not the full judge implementation.
    - It answers: for this already-defined problem, what final answer is acceptable?

    ### Boundary vs World Contract

    - `world_contract` defines the task-semantic layer:
      - paradigm
      - function/object definitions
      - argument-order semantics
      - boundary rules themselves
    - `answer_contract` defines the answer-acceptance layer:
      - how the final answer should be written
      - what the acceptance boundary is
      - which local witness specs may help the judge in high-ambiguity cases
    - In the current architecture:
      - raw answer-contract structure stays in the internal ACB;
      - the public solver-visible output-spec slice may be projected into `world_contract` L4 text downstream.

    - Decision rule:
      - if changing a rule would make the problem itself become a different problem, it belongs to `world_contract`
      - if the problem stays the same but the acceptance boundary of the final answer changes, it belongs to `answer_contract`

    ### Full model (conceptually)

    - `answer_style`
      - how the final answer should be written
    - `answer_semantics`
      - what final answer counts as acceptable
    - `support_witness_spec`
      - which local witness specs may help the judge in high-ambiguity cases

    ### Current implementation constraint

    - The current builder / schema in this repository is still transitional.
    - Derivation is moving from the legacy `derivation_spec` subset toward a three-layer payload: `answer_style / answer_semantics / support_witness`.
    """
).rstrip()


def answer_contract_model_text(lang: str | None = None) -> str:
    lang_norm = (lang or "").strip().lower()
    return ANSWER_CONTRACT_MODEL_TEXT_EN if lang_norm in {"en", "english"} else ANSWER_CONTRACT_MODEL_TEXT_ZH


def _extract_type2_policy(world_contract: Any) -> Dict[str, Any]:
    """Extract Type2 (L4) policy points from world_contract.

    world_contract is a canonical v2 object (sections+points), but this function is
    tolerant to missing/partial shapes.
    """
    out: Dict[str, Any] = {}
    if not isinstance(world_contract, dict):
        return out
    sections = world_contract.get("sections")
    if not isinstance(sections, list):
        return out
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        if str(sec.get("level") or "").strip().upper() != "L4":
            continue
        pts = sec.get("points")
        if not isinstance(pts, list):
            continue
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            axis = _as_str(pt.get("axis")).strip()
            if not axis:
                continue
            out[axis] = pt.get("choice")
        break
    return out


def _as_str_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, tuple):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def extract_answer_contract_context(
    memory: Dict[str, Any],
    *,
    step: int,
) -> Dict[str, Any]:
    """Extract a compact, structured Type2 contract context for evaluation/audit.

    This is internal-only and MUST NOT be exposed to solver prompts.
    """
    mem = dict(memory or {})
    step_i = int(step)

    policy = _extract_type2_policy(mem.get("world_contract"))

    cand = mem.get("answer_contract_validation_candidates")
    cand_step: Dict[str, Any] | None = None
    if isinstance(cand, dict):
        raw = cand.get(str(step_i))
        cand_step = raw if isinstance(raw, dict) else None

    # Load ids from latest cert.
    ids: List[str] = []
    step_certs = mem.get("step_certs") if isinstance(mem.get("step_certs"), list) else []
    for item in reversed(step_certs):
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "") != "answer_contract_cert":
            continue
        try:
            if int(item.get("step")) != step_i:
                continue
        except Exception:
            continue
        raw_ids = item.get("answer_contract_ids")
        if isinstance(raw_ids, list):
            ids = [str(x) for x in raw_ids if str(x).strip()]
        break

    bank = mem.get("answer_contract_bank") if isinstance(mem.get("answer_contract_bank"), list) else []
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in bank:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            by_id[c["id"]] = c
    contracts = [by_id[i] for i in ids if i in by_id] if ids else []
    if not contracts and cand_step is not None:
        # Candidate snapshot already contains summarized contracts.
        return {
            "step": step_i,
            "type2_policy": policy,
            "answer_contract_ids": cand_step.get("answer_contract_ids") or ids,
            "answer_contracts": cand_step.get("answer_contracts"),
            "validation": {
                "error_count": cand_step.get("error_count"),
                "warn_count": cand_step.get("warn_count"),
                "issue_types_error": cand_step.get("issue_types_error"),
                "issue_types_warn": cand_step.get("issue_types_warn"),
            },
        }

    # Prefer canonical bank contracts (summarized).
    csum = [_summarize_contract(c) for c in contracts[:6]]
    validation = {}
    if cand_step is not None:
        validation = {
            "error_count": cand_step.get("error_count"),
            "warn_count": cand_step.get("warn_count"),
            "issue_types_error": cand_step.get("issue_types_error"),
            "issue_types_warn": cand_step.get("issue_types_warn"),
        }
    return {
        "step": step_i,
        "type2_policy": policy,
        "answer_contract_ids": ids,
        "answer_contracts": csum,
        "validation": validation,
    }


def _summarize_output_spec(c: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only solver-visible output-spec fields."""
    out: Dict[str, Any] = {}
    qtype = c.get("question_type")
    if qtype is not None:
        out["question_type"] = qtype

    if _qtype_norm(qtype) == "Derivation":
        answer_style = _sanitize_answer_style(c.get("answer_style"))
        if answer_style:
            out["answer_style"] = answer_style

        semantics = _sanitize_answer_semantics(c.get("answer_semantics"))
        if semantics:
            public_semantics: Dict[str, Any] = {}
            for key in (
                "answer_object",
                "acceptance_mode",
                "branch_policy",
                "allowed_symbols",
                "required_qualifiers",
            ):
                if key in semantics:
                    public_semantics[key] = semantics[key]
            if public_semantics:
                out["answer_semantics"] = public_semantics
        return out

    mode = ((c.get("exactness") or {}) if isinstance(c.get("exactness"), dict) else {}).get("mode")
    if mode is None:
        mode = c.get("mode")
    if mode is not None:
        out["mode"] = mode
    answer_shape = c.get("answer_shape")
    if answer_shape is not None:
        out["answer_shape"] = answer_shape

    approx = c.get("approx") if isinstance(c.get("approx"), dict) else None
    if approx is not None:
        out["approx"] = {
            k: approx.get(k)
            for k in ("regime", "order")
            if k in approx
        }
        if not out["approx"]:
            out["approx"] = {}

    branch = c.get("branch") if isinstance(c.get("branch"), dict) else None
    if branch:
        branch_out = {
            "allow_branches": branch.get("allow_branches"),
            "branch_description_required": branch.get("branch_description_required"),
        }
        out["branch"] = {k: v for k, v in branch_out.items() if v is not None}

    numeric = c.get("numeric") if isinstance(c.get("numeric"), dict) else None
    if numeric:
        numeric_out = {
            "abs_tol": numeric.get("abs_tol"),
            "rel_tol": numeric.get("rel_tol"),
            "sig_figs": numeric.get("sig_figs"),
            "unit": numeric.get("unit"),
        }
        out["numeric"] = {k: v for k, v in numeric_out.items() if v not in (None, "")}

    return out


def extract_answer_output_spec_context(
    memory: Dict[str, Any],
    *,
    step: int,
) -> Dict[str, Any]:
    """Extract solver-visible answer output specs only.

    This is the public/output-facing slice of answer contracts. Internal judge
    configuration such as contract ids, template metadata, provenance, and
    validation state is intentionally excluded.
    """
    mem = dict(memory or {})
    step_i = int(step)

    cand = mem.get("answer_contract_validation_candidates")
    cand_step: Dict[str, Any] | None = None
    if isinstance(cand, dict):
        raw = cand.get(str(step_i))
        cand_step = raw if isinstance(raw, dict) else None

    ids: List[str] = []
    step_certs = mem.get("step_certs") if isinstance(mem.get("step_certs"), list) else []
    for item in reversed(step_certs):
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "") != "answer_contract_cert":
            continue
        try:
            if int(item.get("step")) != step_i:
                continue
        except Exception:
            continue
        raw_ids = item.get("answer_contract_ids")
        if isinstance(raw_ids, list):
            ids = [str(x) for x in raw_ids if str(x).strip()]
        break

    bank = mem.get("answer_contract_bank") if isinstance(mem.get("answer_contract_bank"), list) else []
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in bank:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            by_id[c["id"]] = c
    contracts = [by_id[i] for i in ids if i in by_id] if ids else []
    if not contracts and cand_step is not None:
        raw_specs = cand_step.get("answer_contracts")
        specs = []
        if isinstance(raw_specs, list):
            specs = [_summarize_output_spec(x) for x in raw_specs if isinstance(x, dict)]
        return {
            "step": step_i,
            "answer_output_specs": specs,
        }

    return {
        "step": step_i,
        "answer_output_specs": [_summarize_output_spec(c) for c in contracts[:6]],
    }


def build_answer_output_spec_prompt_section(
    memory: Dict[str, Any],
    *,
    step: int,
    lang: str | None = None,
) -> str:
    """Build a solver-visible prompt block for answer output requirements."""
    ctx = extract_answer_output_spec_context(memory, step=step)
    specs = ctx.get("answer_output_specs")
    if not isinstance(specs, list):
        return ""
    specs = [x for x in specs if isinstance(x, dict) and x]
    if not specs:
        return ""

    use_en = _is_en(lang)
    lines: List[str] = []
    if use_en:
        lines.extend(
            [
                "## Additional Answer Requirements",
                "The following output specs are solver-visible and describe acceptable answer/output requirements only.",
                "If multiple specs are listed, satisfying any one of them is acceptable.",
            ]
        )
        label = "Output spec"
    else:
        lines.extend(
            [
                "## 补充答案要求",
                "以下 output spec 对 solver 可见，只描述可接受的答案/输出要求，不包含内部 judge 配置。",
                "如果列出了多个 spec，满足其中任意一个即可。",
            ]
        )
        label = "Output spec"

    for idx, spec in enumerate(specs, start=1):
        try:
            spec_text = json.dumps(spec, ensure_ascii=False)
        except Exception:
            spec_text = _as_str(spec).strip()
        if not spec_text:
            continue
        lines.append(f"- {label} {idx}: {spec_text}")
    return "\n".join(lines).strip()


def _is_en(lang: str | None) -> bool:
    return str(lang or "").strip().lower() in {"en", "english"}


def _as_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _bounded_append(items: List[Any], item: Any, *, max_items: int) -> List[Any]:
    out = list(items or [])
    out.append(item)
    if max_items > 0 and len(out) > max_items:
        out = out[-max_items:]
    return out


def _bounded_step_dict_insert(d: Dict[str, Any], key_step: int, value: Any, *, max_steps: int) -> Dict[str, Any]:
    out = dict(d or {})
    out[str(int(key_step))] = value
    if max_steps > 0 and len(out) > max_steps:
        # Drop smallest step numbers first (stable + deterministic).
        steps: List[int] = []
        for k in out.keys():
            try:
                steps.append(int(str(k)))
            except Exception:
                continue
        steps.sort()
        for s in steps:
            if len(out) <= max_steps:
                break
            out.pop(str(s), None)
    return out


def _qtype_norm(question_type: Any) -> str:
    v = str(question_type or "").strip()
    if not v:
        return ""
    # Keep canonical names aligned with the rest of the repo.
    v0 = v.lower()
    if v0 in {"mcq", "choice", "single", "single_choice", "single-choice"}:
        return "MCQ"
    if v0 in {"derivation", "derive", "symbolic", "proof"}:
        return "Derivation"
    if v0 in {"numeric", "calc", "calculation", "compute", "number", "estimate", "estimation"}:
        return "Numeric"
    return v


def _coerce_optional_bool(val: Any) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    return None


def _extract_math_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for m in _INLINE_MATH_TOKEN_RE.finditer(str(text or "")):
        token = (m.group(1) or m.group(2) or "").strip()
        if token:
            tokens.append(token)
    return tokens


def _extract_in_terms_of_symbols(question_text: str) -> List[str]:
    q = str(question_text or "")
    if not q.strip():
        return []
    m = _IN_TERMS_OF_RE.search(q)
    if not m:
        return []
    body = m.group("body") or ""
    seen: set[str] = set()
    out: List[str] = []
    for token in _extract_math_tokens(body):
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _sanitize_derivation_spec(val: Any) -> Dict[str, Any]:
    if not isinstance(val, dict):
        return {}

    out: Dict[str, Any] = {}

    def _sanitize_entries(
        entries: Any,
        allowed_fields: Sequence[str],
        *,
        list_field_names: Optional[Sequence[str]] = None,
        bool_field_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        allowed = set(str(x) for x in allowed_fields)
        list_fields = set(str(x) for x in (list_field_names or []))
        bool_fields = set(str(x) for x in (bool_field_names or []))
        items: List[Dict[str, Any]] = []
        if not isinstance(entries, list):
            return items
        for item in entries:
            if not isinstance(item, dict):
                continue
            row: Dict[str, Any] = {}
            for key in allowed:
                if key not in item:
                    continue
                raw = item.get(key)
                if key in list_fields:
                    vals = _as_str_list(raw)
                    if vals:
                        row[key] = vals
                    continue
                if key in bool_fields:
                    b = _coerce_optional_bool(raw)
                    if b is not None:
                        row[key] = b
                    continue
                s = _as_str(raw).strip()
                if s:
                    row[key] = s
            if row:
                items.append(row)
        return items

    answer_form = val.get("answer_form")
    if isinstance(answer_form, dict):
        af: Dict[str, Any] = {}
        for key in ("boxed", "single_expression", "closed_form"):
            b = _coerce_optional_bool(answer_form.get(key))
            if b is not None:
                af[key] = b
        in_terms = _as_str_list(answer_form.get("in_terms_of"))
        if in_terms:
            af["in_terms_of"] = in_terms
        if af:
            out["answer_form"] = af

    allowed_symbols = _as_str_list(val.get("allowed_symbols"))
    if allowed_symbols:
        out["allowed_symbols"] = allowed_symbols

    domain_constraints = _as_str_list(val.get("domain_constraints"))
    if domain_constraints:
        out["domain_constraints"] = domain_constraints

    symbol_signature = _sanitize_entries(
        val.get("symbol_signature"),
        ("symbol", "form", "render_as", "required_in_question"),
        bool_field_names=("required_in_question",),
    )
    if symbol_signature:
        out["symbol_signature"] = symbol_signature

    parameter_order = _sanitize_entries(
        val.get("parameter_order"),
        ("symbol", "args", "render_as", "required_in_question"),
        list_field_names=("args",),
        bool_field_names=("required_in_question",),
    )
    if parameter_order:
        out["parameter_order"] = parameter_order

    boundary_strictness = _sanitize_entries(
        val.get("boundary_strictness"),
        ("phrase", "operator", "required_in_question"),
        bool_field_names=("required_in_question",),
    )
    if boundary_strictness:
        out["boundary_strictness"] = boundary_strictness

    operator_semantics = _sanitize_entries(
        val.get("operator_semantics"),
        ("symbol", "must_preserve", "description"),
        bool_field_names=("must_preserve",),
    )
    if operator_semantics:
        out["operator_semantics"] = operator_semantics

    return out


def _build_default_derivation_spec(question_text: str, answer_text: str) -> Dict[str, Any]:
    q = str(question_text or "")
    q_lower = q.lower()
    answer = str(answer_text or "")

    answer_form: Dict[str, Any] = {}
    if "\\boxed" in answer or "\\boxed" in q:
        answer_form["boxed"] = True
    if any(
        marker in q_lower
        for marker in (
            "single expression",
            "single inequality",
            "single equation",
            "one expression",
            "closed-form expression",
            "single latex",
            "单个表达式",
            "单个不等式",
            "单个方程",
            "一个表达式",
            "一个不等式",
            "一个方程",
        )
    ):
        answer_form["single_expression"] = True
    if "closed-form" in q_lower or "闭式" in q:
        answer_form["closed_form"] = True

    allowed_symbols = _extract_in_terms_of_symbols(q)
    if allowed_symbols:
        answer_form["in_terms_of"] = allowed_symbols

    spec: Dict[str, Any] = {}
    if answer_form:
        spec["answer_form"] = answer_form
    if allowed_symbols:
        spec["allowed_symbols"] = allowed_symbols
    return spec


def _merge_derivation_spec(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if not base:
        return dict(override or {})
    if not override:
        return dict(base)
    out = dict(base)
    for key, val in (override or {}).items():
        if key == "answer_form" and isinstance(val, dict) and isinstance(out.get("answer_form"), dict):
            merged = dict(out["answer_form"])
            merged.update(val)
            out["answer_form"] = merged
            continue
        out[key] = val
    return out


_DERIVATION_FORM_VALUES = {
    "single_expression",
    "single_equation",
    "single_inequality",
    "set",
    "tuple",
}
_DERIVATION_ANSWER_OBJECT_VALUES = {
    "symbolic_expr",
    "equation",
    "inequality",
    "set",
    "tuple",
}
_DERIVATION_ACCEPTANCE_MODE_VALUES = {"exact", "approx", "either"}
_DERIVATION_WITNESS_TYPES = {"branch", "boundary", "signature", "dependency", "equivalence_cue"}


def _compact_json_value(val: Any) -> str:
    try:
        return json.dumps(val, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return _as_str(val).strip()


def _sanitize_answer_style(val: Any) -> Dict[str, Any]:
    if not isinstance(val, dict):
        return {}
    out: Dict[str, Any] = {}
    boxed = _coerce_optional_bool(val.get("boxed"))
    if boxed is not None:
        out["boxed"] = boxed
    form = _as_str(val.get("form")).strip()
    if form in _DERIVATION_FORM_VALUES:
        out["form"] = form
    notes = _as_str_list(val.get("rendering_notes"))
    if notes:
        out["rendering_notes"] = notes
    return out


def _sanitize_branch_policy(val: Any) -> Dict[str, Any]:
    if not isinstance(val, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("allow_branches", "require_complete_enumeration", "branch_description_required"):
        b = _coerce_optional_bool(val.get(key))
        if b is not None:
            out[key] = b
    return out


def _sanitize_answer_semantics(val: Any) -> Dict[str, Any]:
    if not isinstance(val, dict):
        return {}
    out: Dict[str, Any] = {}
    answer_object = _as_str(val.get("answer_object")).strip()
    if answer_object in _DERIVATION_ANSWER_OBJECT_VALUES:
        out["answer_object"] = answer_object

    acceptance_mode = _as_str(val.get("acceptance_mode")).strip().lower()
    if acceptance_mode in _DERIVATION_ACCEPTANCE_MODE_VALUES:
        out["acceptance_mode"] = acceptance_mode

    branch_policy = _sanitize_branch_policy(val.get("branch_policy"))
    if branch_policy:
        out["branch_policy"] = branch_policy

    allowed_symbols = _as_str_list(val.get("allowed_symbols"))
    if allowed_symbols:
        out["allowed_symbols"] = allowed_symbols

    required_qualifiers = _as_str_list(val.get("required_qualifiers"))
    if required_qualifiers:
        out["required_qualifiers"] = required_qualifiers

    equivalence_rules = _as_str_list(val.get("equivalence_rules"))
    if equivalence_rules:
        out["equivalence_rules"] = equivalence_rules

    return out


def _sanitize_support_witness(val: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not isinstance(val, list):
        return items
    for item in val:
        if not isinstance(item, dict):
            continue
        witness_type = _as_str(item.get("type")).strip()
        statement = _as_str(item.get("statement")).strip()
        if not witness_type or not statement:
            continue
        if witness_type not in _DERIVATION_WITNESS_TYPES:
            continue
        items.append({"type": witness_type, "statement": statement})
    return items


def _sanitize_answer_contract_payload(val: Any) -> Dict[str, Any]:
    if not isinstance(val, dict):
        return {}
    out: Dict[str, Any] = {}
    answer_style = _sanitize_answer_style(val.get("answer_style"))
    if answer_style:
        out["answer_style"] = answer_style
    answer_semantics = _sanitize_answer_semantics(val.get("answer_semantics"))
    if answer_semantics:
        out["answer_semantics"] = answer_semantics
    support_witness = _sanitize_support_witness(val.get("support_witness"))
    if support_witness:
        out["support_witness"] = support_witness
    return out


def _legacy_derivation_spec_to_answer_contract_payload(derivation_spec: Any) -> Dict[str, Any]:
    spec = _sanitize_derivation_spec(derivation_spec)
    if not spec:
        return {}

    answer_style: Dict[str, Any] = {}
    answer_semantics: Dict[str, Any] = {}
    support_witness: List[Dict[str, Any]] = []

    answer_form = spec.get("answer_form") if isinstance(spec.get("answer_form"), dict) else {}
    if isinstance(answer_form, dict):
        boxed = _coerce_optional_bool(answer_form.get("boxed"))
        if boxed is not None:
            answer_style["boxed"] = boxed
        if answer_form.get("single_expression") is True:
            answer_style["form"] = "single_expression"

        allowed_symbols = _as_str_list(answer_form.get("in_terms_of"))
        if allowed_symbols:
            answer_semantics["allowed_symbols"] = allowed_symbols

        qualifiers: List[str] = []
        if answer_form.get("closed_form") is True:
            qualifiers.append("closed_form")
        if qualifiers:
            answer_semantics["required_qualifiers"] = qualifiers

    allowed_symbols = _as_str_list(spec.get("allowed_symbols"))
    if allowed_symbols:
        answer_semantics["allowed_symbols"] = allowed_symbols

    domain_constraints = _as_str_list(spec.get("domain_constraints"))
    if domain_constraints:
        existing = list(answer_semantics.get("required_qualifiers") or [])
        answer_semantics["required_qualifiers"] = existing + domain_constraints

    witness_type_map = {
        "symbol_signature": "signature",
        "parameter_order": "signature",
        "boundary_strictness": "boundary",
        "operator_semantics": "equivalence_cue",
    }
    for key, witness_type in witness_type_map.items():
        entries = spec.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            support_witness.append(
                {
                    "type": witness_type,
                    "statement": f"{key}: {_compact_json_value(entry)}",
                }
            )

    payload: Dict[str, Any] = {}
    if answer_style:
        payload["answer_style"] = answer_style
    if answer_semantics:
        payload["answer_semantics"] = answer_semantics
    if support_witness:
        payload["support_witness"] = support_witness
    return _sanitize_answer_contract_payload(payload)


def _merge_answer_contract_payload(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if not base:
        return dict(override or {})
    if not override:
        return dict(base)

    out = dict(base)
    for key in ("answer_style", "answer_semantics"):
        base_part = out.get(key)
        override_part = override.get(key)
        if isinstance(base_part, dict) and isinstance(override_part, dict):
            merged = dict(base_part)
            merged.update(override_part)
            out[key] = merged
        elif isinstance(override_part, dict):
            out[key] = dict(override_part)

    if isinstance(override.get("support_witness"), list):
        out["support_witness"] = list(override.get("support_witness") or [])

    return _sanitize_answer_contract_payload(out)


def _infer_answer_style_form(question_text: str) -> str:
    q = str(question_text or "")
    q_lower = q.lower()
    if any(marker in q_lower for marker in ("single inequality",)) or any(marker in q for marker in ("单个不等式", "一个不等式")):
        return "single_inequality"
    if any(marker in q_lower for marker in ("single equation", "one equation")) or any(marker in q for marker in ("单个方程", "一个方程")):
        return "single_equation"
    if any(
        marker in q_lower
        for marker in ("single expression", "one expression", "closed-form expression", "single latex")
    ) or any(marker in q for marker in ("单个表达式", "一个表达式")):
        return "single_expression"
    return ""


def _answer_object_for_form(form: str) -> str:
    if form == "single_equation":
        return "equation"
    if form == "single_inequality":
        return "inequality"
    if form in {"set", "tuple"}:
        return form
    return "symbolic_expr"


def _answer_shape_for_object(answer_object: str) -> str:
    if answer_object == "equation":
        return "equation"
    if answer_object == "inequality":
        return "inequality"
    if answer_object == "set":
        return "set"
    if answer_object == "tuple":
        return "tuple"
    return "symbolic_expr"


def _build_default_derivation_answer_contract_payload(
    question_text: str,
    answer_text: str,
    *,
    acceptance_mode: str,
) -> Dict[str, Any]:
    q = str(question_text or "")
    answer = str(answer_text or "")

    answer_style: Dict[str, Any] = {}
    if "\\boxed" in answer or "\\boxed" in q:
        answer_style["boxed"] = True

    form = _infer_answer_style_form(q)
    if form:
        answer_style["form"] = form

    answer_semantics: Dict[str, Any] = {
        "answer_object": _answer_object_for_form(form),
        "acceptance_mode": acceptance_mode if acceptance_mode in _DERIVATION_ACCEPTANCE_MODE_VALUES else "exact",
        "branch_policy": {"allow_branches": False, "require_complete_enumeration": False},
    }

    allowed_symbols = _extract_in_terms_of_symbols(q)
    if allowed_symbols:
        answer_semantics["allowed_symbols"] = allowed_symbols

    required_qualifiers: List[str] = []
    if "closed-form" in q.lower() or "闭式" in q:
        required_qualifiers.append("closed_form")
    if required_qualifiers:
        answer_semantics["required_qualifiers"] = required_qualifiers

    payload = {
        "answer_style": answer_style,
        "answer_semantics": answer_semantics,
        "support_witness": [],
    }
    return _sanitize_answer_contract_payload(payload)


def _allows_approx(question_text: str) -> bool:
    """Heuristic: detect explicit 'approx allowed' semantics in the question text.

    NOTE: keep this conservative. Only treat explicit permission as allow_approx.
    """
    q = str(question_text or "").lower()
    if not q.strip():
        return False
    markers = (
        "允许近似",
        "可以近似",
        "允许估算",
        "可以估算",
        "you may approximate",
        "approximation is allowed",
        "approximation allowed",
        "you can approximate",
        "you may use approximation",
        "允许用近似",
        "允许使用近似",
    )
    return any(m in q for m in markers)


def _id_for(step: int, question_type: str, mode: str, version: int = 1) -> str:
    qt = _qtype_norm(question_type) or "unknown"
    m = str(mode or "").strip().lower() or "exact"
    return f"{ACB_ID_PREFIX}step{int(step)}_{qt.lower()}_{m}_v{int(version)}"


def _summarize_contract(c: Dict[str, Any]) -> Dict[str, Any]:
    """Keep a compact snapshot for prompts/debug artifacts."""
    out: Dict[str, Any] = {}
    out["id"] = c.get("id")
    out["source_step"] = c.get("source_step")
    out["question_type"] = c.get("question_type")
    out["mode"] = ((c.get("exactness") or {}) if isinstance(c.get("exactness"), dict) else {}).get("mode")
    out["answer_shape"] = c.get("answer_shape")
    out["judge"] = c.get("judge")
    tmpl = c.get("template") if isinstance(c.get("template"), dict) else {}
    if tmpl:
        out["template_id"] = tmpl.get("template_id")
        out["template_version"] = tmpl.get("template_version")
    approx = c.get("approx") if isinstance(c.get("approx"), dict) else {}
    if approx:
        out["approx"] = {"regime": approx.get("regime"), "order": approx.get("order")}
    branch = c.get("branch") if isinstance(c.get("branch"), dict) else {}
    if branch:
        out["branch"] = {"allow_branches": branch.get("allow_branches"), "branch_description_required": branch.get("branch_description_required")}
    numeric = c.get("numeric") if isinstance(c.get("numeric"), dict) else {}
    if numeric:
        out["numeric"] = {
            "abs_tol": numeric.get("abs_tol"),
            "rel_tol": numeric.get("rel_tol"),
            "sig_figs": numeric.get("sig_figs"),
            "unit": numeric.get("unit"),
        }
    answer_style = _sanitize_answer_style(c.get("answer_style"))
    if answer_style:
        out["answer_style"] = answer_style
    answer_semantics = _sanitize_answer_semantics(c.get("answer_semantics"))
    if answer_semantics:
        out["answer_semantics"] = answer_semantics
    support_witness = _sanitize_support_witness(c.get("support_witness"))
    if support_witness:
        out["support_witness"] = support_witness
    return out


def make_default_answer_contracts(
    *,
    step: int,
    question_type: Any,
    question: str,
    answer: str,
    abs_tol: Any = None,
    rel_tol: Any = None,
    sig_figs: Any = None,
    unit: Any = None,
    answer_contract_payload: Optional[Dict[str, Any]] = None,
    derivation_spec: Optional[Dict[str, Any]] = None,
    version: int = 1,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build step-scoped default answer contracts deterministically.

    Returns (answer_contract_ids, answer_contracts).
    """
    qt = _qtype_norm(question_type)
    ids: List[str] = []
    contracts: List[Dict[str, Any]] = []

    allow_approx = _allows_approx(question or "")
    modes = ["exact", "approx"] if allow_approx else ["exact"]

    for mode in modes:
        cid = _id_for(step, qt, mode, version=version)
        ids.append(cid)

        template_id = f"acb_lite_{(_qtype_norm(qt) or 'unknown').lower()}_{str(mode).strip().lower()}_v{int(version)}"
        c: Dict[str, Any] = {
            "id": cid,
            "source_step": int(step),
            "question_type": qt,
            "exactness": {"mode": mode},
            "template": {"template_id": template_id, "template_version": int(version)},
        }

        if qt == "MCQ":
            c["answer_shape"] = "mcq_single"
            c["judge"] = "mcq_unique_index"
        elif qt == "Numeric":
            c["answer_shape"] = "numeric_scalar"
            c["judge"] = "numeric_equivalence"
            c["numeric"] = {
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
                "sig_figs": sig_figs,
                "unit": _as_str(unit).strip(),
            }
        else:
            # Default Derivation / others: symbolic expression equivalence.
            c["answer_shape"] = "symbolic_expr"
            c["judge"] = "expression_equivalence"
            if qt == "Derivation":
                payload_default = _build_default_derivation_answer_contract_payload(
                    question,
                    answer,
                    acceptance_mode=mode,
                )
                payload_override = _sanitize_answer_contract_payload(answer_contract_payload)
                if not payload_override and isinstance(derivation_spec, dict):
                    payload_override = _legacy_derivation_spec_to_answer_contract_payload(derivation_spec)
                payload = _merge_answer_contract_payload(payload_default, payload_override)
                answer_style = payload.get("answer_style")
                if isinstance(answer_style, dict) and answer_style:
                    c["answer_style"] = answer_style
                semantics = payload.get("answer_semantics")
                if isinstance(semantics, dict) and semantics:
                    semantics2 = dict(semantics)
                    semantics2["acceptance_mode"] = mode
                    c["answer_semantics"] = semantics2
                    c["answer_shape"] = _answer_shape_for_object(
                        _as_str(semantics2.get("answer_object")).strip() or "symbolic_expr"
                    )
                    branch_policy = semantics2.get("branch_policy")
                    if isinstance(branch_policy, dict) and branch_policy:
                        required_qualifiers = _as_str_list(semantics2.get("required_qualifiers"))
                        c["branch"] = {
                            "allow_branches": branch_policy.get("allow_branches"),
                            "branch_description_required": "branch_description" in required_qualifiers,
                        }
                support_witness = payload.get("support_witness")
                if isinstance(support_witness, list) and support_witness:
                    c["support_witness"] = support_witness

        if mode == "approx":
            # Intentionally leave regime/order unset unless upstream provides them.
            c["approx"] = {}

        contracts.append(c)

    return ids, contracts


def validate_answer_contracts(
    contracts: Sequence[Dict[str, Any]],
    *,
    world_contract: Any | None = None,
) -> Tuple[List[str], List[str]]:
    """Validate contract structure and key judging requirements.

    Returns (issue_types_error, issue_types_warn).
    """
    errors_set: set[str] = set()
    warns_set: set[str] = set()

    def _add(issue_type: str, default_sev: str) -> None:
        issue = str(issue_type or "").strip()
        if not issue:
            return
        sev = str(default_sev or "").strip().lower() or "error"
        override = None
        try:
            if isinstance(issue_severity, dict) and issue in issue_severity:
                override = issue_severity.get(issue)
        except Exception:
            override = None
        if override is not None:
            sev = str(override or "").strip().lower() or sev
        if sev == "ignore":
            return
        if sev == "warn" or sev == "warning":
            warns_set.add(issue)
            return
        errors_set.add(issue)

    seen: set[str] = set()
    # Default requirements (override by world_contract(L4) policy points when present).
    approx_requires_default = ["regime", "order"]
    numeric_requires_any_of_default = ["abs_tol", "rel_tol", "sig_figs"]

    policy = _extract_type2_policy(world_contract)
    if "type2.approx_requires" in policy:
        # Allow explicit empty list to mean "no requirements".
        approx_requires = _as_str_list(policy.get("type2.approx_requires"))
    else:
        approx_requires = approx_requires_default
    if "type2.numeric_requires_any_of" in policy:
        numeric_requires_any_of = _as_str_list(policy.get("type2.numeric_requires_any_of"))
    else:
        numeric_requires_any_of = numeric_requires_any_of_default
    numeric_unit_policy = str(policy.get("type2.numeric_unit_policy") or "ignore").strip().lower() or "ignore"
    issue_severity = policy.get("type2.issue_severity") if isinstance(policy.get("type2.issue_severity"), dict) else {}

    for c in contracts or []:
        if not isinstance(c, dict):
            _add("contract_not_dict", "error")
            continue
        cid = c.get("id")
        if not isinstance(cid, str) or not cid.strip():
            _add("missing_contract_id", "error")
        elif not cid.startswith(ACB_ID_PREFIX):
            _add("invalid_contract_id_prefix", "error")

        # Key uniqueness guard: contract id must be unique within the list.
        if isinstance(cid, str) and cid.strip():
            if cid in seen:
                _add("duplicate_contract_id", "error")
            else:
                seen.add(cid)

        qt = _qtype_norm(c.get("question_type"))
        exactness = c.get("exactness") if isinstance(c.get("exactness"), dict) else {}
        mode = str(exactness.get("mode") or "").strip().lower()

        if qt == "Numeric":
            numeric = c.get("numeric") if isinstance(c.get("numeric"), dict) else {}
            if numeric_requires_any_of:
                has_any = any(numeric.get(k) is not None for k in numeric_requires_any_of)
                if not has_any:
                    _add("numeric_missing_tolerance", "error")
            if numeric_unit_policy == "required":
                u = numeric.get("unit")
                if not (isinstance(u, str) and u.strip()):
                    _add("numeric_missing_unit", "warn")

        if mode == "approx":
            approx = c.get("approx") if isinstance(c.get("approx"), dict) else {}
            if approx_requires:
                missing = [k for k in approx_requires if approx.get(k) is None]
                if missing:
                    _add("approx_missing_regime_order", "error")

        branch = c.get("branch") if isinstance(c.get("branch"), dict) else {}
        if branch.get("allow_branches") is True and branch.get("branch_description_required") is True:
            # This is a policy flag; require an explicit branch description prompt on the question side.
            # Validator cannot inspect the question here; keep as warn to avoid over-blocking v0.
            _add("branch_requires_description", "warn")

    errors = sorted(errors_set)
    warns = sorted(warns_set)
    return errors, warns


def persist_answer_contracts(
    memory: Dict[str, Any],
    *,
    step: int,
    where: str,
    answer_contract_ids: Sequence[str],
    answer_contracts: Sequence[Dict[str, Any]],
    issue_types_error: Sequence[str] | None = None,
    issue_types_warn: Sequence[str] | None = None,
    raw_ref: str | None = None,
) -> Dict[str, Any]:
    """Persist contracts + ids + validation debug artifacts into KnownTree memory.

    NOTE: This function is internal governance only; it must not impact solver-visible views.
    """
    mem = dict(memory or {})
    step_i = int(step)

    # 1) Update bank: replace all contracts for this step deterministically.
    bank = mem.get("answer_contract_bank")
    if not isinstance(bank, list):
        bank = []
    bank2: List[Dict[str, Any]] = []
    for item in bank:
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("source_step")) == step_i:
                continue
        except Exception:
            pass
        bank2.append(item)
    for c in answer_contracts or []:
        if isinstance(c, dict):
            bank2.append(dict(c))
    mem["answer_contract_bank"] = bank2

    # 2) Write step link via step_certs (independent entry).
    step_certs = mem.get("step_certs")
    if not isinstance(step_certs, list):
        step_certs = []
    step_certs.append(
        {
            "kind": "answer_contract_cert",
            "step": step_i,
            "answer_contract_ids": [str(x) for x in (answer_contract_ids or []) if str(x).strip()],
            "provenance": {"role": "answer_contract_bank", "raw_ref": raw_ref, "where": str(where or "").strip()},
        }
    )
    mem["step_certs"] = step_certs

    # 3) Persist validation debug artifacts (bounded).
    # Decision: candidates[step] is the single truth source for "current" validation state,
    # and MUST be upserted every time (even when issue lists are empty) to avoid stale errors.
    issue_types_error = [str(x) for x in (issue_types_error or []) if str(x).strip()]
    issue_types_warn = [str(x) for x in (issue_types_warn or []) if str(x).strip()]
    error_count = len(issue_types_error)
    warn_count = len(issue_types_warn)

    cand = mem.get("answer_contract_validation_candidates")
    if not isinstance(cand, dict):
        cand = {}
    cand_payload = {
        "answer_contract_ids": [str(x) for x in (answer_contract_ids or []) if str(x).strip()],
        "answer_contracts": [_summarize_contract(dict(c)) for c in (answer_contracts or []) if isinstance(c, dict)],
        "issue_types_error": issue_types_error,
        "issue_types_warn": issue_types_warn,
        "error_count": error_count,
        "warn_count": warn_count,
    }
    mem["answer_contract_validation_candidates"] = _bounded_step_dict_insert(
        cand, step_i, cand_payload, max_steps=MAX_VALIDATION_CANDIDATE_STEPS
    )

    if error_count or warn_count:
        errs_list = mem.get("answer_contract_validation_errors")
        if not isinstance(errs_list, list):
            errs_list = []
        msg = f"answer contract issues(error) step={step_i}: {','.join(issue_types_error)}"
        if warn_count:
            msg = msg + f" | warn: {','.join(issue_types_warn)}"
        errs_list = _bounded_append(
            errs_list,
            {
                "step": step_i,
                "error": msg,
                "where": str(where or "").strip(),
                "issue_types_error": issue_types_error,
                "issue_types_warn": issue_types_warn,
                "error_count": error_count,
                "warn_count": warn_count,
            },
            max_items=MAX_VALIDATION_ERRORS,
        )
        mem["answer_contract_validation_errors"] = errs_list

    return mem


def build_answer_contract_validation_background(
    memory: Dict[str, Any],
    *,
    step: int,
    lang: str | None = None,
) -> str:
    """Build a concise prompt background for answer_contract revise_diagnose."""
    mem = dict(memory or {})
    step_i = int(step)

    bank = mem.get("answer_contract_bank") if isinstance(mem.get("answer_contract_bank"), list) else []
    # 1) Find ids from step_certs (latest wins).
    ids: List[str] = []
    step_certs = mem.get("step_certs") if isinstance(mem.get("step_certs"), list) else []
    for item in reversed(step_certs):
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "") != "answer_contract_cert":
            continue
        try:
            if int(item.get("step")) != step_i:
                continue
        except Exception:
            continue
        raw_ids = item.get("answer_contract_ids")
        if isinstance(raw_ids, list):
            ids = [str(x) for x in raw_ids if str(x).strip()]
        break

    # 2) Load contracts by ids.
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in bank:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            by_id[c["id"]] = c
    contracts = [by_id[i] for i in ids if i in by_id] if ids else []

    # 3) Fallback to candidates snapshot when bank missing.
    cand = mem.get("answer_contract_validation_candidates")
    cand_step: Dict[str, Any] | None = None
    if isinstance(cand, dict):
        raw = cand.get(str(step_i))
        cand_step = raw if isinstance(raw, dict) else None

    errs = mem.get("answer_contract_validation_errors")
    errs_for_step: List[Dict[str, Any]] = []
    if isinstance(errs, list):
        for e in errs:
            if not isinstance(e, dict):
                continue
            try:
                if int(e.get("step")) != step_i:
                    continue
            except Exception:
                continue
            errs_for_step.append(e)

    use_en = _is_en(lang)
    lines: List[str] = []
    title = "Answer Contract Validation (Type2)" if use_en else "Answer Contract 校验信息（Type2）"
    lines.append(title)

    # Single truth source: use candidates[step] for the "current" issue types/counts.
    # Only show historical errors when the current snapshot still has issues.
    curr_err = 0
    curr_warn = 0
    curr_issue_err: List[str] = []
    curr_issue_warn: List[str] = []
    if cand_step is not None:
        try:
            curr_err = int(cand_step.get("error_count") or 0)
        except Exception:
            curr_err = 0
        try:
            curr_warn = int(cand_step.get("warn_count") or 0)
        except Exception:
            curr_warn = 0
        raw_ie = cand_step.get("issue_types_error")
        if isinstance(raw_ie, list):
            curr_issue_err = [str(x).strip() for x in raw_ie if str(x).strip()]
        raw_iw = cand_step.get("issue_types_warn")
        if isinstance(raw_iw, list):
            curr_issue_warn = [str(x).strip() for x in raw_iw if str(x).strip()]

        lines.append(
            f"- current: errors={curr_err}, warns={curr_warn}" if use_en else f"- 当前状态：errors={curr_err}, warns={curr_warn}"
        )
        if curr_issue_err:
            lines.append(f"  - error_types: {curr_issue_err}" if use_en else f"  - error_types：{curr_issue_err}")
        if curr_issue_warn:
            lines.append(f"  - warn_types: {curr_issue_warn}" if use_en else f"  - warn_types：{curr_issue_warn}")

    if errs_for_step and (curr_err > 0 or curr_warn > 0):
        lines.append("- errors (history):" if use_en else "- 错误摘要（历史）：")
        for e in errs_for_step[-3:]:
            s = _as_str(e.get("error") or "").strip()
            if s:
                lines.append(f"  - {s}")

    if contracts:
        lines.append("- contracts (from bank):" if use_en else "- contracts（来自 bank）：")
        for c in contracts[:3]:
            lines.append(f"  - {_as_str(_summarize_contract(c))}")
    elif cand_step is not None:
        lines.append("- contracts (candidate snapshot):" if use_en else "- contracts（候选快照）：")
        try:
            csum = cand_step.get("answer_contracts")
            if isinstance(csum, list):
                for c in csum[:3]:
                    lines.append(f"  - {_as_str(c)}")
        except Exception:
            pass

    if ids:
        lines.append(f"- answer_contract_ids: {ids}" if use_en else f"- answer_contract_ids：{ids}")
    elif cand_step is not None:
        cid2 = cand_step.get("answer_contract_ids")
        if isinstance(cid2, list) and cid2:
            lines.append(f"- answer_contract_ids: {cid2}" if use_en else f"- answer_contract_ids：{cid2}")

    return "\n".join(lines).strip()


__all__ = [
    "ACB_ID_PREFIX",
    "ANSWER_CONTRACT_MODEL_TEXT_ZH",
    "ANSWER_CONTRACT_MODEL_TEXT_EN",
    "make_default_answer_contracts",
    "validate_answer_contracts",
    "persist_answer_contracts",
    "build_answer_contract_validation_background",
    "build_answer_output_spec_prompt_section",
    "extract_answer_contract_context",
    "extract_answer_output_spec_context",
    "answer_contract_model_text",
]
