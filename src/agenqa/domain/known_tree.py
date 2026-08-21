"""KnownTree v2 helpers: schema definition, normalization, and view building."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List
import json
import logging
from textwrap import dedent

from agenqa.domain.known_utils import normalize_known, parse_known_to_dict
from agenqa.domain.contracts.world_contract import normalize_world_contract as normalize_world_contract_v2

logger = logging.getLogger(__name__)

KNOWN_TREE_DESCRIPTION = dedent(
    """\
    # KnownTree v2 结构与字段含义（共享约定）

    - KnownTree v2 是系统维护的长期记忆（Memory）结构，核心目标是：
      - typed bullets（类型化原子条目）
      - provenance（可溯源）
      - views（按角色/可见性过滤）

    - 顶层结构（v2）：
      - episode_seed：链路的“话题锚定/出题侧元信息”（字段由 contract 定义；`anchor` 可由 `anchor_fields/subject/keywords` 规范化派生）。
        - 重要：`episode_seed` **不作为 solver 输入**（不进入 edge/path 的 solver 可见 view）；它用于 Director/出题者侧的“主题锚定/一致性检查/出题引导”。
      - world_contract：Type1（语义世界观不唯一）治理用的“语义契约（分层 points）”：
        - 以 L1-L4 的 sections + points 表示；points 采用 `{axis, choice, note?}`。
        - 其中 `L1.paradigm_id` 表示范式选择；`L3` 表示题目特定规则（避免语义分叉）。
        - 重要：`world_contract` **不进入 solver 的 KnownTree view**；它用于 Director/ReviseWorldContract 的治理，并会在下游被渲染为独立 `world_contract_text`，再与 `question` 拼接成最终 solver 输入。
      - premise_bank：前提/定义/条件（单调可见；属于 solver 可见输入的一部分）；
      - fact_bank：可复用结论（默认不进入 path 视角）；
      - step_certs：每步推理证书（默认不进入 path 视角）。

    - solver 可见性约定（edge/path 用于求解评测）：
      - edge solver view：`premise_bank + fact_bank/step_certs(<t)`（不含 episode_seed）。
      - path solver view（Path-Fold）：仅 `premise_bank`（不含 episode_seed；题面为 `Q_fold`）。

    - 链式递进强约束：
      - draft_chain 必须显式声明 required_fact_ids / primary_required_fact_id / reuse_plan。
      - step_cert_builder 必须输出 key_fact（答案等价锚点）。
      - 对 t>=2：draft_chain.primary_required_fact_id == key_fact_id(step=t-1)。

    - MCQ 特殊规则：
      - key_fact 不能只给选项字母，必须包含 mcq_choice + mcq_choice_text + statement。
    """
)

KNOWN_TREE_DESCRIPTION_EN = dedent(
    """\
    # KnownTree v2: Structure and Field Semantics (Shared Contract)

    - KnownTree v2 is the long-term memory structure with:
      - typed bullets (atomic typed entries)
      - provenance (traceability)
      - views (role-specific visibility filters)

    - Top-level fields (v2):
      - episode_seed: topic anchor / generator-side metadata (fields are contract-defined; `anchor` can be derived from `anchor_fields/subject/keywords`).
        - Important: `episode_seed` is **NOT part of solver input** (excluded from edge/path solver-visible views); it is used for Director/generator-side anchoring and consistency checks.
      - world_contract: a Type1 semantic world-view contract with layered sections + points (L1-L4).
        - points are `{axis, choice, note?}`; `L1.paradigm_id` is the paradigm choice; `L3` stores question-specific semantic rules.
        - Important: `world_contract` is **NOT injected into the solver's KnownTree view**; downstream renders it as a separate `world_contract_text` block and concatenates it with the question for solver input.
      - premise_bank: premises/definitions/conditions (monotonic visibility; part of solver-visible input).
      - fact_bank: reusable conclusions (hidden from path view).
      - step_certs: per-step reasoning certificates (hidden from path view).

    - Solver-visible views (for evaluation):
      - edge solver view: `premise_bank + fact_bank/step_certs(<t)` (no episode_seed).
      - path solver view (Path-Fold): `premise_bank` only (no episode_seed; `Question` is `Q_fold`).

    - Chain constraints:
      - draft_chain must declare required_fact_ids / primary_required_fact_id / reuse_plan explicitly.
      - step_cert_builder must output a key_fact (answer-equivalent anchor).
      - For t>=2: draft_chain.primary_required_fact_id == key_fact_id(step=t-1).

    - MCQ rule:
      - key_fact must include mcq_choice + mcq_choice_text + statement; do not store just the letter.
    """
)

KNOWN_TREE_FIELD_VISIBILITY: Dict[str, Dict[str, bool]] = {
    # Note: path "solver view" excludes episode_seed by design (generator-only metadata).
    "episode_seed": {"visible_to_path": False},
    # Generator-side semantics governance. Not part of solver-visible KnownTree views; a
    # separate rendered world_contract_text may still be concatenated into solver input.
    "world_contract": {"visible_to_path": False},
    "premise_bank": {"visible_to_path": True},
    "fact_bank": {"visible_to_path": False},
    "step_certs": {"visible_to_path": False},
}


class KnownTree:
    """Helpers to build and query KnownTree v2 structures."""

    DESCRIPTION: str = KNOWN_TREE_DESCRIPTION
    DESCRIPTION_EN: str = KNOWN_TREE_DESCRIPTION_EN
    SCHEMA_VERSION: int = 2
    MAX_EPISODE_KEYWORDS: int = 10

    @staticmethod
    def empty_memory(episode_seed: Dict[str, Any] | None = None) -> Dict[str, Any]:
        seed = KnownTree._normalize_episode_seed(episode_seed)
        return {
            "schema_version": KnownTree.SCHEMA_VERSION,
            "episode_seed": seed,
            "world_contract": KnownTree._normalize_world_contract(None),
            "premise_bank": [],
            "fact_bank": [],
            "step_certs": [],
            # Type2 governance (internal-only; do NOT expose to solvers).
            "answer_contract_bank": [],
            "answer_contract_validation_errors": [],
            "answer_contract_validation_candidates": {},
        }

    @staticmethod
    def _normalize_world_contract(val: Any) -> Dict[str, Any]:
        """Normalize world_contract into canonical v2 object (layered sections + points)."""
        return normalize_world_contract_v2(val)

    @staticmethod
    def normalize_world_contract(val: Any) -> Dict[str, Any]:
        """Public wrapper for world_contract normalization (v2)."""
        return KnownTree._normalize_world_contract(val)

    @staticmethod
    def _normalize_episode_seed(seed: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(seed, dict):
            seed = {}
        anchor = seed.get("anchor")
        anchor_fields = seed.get("anchor_fields")
        subject = seed.get("subject")
        keywords = seed.get("keywords")
        if not isinstance(anchor, str):
            anchor = str(anchor) if anchor is not None else ""
        if not isinstance(anchor_fields, dict):
            anchor_fields = {}
        seed_contract = seed.get("seed_contract")
        if not isinstance(seed_contract, dict):
            seed_contract = {}

        # Normalize anchor_fields as a structured anchor dict (keys are contract-defined).
        af_out: Dict[str, str] = {}
        for key in ("topic_task", "core_objects_symbols", "regime_assumptions"):
            val = anchor_fields.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                s = val.strip()
            else:
                s = str(val).strip()
            if s:
                af_out[key] = s
        # Also keep any other anchor_fields keys (contract-specific), preserving them as strings.
        if isinstance(anchor_fields, dict):
            for k, v in anchor_fields.items():
                ks = str(k).strip()
                if not ks or ks in af_out:
                    continue
                vs = v.strip() if isinstance(v, str) else str(v).strip()
                if vs:
                    af_out[ks] = vs
        if not isinstance(subject, str):
            subject = str(subject) if subject is not None else ""
        if not isinstance(keywords, list):
            keywords = []
        keywords_out: List[str] = []
        for item in keywords:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
            else:
                s = str(item).strip()
            if s:
                keywords_out.append(s)
        if len(keywords_out) > KnownTree.MAX_EPISODE_KEYWORDS:
            keywords_out = keywords_out[: KnownTree.MAX_EPISODE_KEYWORDS]

        if not anchor.strip():
            # If we have structured anchor fields, derive a stable anchor first.
            if af_out:
                template = seed_contract.get("anchor_template")
                if isinstance(template, str) and template.strip():
                    try:
                        anchor = template.format(**af_out).strip()
                    except Exception:
                        anchor = ""
                if not anchor.strip():
                    lines: list[str] = []
                    # Prefer the canonical 3 fields when present.
                    if af_out.get("topic_task"):
                        lines.append(f"Topic/Task: {af_out['topic_task']}")
                    if af_out.get("core_objects_symbols"):
                        lines.append(f"Core objects/symbols: {af_out['core_objects_symbols']}")
                    if af_out.get("regime_assumptions"):
                        lines.append(f"Regime/assumptions: {af_out['regime_assumptions']}")
                    if not lines:
                        # Fallback: deterministic key ordering for unknown schemas.
                        for k in sorted(af_out.keys()):
                            lines.append(f"{k}: {af_out[k]}")
                    anchor = "\n".join(lines).strip()

        if not anchor.strip():
            # Keep episode_seed usable even when only legacy fields exist.
            # Prefer a compact, single-line anchor.
            parts: list[str] = []
            if subject.strip():
                parts.append(subject.strip())
            if keywords_out:
                parts.append(", ".join(keywords_out[: KnownTree.MAX_EPISODE_KEYWORDS]))
            anchor = " | ".join(parts).strip()

        out = dict(seed)
        out["anchor"] = anchor
        out["anchor_fields"] = af_out
        out["subject"] = subject
        out["keywords"] = keywords_out
        return out

    @staticmethod
    def _compact_episode_seed_for_draft_chain(seed: Any) -> Dict[str, Any]:
        """Drop summary-style duplicates from episode_seed for draft generation."""
        seed_out = deepcopy(seed) if isinstance(seed, dict) else {}
        # Keep structured anchor_fields as source of truth; drop duplicated prose anchor.
        seed_out.pop("anchor", None)
        sk = seed_out.get("paper_reasoning_skeleton")
        if isinstance(sk, dict):
            sk_out = deepcopy(sk)
            # Keep the structured skeleton, but remove its summary-style prose duplication.
            sk_out.pop("reasoning_summary", None)
            seed_out["paper_reasoning_skeleton"] = sk_out
        return seed_out

    @staticmethod
    def _compact_world_contract_for_draft_chain(world_contract: Any) -> Dict[str, Any]:
        """Keep semantic contract content, drop internal evolution metadata."""
        wc_out = deepcopy(world_contract) if isinstance(world_contract, dict) else {}
        wc_out.pop("extra_internal", None)
        return wc_out

    @staticmethod
    def normalize_memory(val: Any) -> Dict[str, Any]:
        """Normalize a memory payload into a v2 KnownTree dict."""
        parsed = val
        if isinstance(parsed, str):
            parsed = parse_known_to_dict(parsed) or parsed
        if not isinstance(parsed, dict):
            return KnownTree.empty_memory()
        memory = deepcopy(parsed)
        if not isinstance(memory.get("schema_version"), int):
            memory["schema_version"] = KnownTree.SCHEMA_VERSION
        if not isinstance(memory.get("episode_seed"), dict):
            memory["episode_seed"] = {}
        memory["episode_seed"] = KnownTree._normalize_episode_seed(memory.get("episode_seed"))
        memory["world_contract"] = KnownTree._normalize_world_contract(memory.get("world_contract"))
        for key in ("premise_bank", "fact_bank", "step_certs"):
            if not isinstance(memory.get(key), list):
                memory[key] = []
        if not isinstance(memory.get("answer_contract_bank"), list):
            memory["answer_contract_bank"] = []
        if not isinstance(memory.get("answer_contract_validation_errors"), list):
            memory["answer_contract_validation_errors"] = []
        if not isinstance(memory.get("answer_contract_validation_candidates"), dict):
            memory["answer_contract_validation_candidates"] = {}
        return memory

    @staticmethod
    def update_episode_seed(
        memory: Any,
        subject: str | None = None,
        keywords: List[str] | None = None,
        *,
        anchor: str | None = None,
        seed: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        mem = KnownTree.normalize_memory(memory)
        seed_payload: Dict[str, Any] = dict(mem.get("episode_seed", {}))
        if isinstance(seed, dict) and seed:
            seed_payload.update(dict(seed))
        if anchor is not None:
            seed_payload["anchor"] = anchor
        if subject is not None:
            seed_payload["subject"] = subject
        if keywords is not None:
            seed_payload["keywords"] = keywords
        mem["episode_seed"] = KnownTree._normalize_episode_seed(seed_payload)
        return mem

    @staticmethod
    def key_fact_id_for_step(memory: Any, step: int) -> str | None:
        mem = KnownTree.normalize_memory(memory)
        try:
            step_i = int(step)
        except Exception:
            return None
        # A single step may have multiple cert entries (e.g. chain cert + eval cert).
        # Return the first non-empty key_fact_id among step-matching entries.
        for cert in KnownTree._filter_entries(mem.get("step_certs", [])):
            try:
                if int(cert.get("step")) != step_i:
                    continue
            except Exception:
                continue
            val = cert.get("key_fact_id")
            if isinstance(val, str) and val.strip():
                return val
        return None

    @staticmethod
    def to_json(tree: Any) -> str:
        try:
            return json.dumps(tree, ensure_ascii=False)
        except Exception:
            return "" if tree is None else str(tree)

    @staticmethod
    def _filter_entries(entries: Iterable[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in entries:
            if isinstance(item, dict):
                out.append(item)
        return out

    @staticmethod
    def _step_of_entry(entry: Dict[str, Any], default: int | None = None) -> int | None:
        for key in ("source_step", "step"):
            val = entry.get(key)
            if val is None:
                continue
            try:
                return int(val)
            except Exception:
                continue
        return default

    @staticmethod
    def _strip_provenance(entry: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(entry)
        item.pop("provenance", None)
        return item

    @staticmethod
    def _redact_step_cert_for_llm(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Redact internal/sensitive fields in step_certs for LLM-facing views.

        step_certs live in memory and may contain tool artifacts (e.g. numeric oracle code,
        execution payload). Such fields must never appear in prompts, so we use a strict
        allow-list to avoid accidental leakage.
        """
        if not isinstance(entry, dict):
            return {}

        kind = entry.get("kind") if isinstance(entry.get("kind"), str) else None
        if kind == "numeric_oracle_cert":
            allowed = ("kind", "step", "abs_tol", "rel_tol", "sig_figs", "unit")
        elif kind == "answer_contract_cert":
            # Type2 governance is internal-only; must never appear in solver-facing views.
            return {}
        else:
            allowed = (
                "kind",
                "step",
                "uses_premise_ids",
                "uses_fact_ids",
                "produces_fact_ids",
                "key_fact_id",
                "cert_text",
            )

        out: Dict[str, Any] = {}
        for k in allowed:
            if k in entry:
                out[k] = entry.get(k)
        return out

    @staticmethod
    def compact_kqa_known_view(view: Any) -> Dict[str, Any]:
        """Compact a KnownTree view for KQA exports (edge_kqa / path_kqa).

        Motivation:
        - Reduce artifact size and prompt payload (e.g. final_commenter inputs).
        - Remove trace/debug-only keys that are not needed by solvers (kind/source_step/provenance).

        Notes:
        - This does NOT mutate the underlying state memory.
        - Keeps only solver-relevant text fields (text/statement/cert_text) + minimal ids.
        """
        if not isinstance(view, dict):
            return {"schema_version": KnownTree.SCHEMA_VERSION}

        # KQA exports are solver-facing artifacts; keep only solver-relevant fields.
        # `episode_seed` is generator-only metadata and must not be included here.
        out: Dict[str, Any] = {"schema_version": view.get("schema_version", KnownTree.SCHEMA_VERSION)}

        premise_out: List[Dict[str, Any]] = []
        for item in KnownTree._filter_entries(view.get("premise_bank", [])):
            pid = item.get("id")
            text = item.get("text")
            if text is None or (isinstance(text, str) and not text.strip()):
                text = item.get("statement")
            compact: Dict[str, Any] = {}
            if isinstance(pid, str) and pid.strip():
                compact["id"] = pid
            if text is not None:
                compact["text"] = str(text)
            if compact:
                premise_out.append(compact)
        out["premise_bank"] = premise_out

        fact_out: List[Dict[str, Any]] = []
        for item in KnownTree._filter_entries(view.get("fact_bank", [])):
            fid = item.get("id")
            statement = item.get("statement")
            text = item.get("text")
            compact = {}
            if isinstance(fid, str) and fid.strip():
                compact["id"] = fid
            if statement is not None and not (isinstance(statement, str) and not statement.strip()):
                compact["statement"] = str(statement)
            elif text is not None:
                compact["text"] = str(text)
            mcq_choice = item.get("mcq_choice")
            if isinstance(mcq_choice, str) and mcq_choice.strip():
                compact["mcq_choice"] = mcq_choice
                mcq_choice_text = item.get("mcq_choice_text")
                if mcq_choice_text is not None:
                    compact["mcq_choice_text"] = str(mcq_choice_text)
            tags = item.get("tags")
            if isinstance(tags, list):
                cleaned = [str(t).strip() for t in tags if str(t).strip()]
                if cleaned:
                    compact["tags"] = cleaned
            if compact:
                fact_out.append(compact)
        out["fact_bank"] = fact_out

        cert_out: List[Dict[str, Any]] = []
        for item in KnownTree._filter_entries(view.get("step_certs", [])):
            compact: Dict[str, Any] = {}
            for key in (
                "step",
                "uses_premise_ids",
                "uses_fact_ids",
                "produces_fact_ids",
                "key_fact_id",
                "cert_text",
            ):
                if key in item:
                    compact[key] = item.get(key)
            if compact:
                cert_out.append(compact)
        out["step_certs"] = cert_out

        # Some views use windowed fields; preserve them when present.
        if "fact_window" in view and isinstance(view.get("fact_window"), list):
            out["fact_window"] = [
                {"id": f.get("id"), "text": str((f.get("statement") or f.get("text") or "")).strip()}
                for f in KnownTree._filter_entries(view.get("fact_window", []))
                if isinstance(f.get("id"), str) and str((f.get("statement") or f.get("text") or "")).strip()
            ]
        if "step_certs_window" in view and isinstance(view.get("step_certs_window"), list):
            out["step_certs_window"] = [
                {k: c.get(k) for k in ("step", "key_fact_id", "cert_text") if k in c}
                for c in KnownTree._filter_entries(view.get("step_certs_window", []))
                if any(k in c for k in ("step", "key_fact_id", "cert_text"))
            ]
        if "current_step" in view and isinstance(view.get("current_step"), dict):
            out["current_step"] = dict(view.get("current_step") or {})

        return out

    @staticmethod
    def build_draft_chain_view(memory: Any, step: int, window: int = 2) -> Dict[str, Any]:
        mem = KnownTree.normalize_memory(memory)
        try:
            step_i = int(step)
        except Exception:
            step_i = 0
        premise_bank = KnownTree._filter_entries(mem.get("premise_bank", []))
        fact_bank = KnownTree._filter_entries(mem.get("fact_bank", []))
        step_certs = KnownTree._filter_entries(mem.get("step_certs", []))
        allowed_steps = {step_i - i for i in range(1, max(1, window) + 1) if step_i - i >= 1}
        fact_window = [f for f in fact_bank if KnownTree._step_of_entry(f) in allowed_steps]
        cert_window = [c for c in step_certs if KnownTree._step_of_entry(c) in allowed_steps]
        return {
            "schema_version": mem.get("schema_version", KnownTree.SCHEMA_VERSION),
            "episode_seed": KnownTree._compact_episode_seed_for_draft_chain(mem.get("episode_seed", {})),
            "world_contract": KnownTree._compact_world_contract_for_draft_chain(mem.get("world_contract", {})),
            "premise_bank": premise_bank,
            "fact_window": fact_window,
            "step_certs_window": [r for r in (KnownTree._redact_step_cert_for_llm(c) for c in cert_window) if r],
        }

    @staticmethod
    def build_diagnose_view(
        memory: Any,
        step: int,
        *,
        window: int = 2,
        include_current_step: bool = True,
    ) -> Dict[str, Any]:
        """Build a revise-diagnose view with draft_chain-style windows + current-step evidence.

        Output fields:
        - episode_seed / premise_bank: for grounding (premise_bank is filtered to < step)
        - fact_window / step_certs_window: only recent window steps (< step)
        - current_step: (optional) the existing entries for step==current step, with provenance stripped

        All entries strip the top-level 'provenance' to keep the view semantic-focused.
        """
        mem = KnownTree.normalize_memory(memory)
        try:
            step_i = int(step)
        except Exception:
            step_i = 0
        try:
            window_i = int(window)
        except Exception:
            window_i = 2
        if window_i < 0:
            window_i = 0

        premise_bank_all = KnownTree._filter_entries(mem.get("premise_bank", []))
        fact_bank_all = KnownTree._filter_entries(mem.get("fact_bank", []))
        step_certs_all = KnownTree._filter_entries(mem.get("step_certs", []))

        premise_bank = [
            KnownTree._strip_provenance(p)
            for p in premise_bank_all
            if (KnownTree._step_of_entry(p, default=-1) or -1) < step_i
        ]

        allowed_steps = {step_i - i for i in range(1, window_i + 1) if step_i - i >= 1}
        fact_window = [KnownTree._strip_provenance(f) for f in fact_bank_all if KnownTree._step_of_entry(f) in allowed_steps]
        cert_window = [
            r
            for r in (
                KnownTree._redact_step_cert_for_llm(KnownTree._strip_provenance(c))
                for c in step_certs_all
                if KnownTree._step_of_entry(c) in allowed_steps
            )
            if r
        ]

        out: Dict[str, Any] = {
            "schema_version": mem.get("schema_version", KnownTree.SCHEMA_VERSION),
            "episode_seed": mem.get("episode_seed", {}),
            "world_contract": mem.get("world_contract", {}),
            "premise_bank": premise_bank,
            "fact_window": fact_window,
            "step_certs_window": cert_window,
        }

        if include_current_step:
            current_premise = [
                KnownTree._strip_provenance(p)
                for p in premise_bank_all
                if KnownTree._step_of_entry(p, default=-1) == step_i
            ]
            current_facts = [
                KnownTree._strip_provenance(f) for f in fact_bank_all if KnownTree._step_of_entry(f, default=-1) == step_i
            ]
            current_cert: Dict[str, Any] | None = None
            for cert in step_certs_all:
                try:
                    if int(cert.get("step")) == step_i:
                        current_cert = KnownTree._redact_step_cert_for_llm(KnownTree._strip_provenance(cert))
                except Exception:
                    continue
            out["current_step"] = {
                "step": step_i,
                "premise_delta": current_premise,
                "fact_delta": current_facts,
                "step_cert": current_cert or {},
            }

        return out

    @staticmethod
    def build_edge_solver_view(memory: Any, step: int) -> Dict[str, Any]:
        mem = KnownTree.normalize_memory(memory)
        try:
            step_i = int(step)
        except Exception:
            step_i = 0
        premise_bank = KnownTree._filter_entries(mem.get("premise_bank", []))
        fact_bank = [
            f
            for f in KnownTree._filter_entries(mem.get("fact_bank", []))
            if (KnownTree._step_of_entry(f, default=-1) or -1) < step_i
        ]
        step_certs = [
            c
            for c in KnownTree._filter_entries(mem.get("step_certs", []))
            if (KnownTree._step_of_entry(c, default=-1) or -1) < step_i
        ]
        return {
            "schema_version": mem.get("schema_version", KnownTree.SCHEMA_VERSION),
            "premise_bank": premise_bank,
            "fact_bank": fact_bank,
            "step_certs": [r for r in (KnownTree._redact_step_cert_for_llm(c) for c in step_certs) if r],
        }

    @staticmethod
    def build_path_solver_view(memory: Any, step: int) -> Dict[str, Any]:
        mem = KnownTree.normalize_memory(memory)
        premise_bank = KnownTree._filter_entries(mem.get("premise_bank", []))
        return {
            "schema_version": mem.get("schema_version", KnownTree.SCHEMA_VERSION),
            "premise_bank": premise_bank,
            "fact_bank": [],
            "step_certs": [],
        }

    @staticmethod
    def build_step_cert_view(memory: Any, step: int) -> Dict[str, Any]:
        """Build a minimal memory view for step_cert_builder.

        Only includes < step entries in premise_bank / fact_bank, and each entry keeps only:
        - id
        - text (premise) / text (fact, derived from statement or text)

        Notes:
        - Excludes episode_seed / step_certs and other metadata to reduce context size.
        - Filtering by < step also naturally avoids referencing same-step old entries in revise(overwrite_step) flows.
        """
        mem = KnownTree.normalize_memory(memory)
        try:
            step_i = int(step)
        except Exception:
            step_i = 0

        premise_bank_minimal: List[Dict[str, Any]] = []
        for premise in KnownTree._filter_entries(mem.get("premise_bank", [])):
            pid = premise.get("id")
            if not isinstance(pid, str) or not pid.strip():
                continue
            if (KnownTree._step_of_entry(premise, default=-1) or -1) >= step_i:
                continue
            text = premise.get("text")
            premise_bank_minimal.append({"id": pid, "text": "" if text is None else str(text)})

        fact_bank_minimal: List[Dict[str, Any]] = []
        for fact in KnownTree._filter_entries(mem.get("fact_bank", [])):
            fid = fact.get("id")
            if not isinstance(fid, str) or not fid.strip():
                continue
            if (KnownTree._step_of_entry(fact, default=-1) or -1) >= step_i:
                continue
            text = fact.get("statement")
            if text is None or (isinstance(text, str) and not text.strip()):
                text = fact.get("text")
            fact_bank_minimal.append({"id": fid, "text": "" if text is None else str(text)})

        return {
            "schema_version": mem.get("schema_version", KnownTree.SCHEMA_VERSION),
            "premise_bank": premise_bank_minimal,
            "fact_bank": fact_bank_minimal,
        }

    @staticmethod
    def apply_step_update(
        memory: Any,
        *,
        step: int,
        premise_delta: List[Dict[str, Any]] | None,
        fact_delta: List[Dict[str, Any]] | None,
        step_cert: Dict[str, Any] | None,
        key_fact_id: str | None,
        overwrite_step: bool = False,
        provenance: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        mem = KnownTree.normalize_memory(memory)
        try:
            step_i = int(step)
        except Exception as exc:
            raise ValueError(f"invalid step={step!r}") from exc

        premise_delta = premise_delta or []
        fact_delta = fact_delta or []
        if step_cert is None or not isinstance(step_cert, dict):
            raise ValueError("step_cert_builder output missing step_cert")
        if not key_fact_id or not isinstance(key_fact_id, str):
            raise ValueError("step_cert_builder output missing key_fact_id")
        if not key_fact_id.strip() or key_fact_id.strip() != key_fact_id:
            raise ValueError("key_fact_id must be a non-empty string without leading/trailing whitespace")

        if overwrite_step:
            mem["premise_bank"] = [
                p for p in KnownTree._filter_entries(mem.get("premise_bank", [])) if KnownTree._step_of_entry(p) != step_i
            ]
            mem["fact_bank"] = [
                f for f in KnownTree._filter_entries(mem.get("fact_bank", [])) if KnownTree._step_of_entry(f) != step_i
            ]
            mem["step_certs"] = [
                c for c in KnownTree._filter_entries(mem.get("step_certs", [])) if KnownTree._step_of_entry(c) != step_i
            ]

        existing_premise_ids = {
            pid
            for pid in (p.get("id") for p in KnownTree._filter_entries(mem.get("premise_bank", [])))
            if isinstance(pid, str) and pid.strip()
        }
        existing_fact_ids = {
            fid
            for fid in (f.get("id") for f in KnownTree._filter_entries(mem.get("fact_bank", [])))
            if isinstance(fid, str) and fid.strip()
        }
        existing_all_ids = existing_premise_ids | existing_fact_ids

        def _delta_ids(delta: List[Dict[str, Any]], field: str) -> List[str]:
            ids: List[str] = []
            for idx, entry in enumerate(delta):
                if not isinstance(entry, dict):
                    raise ValueError(f"{field}[{idx}] must be an object (step={step_i})")
                raw = entry.get("id")
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError(f"{field}[{idx}].id must be a non-empty string (step={step_i})")
                if raw.strip() != raw:
                    raise ValueError(
                        f"{field}[{idx}].id must not contain leading/trailing whitespace: {raw!r} (step={step_i})"
                    )
                if "source_step" in entry:
                    try:
                        source_step = int(entry.get("source_step"))
                    except Exception as exc:
                        raise ValueError(f"{field}[{idx}].source_step must be an int (step={step_i})") from exc
                    if source_step != step_i:
                        raise ValueError(
                            f"{field}[{idx}].source_step must equal current step={step_i} (got {source_step})"
                        )
                ids.append(raw)
            return ids

        new_premise_ids = _delta_ids(premise_delta, "premise_delta")
        new_fact_ids = _delta_ids(fact_delta, "fact_delta")
        if key_fact_id not in set(new_fact_ids):
            raise ValueError("key_fact_id must point to a fact_delta entry")

        def _find_duplicates(ids: List[str]) -> List[str]:
            seen: set[str] = set()
            dups: set[str] = set()
            for item in ids:
                if item in seen:
                    dups.add(item)
                else:
                    seen.add(item)
            return sorted(dups)

        premise_dups = _find_duplicates(new_premise_ids)
        if premise_dups:
            raise ValueError(f"premise_delta contains duplicate IDs (step={step_i}): {premise_dups}")
        fact_dups = _find_duplicates(new_fact_ids)
        if fact_dups:
            raise ValueError(f"fact_delta contains duplicate IDs (step={step_i}): {fact_dups}")

        cross_dups = sorted(set(new_premise_ids) & set(new_fact_ids))
        if cross_dups:
            raise ValueError(f"premise_delta and fact_delta share duplicate IDs (step={step_i}): {cross_dups}")

        premise_conflicts = sorted(set(new_premise_ids) & existing_all_ids)
        if premise_conflicts:
            raise ValueError(f"premise_delta IDs conflict with existing memory (step={step_i}): {premise_conflicts}")
        fact_conflicts = sorted(set(new_fact_ids) & existing_all_ids)
        if fact_conflicts:
            raise ValueError(f"fact_delta IDs conflict with existing memory (step={step_i}): {fact_conflicts}")

        def _id_list(cert: Dict[str, Any], key: str) -> List[str]:
            if key not in cert:
                return []
            val = cert.get(key)
            if val is None:
                return []
            if not isinstance(val, list):
                raise ValueError(f"step_cert.{key} must be a list of strings (step={step_i})")
            out: List[str] = []
            for item in val:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"step_cert.{key} must be a list of non-empty strings (step={step_i})")
                if item.strip() != item:
                    raise ValueError(
                        f"step_cert.{key} items must not contain leading/trailing whitespace: {item!r} (step={step_i})"
                    )
                out.append(item)
            return out

        uses_premise_ids = _id_list(step_cert, "uses_premise_ids")
        uses_fact_ids = _id_list(step_cert, "uses_fact_ids")
        produces_fact_ids = _id_list(step_cert, "produces_fact_ids")

        valid_premise_ids = existing_premise_ids | set(new_premise_ids)
        valid_fact_ids = existing_fact_ids | set(new_fact_ids)

        # LLM robustness: sometimes a valid fact id is mistakenly placed into uses_premise_ids (or vice versa).
        # Auto-correct such cross-references before validating unknown IDs.
        misplaced_as_premise = [pid for pid in uses_premise_ids if pid not in valid_premise_ids and pid in valid_fact_ids]
        if misplaced_as_premise:
            logger.warning(
                "Auto-fixing step_cert id lists: moving fact IDs from uses_premise_ids to uses_fact_ids (step=%s): %s",
                step_i,
                sorted(set(misplaced_as_premise)),
            )
            uses_premise_ids = [pid for pid in uses_premise_ids if pid not in set(misplaced_as_premise)]
            uses_fact_ids = uses_fact_ids + misplaced_as_premise

        misplaced_as_fact = [fid for fid in uses_fact_ids if fid not in valid_fact_ids and fid in valid_premise_ids]
        if misplaced_as_fact:
            logger.warning(
                "Auto-fixing step_cert id lists: moving premise IDs from uses_fact_ids to uses_premise_ids (step=%s): %s",
                step_i,
                sorted(set(misplaced_as_fact)),
            )
            uses_fact_ids = [fid for fid in uses_fact_ids if fid not in set(misplaced_as_fact)]
            uses_premise_ids = uses_premise_ids + misplaced_as_fact

        # De-duplicate while preserving order.
        def _dedupe(items: List[str]) -> List[str]:
            seen: set[str] = set()
            out_items: List[str] = []
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                out_items.append(item)
            return out_items

        uses_premise_ids = _dedupe(uses_premise_ids)
        uses_fact_ids = _dedupe(uses_fact_ids)

        invalid_premise_refs = sorted({pid for pid in uses_premise_ids if pid not in valid_premise_ids})
        if invalid_premise_refs:
            raise ValueError(
                f"step_cert.uses_premise_ids contains unknown IDs (step={step_i}): {invalid_premise_refs}"
            )

        invalid_fact_refs = sorted({fid for fid in uses_fact_ids if fid not in valid_fact_ids})
        if invalid_fact_refs:
            raise ValueError(f"step_cert.uses_fact_ids contains unknown IDs (step={step_i}): {invalid_fact_refs}")

        # Executable chain constraint (fail-fast, no implicit compatibility):
        # For executable_chain_cert steps t>=2, step_cert.uses_fact_ids must include the previous step's key_fact_id.
        kind = step_cert.get("kind") if isinstance(step_cert, dict) else None
        if kind == "executable_chain_cert" and step_i >= 2:
            prev_key_fact_id: str | None = None
            for cert in KnownTree._filter_entries(mem.get("step_certs", [])):
                try:
                    if int(cert.get("step")) != (step_i - 1):
                        continue
                except Exception:
                    continue
                val = cert.get("key_fact_id")
                if isinstance(val, str) and val.strip():
                    prev_key_fact_id = val
                    break
            if not prev_key_fact_id:
                raise ValueError(
                    f"executable_chain_cert requires previous step key_fact_id (step={step_i}, prev_step={step_i-1})"
                )
            if prev_key_fact_id not in set(uses_fact_ids):
                raise ValueError(
                    "executable_chain_cert requires step_cert.uses_fact_ids to include previous key_fact_id "
                    f"(step={step_i}, expected={prev_key_fact_id})"
                )

        invalid_produces = sorted({fid for fid in produces_fact_ids if fid not in set(new_fact_ids)})
        if invalid_produces:
            raise ValueError(
                f"step_cert.produces_fact_ids must come from fact_delta IDs (step={step_i}): {invalid_produces}"
            )

        def _attach(entry: Dict[str, Any], source_key: str) -> Dict[str, Any]:
            item = dict(entry)
            if "id" not in item or not isinstance(item.get("id"), str):
                raise ValueError(f"missing {source_key} id")
            if not str(item.get("id")).strip():
                raise ValueError(f"missing {source_key} id")
            if "source_step" not in item:
                item["source_step"] = step_i
            if provenance and "provenance" not in item:
                item["provenance"] = provenance
            return item

        for premise in premise_delta:
            mem["premise_bank"].append(_attach(premise, "premise"))

        for fact in fact_delta:
            mem["fact_bank"].append(_attach(fact, "fact"))

        step_cert = dict(step_cert)
        step_cert.setdefault("step", step_i)
        if step_cert.get("step") != step_i:
            raise ValueError("step_cert.step must equal current step")
        step_cert["uses_premise_ids"] = uses_premise_ids
        step_cert["uses_fact_ids"] = uses_fact_ids
        step_cert["key_fact_id"] = key_fact_id
        if provenance and "provenance" not in step_cert:
            step_cert["provenance"] = provenance
        mem["step_certs"].append(step_cert)
        return mem


__all__ = [
    "KnownTree",
    "KNOWN_TREE_DESCRIPTION",
    "KNOWN_TREE_DESCRIPTION_EN",
    "KNOWN_TREE_FIELD_VISIBILITY",
    "normalize_known",
    "parse_known_to_dict",
]
