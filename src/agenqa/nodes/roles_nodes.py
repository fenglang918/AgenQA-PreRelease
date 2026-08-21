from __future__ import annotations

import logging
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from infra.input_adapters.paper_input_loader import load_one_paper_like_record
from infra.data.ids import generate_paper_id
from utils import ensure_dir

from agenqa.graph.output_manager import compute_step_dir
from agenqa.graph.state import AgentState, KQARecord
from agenqa.memory.store import dump_director_decision_for_step, save_state
from agenqa.domain.known_tree import KnownTree
from agenqa.domain.contracts.world_contract import merge_world_contract
from agenqa.domain.format_schema import format_output_to_dict
from agenqa.domain.draft_chain_schema import FIELD_DRAFT_QUESTION_EXPLICIT, draft_chain_output_to_dict
from agenqa.domain.draft_schema import FIELD_DRAFT_QUESTION
from agenqa.domain.folded_question_schema import FIELD_QUESTION_TEXT, dumps_folded_question
from agenqa.domain.numeric_oracle_schema import (
    FIELD_ABS_TOL,
    FIELD_NOTES,
    FIELD_ORACLE_CODE,
    FIELD_REL_TOL,
    FIELD_SIG_FIGS,
    FIELD_UNIT,
)
from agenqa.domain.path_fold_schema import path_fold_output_to_dict
from agenqa.domain.step_cert_schema import step_cert_output_to_dict
from agenqa.nodes.utils import (
    allowed_question_types_for_step,
    build_director_notes,
    has_zam_cal_marker,
    idealab_session_id_for_step_node,
    is_symbolic_only,
    is_symbolic_only_for_question_type,
    normalize_question_type,
    select_question_type,
    with_idealab_session_id,
)
from agenqa.revise_modes import (
    REVISE_MODE_CORRECTNESS,
    REVISE_MODE_WORLD_CONTRACT,
    REVISE_MODE_ANSWER_CONTRACT,
    REVISE_MODE_REUSE_HIDDEN,
    REVISE_MODE_QUALITY,
    is_reuse_hidden,
    normalize_revise_mode,
)
from agenqa.skills.episode_seed_builder import EpisodeSeedBuilderConfig, EpisodeSeedBuilderRunner
from agenqa.skills.diagnosing import DiagnoseConfig, DiagnoseInput, DiagnoseRunner
from agenqa.skills.draft_chain import DraftChainConfig, DraftChainInput, DraftChainRunner
from agenqa.skills.numeric_oracle import (
    NumericOracleConfig,
    NumericOracleInput,
    NumericOracleOutput,
    NumericOracleRunner,
    build_numeric_answer_format_sentence,
    execute_oracle_code,
    format_numeric_answer,
)
from agenqa.skills.path_fold import PathFoldConfig, PathFoldInput, PathFoldRunner
from agenqa.skills.step_cert_builder import StepCertConfig, StepCertInput, StepCertBuilderRunner
from agenqa.skills.formatting import FormatConfig, FormatInput, FormatRunner
from agenqa.skills.answer_contract_builder import (
    AnswerContractBuilderConfig,
    AnswerContractBuilderInput,
    AnswerContractBuilderRunner,
)
from agenqa.domain.contracts.answer_contract_bank import (
    build_answer_contract_validation_background,
    make_default_answer_contracts,
    persist_answer_contracts,
    validate_answer_contracts,
)
from agenqa.domain.contracts.solver_contract_text import (
    compose_solver_question,
    extract_solver_world_contract_text,
    strip_embedded_contract_blocks,
)
from agenqa.prompts.answer_contract_builder import (
    ANSWER_CONTRACT_BUILDER_V1,
    ANSWER_CONTRACT_BUILDER_V1_EN,
)
from agenqa.prompts.draft_chain import get_draft_chain_prompt
from agenqa.prompts.path_fold import PATH_FOLD_V1, PATH_FOLD_V1_EN
from agenqa.prompts.format import FORMAT_V1, FORMAT_V1_EN, FORMAT_V1_TAGGED, FORMAT_V1_TAGGED_EN
from agenqa.prompts.step_cert_builder import STEP_CERT_BUILDER_V1, STEP_CERT_BUILDER_V1_EN
from agenqa.prompts.episode_seed_builder import EPISODE_SEED_BUILDER_PROMPT, EPISODE_SEED_BUILDER_PROMPT_EN
from agenqa.prompts.diagnose import (
    DIAGNOSE_V1,
    DIAGNOSE_V1_TAGGED,
    DIAGNOSE_REVISE_CORRECTNESS,
    DIAGNOSE_REVISE_CORRECTNESS_TAGGED,
    DIAGNOSE_REVISE_WORLD_CONTRACT,
    DIAGNOSE_REVISE_WORLD_CONTRACT_TAGGED,
    DIAGNOSE_REVISE_ANSWER_CONTRACT,
    DIAGNOSE_REVISE_ANSWER_CONTRACT_TAGGED,
    DIAGNOSE_REVISE_DIFFICULTY,
    DIAGNOSE_REVISE_DIFFICULTY_TAGGED,
    DIAGNOSE_REVISE_REUSE_HIDDEN,
    DIAGNOSE_REVISE_REUSE_HIDDEN_TAGGED,
    DIAGNOSE_V1_EN,
    DIAGNOSE_V1_TAGGED_EN,
    DIAGNOSE_REVISE_CORRECTNESS_EN,
    DIAGNOSE_REVISE_CORRECTNESS_TAGGED_EN,
    DIAGNOSE_REVISE_WORLD_CONTRACT_EN,
    DIAGNOSE_REVISE_WORLD_CONTRACT_TAGGED_EN,
    DIAGNOSE_REVISE_ANSWER_CONTRACT_EN,
    DIAGNOSE_REVISE_ANSWER_CONTRACT_TAGGED_EN,
    DIAGNOSE_REVISE_DIFFICULTY_EN,
    DIAGNOSE_REVISE_DIFFICULTY_TAGGED_EN,
    DIAGNOSE_REVISE_REUSE_HIDDEN_EN,
    DIAGNOSE_REVISE_REUSE_HIDDEN_TAGGED_EN,
)

logger = logging.getLogger(__name__)

_PATH_FOLD_RECENT_EXACT_STEPS = 3
_PATH_FOLD_OLDER_QUESTION_SUMMARY_CHARS = 240
_PATH_FOLD_OLDER_ANSWER_SUMMARY_CHARS = 160


def _append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        return


def _record_numeric_error_chain(
    *,
    state: AgentState,
    step_idx: int,
    round_idx: int,
    op_name: str,
    step_dir: Path,
    raw_value: float,
    shown_value: str,
    unit: str,
    rounding_rule: Dict[str, Any],
    source: str,
) -> None:
    try:
        rel_step_dir = step_dir.relative_to(state.artifacts_dir).as_posix()
    except Exception:
        rel_step_dir = str(step_dir)
    _append_jsonl_row(
        state.artifacts_dir / "numeric_error_chain.jsonl",
        {
            "op": str(op_name),
            "round": int(round_idx),
            "step": int(step_idx),
            "step_dir": rel_step_dir,
            "raw_value": float(raw_value),
            "shown_value": str(shown_value or ""),
            "unit": str(unit or ""),
            "rounding_rule": dict(rounding_rule or {}),
            "source": str(source or ""),
        },
    )


def _get_int_conf(conf: Dict[str, Any], key: str, default: int) -> int:
    try:
        val = conf.get(key)
        if val is None:
            return default
        return int(val)
    except Exception:
        return default


def _get_bool_conf(conf: Dict[str, Any], key: str, default: bool) -> bool:
    try:
        val = conf.get(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            s = val.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        return bool(val)
    except Exception:
        return default


def _compact_text_for_summary(text: Any, limit: int) -> str:
    raw = "" if text is None else str(text)
    compact = " ".join(raw.split())
    if len(compact) <= max(0, int(limit)):
        return compact
    head = max(32, int(limit * 0.65))
    tail = max(16, int(limit * 0.2))
    if head + tail + 5 >= len(compact):
        return compact[:limit]
    return f"{compact[:head].rstrip()} ... {compact[-tail:].lstrip()}"


def _build_path_fold_history_payload(history: list[Any]) -> list[Dict[str, Any]] | Dict[str, Any]:
    records = list(history or [])
    if not records:
        return {"older_summary": [], "recent_steps": []}

    recent_exact_steps = max(1, _PATH_FOLD_RECENT_EXACT_STEPS)
    if len(records) <= recent_exact_steps:
        return {
            "older_summary": [],
            "recent_steps": [
                {
                    "step": int(getattr(rec, "qa_idx", getattr(rec, "step", 0)) or 0),
                    "question": getattr(rec, "question", ""),
                    "answer": getattr(rec, "answer", ""),
                }
                for rec in records
            ],
        }

    older = records[:-recent_exact_steps]
    recent = records[-recent_exact_steps:]
    older_summary: list[Dict[str, Any]] = []
    for rec in older:
        older_summary.append(
            {
                "step": int(getattr(rec, "qa_idx", getattr(rec, "step", 0)) or 0),
                "question_summary": _compact_text_for_summary(
                    getattr(rec, "question", ""),
                    _PATH_FOLD_OLDER_QUESTION_SUMMARY_CHARS,
                ),
                "answer_summary": _compact_text_for_summary(
                    getattr(rec, "answer", ""),
                    _PATH_FOLD_OLDER_ANSWER_SUMMARY_CHARS,
                ),
            }
        )

    recent_steps: list[Dict[str, Any]] = []
    for rec in recent:
        recent_steps.append(
            {
                "step": int(getattr(rec, "qa_idx", getattr(rec, "step", 0)) or 0),
                "question": getattr(rec, "question", ""),
                "answer": getattr(rec, "answer", ""),
            }
        )
    return {"older_summary": older_summary, "recent_steps": recent_steps}


def _persist_type2_answer_contracts(
    *,
    mem: Dict[str, Any],
    step_idx: int,
    question_type: Any,
    question: str,
    answer: str,
    where: str,
    raw_ref: str | None,
    numeric_oracle_out: Any | None = None,
    answer_contract_payload: Dict[str, Any] | None = None,
    report_path: Path | None = None,
) -> Dict[str, Any]:
    """Persist Type2 contracts (ACB-lite) deterministically.

    NOTE: This is internal-only governance; answer contracts must not be visible to solvers.
    """
    abs_tol = getattr(numeric_oracle_out, "abs_tol", None) if numeric_oracle_out is not None else None
    rel_tol = getattr(numeric_oracle_out, "rel_tol", None) if numeric_oracle_out is not None else None
    sig_figs = getattr(numeric_oracle_out, "sig_figs", None) if numeric_oracle_out is not None else None
    unit = getattr(numeric_oracle_out, "unit", None) if numeric_oracle_out is not None else None

    ids, contracts = make_default_answer_contracts(
        step=int(step_idx),
        question_type=question_type,
        question=question,
        answer=answer,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        sig_figs=sig_figs,
        unit=unit,
        answer_contract_payload=answer_contract_payload,
    )
    issue_err, issue_warn = validate_answer_contracts(contracts, world_contract=mem.get("world_contract"))
    mem2 = persist_answer_contracts(
        mem,
        step=int(step_idx),
        where=str(where or "").strip(),
        answer_contract_ids=ids,
        answer_contracts=contracts,
        issue_types_error=issue_err,
        issue_types_warn=issue_warn,
        raw_ref=raw_ref,
    )
    if report_path is not None:
        try:
            report_path.write_text(
                json.dumps(
                    {
                        "step": int(step_idx),
                        "question_type": str(question_type or "").strip(),
                        "where": str(where or "").strip(),
                        "answer_contract_ids": list(ids),
                        "answer_contract_payload": dict(answer_contract_payload or {}),
                        "issue_types_error": list(issue_err),
                        "issue_types_warn": list(issue_warn),
                        "error_count": int(len(issue_err)),
                        "warn_count": int(len(issue_warn)),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    return mem2


def _run_answer_contract_builder(
    *,
    state: AgentState,
    op_name: str,
    step_idx: int,
    step_dir: Path,
    op_conf: Dict[str, Any],
    generator: Dict[str, Any],
    agent_lang: str | None,
    question_type: Any,
    question: str,
    world_contract_text: str | None,
    answer: str,
) -> Dict[str, Any]:
    """Extract a builder-supplied v2 payload for Derivation answer contracts.

    This is a bounded augmentation layer over ACB-lite:
    - only Derivation uses the builder today
    - deterministic ACB generation remains the canonical contract writer
    - builder failures fall back to deterministic defaults instead of breaking Extend/Revise
    """
    if normalize_question_type(question_type) != "Derivation":
        return {
            "status": "skipped",
            "reason": "question_type_not_derivation",
            "answer_style": {},
            "answer_semantics": {},
            "support_witness": [],
        }

    builder_generator = (
        op_conf.get("answer_contract_builder_generator")
        or op_conf.get("answer_contract_generator")
        or op_conf.get("struct_generator")
        or generator
    )
    builder_generator = with_idealab_session_id(
        builder_generator,
        idealab_session_id_for_step_node(state, op_name, step_idx),
    )
    use_en = str(agent_lang or "").lower() in {"en", "english"}
    builder_prompt = ANSWER_CONTRACT_BUILDER_V1_EN if use_en else ANSWER_CONTRACT_BUILDER_V1

    solver_visible_question = compose_solver_question(question, world_contract_text)
    if not solver_visible_question.strip():
        solver_visible_question = str(question or "").strip()

    builder_runner = AnswerContractBuilderRunner(
        AnswerContractBuilderConfig(
            generator=builder_generator,
            prompt_path=Path(
                op_conf.get("answer_contract_builder_prompt_path")
                or op_conf.get("answer_contract_prompt_path")
                or "src/agenqa/prompts/answer_contract_builder.py"
            ),
            prompt_text=builder_prompt,
            lang=agent_lang,
        )
    )

    try:
        builder_out = builder_runner.run_one(
            AnswerContractBuilderInput(
                step=step_idx,
                question=solver_visible_question,
                answer=answer,
                question_type=normalize_question_type(question_type) or str(question_type or ""),
            ),
            snapshot_dir=step_dir / "subruns_raw" / "answer_contract_builder",
            unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "AnswerContractBuilder failed (%s step=%s), fallback to deterministic defaults: %s",
            str(op_name),
            str(step_idx),
            str(exc),
        )
        return {
            "status": "fallback_default",
            "reason": "builder_failed",
            "error": str(exc),
            "answer_style": {},
            "answer_semantics": {},
            "support_witness": [],
        }

    return {
        "status": "ok",
        "answer_style": dict(getattr(builder_out, "answer_style", None) or {}),
        "answer_semantics": dict(getattr(builder_out, "answer_semantics", None) or {}),
        "support_witness": list(getattr(builder_out, "support_witness", None) or []),
    }


def _append_symbolic_constraints(prompt_text: str, use_en: bool, question_type: Any) -> str:
    if not prompt_text:
        return prompt_text
    qtype = normalize_question_type(question_type) or str(question_type or "").strip()
    if qtype == "MCQ":
        if use_en:
            tail = (
                "\n\n"
                "[MCQ anti-numeric-evaluation constraints]\n"
                "- The question/options may contain concrete numbers, but do NOT require numeric evaluation "
                "(no decimal approximations, no tolerance rules like abs_tol/rel_tol, no rounding-to-N-decimals requirements).\n"
                "- Keep the MCQ protocol: the answer must be the option letter only (e.g., \\boxed{A})."
            )
        else:
            tail = (
                "\n\n"
                "【MCQ 禁数值求值口径约束】\n"
                "- 题干/选项允许出现具体数值，但禁止要求数值求值（例如“给出精确数值/保留 N 位小数/有效数字/abs_tol/rel_tol/误差口径”等）。\n"
                "- 严格遵守 MCQ 协议：Answer 仅输出唯一正确选项字母（如 \\boxed{A}），不要输出数值或解释。"
            )
    else:
        if use_en:
            tail = (
                "\n\n"
                "[Semantic-expression ONLY constraints]\n"
                "- The Question/Answer must not include any concrete numeric values or orders of magnitude "
                "(e.g., 1.2e3, 2.6*10^3, 0.155).\n"
                "- Use symbols for all physical quantities/variables; you may use small dimensionless integers as coefficients/exponents.\n"
                "- The Answer must be a symbolic expression wrapped in \\boxed{...}; do not provide decimal approximations or numeric evaluations."
            )
        else:
            tail = (
                "\n\n"
                "【符号表达式 ONLY 约束】\n"
                "- Question/Answer 禁止给出具体数值或数量级（例如 1.2e3、2.6*10^3、0.155）。\n"
                "- 所有物理量/变量一律使用符号表示，可用少量无量纲整数作为系数或指数。\n"
                "- Answer 必须给出符号解析式（用 \\boxed{...} 包裹），不要提供小数近似或单位化数值。"
            )
    return f"{prompt_text}{tail}"


def _strip_trailing_answer_format_paragraph(text: str) -> str:
    """Remove a trailing 'answer format' paragraph (best-effort) before appending a deterministic one."""
    if not isinstance(text, str) or not text.strip():
        return "" if text is None else str(text)
    s = text.rstrip()
    parts = s.split("\n\n")
    if not parts:
        return s
    last = parts[-1].strip()
    if last.startswith("答案格式") or last.lower().startswith("answer format"):
        parts = parts[:-1]
        return "\n\n".join(parts).rstrip()
    return s


def _strip_solver_contract_from_question(text: str) -> str:
    """Keep the core question body separate from solver-visible contract text."""
    raw = str(text or "").strip()
    if not raw:
        return raw
    cleaned = strip_embedded_contract_blocks(raw)
    cleaned = _strip_trailing_answer_format_paragraph(cleaned).strip()
    return cleaned or raw


def _is_calc_prompt_marker(val: Any) -> bool:
    return has_zam_cal_marker(val)


def episode_seed_builder_node(agent_conf: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    """Known-Init：基于单一 contract 从 paper text 构造 episode_seed。"""
    if state.history:
        logger.warning("episode_seed_builder_node 调用时 history 非空，跳过 Known‑Init 链路。")
        return {}

    from infra.config.init_config import parse_init_config

    init_cfg = parse_init_config(agent_conf)
    data_conf = agent_conf.get("data") or {}

    agent_lang = str((agent_conf.get("agent") or {}).get("lang") or "").lower() or None
    use_en = agent_lang in {"en", "english"}

    summary_dir = state.artifacts_dir / "00_Summary"
    ensure_dir(str(summary_dir))

    step_idx = 0
    round_idx = state.current_round_index()
    step_dir = compute_step_dir(state.artifacts_dir, "init", step_idx, round_idx)
    step_dir.mkdir(parents=True, exist_ok=True)
    dump_director_decision_for_step(state, step_dir, step_idx)

    # 会话亲和：init 全链路共享一个 step session id。
    step_session_id = idealab_session_id_for_step_node(state, "init", step_idx)

    paper: Dict[str, Any] = {}
    extracted_text = ""
    extra_role_outputs: Dict[str, Any] = {}
    if getattr(init_cfg, "source", None) is None:
        raise RuntimeError("Known-Init missing init.source")

    source_type = str(getattr(init_cfg.source, "type", "") or "").strip().lower() or "paper"
    if source_type == "paper":
        try:
            paper = load_one_paper_like_record(agent_conf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("episode_seed_builder_node 无法读取论文输入，终止 Known‑Init 链路: %s", str(exc))
            state.stop_reason = "qa_init_no_paper"
            return {}
    elif source_type == "domain_seed_walk":
        try:
            from datetime import datetime
            from copy import deepcopy as _dc

            from infra.domain_seed_walk.run import WalkConfig, run_walk
            from infra.domain_seed_walk.playback import build_playback
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"domain_seed_walk import failed: {exc}") from exc

        dsw = getattr(init_cfg.source, "domain_seed_walk", None) or {}
        if not isinstance(dsw, dict):
            dsw = {}
        root_domain = dsw.get("root_domain")
        if not isinstance(root_domain, str) or not root_domain.strip():
            raise RuntimeError("init.source.domain_seed_walk.root_domain is required")
        root_domain = root_domain.strip()

        def _int_conf(key: str, default: int) -> int:
            try:
                return max(1, int(dsw.get(key, default)))
            except Exception:
                return default

        depth = _int_conf("depth", 4)
        branching = _int_conf("branching", 5)
        keywords_per_leaf = _int_conf("keywords_per_leaf", 10)

        dsw_lang = str(dsw.get("lang") or "").strip().lower() or ("en" if use_en else "zh")
        dsw_lang = "zh" if dsw_lang in {"zh", "cn", "zh-cn", "zh-hans"} else "en"

        temperature = None
        if dsw.get("temperature") is not None:
            try:
                temperature = float(dsw.get("temperature"))
            except Exception:
                temperature = None
        max_tokens = None
        if dsw.get("max_tokens") is not None:
            try:
                max_tokens = int(dsw.get("max_tokens"))
            except Exception:
                max_tokens = None

        # generator: allow per-source override; otherwise fallback to init.generator
        dsw_gen = dsw.get("generator") if isinstance(dsw.get("generator"), dict) and dsw.get("generator") else None
        if dsw_gen is None:
            dsw_gen = getattr(init_cfg, "generator", None)
        if not isinstance(dsw_gen, dict) or not dsw_gen:
            raise RuntimeError("domain_seed_walk requires init.generator or init.source.domain_seed_walk.generator")

        # Avoid mutating init.generator; domain_seed_walk may override generation knobs.
        dsw_gen = _dc(dsw_gen)
        dsw_gen = with_idealab_session_id(dsw_gen, step_session_id)

        subruns_dir = step_dir / "subruns_raw" / "domain_seed_walk"
        subruns_dir.mkdir(parents=True, exist_ok=True)
        run_dir = subruns_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        walk_conf = WalkConfig(
            root_domain=root_domain,
            depth=depth,
            branching=branching,
            keywords_per_leaf=keywords_per_leaf,
            lang=dsw_lang,
            output_dir=run_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            generator=dsw_gen,
        )
        setattr(walk_conf, "no_trace_context", bool(dsw.get("no_trace_context", False)))
        dsw_result = run_walk(walk_conf)
        try:
            (run_dir / "playback.md").write_text(build_playback(run_dir, preview_chars=0), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write domain_seed_walk playback.md (ignored).")

        # Render a compact, deterministic "material text" for EpisodeSeedBuilder.
        path_trace = dsw_result.get("path_trace") if isinstance(dsw_result, dict) else None
        if not isinstance(path_trace, list):
            path_trace = []

        if dsw_lang == "zh":
            lines = [
                "材料类型：研究领域分解（domain taxonomy）+ 叶子关键词（problem keywords）",
                f"根域（root domain）：{root_domain}",
                "",
                "路径轨迹（chosen path）：",
            ]
            for e in path_trace:
                if not isinstance(e, dict):
                    continue
                lvl = e.get("level")
                inp = e.get("input_domain")
                ch = e.get("chosen")
                tags = e.get("chosen_context_tags") if isinstance(e.get("chosen_context_tags"), list) else []
                tags = [str(x).strip() for x in tags if str(x).strip()]
                tags_part = f"（tags: {', '.join(tags)}）" if tags else ""
                if lvl and inp and ch:
                    lines.append(f"- L{lvl}: {inp} -> {ch}{tags_part}")
            leaf_domain = str(dsw_result.get("leaf_domain") or "").strip()
            if leaf_domain:
                lines.extend(["", f"叶子域（leaf domain）：{leaf_domain}"])
            kws = dsw_result.get("problem_keywords") if isinstance(dsw_result.get("problem_keywords"), list) else []
            kws = [str(x).strip() for x in kws if str(x).strip()]
            if kws:
                lines.extend(["", "叶子关键词（problem keywords）："])
                lines.extend([f"- {kw}" for kw in kws])
            extracted_text = "\n".join(lines).strip()
            paper = {
                "paper_id": generate_paper_id({"domain_seed_walk": dsw_result}),
                "title": f"Domain seed (domain_seed_walk): {root_domain}",
                "abstract": "",
                "text": extracted_text,
                "meta": {"source_kind": "domain_seed_walk", "domain_seed_walk_run_dir": str(run_dir)},
            }
        else:
            lines = [
                "Material type: domain taxonomy + leaf problem keywords",
                f"Root domain: {root_domain}",
                "",
                "Chosen path trace:",
            ]
            for e in path_trace:
                if not isinstance(e, dict):
                    continue
                lvl = e.get("level")
                inp = e.get("input_domain")
                ch = e.get("chosen")
                tags = e.get("chosen_context_tags") if isinstance(e.get("chosen_context_tags"), list) else []
                tags = [str(x).strip() for x in tags if str(x).strip()]
                tags_part = f" (tags: {', '.join(tags)})" if tags else ""
                if lvl and inp and ch:
                    lines.append(f"- L{lvl}: {inp} -> {ch}{tags_part}")
            leaf_domain = str(dsw_result.get("leaf_domain") or "").strip()
            if leaf_domain:
                lines.extend(["", f"Leaf domain: {leaf_domain}"])
            kws = dsw_result.get("problem_keywords") if isinstance(dsw_result.get("problem_keywords"), list) else []
            kws = [str(x).strip() for x in kws if str(x).strip()]
            if kws:
                lines.extend(["", "Leaf problem keywords:"])
                lines.extend([f"- {kw}" for kw in kws])
            extracted_text = "\n".join(lines).strip()
            paper = {
                "paper_id": generate_paper_id({"domain_seed_walk": dsw_result}),
                "title": f"Domain seed (domain_seed_walk): {root_domain}",
                "abstract": "",
                "text": extracted_text,
                "meta": {"source_kind": "domain_seed_walk", "domain_seed_walk_run_dir": str(run_dir)},
            }

        inputs_dir = step_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        (inputs_dir / "domain_seed_walk_material.txt").write_text(extracted_text, encoding="utf-8")
        extra_role_outputs["domain_seed_walk"] = {"run_dir": str(run_dir), "result": dsw_result}
        # Stash to state for later debug (optional)
        try:
            (inputs_dir / "domain_seed_walk.result.json").write_text(
                json.dumps(dsw_result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
    else:
        raise RuntimeError(f"Unsupported init.source.type={source_type!r}")

    if not extracted_text:
        try:
            extracted_parts: list[str] = []
            title = paper.get("title")
            abstract = paper.get("abstract")
            text = paper.get("text") or paper.get("content")
            meta = paper.get("meta") if isinstance(paper.get("meta"), dict) else {}
            if isinstance(title, str) and title.strip():
                extracted_parts.append(f"Title: {title.strip()}")
            if isinstance(abstract, str) and abstract.strip():
                extracted_parts.append(f"Abstract: {abstract.strip()}")
            if isinstance(text, str) and text.strip():
                extracted_parts.append(text.strip())
            elif isinstance(meta, dict) and meta.get("source_kind") == "pdf" and meta.get("pdf_attachment"):
                extracted_parts.append("NOTE: The full paper is provided as an attached PDF document. Please read the attachment.")
            extracted_text = "\n\n".join(extracted_parts).strip()
            if extracted_text:
                inputs_dir = step_dir / "inputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                (inputs_dir / "paper_extracted.txt").write_text(extracted_text, encoding="utf-8")
        except Exception:
            pass

    if not extracted_text:
        try:
            extracted_text = json.dumps(paper, ensure_ascii=False, indent=2)
        except Exception:
            extracted_text = ""
    if not extracted_text:
        logger.error("Known-Init 缺少可用的论文文本。")
        raise RuntimeError("Known-Init missing paper text")

    enabled = bool(getattr(getattr(init_cfg, "episode_seed", None), "enable", True))
    if not enabled:
        raise RuntimeError("init.episode_seed.enable=false 时无法生成 episode_seed（无 fallback）。")

    seed_raw_dir = step_dir / "subruns_raw" / "episode_seed_builder"
    seed_raw_dir.mkdir(parents=True, exist_ok=True)

    # generator: allow per-module override; otherwise fallback to init.generator
    seed_gen = getattr(init_cfg.episode_seed, "generator", None) or getattr(init_cfg, "generator", None)
    if not isinstance(seed_gen, dict) or not seed_gen:
        raise RuntimeError("EpisodeSeedBuilder requires init.generator or init.episode_seed.generator.")
    seed_gen = with_idealab_session_id(seed_gen, step_session_id)

    override_prompt_path = getattr(init_cfg.episode_seed, "prompt_path", None)
    default_prompt_path = "src/agenqa/prompts/episode_seed_builder.prompt"
    default_prompt_text = EPISODE_SEED_BUILDER_PROMPT_EN if use_en else EPISODE_SEED_BUILDER_PROMPT
    runner = EpisodeSeedBuilderRunner(
        EpisodeSeedBuilderConfig(
            generator=seed_gen,
            prompt_path=Path(
                str(override_prompt_path)
                if isinstance(override_prompt_path, Path)
                else (str(override_prompt_path).strip() if isinstance(override_prompt_path, str) else "")
            )
            if (override_prompt_path is not None and str(override_prompt_path).strip())
            else Path(default_prompt_path),
            prompt_text=default_prompt_text,
            lang=agent_lang,
        )
    )

    contract = getattr(init_cfg.episode_seed, "contract", None) or {}
    if not isinstance(contract, dict) or not contract:
        raise RuntimeError("EpisodeSeedBuilder missing contract (init.episode_seed.contract).")

    validation = getattr(init_cfg.episode_seed, "validation", None) or {}
    strict_validation = bool(validation.get("strict", True))
    try:
        retry_on_fail = int(validation.get("retry_on_fail", 0) or 0)
    except Exception:
        retry_on_fail = 0
    retry_on_fail = max(0, retry_on_fail)

    seed_raw: Dict[str, Any] = {}
    last_exc: Exception | None = None
    for attempt in range(retry_on_fail + 1):
        attempt_dir = seed_raw_dir / f"attempt_{attempt+1}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        try:
            seed_raw = runner.run_one(
                extracted_text,
                paper=paper,
                contract=contract,
                strict_validation=strict_validation,
                snapshot_dir=attempt_dir,
                unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
            )
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "EpisodeSeedBuilder 尝试失败（%s/%s）: %s",
                attempt + 1,
                retry_on_fail + 1,
                str(exc),
            )
    if last_exc is not None:
        raise last_exc

    qa_ctx: Dict[str, Any] = {
        "paper": paper,
        "init_cfg": init_cfg,
        "agent_lang": agent_lang,
        "data_conf": data_conf,
        "summary_dir": summary_dir,
        "step_dir": step_dir,
        "step_idx": step_idx,
        "round_idx": round_idx,
        "episode_seed": seed_raw,
    }
    role_outputs = qa_ctx.get("role_outputs") or {}
    if isinstance(extra_role_outputs, dict) and extra_role_outputs:
        role_outputs.update(extra_role_outputs)
    role_outputs["episode_seed_builder"] = {
        "episode_seed": seed_raw,
        "contract_path": str(getattr(init_cfg.episode_seed, "contract_path", "") or ""),
    }
    qa_ctx["role_outputs"] = role_outputs
    return qa_ctx


def known_init_seed_node(
    agent_conf: Dict[str, Any],
    state: AgentState,
    qa_ctx: Dict[str, Any],
) -> tuple[AgentState, Dict[str, Any], Path, int, int]:
    """Known-Init 终点：仅生成 episode_seed，写入 state.memory，不产 QA。"""
    qa_ctx = deepcopy(qa_ctx or {})
    if not qa_ctx:
        logger.error("known_init_seed_node 缺少 known_init 上下文。")
        raise RuntimeError("Known-Init Seed 缺少上下文")

    paper = qa_ctx.get("paper") or {}
    step_dir: Path = qa_ctx.get("step_dir")
    step_idx = int(qa_ctx.get("step_idx") or 0)
    round_idx = int(qa_ctx.get("round_idx") or state.current_round_index())

    seed = qa_ctx.get("episode_seed") or {}
    if not isinstance(seed, dict):
        logger.error("Known-Init Seed missing episode_seed payload.")
        raise RuntimeError("Known-Init Seed missing episode_seed")

    state.memory = KnownTree.update_episode_seed(state.memory, seed=seed)
    state.paper_id = str(paper.get("paper_id") or generate_paper_id(paper))
    subj = seed.get("subject")
    state.subject = subj.strip() if isinstance(subj, str) and subj.strip() else None

    role_outputs = qa_ctx.get("role_outputs") or {}
    role_outputs["seed_init"] = {
        "episode_seed": KnownTree.normalize_memory(state.memory).get("episode_seed") or {},
    }
    qa_ctx["role_outputs"] = role_outputs
    save_state(state)
    return state, role_outputs, step_dir, step_idx, round_idx


# Backward-compatible name (deprecated)
known_init_background_node = known_init_seed_node


# Legacy QA-Init draft/format nodes removed.
# 当前架构中，QA-Init（旧名）等价于：
# EpisodeSeedBuilder（单次 LLM + contract）→ SeedInit（写入 episode_seed）→ Extend（Draft+Format 出第一题）。


def extend_draft_node(agent_conf: Dict[str, Any], state: AgentState) -> tuple[AgentState, Dict[str, Any]]:
    """Extend 第一步：生成 DraftChain 草稿。"""
    memory = KnownTree.normalize_memory(getattr(state, "memory", None))
    episode_seed = memory.get("episode_seed") or {}
    anchor = episode_seed.get("anchor")
    has_anchor = isinstance(anchor, str) and anchor.strip()
    if not has_anchor and not (episode_seed.get("subject") or episode_seed.get("keywords")):
        logger.warning("extend_draft_node：episode_seed 为空，跳过。")
        state.stop_reason = "extend_no_seed"
        return state, {}

    ops_conf = (agent_conf.get("operators") or {})
    op_conf = (ops_conf.get("extend") or {}) or {}
    generator = op_conf.get("generator") or {
        "service_type": "private_endpoint",
        "service_id": op_conf.get("service_id"),
    }
    agent_block = agent_conf.get("agent") or {}
    agent_lang = str(agent_block.get("lang") or "").lower() or None
    use_en = agent_lang in {"en", "english"}
    draft_chain_window = _get_int_conf(agent_block, "draft_chain_window", 2)

    prev_step = int(state.qa_idx) if state.history else 0
    step_idx = prev_step + 1
    # 会话亲和：extend(step_idx) 内的所有子调用共享一个 session id（DraftChain/Format/...）。
    generator = with_idealab_session_id(generator, idealab_session_id_for_step_node(state, "extend", step_idx))
    round_idx = state.current_round_index()
    step_dir = compute_step_dir(state.artifacts_dir, "extend", step_idx, round_idx)
    step_dir.mkdir(parents=True, exist_ok=True)
    dump_director_decision_for_step(state, step_dir, step_idx)

    question_type = select_question_type(state)
    # On the first extend there may be no Director decision yet. In that case,
    # honor the configured question-type policy instead of falling back to MCQ.
    try:
        has_director_qtype = False
        params = getattr(getattr(state, "last_decision", None), "params", None)
        if isinstance(params, dict):
            has_director_qtype = normalize_question_type(params.get("question_type")) is not None
        if not has_director_qtype:
            allowed_qtypes_for_step = allowed_question_types_for_step(agent_conf, step_idx)
            if allowed_qtypes_for_step:
                question_type = allowed_qtypes_for_step[0]
    except Exception:
        pass
    symbolic_only = is_symbolic_only_for_question_type(agent_conf, question_type)
    director_notes = build_director_notes(state, include_solver_feedback=False)
    if symbolic_only:
        qtype = normalize_question_type(question_type) or str(question_type or "").strip()
        if qtype == "MCQ":
            guard = (
                "【约束提醒】本步为 MCQ（禁数值求值口径）：题干可包含数值，但不得要求数值计算/小数近似/误差口径；"
                "Answer 必须仅输出选项字母 \\boxed{A/B/C/D}。若 director_notes 中包含数值求值/abs_tol/保留几位等建议，请忽略。"
            )
        else:
            guard = (
                "【约束提醒】本步为符号表达式 ONLY：不得在 Question/Answer 中要求或给出数值求值/小数近似/误差口径。"
                "若 director_notes 中包含此类数值计算建议，请忽略并改为符号推导任务。"
            )
        director_notes = (guard if not director_notes else f"{guard} | {director_notes}")
    chain_view = KnownTree.build_draft_chain_view(state.memory, step_idx, window=draft_chain_window)
    chain_view_json = KnownTree.to_json(chain_view)
    expected_primary = KnownTree.key_fact_id_for_step(state.memory, step_idx - 1) if step_idx >= 2 else ""

    use_calc = _is_calc_prompt_marker(op_conf.get("prompt_path")) or _is_calc_prompt_marker(
        op_conf.get("draft_chain_prompt_path")
    )
    # Default ON: always keep a structured world_contract object in memory to prevent
    # cross-step paradigm drift. Users may explicitly disable it to reduce burden/cost.
    wc_enable = _get_bool_conf(op_conf, "enable_world_contract", True)
    prompt_text = get_draft_chain_prompt(
        question_type=question_type,
        use_en=use_en,
        calc=use_calc,
        world_contract=wc_enable,
    )
    if symbolic_only:
        prompt_text = _append_symbolic_constraints(prompt_text, use_en, question_type)

    qtype_key = str(question_type or "").strip().lower() or "derivation"
    default_prompt_path = f"src/agenqa/prompts/draft_chain_{qtype_key}{'_calc' if use_calc else ''}.prompt"
    draft_runner = DraftChainRunner(
        DraftChainConfig(
            generator=generator,
            prompt_path=Path(op_conf.get("draft_chain_prompt_path") or default_prompt_path),
            prompt_text=prompt_text,
            lang=agent_lang,
        )
    )
    draft_in = DraftChainInput(
        chain_view_json=chain_view_json,
        prev_step=prev_step,
        step=step_idx,
        director_notes=director_notes,
        question_type=question_type,
        expected_primary_fact_id=expected_primary,
    )
    draft_out = draft_runner.run_one(
        draft_in,
        snapshot_dir=step_dir / "subruns_raw" / "draft_chain",
        unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
    )

    if not draft_out:
        logger.error("Extend DraftChain 未产出有效结果。")
        raise RuntimeError("Extend DraftChain failed")

    contract_violations: list[str] = []
    if step_idx >= 2:
        required = [str(x) for x in (draft_out.required_fact_ids or []) if str(x).strip()]
        primary = str(draft_out.primary_required_fact_id or "")
        if not expected_primary:
            # No hard-stop: record the mismatch and keep going.
            contract_violations.append(
                "missing expected_primary_fact_id from previous step (memory mismatch; skipping strict enforcement)"
            )
            if primary and primary.strip() and primary not in required:
                contract_violations.append(f"required_fact_ids missing primary_required_fact_id: {primary!r}")
        else:
            if (primary or "").strip() != expected_primary:
                contract_violations.append(
                    f"primary_required_fact_id mismatch: {draft_out.primary_required_fact_id!r} vs expected={expected_primary!r}"
                )
            if expected_primary not in required:
                contract_violations.append(f"required_fact_ids missing expected_primary_fact_id: {expected_primary!r}")
    else:
        if draft_out.required_fact_ids or draft_out.primary_required_fact_id:
            contract_violations.append("step=1 must not require prior facts")

    draft_dict = draft_chain_output_to_dict(draft_out)
    if contract_violations:
        logger.warning("DraftChain contract violations (observed; no auto-fix): %s", "; ".join(contract_violations))
    ext_ctx: Dict[str, Any] = {
        "step_idx": step_idx,
        "round_idx": round_idx,
        "step_dir": step_dir,
        "op_conf": op_conf,
        "generator": generator,
        "agent_lang": agent_lang,
        "question_type": question_type,
        "director_notes": director_notes,
        "chain_view_json": chain_view_json,
        "draft_chain_dict": draft_dict,
        "role_outputs": {
            "draft_chain": draft_dict,
            "draft_chain_contract": {
                "step": step_idx,
                "expected_primary_fact_id": expected_primary,
                "observed_primary_required_fact_id": draft_out.primary_required_fact_id,
                "observed_required_fact_ids": list(draft_out.required_fact_ids or []),
                "violations": contract_violations,
            },
        },
    }
    return state, ext_ctx


def extend_format_node(
    agent_conf: Dict[str, Any],
    state: AgentState,
    ext_ctx: Dict[str, Any],
) -> tuple[AgentState, Dict[str, Any], Path, int, int]:
    """Extend 第二步：Format + StepCertBuilder，并写入 memory/history。"""
    ext_ctx = deepcopy(ext_ctx or {})
    if not ext_ctx:
        logger.error("extend_format_node 缺少 extend 上下文。")
        raise RuntimeError("Extend format 缺少上下文")

    step_idx = int(ext_ctx.get("step_idx") or (state.step or 0) + 1)
    round_idx = int(ext_ctx.get("round_idx") or state.current_round_index())
    step_dir: Path = ext_ctx.get("step_dir")
    op_conf = ext_ctx.get("op_conf") or {}
    generator = ext_ctx.get("generator") or {}
    agent_lang = ext_ctx.get("agent_lang")
    agent_block = agent_conf.get("agent") or {}
    use_en = str(agent_lang or "").lower() in {"en", "english"}
    roles_protocol = str(
        agent_block.get("roles_protocol") or agent_block.get("draft_protocol") or ""
    ).strip().lower() or None

    question_type = ext_ctx.get("question_type")
    symbolic_only = is_symbolic_only_for_question_type(agent_conf, question_type)
    draft_chain_dict = ext_ctx.get("draft_chain_dict") or {}

    # Format
    format_prompt = FORMAT_V1_TAGGED_EN if use_en and roles_protocol == "tagged" else (
        FORMAT_V1_TAGGED if (roles_protocol == "tagged") else (FORMAT_V1_EN if use_en else FORMAT_V1)
    )
    if symbolic_only:
        format_prompt = _append_symbolic_constraints(format_prompt, use_en, question_type)
    draft_pack = dict(draft_chain_dict)
    draft_pack[FIELD_DRAFT_QUESTION] = str(draft_chain_dict.get(FIELD_DRAFT_QUESTION_EXPLICIT) or "")
    if FIELD_DRAFT_QUESTION_EXPLICIT in draft_pack:
        del draft_pack[FIELD_DRAFT_QUESTION_EXPLICIT]
    format_generator = (
        op_conf.get("format_generator")
        or op_conf.get("struct_generator")
        or (op_conf.get("format") or {}).get("generator")
        or generator
    )
    format_generator = with_idealab_session_id(
        format_generator,
        idealab_session_id_for_step_node(state, "extend", step_idx),
    )
    fmt_runner = FormatRunner(
        FormatConfig(
            generator=format_generator,
            prompt_path=Path(op_conf.get("format_prompt_path") or "src/agenqa/prompts/format.prompt"),
            prompt_text=format_prompt,
            lang=agent_lang,
            protocol=roles_protocol,
        )
    )
    fmt_in = FormatInput(
        draft_json=json.dumps(draft_pack, ensure_ascii=False),
        prev_step=step_idx - 1,
        step=step_idx,
        question_type=question_type,
    )
    fmt_out = fmt_runner.run_one(
        fmt_in,
        snapshot_dir=step_dir / "subruns_raw" / "format",
        unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
    )
    if not fmt_out:
        logger.error("Extend Format 未产出有效结果。")
        raise RuntimeError("Extend Format failed")

    # Numeric oracle (tool-assisted): compute GT value + tolerance, then override Answer deterministically.
    numeric_oracle_out = None
    if str(question_type or "").strip() == "Numeric":
        oracle_block = agent_conf.get("numeric_oracle") or {}
        oracle_source = "unknown"
        # Prefer draft-time oracle_code from DraftChain and execute it locally; fall back to the
        # NumericOracle LLM role only when oracle_code is missing or execution fails.
        draft_oracle_code = ""
        try:
            draft_oracle_code = str(draft_chain_dict.get(FIELD_ORACLE_CODE) or "").strip()
        except Exception:
            draft_oracle_code = ""

        if draft_oracle_code:
            snap_dir = step_dir / "subruns_raw" / "numeric_oracle"
            try:
                snap_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                (snap_dir / "input_view.json").write_text(
                    json.dumps(
                        {
                            "source": "draft_chain",
                            "mode": "extend",
                            "step": step_idx,
                            "question_type": str(question_type or ""),
                            "timeout_seconds": float(oracle_block.get("timeout_seconds", 10.0)),
                            "memory_limit_mb": int(oracle_block.get("memory_limit_mb", 4096)),
                            "temp_dir": str(oracle_block.get("temp_dir") or str(Path("/tmp").resolve())),
                            "python_bin": str(oracle_block.get("python_bin") or sys.executable),
                            "question": fmt_out.question,
                            "solution": fmt_out.solution,
                            FIELD_ABS_TOL: draft_chain_dict.get(FIELD_ABS_TOL),
                            FIELD_REL_TOL: draft_chain_dict.get(FIELD_REL_TOL),
                            FIELD_SIG_FIGS: draft_chain_dict.get(FIELD_SIG_FIGS),
                            FIELD_UNIT: draft_chain_dict.get(FIELD_UNIT),
                            FIELD_NOTES: draft_chain_dict.get(FIELD_NOTES),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (snap_dir / "oracle_code.py").write_text(draft_oracle_code, encoding="utf-8")
            except Exception:
                pass

            try:
                gt, exec_payload = execute_oracle_code(
                    draft_oracle_code,
                    timeout_seconds=float(oracle_block.get("timeout_seconds", 10.0)),
                    memory_limit_mb=int(oracle_block.get("memory_limit_mb", 4096)),
                    temp_dir=str(oracle_block.get("temp_dir") or str(Path("/tmp").resolve())),
                    python_bin=str(oracle_block.get("python_bin") or sys.executable),
                )
                numeric_oracle_out = NumericOracleOutput(
                    abs_tol=draft_chain_dict.get(FIELD_ABS_TOL),
                    rel_tol=draft_chain_dict.get(FIELD_REL_TOL),
                    sig_figs=draft_chain_dict.get(FIELD_SIG_FIGS),
                    unit=str(draft_chain_dict.get(FIELD_UNIT) or "").strip(),
                    oracle_code=draft_oracle_code,
                    gt_value=gt,
                    exec_payload=exec_payload,
                    notes=str(draft_chain_dict.get(FIELD_NOTES) or "").strip(),
                )
                try:
                    (snap_dir / "executor_result.json").write_text(
                        json.dumps(exec_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                if numeric_oracle_out.gt_value is None:
                    raise RuntimeError("draft_chain oracle_code executed but gt_value is None")
                oracle_source = "draft_chain_oracle_code_exec"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DraftChain oracle_code execution failed, falling back to NumericOracle role: %s",
                    str(exc),
                )
                numeric_oracle_out = None

        if numeric_oracle_out is None:
            oracle_gen = (
                op_conf.get("numeric_oracle_generator")
                or op_conf.get("oracle_generator")
                or op_conf.get("struct_generator")
                or generator
            )
            oracle_gen = with_idealab_session_id(
                oracle_gen, idealab_session_id_for_step_node(state, "extend", step_idx)
            )
            oracle_runner = NumericOracleRunner(
                NumericOracleConfig(
                    generator=oracle_gen,
                    prompt_path=Path(op_conf.get("numeric_oracle_prompt_path") or "src/agenqa/prompts/numeric_oracle.py"),
                    lang=agent_lang,
                    timeout_seconds=float(oracle_block.get("timeout_seconds", 10.0)),
                    memory_limit_mb=int(oracle_block.get("memory_limit_mb", 4096)),
                    temp_dir=str(oracle_block.get("temp_dir") or str(Path("/tmp").resolve())),
                    python_bin=str(oracle_block.get("python_bin") or sys.executable),
                )
            )
            numeric_oracle_out = oracle_runner.run_one(
                NumericOracleInput(step=step_idx, question=fmt_out.question, solution=fmt_out.solution),
                snapshot_dir=step_dir / "subruns_raw" / "numeric_oracle",
                unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
            )
            if numeric_oracle_out.gt_value is None:
                raise RuntimeError("NumericOracle failed to produce gt_value")
            oracle_source = "numeric_oracle_role"
        fmt_out.answer = format_numeric_answer(numeric_oracle_out.gt_value, sig_figs=numeric_oracle_out.sig_figs)
        tolerance_sentence = build_numeric_answer_format_sentence(
            abs_tol=numeric_oracle_out.abs_tol,
            rel_tol=numeric_oracle_out.rel_tol,
            sig_figs=numeric_oracle_out.sig_figs,
            lang=agent_lang,
        )
        _record_numeric_error_chain(
            state=state,
            step_idx=step_idx,
            round_idx=round_idx,
            op_name="extend",
            step_dir=step_dir,
            raw_value=float(numeric_oracle_out.gt_value),
            shown_value=str(fmt_out.answer or ""),
            unit=str(numeric_oracle_out.unit or ""),
            rounding_rule={
                "sig_figs": numeric_oracle_out.sig_figs,
                "abs_tol": numeric_oracle_out.abs_tol,
                "rel_tol": numeric_oracle_out.rel_tol,
                "tolerance_sentence": tolerance_sentence,
            },
            source=oracle_source,
        )

    raw_question_for_contracts = str(fmt_out.question or "")
    fmt_out.world_contract = str(getattr(fmt_out, "world_contract", "") or "").strip()
    fmt_out.question = _strip_solver_contract_from_question(raw_question_for_contracts)
    fmt_dict = format_output_to_dict(fmt_out)

    format_validation = None
    if getattr(fmt_out, "validation_passed", True) is False:
        format_validation = {
            "mode": "extend",
            "attempt_step": int(getattr(fmt_out, "step", step_idx) or step_idx),
            "validation_passed": False,
            "validation_errors": list(getattr(fmt_out, "validation_errors", None) or []),
        }

    # StepCertBuilder
    cert_prompt = STEP_CERT_BUILDER_V1_EN if use_en else STEP_CERT_BUILDER_V1
    cert_generator = op_conf.get("step_cert_generator") or op_conf.get("struct_generator") or generator
    cert_generator = with_idealab_session_id(
        cert_generator,
        idealab_session_id_for_step_node(state, "extend", step_idx),
    )
    cert_runner = StepCertBuilderRunner(
        StepCertConfig(
            generator=cert_generator,
            prompt_path=Path(op_conf.get("step_cert_prompt_path") or "src/agenqa/prompts/step_cert_builder.prompt"),
            prompt_text=cert_prompt,
            lang=agent_lang,
        )
    )
    cert_in = StepCertInput(
        step=step_idx,
        question=fmt_out.question,
        solution=fmt_out.solution,
        answer=fmt_out.answer,
        question_type=question_type,
        memory_json=KnownTree.to_json(KnownTree.build_step_cert_view(state.memory, step_idx)),
    )
    cert_out = cert_runner.run_one(
        cert_in,
        snapshot_dir=step_dir / "subruns_raw" / "step_cert_builder",
        unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
    )
    cert_dict = step_cert_output_to_dict(cert_out)

    raw_ref = None
    try:
        raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/step_cert_builder/"
    except Exception:
        raw_ref = str(step_dir / "subruns_raw" / "step_cert_builder")
    provenance = {"role": "step_cert_builder", "raw_ref": raw_ref}

    state.memory = KnownTree.apply_step_update(
        state.memory,
        step=step_idx,
        premise_delta=cert_out.premise_delta,
        fact_delta=cert_out.fact_delta,
        step_cert=cert_out.step_cert,
        key_fact_id=cert_out.key_fact_id,
        overwrite_step=False,
        provenance=provenance,
    )

    # Persist Numeric oracle signals into step_certs (edge-only).
    if numeric_oracle_out is not None and numeric_oracle_out.gt_value is not None:
        raw_ref = None
        try:
            raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/numeric_oracle/"
        except Exception:
            raw_ref = str(step_dir / "subruns_raw" / "numeric_oracle")
        mem = KnownTree.normalize_memory(state.memory)
        step_certs = mem.get("step_certs")
        if not isinstance(step_certs, list):
            step_certs = []
        step_certs.append(
            {
                "kind": "numeric_oracle_cert",
                "step": int(step_idx),
                "abs_tol": numeric_oracle_out.abs_tol,
                "rel_tol": numeric_oracle_out.rel_tol,
                "sig_figs": numeric_oracle_out.sig_figs,
                "unit": numeric_oracle_out.unit,
                "extra_internal": {
                    "oracle_code": numeric_oracle_out.oracle_code,
                    "gt_value": numeric_oracle_out.gt_value,
                    "exec_payload": numeric_oracle_out.exec_payload,
                    "notes": numeric_oracle_out.notes,
                },
                "provenance": {"role": "numeric_oracle", "raw_ref": raw_ref},
            }
        )
        mem["step_certs"] = step_certs
        state.memory = mem

    # Persist world_contract into memory when DraftChain provides it (extend-side governance).
    # Note: default DraftChain prompt omits world_contract; it is present only when the world_contract
    # prompt variant is used.
    try:
        wc = (draft_chain_dict or {}).get("world_contract")
    except Exception:
        wc = None
    if isinstance(wc, dict):
        raw_ref = None
        try:
            raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/draft_chain/"
        except Exception:
            raw_ref = str(step_dir / "subruns_raw" / "draft_chain")
        mem = KnownTree.normalize_memory(state.memory)
        mem["world_contract"] = merge_world_contract(
            mem.get("world_contract"),
            wc,
            role="extend_world_contract",
            step=int(step_idx),
            round=int(round_idx),
            raw_ref=raw_ref,
        )
        state.memory = mem

    # Persist Type2 answer contracts (ACB-lite): internal-only governance for judging/output protocol.
    builder_result: Dict[str, Any] = {
        "status": "not_run",
        "answer_style": {},
        "answer_semantics": {},
        "support_witness": [],
    }
    try:
        raw_ref = None
        try:
            raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/format/"
        except Exception:
            raw_ref = str(step_dir / "subruns_raw" / "format")
        if str(fmt_out.world_contract or "").strip():
            answer_contract_question = compose_solver_question(fmt_out.question, fmt_out.world_contract)
        else:
            answer_contract_question = raw_question_for_contracts
        if not answer_contract_question.strip():
            answer_contract_question = str(fmt_out.question or "").strip()
        builder_result = _run_answer_contract_builder(
            state=state,
            op_name="extend",
            step_idx=int(step_idx),
            step_dir=step_dir,
            op_conf=op_conf,
            generator=generator,
            agent_lang=agent_lang,
            question_type=question_type,
            question=answer_contract_question,
            world_contract_text=None,
            answer=str(fmt_out.answer or ""),
        )
        if str(builder_result.get("status") or "").strip().lower() == "ok":
            try:
                raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/answer_contract_builder/"
            except Exception:
                raw_ref = str(step_dir / "subruns_raw" / "answer_contract_builder")
        answer_contract_payload = {
            "answer_style": builder_result.get("answer_style"),
            "answer_semantics": builder_result.get("answer_semantics"),
            "support_witness": builder_result.get("support_witness"),
        }
        mem = KnownTree.normalize_memory(state.memory)
        mem = _persist_type2_answer_contracts(
            mem=mem,
            step_idx=int(step_idx),
            question_type=question_type,
            question=answer_contract_question,
            answer=str(fmt_out.answer or ""),
            where="extend_format",
            raw_ref=raw_ref,
            numeric_oracle_out=numeric_oracle_out,
            answer_contract_payload=answer_contract_payload,
            report_path=step_dir / "answer_contract_report.json",
        )
        state.memory = mem
    except Exception as exc:  # noqa: BLE001
        # Fail-fast on internal contract persistence bugs: this should never silently degrade.
        logger.error("persist Type2 answer contracts failed: %s", str(exc))
        raise

    world_contract_text = extract_solver_world_contract_text(
        state.memory,
        step=step_idx,
        lang=agent_lang,
        explicit_world_contract_text=fmt_out.world_contract,
    )

    known_view = KnownTree.build_edge_solver_view(state.memory, step_idx)
    known_text = KnownTree.to_json(known_view)
    qtype_norm = normalize_question_type(question_type)
    allowed_qtypes_for_step = allowed_question_types_for_step(agent_conf, step_idx)
    symbolic_only_semantics = "off"
    if symbolic_only and qtype_norm == "Derivation":
        symbolic_only_semantics = "derivation_symbolic_only"
    elif symbolic_only and qtype_norm == "MCQ":
        symbolic_only_semantics = "mcq_no_numeric_eval"
    state.append_history(
        KQARecord(
            paper_id=state.paper_id or "",
            step=step_idx,
            known=known_text,
            question=fmt_out.question,
            world_contract_text=world_contract_text,
            answer=fmt_out.answer,
            chain=f"k{step_idx},q{step_idx},a{step_idx}",
            subject=state.subject,
            question_type=qtype_norm,
            question_type_constraints={
                "locked_question_type": qtype_norm,
                "allowed_question_types_for_step": allowed_qtypes_for_step,
                "symbolic_only_semantics": symbolic_only_semantics,
            },
        )
    )

    # Path-Fold: generate a folded path question for evaluation (two variants).
    try:
        fold_prompt = PATH_FOLD_V1_EN if use_en else PATH_FOLD_V1
        if symbolic_only:
            fold_prompt = _append_symbolic_constraints(fold_prompt, use_en, question_type)
        fold_generator = op_conf.get("path_fold_generator") or op_conf.get("struct_generator") or generator
        fold_generator = with_idealab_session_id(
            fold_generator,
            idealab_session_id_for_step_node(state, "extend", step_idx),
        )
        path_view = KnownTree.build_path_solver_view(state.memory, step_idx)
        path_view = KnownTree.compact_kqa_known_view(path_view)
        premise_bank_json = json.dumps(path_view.get("premise_bank", []), ensure_ascii=False)
        history_payload = _build_path_fold_history_payload(state.history)
        history_json = json.dumps(history_payload, ensure_ascii=False)

        fold_runner = PathFoldRunner(
            PathFoldConfig(
                generator=fold_generator,
                prompt_path=Path(op_conf.get("path_fold_prompt_path") or "src/agenqa/prompts/path_fold.prompt"),
                prompt_text=fold_prompt,
                lang=agent_lang,
            )
        )
        fold_in = PathFoldInput(
            step=step_idx,
            question_type=str(question_type or ""),
            premise_bank_json=premise_bank_json,
            history_json=history_json,
        )
        fold_out = fold_runner.run_one(
            fold_in,
            snapshot_dir=step_dir / "subruns_raw" / "path_fold",
            unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
        )
        fold_out.question_scaffolded = _strip_solver_contract_from_question(fold_out.question_scaffolded)
        fold_out.question_direct = _strip_solver_contract_from_question(fold_out.question_direct)
        try:
            tail = state.history[-1]
            track_out = str((agent_conf.get("agent") or {}).get("track") or "").strip().lower() or "unified"
            if track_out not in {"unified", "semantic"}:
                track_out = "unified"
            if isinstance(fold_out.question_scaffolded, str) and fold_out.question_scaffolded.strip():
                tail.path_question_scaffolded = dumps_folded_question(
                    track=track_out,
                    variant="scaffolded",
                    payload={FIELD_QUESTION_TEXT: fold_out.question_scaffolded.strip()},
                )
            else:
                tail.path_question_scaffolded = None
            if isinstance(fold_out.question_direct, str) and fold_out.question_direct.strip():
                tail.path_question_direct = dumps_folded_question(
                    track=track_out,
                    variant="direct",
                    payload={FIELD_QUESTION_TEXT: fold_out.question_direct.strip()},
                )
            else:
                tail.path_question_direct = None
            tail.path_fold_notes = fold_out.fold_notes
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("PathFold failed (extend step=%s): %s", str(step_idx), str(exc))
        fold_out = None

    save_state(state)

    role_outputs = ext_ctx.get("role_outputs") or {}
    role_outputs["format"] = fmt_dict
    if isinstance(builder_result, dict):
        role_outputs["answer_contract_builder"] = builder_result
    if numeric_oracle_out is not None:
        role_outputs["numeric_oracle"] = {
            "abs_tol": numeric_oracle_out.abs_tol,
            "rel_tol": numeric_oracle_out.rel_tol,
            "sig_figs": numeric_oracle_out.sig_figs,
            "unit": numeric_oracle_out.unit,
            "gt_value": numeric_oracle_out.gt_value,
            "notes": numeric_oracle_out.notes,
        }
    role_outputs["world_contract_text"] = world_contract_text
    role_outputs["step_cert_builder"] = cert_dict
    if fold_out is not None:
        role_outputs["path_fold"] = path_fold_output_to_dict(fold_out)
    if format_validation is not None:
        role_outputs["format_validation"] = format_validation
    return state, role_outputs, step_dir, step_idx, round_idx


def revise_diagnose_node(agent_conf: Dict[str, Any], state: AgentState) -> tuple[AgentState, Dict[str, Any]]:
    """Revise 第一步：Diagnose 当前最后一题。"""
    if not state.history:
        logger.warning("revise_diagnose_node 在 history 为空时被调用，跳过。")
        return state, {}

    ops_conf = (agent_conf.get("operators") or {})
    op_conf = (ops_conf.get("revise") or ops_conf.get("extend") or {}) or {}
    generator = op_conf.get("generator") or {
        "service_type": "private_endpoint",
        "service_id": op_conf.get("service_id"),
    }
    agent_block = agent_conf.get("agent") or {}
    agent_lang = str(agent_block.get("lang") or "").lower() or None
    use_en = agent_lang in {"en", "english"}
    draft_chain_window = _get_int_conf(agent_block, "draft_chain_window", 2)
    diagnose_window = _get_int_conf(agent_block, "diagnose_window", draft_chain_window)

    try:
        step_idx = int(state.step)
    except Exception:
        step_idx = max(len(state.history) - 1, 0)

    round_idx = state.current_round_index()
    # 先推断 revise_mode，用于目录命名
    from agenqa.nodes.op_revise import _infer_revise_mode
    revise_mode = normalize_revise_mode(_infer_revise_mode(state)) or REVISE_MODE_REUSE_HIDDEN
    # 会话亲和：revise(step_idx) 内的所有子调用共享一个 session id（Diagnose/Draft/Format/...）。
    revise_session_id = idealab_session_id_for_step_node(state, "revise", step_idx)
    generator = with_idealab_session_id(generator, revise_session_id)
    step_dir = compute_step_dir(state.artifacts_dir, "revise", step_idx, round_idx, revise_mode=revise_mode)
    step_dir.mkdir(parents=True, exist_ok=True)
    dump_director_decision_for_step(state, step_dir, step_idx)

    target = state.history[-1]
    known_view = KnownTree.build_diagnose_view(
        state.memory,
        step_idx,
        window=diagnose_window,
        include_current_step=True,
    )
    known_0_for_roles = KnownTree.to_json(known_view)

    # 轻量 solver feedback（延用现有逻辑）
    from agenqa.nodes.op_revise import (
        _build_solver_feedback_text,
        _build_solver_answers_text,
        _build_solver_reasoning_text,
    )

    solver_fb = _build_solver_feedback_text(state)
    solver_answers = _build_solver_answers_text(state)
    solver_reasoning = _build_solver_reasoning_text(state)
    # Revise needs full solver_feedback to diagnose and fix the current question.
    director_notes = build_director_notes(state, include_solver_feedback=True)
    roles_protocol = str(
        agent_block.get("roles_protocol") or agent_block.get("draft_protocol") or ""
    ).strip().lower() or None

    # 根据 revise_mode 选择对应的 Diagnose prompt
    if revise_mode == REVISE_MODE_CORRECTNESS:
        if use_en:
            diag_prompt_text = (
                DIAGNOSE_REVISE_CORRECTNESS_TAGGED_EN
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_CORRECTNESS_EN
            )
        else:
            diag_prompt_text = (
                DIAGNOSE_REVISE_CORRECTNESS_TAGGED
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_CORRECTNESS
            )
    elif revise_mode == REVISE_MODE_WORLD_CONTRACT:
        if use_en:
            diag_prompt_text = (
                DIAGNOSE_REVISE_WORLD_CONTRACT_TAGGED_EN
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_WORLD_CONTRACT_EN
            )
        else:
            diag_prompt_text = (
                DIAGNOSE_REVISE_WORLD_CONTRACT_TAGGED
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_WORLD_CONTRACT
            )
    elif revise_mode == REVISE_MODE_ANSWER_CONTRACT:
        if use_en:
            diag_prompt_text = (
                DIAGNOSE_REVISE_ANSWER_CONTRACT_TAGGED_EN
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_ANSWER_CONTRACT_EN
            )
        else:
            diag_prompt_text = (
                DIAGNOSE_REVISE_ANSWER_CONTRACT_TAGGED
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_ANSWER_CONTRACT
            )
    elif revise_mode == REVISE_MODE_QUALITY:
        if use_en:
            diag_prompt_text = (
                DIAGNOSE_REVISE_DIFFICULTY_TAGGED_EN
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_DIFFICULTY_EN
            )
        else:
            diag_prompt_text = (
                DIAGNOSE_REVISE_DIFFICULTY_TAGGED
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_DIFFICULTY
            )
    elif revise_mode == REVISE_MODE_REUSE_HIDDEN:
        if use_en:
            diag_prompt_text = (
                DIAGNOSE_REVISE_REUSE_HIDDEN_TAGGED_EN
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_REUSE_HIDDEN_EN
            )
        else:
            diag_prompt_text = (
                DIAGNOSE_REVISE_REUSE_HIDDEN_TAGGED
                if roles_protocol == "tagged"
                else DIAGNOSE_REVISE_REUSE_HIDDEN
            )
    else:
        # 回退到通用版本（理论上不应该发生）
        if use_en:
            diag_prompt_text = DIAGNOSE_V1_TAGGED_EN if roles_protocol == "tagged" else DIAGNOSE_V1_EN
        else:
            diag_prompt_text = DIAGNOSE_V1_TAGGED if roles_protocol == "tagged" else DIAGNOSE_V1

    diag_generator = op_conf.get("diagnose_generator") or op_conf.get("struct_generator") or generator
    diag_generator = with_idealab_session_id(diag_generator, revise_session_id)
    diag_runner = DiagnoseRunner(
        DiagnoseConfig(
            generator=diag_generator,
            prompt_path=Path(op_conf.get("diagnose_prompt_path") or "src/agenqa/prompts/diagnose.prompt"),
            prompt_text=diag_prompt_text,
            lang=agent_lang,
            protocol=roles_protocol,
        )
    )
    diag_in = DiagnoseInput(
        known_0=known_0_for_roles or str(target.known),
        question=target.question,
        answer=target.answer,
        solver_feedback=solver_fb,
        director_notes=director_notes,
        solver_answers=solver_answers,
        solver_reasoning=solver_reasoning,
        background=(
            build_answer_contract_validation_background(
                KnownTree.normalize_memory(state.memory),
                step=step_idx,
                lang=agent_lang,
            )
            if revise_mode == REVISE_MODE_ANSWER_CONTRACT
            else None
        ),
    )
    # revise_mode 已在上面推断，用于 prompt snapshot 命名和后续节点传递
    # 方案 B：在 diagnose prompt snapshot 的 name_prefix 中包含 revise_mode
    diagnose_name_prefix = f"prompt_used.diagnose_revise.{revise_mode}.diagnose."

    diag_out = diag_runner.run_one(
        diag_in,
        snapshot_dir=step_dir / "subruns_raw" / "diagnose",
        unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
        name_prefix=diagnose_name_prefix,
    )
    if not diag_out:
        logger.error("Revise Diagnose 未产出有效结果。")
        raise RuntimeError("Revise Diagnose failed")

    role_outputs = {
        "diagnose": {
            "diagnosis": diag_out.diagnosis,
            "issues": diag_out.issues,
            "fix_suggestions": diag_out.fix_suggestions,
        }
    }
    diag_summary = diag_out.diagnosis
    if diag_out.issues:
        diag_summary = f"Issues: {', '.join(diag_out.issues)} | {diag_summary}"

    # 不再构建 history_brief，revise draft/format 直接使用 Known JSON

    rev_ctx: Dict[str, Any] = {
        "step_idx": step_idx,
        "round_idx": round_idx,
        "step_dir": step_dir,
        "target": target,
        "known_0_for_roles": known_0_for_roles or str(target.known),
        "diag_summary": diag_summary,
        "generator": generator,
        "agent_lang": agent_lang,
        "director_notes": director_notes,
        "op_conf": op_conf,
        "role_outputs": role_outputs,
        "revise_mode": revise_mode,  # 传递 revise_mode 到后续节点
        "revise_session_id": revise_session_id,
    }
    return state, rev_ctx


def revise_draft_node(
    agent_conf: Dict[str, Any],
    state: AgentState,
    rev_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """Revise 第二步：生成 DraftChain 草稿。"""
    rev_ctx = deepcopy(rev_ctx or {})
    if not rev_ctx:
        logger.error("revise_draft_node 缺少 revise 上下文。")
        raise RuntimeError("Revise draft 缺少上下文")

    step_idx = int(rev_ctx.get("step_idx") or state.step or 0)
    diag_summary = rev_ctx.get("diag_summary")
    target = rev_ctx.get("target")
    generator = rev_ctx.get("generator") or {}
    agent_lang = rev_ctx.get("agent_lang")
    op_conf = rev_ctx.get("op_conf") or {}
    agent_block = agent_conf.get("agent") or {}
    roles_protocol = str(
        agent_block.get("roles_protocol") or agent_block.get("draft_protocol") or ""
    ).strip().lower() or None
    use_en = str(agent_lang or "").lower() in {"en", "english"}
    draft_chain_window = _get_int_conf(agent_block, "draft_chain_window", 2)

    # 题型推断延用现有逻辑
    from agenqa.nodes.op_revise import _infer_question_type

    question_type = _infer_question_type(agent_conf, state, step_idx)
    symbolic_only = is_symbolic_only_for_question_type(agent_conf, question_type)

    director_notes = rev_ctx.get("director_notes") or ""
    if diag_summary:
        director_notes = f"{director_notes}\n\n[Diagnose]\n{diag_summary}".strip()
    if target and (getattr(target, "question", None) or getattr(target, "answer", None)):
        cur_q = getattr(target, "question", "") or ""
        cur_a = getattr(target, "answer", "") or ""
        if use_en:
            director_notes = (
                f"{director_notes}\n\n"
                "[Current QA to revise]\n"
                f"Question:\n{cur_q}\n\n"
                f"Answer:\n{cur_a}\n"
            ).strip()
        else:
            director_notes = (
                f"{director_notes}\n\n"
                "【当前待修订 QA】\n"
                f"Question:\n{cur_q}\n\n"
                f"Answer:\n{cur_a}\n"
            ).strip()

    # World-contract revise uses a dedicated DraftChain prompt variant (minimal-context design).
    try:
        revise_mode = str(rev_ctx.get("revise_mode") or "").strip().lower()
    except Exception:
        revise_mode = ""
    if normalize_revise_mode(revise_mode) == REVISE_MODE_WORLD_CONTRACT:
        guard = (
            "【约束提醒】本次为 world_contract 修订：优先把语义世界观钉死（L1-first：先选范式 L1，再补齐必要的 L3 规则）。"
            "不要为了对齐旧 GT 盲目改推导；仅当“语义钉死后旧 Answer/GT 不再成立”时，才允许同步更新 Answer/GT，并在 Diagnose/Draft 中明确说明原因。"
        )
        director_notes = f"{director_notes}\n\n{guard}".strip()
    elif normalize_revise_mode(revise_mode) == REVISE_MODE_ANSWER_CONTRACT:
        guard = (
            "【约束提醒】本次为 answer_contract 修订（Type2 输出协议/判对口径）：优先澄清题面与答案的作答要求，"
            "例如 exact/approx、容差/单位、分支/等价类、唯一答案约束等。"
            "尽量不要改变题目的语义世界观（Type1/L1-L3）；除非 Diagnose 明确指出“现有题面口径与 GT 不一致/不可判”，"
            "否则不要重写推导逻辑。"
        )
        director_notes = f"{director_notes}\n\n{guard}".strip()

    if symbolic_only:
        qtype = normalize_question_type(question_type) or str(question_type or "").strip()
        if qtype == "MCQ":
            guard = (
                "【约束提醒】本步为 MCQ（禁数值求值口径）：题干可包含数值，但不得要求数值计算/小数近似/误差口径；"
                "Answer 必须仅输出选项字母 \\boxed{A/B/C/D}。若 director_notes 中包含数值求值/abs_tol/保留几位等建议，请忽略。"
            )
        else:
            guard = (
                "【约束提醒】本步为符号表达式 ONLY：不得在 Question/Answer 中要求或给出数值求值/小数近似/误差口径。"
                "若 director_notes 中包含此类数值计算建议，请忽略并改为符号推导任务。"
            )
        director_notes = guard if not director_notes else f"{guard} | {director_notes}"

    chain_view = KnownTree.build_draft_chain_view(state.memory, step_idx, window=draft_chain_window)
    chain_view_json = KnownTree.to_json(chain_view)
    expected_primary = KnownTree.key_fact_id_for_step(state.memory, step_idx - 1) if step_idx >= 2 else ""

    use_calc = _is_calc_prompt_marker(op_conf.get("prompt_path")) or _is_calc_prompt_marker(
        op_conf.get("draft_chain_prompt_path")
    )
    prompt_text = get_draft_chain_prompt(
        question_type=question_type,
        use_en=use_en,
        calc=use_calc,
        world_contract=(normalize_revise_mode(revise_mode) == REVISE_MODE_WORLD_CONTRACT),
    )
    if symbolic_only:
        prompt_text = _append_symbolic_constraints(prompt_text, use_en, question_type)
    qtype_key = str(question_type or "").strip().lower() or "derivation"
    default_prompt_path = f"src/agenqa/prompts/draft_chain_{qtype_key}{'_calc' if use_calc else ''}.prompt"
    draft_runner = DraftChainRunner(
        DraftChainConfig(
            generator=generator,
            prompt_path=Path(op_conf.get("draft_chain_prompt_path") or default_prompt_path),
            prompt_text=prompt_text,
            lang=agent_lang,
        )
    )
    draft_in = DraftChainInput(
        chain_view_json=chain_view_json,
        prev_step=step_idx - 1,
        step=step_idx,
        director_notes=director_notes,
        question_type=question_type,
        expected_primary_fact_id=expected_primary,
    )
    draft_out = draft_runner.run_one(
        draft_in,
        snapshot_dir=Path(rev_ctx.get("step_dir")) / "subruns_raw" / "draft_chain",
        unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
    )
    if not draft_out:
        logger.error("Revise DraftChain 未产出有效结果。")
        raise RuntimeError("Revise DraftChain failed")
    contract_violations: list[str] = []
    if step_idx >= 2:
        required = [str(x) for x in (draft_out.required_fact_ids or []) if str(x).strip()]
        primary = str(draft_out.primary_required_fact_id or "")
        if not expected_primary:
            contract_violations.append(
                "missing expected_primary_fact_id from previous step (memory mismatch; skipping strict enforcement)"
            )
            if primary and primary.strip() and primary not in required:
                contract_violations.append(f"required_fact_ids missing primary_required_fact_id: {primary!r}")
        else:
            if (primary or "").strip() != expected_primary:
                contract_violations.append(
                    f"primary_required_fact_id mismatch: {draft_out.primary_required_fact_id!r} vs expected={expected_primary!r}"
                )
            if expected_primary not in required:
                contract_violations.append(f"required_fact_ids missing expected_primary_fact_id: {expected_primary!r}")
    else:
        if draft_out.required_fact_ids or draft_out.primary_required_fact_id:
            contract_violations.append("step=1 must not require prior facts")

    draft_dict = draft_chain_output_to_dict(draft_out)
    if contract_violations:
        logger.warning("Revise DraftChain contract violations (observed; no auto-fix): %s", "; ".join(contract_violations))
    role_outputs = rev_ctx.get("role_outputs") or {}
    role_outputs["draft_chain"] = draft_dict
    role_outputs["draft_chain_contract"] = {
        "step": step_idx,
        "expected_primary_fact_id": expected_primary,
        "observed_primary_required_fact_id": draft_out.primary_required_fact_id,
        "observed_required_fact_ids": list(draft_out.required_fact_ids or []),
        "violations": contract_violations,
    }
    rev_ctx["role_outputs"] = role_outputs
    rev_ctx["draft_chain_dict"] = draft_dict
    rev_ctx["chain_view_json"] = chain_view_json
    rev_ctx["question_type"] = question_type
    return rev_ctx


def revise_format_node(
    agent_conf: Dict[str, Any],
    state: AgentState,
    rev_ctx: Dict[str, Any],
) -> tuple[AgentState, Dict[str, Any], Path, int, int]:
    """Revise 第三步：Format + StepCertBuilder，并覆盖 last history。"""
    rev_ctx = deepcopy(rev_ctx or {})
    if not rev_ctx:
        logger.error("revise_format_node 缺少 revise 上下文。")
        raise RuntimeError("Revise format 缺少上下文")

    step_idx = int(rev_ctx.get("step_idx") or state.step or 0)
    step_dir: Path = rev_ctx.get("step_dir")
    round_idx = int(rev_ctx.get("round_idx") or state.current_round_index())
    generator = rev_ctx.get("generator") or {}
    revise_mode = str(rev_ctx.get("revise_mode") or "").strip().lower()
    agent_lang = rev_ctx.get("agent_lang")
    op_conf = rev_ctx.get("op_conf") or {}
    question_type = rev_ctx.get("question_type")
    agent_block = agent_conf.get("agent") or {}
    use_en = str(agent_lang or "").lower() in {"en", "english"}
    symbolic_only = is_symbolic_only_for_question_type(agent_conf, question_type)
    roles_protocol = str(
        agent_block.get("roles_protocol") or agent_block.get("draft_protocol") or ""
    ).strip().lower() or None

    draft_chain_dict = rev_ctx.get("draft_chain_dict") or {}
    revise_session_id = str(rev_ctx.get("revise_session_id") or "").strip() or idealab_session_id_for_step_node(
        state, "revise", step_idx
    )

    # Format
    format_prompt = FORMAT_V1_TAGGED_EN if use_en and roles_protocol == "tagged" else (
        FORMAT_V1_TAGGED if (roles_protocol == "tagged") else (FORMAT_V1_EN if use_en else FORMAT_V1)
    )
    if symbolic_only:
        format_prompt = _append_symbolic_constraints(format_prompt, use_en, question_type)
    draft_pack = dict(draft_chain_dict)
    draft_pack[FIELD_DRAFT_QUESTION] = str(draft_chain_dict.get(FIELD_DRAFT_QUESTION_EXPLICIT) or "")
    if FIELD_DRAFT_QUESTION_EXPLICIT in draft_pack:
        del draft_pack[FIELD_DRAFT_QUESTION_EXPLICIT]
    format_generator = (
        op_conf.get("format_generator")
        or op_conf.get("struct_generator")
        or (op_conf.get("format") or {}).get("generator")
        or generator
    )
    format_generator = with_idealab_session_id(
        format_generator,
        revise_session_id,
    )
    fmt_runner = FormatRunner(
        FormatConfig(
            generator=format_generator,
            prompt_path=Path(op_conf.get("format_prompt_path") or "src/agenqa/prompts/format.prompt"),
            prompt_text=format_prompt,
            lang=agent_lang,
            protocol=roles_protocol,
        )
    )
    fmt_in = FormatInput(
        draft_json=json.dumps(draft_pack, ensure_ascii=False),
        prev_step=step_idx - 1,
        step=step_idx,
        question_type=question_type,
    )
    fmt_out = fmt_runner.run_one(
        fmt_in,
        snapshot_dir=step_dir / "subruns_raw" / "format",
        unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
    )
    if not fmt_out:
        logger.error("Revise Format 未产出有效结果。")
        raise RuntimeError("Revise Format failed")

    # Numeric oracle (tool-assisted): recompute GT value + tolerance, then override Answer deterministically.
    numeric_oracle_out = None
    if str(question_type or "").strip() == "Numeric":
        oracle_block = agent_conf.get("numeric_oracle") or {}
        oracle_source = "unknown"
        draft_oracle_code = ""
        try:
            draft_oracle_code = str(draft_chain_dict.get(FIELD_ORACLE_CODE) or "").strip()
        except Exception:
            draft_oracle_code = ""

        if draft_oracle_code:
            snap_dir = step_dir / "subruns_raw" / "numeric_oracle"
            try:
                snap_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                (snap_dir / "input_view.json").write_text(
                    json.dumps(
                        {
                            "source": "draft_chain",
                            "mode": "revise",
                            "step": step_idx,
                            "question_type": str(question_type or ""),
                            "timeout_seconds": float(oracle_block.get("timeout_seconds", 10.0)),
                            "memory_limit_mb": int(oracle_block.get("memory_limit_mb", 4096)),
                            "temp_dir": str(oracle_block.get("temp_dir") or str(Path("/tmp").resolve())),
                            "python_bin": str(oracle_block.get("python_bin") or sys.executable),
                            "question": fmt_out.question,
                            "solution": fmt_out.solution,
                            FIELD_ABS_TOL: draft_chain_dict.get(FIELD_ABS_TOL),
                            FIELD_REL_TOL: draft_chain_dict.get(FIELD_REL_TOL),
                            FIELD_SIG_FIGS: draft_chain_dict.get(FIELD_SIG_FIGS),
                            FIELD_UNIT: draft_chain_dict.get(FIELD_UNIT),
                            FIELD_NOTES: draft_chain_dict.get(FIELD_NOTES),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (snap_dir / "oracle_code.py").write_text(draft_oracle_code, encoding="utf-8")
            except Exception:
                pass

            try:
                gt, exec_payload = execute_oracle_code(
                    draft_oracle_code,
                    timeout_seconds=float(oracle_block.get("timeout_seconds", 10.0)),
                    memory_limit_mb=int(oracle_block.get("memory_limit_mb", 4096)),
                    temp_dir=str(oracle_block.get("temp_dir") or str(Path("/tmp").resolve())),
                    python_bin=str(oracle_block.get("python_bin") or sys.executable),
                )
                numeric_oracle_out = NumericOracleOutput(
                    abs_tol=draft_chain_dict.get(FIELD_ABS_TOL),
                    rel_tol=draft_chain_dict.get(FIELD_REL_TOL),
                    sig_figs=draft_chain_dict.get(FIELD_SIG_FIGS),
                    unit=str(draft_chain_dict.get(FIELD_UNIT) or "").strip(),
                    oracle_code=draft_oracle_code,
                    gt_value=gt,
                    exec_payload=exec_payload,
                    notes=str(draft_chain_dict.get(FIELD_NOTES) or "").strip(),
                )
                try:
                    (snap_dir / "executor_result.json").write_text(
                        json.dumps(exec_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                if numeric_oracle_out.gt_value is None:
                    raise RuntimeError("draft_chain oracle_code executed but gt_value is None")
                oracle_source = "draft_chain_oracle_code_exec"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DraftChain oracle_code execution failed (revise), falling back to NumericOracle role: %s",
                    str(exc),
                )
                numeric_oracle_out = None

        if numeric_oracle_out is None:
            oracle_gen = (
                op_conf.get("numeric_oracle_generator")
                or op_conf.get("oracle_generator")
                or op_conf.get("struct_generator")
                or generator
            )
            oracle_gen = with_idealab_session_id(oracle_gen, revise_session_id)
            oracle_runner = NumericOracleRunner(
                NumericOracleConfig(
                    generator=oracle_gen,
                    prompt_path=Path(op_conf.get("numeric_oracle_prompt_path") or "src/agenqa/prompts/numeric_oracle.py"),
                    lang=agent_lang,
                    timeout_seconds=float(oracle_block.get("timeout_seconds", 10.0)),
                    memory_limit_mb=int(oracle_block.get("memory_limit_mb", 4096)),
                    temp_dir=str(oracle_block.get("temp_dir") or str(Path("/tmp").resolve())),
                    python_bin=str(oracle_block.get("python_bin") or sys.executable),
                )
            )
            numeric_oracle_out = oracle_runner.run_one(
                NumericOracleInput(step=step_idx, question=fmt_out.question, solution=fmt_out.solution),
                snapshot_dir=step_dir / "subruns_raw" / "numeric_oracle",
                unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
            )
            if numeric_oracle_out.gt_value is None:
                raise RuntimeError("NumericOracle failed to produce gt_value")
            oracle_source = "numeric_oracle_role"
        fmt_out.answer = format_numeric_answer(numeric_oracle_out.gt_value, sig_figs=numeric_oracle_out.sig_figs)
        tolerance_sentence = build_numeric_answer_format_sentence(
            abs_tol=numeric_oracle_out.abs_tol,
            rel_tol=numeric_oracle_out.rel_tol,
            sig_figs=numeric_oracle_out.sig_figs,
            lang=agent_lang,
        )
        _record_numeric_error_chain(
            state=state,
            step_idx=step_idx,
            round_idx=round_idx,
            op_name="revise",
            step_dir=step_dir,
            raw_value=float(numeric_oracle_out.gt_value),
            shown_value=str(fmt_out.answer or ""),
            unit=str(numeric_oracle_out.unit or ""),
            rounding_rule={
                "sig_figs": numeric_oracle_out.sig_figs,
                "abs_tol": numeric_oracle_out.abs_tol,
                "rel_tol": numeric_oracle_out.rel_tol,
                "tolerance_sentence": tolerance_sentence,
            },
            source=oracle_source,
        )

    raw_question_for_contracts = str(fmt_out.question or "")
    fmt_out.world_contract = str(getattr(fmt_out, "world_contract", "") or "").strip()
    fmt_out.question = _strip_solver_contract_from_question(raw_question_for_contracts)
    fmt_dict = format_output_to_dict(fmt_out)

    format_validation = None
    if getattr(fmt_out, "validation_passed", True) is False:
        format_validation = {
            "mode": "revise",
            "attempt_step": int(getattr(fmt_out, "step", step_idx) or step_idx),
            "validation_passed": False,
            "validation_errors": list(getattr(fmt_out, "validation_errors", None) or []),
        }

    # StepCertBuilder (overwrite memory for this step)
    cert_prompt = STEP_CERT_BUILDER_V1_EN if use_en else STEP_CERT_BUILDER_V1
    cert_generator = op_conf.get("step_cert_generator") or op_conf.get("struct_generator") or generator
    cert_generator = with_idealab_session_id(
        cert_generator,
        revise_session_id,
    )
    cert_runner = StepCertBuilderRunner(
        StepCertConfig(
            generator=cert_generator,
            prompt_path=Path(op_conf.get("step_cert_prompt_path") or "src/agenqa/prompts/step_cert_builder.prompt"),
            prompt_text=cert_prompt,
            lang=agent_lang,
        )
    )
    cert_in = StepCertInput(
        step=step_idx,
        question=fmt_out.question,
        solution=fmt_out.solution,
        answer=fmt_out.answer,
        question_type=question_type,
        memory_json=KnownTree.to_json(KnownTree.build_step_cert_view(state.memory, step_idx)),
    )
    cert_out = cert_runner.run_one(
        cert_in,
        snapshot_dir=step_dir / "subruns_raw" / "step_cert_builder",
        unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
    )
    cert_dict = step_cert_output_to_dict(cert_out)

    raw_ref = None
    try:
        raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/step_cert_builder/"
    except Exception:
        raw_ref = str(step_dir / "subruns_raw" / "step_cert_builder")
    provenance = {"role": "step_cert_builder", "raw_ref": raw_ref}

    state.memory = KnownTree.apply_step_update(
        state.memory,
        step=step_idx,
        premise_delta=cert_out.premise_delta,
        fact_delta=cert_out.fact_delta,
        step_cert=cert_out.step_cert,
        key_fact_id=cert_out.key_fact_id,
        overwrite_step=True,
        provenance=provenance,
    )

    # Persist Type1 world_contract into memory when world_contract revise is active.
    if revise_mode == REVISE_MODE_WORLD_CONTRACT:
        wc = (rev_ctx.get("draft_chain_dict") or {}).get("world_contract")
        if isinstance(wc, dict):
            raw_ref = None
            try:
                raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/draft_chain/"
            except Exception:
                raw_ref = str(step_dir / "subruns_raw" / "draft_chain")
            mem = KnownTree.normalize_memory(state.memory)
            mem["world_contract"] = merge_world_contract(
                mem.get("world_contract"),
                wc,
                role="revise_world_contract",
                step=int(step_idx),
                round=int(round_idx),
                raw_ref=raw_ref,
            )
            state.memory = mem

    # Persist Numeric oracle signals into step_certs (edge-only).
    if numeric_oracle_out is not None and numeric_oracle_out.gt_value is not None:
        raw_ref = None
        try:
            raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/numeric_oracle/"
        except Exception:
            raw_ref = str(step_dir / "subruns_raw" / "numeric_oracle")
        mem = KnownTree.normalize_memory(state.memory)
        step_certs = mem.get("step_certs")
        if not isinstance(step_certs, list):
            step_certs = []
        step_certs.append(
            {
                "kind": "numeric_oracle_cert",
                "step": int(step_idx),
                "abs_tol": numeric_oracle_out.abs_tol,
                "rel_tol": numeric_oracle_out.rel_tol,
                "sig_figs": numeric_oracle_out.sig_figs,
                "unit": numeric_oracle_out.unit,
                "extra_internal": {
                    "oracle_code": numeric_oracle_out.oracle_code,
                    "gt_value": numeric_oracle_out.gt_value,
                    "exec_payload": numeric_oracle_out.exec_payload,
                    "notes": numeric_oracle_out.notes,
                },
                "provenance": {"role": "numeric_oracle", "raw_ref": raw_ref},
            }
        )
        mem["step_certs"] = step_certs
        state.memory = mem

    # Persist Type2 answer contracts (ACB-lite) for this revised step.
    builder_result: Dict[str, Any] = {
        "status": "not_run",
        "answer_style": {},
        "answer_semantics": {},
        "support_witness": [],
    }
    try:
        raw_ref = None
        try:
            raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/format/"
        except Exception:
            raw_ref = str(step_dir / "subruns_raw" / "format")
        if str(fmt_out.world_contract or "").strip():
            answer_contract_question = compose_solver_question(fmt_out.question, fmt_out.world_contract)
        else:
            answer_contract_question = raw_question_for_contracts
        if not answer_contract_question.strip():
            answer_contract_question = str(fmt_out.question or "").strip()
        builder_result = _run_answer_contract_builder(
            state=state,
            op_name="revise",
            step_idx=int(step_idx),
            step_dir=step_dir,
            op_conf=op_conf,
            generator=generator,
            agent_lang=agent_lang,
            question_type=question_type,
            question=answer_contract_question,
            world_contract_text=None,
            answer=str(fmt_out.answer or ""),
        )
        if str(builder_result.get("status") or "").strip().lower() == "ok":
            try:
                raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/answer_contract_builder/"
            except Exception:
                raw_ref = str(step_dir / "subruns_raw" / "answer_contract_builder")
        answer_contract_payload = {
            "answer_style": builder_result.get("answer_style"),
            "answer_semantics": builder_result.get("answer_semantics"),
            "support_witness": builder_result.get("support_witness"),
        }
        mem = KnownTree.normalize_memory(state.memory)
        mem = _persist_type2_answer_contracts(
            mem=mem,
            step_idx=int(step_idx),
            question_type=question_type,
            question=answer_contract_question,
            answer=str(fmt_out.answer or ""),
            where="revise_format",
            raw_ref=raw_ref,
            numeric_oracle_out=numeric_oracle_out,
            answer_contract_payload=answer_contract_payload,
            report_path=step_dir / "answer_contract_report.json",
        )
        state.memory = mem
    except Exception as exc:  # noqa: BLE001
        logger.error("persist Type2 answer contracts failed (revise): %s", str(exc))
        raise

    world_contract_text = extract_solver_world_contract_text(
        state.memory,
        step=step_idx,
        lang=agent_lang,
        explicit_world_contract_text=fmt_out.world_contract,
    )

    known_view = KnownTree.build_edge_solver_view(state.memory, step_idx)
    known_text = KnownTree.to_json(known_view)
    qtype_norm = normalize_question_type(question_type)
    allowed_qtypes_for_step = allowed_question_types_for_step(agent_conf, step_idx)
    symbolic_only_semantics = "off"
    if symbolic_only and qtype_norm == "Derivation":
        symbolic_only_semantics = "derivation_symbolic_only"
    elif symbolic_only and qtype_norm == "MCQ":
        symbolic_only_semantics = "mcq_no_numeric_eval"
    revised = KQARecord(
        paper_id=state.paper_id or "",
        step=step_idx,
        known=known_text,
        question=fmt_out.question,
        world_contract_text=world_contract_text,
        answer=fmt_out.answer,
        chain=f"k{step_idx},q{step_idx},a{step_idx}",
        subject=state.subject,
        question_type=qtype_norm,
        question_type_constraints={
            "locked_question_type": qtype_norm,
            "allowed_question_types_for_step": allowed_qtypes_for_step,
            "symbolic_only_semantics": symbolic_only_semantics,
        },
    )
    state.history[-1] = revised

    # Path-Fold: regenerate folded path questions after revise.
    try:
        fold_prompt = PATH_FOLD_V1_EN if use_en else PATH_FOLD_V1
        if symbolic_only:
            fold_prompt = _append_symbolic_constraints(fold_prompt, use_en, question_type)
        fold_generator = op_conf.get("path_fold_generator") or op_conf.get("struct_generator") or generator
        fold_generator = with_idealab_session_id(fold_generator, revise_session_id)
        path_view = KnownTree.build_path_solver_view(state.memory, step_idx)
        path_view = KnownTree.compact_kqa_known_view(path_view)
        premise_bank_json = json.dumps(path_view.get("premise_bank", []), ensure_ascii=False)
        history_payload = _build_path_fold_history_payload(state.history)
        history_json = json.dumps(history_payload, ensure_ascii=False)

        fold_runner = PathFoldRunner(
            PathFoldConfig(
                generator=fold_generator,
                prompt_path=Path(op_conf.get("path_fold_prompt_path") or "src/agenqa/prompts/path_fold.prompt"),
                prompt_text=fold_prompt,
                lang=agent_lang,
            )
        )
        fold_in = PathFoldInput(
            step=step_idx,
            question_type=str(question_type or ""),
            premise_bank_json=premise_bank_json,
            history_json=history_json,
        )
        fold_out = fold_runner.run_one(
            fold_in,
            snapshot_dir=step_dir / "subruns_raw" / "path_fold",
            unified_prompt_dir=state.artifacts_dir / "00_Prompts_Snapshot",
        )
        fold_out.question_scaffolded = _strip_solver_contract_from_question(fold_out.question_scaffolded)
        fold_out.question_direct = _strip_solver_contract_from_question(fold_out.question_direct)
        track_out = str((agent_conf.get("agent") or {}).get("track") or "").strip().lower() or "unified"
        if track_out not in {"unified", "semantic"}:
            track_out = "unified"
        if isinstance(fold_out.question_scaffolded, str) and fold_out.question_scaffolded.strip():
            revised.path_question_scaffolded = dumps_folded_question(
                track=track_out,
                variant="scaffolded",
                payload={FIELD_QUESTION_TEXT: fold_out.question_scaffolded.strip()},
            )
        else:
            revised.path_question_scaffolded = None
        if isinstance(fold_out.question_direct, str) and fold_out.question_direct.strip():
            revised.path_question_direct = dumps_folded_question(
                track=track_out,
                variant="direct",
                payload={FIELD_QUESTION_TEXT: fold_out.question_direct.strip()},
            )
        else:
            revised.path_question_direct = None
        revised.path_fold_notes = fold_out.fold_notes
    except Exception as exc:  # noqa: BLE001
        logger.warning("PathFold failed (revise step=%s): %s", str(step_idx), str(exc))
        fold_out = None

    save_state(state)

    role_outputs = rev_ctx.get("role_outputs") or {}
    role_outputs["format"] = fmt_dict
    if isinstance(builder_result, dict):
        role_outputs["answer_contract_builder"] = builder_result
    if numeric_oracle_out is not None:
        role_outputs["numeric_oracle"] = {
            "abs_tol": numeric_oracle_out.abs_tol,
            "rel_tol": numeric_oracle_out.rel_tol,
            "sig_figs": numeric_oracle_out.sig_figs,
            "unit": numeric_oracle_out.unit,
            "gt_value": numeric_oracle_out.gt_value,
            "notes": numeric_oracle_out.notes,
        }
    role_outputs["world_contract_text"] = world_contract_text
    role_outputs["step_cert_builder"] = cert_dict
    if fold_out is not None:
        role_outputs["path_fold"] = path_fold_output_to_dict(fold_out)
    if format_validation is not None:
        role_outputs["format_validation"] = format_validation
    return state, role_outputs, step_dir, step_idx, round_idx
