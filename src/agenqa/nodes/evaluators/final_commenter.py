"""Final Commenter node: comment on the final Path question.

Writes: `00_Summary/final_comment.json`
"""

from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from infra.llm.inference import resolve_inference
from infra.data.io import read_jsonl
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import snapshot_prompt_used, snapshot_rendered_prompt
from agenqa.prompts.final_commenter import FINAL_COMMENTER_TEMPLATE, build_final_commenter_v1_body
from infra.text.json_policy import clean_json_text
from agenqa.graph.state import AgentState
from agenqa.memory.store import dump_edge_kqa_for_step, dump_path_kqa_for_step

logger = logging.getLogger(__name__)


def _solver_status_label(value: Any) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "incorrect"
    return "unknown"


def _solver_key_for_report(tier_name: str, row: Any) -> tuple[str, str, str]:
    tier = str(tier_name or "").strip() or "strong"
    model = ""
    service_id = ""
    if row is not None:
        model = str(getattr(row, "model", None) or "").strip()
        service_id = str(getattr(row, "service_id", None) or "").strip()
    solver_key = service_id or model or tier
    return solver_key, model, service_id


def _well_posed_label(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _build_strong_solver_round_stats(state: AgentState) -> Dict[str, Any]:
    round_solver_rows: Dict[int, Dict[str, Dict[str, Any]]] = {}
    # `solver_index` shape is step -> round -> target -> tier -> SolverResult.
    for step_idx, rounds_dict in sorted((state.solver_index or {}).items()):
        if not isinstance(rounds_dict, dict):
            continue
        for round_idx, targets_dict in sorted(rounds_dict.items()):
            if not isinstance(targets_dict, dict):
                continue
            round_int = int(round_idx)
            round_bucket = round_solver_rows.setdefault(round_int, {})
            for view_name in ("edge", "path"):
                view_dict = targets_dict.get(view_name)
                if not isinstance(view_dict, dict):
                    continue
                for tier_name, result in sorted(view_dict.items()):
                    if not (str(tier_name) == "strong" or str(tier_name).startswith("strong_")):
                        continue
                    solver_key, model, service_id = _solver_key_for_report(str(tier_name), result)
                    row = round_bucket.setdefault(
                        solver_key,
                        {
                            "round": round_int,
                            "step": int(step_idx),
                            "tier": str(tier_name),
                            "model": model,
                            "service_id": service_id,
                            "edge": "unknown",
                            "edge_well_posed": "unknown",
                            "path": "unknown",
                            "path_well_posed": "unknown",
                        },
                    )
                    row["step"] = min(int(row.get("step", step_idx)), int(step_idx))
                    if not row.get("model") and model:
                        row["model"] = model
                    if not row.get("service_id") and service_id:
                        row["service_id"] = service_id
                    if str(row.get("tier") or "").startswith("strong_") and str(tier_name) == "strong":
                        row["tier"] = "strong"
                    row[view_name] = _solver_status_label(getattr(result, "correct", None))
                    row[f"{view_name}_well_posed"] = _well_posed_label(getattr(result, "question_well_posed", None))

    summary_rows: list[Dict[str, Any]] = []
    rounds_json: list[Dict[str, Any]] = []
    lines: list[str] = [
        "# Strong Solver Round Stats",
        "",
        "Per-round edge/path outcomes for every strong solver recorded in `state.solver_index`.",
        "",
        "## Round Summary",
        "",
        "| Round | Step | Tier | Model | Service ID | Edge | Edge Well-Posed | Path | Path Well-Posed | Edge Correct | Edge Incorrect | Edge Unknown | Path Correct | Path Incorrect | Path Unknown |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for round_idx in sorted(round_solver_rows.keys()):
        solver_rows = round_solver_rows[round_idx]
        for solver_key in sorted(solver_rows.keys()):
            row = solver_rows[solver_key]
            edge_status = str(row.get("edge") or "unknown")
            path_status = str(row.get("path") or "unknown")
            edge_well_posed = str(row.get("edge_well_posed") or "unknown")
            path_well_posed = str(row.get("path_well_posed") or "unknown")
            summary_row = {
                "round": round_idx,
                "step": int(row.get("step") or 0),
                "tier": str(row.get("tier") or ""),
                "model": str(row.get("model") or ""),
                "service_id": str(row.get("service_id") or ""),
                "edge": edge_status,
                "edge_well_posed": edge_well_posed,
                "path": path_status,
                "path_well_posed": path_well_posed,
                "edge_correct": 1 if edge_status == "correct" else 0,
                "edge_incorrect": 1 if edge_status == "incorrect" else 0,
                "edge_unknown": 1 if edge_status == "unknown" else 0,
                "path_correct": 1 if path_status == "correct" else 0,
                "path_incorrect": 1 if path_status == "incorrect" else 0,
                "path_unknown": 1 if path_status == "unknown" else 0,
            }
            summary_rows.append(summary_row)
            lines.append(
                "| {round} | {step} | {tier} | {model} | {service_id} | {edge} | {edge_well_posed} | {path} | {path_well_posed} | "
                "{edge_correct} | {edge_incorrect} | {edge_unknown} | {path_correct} | "
                "{path_incorrect} | {path_unknown} |".format(**summary_row)
            )

    if not summary_rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
        lines.extend(["", "_No strong solver results were recorded._"])
    else:
        lines.extend(["", "## Round Details", ""])
        for round_idx in sorted(round_solver_rows.keys()):
            solver_rows = round_solver_rows[round_idx]
            edge_correct_count = sum(1 for row in solver_rows.values() if str(row.get("edge") or "unknown") == "correct")
            edge_incorrect_count = sum(1 for row in solver_rows.values() if str(row.get("edge") or "unknown") == "incorrect")
            edge_unknown_count = sum(1 for row in solver_rows.values() if str(row.get("edge") or "unknown") == "unknown")
            path_correct_count = sum(1 for row in solver_rows.values() if str(row.get("path") or "unknown") == "correct")
            path_incorrect_count = sum(1 for row in solver_rows.values() if str(row.get("path") or "unknown") == "incorrect")
            path_unknown_count = sum(1 for row in solver_rows.values() if str(row.get("path") or "unknown") == "unknown")
            round_json = {
                "round": round_idx,
                "strong_summary": {
                    "edge": {
                        "correct_count": edge_correct_count,
                        "incorrect_count": edge_incorrect_count,
                        "unknown_count": edge_unknown_count,
                    },
                    "path": {
                        "correct_count": path_correct_count,
                        "incorrect_count": path_incorrect_count,
                        "unknown_count": path_unknown_count,
                    },
                },
                "solver_rows": [],
            }

            lines.append(f"### Round {round_idx}")
            lines.append("")
            lines.append("#### Strong Summary")
            lines.append("")
            lines.append("| View | correct_count | incorrect_count | unknown_count |")
            lines.append("| --- | --- | --- | --- |")
            lines.append(f"| edge | {edge_correct_count} | {edge_incorrect_count} | {edge_unknown_count} |")
            lines.append(f"| path | {path_correct_count} | {path_incorrect_count} | {path_unknown_count} |")
            lines.append("")
            lines.append("#### Per-Solver Details")
            lines.append("")
            lines.append("| Step | Tier | Model | Service ID | Edge | Edge Well-Posed | Path | Path Well-Posed |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for solver_key in sorted(solver_rows.keys()):
                row = solver_rows[solver_key]
                round_json["solver_rows"].append(
                    {
                        "step": int(row.get("step") or 0),
                        "tier": str(row.get("tier") or ""),
                        "model": str(row.get("model") or ""),
                        "service_id": str(row.get("service_id") or ""),
                        "edge": str(row.get("edge") or "unknown"),
                        "edge_well_posed": str(row.get("edge_well_posed") or "unknown"),
                        "path": str(row.get("path") or "unknown"),
                        "path_well_posed": str(row.get("path_well_posed") or "unknown"),
                    }
                )
                lines.append(
                    "| {step} | {tier} | {model} | {service_id} | {edge} | {edge_well_posed} | {path} | {path_well_posed} |".format(
                        step=row.get("step") or "",
                        tier=row.get("tier") or "",
                        model=row.get("model") or "",
                        service_id=row.get("service_id") or "",
                        edge=row.get("edge") or "unknown",
                        edge_well_posed=row.get("edge_well_posed") or "unknown",
                        path=row.get("path") or "unknown",
                        path_well_posed=row.get("path_well_posed") or "unknown",
                    )
                )
            lines.append("")
            rounds_json.append(round_json)

    return {
        "rows": summary_rows,
        "rounds": rounds_json,
        "markdown": "\n".join(lines).strip() + "\n",
    }


def _extract_fenced_json(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return ""
    # Drop think blocks if present
    for tag in ("</think>", "</analysis>"):
        if tag in candidate:
            candidate = candidate.split(tag, 1)[1].strip()
    if candidate.startswith("```"):
        fence = "```json" if candidate.startswith("```json") else "```"
        end = candidate.find("```", len(fence))
        if end != -1:
            candidate = candidate[len(fence) : end].strip()
    return candidate


def _parse_json_obj(text: str) -> Dict[str, Any] | None:
    candidate = _extract_fenced_json(text)
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Fallback: try extracting the outermost {...}
    try:
        i = candidate.find("{")
        j = candidate.rfind("}")
        if i != -1 and j != -1 and j > i:
            sub = candidate[i : j + 1]
            obj = json.loads(sub)
            return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Last resort: python literal eval (single quotes, etc.)
    try:
        obj = ast.literal_eval(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_json_keys(obj: Any) -> Any:
    """Best-effort: fix common model glitches like newlines inside keys (e.g., 'well\\n_posed')."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            # Collapse whitespace (incl. newlines/tabs) in keys.
            ks = "".join(ks.split())
            out[ks] = _normalize_json_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_json_keys(x) for x in obj]
    return obj


def _final_commenter_block(agent_conf: Dict[str, Any]) -> Dict[str, Any]:
    block = agent_conf.get("final_commenter") or {}
    return block if isinstance(block, dict) else {}


def _commenter_enabled(agent_conf: Dict[str, Any]) -> bool:
    block = _final_commenter_block(agent_conf)
    enabled = block.get("enabled", True)
    return bool(enabled)


def _resolve_generator(agent_conf: Dict[str, Any]) -> Dict[str, Any]:
    block = _final_commenter_block(agent_conf)
    gen = block.get("generator")
    if isinstance(gen, dict) and gen:
        return gen
    # Default fallback to director generator (same service endpoint)
    director_conf = agent_conf.get("director") or {}
    gen2 = director_conf.get("generator")
    if isinstance(gen2, dict) and gen2:
        return gen2
    return {"service_type": "private_endpoint", "service_id": director_conf.get("service_id")}


def _first_row(path: Path) -> Dict[str, Any] | None:
    try:
        for row in read_jsonl(path, schema=None, max_lines=1):
            return row if isinstance(row, dict) else None
    except Exception:
        return None
    return None


def _solver_result_dict(state: AgentState, step_idx: int, target: str, tier: str) -> Dict[str, Any] | None:
    res = state.get_latest_solver(step_idx, target, tier)
    if res is None:
        return None
    return {
        "correct": res.correct,
        "token_ratio": res.token_ratio,
        "model": res.model,
        "service_id": res.service_id,
        "question_well_posed": res.question_well_posed,
    }


def final_commenter_node(agent_conf: Dict[str, Any], state: AgentState) -> AgentState:
    """Best-effort final commenter: never blocks episode completion."""
    summary_dir = Path(state.artifacts_dir) / "00_Summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = summary_dir / "subruns_raw" / "final_commenter"
    raw_dir.mkdir(parents=True, exist_ok=True)
    round_stats = _build_strong_solver_round_stats(state)
    round_stats_md_path = summary_dir / "strong_solver_round_stats.md"
    round_stats_json_path = summary_dir / "strong_solver_round_stats.json"
    try:
        round_stats_md_path.write_text(str(round_stats.get("markdown") or ""), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write strong solver round stats markdown: %s", str(round_stats_md_path))
    try:
        round_stats_json_path.write_text(
            json.dumps(
                {
                    "rows": round_stats.get("rows") or [],
                    "rounds": round_stats.get("rounds") or [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("Failed to write strong solver round stats json: %s", str(round_stats_json_path))

    if not _commenter_enabled(agent_conf):
        return state
    if not state.history:
        return state

    try:
        step_idx = int(state.step or 0)
    except Exception:
        step_idx = 0

    # Build canonical inputs (also convenient for debugging)
    edge_kqa_path = dump_edge_kqa_for_step(state, raw_dir, filename="edge_kqa.jsonl")
    agent_block = agent_conf.get("agent") or {}
    fold_variant = str(agent_block.get("path_kqa_variant") or agent_block.get("path_fold_variant") or "direct").strip().lower()
    if fold_variant not in {"direct", "scaffolded"}:
        fold_variant = "direct"
    path_kqa_path = dump_path_kqa_for_step(state, raw_dir, filename="path_kqa.jsonl", fold_variant=fold_variant)
    edge_kqa = _first_row(edge_kqa_path) or {}
    path_kqa = _first_row(path_kqa_path) or {}

    solver_summary: Dict[str, Any] = {
        "step": step_idx,
        "edge": {
            "medium": _solver_result_dict(state, step_idx, "edge", "medium"),
            "strong": _solver_result_dict(state, step_idx, "edge", "strong"),
        },
        "path": {
            "medium": _solver_result_dict(state, step_idx, "path", "medium"),
            "strong": _solver_result_dict(state, step_idx, "path", "strong"),
        },
    }

    payload: Dict[str, Any] = {
        "edge_kqa": edge_kqa,
        "path_kqa": path_kqa,
        "solver_summary": solver_summary,
        "strong_solver_round_stats": round_stats.get("rows") or [],
    }

    block = _final_commenter_block(agent_conf)
    prompt_path = Path(block.get("prompt_path") or "src/agenqa/prompts/final_commenter.prompt")
    generator = _resolve_generator(agent_conf)
    max_attempts = max(1, int(block.get("max_retries", 2) or 2))
    retry_delay = float(block.get("retry_delay_seconds", 1.0) or 1.0)
    agent_lang = str((agent_conf.get("agent") or {}).get("lang") or "").lower() or None

    out_path = summary_dir / "final_comment.json"
    raw_resp_path = raw_dir / "response.txt"

    last_exc: Exception | None = None
    parsed: Dict[str, Any] | None = None
    raw_text: str = ""

    for attempt in range(max_attempts):
        try:
            resolved = resolve_inference(generator)
            sess = resolved.session

            prompt_text = build_final_commenter_v1_body(payload, lang=agent_lang)
            try:
                template_text = "\n\n".join(section.text for section in FINAL_COMMENTER_TEMPLATE.sections)
                snapshot_prompt_used(
                    prompt_path,
                    state.artifacts_dir / "00_Prompts_Snapshot",
                    content=template_text,
                    name_prefix="prompt_used.final_commenter.",
                    logger=logger,
                )
                snapshot_rendered_prompt(prompt_text, raw_dir, filename="prompt_rendered.txt", logger=logger)
            except Exception:
                pass

            messages = build_messages_with_background(prompt_text, lang=agent_lang)
            resp = sess.chat(messages, **resolved.chat_args)
            raw_text = sess.extract_text(resp, default="").strip()
            try:
                raw_resp_path.write_text(raw_text, encoding="utf-8")
            except Exception:
                pass

            parsed = _parse_json_obj(raw_text)
            if parsed is None:
                cleaned = None
                try:
                    cleaned = clean_json_text(
                        raw_text,
                        generator=generator,
                        task_name="final_commenter",
                        lang=agent_lang or "zh",
                        # Keep it permissive; we mainly want a dict-shaped JSON back.
                        required_keys=["well_posed"],
                        prompt_body=prompt_text,
                        snapshot_dir=raw_dir,
                        allow_python=True,
                    )
                except Exception:
                    cleaned = None
                if cleaned:
                    try:
                        (raw_dir / "response_cleaned_text.txt").write_text(cleaned, encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        parsed = json.loads(cleaned)
                    except Exception:
                        parsed = _parse_json_obj(cleaned)
                if parsed is None:
                    raise ValueError("Final commenter returned non-JSON output")

            parsed = _normalize_json_keys(parsed)

            # Write final comment
            output_payload = {
                "step": step_idx,
                "run_id": getattr(state, "run_id", None),
                "stop_reason": state.stop_reason,
                "model": getattr(sess, "model_name", None),
                "service_id": getattr(sess, "service_id", None),
                "strong_solver_round_stats_path": str(round_stats_md_path),
                "strong_solver_round_stats_json_path": str(round_stats_json_path),
                "comment": parsed,
            }
            out_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return state
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.error("Final commenter failed (%s/%s): %s", attempt + 1, max_attempts, str(exc))
            if attempt < max_attempts - 1:
                try:
                    time.sleep(retry_delay)
                except Exception:
                    pass

    # Failure fallback: write an error payload for visibility.
    err_payload = {
        "step": step_idx,
        "run_id": getattr(state, "run_id", None),
        "stop_reason": state.stop_reason,
        "error": str(last_exc) if last_exc else "unknown_error",
        "raw_text": raw_text[:4000] if isinstance(raw_text, str) else "",
    }
    try:
        out_path.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return state
