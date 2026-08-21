from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agenqa.downstream.sft.collector import discover_step_artifacts, load_step_snapshot
from agenqa.downstream.sft.filters import Phase1FilterConfig, decide_edge, decide_path_direct
from agenqa.evaluation.path_hardcase_audit import audit_candidate_dir, load_run_state_for_audit
from agenqa.graph.state import AgentState


REPO_ROOT = Path(__file__).resolve().parents[3]
_DATE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-20\d{2}(?:[-_]\d{2}[-_]\d{2}|\d{4,})$")
_SOLVE_STRONG_FILE_RE = re.compile(r"^solve_strong_(\d+)\.jsonl$")
_SOLVE_PATH_STRONG_FILE_RE = re.compile(r"^solve_path_strong_(\d+)\.jsonl$")


@dataclass(frozen=True)
class PricingEntry:
    input_per_1m_usd: float
    output_per_1m_usd: float


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _read_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _resolve_repo_path(value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _resolve_lineage_root(batch_dir: Path) -> Path:
    cur = batch_dir.resolve()
    seen: set[str] = set()
    while True:
        key = str(cur)
        if key in seen:
            return cur
        seen.add(key)
        manifest_path = cur / "batch_manifest.json"
        if not manifest_path.exists():
            return cur
        manifest = _read_optional_json(manifest_path)
        if not isinstance(manifest, dict):
            return cur
        hb_root = manifest.get("heartbeat_root_batch_dir")
        if isinstance(hb_root, str) and hb_root.strip():
            return _resolve_repo_path(hb_root)
        resumed_from = manifest.get("resumed_from_batch_dir")
        if not isinstance(resumed_from, str) or not resumed_from.strip():
            return cur
        cur = _resolve_repo_path(resumed_from)


def _read_batch_results_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("task_id"):
            out.append(row)
    return out


def _load_batch_lineage(root_batch_dir: Path) -> list[dict[str, Any]]:
    root_batch_dir = _resolve_lineage_root(root_batch_dir)
    root_manifest = _read_json(root_batch_dir / "batch_manifest.json")
    candidates: list[dict[str, Any]] = [{"path": root_batch_dir, "manifest": root_manifest}]
    for child in sorted(root_batch_dir.parent.iterdir()):
        if not child.is_dir() or child == root_batch_dir:
            continue
        manifest_path = child / "batch_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_optional_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        candidates.append({"path": child.resolve(), "manifest": manifest})

    selected: dict[str, dict[str, Any]] = {str(root_batch_dir): candidates[0]}
    changed = True
    while changed:
        changed = False
        selected_paths = {str(Path(key).resolve()) for key in selected}
        for item in candidates[1:]:
            path = Path(item["path"]).resolve()
            if str(path) in selected:
                continue
            manifest = item["manifest"]
            hb_root = manifest.get("heartbeat_root_batch_dir")
            if isinstance(hb_root, str) and hb_root.strip():
                if str(_resolve_repo_path(hb_root)) == str(root_batch_dir.resolve()):
                    selected[str(path)] = item
                    changed = True
                    continue
            resumed_from = manifest.get("resumed_from_batch_dir")
            if isinstance(resumed_from, str) and resumed_from.strip():
                if str(_resolve_repo_path(resumed_from)) in selected_paths:
                    selected[str(path)] = item
                    changed = True

    lineage = list(selected.values())
    lineage.sort(key=lambda item: (str(item["manifest"].get("created_at") or ""), str(item["path"])))
    for idx, item in enumerate(lineage):
        item["order"] = idx
        item["results"] = _read_batch_results_jsonl(Path(item["path"]) / "batch_results.jsonl")
    return lineage


def _collect_latest_lineage_rows(root_batch_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for item in _load_batch_lineage(root_batch_dir):
        for row in item.get("results") or []:
            task_id = str(row.get("task_id") or "")
            if task_id:
                latest[task_id] = dict(row)
    return latest


def _hardcase_label_key(row: dict[str, Any]) -> tuple[str, int | None, str, int | None]:
    return (
        str(row.get("run_id") or ""),
        _coerce_int(row.get("round")),
        str(row.get("stage") or ""),
        _coerce_int(row.get("step")),
    )


def _load_hardcase_label_overrides(path: Path | None) -> dict[tuple[str, int | None, str, int | None], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    overrides: dict[tuple[str, int | None, str, int | None], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = _hardcase_label_key(row)
        if not key[0]:
            continue
        overrides[key] = row
    return overrides


def _apply_hardcase_override(candidate_row: dict[str, Any], overrides: dict[tuple[str, int | None, str, int | None], dict[str, Any]]) -> dict[str, Any]:
    override = overrides.get(_hardcase_label_key(candidate_row))
    if not override:
        return candidate_row
    merged = dict(candidate_row)
    for key in (
        "hard_case_observed",
        "hard_case_ge_2",
        "hard_case_ge_3",
        "hard_case_majority",
        "primary_label",
        "evidence_codes",
        "review_priority",
        "evaluable",
    ):
        if key in override:
            merged[key] = override[key]
    return merged


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _price_value(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _load_pricing_manifest(path: Path) -> tuple[dict[str, PricingEntry], dict[str, str], dict[str, Any]]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected pricing manifest object at {path}")
    models_raw = manifest.get("models")
    if not isinstance(models_raw, dict) or not models_raw:
        raise ValueError(f"pricing manifest missing non-empty models map: {path}")
    pricing: dict[str, PricingEntry] = {}
    aliases: dict[str, str] = {}
    for model_name, payload in models_raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"pricing manifest model={model_name} must map to an object")
        input_price = _price_value(payload, "input_per_1m_usd", "prompt_per_1m_usd")
        output_price = _price_value(payload, "output_per_1m_usd", "completion_per_1m_usd")
        if input_price is None or output_price is None:
            raise ValueError(f"pricing manifest model={model_name} missing input/output per-1m usd")
        pricing[str(model_name)] = PricingEntry(
            input_per_1m_usd=input_price,
            output_per_1m_usd=output_price,
        )
        for alias in payload.get("aliases") or []:
            if isinstance(alias, str) and alias.strip():
                aliases[alias] = str(model_name)
    return pricing, aliases, manifest


def _resolve_pricing_model(model: str | None, pricing: dict[str, PricingEntry], aliases: dict[str, str]) -> str | None:
    if not model:
        return None
    if model in pricing:
        return model
    if model in aliases:
        return aliases[model]
    match = _DATE_SUFFIX_RE.match(model)
    if match:
        base = match.group("base")
        if base in pricing:
            return base
        if base in aliases:
            return aliases[base]
    return None


def _usage_pair(usage: Any) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    return _coerce_int(usage.get("prompt_tokens")), _coerce_int(usage.get("completion_tokens"))


def _empty_breakdown() -> dict[str, float]:
    return {}


def _increment_cost(breakdown: dict[str, float], key: str, amount: float) -> None:
    if not amount:
        return
    breakdown[key] = breakdown.get(key, 0.0) + amount


def _merge_breakdown(dst: dict[str, float], src: dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + float(value or 0.0)


def _calc_partial_cost(
    *,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing: dict[str, PricingEntry],
    aliases: dict[str, str],
) -> tuple[float, str | None, list[str]]:
    missing: list[str] = []
    resolved_model = _resolve_pricing_model(model, pricing, aliases)
    if resolved_model is None:
        missing.append("pricing_missing")
        return 0.0, None, missing
    entry = pricing[resolved_model]
    cost = 0.0
    if prompt_tokens is None:
        missing.append("prompt_tokens_missing")
    else:
        cost += (prompt_tokens / 1_000_000.0) * entry.input_per_1m_usd
    if completion_tokens is None:
        missing.append("completion_tokens_missing")
    else:
        cost += (completion_tokens / 1_000_000.0) * entry.output_per_1m_usd
    return cost, resolved_model, missing


def _run_config_models(run_dir: Path) -> dict[str, Any]:
    obj = _read_json(run_dir / "run_config.json")
    resolved = obj.get("resolved_config") if isinstance(obj, dict) else None
    if not isinstance(resolved, dict):
        return {}
    strong_models = [str(item.get("generator", {}).get("model_name") or item.get("id") or "") for item in resolved.get("solvers", {}).get("strong", []) if isinstance(item, dict)]
    return {
        "init": str(resolved.get("init", {}).get("generator", {}).get("model_name") or ""),
        "director": str(resolved.get("director", {}).get("generator", {}).get("model_name") or ""),
        "extend": {
            "draft_chain": str(resolved.get("operators", {}).get("extend", {}).get("generator", {}).get("model_name") or ""),
            "format": str(resolved.get("operators", {}).get("extend", {}).get("format_generator", {}).get("model_name") or ""),
            "step_cert_builder": str(resolved.get("operators", {}).get("extend", {}).get("step_cert_generator", {}).get("model_name") or ""),
            "path_fold": str(resolved.get("operators", {}).get("extend", {}).get("path_fold_generator", {}).get("model_name") or ""),
        },
        "revise": {
            "draft_chain": str(resolved.get("operators", {}).get("revise", {}).get("generator", {}).get("model_name") or ""),
            "format": str(resolved.get("operators", {}).get("revise", {}).get("format_generator", {}).get("model_name") or ""),
            "step_cert_builder": str(resolved.get("operators", {}).get("revise", {}).get("step_cert_generator", {}).get("model_name") or ""),
            "path_fold": str(resolved.get("operators", {}).get("revise", {}).get("path_fold_generator", {}).get("model_name") or ""),
        },
        "medium": str(resolved.get("solvers", {}).get("medium", {}).get("generator", {}).get("model_name") or ""),
        "strong": strong_models,
        "expression_judge": str(resolved.get("expression_judge", {}).get("generator", {}).get("model_name") or ""),
    }


def _stage_profile(step_dir: Path) -> str:
    return "revise" if step_dir.name.startswith("revise") else "extend"


def _request_meta_model(raw_dir: Path) -> str | None:
    meta = _read_optional_json(raw_dir / "request_meta.json")
    if isinstance(meta, dict):
        model = meta.get("model")
        if isinstance(model, str) and model.strip():
            return model
    return None


def _response_model(response_obj: Any) -> str | None:
    if isinstance(response_obj, dict):
        model = response_obj.get("model")
        if isinstance(model, str) and model.strip():
            return model
    return None


def _add_component(
    *,
    row: dict[str, Any],
    stage: str,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    source_path: str | None,
    pricing: dict[str, PricingEntry],
    aliases: dict[str, str],
    required: bool = True,
) -> None:
    partial_cost, priced_model, missing = _calc_partial_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pricing=pricing,
        aliases=aliases,
    )
    row["cost_usd_total"] += partial_cost
    if partial_cost:
        _increment_cost(row["cost_usd_by_stage"], stage, partial_cost)
    if partial_cost and priced_model:
        _increment_cost(row["cost_usd_by_model"], priced_model, partial_cost)
    if required and missing:
        row["cost_incomplete"] = True
        row["missing_cost_components"].append(
            {
                "stage": stage,
                "model": model,
                "pricing_model": priced_model,
                "missing": missing,
                "source_path": source_path,
            }
        )


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _add_init_cost(
    *,
    row: dict[str, Any],
    run_dir: Path,
    config_models: dict[str, Any],
    pricing: dict[str, PricingEntry],
    aliases: dict[str, str],
) -> None:
    response_path = _first_existing(
        sorted(run_dir.rglob("episode_seed_builder_response.json"), key=lambda p: str(p))
    )
    if response_path is None:
        row["cost_incomplete"] = True
        row["missing_cost_components"].append(
            {
                "stage": "init",
                "model": config_models.get("init"),
                "pricing_model": _resolve_pricing_model(config_models.get("init"), pricing, aliases),
                "missing": ["response_missing"],
                "source_path": None,
            }
        )
        return
    obj = _read_optional_json(response_path)
    prompt_tokens, completion_tokens = _usage_pair(obj.get("usage") if isinstance(obj, dict) else None)
    model = config_models.get("init") or _response_model(obj)
    _add_component(
        row=row,
        stage="init",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        source_path=str(response_path),
        pricing=pricing,
        aliases=aliases,
    )


def _generation_stage_specs(profile: str) -> list[tuple[str, str, str]]:
    _ = profile
    return [
        ("draft_chain", "draft_chain", "raw_response.json"),
        ("format", "format", "response.json"),
        ("step_cert_builder", "step_cert_builder", "raw_response.json"),
        ("path_fold", "path_fold", "raw_response.json"),
    ]


def _add_generation_costs(
    *,
    row: dict[str, Any],
    step_dir: Path,
    config_models: dict[str, Any],
    pricing: dict[str, PricingEntry],
    aliases: dict[str, str],
) -> None:
    profile = _stage_profile(step_dir)
    profile_models = config_models.get(profile) or {}
    for stage_name, subdir, response_file in _generation_stage_specs(profile):
        raw_dir = step_dir / "subruns_raw" / subdir
        response_path = raw_dir / response_file
        if not response_path.exists():
            continue
        response_obj = _read_optional_json(response_path)
        prompt_tokens, completion_tokens = _usage_pair(response_obj.get("usage") if isinstance(response_obj, dict) else None)
        model = _request_meta_model(raw_dir) or _response_model(response_obj) or profile_models.get(stage_name)
        _add_component(
            row=row,
            stage=stage_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source_path=str(response_path),
            pricing=pricing,
            aliases=aliases,
        )

    director_decision_path = step_dir / "director_decision.json"
    if director_decision_path.exists():
        _add_component(
            row=row,
            stage="director",
            model=config_models.get("director"),
            prompt_tokens=None,
            completion_tokens=None,
            source_path=str(director_decision_path),
            pricing=pricing,
            aliases=aliases,
        )


def _solve_row_sort_key(path: Path, prefix_re: re.Pattern[str]) -> tuple[int, str]:
    match = prefix_re.fullmatch(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10_000, path.name


def _solve_component_paths(step_dir: Path) -> list[tuple[str, Path]]:
    solve_dir = step_dir / "solve"
    if not solve_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    if (solve_dir / "solve_medium.jsonl").exists():
        out.append(("solve_edge_medium", solve_dir / "solve_medium.jsonl"))
    if (solve_dir / "solve_path_medium.jsonl").exists():
        out.append(("solve_path_medium", solve_dir / "solve_path_medium.jsonl"))
    strong_files = sorted(
        [path for path in solve_dir.glob("solve_strong_*.jsonl") if _SOLVE_STRONG_FILE_RE.fullmatch(path.name)],
        key=lambda path: _solve_row_sort_key(path, _SOLVE_STRONG_FILE_RE),
    )
    out.extend(("solve_edge_strong", path) for path in strong_files)
    path_strong_files = sorted(
        [path for path in solve_dir.glob("solve_path_strong_*.jsonl") if _SOLVE_PATH_STRONG_FILE_RE.fullmatch(path.name)],
        key=lambda path: _solve_row_sort_key(path, _SOLVE_PATH_STRONG_FILE_RE),
    )
    out.extend(("solve_path_strong", path) for path in path_strong_files)
    return out


def _solve_prompt_tokens(row: dict[str, Any]) -> tuple[int | None, int | None, dict[str, Any]]:
    metrics = row.get("metrics")
    prompt_tokens = None
    completion_tokens = None
    extras: dict[str, Any] = {}
    if isinstance(metrics, dict):
        prompt_tokens = _coerce_int(metrics.get("prompt_tokens"))
        completion_tokens = _coerce_int(metrics.get("completion_tokens"))
        if prompt_tokens is None and _coerce_int(metrics.get("kq_tokens")) is not None:
            extras["kq_tokens"] = _coerce_int(metrics.get("kq_tokens"))
    return prompt_tokens, completion_tokens, extras


def _add_solve_and_judge_costs(
    *,
    row: dict[str, Any],
    step_dir: Path,
    config_models: dict[str, Any],
    pricing: dict[str, PricingEntry],
    aliases: dict[str, str],
) -> None:
    judge_model = config_models.get("expression_judge")
    for stage_name, path in _solve_component_paths(step_dir):
        solve_rows = _read_jsonl(path)
        if not solve_rows:
            continue
        solve_row = solve_rows[0]
        if str(solve_row.get("solver_status") or "") != "success":
            continue
        prompt_tokens, completion_tokens, extras = _solve_prompt_tokens(solve_row)
        model = str(solve_row.get("model") or "").strip() or None
        _add_component(
            row=row,
            stage=stage_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source_path=str(path),
            pricing=pricing,
            aliases=aliases,
        )
        if extras:
            if row["missing_cost_components"]:
                latest = row["missing_cost_components"][-1]
                if latest.get("source_path") == str(path):
                    latest["extras"] = extras
        expression_judge = solve_row.get("expression_judge")
        if not isinstance(expression_judge, dict):
            continue
        for payload in expression_judge.values():
            if not isinstance(payload, dict):
                continue
            if str(payload.get("status") or "") != "success":
                continue
            _add_component(
                row=row,
                stage="expression_judge",
                model=judge_model,
                prompt_tokens=None,
                completion_tokens=None,
                source_path=str(path),
                pricing=pricing,
                aliases=aliases,
            )


def _initial_cost_row(task_row: dict[str, Any], candidate_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_row["task_id"],
        "paper_path": task_row["paper_path"],
        "run_id": candidate_row["run_id"],
        "run_dir": task_row["run_dir"],
        "candidate_id": candidate_row["candidate_id"],
        "round": candidate_row["round"],
        "step": candidate_row["step"],
        "hard_round": candidate_row["round"],
        "hard_step": candidate_row["step"],
        "question_type": candidate_row["question_type"],
        "hard_label_observed": "observed_hard" if candidate_row["hard_case_observed"] else "not_hard",
        "hard_label_primary": candidate_row["primary_label"],
        "is_observed_hard": bool(candidate_row["hard_case_observed"]),
        "is_likely_true_hard": candidate_row["primary_label"] == "likely_true_hard",
        "paper_id": candidate_row.get("paper_id"),
        "chain": candidate_row.get("chain"),
        "subject": candidate_row.get("subject"),
        "source": candidate_row.get("source"),
        "cost_usd_total": 0.0,
        "cost_usd_by_stage": _empty_breakdown(),
        "cost_usd_by_model": _empty_breakdown(),
        "cost_incomplete": False,
        "missing_cost_components": [],
        "byproduct_sft_sample_count_total": 0,
        "byproduct_sft_edge_count": 0,
        "byproduct_sft_path_count": 0,
        "byproduct_round_count": 0,
        "byproduct_steps": [],
        "cost_usd_per_byproduct_sft_sample": None,
        "audit_evidence_codes": list(candidate_row.get("evidence_codes") or []),
        "audit_source_paths": list(candidate_row.get("source_paths") or []),
    }


def _count_byproduct_sft_samples(
    *,
    run_dir: Path,
    hard_step: int,
    config: Phase1FilterConfig,
) -> tuple[int, int, int, list[int]]:
    state = AgentState.load_from_file(run_dir / "state.json")
    edge_count = 0
    path_count = 0
    accepted_steps: list[int] = []
    for artifacts in discover_step_artifacts(run_dir):
        if int(artifacts.step) >= int(hard_step):
            continue
        snapshot = load_step_snapshot(artifacts, state=state)
        step_has_sample = False
        edge_decision = decide_edge(snapshot, config)
        if edge_decision.accepted:
            edge_count += 1
            step_has_sample = True
        path_decision = decide_path_direct(snapshot, config)
        if path_decision.accepted:
            path_count += 1
            step_has_sample = True
        if step_has_sample:
            accepted_steps.append(int(artifacts.step))
    accepted_steps = sorted(set(accepted_steps))
    return edge_count + path_count, edge_count, path_count, accepted_steps


def _candidate_rows_for_winning_chain(
    *,
    task_row: dict[str, Any],
    root_batch_id: str,
    hardcase_overrides: dict[tuple[str, int | None, str, int | None], dict[str, Any]],
) -> list[tuple[dict[str, Any], Path]]:
    run_dir = Path(task_row["run_dir"]).resolve()
    state = load_run_state_for_audit(run_dir)
    candidate_rows: list[tuple[dict[str, Any], Path]] = []
    artifacts_by_step = {int(art.step): art for art in discover_step_artifacts(run_dir)}
    for step in sorted(artifacts_by_step):
        artifacts = artifacts_by_step[step]
        candidate_dir = artifacts.step_dir
        candidate_rows.append(
            (
                _apply_hardcase_override(
                    audit_candidate_dir(
                        run_dir=run_dir,
                        candidate_dir=candidate_dir,
                        state=state,
                        batch_id=root_batch_id,
                        batch_result=task_row,
                    ),
                    hardcase_overrides,
                ),
                candidate_dir,
            )
        )
    return candidate_rows


def _build_hard_question_rows(
    *,
    task_row: dict[str, Any],
    root_batch_id: str,
    hardcase_overrides: dict[tuple[str, int | None, str, int | None], dict[str, Any]],
    pricing: dict[str, PricingEntry],
    aliases: dict[str, str],
    filter_config: Phase1FilterConfig,
) -> list[dict[str, Any]]:
    run_dir = Path(task_row["run_dir"]).resolve()
    config_models = _run_config_models(run_dir)
    candidate_rows = _candidate_rows_for_winning_chain(
        task_row=task_row,
        root_batch_id=root_batch_id,
        hardcase_overrides=hardcase_overrides,
    )
    candidate_by_step = {int(row["step"]): (row, candidate_dir) for row, candidate_dir in candidate_rows}
    out: list[dict[str, Any]] = []
    for step in sorted(candidate_by_step):
        candidate_row, _ = candidate_by_step[step]
        if candidate_row.get("hard_case_observed") is not True:
            continue
        row = _initial_cost_row(task_row, candidate_row)
        _add_init_cost(row=row, run_dir=run_dir, config_models=config_models, pricing=pricing, aliases=aliases)
        for current_step in sorted(candidate_by_step):
            if int(current_step) > int(step):
                break
            _, candidate_dir = candidate_by_step[current_step]
            _add_generation_costs(
                row=row,
                step_dir=candidate_dir,
                config_models=config_models,
                pricing=pricing,
                aliases=aliases,
            )
            _add_solve_and_judge_costs(
                row=row,
                step_dir=candidate_dir,
                config_models=config_models,
                pricing=pricing,
                aliases=aliases,
            )
        total_samples, edge_count, path_count, accepted_steps = _count_byproduct_sft_samples(
            run_dir=run_dir,
            hard_step=int(step),
            config=filter_config,
        )
        row["byproduct_sft_sample_count_total"] = total_samples
        row["byproduct_sft_edge_count"] = edge_count
        row["byproduct_sft_path_count"] = path_count
        row["byproduct_round_count"] = len(accepted_steps)
        row["byproduct_steps"] = accepted_steps
        if total_samples > 0:
            row["cost_usd_per_byproduct_sft_sample"] = row["cost_usd_total"] / float(total_samples)
        out.append(row)
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return ordered[rank]


def _summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete_rows = [row for row in rows if row.get("cost_incomplete") is not True]
    complete_costs = [float(row.get("cost_usd_total") or 0.0) for row in complete_rows]
    total_stage_costs: dict[str, float] = {}
    total_model_costs: dict[str, float] = {}
    missing_component_counts = Counter()
    for row in rows:
        for component in row.get("missing_cost_components") or []:
            missing_component_counts[str(component.get("stage") or "unknown")] += 1
    for row in complete_rows:
        _merge_breakdown(total_stage_costs, row.get("cost_usd_by_stage") or {})
        _merge_breakdown(total_model_costs, row.get("cost_usd_by_model") or {})
    avg_stage_costs = {
        key: value / len(complete_rows)
        for key, value in sorted(total_stage_costs.items())
        if complete_rows
    }
    avg_model_costs = {
        key: value / len(complete_rows)
        for key, value in sorted(total_model_costs.items())
        if complete_rows
    }
    byproduct_total = sum(int(row.get("byproduct_sft_sample_count_total") or 0) for row in rows)
    byproduct_edge_total = sum(int(row.get("byproduct_sft_edge_count") or 0) for row in rows)
    byproduct_path_total = sum(int(row.get("byproduct_sft_path_count") or 0) for row in rows)
    return {
        "hard_question_count": len(rows),
        "complete_cost_count": len(complete_rows),
        "incomplete_cost_count": len(rows) - len(complete_rows),
        "mean_cost_usd": _mean(complete_costs),
        "median_cost_usd": _median(complete_costs),
        "p90_cost_usd": _p90(complete_costs),
        "avg_cost_usd_by_stage": avg_stage_costs,
        "avg_cost_usd_by_model": avg_model_costs,
        "mean_byproduct_sft_sample_count_total": _mean([int(row.get("byproduct_sft_sample_count_total") or 0) for row in rows]),
        "mean_byproduct_sft_edge_count": _mean([int(row.get("byproduct_sft_edge_count") or 0) for row in rows]),
        "mean_byproduct_sft_path_count": _mean([int(row.get("byproduct_sft_path_count") or 0) for row in rows]),
        "byproduct_sft_sample_count_total": byproduct_total,
        "byproduct_sft_edge_count_total": byproduct_edge_total,
        "byproduct_sft_path_count_total": byproduct_path_total,
        "mean_cost_usd_per_byproduct_sft_sample": _mean(
            [
                float(row["cost_usd_per_byproduct_sft_sample"])
                for row in rows
                if row.get("cost_usd_per_byproduct_sft_sample") is not None
            ]
        ),
        "missing_cost_component_counts": dict(sorted(missing_component_counts.items())),
    }


def _format_usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _render_breakdown_md(title: str, breakdown: dict[str, float]) -> str:
    if not breakdown:
        return f"### {title}\n\n_None_\n"
    lines = [f"### {title}", ""]
    for key, value in breakdown.items():
        if float(value).is_integer():
            rendered = str(int(value))
        else:
            rendered = f"{value:.6f}"
        lines.append(f"- `{key}`: `{rendered}`")
    lines.append("")
    return "\n".join(lines)


def _render_summary_md(summary: dict[str, Any]) -> str:
    def render_section(title: str, payload: dict[str, Any]) -> list[str]:
        lines = [
            f"## {title}",
            "",
            f"- hard_question_count: `{payload['hard_question_count']}`",
            f"- complete_cost_count: `{payload['complete_cost_count']}`",
            f"- incomplete_cost_count: `{payload['incomplete_cost_count']}`",
            f"- mean_cost_usd: `{_format_usd(payload['mean_cost_usd'])}`",
            f"- median_cost_usd: `{_format_usd(payload['median_cost_usd'])}`",
            f"- p90_cost_usd: `{_format_usd(payload['p90_cost_usd'])}`",
            f"- mean_byproduct_sft_sample_count_total: `{payload['mean_byproduct_sft_sample_count_total']}`",
            f"- mean_byproduct_sft_edge_count: `{payload['mean_byproduct_sft_edge_count']}`",
            f"- mean_byproduct_sft_path_count: `{payload['mean_byproduct_sft_path_count']}`",
            f"- byproduct_sft_sample_count_total: `{payload['byproduct_sft_sample_count_total']}`",
            f"- mean_cost_usd_per_byproduct_sft_sample: `{_format_usd(payload['mean_cost_usd_per_byproduct_sft_sample'])}`",
            "",
        ]
        lines.append(_render_breakdown_md("Average Cost By Stage", payload["avg_cost_usd_by_stage"]).rstrip())
        lines.append("")
        lines.append(_render_breakdown_md("Average Cost By Model", payload["avg_cost_usd_by_model"]).rstrip())
        lines.append("")
        lines.append(_render_breakdown_md("Missing Cost Component Counts", {k: float(v) for k, v in payload["missing_cost_component_counts"].items()}).rstrip())
        lines.append("")
        return lines

    lines = [
        "# Hard Question Cost Audit",
        "",
        "## Summary",
        f"- generated_at: `{summary['generated_at']}`",
        f"- batch_dir: `{summary['input']['batch_dir']}`",
        f"- root_batch_dir: `{summary['input']['root_batch_dir']}`",
        f"- pricing_manifest: `{summary['input']['pricing_manifest']}`",
        f"- pricing_version: `{summary['input']['pricing_version']}`",
        f"- winning_run_count: `{summary['input']['winning_run_count']}`",
        "",
    ]
    lines.extend(render_section("Observed Hard", summary["observed_hard"]))
    lines.extend(render_section("Likely True Hard", summary["likely_true_hard"]))
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_hard_question_cost_audit(
    *,
    batch_dir: Path,
    output_dir: Path,
    pricing_manifest: Path,
    hardcase_jsonl: Path | None = None,
    filter_config: Phase1FilterConfig | None = None,
) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    root_batch_dir = _resolve_lineage_root(batch_dir)
    root_manifest = _read_json(root_batch_dir / "batch_manifest.json")
    pricing, aliases, pricing_meta = _load_pricing_manifest(pricing_manifest.resolve())
    hardcase_overrides = _load_hardcase_label_overrides(hardcase_jsonl.resolve() if hardcase_jsonl else None)
    latest_rows = _collect_latest_lineage_rows(root_batch_dir)
    winning_rows = [
        dict(row)
        for _, row in sorted(latest_rows.items())
        if str(row.get("status") or "") == "success" and row.get("run_dir")
    ]
    cfg = filter_config or Phase1FilterConfig()

    hard_rows: list[dict[str, Any]] = []
    for row in winning_rows:
        hard_rows.extend(
            _build_hard_question_rows(
                task_row=row,
                root_batch_id=str(root_manifest.get("batch_id") or root_batch_dir.name),
                hardcase_overrides=hardcase_overrides,
                pricing=pricing,
                aliases=aliases,
                filter_config=cfg,
            )
        )

    observed_rows = list(hard_rows)
    likely_true_rows = [row for row in hard_rows if row.get("is_likely_true_hard") is True]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "batch_dir": str(batch_dir),
            "root_batch_dir": str(root_batch_dir),
            "pricing_manifest": str(pricing_manifest.resolve()),
            "pricing_version": pricing_meta.get("version"),
            "hardcase_jsonl": str(hardcase_jsonl.resolve()) if hardcase_jsonl else None,
            "winning_run_count": len(winning_rows),
        },
        "observed_hard": _summarize_subset(observed_rows),
        "likely_true_hard": _summarize_subset(likely_true_rows),
    }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "hard_question_costs.jsonl", hard_rows)
    _write_json(output_dir / "hard_question_cost_summary.json", summary)
    (output_dir / "hard_question_cost_summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate net USD production cost for upstream content-hard questions and same-chain SFT byproducts."
    )
    parser.add_argument("--batch-dir", type=Path, required=True, help="Current formal-line root batch directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for audit outputs.")
    parser.add_argument("--pricing-manifest", type=Path, required=True, help="Versioned USD pricing manifest JSON.")
    parser.add_argument(
        "--hardcase-jsonl",
        type=Path,
        default=None,
        help="Optional candidates.jsonl/hard_cases.jsonl from path_hardcase_audit. If absent, labels are recomputed on winning candidates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_hard_question_cost_audit(
        batch_dir=args.batch_dir,
        output_dir=args.output_dir,
        pricing_manifest=args.pricing_manifest,
        hardcase_jsonl=args.hardcase_jsonl,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
