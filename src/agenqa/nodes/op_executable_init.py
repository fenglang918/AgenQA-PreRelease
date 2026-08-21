from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import contextlib
import logging
import os
import json

from infra.data.ids import generate_paper_id
from infra.data.io import read_jsonl, read_text_file
from agenqa.domain.known_tree import KnownTree
from agenqa.domain.node_result import NodeResult, OutputSpec
from agenqa.graph.output_manager import OutputContext, compute_step_dir
from agenqa.graph.state import AgentState
from agenqa.memory.store import dump_director_decision_for_step, save_state
from agenqa.nodes.utils import build_director_notes

logger = logging.getLogger(__name__)


def _truncate_text(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        return ""
    max_chars = int(max_chars)
    if max_chars <= 0:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)].rstrip() + "..."


@dataclass
class ExecutableInitSettings:
    source: str
    papers_path: Path
    scicode_root: Path
    split: str
    problem_id: Optional[str]
    with_background: bool
    hf_use_proxy: bool
    hf_home: Optional[Path]
    init_generator: Optional[Dict[str, Any]]
    extract_enabled: bool
    extract_prompt_path: Path
    extract_generator: Optional[Dict[str, Any]]
    dependencies_whitelist: str
    paper_brief_enabled: bool
    paper_brief_version: str
    paper_brief_prompt_path: Optional[Path]
    paper_brief_generator: Optional[Dict[str, Any]]
    paper_brief_max_chars: int
    keep_paper_text_in_memory: bool


def _resolve_settings(agent_conf: Dict[str, Any]) -> ExecutableInitSettings:
    from infra.config.init_config import assert_no_legacy_init_config, normalize_paper_brief_mode

    assert_no_legacy_init_config(agent_conf)

    init_conf = agent_conf.get("init") if isinstance(agent_conf, dict) else None
    if not isinstance(init_conf, dict):
        raise ValueError("Executable init requires top-level init config block.")

    source_conf = init_conf.get("source") if isinstance(init_conf.get("source"), dict) else None
    if not isinstance(source_conf, dict):
        raise ValueError("Executable init requires init.source block.")

    source = str(source_conf.get("type") or "scicode").strip().lower()
    path_raw = source_conf.get("path")
    papers_path = Path(str(path_raw).strip()) if isinstance(path_raw, str) and str(path_raw).strip() else None
    if source == "paper" and papers_path is None:
        raise ValueError("init.source.path is required when init.source.type=paper.")
    if papers_path is None:
        papers_path = Path("data/input/papers/first_line.jsonl")

    scicode_root = Path(source_conf.get("scicode_root") or "external/eval_inno/scicode/SciCode")
    split = str(source_conf.get("split") or "validation").strip()
    problem_id = source_conf.get("problem_id")
    problem_id = str(problem_id).strip() if isinstance(problem_id, (str, int)) and str(problem_id).strip() else None
    with_background = bool(source_conf.get("with_background", False))
    hf_use_proxy = bool(source_conf.get("hf_use_proxy", False))
    hf_home_raw = source_conf.get("hf_home")
    hf_home = Path(hf_home_raw) if isinstance(hf_home_raw, str) and hf_home_raw.strip() else None

    init_generator = init_conf.get("generator") if isinstance(init_conf.get("generator"), dict) else None

    extract_conf = init_conf.get("extract") if isinstance(init_conf.get("extract"), dict) else {}
    extract_enabled = bool(extract_conf.get("enable", True))
    extract_prompt_path = Path(extract_conf.get("prompt_path") or "src/agenqa/prompts/executable_extract.prompt")

    extract_generator = extract_conf.get("generator")
    extract_generator = extract_generator if isinstance(extract_generator, dict) else None

    whitelist = extract_conf.get("dependencies_whitelist", None)
    if isinstance(whitelist, list):
        dependencies_whitelist = "\n".join(str(item) for item in whitelist if str(item).strip())
    elif isinstance(whitelist, str):
        dependencies_whitelist = whitelist.strip()
    else:
        dependencies_whitelist = "\n".join(["numpy", "scipy", "sympy", "h5py"])

    paper_brief_block = init_conf.get("paper_brief") if isinstance(init_conf.get("paper_brief"), dict) else {}
    paper_brief_enabled_raw = paper_brief_block.get("enable", None)
    # Default: for paper source, enable PaperBrief when with_background=true (avoid dumping full paper text).
    if paper_brief_enabled_raw is None:
        paper_brief_enabled = (source == "paper") and with_background
    else:
        paper_brief_enabled = bool(paper_brief_enabled_raw)
    paper_brief_version = normalize_paper_brief_mode(paper_brief_block.get("version"), default="subject-keywords-skeleton")

    paper_brief_prompt_path = paper_brief_block.get("prompt_path")
    paper_brief_prompt_path = (
        Path(paper_brief_prompt_path)
        if isinstance(paper_brief_prompt_path, str) and paper_brief_prompt_path.strip()
        else None
    )

    paper_brief_generator = paper_brief_block.get("generator")
    paper_brief_generator = paper_brief_generator if isinstance(paper_brief_generator, dict) else None

    paper_brief_max_chars = int(paper_brief_block.get("max_chars") or 6000)
    keep_paper_text_in_memory = bool(paper_brief_block.get("keep_paper_text", False))
    return ExecutableInitSettings(
        source=source,
        papers_path=papers_path,
        scicode_root=scicode_root,
        split=split,
        problem_id=problem_id,
        with_background=with_background,
        hf_use_proxy=hf_use_proxy,
        hf_home=hf_home,
        init_generator=init_generator,
        extract_enabled=extract_enabled,
        extract_prompt_path=extract_prompt_path,
        extract_generator=extract_generator,
        dependencies_whitelist=dependencies_whitelist,
        paper_brief_enabled=paper_brief_enabled,
        paper_brief_version=paper_brief_version,
        paper_brief_prompt_path=paper_brief_prompt_path,
        paper_brief_generator=paper_brief_generator,
        paper_brief_max_chars=paper_brief_max_chars,
        keep_paper_text_in_memory=keep_paper_text_in_memory,
    )


@contextlib.contextmanager
def _temporary_unset_env(keys: list[str]):
    old: Dict[str, Optional[str]] = {}
    for k in keys:
        old[k] = os.environ.get(k)
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _temporary_set_env(values: dict[str, str]):
    old: Dict[str, Optional[str]] = {}
    for k, v in values.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    val = value.strip().lower()
    return val not in {"", "0", "false", "off", "no"}


def _extract_scicode_problem_background(record: Dict[str, Any]) -> str:
    for key in (
        # SciCode jsonl fields (preferred)
        "problem_description_main",
        "problem_background_main",
        "problem_description",
        "problem_background",
        "background",
        # Fallback: IO block often contains full specification
        "problem_io",
    ):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Paper/text fallback (supports json/text inputs)
    paper_text = _compose_paper_text(record)
    if paper_text:
        return paper_text
    pages = record.get("pages")
    if isinstance(pages, list):
        chunks = []
        for page in pages:
            if isinstance(page, dict):
                text = page.get("text")
            else:
                text = page
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        if chunks:
            return "\n\n".join(chunks)
    return ""


def _load_scicode_record(*, split: str, problem_id: str | None, use_proxy: bool) -> Dict[str, Any]:
    proxy_keys = [
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
    ]
    proxy_ctx = contextlib.nullcontext() if use_proxy else _temporary_unset_env(proxy_keys)
    # Default to offline/local HF cache unless explicitly forced online.
    force_online = _env_truthy(os.environ.get("SCICLONE_HF_FORCE_ONLINE"))
    offline_enabled = (not use_proxy) and (not force_online)
    if offline_enabled:
        offline_env = {"HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    elif force_online:
        offline_env = {"HF_DATASETS_OFFLINE": "0", "HF_HUB_OFFLINE": "0"}
    else:
        offline_env = {}

    with proxy_ctx, _temporary_set_env(offline_env):
        try:
            from datasets import config as ds_config
            from datasets import load_dataset
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Missing dependency `datasets`. Install requirements_scicode_import.txt to use SciCode source."
            ) from exc
        prev_ds_offline = bool(getattr(ds_config, "HF_DATASETS_OFFLINE", False))
        prev_hub_offline = bool(getattr(ds_config, "HF_HUB_OFFLINE", False))
        try:
            if offline_enabled:
                ds_config.HF_DATASETS_OFFLINE = True
                ds_config.HF_HUB_OFFLINE = True
            elif force_online:
                ds_config.HF_DATASETS_OFFLINE = False
                ds_config.HF_HUB_OFFLINE = False
            dataset = load_dataset("SciCode1/SciCode", split=split)
        finally:
            ds_config.HF_DATASETS_OFFLINE = prev_ds_offline
            ds_config.HF_HUB_OFFLINE = prev_hub_offline
        # NOTE: datasets.Dataset supports iteration yielding dict rows.
        for row in dataset:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("problem_id") or "")
            if problem_id is None or pid == problem_id:
                return dict(row)
    raise ValueError(f"SciCode problem_id={problem_id!r} not found in split={split!r}")


def _compose_paper_text(record: Dict[str, Any]) -> str:
    title = record.get("title") or (record.get("metadata") or {}).get("title")
    abstract = record.get("abstract") or (record.get("metadata") or {}).get("abstract")
    text = record.get("text") or record.get("paper_text")
    parts = []
    if isinstance(title, str) and title.strip():
        parts.append(f"Title: {title.strip()}")
    if isinstance(abstract, str) and abstract.strip():
        parts.append(f"Abstract: {abstract.strip()}")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    if parts:
        return "\n\n".join(parts)
    return ""


def _load_paper_record(papers_path: Path) -> Dict[str, Any]:
    if not papers_path.exists():
        raise FileNotFoundError(f"papers_path not found: {papers_path}")

    suffix = papers_path.suffix.lower()
    record: Dict[str, Any]
    if suffix in {".txt", ".md", ""}:
        text = read_text_file(papers_path)
        record = {"text": text}
    elif suffix == ".jsonl":
        record = next(read_jsonl(papers_path), None) or {}
    elif suffix == ".json":
        raw = json.loads(read_text_file(papers_path))
        if isinstance(raw, list):
            record = raw[0] if raw else {}
        elif isinstance(raw, dict):
            record = raw
        else:
            record = {}
    else:
        text = read_text_file(papers_path)
        record = {"text": text}

    if not isinstance(record, dict):
        raise ValueError(f"Unsupported paper record in {papers_path}")

    if not isinstance(record.get("text"), str) or not record.get("text", "").strip():
        pages = record.get("pages")
        if isinstance(pages, list):
            chunks = []
            for page in pages:
                if isinstance(page, dict):
                    text = page.get("text")
                else:
                    text = page
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            if chunks:
                record["text"] = "\n\n".join(chunks)

    if "paper_id" not in record:
        record["paper_id"] = generate_paper_id(record)
    return record


def run_executable_init(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    """ExecutableInit: initialize a executable episode seed (no executable step output yet)."""
    settings = _resolve_settings(agent_conf)
    if settings.source not in {"scicode", "paper"}:
        raise ValueError(f"Unsupported init.source.type={settings.source!r} (supported: scicode, paper)")

    if settings.source == "scicode":
        if settings.hf_home:
            os.environ.setdefault("HF_HOME", str(settings.hf_home))
            os.environ.setdefault("HF_HUB_CACHE", str(settings.hf_home / "hub"))
            os.environ.setdefault("HF_DATASETS_CACHE", str(settings.hf_home / "datasets"))

        record = _load_scicode_record(split=settings.split, problem_id=settings.problem_id, use_proxy=settings.hf_use_proxy)
        prob_id = str(record.get("problem_id") or settings.problem_id or "unknown")
    else:
        record = _load_paper_record(settings.papers_path)
        prob_id = str(record.get("paper_id") or "unknown")

    # init is step 0 (seed stage)
    step_idx = 0
    round_idx = state.current_round_index()
    if output_manager:
        ctx: OutputContext = output_manager.begin("init", step_idx, round_idx)
        step_dir = ctx.step_dir
        ctx.dump_director_decision(state, step_idx)
    else:
        step_dir = compute_step_dir(state.artifacts_dir, "init", step_idx, round_idx)
        step_dir.mkdir(parents=True, exist_ok=True)
        dump_director_decision_for_step(state, step_dir, step_idx)

    mem = KnownTree.normalize_memory(getattr(state, "memory", None))
    seed: Dict[str, Any] = {
        "source": settings.source,
        "problem_id": prob_id,
        "with_background": settings.with_background,
        "record": record,
    }
    if settings.source == "scicode":
        seed.update(
            {
                "split": settings.split,
                "hf_use_proxy": settings.hf_use_proxy,
                "scicode_root": str(settings.scicode_root),
            }
        )
    else:
        seed.update(
            {
                "papers_path": str(settings.papers_path),
                "paper_id": record.get("paper_id"),
            }
        )
    mem["executable_seed"] = seed
    # Align HEAD abstraction with semantic pipeline: always provide an episode_seed anchor.
    try:
        if settings.source == "scicode":
            subject = str(record.get("subject") or "").strip()
            if not subject:
                subject = f"SciCode executable problem {prob_id}"
            keywords = [f"executable", f"scicode", f"problem_id={prob_id}"]
        else:
            subject = str(record.get("title") or (record.get("metadata") or {}).get("title") or "").strip()
            if not subject:
                subject = f"Paper executable seed {prob_id}"
            keywords = [f"executable", f"paper", f"paper_id={prob_id}"]
        mem = KnownTree.update_episode_seed(
            mem,
            subject=subject,
            keywords=keywords,
        )
    except Exception:
        pass
    state.memory = mem
    save_state(state)

    # `KnownTree.update_episode_seed()` normalizes memory via deepcopy; re-bind `record` to the in-memory
    # instance so later mutations (PaperBrief/Extract) affect the persisted seed.
    try:
        seed_mem = mem.get("executable_seed") if isinstance(mem, dict) else None
        record_mem = seed_mem.get("record") if isinstance(seed_mem, dict) else None
        if isinstance(record_mem, dict):
            record = record_mem
    except Exception:
        pass

    role_outputs: Dict[str, Any] = {}

    # For paper seeds: optionally generate a compact brief (subject/keywords/summary/structured fields),
    # instead of carrying the full paper text in `executable_seed.record`.
    if settings.source == "paper" and settings.paper_brief_enabled:
        try:
            from agenqa.skills.paper_brief import PaperBriefConfig, PaperBriefRunner, render_brief_text
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"paper_brief import failed: {exc}") from exc

        # Snapshot the raw paper record to disk for debugging; keep memory lean downstream.
        try:
            (step_dir / "paper_record_raw.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        lang = (agent_conf.get("agent") or {}).get("lang")
        # Prefer explicit generator; fallback to init.generator (no cross-module fallback).
        brief_gen = settings.paper_brief_generator or settings.init_generator
        if not isinstance(brief_gen, dict) or not brief_gen:
            raise RuntimeError("PaperBrief requires init.generator or init.paper_brief.generator.")

        version = str(settings.paper_brief_version or "subject-keywords-skeleton").strip().lower()
        version = normalize_paper_brief_mode(version, default="subject-keywords-skeleton")
        if settings.paper_brief_prompt_path is not None:
            prompt_path = settings.paper_brief_prompt_path
        elif version == "subject-keywords":
            prompt_path = Path("src/agenqa/prompts/paper_brief_v3_seed.prompt")
        elif version == "subject-keywords-brief":
            prompt_path = Path("src/agenqa/prompts/paper_brief.prompt")
        else:
            # default reasoning
            prompt_path = Path("src/agenqa/prompts/paper_brief_v2_reasoning.prompt")

        brief_runner = PaperBriefRunner(
            PaperBriefConfig(
                generator=brief_gen,
                prompt_path=prompt_path,
                lang=lang,
            )
        )
        brief = brief_runner.run_one(
            record,
            snapshot_dir=step_dir / "subruns_raw" / "paper_brief",
            unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
        )
        if not isinstance(brief, dict):
            raise RuntimeError("paper_brief returned empty output")

        brief_text = _truncate_text(render_brief_text(brief), settings.paper_brief_max_chars)

        record["subject"] = str(brief.get("subject") or record.get("title") or "").strip() or record.get("subject")
        if isinstance(brief.get("keywords"), list):
            record["keywords"] = brief.get("keywords")
        record["paper_brief"] = brief
        record["background"] = brief_text
        if not settings.keep_paper_text_in_memory:
            record.pop("text", None)
            record.pop("paper_text", None)
            record.pop("content", None)
            if isinstance(record.get("pages"), list) and len(record.get("pages") or []) > 0:
                record.pop("pages", None)

        # Update episode_seed to match the compact brief.
        try:
            subj = str(record.get("subject") or "").strip() or f"Paper executable seed {prob_id}"
            kws = record.get("keywords") if isinstance(record.get("keywords"), list) else None
            keywords = [str(k) for k in (kws or []) if isinstance(k, str) and k.strip()]
            if not keywords:
                keywords = [f"executable", f"paper", f"paper_id={prob_id}"]
            mem = KnownTree.update_episode_seed(mem, subject=subj, keywords=keywords)
            state.memory = mem
            save_state(state)
        except Exception:
            pass

        role_outputs["paper_brief"] = {
            "subject": record.get("subject"),
            "keywords": record.get("keywords"),
            "brief_text": brief_text,
            "version": version,
            "prompt_path": str(prompt_path),
            "keep_paper_text_in_memory": settings.keep_paper_text_in_memory,
        }

    # Keep a small on-disk seed snapshot for debugging.
    try:
        (step_dir / "executable_seed.json").write_text(
            json.dumps(mem.get("executable_seed") or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Optional: seed-stage ExecutableExtract (produce task_sketch for llm-generated extend).
    try:
        extend_conf = (agent_conf.get("operators") or {}).get("extend") or {}
        code_source = str(extend_conf.get("code_source") or "llm-generated").strip().lower()
        should_extract = settings.extract_enabled and code_source in {"llm", "llm-generated", "generated"}
    except Exception:
        should_extract = False

    if should_extract:
        seed = mem.get("executable_seed") if isinstance(mem, dict) else None
        if isinstance(seed, dict) and not str(seed.get("task_sketch") or "").strip():
            try:
                from agenqa.skills.executable_extract import ExecutableExtractConfig, ExecutableExtractInput, ExecutableExtractRunner
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"executable_extract import failed: {exc}") from exc

            extract_gen = settings.extract_generator or settings.init_generator or {}
            if not extract_gen:
                raise RuntimeError("ExecutableExtract requires init.generator or init.extract.generator.")

            director_notes = build_director_notes(state, include_solver_feedback=False) or ""
            background = _extract_scicode_problem_background(record)

            extract_runner = ExecutableExtractRunner(
                ExecutableExtractConfig(
                    generator=extract_gen,
                    prompt_path=settings.extract_prompt_path,
                    lang=(agent_conf.get("agent") or {}).get("lang"),
                )
            )
            extract_out = extract_runner.run_one(
                ExecutableExtractInput(
                    director_notes=director_notes,
                    paper_background=background,
                    problem_description=background,
                    dependencies_whitelist=settings.dependencies_whitelist,
                ),
                snapshot_dir=step_dir / "subruns_raw" / "executable_extract",
                unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
            )
            if not extract_out.executable_suitable:
                raise RuntimeError("executable_extract marked as not suitable")

            # Persist extract hints into executable_seed.
            seed["task_sketch"] = extract_out.task_sketch
            seed["extract_notes"] = extract_out.notes
            seed["extract_initial_sub_steps"] = [s.to_dict() for s in extract_out.initial_sub_steps]
            mem["executable_seed"] = seed
            state.memory = mem
            save_state(state)

            role_outputs["executable_extract"] = {
                "executable_suitable": extract_out.executable_suitable,
                "notes": extract_out.notes,
                "task_sketch": extract_out.task_sketch,
                "initial_sub_steps": [s.to_dict() for s in extract_out.initial_sub_steps],
                "estimated_difficulty": extract_out.estimated_difficulty,
            }

    if output_manager:
        return NodeResult(
            state=state,
            step_idx=step_idx,
            round_idx=round_idx,
            outputs={},
            role_outputs=role_outputs,
            step_dir=step_dir,
        )
    return state


__all__ = [
    "run_executable_init",
    "ExecutableInitSettings",
]
