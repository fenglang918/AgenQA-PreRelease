from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import sys

from agenqa.domain.executable_schema import (
    ExecutableRecord,
    ExecutableSubStep,
    ExecutableTestCase,
    FIELD_INPUTS_ARTIFACT_RELPATH,
    FIELD_INPUTS_SHA256,
)
from agenqa.domain.known_tree import KnownTree
from agenqa.domain.node_result import NodeResult
from agenqa.graph.output_manager import OutputContext, compute_step_dir
from agenqa.graph.state import AgentState
from agenqa.memory.store import (
    dump_director_decision_for_step,
    dump_edge_executable_for_step,
    dump_path_executable_for_step,
    save_state,
)
from agenqa.nodes.executable_e2e_oracle import (
    build_e2e_sub_step,
    build_reference_solve_wrapper_code,
    parse_e2e_spec,
)
from agenqa.nodes.executable_path_fold import apply_executable_path_fold
from agenqa.nodes.utils import build_director_notes

logger = logging.getLogger(__name__)


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
    # Paper/text fallback
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


def _looks_like_python_import_block(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("import ") or s.startswith("from "):
            return True
    return False


@dataclass
class ExecutableExtendSettings:
    max_background_step: Optional[int]
    code_source: str
    dependencies_whitelist: str
    extract_prompt_path: Path
    draft_prompt_path: Path
    derive_prompt_path: Path
    generator: Dict[str, Any]
    extract_generator: Optional[Dict[str, Any]]
    draft_generator: Optional[Dict[str, Any]]
    derive_generator: Optional[Dict[str, Any]]
    extract_enabled: bool
    max_inputs: int
    max_case_chars: int


def _resolve_settings(agent_conf: Dict[str, Any]) -> ExecutableExtendSettings:
    op_conf = (agent_conf.get("operators") or {}).get("extend") or {}
    try:
        max_background_step = int(op_conf.get("max_background_step", 2))
    except Exception:
        max_background_step = op_conf.get("max_background_step", 2)
    code_source = str(op_conf.get("code_source") or "llm-generated").strip().lower()

    whitelist = op_conf.get("dependencies_whitelist")
    if isinstance(whitelist, list):
        dependencies_whitelist = "\n".join(str(item) for item in whitelist if str(item).strip())
    elif isinstance(whitelist, str):
        dependencies_whitelist = whitelist.strip()
    else:
        dependencies_whitelist = "\n".join(["numpy", "scipy", "sympy", "h5py"])

    extract_prompt_path = Path(op_conf.get("extract_prompt_path") or "src/agenqa/prompts/executable_extract.prompt")
    draft_prompt_path = Path(op_conf.get("draft_prompt_path") or "src/agenqa/prompts/executable_draft_step.prompt")
    derive_prompt_path = Path(op_conf.get("derive_prompt_path") or "src/agenqa/prompts/executable_test_inputs.prompt")

    generator = op_conf.get("generator") or {}
    extract_generator = op_conf.get("extract_generator")
    draft_generator = op_conf.get("draft_generator")
    derive_generator = op_conf.get("derive_generator")
    extract_enabled = bool(op_conf.get("extract_enabled", True))
    max_inputs = int(op_conf.get("max_inputs", 8))
    max_case_chars = int(op_conf.get("max_case_chars", 4000))

    return ExecutableExtendSettings(
        max_background_step=max_background_step,
        code_source=code_source,
        dependencies_whitelist=dependencies_whitelist,
        extract_prompt_path=extract_prompt_path,
        draft_prompt_path=draft_prompt_path,
        derive_prompt_path=derive_prompt_path,
        generator=generator if isinstance(generator, dict) else {},
        extract_generator=extract_generator if isinstance(extract_generator, dict) else None,
        draft_generator=draft_generator if isinstance(draft_generator, dict) else None,
        derive_generator=derive_generator if isinstance(derive_generator, dict) else None,
        extract_enabled=extract_enabled,
        max_inputs=max_inputs,
        max_case_chars=max_case_chars,
    )


def _extract_step_text(sub_step: Dict[str, Any]) -> str:
    # Prefer SciCode's step_description_prompt field.
    val = sub_step.get("step_description_prompt")
    if isinstance(val, str) and val.strip():
        return val.strip()
    val = sub_step.get("step_description")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return ""


def _build_executable_record_from_scicode_seed(
    seed: Dict[str, Any],
    *,
    step_idx: int,
    with_background: bool,
) -> ExecutableRecord:
    record = seed.get("record") if isinstance(seed, dict) else None
    if not isinstance(record, dict):
        raise ValueError("executable_seed.record missing or invalid")

    prob_id = str(record.get("problem_id") or seed.get("problem_id") or "unknown")
    sub_steps = record.get("sub_steps") or []
    if not isinstance(sub_steps, list) or not sub_steps:
        raise ValueError(f"SciCode record has empty sub_steps (problem_id={prob_id})")

    idx = step_idx - 1
    if idx < 0 or idx >= len(sub_steps):
        raise IndexError(f"step_idx={step_idx} out of range for SciCode problem_id={prob_id} (len={len(sub_steps)})")

    visible: List[ExecutableSubStep] = []
    per_step_golden: Dict[str, str] = {}
    for i in range(0, idx + 1):
        step = sub_steps[i]
        if not isinstance(step, dict):
            continue
        step_number = str(step.get("step_number") or "")
        visible.append(
            ExecutableSubStep(
                step_number=step_number,
                step_description=_extract_step_text(step),
                function_header=str(step.get("function_header") or ""),
                return_line=str(step.get("return_line") or ""),
                step_background=(str(step.get("step_background") or "") if with_background else None),
            )
        )
        gt = step.get("ground_truth_code")
        if isinstance(gt, str) and step_number:
            per_step_golden[step_number] = gt

    tail_step = sub_steps[idx]
    tail_step_number = str(tail_step.get("step_number") or "")
    raw_tests = tail_step.get("test_cases") or []
    tests: List[ExecutableTestCase] = []
    if isinstance(raw_tests, list):
        for t_idx, t in enumerate(raw_tests):
            if not isinstance(t, str):
                continue
            tests.append(
                ExecutableTestCase(
                    test_id=f"{tail_step_number}_test_{t_idx}",
                    test_code=t,
                    derived_from_golden=False,
                )
            )

    deps = record.get("required_dependencies", "")
    if isinstance(deps, list):
        deps = "\n".join(str(item) for item in deps)
    elif deps is None:
        deps = ""

    background = _extract_scicode_problem_background(record)

    out = ExecutableRecord(
        paper_id=None,
        problem_id=prob_id,
        qa_idx=step_idx,
        background=background,
        required_dependencies=str(deps),
        sub_steps=visible,
        test_cases={tail_step_number: tests} if tail_step_number else {},
        golden_code=None,
        per_step_golden=per_step_golden,
        estimated_difficulty=(str(record.get("estimated_difficulty")) if isinstance(record.get("estimated_difficulty"), str) else None),
        subject=(str(record.get("subject")) if isinstance(record.get("subject"), str) else None),
        source="scicode",
    )
    return out


def _clone_executable_record(rec: ExecutableRecord) -> ExecutableRecord:
    return ExecutableRecord(
        paper_id=rec.paper_id,
        problem_id=rec.problem_id,
        qa_idx=rec.qa_idx,
        background=rec.background,
        required_dependencies=rec.required_dependencies,
        sub_steps=list(rec.sub_steps),
        test_cases={k: list(v) for k, v in (rec.test_cases or {}).items()},
        inputs_artifacts=dict(rec.inputs_artifacts or {}),
        golden_code=rec.golden_code,
        per_step_golden=dict(rec.per_step_golden or {}),
        estimated_difficulty=rec.estimated_difficulty,
        subject=rec.subject,
        source=rec.source,
        path_question_scaffolded=rec.path_question_scaffolded,
        path_question_direct=rec.path_question_direct,
        path_fold_notes=rec.path_fold_notes,
    )


def _build_base_record_from_seed(seed: Dict[str, Any]) -> ExecutableRecord:
    record = seed.get("record") if isinstance(seed, dict) else None
    prob_id = str(seed.get("problem_id") or "unknown")
    paper_id = seed.get("paper_id")
    background = ""
    required_dependencies = ""
    estimated_difficulty = None
    subject = None
    if isinstance(record, dict):
        prob_id = str(record.get("problem_id") or prob_id)
        paper_id = record.get("paper_id") or paper_id
        background = _extract_scicode_problem_background(record)
        deps = record.get("required_dependencies", "")
        if isinstance(deps, list):
            required_dependencies = "\n".join(str(item) for item in deps)
        elif isinstance(deps, str):
            required_dependencies = deps
        if isinstance(record.get("estimated_difficulty"), str):
            estimated_difficulty = record.get("estimated_difficulty")
        if isinstance(record.get("subject"), str):
            subject = record.get("subject")
    return ExecutableRecord(
        paper_id=str(paper_id) if paper_id else None,
        problem_id=prob_id,
        qa_idx=None,
        background=background,
        required_dependencies=required_dependencies,
        sub_steps=[],
        test_cases={},
        inputs_artifacts={},
        golden_code=None,
        per_step_golden={},
        estimated_difficulty=estimated_difficulty,
        subject=subject,
        source="llm_generated",
    )


def run_executable_extend(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    """ExecutableExtend: append the next executable step based on the executable seed."""
    settings = _resolve_settings(agent_conf)

    mem = getattr(state, "memory", None)
    if not isinstance(mem, dict):
        raise RuntimeError("state.memory is missing (expected KnownTree v2 dict)")
    seed = mem.get("executable_seed")
    if not isinstance(seed, dict):
        raise RuntimeError("executable_seed is missing; run init first")

    with_background = bool(seed.get("with_background", False))

    # Next step index: mirror semantic extend (step starts at 1; step 0 is init).
    try:
        step_idx = int(state.step) + 1
    except Exception:
        step_idx = 1
    round_idx = state.current_round_index()

    if output_manager:
        ctx: OutputContext = output_manager.begin("extend", step_idx, round_idx)
        step_dir = ctx.step_dir
        ctx.dump_director_decision(state, step_idx)
    else:
        step_dir = compute_step_dir(state.artifacts_dir, "extend", step_idx, round_idx)
        step_dir.mkdir(parents=True, exist_ok=True)
        dump_director_decision_for_step(state, step_dir, step_idx)

    role_outputs: Dict[str, Any] = {}
    code_source = str(settings.code_source or "llm-generated").strip().lower()

    if code_source in {"llm", "llm-generated", "generated"}:
        latest = None
        try:
            latest = state.latest_executable_record()
        except Exception:
            latest = None
        rec = _clone_executable_record(latest) if isinstance(latest, ExecutableRecord) else _build_base_record_from_seed(seed)
        rec.qa_idx = step_idx

        director_notes = build_director_notes(state, include_solver_feedback=False) or ""
        expected_primary_fact_id = KnownTree.key_fact_id_for_step(state.memory, step_idx - 1) if step_idx >= 2 else ""

        task_sketch = str(seed.get("task_sketch") or "")
        if (not task_sketch) and settings.extract_enabled:
            try:
                from agenqa.skills.executable_extract import ExecutableExtractConfig, ExecutableExtractInput, ExecutableExtractRunner
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"executable_extract import failed: {exc}") from exc

            extract_gen = settings.extract_generator or settings.generator
            if not extract_gen:
                director_conf = agent_conf.get("director") or {}
                extract_gen = director_conf.get("generator") or {
                    "service_type": "private_endpoint",
                    "service_id": director_conf.get("service_id"),
                }
            extract_runner = ExecutableExtractRunner(
                ExecutableExtractConfig(
                    generator=extract_gen,
                    prompt_path=settings.extract_prompt_path,
                    lang=(agent_conf.get("agent") or {}).get("lang"),
                )
            )
            extract_in = ExecutableExtractInput(
                director_notes=director_notes,
                paper_background=rec.background,
                problem_description=rec.background,
                dependencies_whitelist=settings.dependencies_whitelist,
            )
            extract_out = extract_runner.run_one(
                extract_in,
                snapshot_dir=step_dir / "subruns_raw" / "executable_extract",
                unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
            )
            if not extract_out.executable_suitable:
                raise RuntimeError("executable_extract marked as not suitable")
            task_sketch = extract_out.task_sketch
            if extract_out.estimated_difficulty and not rec.estimated_difficulty:
                rec.estimated_difficulty = extract_out.estimated_difficulty
            # persist extract hints to memory seed
            seed["task_sketch"] = task_sketch
            seed["extract_notes"] = extract_out.notes
            seed["extract_initial_sub_steps"] = [s.to_dict() for s in extract_out.initial_sub_steps]
            mem["executable_seed"] = seed
            state.memory = mem
            role_outputs["executable_extract"] = {
                "executable_suitable": extract_out.executable_suitable,
                "notes": extract_out.notes,
                "task_sketch": extract_out.task_sketch,
                "initial_sub_steps": [s.to_dict() for s in extract_out.initial_sub_steps],
                "estimated_difficulty": extract_out.estimated_difficulty,
            }

        try:
            from agenqa.skills.executable_draft import ExecutableDraftConfig, ExecutableDraftInput, ExecutableDraftRunner
            from agenqa.skills.executable_step_cert_builder import (
                ExecutableStepCertBuilderConfig,
                ExecutableStepCertBuilderInput,
                ExecutableStepCertBuilderRunner,
            )
            from agenqa.skills.test_derive import TestDeriveConfig, TestDeriveInput, TestDeriveRunner
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"executable draft/derive import failed: {exc}") from exc

        draft_gen = settings.draft_generator or settings.generator
        if not draft_gen:
            director_conf = agent_conf.get("director") or {}
            draft_gen = director_conf.get("generator") or {
                "service_type": "private_endpoint",
                "service_id": director_conf.get("service_id"),
            }
        op_conf = (agent_conf.get("operators") or {}).get("extend") or {}
        struct_gen = op_conf.get("struct_generator")
        if not isinstance(struct_gen, dict) or not struct_gen:
            struct_gen = settings.generator or draft_gen
        draft_runner = ExecutableDraftRunner(
            ExecutableDraftConfig(
                generator=draft_gen,
                format_generator=op_conf.get("format_generator") if isinstance(op_conf.get("format_generator"), dict) else struct_gen,
                prompt_path=settings.draft_prompt_path,
                lang=(agent_conf.get("agent") or {}).get("lang"),
            )
        )
        draft_out = draft_runner.run_one(
            ExecutableDraftInput(
                step=step_idx,
                director_notes=director_notes,
                task_sketch=task_sketch,
                background=rec.background,
                prev_sub_steps=list(rec.sub_steps),
                dependencies_whitelist=settings.dependencies_whitelist,
                expected_primary_fact_id=expected_primary_fact_id,
            ),
            snapshot_dir=step_dir / "subruns_raw" / "executable_draft_step",
            unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
        )

        step_number = draft_out.step_number or str(step_idx)
        draft_out.sub_step.step_number = step_number
        rec.sub_steps.append(draft_out.sub_step)
        if isinstance(draft_out.dependencies, str) and draft_out.dependencies.strip():
            deps_text = draft_out.dependencies.strip()
            # `required_dependencies` is executed as Python imports by TestDerive; do not overwrite with package names.
            if _looks_like_python_import_block(deps_text):
                rec.required_dependencies = deps_text
        rec.per_step_golden[step_number] = draft_out.golden_step_code
        rec.source = "llm_generated"
        role_outputs["executable_draft_step"] = {
            "step_number": step_number,
            "sub_step": draft_out.sub_step.to_dict(),
            "golden_step_code": draft_out.golden_step_code,
            "dependencies": draft_out.dependencies,
            "required_fact_ids": list(draft_out.required_fact_ids or []),
            "primary_required_fact_id": draft_out.primary_required_fact_id,
            "reuse_plan": list(draft_out.reuse_plan or []),
        }

        # --- ExecutableStepCertBuilder: materialize contract/premise/fact into KnownTree memory (append) ---
        memory_json = KnownTree.to_json(KnownTree.build_step_cert_view(state.memory, step_idx))
        op_conf_step_cert = (agent_conf.get("operators") or {}).get("extend") or {}
        cert_gen = (
            op_conf_step_cert.get("step_cert_generator")
            or op_conf_step_cert.get("path_fold_generator")
            or op_conf_step_cert.get("struct_generator")
            or settings.generator
            or draft_gen
        )
        if not isinstance(cert_gen, dict) or not cert_gen:
            director_conf = agent_conf.get("director") or {}
            cert_gen = director_conf.get("generator") or {
                "service_type": "private_endpoint",
                "service_id": director_conf.get("service_id"),
            }
        cert_runner = ExecutableStepCertBuilderRunner(
            ExecutableStepCertBuilderConfig(
                generator=cert_gen,
                format_generator=op_conf_step_cert.get("format_generator") if isinstance(op_conf_step_cert.get("format_generator"), dict) else struct_gen,
                prompt_path=Path(op_conf_step_cert.get("step_cert_prompt_path") or "src/agenqa/prompts/executable_step_cert_builder.prompt"),
                lang=(agent_conf.get("agent") or {}).get("lang"),
            )
        )
        cert_out = cert_runner.run_one(
            ExecutableStepCertBuilderInput(
                step=step_idx,
                director_notes=director_notes,
                task_sketch=task_sketch,
                background=rec.background,
                prev_sub_steps=list(rec.sub_steps[:-1]),
                tail_sub_step=rec.sub_steps[-1],
                golden_step_code=draft_out.golden_step_code,
                memory_json=memory_json,
                expected_primary_fact_id=expected_primary_fact_id or "",
                observed_required_fact_ids=list(draft_out.required_fact_ids or []),
                observed_primary_required_fact_id=str(draft_out.primary_required_fact_id or ""),
            ),
            snapshot_dir=step_dir / "subruns_raw" / "executable_step_cert_builder",
            unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
        )
        # Deterministic metadata normalization (no semantic change): tag this cert kind.
        try:
            cert_out.step_cert.setdefault("kind", "executable_chain_cert")
        except Exception:
            pass
        # Deterministic protocol enforcement: step_cert.step must match the current step.
        try:
            cert_out.step_cert["step"] = int(step_idx)
        except Exception:
            pass
        raw_ref = f"{step_dir.relative_to(state.artifacts_dir).as_posix()}/subruns_raw/executable_step_cert_builder/"
        provenance = {"role": "executable_step_cert_builder", "raw_ref": raw_ref}
        overwrite_step = bool(op_conf_step_cert.get("overwrite_step", False) or op_conf_step_cert.get("overwrite_existing_step", False))
        if not overwrite_step:
            step_i = int(step_idx)
            mem = state.memory if isinstance(state.memory, dict) else {}
            for bank_key in ("premise_bank", "fact_bank"):
                for entry in mem.get(bank_key, []) or []:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        entry_step = int(entry.get("source_step", -1) or -1)
                    except Exception:
                        continue
                    if entry_step == step_i:
                        overwrite_step = True
                        break
                if overwrite_step:
                    break
            if not overwrite_step:
                for entry in mem.get("step_certs", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        entry_step = int(entry.get("step", -1) or -1)
                    except Exception:
                        continue
                    if entry_step == step_i:
                        overwrite_step = True
                        break
        if overwrite_step:
            logger.warning("Detected existing KnownTree entries for step=%s; overwriting step memory to avoid ID conflicts.", step_idx)
        state.memory = KnownTree.apply_step_update(
            state.memory,
            step=step_idx,
            premise_delta=cert_out.premise_delta,
            fact_delta=cert_out.fact_delta,
            step_cert=cert_out.step_cert,
            key_fact_id=cert_out.key_fact_id,
            overwrite_step=overwrite_step,
            provenance=provenance,
        )
        save_state(state)
        role_outputs["executable_step_cert_builder"] = {
            "premise_delta": cert_out.premise_delta,
            "fact_delta": cert_out.fact_delta,
            "step_cert": cert_out.step_cert,
            "key_fact_id": cert_out.key_fact_id,
        }

        eval_conf = agent_conf.get("executable_eval") or {}
        derive_gen = settings.derive_generator or settings.generator or draft_gen
        derive_runner = TestDeriveRunner(
            TestDeriveConfig(
                generator=derive_gen,
                prompt_path=settings.derive_prompt_path,
                lang=(agent_conf.get("agent") or {}).get("lang"),
                step_timeout=float(eval_conf.get("step_timeout", 120)),
                memory_mb=int(eval_conf.get("memory_mb", 16384)),
                temp_dir=str(eval_conf.get("temp_dir") or str(Path("infra/code_verifier/.tmp").resolve())),
                python_bin=str(eval_conf.get("python_bin") or sys.executable),
                max_inputs=settings.max_inputs,
                max_case_chars=settings.max_case_chars,
            )
        )
        derive_out = derive_runner.run_one(
            TestDeriveInput(
                background=rec.background,
                required_dependencies=rec.required_dependencies,
                sub_steps=list(rec.sub_steps),
                per_step_golden=dict(rec.per_step_golden),
                step_number=step_number,
                golden_step_code=draft_out.golden_step_code,
            ),
            step_dir=step_dir,
            snapshot_dir=step_dir / "subruns_raw" / "executable_test_inputs",
            unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
        )
        rec.test_cases[step_number] = derive_out.test_cases_for_step
        if derive_out.inputs_artifact_relpath and derive_out.inputs_sha256:
            rec.inputs_artifacts[step_number] = {
                FIELD_INPUTS_ARTIFACT_RELPATH: derive_out.inputs_artifact_relpath,
                FIELD_INPUTS_SHA256: derive_out.inputs_sha256,
            }
        role_outputs["executable_test_inputs"] = {
            "inputs_artifact_relpath": derive_out.inputs_artifact_relpath,
            "inputs_sha256": derive_out.inputs_sha256,
            "test_cases_for_step": [t.to_dict() for t in derive_out.test_cases_for_step],
        }

        enable_e2e_oracle = bool(op_conf_step_cert.get("enable_e2e_oracle", False) or op_conf_step_cert.get("e2e_oracle_enabled", False))
        if enable_e2e_oracle:
            try:
                raw_spec = (cert_out.step_cert or {}).get("e2e_spec")
                spec = parse_e2e_spec(raw_spec)
                e2e_step = build_e2e_sub_step(record=rec, spec=spec)
                wrapper_code = build_reference_solve_wrapper_code(spec=spec, record=rec)
                derive_out_e2e = derive_runner.run_one(
                    TestDeriveInput(
                        background=rec.background,
                        required_dependencies=rec.required_dependencies,
                        sub_steps=list(rec.sub_steps) + [e2e_step],
                        per_step_golden=dict(rec.per_step_golden),
                        step_number="e2e",
                        golden_step_code=wrapper_code,
                    ),
                    step_dir=step_dir,
                    snapshot_dir=step_dir / "subruns_raw" / "executable_test_inputs_e2e",
                    unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
                    derive_dirname="derive_e2e",
                )
                rec.test_cases["e2e"] = derive_out_e2e.test_cases_for_step
                if derive_out_e2e.inputs_artifact_relpath and derive_out_e2e.inputs_sha256:
                    rec.inputs_artifacts["e2e"] = {
                        FIELD_INPUTS_ARTIFACT_RELPATH: derive_out_e2e.inputs_artifact_relpath,
                        FIELD_INPUTS_SHA256: derive_out_e2e.inputs_sha256,
                    }
                role_outputs["executable_test_inputs_e2e"] = {
                    "inputs_artifact_relpath": derive_out_e2e.inputs_artifact_relpath,
                    "inputs_sha256": derive_out_e2e.inputs_sha256,
                    "test_cases_for_step": [t.to_dict() for t in derive_out_e2e.test_cases_for_step],
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("executable_e2e_oracle derive failed; skip e2e. error=%s", str(exc))
                role_outputs["executable_test_inputs_e2e_error"] = {"error": str(exc)}
    elif code_source in {"scicode", "ground-truth", "groundtruth", "gt", "golden"}:
        seed_source = str(seed.get("source") or "").strip().lower()
        if seed_source not in {"scicode", ""}:
            raise ValueError(
                f"Unsupported code_source={code_source!r} for extend when executable_seed.source={seed_source!r} "
                "(supported: llm-generated; for scicode seeds: scicode/ground-truth)"
            )
        rec = _build_executable_record_from_scicode_seed(seed, step_idx=step_idx, with_background=with_background)
    else:
        raise ValueError(f"Unsupported code_source={code_source!r} for extend (supported: llm-generated, scicode)")

    fold_payload = apply_executable_path_fold(agent_conf, state, rec, step_dir=step_dir, op_name="extend")
    state.append_record(rec)
    save_state(state)

    if not output_manager:
        dump_edge_executable_for_step(state, step_dir)
        dump_path_executable_for_step(state, step_dir)

    if output_manager:
        return NodeResult(
            state=state,
            step_idx=step_idx,
            round_idx=round_idx,
            outputs={},
            role_outputs={**({"executable_path_fold": fold_payload} if fold_payload else {}), **role_outputs},
            step_dir=step_dir,
        )
    return state


__all__ = [
    "run_executable_extend",
]
