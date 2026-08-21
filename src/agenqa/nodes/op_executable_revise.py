from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import logging
import json

from agenqa.domain.node_result import NodeResult
from agenqa.graph.output_manager import OutputContext, compute_step_dir
from agenqa.graph.state import AgentState
from agenqa.nodes.utils import build_director_notes
from agenqa.domain.executable_schema import FIELD_INPUTS_ARTIFACT_RELPATH, FIELD_INPUTS_SHA256
from agenqa.domain.known_tree import KnownTree
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

logger = logging.getLogger(__name__)


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


def _normalize_whitelist(val: Any) -> str:
    if isinstance(val, list):
        return "\n".join(str(item) for item in val if str(item).strip())
    if isinstance(val, str):
        return val.strip()
    return "\n".join(["numpy", "scipy", "sympy", "h5py"])


def _resolve_generator(agent_conf: Dict[str, Any], op_conf: Dict[str, Any]) -> Dict[str, Any]:
    gen = op_conf.get("generator") or {}
    if isinstance(gen, dict) and gen:
        return gen
    director_conf = agent_conf.get("director") or {}
    gen = director_conf.get("generator") or {
        "service_type": "private_endpoint",
        "service_id": director_conf.get("service_id"),
    }
    return gen if isinstance(gen, dict) else {}


def run_executable_revise(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    """ExecutableRevise: revise the latest executable step (step-wise, SciCode-like)."""
    mem = getattr(state, "memory", None)
    if not isinstance(mem, dict):
        raise RuntimeError("state.memory is missing (expected KnownTree v2 dict)")
    seed = mem.get("executable_seed")
    if not isinstance(seed, dict):
        raise RuntimeError("executable_seed is missing; run init first")

    with_background = bool(seed.get("with_background", False))

    try:
        step_idx = int(state.step)
    except Exception:
        step_idx = 0
    round_idx = state.current_round_index()

    if output_manager:
        ctx: OutputContext = output_manager.begin("revise", step_idx, round_idx)
        step_dir = ctx.step_dir
        ctx.dump_director_decision(state, step_idx)
    else:
        step_dir = compute_step_dir(state.artifacts_dir, "revise", step_idx, round_idx)
        step_dir.mkdir(parents=True, exist_ok=True)
        dump_director_decision_for_step(state, step_dir, step_idx)

    if not state.latest_executable_record():
        raise RuntimeError("executable history is empty; nothing to revise")

    op_conf = (agent_conf.get("operators") or {}).get("revise") or {}
    code_source = (
        str(op_conf.get("code_source") or "").strip().lower()
        or str(((agent_conf.get("operators") or {}).get("extend") or {}).get("code_source") or "llm-generated").strip().lower()
    )

    role_outputs: Dict[str, Any] = {}

    if code_source in {"llm", "llm-generated", "generated"}:
        latest = state.latest_executable_record()
        if not latest:
            raise RuntimeError("executable history is empty; nothing to revise")
        rec = latest
        if not rec.sub_steps:
            raise RuntimeError("executable_record.sub_steps is empty; nothing to revise")
        tail = rec.sub_steps[-1]
        step_number = tail.step_number or str(step_idx)

        dependencies_whitelist = _normalize_whitelist(op_conf.get("dependencies_whitelist"))
        director_notes = build_director_notes(state, include_solver_feedback=True) or ""
        expected_primary_fact_id = KnownTree.key_fact_id_for_step(state.memory, step_idx - 1) if step_idx >= 2 else ""
        eval_error = ""
        try:
            res = state.get_latest_solver(step_idx, "edge", "strong") or state.get_latest_solver(step_idx, "edge", "medium")
            eval_error = str(getattr(res, "correctness_feedback", None) or "") if res else ""
        except Exception:
            eval_error = ""

        tail_payload = {
            "step_number": step_number,
            "sub_step": tail.to_dict(),
            "required_dependencies": rec.required_dependencies,
            "inputs_artifacts": (rec.inputs_artifacts or {}).get(step_number),
            "test_cases_count": len((rec.test_cases or {}).get(step_number) or []),
        }

        try:
            from agenqa.skills.executable_diagnose import (
                ExecutableDiagnoseConfig,
                ExecutableDiagnoseInput,
                ExecutableDiagnoseRunner,
            )
            from agenqa.skills.executable_revise_step import (
                ExecutableReviseStepConfig,
                ExecutableReviseStepInput,
                ExecutableReviseStepRunner,
            )
            from agenqa.skills.executable_step_cert_builder import (
                ExecutableStepCertBuilderConfig,
                ExecutableStepCertBuilderInput,
                ExecutableStepCertBuilderRunner,
            )
            from agenqa.skills.test_derive import TestDeriveConfig, TestDeriveInput, TestDeriveRunner
            import sys
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"executable revise imports failed: {exc}") from exc

        generator = _resolve_generator(agent_conf, op_conf)
        struct_gen = op_conf.get("struct_generator")
        if not isinstance(struct_gen, dict) or not struct_gen:
            struct_gen = generator
        lang = (agent_conf.get("agent") or {}).get("lang")

        diagnose_prompt_path = Path(op_conf.get("diagnose_prompt_path") or "src/agenqa/prompts/executable_diagnose.prompt")
        revise_prompt_path = Path(op_conf.get("revise_prompt_path") or "src/agenqa/prompts/executable_revise_step.prompt")
        derive_prompt_path = Path(op_conf.get("derive_prompt_path") or "src/agenqa/prompts/executable_test_inputs.prompt")

        diag_runner = ExecutableDiagnoseRunner(
            ExecutableDiagnoseConfig(
                generator=generator,
                prompt_path=diagnose_prompt_path,
                lang=lang,
            )
        )
        diag_out = diag_runner.run_one(
            ExecutableDiagnoseInput(
                step=step_idx,
                director_notes=director_notes,
                executable_tail_json=json.dumps(tail_payload, ensure_ascii=False),
                eval_error=eval_error,
            ),
            snapshot_dir=step_dir / "subruns_raw" / "executable_diagnose",
            unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
        )
        role_outputs["executable_diagnose"] = {
            "issues": list(diag_out.issues),
            "fix_suggestions": list(diag_out.fix_suggestions),
            "diagnosis": diag_out.diagnosis,
        }
        director_notes_for_revise = director_notes
        if isinstance(diag_out.diagnosis, str) and diag_out.diagnosis.strip():
            director_notes_for_revise = f"{director_notes}\n\n[Diagnose]\n{diag_out.diagnosis}".strip()

        task_sketch = str(seed.get("task_sketch") or "")
        revise_runner = ExecutableReviseStepRunner(
            ExecutableReviseStepConfig(
                generator=generator,
                prompt_path=revise_prompt_path,
                lang=lang,
            )
        )
        revise_out = revise_runner.run_one(
            ExecutableReviseStepInput(
                step=step_idx,
                director_notes=director_notes_for_revise,
                task_sketch=task_sketch,
                background=str(rec.background or ""),
                prev_sub_steps=list(rec.sub_steps[:-1]),
                current_sub_step=tail,
                current_golden_step_code=str((rec.per_step_golden or {}).get(step_number) or ""),
                diagnose_json=json.dumps(role_outputs["executable_diagnose"], ensure_ascii=False),
                dependencies_whitelist=dependencies_whitelist,
            ),
            snapshot_dir=step_dir / "subruns_raw" / "executable_revise_step",
            unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
        )

        revise_step_number = revise_out.step_number or step_number
        revise_out.sub_step.step_number = revise_step_number
        if revise_step_number != step_number:
            revise_step_number = step_number
            revise_out.sub_step.step_number = step_number

        rec.sub_steps[-1] = revise_out.sub_step
        if isinstance(revise_out.dependencies, str) and revise_out.dependencies.strip():
            deps_text = revise_out.dependencies.strip()
            # `required_dependencies` is executed as Python imports by verifiers; do not overwrite with package names.
            if _looks_like_python_import_block(deps_text):
                rec.required_dependencies = deps_text
        rec.per_step_golden[step_number] = revise_out.golden_step_code
        rec.source = "llm_generated"

        role_outputs["executable_revise_step"] = {
            "step_number": step_number,
            "sub_step": revise_out.sub_step.to_dict(),
            "golden_step_code": revise_out.golden_step_code,
            "dependencies": revise_out.dependencies,
        }

        # --- ExecutableStepCertBuilder: overwrite chain cert for this step in KnownTree memory ---
        memory_json = KnownTree.to_json(KnownTree.build_step_cert_view(state.memory, step_idx))
        cert_runner = ExecutableStepCertBuilderRunner(
            ExecutableStepCertBuilderConfig(
                generator=struct_gen,
                format_generator=op_conf.get("format_generator") if isinstance(op_conf.get("format_generator"), dict) else struct_gen,
                prompt_path=Path(op_conf.get("step_cert_prompt_path") or "src/agenqa/prompts/executable_step_cert_builder.prompt"),
                lang=lang,
            )
        )
        cert_out = cert_runner.run_one(
            ExecutableStepCertBuilderInput(
                step=step_idx,
                director_notes=director_notes_for_revise,
                task_sketch=task_sketch,
                background=str(rec.background or ""),
                prev_sub_steps=list(rec.sub_steps[:-1]),
                tail_sub_step=rec.sub_steps[-1],
                golden_step_code=revise_out.golden_step_code,
                memory_json=memory_json,
                expected_primary_fact_id=expected_primary_fact_id or "",
                observed_required_fact_ids=[],
                observed_primary_required_fact_id="",
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
        save_state(state)
        role_outputs["executable_step_cert_builder"] = {
            "premise_delta": cert_out.premise_delta,
            "fact_delta": cert_out.fact_delta,
            "step_cert": cert_out.step_cert,
            "key_fact_id": cert_out.key_fact_id,
        }

        eval_conf = agent_conf.get("executable_eval") or {}
        derive_runner = TestDeriveRunner(
            TestDeriveConfig(
                generator=generator,
                prompt_path=derive_prompt_path,
                lang=lang,
                step_timeout=float(eval_conf.get("step_timeout", 120)),
                memory_mb=int(eval_conf.get("memory_mb", 16384)),
                temp_dir=str(eval_conf.get("temp_dir") or str(Path("infra/code_verifier/.tmp").resolve())),
                python_bin=str(eval_conf.get("python_bin") or sys.executable),
                max_inputs=int(op_conf.get("max_inputs", 8)),
                max_case_chars=int(op_conf.get("max_case_chars", 4000)),
            )
        )
        derive_out = derive_runner.run_one(
            TestDeriveInput(
                background=str(rec.background or ""),
                required_dependencies=str(rec.required_dependencies or ""),
                sub_steps=list(rec.sub_steps),
                per_step_golden=dict(rec.per_step_golden or {}),
                step_number=step_number,
                golden_step_code=revise_out.golden_step_code,
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

        enable_e2e_oracle = bool(op_conf.get("enable_e2e_oracle", False) or op_conf.get("e2e_oracle_enabled", False))
        if enable_e2e_oracle:
            try:
                raw_spec = (cert_out.step_cert or {}).get("e2e_spec")
                spec = parse_e2e_spec(raw_spec)
                e2e_step = build_e2e_sub_step(record=rec, spec=spec)
                wrapper_code = build_reference_solve_wrapper_code(spec=spec, record=rec)
                derive_out_e2e = derive_runner.run_one(
                    TestDeriveInput(
                        background=str(rec.background or ""),
                        required_dependencies=str(rec.required_dependencies or ""),
                        sub_steps=list(rec.sub_steps) + [e2e_step],
                        per_step_golden=dict(rec.per_step_golden or {}),
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
    else:
        raise ValueError(f"Unsupported code_source={code_source!r} for revise (only llm-generated is supported)")

    fold_payload = apply_executable_path_fold(agent_conf, state, rec, step_dir=step_dir, op_name="revise")
    # Replace the latest executable record in-place to keep step index unchanged.
    for i in range(len(state.history) - 1, -1, -1):
        if isinstance(state.history[i], type(rec)):
            state.history[i] = rec
            break
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
    "run_executable_revise",
]
