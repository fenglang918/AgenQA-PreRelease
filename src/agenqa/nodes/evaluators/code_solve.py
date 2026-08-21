"""Executable solve/eval node: execute code and judge correctness via verifier.

Phase R2 (pipeline road) runs llm-generated steps only and verifies via execution.
- eval_backend=native|mcp: run locally (PythonExecutor) or via MCP verifier server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
from string import Template
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from infra.code_verifier.executor import ExecutionConfig, ExecutionResult
from infra.code_verifier.python_executor import PythonExecutor
from infra.data.io import write_jsonl
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import snapshot_prompt_used, snapshot_rendered_prompt
from infra.llm.inference import resolve_inference
from infra.llm.service_loader import load_llm_service_full_config
from agenqa.domain.executable_error_taxonomy import classify_row
from agenqa.domain.executable_schema import ExecutableRecord
from agenqa.domain.folded_question_schema import extract_executable_payload
from agenqa.domain.known_tree import KnownTree
from agenqa.domain.node_result import NodeResult
from agenqa.graph.output_manager import compute_step_dir
from agenqa.graph.state import AgentState, SolverResult
from agenqa.memory.store import save_state
from agenqa.skills.base import BaseSkillRunner
from agenqa.prompts.executable_solver import EXECUTABLE_SOLVER_V1, EXECUTABLE_SOLVER_V1_EN

logger = logging.getLogger(__name__)


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    matches = [m.strip() for m in _CODE_BLOCK_RE.findall(text) if m.strip()]
    if matches:
        return "\n\n".join(matches)
    # Fallback: start from first def/class/import line.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*(def|class|import|from)\b", line):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def _extract_function_name(function_header: str) -> str:
    for pattern in (r"\bdef\s+(\w+)\s*\(", r"\bclass\s+(\w+)\s*\("):
        match = re.search(pattern, function_header or "")
        if match:
            return match.group(1)
    raise ValueError("Function or class name not found in function_header")


def _get_function_from_code(code_string: str, function_name: str) -> Optional[str]:
    if not code_string:
        return None
    try:
        import ast

        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception:
        return code_string
    return None


def _fold_variant_from_settings(path_variant: str) -> str:
    v = str(path_variant or "").strip().lower()
    return "direct" if v in {"direct", "e2e"} else "scaffolded"


def _extract_fold_plan(rec: ExecutableRecord, *, path_variant: str) -> tuple[list[Dict[str, Any]], str, str, list[str]] | None:
    variant = _fold_variant_from_settings(path_variant)
    raw = rec.path_question_direct if variant == "direct" else rec.path_question_scaffolded
    if not (isinstance(raw, str) and raw.strip()):
        return None
    try:
        obj = extract_executable_payload(raw, variant=variant)  # type: ignore[arg-type]
    except Exception:
        return None
    sub_steps = obj.get("sub_steps")
    if not isinstance(sub_steps, list) or not sub_steps:
        return None
    head = sub_steps[0] if isinstance(sub_steps[0], dict) else {}
    fn_header = str(head.get("function_header") or "").strip()
    return_line = str(head.get("return_line") or "").strip()

    tests_map = obj.get("test_cases") or {}
    tests: list[str] = []
    if isinstance(tests_map, dict):
        for _k, raw in tests_map.items():
            if not isinstance(raw, list):
                continue
            for item in raw:
                if isinstance(item, dict) and isinstance(item.get("test_code"), str) and item["test_code"].strip():
                    tests.append(item["test_code"])
            if tests:
                break
    if not tests:
        return None
    return sub_steps, fn_header, return_line, tests


def _build_scicode_step_script(
    *,
    dependencies: str,
    code_content: str,
    previous_steps: List[Tuple[str, str]],  # (function_header, code_text)
    step_id: str,
    test_cases: List[str],
    h5_file: Path,
    scicode_root: Path,
) -> str:
    scicode_src = scicode_root / "src"

    lines: List[str] = []

    # 0) Dependencies (SciCode requires the harness to provide imports).
    dep_lines = []
    for line in (dependencies or "").splitlines():
        if line.strip():
            dep_lines.append(line.rstrip())
    if dep_lines:
        lines.extend(dep_lines)
        lines.append("")

    # 1) Previous step function bodies (keep only the required function/class)
    for fn_header, raw in previous_steps:
        fn_name = _extract_function_name(fn_header)
        code = _extract_code(raw)
        fn_code = _get_function_from_code(code, fn_name) or code
        lines.extend(fn_code.splitlines())
        if lines and lines[-1] != "":
            lines.append("")

    # 2) Current step code
    curr_code_lines = _extract_code(code_content).splitlines()
    # Inject SciCode src path after __future__ imports (if any).
    if scicode_src.exists():
        insert_at = 0
        while insert_at < len(curr_code_lines) and curr_code_lines[insert_at].startswith("from __future__ import"):
            insert_at += 1
        injection = [
            "import sys",
            f"sys.path.insert(0, {repr(str(scicode_src.resolve()))})",
            "",
        ]
        curr_code_lines = curr_code_lines[:insert_at] + injection + curr_code_lines[insert_at:]
    lines.extend(curr_code_lines)
    if lines and lines[-1] != "":
        lines.append("")

    # 3) SciCode HDF5-backed targets + tests
    lines.append("from scicode.parse.parse import process_hdf5_to_tuple")
    lines.append("")
    lines.append(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_cases)}, {repr(str(h5_file))})")
    lines.append("")
    for i, test in enumerate(test_cases):
        lines.append(f"target = targets[{i}]")
        lines.append("")
        lines.extend((test or "").splitlines())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_llm_step_script(
    *,
    dependencies: str,
    code_content: str,
    previous_steps: List[Tuple[str, str]],  # (function_header, code_text)
    test_cases: List[str],
) -> str:
    lines: List[str] = []
    dep_lines = []
    for line in (dependencies or "").splitlines():
        if line.strip():
            dep_lines.append(line.rstrip())
    if dep_lines:
        lines.extend(dep_lines)
        lines.append("")

    for fn_header, raw in previous_steps:
        fn_name = _extract_function_name(fn_header)
        code = _extract_code(raw)
        fn_code = _get_function_from_code(code, fn_name) or code
        lines.extend(fn_code.splitlines())
        if lines and lines[-1] != "":
            lines.append("")

    curr_code_lines = _extract_code(code_content).splitlines()
    lines.extend(curr_code_lines)
    if lines and lines[-1] != "":
        lines.append("")

    # Provide a generic deep-equal helper for tests that rely on it.
    lines.extend(
        [
            "def _norm(v):",
            "    if hasattr(v, 'tolist'):",
            "        try:",
            "            return v.tolist()",
            "        except Exception:",
            "            return v",
            "    return v",
            "def _deep_equal(a, b, tol=1e-6):",
            "    a = _norm(a)",
            "    b = _norm(b)",
            "    if isinstance(a, bool) or isinstance(b, bool):",
            "        return a == b",
            "    if isinstance(a, (int, float)) and isinstance(b, (int, float)):",
            "        return abs(a - b) <= tol",
            "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):",
            "        return len(a) == len(b) and all(_deep_equal(x, y, tol=tol) for x, y in zip(a, b))",
            "    if isinstance(a, dict) and isinstance(b, dict):",
            "        if set(a.keys()) != set(b.keys()):",
            "            return False",
            "        return all(_deep_equal(a[k], b[k], tol=tol) for k in a.keys())",
            "    return a == b",
            "",
        ]
    )

    for test in test_cases:
        lines.extend((test or "").splitlines())
        if lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _executable_path_view(sub_steps: list[Dict[str, Any]], *, variant: str) -> list[Dict[str, Any]]:
    if not sub_steps:
        return []
    head = sub_steps[0]
    tail = sub_steps[-1]
    if variant == "direct":
        return [head, tail] if len(sub_steps) > 1 else [head]
    if variant == "scaffolded":
        out: list[Dict[str, Any]] = [head]
        for mid in sub_steps[1:-1]:
            out.append(
                {
                    "step_number": mid.get("step_number", ""),
                    "step_description": "",
                    "function_header": mid.get("function_header", ""),
                    "return_line": mid.get("return_line", ""),
                    "step_background": None,
                }
            )
        if len(sub_steps) > 1:
            out.append(tail)
        return out
    return list(sub_steps)


def _normalize_solver_confs(agent_conf: Dict[str, Any], tier: str) -> list[Dict[str, Any]]:
    solvers_block = agent_conf.get("solvers") or {}
    if not isinstance(solvers_block, dict):
        return []
    raw = solvers_block.get(tier)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict) and item]
    if isinstance(raw, dict) and raw:
        return [raw]
    return []


DEFAULT_SERVICE_CONFIG = Path(
    os.getenv(
        "LLM_SERVICES_JSON",
        "config/services.json",
    )
)


def _merge_generator_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge with deep merge for common nested keys (client/generation)."""
    merged: Dict[str, Any] = dict(base or {})
    for key, val in (override or {}).items():
        if key in {"client", "generation"} and isinstance(merged.get(key), dict) and isinstance(val, dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def _resolve_solver_generator(solve_conf: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a usable generator dict for ExecutableSolver.

    - If `generator` is provided and already contains api_base+model_name, use it.
    - If only `service_id` is provided, resolve it via llm_service `services.json`.
    """
    generator = solve_conf.get("generator")
    if isinstance(generator, dict) and generator:
        service_id = generator.get("service_id")
        has_api_base = bool(generator.get("api_base") or generator.get("base_url"))
        has_model = bool(generator.get("model_name"))
        if isinstance(service_id, str) and service_id.strip() and (not has_api_base or not has_model):
            service_cfg_path = Path(solve_conf.get("service_config") or DEFAULT_SERVICE_CONFIG)
            base = load_llm_service_full_config(
                service_cfg_path,
                service_id.strip(),
                explicit_model=solve_conf.get("service_model"),
                fallback_model=generator.get("model_name"),
            )
            return _merge_generator_configs(base, generator)
        return generator

    # Backward compatible: allow using the solver block itself as a generator.
    if any(k in solve_conf for k in ("api_base", "base_url", "model_name")):
        return solve_conf

    # Backward compatible: service_id at tier level
    service_id = solve_conf.get("service_id")
    if isinstance(service_id, str) and service_id.strip():
        service_cfg_path = Path(solve_conf.get("service_config") or DEFAULT_SERVICE_CONFIG)
        base = load_llm_service_full_config(
            service_cfg_path,
            service_id.strip(),
            explicit_model=solve_conf.get("service_model"),
        )
        # Allow overriding client/generation at tier level
        override: Dict[str, Any] = {}
        for k in ("client", "generation"):
            if isinstance(solve_conf.get(k), dict):
                override[k] = solve_conf.get(k)
        return _merge_generator_configs(base, override)

    return {}


def _render_executable_solver_prompt(
    *,
    payload: Dict[str, Any],
    lang: str | None,
    prompt_text: str,
) -> str:
    tmpl = Template(prompt_text)
    body = tmpl.safe_substitute(payload)
    lang_norm = (lang or "zh").lower()
    header = f"Role: ExecutableSolver | Language: {lang_norm}" if lang_norm in {"en", "english"} else f"【ExecutableSolver 角色】语言: {lang_norm}"
    return "\n".join([header, "", body])


def _run_executable_solver_once(
    *,
    generator: Dict[str, Any],
    prompt_text: str,
    prompt_path: Path,
    payload: Dict[str, Any],
    lang: str | None,
    dependencies: str,
    previous_steps: List[Tuple[str, str]],
    test_snippets: List[str],
    eval_backend: str,
    mcp_url: str | None,
    mcp_use_proxy: bool,
    step_dir: Path | None,
    name_suffix: str,
    step_timeout: float,
    memory_mb: int,
    temp_dir: str,
    python_bin: str,
) -> Tuple[Dict[str, Any], bool, str, float]:
    resolved = resolve_inference(generator)
    sess = resolved.session
    prompt = _render_executable_solver_prompt(payload=payload, lang=lang, prompt_text=prompt_text)
    messages = build_messages_with_background(prompt, lang=lang or "zh")
    response = sess.chat(messages, **resolved.chat_args)
    BaseSkillRunner._check_finish_reason(response, f"ExecutableSolver[{name_suffix}]")
    text = sess.extract_text(response, default="")
    code = _extract_code(text)
    if not code:
        raise RuntimeError(f"executable_solver_empty_code[{name_suffix}]")

    if step_dir is not None:
        try:
            snap_dir = step_dir / "subruns_raw" / f"code_solve_{name_suffix}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "input_view.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            snapshot_prompt_used(
                prompt_path,
                snap_dir,
                content=prompt_text,
                name_prefix="prompt_used.",
                logger=logger,
            )
            snapshot_rendered_prompt(prompt, snap_dir, filename="prompt_rendered.txt", logger=logger)
            request_meta = {"model": sess.model_name, "service_id": sess.service_id, "chat_args": resolved.chat_args}
            (snap_dir / "request_meta.json").write_text(
                json.dumps(request_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                (snap_dir / "raw_response.json").write_text(
                    json.dumps(response, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            (snap_dir / "output.json").write_text(
                json.dumps({"code": code}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    script = _build_llm_step_script(
        dependencies=dependencies,
        code_content=code,
        previous_steps=previous_steps,
        test_cases=test_snippets,
    )

    ok = False
    err = ""
    exec_time = 0.0
    if eval_backend == "native":
        result = _run_native(
            script,
            timeout=float(step_timeout),
            memory_mb=int(memory_mb),
            temp_dir=str(temp_dir),
            python_bin=str(python_bin),
        )
        ok = bool(result.success)
        err = str(result.error or "")
        exec_time = float(result.execution_time or 0.0)
    elif eval_backend == "mcp":
        if not (isinstance(mcp_url, str) and mcp_url.strip()):
            raise ValueError("executable_eval.mcp_url is required when eval_backend='mcp'")
        payload_out = _run_mcp(
            script,
            timeout=float(step_timeout),
            mcp_url=mcp_url,
            use_proxy=bool(mcp_use_proxy),
        )
        ok = bool(payload_out.get("success") is True)
        err = str(payload_out.get("error") or "")
        try:
            exec_time = float(payload_out.get("execution_time") or 0.0)
        except Exception:
            exec_time = 0.0
    else:
        raise ValueError(f"Unsupported executable_eval.eval_backend={eval_backend!r} (expected 'native' or 'mcp')")

    row = {
        "correct": ok,
        "execution_time": exec_time,
        "error": err if not ok else "",
        "model": sess.model_name,
        "service_id": sess.service_id,
        "code": code,
    }
    return row, ok, err, exec_time


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


def _resolve_settings(agent_conf: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    conf = agent_conf.get("executable_eval") or {}
    if not isinstance(conf, dict):
        conf = {}
    seed = {}
    try:
        mem = getattr(state, "memory", None)
        if isinstance(mem, dict) and isinstance(mem.get("executable_seed"), dict):
            seed = mem["executable_seed"]
    except Exception:
        seed = {}

    scicode_root = Path(conf.get("scicode_root") or seed.get("scicode_root") or "external/eval_inno/scicode/SciCode")
    h5_file = Path(conf.get("h5_file") or os.getenv("AGENQA_EXECUTABLE_H5", "data/test_data.h5"))
    eval_backend = str(conf.get("eval_backend") or "native").strip().lower()
    mcp_url = conf.get("mcp_url")
    step_timeout = float(conf.get("step_timeout", 120))
    memory_mb = int(conf.get("memory_mb", 16384))
    temp_dir = str(conf.get("temp_dir") or str((Path("infra/code_verifier/.tmp")).resolve()))
    python_bin = str(conf.get("python_bin") or sys.executable)
    code_source = str(conf.get("code_source") or "llm-generated").strip().lower()
    mcp_use_proxy = bool(conf.get("mcp_use_proxy", False))
    path_variant = str(conf.get("path_variant") or "scaffolded").strip().lower()

    return {
        "scicode_root": scicode_root,
        "h5_file": h5_file,
        "eval_backend": eval_backend,
        "mcp_url": mcp_url,
        "step_timeout": step_timeout,
        "memory_mb": memory_mb,
        "temp_dir": temp_dir,
        "python_bin": python_bin,
        "code_source": code_source,
        "mcp_use_proxy": mcp_use_proxy,
        "path_variant": path_variant,
    }


def _run_native(script: str, *, timeout: float, memory_mb: int, temp_dir: str, python_bin: str) -> ExecutionResult:
    conf = ExecutionConfig(timeout=float(timeout), memory_limit_mb=int(memory_mb), temp_dir=str(temp_dir), python_bin=str(python_bin))
    executor = PythonExecutor(conf)
    return asyncio.run(executor.execute_single(script))


def _run_mcp(script: str, *, timeout: float, mcp_url: str, use_proxy: bool) -> Dict[str, Any]:
    try:
        from fastmcp.client import Client
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing optional dependency `fastmcp`. Install requirements_code_verifier.txt for MCP backend."
        ) from exc

    async def _call() -> str:
        async with Client(mcp_url) as client:
            result = await client.call_tool("execute_code", {"code": script, "timeout": float(timeout)})
            text = ""
            try:
                if result and getattr(result, "content", None):
                    text = result.content[0].text  # type: ignore[attr-defined]
            except Exception:
                text = ""
            return text

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
    ctx = contextlib.nullcontext() if use_proxy else _temporary_unset_env(proxy_keys)
    with ctx:
        raw = asyncio.run(_call())
    try:
        payload = json.loads(raw) if raw else {}
    except Exception:
        payload = {"success": False, "error": f"invalid_mcp_response\n{raw}"}
    return payload if isinstance(payload, dict) else {"success": False, "error": f"invalid_mcp_response\n{raw}"}


def _latest_executable_record(state: AgentState) -> ExecutableRecord:
    rec = state.latest_executable_record()
    if not isinstance(rec, ExecutableRecord):
        raise RuntimeError("executable history is empty or invalid; run extend first")
    return rec


def code_solve(agent_conf: Dict[str, Any], state: AgentState, output_manager: Any | None = None) -> AgentState | NodeResult:
    """Evaluate the latest executable step via execution oracle."""
    settings = _resolve_settings(agent_conf, state)
    rec = _latest_executable_record(state)

    try:
        step_idx = int(state.step or 0)
    except Exception:
        step_idx = int(rec.qa_idx or 0)

    op_raw = (state.last_decision.operation if state.last_decision else "extend")  # type: ignore[union-attr]
    op_name = str(op_raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "revise" in op_name:
        op_name = "revise"
    elif "extend" in op_name or "init" in op_name:
        op_name = "extend"
    if op_name not in {"extend", "revise"}:
        op_name = "extend"

    round_idx = state.current_round_index()
    step_dir = compute_step_dir(state.artifacts_dir, op_name, step_idx, round_idx)
    step_dir.mkdir(parents=True, exist_ok=True)
    solve_dir = step_dir / "solve"
    solve_dir.mkdir(parents=True, exist_ok=True)

    # SciCode: tests are keyed by step_number (string), use the latest visible step.
    if not rec.sub_steps:
        raise RuntimeError("executable_record.sub_steps is empty")
    tail = rec.sub_steps[-1]
    step_id = tail.step_number
    tests = rec.test_cases.get(step_id) or []
    test_snippets = [t.test_code for t in tests if isinstance(t.test_code, str) and t.test_code.strip()]

    # Build previous code list (function bodies only).
    previous_steps: List[Tuple[str, str]] = []
    for s in rec.sub_steps[:-1]:
        if not s.step_number:
            continue
        previous_steps.append((s.function_header, rec.per_step_golden.get(s.step_number, "")))

    code_text = rec.per_step_golden.get(step_id, "")
    if not (isinstance(code_text, str) and code_text.strip()):
        raise RuntimeError(f"Missing per_step_golden for step_id={step_id!r}")

    eval_backend = settings["eval_backend"]
    code_source = settings["code_source"]

    # ---- Multi-solver setup (llm-generated) ----
    medium_confs = _normalize_solver_confs(agent_conf, "medium")
    strong_confs = _normalize_solver_confs(agent_conf, "strong")

    agent_lang = str((agent_conf.get("agent") or {}).get("lang") or "").lower() or None
    use_en = str(agent_lang or "").lower().strip() in {"en", "english"}
    default_prompt_text = EXECUTABLE_SOLVER_V1_EN if use_en else EXECUTABLE_SOLVER_V1

    sub_steps_full = [s.to_dict() for s in rec.sub_steps]
    path_variant = settings.get("path_variant") or "scaffolded"

    fold_plan = _extract_fold_plan(rec, path_variant=str(path_variant))
    if fold_plan is not None:
        sub_steps_path, path_function_header, path_return_line, test_snippets_path = fold_plan
        previous_steps_path: List[Tuple[str, str]] = []
    else:
        sub_steps_path = _executable_path_view(sub_steps_full, variant=str(path_variant))
        path_function_header = tail.function_header
        path_return_line = tail.return_line
        test_snippets_path = test_snippets
        previous_steps_path = previous_steps

    def _truncate_error(val: Any, *, max_chars: int = 400) -> str:
        if not isinstance(val, str):
            return ""
        s = val.strip()
        if not s:
            return ""
        if len(s) <= max_chars:
            return s
        return s[: max_chars - 1] + "…"

    def _executable_extra_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        cls = classify_row(row)
        extra: Dict[str, Any] = {
            "eval_backend": row.get("eval_backend") or eval_backend,
            "code_source": row.get("code_source") or code_source,
            "execution_time": row.get("execution_time"),
            "error_type": cls.type,
            "recoverable": cls.recoverable,
        }
        err_short = _truncate_error(row.get("error"))
        if err_short:
            extra["error"] = err_short
        step_id_local = row.get("scicode_step_id")
        if isinstance(step_id_local, str) and step_id_local.strip():
            extra["scicode_step_id"] = step_id_local.strip()
        return extra

    def _solver_payload(target: str) -> Dict[str, Any]:
        view = sub_steps_full if target == "edge" else sub_steps_path
        known_view = (
            KnownTree.build_edge_solver_view(state.memory, step_idx)
            if target == "edge"
            else KnownTree.build_path_solver_view(state.memory, step_idx)
        )
        known_view = KnownTree.compact_kqa_known_view(known_view)
        if target == "path":
            fn_header = path_function_header
            ret_line = path_return_line
        else:
            fn_header = tail.function_header
            ret_line = tail.return_line
        return {
            "background": rec.background,
            "known_json": json.dumps(known_view, ensure_ascii=False),
            "required_dependencies": rec.required_dependencies,
            "sub_steps_json": json.dumps(view, ensure_ascii=False),
            "step_number": ("e2e" if target == "path" and fold_plan is not None else step_id),
            "function_header": fn_header,
            "return_line": ret_line,
        }

    def _run_tier(confs: list[Dict[str, Any]], tier_label: str) -> Tuple[Optional[Path], Optional[Path], Dict[str, SolverResult]]:
        edge_path = None
        path_path = None
        results: Dict[str, SolverResult] = {}
        for idx, conf in enumerate(confs):
            if tier_label == "strong":
                suffix = f"strong_{idx}"
            else:
                suffix = tier_label if idx == 0 else f"{tier_label}_{idx}"
            generator = _resolve_solver_generator(conf)
            if not generator:
                continue
            prompt_text = conf.get("executable_prompt_text") or default_prompt_text
            prompt_path = Path(conf.get("executable_prompt_path") or "src/agenqa/prompts/executable_solver.prompt")
            for target in ("edge", "path"):
                prev_steps = previous_steps if target == "edge" else previous_steps_path
                tests_local = test_snippets if target == "edge" else test_snippets_path
                payload = _solver_payload(target)
                row_base = {
                    "step": step_idx,
                    "problem_id": rec.problem_id,
                    "scicode_step_id": step_id,
                    "eval_backend": eval_backend,
                    "code_source": code_source,
                    "target": target,
                    "tier": suffix,
                }
                try:
                    row, ok_s, err_s, _exec_s = _run_executable_solver_once(
                        generator=generator,
                        prompt_text=str(prompt_text),
                        prompt_path=prompt_path,
                        payload=payload,
                        lang=agent_lang,
                        dependencies=str(rec.required_dependencies or ""),
                        previous_steps=prev_steps,
                        test_snippets=tests_local,
                        eval_backend=eval_backend,
                        mcp_url=settings.get("mcp_url"),
                        mcp_use_proxy=bool(settings.get("mcp_use_proxy")),
                        step_dir=step_dir,
                        name_suffix=f"{suffix}_{target}",
                        step_timeout=float(settings["step_timeout"]),
                        memory_mb=int(settings["memory_mb"]),
                        temp_dir=str(settings["temp_dir"]),
                        python_bin=str(settings["python_bin"]),
                    )
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "correct": False,
                        "execution_time": 0.0,
                        "error": f"solver_failed: {exc}",
                        "model": None,
                        "service_id": None,
                        "code": "",
                    }
                    ok_s = False
                    err_s = row["error"]
                row_full = {**row_base, **row}
                out_name = f"solve_{suffix}.jsonl" if target == "edge" else f"solve_path_{suffix}.jsonl"
                out_path = solve_dir / out_name
                write_jsonl([row_full], out_path, append=False)
                if target == "edge" and idx == 0:
                    edge_path = out_path
                if target == "path" and idx == 0:
                    path_path = out_path
                results[f"{target}:{suffix}"] = SolverResult(
                    correct=ok_s,
                    model=row.get("model"),
                    service_id=row.get("service_id"),
                    correctness_feedback=err_s,
                    extra=_executable_extra_from_row(row_full),
                )
        return edge_path, path_path, results

    def _update_metrics_and_solver_index_from_results(results: Dict[str, SolverResult], *, tier_label: str) -> Optional[bool]:
        res = results.get(f"edge:{tier_label}")
        ok_tier = res.correct if isinstance(res, SolverResult) else None
        try:
            for key, r in results.items():
                if ":" not in key:
                    continue
                target, tier = key.split(":", 1)
                if isinstance(r, SolverResult):
                    state.update_solver_index(step=step_idx, round=round_idx, target=target, tier=tier, result=r)
        except Exception:
            pass
        return ok_tier

    if code_source not in {"llm-generated", "llm", "generated"}:
        raise ValueError(f"Unsupported executable_eval.code_source={code_source!r} (only llm-generated is supported)")

    # ---- LLM-generated executable: verify golden + run solver(s) ----
    script = _build_llm_step_script(
        dependencies=str(rec.required_dependencies or ""),
        code_content=code_text,
        previous_steps=previous_steps,
        test_cases=test_snippets,
    )

    ok = False
    err = ""
    exec_time = 0.0
    if eval_backend == "native":
        result = _run_native(
            script,
            timeout=float(settings["step_timeout"]),
            memory_mb=int(settings["memory_mb"]),
            temp_dir=str(settings["temp_dir"]),
            python_bin=str(settings["python_bin"]),
        )
        ok = bool(result.success)
        err = str(result.error or "")
        exec_time = float(result.execution_time or 0.0)
    elif eval_backend == "mcp":
        mcp_url = settings["mcp_url"]
        if not (isinstance(mcp_url, str) and mcp_url.strip()):
            raise ValueError("executable_eval.mcp_url is required when eval_backend='mcp'")
        payload = _run_mcp(
            script,
            timeout=float(settings["step_timeout"]),
            mcp_url=mcp_url,
            use_proxy=bool(settings["mcp_use_proxy"]),
        )
        ok = bool(payload.get("success") is True)
        err = str(payload.get("error") or "")
        try:
            exec_time = float(payload.get("execution_time") or 0.0)
        except Exception:
            exec_time = 0.0
    else:
        raise ValueError(f"Unsupported executable_eval.eval_backend={eval_backend!r} (expected 'native' or 'mcp')")

    # Write golden verification logs.
    for target in ("edge", "path"):
        try:
            logs_dir = step_dir / "evaluation_logs" / target
            logs_dir.mkdir(parents=True, exist_ok=True)
            status = "pass" if ok else "fail"
            (logs_dir / f"{step_id}.log").write_text((status if ok else f"{status}\n{err}".rstrip()), encoding="utf-8")
        except Exception:
            pass

    golden_row = {
        "step": step_idx,
        "problem_id": rec.problem_id,
        "scicode_step_id": step_id,
        "correct": ok,
        "eval_backend": eval_backend,
        "code_source": code_source,
        "execution_time": exec_time,
        "error": err if not ok else "",
        "model": "golden",
        "service_id": "golden",
    }
    golden_extra = _executable_extra_from_row(golden_row)
    try:
        write_jsonl([golden_row], solve_dir / "solve_golden.jsonl", append=False)
    except Exception:
        pass
    try:
        state.update_solver_index(
            step=step_idx,
            round=round_idx,
            target="edge",
            tier="golden",
            result=SolverResult(correct=ok, model="golden", service_id="golden", correctness_feedback=err, extra=golden_extra),
        )
        state.update_solver_index(
            step=step_idx,
            round=round_idx,
            target="path",
            tier="golden",
            result=SolverResult(correct=ok, model="golden", service_id="golden", correctness_feedback=err, extra=golden_extra),
        )
    except Exception:
        pass

    if not ok:
        medium_path = solve_dir / "solve_medium.jsonl"
        strong_path = solve_dir / "solve_strong_0.jsonl"
        ht_medium_path = solve_dir / "solve_path_medium.jsonl"
        ht_strong_path = solve_dir / "solve_path_strong_0.jsonl"
        write_jsonl([golden_row], medium_path, append=False)
        write_jsonl([golden_row], strong_path, append=False)
        write_jsonl([golden_row], ht_medium_path, append=False)
        write_jsonl([golden_row], ht_strong_path, append=False)

        state.metrics.correct_medium = ok
        state.metrics.token_ratio_medium = None
        try:
            state.update_solver_index(
                step=step_idx,
                round=round_idx,
                target="edge",
                tier="medium",
                result=SolverResult(correct=ok, model="golden", service_id="golden", correctness_feedback=err, extra=golden_extra),
            )
            state.update_solver_index(
                step=step_idx,
                round=round_idx,
                target="edge",
                tier="strong_0",
                result=SolverResult(correct=ok, model="golden", service_id="golden", correctness_feedback=err, extra=golden_extra),
            )
            state.update_solver_index(
                step=step_idx,
                round=round_idx,
                target="path",
                tier="medium",
                result=SolverResult(correct=ok, model="golden", service_id="golden", correctness_feedback=err, extra=golden_extra),
            )
            state.update_solver_index(
                step=step_idx,
                round=round_idx,
                target="path",
                tier="strong",
                result=SolverResult(correct=ok, model="golden", service_id="golden", correctness_feedback=err, extra=golden_extra),
            )
        except Exception:
            pass

        state.rounds = round_idx
        save_state(state)

        if output_manager:
            return NodeResult(
                state=state,
                step_idx=step_idx,
                round_idx=round_idx,
                outputs={
                    "solve_medium": medium_path,
                    "solve_strong": strong_path,
                    "solve_path_medium": ht_medium_path,
                    "solve_path_strong": ht_strong_path,
                },
                step_dir=solve_dir,
            )
        return state

    # Golden OK: run configured solvers (if any), otherwise fallback to the golden row.
    if medium_confs or strong_confs:
        m_edge, m_path, m_results = _run_tier(medium_confs, "medium")
        s_edge, s_path, s_results = _run_tier(strong_confs, "strong")

        state.metrics.correct_medium = _update_metrics_and_solver_index_from_results(m_results, tier_label="medium")
        _update_metrics_and_solver_index_from_results(s_results, tier_label="strong_0")
        state.metrics.token_ratio_medium = None

        state.rounds = round_idx
        save_state(state)
        if output_manager:
            return NodeResult(
                state=state,
                step_idx=step_idx,
                round_idx=round_idx,
                outputs={
                    "solve_medium": m_edge,
                    "solve_strong": s_edge,
                    "solve_path_medium": m_path,
                    "solve_path_strong": s_path,
                },
                step_dir=solve_dir,
            )
        return state

    medium_path = solve_dir / "solve_medium.jsonl"
    strong_path = solve_dir / "solve_strong_0.jsonl"
    ht_medium_path = solve_dir / "solve_path_medium.jsonl"
    ht_strong_path = solve_dir / "solve_path_strong_0.jsonl"
    write_jsonl([golden_row], medium_path, append=False)
    write_jsonl([golden_row], strong_path, append=False)
    write_jsonl([golden_row], ht_medium_path, append=False)
    write_jsonl([golden_row], ht_strong_path, append=False)

    state.metrics.correct_medium = ok
    state.metrics.token_ratio_medium = None
    try:
        for target in ("edge", "path"):
            for tier in ("medium", "strong_0"):
                state.update_solver_index(
                    step=step_idx,
                    round=round_idx,
                    target=target,
                    tier=tier,
                    result=SolverResult(correct=ok, model="golden", service_id="golden", correctness_feedback=err, extra=golden_extra),
                )
    except Exception:
        pass

    state.rounds = round_idx
    save_state(state)
    if output_manager:
        return NodeResult(
            state=state,
            step_idx=step_idx,
            round_idx=round_idx,
            outputs={
                "solve_medium": medium_path,
                "solve_strong": strong_path,
                "solve_path_medium": ht_medium_path,
                "solve_path_strong": ht_strong_path,
            },
            step_dir=solve_dir,
        )
    return state


__all__ = ["code_solve"]
