from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agenqa.domain.executable_schema import ExecutableRecord, ExecutableSubStep


_DEF_NAME_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.ASCII)
_CLASS_NAME_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|:)", re.ASCII)


def _extract_def_name(function_header: str) -> str:
    if not isinstance(function_header, str):
        return ""
    for line in function_header.splitlines():
        m = _DEF_NAME_RE.match(line)
        if m:
            return m.group(1)
        m = _CLASS_NAME_RE.match(line)
        if m:
            return m.group(1)
    return ""


def _is_valid_py_ident(name: str) -> bool:
    return isinstance(name, str) and name.isidentifier()


@dataclass(frozen=True)
class E2EParam:
    name: str
    default: Any = None
    has_default: bool = False


@dataclass(frozen=True)
class E2ECall:
    step_number: str
    bind: Optional[str]
    is_return: bool
    kwargs: Dict[str, str]


@dataclass(frozen=True)
class E2ESpec:
    function_name: str
    params: List[E2EParam]
    calls: List[E2ECall]


def parse_e2e_spec(obj: Any) -> E2ESpec:
    if not isinstance(obj, dict):
        raise ValueError("e2e_spec must be an object")
    function_name = str(obj.get("function_name") or "").strip()
    if function_name != "solve":
        raise ValueError("e2e_spec.function_name must be 'solve'")

    params_raw = obj.get("params") or []
    if not isinstance(params_raw, list) or not params_raw:
        raise ValueError("e2e_spec.params must be a non-empty list")
    params: List[E2EParam] = []
    seen: set[str] = set()
    for i, item in enumerate(params_raw):
        if not isinstance(item, dict):
            raise ValueError(f"e2e_spec.params[{i}] must be an object")
        name = str(item.get("name") or "").strip()
        if not _is_valid_py_ident(name):
            raise ValueError(f"e2e_spec.params[{i}].name must be a valid identifier: {name!r}")
        if name in seen:
            raise ValueError(f"e2e_spec.params contains duplicate param: {name!r}")
        seen.add(name)
        if "default" in item:
            default = item.get("default")
            if default is not None and not isinstance(default, (bool, int, float, str)):
                raise ValueError(f"e2e_spec.params[{i}].default must be JSON-scalar or null: {default!r}")
            params.append(E2EParam(name=name, default=default, has_default=True))
        else:
            params.append(E2EParam(name=name))

    calls_raw = obj.get("calls") or []
    if not isinstance(calls_raw, list) or not calls_raw:
        raise ValueError("e2e_spec.calls must be a non-empty list")
    calls: List[E2ECall] = []
    for i, item in enumerate(calls_raw):
        if not isinstance(item, dict):
            raise ValueError(f"e2e_spec.calls[{i}] must be an object")
        step_number = str(item.get("step_number") or "").strip()
        if not step_number:
            raise ValueError(f"e2e_spec.calls[{i}].step_number is required")
        bind = item.get("bind")
        bind_s = str(bind).strip() if isinstance(bind, str) else None
        is_return = bool(item.get("return") is True)
        if is_return and bind_s:
            raise ValueError(f"e2e_spec.calls[{i}]: return=true should not set bind")
        if (not is_return) and (not bind_s):
            raise ValueError(f"e2e_spec.calls[{i}]: non-return call must set bind")
        if bind_s and not _is_valid_py_ident(bind_s):
            raise ValueError(f"e2e_spec.calls[{i}].bind must be a valid identifier: {bind_s!r}")
        kwargs_raw = item.get("kwargs") or {}
        if not isinstance(kwargs_raw, dict) or not kwargs_raw:
            raise ValueError(f"e2e_spec.calls[{i}].kwargs must be a non-empty object")
        kwargs: Dict[str, str] = {}
        for k, v in kwargs_raw.items():
            if not _is_valid_py_ident(str(k)):
                raise ValueError(f"e2e_spec.calls[{i}].kwargs has invalid key: {k!r}")
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"e2e_spec.calls[{i}].kwargs[{k!r}] must be a non-empty string")
            kwargs[str(k)] = str(v).strip()
        calls.append(E2ECall(step_number=step_number, bind=bind_s, is_return=is_return, kwargs=kwargs))
    if not any(call.is_return for call in calls):
        last = calls[-1]
        calls[-1] = E2ECall(step_number=last.step_number, bind=None, is_return=True, kwargs=last.kwargs)
    elif not calls[-1].is_return:
        raise ValueError("e2e_spec.calls return=true must be on the last item")
    return E2ESpec(function_name=function_name, params=params, calls=calls)


def build_solve_function_header(spec: E2ESpec) -> str:
    params: List[str] = []
    for p in spec.params:
        if p.has_default:
            params.append(f"{p.name}={repr(p.default)}")
        else:
            params.append(p.name)
    return f"def solve({', '.join(params)}):"


def _resolve_ref_expr(
    ref: str,
    *,
    params: set[str],
    bound_vars: set[str],
) -> str:
    if ref.startswith("param:"):
        name = ref[len("param:") :].strip()
        if name not in params:
            raise ValueError(f"e2e_spec reference uses unknown param: {ref!r}")
        return name
    if ref.startswith("var:"):
        name = ref[len("var:") :].strip()
        if name not in bound_vars:
            raise ValueError(f"e2e_spec reference uses unknown var: {ref!r}")
        return name
    raise ValueError(f"e2e_spec reference must start with 'param:' or 'var:': {ref!r}")


def build_reference_solve_wrapper_code(
    *,
    spec: E2ESpec,
    record: ExecutableRecord,
) -> str:
    if not record.sub_steps:
        raise ValueError("record.sub_steps is empty")
    tail_step = record.sub_steps[-1]
    tail_id = str(tail_step.step_number or "").strip()
    if not tail_id:
        raise ValueError("tail step_number is empty")

    step_to_func: Dict[str, str] = {}
    for s in record.sub_steps:
        sid = str(s.step_number or "").strip()
        if not sid:
            continue
        fn = _extract_def_name(s.function_header)
        if not fn:
            raise ValueError(f"Unsupported sub_step.function_header (only def/class is supported): step_number={sid!r}")
        step_to_func[sid] = fn

    if not spec.calls[-1].is_return:
        raise ValueError("e2e_spec.calls last item must have return=true")
    if str(spec.calls[-1].step_number).strip() != tail_id:
        raise ValueError(f"e2e_spec.calls last step_number must equal tail step_number={tail_id!r}")

    params = {p.name for p in spec.params}
    bound_vars: set[str] = set()

    lines: List[str] = [build_solve_function_header(spec)]
    indent = " " * 4
    for idx, call in enumerate(spec.calls):
        step_id = str(call.step_number).strip()
        fn_name = step_to_func.get(step_id)
        if not fn_name:
            raise ValueError(f"e2e_spec.calls[{idx}].step_number not found in record.sub_steps: {step_id!r}")

        parts: List[str] = []
        for arg_name, ref in call.kwargs.items():
            expr = _resolve_ref_expr(ref, params=params, bound_vars=bound_vars)
            parts.append(f"{arg_name}={expr}")
        call_expr = f"{fn_name}({', '.join(parts)})"

        if call.is_return:
            lines.append(f"{indent}return {call_expr}")
        else:
            assert call.bind
            lines.append(f"{indent}{call.bind} = {call_expr}")
            bound_vars.add(call.bind)

    return "\n".join(lines).rstrip() + "\n"


def build_e2e_sub_step(*, record: ExecutableRecord, spec: E2ESpec) -> ExecutableSubStep:
    tail = record.sub_steps[-1] if record.sub_steps else None
    return ExecutableSubStep(
        step_number="e2e",
        step_description="实现整题入口函数 solve(...)：仅基于外部输入（premises）产出最终输出（tail output）。",
        function_header=build_solve_function_header(spec),
        return_line=(tail.return_line if tail else "return result"),
        step_background=None,
    )


def find_step_cert_for_step(memory: Any, *, step: int) -> Optional[Dict[str, Any]]:
    if not isinstance(memory, dict):
        return None
    certs = memory.get("step_certs")
    if not isinstance(certs, list):
        return None
    for item in reversed(certs):
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("step")) != int(step):
                continue
        except Exception:
            continue
        # Only chain certs carry e2e_spec; eval certs are appended later and should not shadow the chain cert.
        if item.get("kind") == "executable_chain_cert":
            return item
    return None


def resolve_e2e_spec_from_memory(memory: Any, *, step: int) -> Optional[E2ESpec]:
    cert = find_step_cert_for_step(memory, step=step)
    if not isinstance(cert, dict):
        return None
    raw = cert.get("e2e_spec")
    if raw is None:
        return None
    return parse_e2e_spec(raw)


__all__ = [
    "E2EParam",
    "E2ECall",
    "E2ESpec",
    "parse_e2e_spec",
    "build_solve_function_header",
    "build_reference_solve_wrapper_code",
    "build_e2e_sub_step",
    "resolve_e2e_spec_from_memory",
]
