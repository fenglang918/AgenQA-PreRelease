"""TestDerive runner: generate inputs, execute golden code, and build tests."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from infra.code_verifier.executor import ExecutionConfig, ExecutionResult
from infra.code_verifier.python_executor import PythonExecutor
from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used, snapshot_rendered_prompt
from infra.llm.service_client import LLMServiceSession
from agenqa.domain.executable_schema import (
    ExecutableSubStep,
    ExecutableTestCase,
    ExecutableTestDeriveOutput,
    FIELD_TEST_INPUTS,
    FIELD_INPUTS_ARTIFACT_RELPATH,
    FIELD_INPUTS_SHA256,
    FIELD_TEST_CASES_FOR_STEP,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text

from agenqa.prompts.executable_test_inputs import EXECUTABLE_TEST_INPUTS_V1, EXECUTABLE_TEST_INPUTS_V1_EN

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class TestDeriveInput:
    background: str
    required_dependencies: str
    sub_steps: List[ExecutableSubStep]
    per_step_golden: Dict[str, str]
    step_number: str
    golden_step_code: str


@dataclass
class TestDeriveConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    step_timeout: float = 120.0
    memory_mb: int = 16384
    temp_dir: str = str((Path("infra/code_verifier") / ".tmp").resolve())
    python_bin: str = sys.executable
    max_inputs: int = 8
    max_case_chars: int = 4000
    golden_retries: int = 1
    input_retries: int = 1


def _truncate_text(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        return ""
    if max_chars <= 0:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)].rstrip() + "..."


def _extract_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    matches = [m.strip() for m in _CODE_BLOCK_RE.findall(text) if m.strip()]
    if matches:
        return "\n\n".join(matches)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*(def|class|import|from)\b", line):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def _extract_function_name(function_header: str) -> str:
    if not isinstance(function_header, str) or not function_header.strip():
        raise ValueError("Function or class name not found in function_header")
    for line in function_header.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"def\s+(\w+)\s*\(", s)
        if m:
            return m.group(1)
        m = re.match(r"class\s+(\w+)\s*(?:\(|:)", s)
        if m:
            return m.group(1)
    raise ValueError("Function or class name not found in function_header")


def _extract_allowed_kwargs_from_header(function_header: str) -> Optional[set[str]]:
    if not isinstance(function_header, str) or not function_header.strip():
        return None
    lines = [ln.strip() for ln in function_header.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return None

    is_class = lines[0].startswith("class ")
    sig_line: Optional[str] = None
    if is_class:
        for ln in lines[1:]:
            if ln.startswith("def __init__"):
                sig_line = ln
                break
    else:
        for ln in lines:
            if ln.startswith("def "):
                sig_line = ln
                break

    if not sig_line:
        return None

    m = re.search(r"\((.*)\)", sig_line)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return set()

    allowed: set[str] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token in {"/", "*"}:
            continue
        if token.startswith("**") or token.startswith("*"):
            continue
        if "=" in token:
            token = token.split("=", 1)[0].strip()
        if token in {"self", "cls"}:
            continue
        if token.isidentifier():
            allowed.add(token)
    return allowed


def _extract_param_order_from_header(function_header: str) -> Optional[list[str]]:
    if not isinstance(function_header, str) or not function_header.strip():
        return None
    lines = [ln.strip() for ln in function_header.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return None

    is_class = lines[0].startswith("class ")
    sig_line: Optional[str] = None
    if is_class:
        for ln in lines[1:]:
            if ln.startswith("def __init__"):
                sig_line = ln
                break
    else:
        for ln in lines:
            if ln.startswith("def "):
                sig_line = ln
                break
    if not sig_line:
        return None

    m = re.search(r"\((.*)\)", sig_line)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return []

    params: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token in {"/", "*"}:
            continue
        if token.startswith("**") or token.startswith("*"):
            continue
        if "=" in token:
            token = token.split("=", 1)[0].strip()
        if token in {"self", "cls"}:
            continue
        if token.isidentifier():
            params.append(token)
    return params


def _format_signature_args(args: ast.arguments) -> str:
    parts: List[str] = []
    for a in getattr(args, "posonlyargs", []) or []:
        parts.append(a.arg)
    if getattr(args, "posonlyargs", None):
        parts.append("/")
    for a in args.args or []:
        parts.append(a.arg)
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs or []:
        parts.append(a.arg)
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _infer_function_header_from_code(function_header: str, code: str) -> str:
    if not isinstance(function_header, str) or not function_header.strip():
        return function_header
    if not isinstance(code, str) or not code.strip():
        return function_header

    try:
        name = _extract_function_name(function_header)
    except Exception:
        return function_header

    try:
        tree = ast.parse(_extract_code(code))
    except Exception:
        return function_header

    lines = [ln.strip() for ln in function_header.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return function_header

    is_class = lines[0].startswith("class ")
    if is_class:
        if any(ln.strip().startswith("def __init__") for ln in function_header.splitlines()):
            return function_header
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == name:
                init_node = next(
                    (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
                    None,
                )
                if init_node is None:
                    return function_header
                args_text = _format_signature_args(init_node.args)
                sig_line = f"def __init__({args_text}):"
                return "\n".join([f"class {name}:", f"    {sig_line}"])
        return function_header

    if any(ln.strip().startswith("def ") and "(" in ln and ")" in ln for ln in function_header.splitlines()):
        return function_header
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args_text = _format_signature_args(node.args)
            return f"def {name}({args_text}):"
    return function_header


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


def _normalize_input_cases(
    inputs: Any,
    *,
    max_inputs: int,
    max_case_chars: int,
    allowed_kwargs: Optional[set[str]] = None,
    param_order: Optional[list[str]] = None,
) -> List[Dict[str, Any]]:
    def _list_depth(x: Any) -> int:
        d = 0
        cur = x
        while isinstance(cur, list):
            d += 1
            if not cur:
                break
            cur = cur[0]
        return d

    def _ensure_3d_list(x: Any) -> Any:
        if x is None:
            return x
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            v = float(x)
            return [[[v, v], [v, v]], [[v, v], [v, v]]]
        if isinstance(x, list):
            out = x
            d = _list_depth(out)
            while d < 3:
                out = [out]
                d += 1
            # Ensure minimal shape (>=2) along each axis for common FFT-based routines.
            # out is expected to be list[list[list[number]]].
            def _dup2(lst: Any) -> Any:
                if not isinstance(lst, list):
                    return lst
                if len(lst) >= 2:
                    return lst
                if len(lst) == 1:
                    return [lst[0], lst[0]]
                # Empty list: fall back to two zeros (keeps JSON-serializable and avoids index errors).
                return [0.0, 0.0]

            out = _dup2(out)
            if isinstance(out, list) and out:
                out[0] = _dup2(out[0])
                if len(out) >= 2:
                    out[1] = _dup2(out[1])
                if isinstance(out[0], list) and out[0]:
                    out[0][0] = _dup2(out[0][0])
                    if len(out[0]) >= 2:
                        out[0][1] = _dup2(out[0][1])
                if len(out) >= 2 and isinstance(out[1], list) and out[1]:
                    out[1][0] = _dup2(out[1][0])
                    if len(out[1]) >= 2:
                        out[1][1] = _dup2(out[1][1])

            # Avoid fully-constant cubes which can trigger degenerate k-selection in some golden implementations.
            try:
                a = out[0][0][0]
                b = out[0][0][1] if isinstance(out[0][0], list) and len(out[0][0]) > 1 else a
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b):
                    out[0][0][0] = float(a) + 1.0
            except Exception:
                pass
            return out
        return x

    if not isinstance(inputs, list):
        raise ValueError("test_inputs must be a JSON array")
    cases: List[Dict[str, Any]] = []
    for item in inputs:
        if isinstance(item, dict):
            has_explicit = "args" in item or "kwargs" in item
            if has_explicit:
                args = item.get("args") if isinstance(item.get("args"), list) else []
                kwargs = item.get("kwargs") if isinstance(item.get("kwargs"), dict) else {}
            else:
                args = []
                kwargs = dict(item)
        elif isinstance(item, list):
            args = item
            kwargs = {}
        else:
            raise ValueError("test_inputs items must be object or list")

        # Common LLM slip: wraps named params into a single dict positional arg.
        # If the dict keys look like valid parameter names, merge it into kwargs.
        if isinstance(args, list) and len(args) == 1 and isinstance(args[0], dict) and isinstance(allowed_kwargs, set):
            dict_arg = args[0]
            if dict_arg:
                all_keys_allowed = all((k in allowed_kwargs) for k in dict_arg.keys())
                has_single_allowed_key = (len(dict_arg) == 1) and next(iter(dict_arg.keys())) in allowed_kwargs
                if all_keys_allowed or has_single_allowed_key:
                    merged = {k: v for k, v in dict_arg.items() if k not in kwargs}
                    if merged:
                        kwargs = {**kwargs, **merged}
                        args = []

        if isinstance(allowed_kwargs, set):
            kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

        # Heuristic: parameters named like "*cube*" almost always expect a 3D array.
        # Keep the transformation JSON-serializable so it works for both golden execution and derived tests.
        for k in list(kwargs.keys()):
            if "cube" in str(k).lower():
                kwargs[k] = _ensure_3d_list(kwargs.get(k))

        if isinstance(param_order, list) and isinstance(args, list) and args:
            for i, val in enumerate(list(args)):
                if i >= len(param_order):
                    break
                name = str(param_order[i] or "")
                if "cube" in name.lower():
                    args[i] = _ensure_3d_list(val)
        case = {"args": args, "kwargs": kwargs}
        blob = json.dumps(case, ensure_ascii=False)
        if len(blob) > int(max_case_chars):
            raise ValueError("test_inputs item too large")
        cases.append(case)
        if len(cases) >= int(max_inputs):
            break
    if not cases:
        raise ValueError("test_inputs is empty after normalization")
    return cases


def _build_golden_script(
    *,
    dependencies: str,
    previous_steps: List[Tuple[str, str]],
    current_step_code: str,
    function_name: str,
    inputs: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    # `dependencies` is executed verbatim; only allow safe import lines here.
    for line in (dependencies or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("import ") or s.startswith("from "):
            lines.append(line.rstrip())
    if lines:
        lines.append("")

    for fn_header, raw in previous_steps:
        fn_name = _extract_function_name(fn_header)
        code = _extract_code(raw)
        fn_code = _get_function_from_code(code, fn_name) or code
        lines.extend(fn_code.splitlines())
        if lines and lines[-1] != "":
            lines.append("")

    lines.extend(_extract_code(current_step_code).splitlines())
    if lines and lines[-1] != "":
        lines.append("")

    lines.extend(
        [
            "import json",
            "import io",
            "import contextlib",
            "",
            "# Best-effort: convert JSON list inputs into numpy arrays when possible.",
            "# Check if numpy is required by the problem dependencies.",
            f"dependencies = {repr(dependencies or '')}",
            "is_numpy_required = 'numpy' in dependencies",
            "try:",
            "    import numpy as _np",
            "except ImportError:",
            "    if is_numpy_required:",
                "        raise ImportError('Numpy is required by dependencies but failed to import.')",
            "    _np = None",
            "except Exception:",
            "    _np = None",
            "",
            "# Many generated solutions use the conventional `np` alias but may omit `import numpy as np`.",
            "# Provide a safe alias when numpy is available to reduce brittle failures during golden execution.",
            "if _np is not None:",
            "    np = _np",
            "",
            "def _is_number(x):",
            "    return isinstance(x, (int, float)) and not isinstance(x, bool)",
            "",
            "def _is_numeric_tree(x):",
            "    if x is None:",
            "        return True",
            "    if isinstance(x, bool):",
            "        return True",
            "    if _is_number(x):",
            "        return True",
            "    if isinstance(x, list):",
            "        return all(_is_numeric_tree(v) for v in x)",
            "    return False",
            "",
            "def _auto_np(obj):",
            "    if _np is None:",
            "        return obj",
            "    if isinstance(obj, list):",
            "        if _is_numeric_tree(obj):",
            "            try:",
            "                return _np.array(obj)",
            "            except Exception as e:",
            "                raise TypeError(f'Failed to convert list to numpy array: {e}') from e",
            "        return [_auto_np(v) for v in obj]",
            "    if isinstance(obj, dict):",
            "        return {k: _auto_np(v) for k, v in obj.items()}",
            "    return obj",
            "",
            "def _coerce_special(obj):",
            "    if isinstance(obj, str):",
            "        s = obj.strip()",
            "        if s in {'None', 'null', 'NULL'}:",
            "            return None",
            "    if isinstance(obj, list):",
            "        return [_coerce_special(v) for v in obj]",
            "    if isinstance(obj, dict):",
            "        return {k: _coerce_special(v) for k, v in obj.items()}",
            "    return obj",
            "",
            "def _is_jsonable(obj):",
            "    try:",
            "        json.dumps(obj, ensure_ascii=False)",
            "        return True",
            "    except Exception:",
            "        return False",
            "",
            "def _to_jsonable(obj):",
            "    if obj is None or isinstance(obj, (bool, int, float, str)):",
            "        return obj",
            "    if isinstance(obj, (list, tuple)):",
            "        return [_to_jsonable(v) for v in obj]",
            "    if isinstance(obj, dict):",
            "        out = {}",
            "        for k, v in obj.items():",
            "            out[str(k)] = _to_jsonable(v)",
            "        return out",
            "    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):",
            "        try:",
            "            return _to_jsonable(obj.item())",
            "        except Exception:",
            "            pass",
            "    if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):",
            "        try:",
            "            return _to_jsonable(obj.tolist())",
            "        except Exception:",
            "            pass",
            "    if hasattr(obj, '__dict__') and isinstance(getattr(obj, '__dict__'), dict):",
            "        try:",
            "            return _to_jsonable(obj.__dict__)",
            "        except Exception:",
            "            pass",
            "    return repr(obj)",
            "",
            f"_inputs = json.loads({json.dumps(inputs, ensure_ascii=False)!r})",
            "_outputs = []",
            "for _case in _inputs:",
            "    _args = _case.get('args') or []",
            "    _kwargs = _case.get('kwargs') or {}",
            "    _args = [_auto_np(_coerce_special(v)) for v in _args]",
            "    _kwargs = {k: _auto_np(_coerce_special(v)) for k, v in _kwargs.items()}",
            "    with contextlib.redirect_stdout(io.StringIO()):",
            f"        _out = {function_name}(*_args, **_kwargs)",
            "    _out = _to_jsonable(_out)",
            "    if not _is_jsonable(_out):",
            "        raise TypeError('output_not_json_serializable')",
            "    _outputs.append(_out)",
            "print(json.dumps(_outputs, ensure_ascii=False))",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _build_test_cases(
    *,
    step_number: str,
    function_name: str,
    inputs: List[Dict[str, Any]],
    outputs: List[Any],
) -> List[ExecutableTestCase]:
    tests: List[ExecutableTestCase] = []
    for idx, (case, out) in enumerate(zip(inputs, outputs)):
        args_json = json.dumps(case.get("args") or [], ensure_ascii=False)
        kwargs_json = json.dumps(case.get("kwargs") or {}, ensure_ascii=False)
        expected_json = json.dumps(out, ensure_ascii=False)
        code = "\n".join(
            [
                "import json",
                "",
                "# Best-effort: convert JSON list inputs into numpy arrays when possible.",
                "try:",
                "    import numpy as _np",
                "except Exception:  # noqa: BLE001",
                "    _np = None",
                "",
                "if _np is not None:",
                "    np = _np",
                "",
                "def _to_jsonable(obj):",
                "    if obj is None or isinstance(obj, (bool, int, float, str)):",
                "        return obj",
                "    if isinstance(obj, (list, tuple)):",
                "        return [_to_jsonable(v) for v in obj]",
                "    if isinstance(obj, dict):",
                "        out = {}",
                "        for k, v in obj.items():",
                "            out[str(k)] = _to_jsonable(v)",
                "        return out",
                "    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):",
                "        try:",
                "            return _to_jsonable(obj.item())",
                "        except Exception:",
                "            pass",
                "    if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):",
                "        try:",
                "            return _to_jsonable(obj.tolist())",
                "        except Exception:",
                "            pass",
                "    if hasattr(obj, '__dict__') and isinstance(getattr(obj, '__dict__'), dict):",
                "        try:",
                "            return _to_jsonable(obj.__dict__)",
                "        except Exception:",
                "            pass",
                "    return repr(obj)",
                "",
                "def _is_number(x):",
                "    return isinstance(x, (int, float)) and not isinstance(x, bool)",
                "",
                "def _is_numeric_tree(x):",
                "    if x is None:",
                "        return True",
                "    if isinstance(x, bool):",
                "        return True",
                "    if _is_number(x):",
                "        return True",
                "    if isinstance(x, list):",
                "        return all(_is_numeric_tree(v) for v in x)",
                "    return False",
                "",
                "def _auto_np(obj):",
                "    if _np is None:",
                "        return obj",
                "    if isinstance(obj, list):",
                "        if _is_numeric_tree(obj):",
                "            try:",
                "                return _np.array(obj)",
                "            except Exception:  # noqa: BLE001",
                "                pass",
                "        return [_auto_np(v) for v in obj]",
                "    if isinstance(obj, dict):",
                "        return {k: _auto_np(v) for k, v in obj.items()}",
                "    return obj",
                "def _coerce_special(obj):",
                "    if isinstance(obj, str):",
                "        s = obj.strip()",
                "        if s in {'None', 'null', 'NULL'}:",
                "            return None",
                "    if isinstance(obj, list):",
                "        return [_coerce_special(v) for v in obj]",
                "    if isinstance(obj, dict):",
                "        return {k: _coerce_special(v) for k, v in obj.items()}",
                "    return obj",
                f"args = json.loads({args_json!r})",
                f"kwargs = json.loads({kwargs_json!r})",
                f"expected = json.loads({expected_json!r})",
                "args = [_auto_np(_coerce_special(v)) for v in args]",
                "kwargs = {k: _auto_np(_coerce_special(v)) for k, v in kwargs.items()}",
                f"got = {function_name}(*args, **kwargs)",
                "got = _to_jsonable(got)",
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
                "assert _deep_equal(got, expected)",
            ]
        )
        tests.append(
            ExecutableTestCase(
                test_id=f"{step_number}_test_{idx}",
                test_code=code,
                derived_from_golden=True,
            )
        )
    return tests


class TestDeriveRunner:
    def __init__(self, config: TestDeriveConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = EXECUTABLE_TEST_INPUTS_V1_EN if use_en else EXECUTABLE_TEST_INPUTS_V1
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(
        self,
        sub_step: ExecutableSubStep,
        *,
        function_header: Optional[str] = None,
        step_background: Optional[str] = None,
    ) -> str:
        payload = {
            "function_header": function_header if function_header is not None else sub_step.function_header,
            "return_line": sub_step.return_line,
            "step_description": sub_step.step_description,
            "step_background": step_background if step_background is not None else (sub_step.step_background or ""),
        }
        body = self.prompt_template.safe_substitute(payload)
        lang = (self.config.lang or "zh").lower()
        header = f"Role: ExecutableTestInputs | Language: {lang}" if lang in {"en", "english"} else f"【ExecutableTestInputs 角色】语言: {lang}"
        return "\n".join([header, "", body])

    @staticmethod
    def _extract_first_json_substring(text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        s = text.strip()
        if not s:
            return None

        starts = []
        for i, ch in enumerate(s):
            if ch in "{[":
                starts.append(i)
                break
        if not starts:
            return None

        start = starts[0]
        open_ch = s[start]
        close_ch = "}" if open_ch == "{" else "]"

        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == open_ch:
                depth += 1
                continue
            if ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1].strip()
        return None

    @staticmethod
    def _strip_json_comments(text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        out: List[str] = []
        in_str = False
        escaped = False
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if in_str:
                out.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue

            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue

            if ch == "/" and i + 1 < n:
                nxt = text[i + 1]
                if nxt == "/":
                    i += 2
                    while i < n and text[i] not in {"\n", "\r"}:
                        i += 1
                    continue
                if nxt == "*":
                    i += 2
                    while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                        i += 1
                    i += 2 if i + 1 < n else 0
                    continue

            out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def _strip_trailing_commas(text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        out: List[str] = []
        in_str = False
        escaped = False
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if in_str:
                out.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue

            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue

            if ch == ",":
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j] in {"}", "]"}:
                    i += 1
                    continue

            out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def _normalize_jsonish_literals(text: str) -> str:
        """Normalize common Python/JSON literal mismatches outside strings.

        Models sometimes mix Python literals (True/False/None) into JSON blocks.
        This function rewrites them to valid JSON tokens (true/false/null).
        """
        if not isinstance(text, str) or not text:
            return "" if text is None else str(text)
        out: List[str] = []
        i = 0
        n = len(text)
        in_str = False
        escape = False

        def _boundary_ok(prev_ch: str, next_ch: str) -> bool:
            prev_ok = (prev_ch == "") or prev_ch.isspace() or prev_ch in {":", ",", "[", "{", "("}
            next_ok = (next_ch == "") or next_ch.isspace() or next_ch in {",", "]", "}", ")", ":"}
            return prev_ok and next_ok

        repl = {
            "True": "true",
            "False": "false",
            "None": "null",
            "TRUE": "true",
            "FALSE": "false",
            "NULL": "null",
        }
        keys = sorted(repl.keys(), key=len, reverse=True)

        while i < n:
            ch = text[i]
            if in_str:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue

            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue

            prev_ch = text[i - 1] if i > 0 else ""
            replaced = False
            for k in keys:
                if text.startswith(k, i):
                    next_ch = text[i + len(k)] if i + len(k) < n else ""
                    if _boundary_ok(prev_ch, next_ch):
                        out.append(repl[k])
                        i += len(k)
                        replaced = True
                        break
            if replaced:
                continue

            out.append(ch)
            i += 1
        return "".join(out)

    @classmethod
    def _loads_json_like(cls, text: str) -> Any:
        candidates: List[str] = []

        def _add(v: Optional[str]) -> None:
            if not v:
                return
            v = str(v).strip()
            if v and v not in candidates:
                candidates.append(v)

        _add(text)
        _add(cls._extract_first_json_substring(text))

        for base in list(candidates):
            _add(cls._normalize_jsonish_literals(base))
            _add(sanitize_json_text(base))
            stripped = cls._strip_json_comments(base)
            _add(stripped)
            _add(cls._normalize_jsonish_literals(stripped))
            _add(sanitize_json_text(stripped))
            _add(cls._strip_trailing_commas(stripped))
            _add(sanitize_json_text(cls._strip_trailing_commas(stripped)))
            _add(cls._normalize_jsonish_literals(cls._strip_trailing_commas(stripped)))

        last_err: Optional[Exception] = None
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue

        # Final fallback: allow Python literals (safe) when model emits JSON-ish objects.
        for cand in candidates:
            try:
                return ast.literal_eval(cand)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue

        raise ValueError(f"ExecutableTestInputs output is not parseable as JSON-like data: {last_err}")

    def _parse_inputs(self, text: str, *, function_header: str) -> List[Dict[str, Any]]:
        if not text or not str(text).strip():
            raise ValueError("ExecutableTestInputs output is empty")
        candidate = str(text).strip()
        think_end = "</think>"
        if think_end in candidate:
            candidate = candidate.split(think_end, 1)[1].strip()

        fenced = extract_preferred_fenced_block(candidate, preferred_langs=("json",))
        if fenced:
            candidate = fenced

        data = self._loads_json_like(candidate)

        raw_inputs: Any = None
        if isinstance(data, list):
            raw_inputs = data
        elif isinstance(data, dict):
            raw_inputs = data.get(FIELD_TEST_INPUTS)
            if raw_inputs is None:
                for alt_key in ("inputs", "cases", "test_cases", "tests", "examples"):
                    if alt_key in data:
                        raw_inputs = data[alt_key]
                        break
        else:
            raise ValueError("ExecutableTestInputs output is not a JSON object or array")
        allowed_kwargs = _extract_allowed_kwargs_from_header(function_header)
        param_order = _extract_param_order_from_header(function_header)
        return _normalize_input_cases(
            raw_inputs,
            max_inputs=int(self.config.max_inputs),
            max_case_chars=int(self.config.max_case_chars),
            allowed_kwargs=allowed_kwargs,
            param_order=param_order,
        )

    def _execute_golden(self, derive_in: TestDeriveInput, inputs: List[Dict[str, Any]]) -> List[Any]:
        step_number = derive_in.step_number
        step = None
        for s in derive_in.sub_steps:
            if s.step_number == step_number:
                step = s
                break
        if step is None:
            step = derive_in.sub_steps[-1] if derive_in.sub_steps else None
        if step is None:
            raise ValueError("sub_steps is empty")

        try:
            fn_name = _extract_function_name(step.function_header)
        except Exception:
            # Be robust to imperfect draft schema: infer from golden code when header is missing.
            fn_name = _extract_function_name(derive_in.golden_step_code)
        previous_steps: List[Tuple[str, str]] = []
        for s in derive_in.sub_steps:
            if s.step_number == step_number:
                break
            if not s.step_number:
                continue
            previous_steps.append((s.function_header, derive_in.per_step_golden.get(s.step_number, "")))

        script = _build_golden_script(
            dependencies=str(derive_in.required_dependencies or ""),
            previous_steps=previous_steps,
            current_step_code=derive_in.golden_step_code,
            function_name=fn_name,
            inputs=inputs,
        )

        conf = ExecutionConfig(
            timeout=float(self.config.step_timeout),
            memory_limit_mb=int(self.config.memory_mb),
            temp_dir=str(self.config.temp_dir),
            python_bin=str(self.config.python_bin),
        )
        executor = PythonExecutor(conf)
        result: ExecutionResult = asyncio.run(executor.execute_single(script))
        if not result.success:
            raise RuntimeError(f"golden_execute_failed: {result.error}")
        out_text = (result.output or "").strip()
        if not out_text:
            raise RuntimeError("golden_execute_empty_output")
        try:
            outputs = json.loads(out_text)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"golden_execute_invalid_json: {exc}") from exc
        if not isinstance(outputs, list):
            raise RuntimeError("golden_execute_output_not_list")
        return outputs

    def run_one(
        self,
        derive_in: TestDeriveInput,
        *,
        step_dir: Optional[Path] = None,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.executable_test_inputs.",
        derive_dirname: str = "derive",
    ) -> ExecutableTestDeriveOutput:
        step_number = derive_in.step_number
        step = next((s for s in derive_in.sub_steps if s.step_number == step_number), None)
        if step is None:
            raise ValueError(f"step_number={step_number!r} not found in sub_steps")

        effective_header = _infer_function_header_from_code(step.function_header, derive_in.golden_step_code)

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ExecutableTestInputs snapshot write failed: %s", str(exc))
                snap_path = None

        if unified_prompt_dir is not None:
            try:
                snapshot_prompt_used(
                    Path(str(self.config.prompt_path)),
                    Path(unified_prompt_dir),
                    content=self.prompt_text,
                    name_prefix=name_prefix,
                    logger=logger,
                )
            except Exception:
                pass

        # 1) Generate & parse inputs (retry on empty/parse failure).
        prompt_body = ""
        text = ""
        inputs: List[Dict[str, Any]] = []
        max_input_retries = max(0, int(getattr(self.config, "input_retries", 0)))
        last_parse_exc: Optional[Exception] = None
        for attempt in range(max_input_retries + 1):
            prompt_body = self._build_prompt(step, function_header=effective_header)
            messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
            response = self.session.chat(messages, **self._chat_args)
            BaseSkillRunner._check_finish_reason(response, f"ExecutableTestInputs(attempt={attempt})")
            text = self.session.extract_text(response, default="")

            if snap_path is not None:
                try:
                    attempt_dir = snap_path if attempt == 0 else (snap_path / f"retry_{attempt}")
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    if attempt == 0:
                        input_payload = {
                            "step_number": step_number,
                            "function_header": step.function_header,
                            "function_header_effective": effective_header,
                            "return_line": step.return_line,
                            "step_description": step.step_description,
                            "step_background": step.step_background,
                        }
                        (attempt_dir / "input_view.json").write_text(
                            json.dumps(input_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        snapshot_prompt_used(
                            Path(str(self.config.prompt_path)),
                            attempt_dir,
                            content=self.prompt_text,
                            name_prefix="prompt_used.",
                            logger=logger,
                        )
                    snapshot_rendered_prompt(prompt_body, attempt_dir, filename="prompt_rendered.txt", logger=logger)
                    request_meta = {
                        "model": self.session.model_name,
                        "service_id": self.session.service_id,
                        "chat_args": self._chat_args,
                    }
                    (attempt_dir / "request_meta.json").write_text(
                        json.dumps(request_meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    try:
                        (attempt_dir / "raw_response.json").write_text(
                            json.dumps(response, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                    (attempt_dir / "raw_text.txt").write_text(text or "", encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ExecutableTestInputs snapshot write failed: %s", str(exc))

            try:
                inputs = self._parse_inputs(text, function_header=effective_header)
                last_parse_exc = None
                if snap_path is not None:
                    try:
                        attempt_dir = snap_path if attempt == 0 else (snap_path / f"retry_{attempt}")
                        (attempt_dir / "output.json").write_text(
                            json.dumps({FIELD_TEST_INPUTS: inputs}, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("ExecutableTestInputs snapshot write failed: %s", str(exc))
                break
            except Exception as exc:  # noqa: BLE001
                last_parse_exc = exc
                if snap_path is not None:
                    try:
                        attempt_dir = snap_path if attempt == 0 else (snap_path / f"retry_{attempt}")
                        (attempt_dir / "parse_error.txt").write_text(str(exc), encoding="utf-8")
                    except Exception:
                        pass
                if attempt >= max_input_retries:
                    raise
                continue

        max_retries = max(0, int(getattr(self.config, "golden_retries", 0)))
        outputs: List[Any] = []
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                outputs = self._execute_golden(derive_in, inputs)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= max_retries:
                    raise
                err_text = _truncate_text(str(exc), 1600)
                bg = (step.step_background or "").strip()
                bg = (bg + "\n\n[Golden execution error]\n" + err_text).strip()
                retry_prompt_body = self._build_prompt(step, function_header=effective_header, step_background=bg)
                retry_messages = build_messages_with_background(retry_prompt_body, lang=self.config.lang or "zh")
                retry_response = self.session.chat(retry_messages, **self._chat_args)
                BaseSkillRunner._check_finish_reason(retry_response, f"ExecutableTestInputs(retry_{attempt + 1})")
                retry_text = self.session.extract_text(retry_response, default="")
                retry_snap: Optional[Path] = None
                if snap_path is not None:
                    try:
                        retry_snap = snap_path / f"retry_{attempt + 1}"
                        retry_snap.mkdir(parents=True, exist_ok=True)
                        snapshot_rendered_prompt(retry_prompt_body, retry_snap, filename="prompt_rendered.txt", logger=logger)
                        request_meta = {
                            "model": self.session.model_name,
                            "service_id": self.session.service_id,
                            "chat_args": self._chat_args,
                            "retry_for_error": err_text,
                        }
                        (retry_snap / "request_meta.json").write_text(
                            json.dumps(request_meta, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        try:
                            (retry_snap / "raw_response.json").write_text(
                                json.dumps(retry_response, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                        (retry_snap / "raw_text.txt").write_text(retry_text or "", encoding="utf-8")
                    except Exception as snap_exc:  # noqa: BLE001
                        logger.warning("ExecutableTestInputs retry snapshot write failed: %s", str(snap_exc))
                        retry_snap = None
                try:
                    inputs = self._parse_inputs(retry_text, function_header=effective_header)
                except Exception as parse_exc:  # noqa: BLE001
                    if retry_snap is not None:
                        try:
                            (retry_snap / "parse_error.txt").write_text(str(parse_exc), encoding="utf-8")
                        except Exception:
                            pass
                    raise
                if retry_snap is not None:
                    try:
                        (retry_snap / "output.json").write_text(
                            json.dumps({FIELD_TEST_INPUTS: inputs}, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

        test_cases = _build_test_cases(
            step_number=step_number,
            function_name=_extract_function_name(step.function_header),
            inputs=inputs,
            outputs=outputs,
        )

        inputs_rel = ""
        inputs_sha = ""
        if isinstance(step_dir, Path):
            safe_name = str(derive_dirname or "derive").strip()
            if not safe_name:
                safe_name = "derive"
            derive_dir = step_dir / safe_name
            derive_dir.mkdir(parents=True, exist_ok=True)
            inputs_path = derive_dir / "inputs.jsonl"
            try:
                with inputs_path.open("w", encoding="utf-8") as f:
                    for row in inputs:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                inputs_rel = str(inputs_path.relative_to(step_dir))
                sha256 = hashlib.sha256()
                sha256.update(inputs_path.read_bytes())
                inputs_sha = sha256.hexdigest()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"write_inputs_failed: {exc}") from exc

        return ExecutableTestDeriveOutput(
            inputs=inputs,
            inputs_artifact_relpath=inputs_rel,
            inputs_sha256=inputs_sha,
            test_cases_for_step=test_cases,
        )


__all__ = [
    "TestDeriveInput",
    "TestDeriveConfig",
    "TestDeriveRunner",
]
