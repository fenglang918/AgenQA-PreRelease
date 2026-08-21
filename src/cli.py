"""AgenQA 命令行接口"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import importlib.util
import json
import sys
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
from typing import Dict, Any, Optional
import logging
import subprocess
import shutil
from uuid import uuid4
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_CLI_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CLI_DIR.parent if _CLI_DIR.name == "src" else _CLI_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import load_config, ensure_dir
from infra.llm.service_loader import load_llm_service_full_config
from infra.logging import setup_logger
from agenqa.skills.solving import SolverRunner, SolverConfig
from agenqa.skills.head_tail import HeadTailComposer, HeadTailConfig
from agenqa.graph import run_episode

# 允许通过环境变量覆盖 llm_service services.json 路径
DEFAULT_SERVICE_CONFIG = Path(
    os.getenv(
        "LLM_SERVICES_JSON",
        "config/services.json",
    )
)

# Portable public defaults. Provider-specific endpoints can still be supplied
# through their existing environment variables or --base-url.
SJTU_DEFAULT_BASE_URL = os.getenv("SJTU_BASE_URL", "https://api.openai.com/v1")
SJTU_DEFAULT_CONFIG = "config/agent_openai.yaml"
SJTU_DEFAULT_CONFIG_EN = "config/agent_openai.yaml"
IDEALAB_DEFAULT_BASE_URL = os.getenv("IDEALAB_BASE_URL", "")
IDEALAB_DEFAULT_CONFIG = "config/agent_openai.yaml"
IDEALAB_DEFAULT_CONFIG_EN = "config/agent_openai.yaml"
AIMUX_DEFAULT_BASE_URL = os.getenv("AIMUX_BASE_URL", "")
DMXAPI_DEFAULT_BASE_URL = os.getenv("DMXAPI_BASE_URL", "https://www.dmxapi.cn/v1")


def setup_basic_logging():
    """设置基本日志（简洁格式，突出关键信息）"""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    # 简洁格式：时间(HH:MM:SS) [级别] 模块简名: 消息
    console_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s [%(levelname).1s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    )
    root_logger.addHandler(console_handler)


## 已迁移到 infra/llm/service_loader.py：load_llm_service_full_config


def _sanitize_component(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    cleaned = re.sub(r"[^\w.-]+", "-", value)
    cleaned = cleaned.strip("-_")
    return cleaned or fallback


def _clone_resume_run_dir(orig: Path, *, suffix: str) -> Path:
    """Clone a previous run dir to a new sibling dir for resume comparisons.

    The new directory name is: {orig.name}_{suffix}{n}, where n starts from 1.
    """
    if not orig.exists():
        raise SystemExit(f"[ERROR] resume_run_dir does not exist: {orig}")
    parent = orig.parent
    base_name = orig.name
    suffix_norm = _sanitize_component(suffix, "resume")
    n = 1
    while True:
        candidate = parent / f"{base_name}_{suffix_norm}{n}"
        if not candidate.exists():
            return candidate
        n += 1


def _next_archive_attempt_dir(archive_root: Path) -> tuple[Path, int]:
    archive_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = archive_root / f"attempt_{n:03d}"
        if not candidate.exists():
            return candidate, n
        n += 1


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _find_latest_round_dir(run_dir: Path) -> Optional[Path]:
    latest_num = -1
    latest_path: Optional[Path] = None
    for candidate in run_dir.glob("round_*"):
        if not candidate.is_dir():
            continue
        m = re.fullmatch(r"round_(\d+)", candidate.name)
        if not m:
            continue
        num = int(m.group(1))
        if num > latest_num:
            latest_num = num
            latest_path = candidate
    return latest_path


def _read_state_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _append_resume_history_line(root_dir: Path, payload: Dict[str, Any]) -> None:
    history_path = root_dir / "resume_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _prepare_run_dir_for_in_place_resume(
    run_dir: Path,
    *,
    resume_reason: str,
    batch_id: Optional[str] = None,
    batch_archive_attempt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"[ERROR] resume_run_dir does not exist: {run_dir}")
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise SystemExit(f"[ERROR] missing state.json under resume run dir: {run_dir}")

    archive_root = run_dir / "_resume_archive"
    attempt_dir, attempt_index = _next_archive_attempt_dir(archive_root)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    latest_round_dir = _find_latest_round_dir(run_dir)
    root_candidates = [
        state_path,
        run_dir / "run.log",
        run_dir / "run_config.json",
        run_dir / "00_Summary",
        run_dir / "00_Prompts_Snapshot",
    ]
    root_candidates.extend(sorted(run_dir.glob("run_playback_*.md")))
    if latest_round_dir is not None:
        root_candidates.append(latest_round_dir)

    archived_paths: list[str] = []
    removed_paths: list[str] = []
    for src in root_candidates:
        if not src.exists():
            continue
        dst = attempt_dir / src.name
        _copy_path(src, dst)
        archived_paths.append(str(src.relative_to(run_dir)))

    removable = [
        run_dir / "run.log",
        run_dir / "00_Summary",
        run_dir / "00_Prompts_Snapshot",
    ]
    removable.extend(sorted(run_dir.glob("run_playback_*.md")))
    if latest_round_dir is not None:
        removable.append(latest_round_dir)
    for path in removable:
        if not path.exists():
            continue
        removed_paths.append(str(path.relative_to(run_dir)))
        _remove_path(path)

    state_snapshot = _read_state_snapshot(state_path)
    manifest = {
        "attempt_index": attempt_index,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "resume_reason": resume_reason,
        "run_dir": str(run_dir),
        "batch_id": batch_id or "",
        "batch_archive_attempt_dir": str(batch_archive_attempt_dir) if batch_archive_attempt_dir else "",
        "source_stop_reason": state_snapshot.get("stop_reason"),
        "source_error": "",
        "archived_paths": archived_paths,
        "removed_paths": removed_paths,
        "resumed_from_state": {
            "step": state_snapshot.get("step"),
            "history_len": len(state_snapshot.get("history") or []),
            "stop_reason": state_snapshot.get("stop_reason"),
        },
        "post_resume_state": None,
    }
    (attempt_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_resume_history_line(
        run_dir,
        {
            "timestamp": manifest["archived_at"],
            "event_type": "resume_archive_created",
            "attempt_index": attempt_index,
            "resume_reason": resume_reason,
            "run_dir": str(run_dir),
            "batch_id": batch_id or "",
            "source_stop_reason": manifest["source_stop_reason"],
            "archived_paths": archived_paths,
        },
    )
    return {
        "run_dir": run_dir,
        "archive_dir": attempt_dir,
        "attempt_index": attempt_index,
    }


def _finalize_run_resume_archive(
    run_dir: Path,
    *,
    success: bool,
    error: str = "",
) -> None:
    run_dir = run_dir.resolve()
    archive_root = run_dir / "_resume_archive"
    if not archive_root.exists():
        return
    attempts = sorted(
        [p for p in archive_root.glob("attempt_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not attempts:
        return
    attempt_dir = attempts[-1]
    manifest_path = attempt_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return
    except Exception:
        return
    state_snapshot = _read_state_snapshot(run_dir / "state.json")
    manifest["source_error"] = manifest.get("source_error") or error
    manifest["post_resume_state"] = {
        "success": bool(success),
        "error": error,
        "step": state_snapshot.get("step"),
        "history_len": len(state_snapshot.get("history") or []),
        "stop_reason": state_snapshot.get("stop_reason"),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_resume_history_line(
        run_dir,
        {
            "timestamp": manifest["post_resume_state"]["finalized_at"],
            "event_type": "resume_archive_finalized",
            "attempt_index": manifest.get("attempt_index"),
            "run_dir": str(run_dir),
            "success": bool(success),
            "error": error,
            "stop_reason": state_snapshot.get("stop_reason"),
        },
    )


def _materialize_run_dir(
    *,
    source_run_dir: Path,
    target_run_dir: Path,
    archive_dir: Optional[Path],
) -> Dict[str, Any]:
    source_run_dir = source_run_dir.resolve()
    target_run_dir = target_run_dir.resolve()
    if source_run_dir == target_run_dir:
        return {
            "materialized": False,
            "source_run_dir": str(source_run_dir),
            "target_run_dir": str(target_run_dir),
            "archived_previous_target": "",
        }

    archived_previous_target = ""
    if target_run_dir.exists():
        if archive_dir is not None:
            backup_dir = archive_dir / "pre_materialize_runs" / target_run_dir.name
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
            _copy_path(target_run_dir, backup_dir)
            archived_previous_target = str(backup_dir)
        _remove_path(target_run_dir)

    target_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, target_run_dir)
    return {
        "materialized": True,
        "source_run_dir": str(source_run_dir),
        "target_run_dir": str(target_run_dir),
        "archived_previous_target": archived_previous_target,
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts, giving precedence to `override`.

    - Copies `base` and overlays keys from `override`.
    - Nested dicts are merged recursively.
    """
    from copy import deepcopy

    result: Dict[str, Any] = deepcopy(base)
    for k, v in (override or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _now_id() -> str:
    """返回北京时间时间戳（用于 run_id 等），格式：YYYYmmdd-HHMMSS。"""
    bj_tz = timezone(timedelta(hours=8))
    return datetime.now(timezone.utc).astimezone(bj_tz).strftime('%Y%m%d-%H%M%S')


def _new_run_id(prefix: Optional[str] = None) -> str:
    """Return a collision-resistant run id for concurrent launches."""
    bj_tz = timezone(timedelta(hours=8))
    ts = datetime.now(timezone.utc).astimezone(bj_tz).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = f"{ts}_{uuid4().hex[:6]}"
    prefix_norm = _sanitize_component(prefix, "") if prefix else ""
    return f"{prefix_norm}_{base}" if prefix_norm else base


def _summarize_generator_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary of key generator config for logging."""
    client = cfg.get('client') or {}
    client = client if isinstance(client, dict) else {}
    gen = cfg.get('generation') or {}
    gen = gen if isinstance(gen, dict) else {}
    gen_compact = {k: v for k, v in gen.items() if v is not None}
    return {
        'service_id': cfg.get('service_id'),
        'api_base': cfg.get('api_base') or cfg.get('base_url'),
        'model_name': cfg.get('model_name'),
        'client': {
            'timeout': client.get('timeout'),
            'stream': client.get('stream'),
        },
        'generation': gen_compact,
    }


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_assignment(raw: str) -> tuple[str, str]:
    if not isinstance(raw, str):
        raise ValueError("env assignment must be a string")
    if "=" not in raw:
        raise ValueError(f"invalid env assignment (expected KEY=VALUE): {raw!r}")
    key, val = raw.split("=", 1)
    key = key.strip()
    if not key or not _ENV_KEY_RE.match(key):
        raise ValueError(f"invalid env var name: {key!r}")
    return key, val


def _apply_env_assignments(
    assignments: Any,
    *,
    override: bool,
    source: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not assignments:
        return
    if not isinstance(assignments, list):
        assignments = [assignments]
    for raw in assignments:
        try:
            key, val = _parse_env_assignment(str(raw))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[ERROR] invalid {source}: {raw!r} ({exc})") from exc
        if not override and os.environ.get(key) is not None:
            continue
        os.environ[key] = val
        if logger:
            logger.info("Set env from %s: %s=%s", source, key, val if val else "")


def _apply_env_unsets(
    keys: Any,
    *,
    source: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not keys:
        return
    if not isinstance(keys, list):
        keys = [keys]
    for raw in keys:
        key = str(raw or "").strip()
        if not key or not _ENV_KEY_RE.match(key):
            raise SystemExit(f"[ERROR] invalid env var name for {source}: {key!r}")
        if key in os.environ:
            os.environ.pop(key, None)
            if logger:
                logger.info("Unset env from %s: %s", source, key)


def _apply_runtime_env_from_config(
    config: Dict[str, Any],
    *,
    override: bool,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not isinstance(config, dict):
        return

    merged: Dict[str, Any] = {}
    top = config.get("runtime_env")
    if isinstance(top, dict):
        merged.update(top)
    agent_block = config.get("agent")
    if isinstance(agent_block, dict) and isinstance(agent_block.get("runtime_env"), dict):
        merged.update(agent_block.get("runtime_env") or {})

    if not merged:
        return

    for key, val in merged.items():
        key_str = str(key or "").strip()
        if not key_str or not _ENV_KEY_RE.match(key_str):
            raise SystemExit(f"[ERROR] invalid runtime_env key in config: {key!r}")
        if val is None:
            if key_str in os.environ:
                os.environ.pop(key_str, None)
                if logger:
                    logger.info("Unset env from config runtime_env: %s", key_str)
            continue
        if not override and os.environ.get(key_str) is not None:
            continue
        val_str = os.path.expandvars(str(val))
        os.environ[key_str] = val_str
        if logger:
            logger.info("Set env from config runtime_env: %s=%s", key_str, val_str if val_str else "")


def _force_client_stream(cfg: Any) -> None:
    """Recursively set client.stream=True for all generator blocks in a config dict."""
    if isinstance(cfg, dict):
        for key, val in cfg.items():
            if key == "generator" and isinstance(val, dict):
                client = val.get("client")
                if not isinstance(client, dict):
                    client = {}
                    val["client"] = client
                client["stream"] = True
            else:
                _force_client_stream(val)
    elif isinstance(cfg, list):
        for item in cfg:
            _force_client_stream(item)


def _parse_no_mcq_from_step(raw: Any) -> int | None:
    """Parse CLI/config value for no_mcq_from_step.

    - int>=1: disallow MCQ starting from that step (next_step>=N)
    - 0/off/none/null/disable: disable the policy (allow MCQ)
    - None: no override (caller decides)
    """
    if raw is None:
        return None
    if raw is False:
        return None
    if isinstance(raw, (int, float)):
        try:
            v = int(raw)
            return v if v >= 1 else None
        except Exception:
            return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            return None
        if s in {"off", "none", "null", "disable", "disabled"}:
            return None
        try:
            v = int(s)
            return v if v >= 1 else None
        except Exception:
            return None
    return None


def _normalize_question_type_name(raw: Any) -> str | None:
    """Normalize question type label into MCQ/Derivation/Numeric."""
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    if not v:
        return None
    if v in {"mcq", "choice", "single", "single_choice", "single-choice"}:
        return "MCQ"
    if v in {"derivation", "derive", "symbolic", "proof", "algebra", "formula"}:
        return "Derivation"
    if v in {"numeric", "calc", "calculation", "compute", "number", "estimate", "estimation", "approx", "approximation"}:
        return "Numeric"
    return None


def _parse_allowed_question_types(raw: Any) -> list[str] | None:
    """Parse CLI/config value for allowed_question_types.

    - list/tuple of strings: normalize and dedupe, preserving order
    - str: split by comma/whitespace
    - off/none/null/disable: return None (disable whitelist)
    - invalid/empty: raise ValueError (fail-fast)
    """
    if raw is None:
        return None
    if raw is False:
        return None

    items: list[str] = []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s.strip().lower() in {"off", "none", "null", "disable", "disabled"}:
            return None
        items = [p for p in re.split(r"[,\s]+", s) if p]
    elif isinstance(raw, (list, tuple)):
        for x in raw:
            if x is None:
                continue
            if isinstance(x, str):
                xs = x.strip()
                if not xs:
                    continue
                # Allow "MCQ,Derivation" in a single token.
                items.extend([p for p in re.split(r"[,\s]+", xs) if p])
            else:
                items.append(str(x))
    else:
        raise ValueError(f"invalid allowed_question_types type={type(raw)!r} (expected list[str] or str)")

    if len(items) == 1 and str(items[0]).strip().lower() in {"off", "none", "null", "disable", "disabled"}:
        return None

    if not items:
        raise ValueError("allowed_question_types is empty")

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        qt = _normalize_question_type_name(item)
        if not qt:
            raise ValueError(f"invalid question type in allowed_question_types: {item!r} (expected MCQ/Derivation/Numeric)")
        if qt in seen:
            continue
        seen.add(qt)
        out.append(qt)
    if not out:
        raise ValueError("allowed_question_types is empty after normalization")
    return out


def _override_allowed_question_types(config: Dict[str, Any], raw: Any, *, logger: logging.Logger | None = None) -> None:
    """Override agent.question_type_policy.allowed_question_types in-place (CLI helper)."""
    parsed = _parse_allowed_question_types(raw)
    agent_block = config.setdefault("agent", {})
    if not isinstance(agent_block, dict):
        raise ValueError("config.agent must be a dict to override allowed_question_types")
    policy = agent_block.get("question_type_policy")
    policy = policy if isinstance(policy, dict) else {}
    policy = dict(policy)
    if parsed is None:
        policy.pop("allowed_question_types", None)
        agent_block["question_type_policy"] = policy
        if logger:
            logger.info("命令行覆盖: agent.question_type_policy.allowed_question_types=off")
        return
    policy["allowed_question_types"] = list(parsed)
    agent_block["question_type_policy"] = policy
    if logger:
        logger.info("命令行覆盖: agent.question_type_policy.allowed_question_types=%s", ",".join(parsed))


def _reorder_global_cli_args(argv: list[str], *, commands: set[str]) -> list[str]:
    """允许将全局参数放在 subcommand 之后（更贴近常见 CLI 习惯）。

    argparse 对 subparsers 的默认行为是：一旦遇到 subcommand，后续参数都会交给子解析器；
    因此像 `src/cli.py agent -c xxx.yaml ...` 这种写法会报 “unrecognized arguments: -c ...”。

    这里做一个轻量预处理：把 subcommand 之后出现的 `-c/--config` 与 `--log-level`
    重新挪回 subcommand 之前，让两种写法都可用。
    """
    if not argv or not commands:
        return argv

    # 找到 subcommand 的位置（跳过全局参数的 value，避免 value 恰好命中 command 名）
    command_index: Optional[int] = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-c", "--config", "--log-level"):
            i += 2
            continue
        if tok.startswith("--config=") or tok.startswith("--log-level="):
            i += 1
            continue
        if tok in commands:
            command_index = i
            break
        i += 1

    if command_index is None:
        return argv

    prefix = argv[:command_index]
    command = argv[command_index]
    suffix = argv[command_index + 1 :]

    extracted: list[str] = []
    kept: list[str] = []
    j = 0
    while j < len(suffix):
        tok = suffix[j]
        if tok in ("-c", "--config", "--log-level"):
            extracted.append(tok)
            if j + 1 < len(suffix):
                extracted.append(suffix[j + 1])
                j += 2
            else:
                j += 1
            continue
        if tok.startswith("--config=") or tok.startswith("--log-level="):
            extracted.append(tok)
            j += 1
            continue
        kept.append(tok)
        j += 1

    if not extracted:
        return argv

    return prefix + extracted + [command] + kept


def _preflight_validate_agent_papers_path(config: Dict[str, Any], logger: logging.Logger) -> None:
    """在运行 agent-run 前检查论文输入路径，避免深层栈报错。"""
    try:
        init_conf = config.get("init") if isinstance(config, dict) else None
        if not isinstance(init_conf, dict):
            logger.error("agent-run 需要顶层 init 配置块（例如 init.source.path / init.generator）。")
            logger.error("示例：python3 src/cli.py -c config/agent_paper_path_v2.yaml agent-run ...")
            sys.exit(2)
        src_conf = init_conf.get("source") if isinstance(init_conf.get("source"), dict) else None
        if not isinstance(src_conf, dict):
            logger.error("agent-run 需要 init.source（例如 init.source.path）。")
            sys.exit(2)

        source_type = str(src_conf.get("type") or "paper").strip().lower() or "paper"
        if source_type not in {"paper", "paper-like", "paperlike"}:
            return

        raw_value = src_conf.get("path")
        if not isinstance(raw_value, str) or not raw_value.strip():
            logger.error("agent-run 需要 init.source.path（支持 .txt/.json/.jsonl/.pdf；建议用 env 覆盖：SCICLONE_PAPER_PATH）。")
            sys.exit(2)
        raw_value = raw_value.strip()

        # 兜底一次环境变量展开（防止非标准加载路径导致 ${VAR} 未被替换）
        expanded = os.path.expandvars(raw_value)
        if expanded != raw_value:
            src_conf["path"] = expanded
            raw_value = expanded

        m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", raw_value)
        if m:
            env_name = m.group(1)
            logger.error("配置 init.source.path=%s 未解析（环境变量 %s 未设置）。", raw_value, env_name)
            logger.error(
                '请先设置：export %s="experiments/upstream/generation/_template/inputs/papers_selected/paper_cf2525e10390.txt"',
                env_name,
            )
            logger.error("或在配置文件里将 init.source.path 改为一个存在的 .txt/.json/.jsonl/.pdf 路径。")
            sys.exit(2)

        papers_path = Path(raw_value)
        if not papers_path.exists():
            logger.error("论文输入不存在：init.source.path=%s", raw_value)
            logger.error(
                '示例：export SCICLONE_PAPER_PATH="experiments/upstream/generation/_template/inputs/papers_selected/paper_cf2525e10390.txt"'
            )
            logger.error('或（SciPedia）：export SCICLONE_SCIPEDIA_PATH="/path/to/CPT_scipedia.jsonl"')
            sys.exit(2)
    except SystemExit:
        raise
    except Exception:
        # 预检失败不应阻断主流程（若后续仍失败会打印完整异常栈）
        logger.warning("papers_path 预检失败，继续执行（可能在后续节点报错）", exc_info=True)


IDEALAB_MAIN_MODEL_ALIASES = {
    # 常用主模型的简写序号/别名（仅用于 agent --source idealab）
    "1": "gpt-51-1113-global",
    "2": "qwen3-max",
    "3": "gemini-2.5-flash-06-17",
    "4": "gemini-2.5-pro-06-17",
    "5": "gemini-3-pro-preview",
    "6": "gpt-5-0807-global",
    "7": "claude_sonnet4_5",
    "8": "gpt-5-mini-0807-global",
    "9": "gpt-5.2-1211-global",
    "10": "qwen3-next-80b-a3b-thinking",
}

def _sanitize_generation_for_model(
    generator: Dict[str, Any],
    model_name: str,
    *,
    logger: Optional[logging.Logger] = None,
    context: str = "",
) -> None:
    """Best-effort sanitize generation params for specific models.

    Some backends only accept default values for certain params (e.g. temperature).
    When users override model_name via CLI aliases, we keep existing generation
    settings; this helper prevents avoidable hard failures.
    """
    if not isinstance(generator, dict):
        return
    generation = generator.get("generation")
    if not isinstance(generation, dict):
        return

    model_lc = str(model_name or "").strip().lower()
    # Observed via Idealab gateway: gpt-5-mini only supports default temperature (1).
    # Passing any explicit temperature can trigger "unsupported_value".
    if "gpt-5-mini" in model_lc:
        if "temperature" in generation:
            old = generation.pop("temperature", None)
            if logger and old is not None:
                prefix = f"{context}: " if context else ""
                logger.info("%s移除不兼容参数: generation.temperature=%s（model=%s）", prefix, old, model_name)


def _resolve_idealab_main_model_alias(raw: str) -> str:
    """将 --main-model 的数字序号/简写解析为 Idealab 模型名。

    - 支持数字序号（\"1\"/\"2\"/...），方便快速切换主模型；
    - 未命中别名时原样返回（兼容直接传完整模型名）。
    """
    key = str(raw).strip()
    if not key:
        return key
    return IDEALAB_MAIN_MODEL_ALIASES.get(key, key)


def _override_idealab_main_model(config: Dict[str, Any], main_model: str, logger: Optional[logging.Logger] = None) -> None:
    """Override the primary Idealab model across instruct roles & strong solver.

    This is a convenience for agent --source idealab, allowing a single
    CLI flag to switch the main model without editing YAML:
    - init.generator.model_name
    - init.paper_brief.generator.model_name (optional)
    - init.episode_seed.generator.model_name (optional override)
    - director.generator.model_name
    - operators.extend.generator.model_name
    - operators.revise.generator.model_name
    - solvers.strong.generator.model_name
    """
    targets = [
        ("init",),
        ("init", "paper_brief"),
        ("init", "episode_seed"),
        ("director",),
        ("operators", "extend"),
        ("operators", "revise"),
        ("solvers", "strong"),
    ]
    changed: list[str] = []

    for path in targets:
        cur: Any = config
        for seg in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(seg)
        # solvers.strong can be a dict (legacy) or a list (multi-strong).
        if path == ("solvers", "strong") and isinstance(cur, list):
            primary = None
            for item in cur:
                if isinstance(item, dict) and item.get("primary") is True:
                    primary = item
                    break
            if primary is None:
                primary = cur[0] if cur and isinstance(cur[0], dict) else None
            if not isinstance(primary, dict):
                continue
            gen = primary.get("generator")
            if not isinstance(gen, dict):
                continue
            old = gen.get("model_name")
            gen["model_name"] = main_model
            path_str = ".".join(path) + "[primary]"
            _sanitize_generation_for_model(gen, main_model, logger=logger, context=path_str)
            if old and old != main_model:
                changed.append(f"{path_str}: {old} -> {main_model}")
            else:
                changed.append(f"{path_str}: {main_model}")
            continue

        if not isinstance(cur, dict):
            continue
        gen = cur.get("generator")
        if not isinstance(gen, dict):
            continue
        old = gen.get("model_name")
        gen["model_name"] = main_model
        path_str = ".".join(path)
        _sanitize_generation_for_model(gen, main_model, logger=logger, context=path_str)
        if old and old != main_model:
            changed.append(f"{path_str}: {old} -> {main_model}")
        else:
            changed.append(f"{path_str}: {main_model}")

    if logger and changed:
        logger.info("Idealab 主模型覆盖 (--main-model=%s): %s", main_model, "; ".join(changed))


def _parse_model_list(raw: str) -> list[str]:
    """Parse a comma/space separated model list."""
    if not isinstance(raw, str):
        return []
    tokens = re.split(r"[,\\s]+", raw.strip())
    return [t for t in (tok.strip() for tok in tokens) if t]


def _override_strong_models(config: Dict[str, Any], model_names: list[str], logger: Optional[logging.Logger] = None) -> None:
    """Override solvers.strong into a list using model_names (first becomes primary).

    This is a convenience for agent runs, to avoid editing YAML when trying
    multi-strong consensus quickly.
    """
    if not model_names:
        return
    solvers = config.setdefault("solvers", {})
    strong = solvers.get("strong")

    # Choose a template strong config to clone.
    from copy import deepcopy

    template: Optional[Dict[str, Any]] = None
    templates: list[Dict[str, Any]] = []
    if isinstance(strong, dict):
        template = strong
    elif isinstance(strong, list):
        templates = [it for it in strong if isinstance(it, dict)]
        template = templates[0] if templates else None

    if template is None:
        return

    new_list: list[Dict[str, Any]] = []
    for idx, model_name in enumerate(model_names):
        base = templates[idx] if idx < len(templates) else template
        conf = deepcopy(base) if isinstance(base, dict) else {}
        conf["primary"] = idx == 0
        gen = conf.get("generator")
        if not isinstance(gen, dict):
            gen = {}
        gen["model_name"] = model_name
        _sanitize_generation_for_model(gen, model_name, logger=logger, context=f"solvers.strong[{idx}]")
        conf["generator"] = gen
        conf.setdefault("id", model_name)
        new_list.append(conf)

    solvers["strong"] = new_list
    if logger:
        logger.info("命令行覆盖: solvers.strong=%s（primary=%s）", ",".join(model_names), model_names[0])


def _override_medium_model(config: Dict[str, Any], model_name: str, logger: Optional[logging.Logger] = None) -> None:
    """Override solvers.medium.generator.model_name."""
    solvers = config.setdefault("solvers", {})
    medium = solvers.get("medium")
    if not isinstance(medium, dict):
        return
    gen = medium.get("generator")
    if not isinstance(gen, dict):
        return
    old = gen.get("model_name")
    gen["model_name"] = model_name
    _sanitize_generation_for_model(gen, model_name, logger=logger, context="solvers.medium")
    if logger:
        if old and old != model_name:
            logger.info("命令行覆盖: solvers.medium=%s -> %s", old, model_name)
        else:
            logger.info("命令行覆盖: solvers.medium=%s", model_name)


def _override_struct_model(config: Dict[str, Any], model_name: str, logger: Optional[logging.Logger] = None) -> None:
    """Override operators.*.struct_generator.model_name.

    This "struct" family is intended for programming/structured roles
    (e.g. Diagnose/Format/StepCertBuilder) where robustness and protocol
    adherence are prioritized.
    """
    from copy import deepcopy

    ops = config.setdefault("operators", {})
    changed: list[str] = []
    for op_name in ("init", "extend", "revise"):
        op_conf = ops.get(op_name)
        if not isinstance(op_conf, dict):
            continue
        struct_gen = op_conf.get("struct_generator")
        if not isinstance(struct_gen, dict):
            base = op_conf.get("generator")
            struct_gen = deepcopy(base) if isinstance(base, dict) else {}
        old = struct_gen.get("model_name")
        struct_gen["model_name"] = model_name
        _sanitize_generation_for_model(struct_gen, model_name, logger=logger, context=f"operators.{op_name}.struct_generator")
        op_conf["struct_generator"] = struct_gen
        if old and old != model_name:
            changed.append(f"operators.{op_name}.struct_generator: {old} -> {model_name}")
        else:
            changed.append(f"operators.{op_name}.struct_generator: {model_name}")
    if logger and changed:
        logger.info("命令行覆盖: %s", "; ".join(changed))


def _override_format_model(config: Dict[str, Any], model_name: str, logger: Optional[logging.Logger] = None) -> None:
    """Override operators.*.format_generator.model_name."""
    from copy import deepcopy

    ops = config.setdefault("operators", {})
    changed: list[str] = []
    for op_name in ("init", "extend", "revise"):
        op_conf = ops.get(op_name)
        if not isinstance(op_conf, dict):
            continue
        fmt_gen = op_conf.get("format_generator")
        if not isinstance(fmt_gen, dict):
            base = op_conf.get("generator")
            fmt_gen = deepcopy(base) if isinstance(base, dict) else {}
        old = fmt_gen.get("model_name")
        fmt_gen["model_name"] = model_name
        _sanitize_generation_for_model(fmt_gen, model_name, logger=logger, context=f"operators.{op_name}.format_generator")
        op_conf["format_generator"] = fmt_gen
        if old and old != model_name:
            changed.append(f"operators.{op_name}.format_generator: {old} -> {model_name}")
        else:
            changed.append(f"operators.{op_name}.format_generator: {model_name}")
    if logger and changed:
        logger.info("命令行覆盖: %s", "; ".join(changed))

def _scrub_secrets(obj: Any) -> Any:
    """浅层清洗配置中的潜在敏感字段（api_key / token / secret 等）。"""
    import re as _re

    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            if _re.search(r"(api_key|apikey|secret|token)", str(k), _re.IGNORECASE):
                cleaned[k] = "<redacted>"
            else:
                cleaned[k] = _scrub_secrets(v)
        return cleaned
    if isinstance(obj, list):
        return [_scrub_secrets(it) for it in obj]
    return obj


def _collect_git_info(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Collect best-effort git metadata for reproducibility/debugging.

    Returns None when git is unavailable or repo_root is not a git repo.
    """
    try:
        git_root = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

    def _git(args: list[str]) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "-C", git_root, *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    head = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    describe = _git(["describe", "--always", "--dirty", "--tags"])
    commit_time = _git(["show", "-s", "--format=%cI", "HEAD"])
    subject = _git(["show", "-s", "--format=%s", "HEAD"])

    dirty_count: Optional[int] = None
    try:
        status = subprocess.check_output(
            ["git", "-C", git_root, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty_lines = [ln for ln in status.splitlines() if ln.strip()]
        dirty_count = len(dirty_lines)
    except Exception:
        dirty_count = None

    info: Dict[str, Any] = {
        "git_root": git_root,
        "head": head,
        "head_short": head[:12] if isinstance(head, str) and len(head) >= 12 else head,
        "branch": branch,
        "describe": describe,
        "commit_time": commit_time,
        "subject": subject,
        "dirty_count": dirty_count,
        "is_dirty": (dirty_count or 0) > 0 if dirty_count is not None else None,
    }
    # Drop nulls to keep payload compact.
    return {k: v for k, v in info.items() if v is not None}


def _save_run_config(run_dir: Path, args: argparse.Namespace, config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    """将本次运行使用的配置与关键参数保存到 run 目录，便于后续回溯。

    - 记录 CLI 参数（去除不可序列化字段，并对可能含密的字段做脱敏）。
    - 记录经过 CLI 覆盖后的最终 config（同样做脱敏）。
    - 可选 extra 用于记录 agent 等上层入口的附加信息。
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {}

        cli_args = {}
        for k, v in vars(args).items():
            if k == "func":
                continue
            if isinstance(v, Path):
                cli_args[k] = str(v)
            else:
                cli_args[k] = v
        cli_args = _scrub_secrets(cli_args)  # type: ignore[assignment]

        payload["cli_args"] = cli_args
        payload["config_path"] = getattr(args, "config", None)
        payload["resolved_config"] = _scrub_secrets(config)
        payload["git"] = _collect_git_info(REPO_ROOT)
        payload["saved_at"] = datetime.now(timezone.utc).isoformat()
        if extra:
            payload["extra"] = extra

        target = run_dir / "run_config.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # 配置快照失败不影响主链路
        logging.getLogger("agenqa.agent").warning("保存 run_config.json 失败", exc_info=True)


def _read_batch_paper_list(
    path: Path,
    *,
    dest_root: Optional[Path] = None,
    path_field: Optional[str] = None,
    relpath_field: Optional[str] = None,
) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".lst"}:
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
        return items
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"[ERROR] paper list json must be an array: {path}")
        out = []
        for item in data:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                raw = item.get("paper_path") or item.get("path")
                if isinstance(raw, str) and raw.strip():
                    out.append(raw.strip())
        return out
    if suffix == ".jsonl":
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if isinstance(row, str) and row.strip():
                out.append(row.strip())
            elif isinstance(row, dict):
                raw = row.get("paper_path") or row.get("path")
                if isinstance(raw, str) and raw.strip():
                    out.append(raw.strip())
        return out
    if suffix == ".csv":
        out = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue

                chosen_path = ""
                rel_raw = ""

                field_candidates = [path_field] if isinstance(path_field, str) and path_field.strip() else []
                field_candidates.extend(["paper_path", "path"])
                for key in field_candidates:
                    raw = row.get(key) if isinstance(key, str) else None
                    if isinstance(raw, str) and raw.strip():
                        chosen_path = raw.strip()
                        break

                rel_candidates = [relpath_field] if isinstance(relpath_field, str) and relpath_field.strip() else []
                rel_candidates.extend(["localRelpath"])
                if not chosen_path:
                    for key in rel_candidates:
                        raw = row.get(key) if isinstance(key, str) else None
                        if isinstance(raw, str) and raw.strip():
                            rel_raw = raw.strip()
                            break

                if chosen_path:
                    out.append(chosen_path)
                elif rel_raw:
                    base = dest_root or REPO_ROOT
                    out.append(str((base / rel_raw).resolve()))
        return out
    raise SystemExit(f"[ERROR] unsupported paper list format: {path}")


def _read_batch_results_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_structured_records(path: Path) -> list[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"[ERROR] structured manifest json must be an array: {path}")
        return [row for row in data if isinstance(row, dict)]
    if suffix == ".jsonl":
        rows: list[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    raise SystemExit(f"[ERROR] unsupported structured manifest format: {path}")


def _resolve_manifest_row_local_path(
    row: Dict[str, Any],
    *,
    dest_root: Optional[Path] = None,
    path_field: Optional[str] = None,
    relpath_field: Optional[str] = None,
) -> Path:
    field_candidates = [path_field] if isinstance(path_field, str) and path_field.strip() else []
    field_candidates.extend(["paper_path", "path"])
    for key in field_candidates:
        raw = row.get(key) if isinstance(key, str) else None
        if isinstance(raw, str) and raw.strip():
            path = Path(raw.strip())
            return path if path.is_absolute() else (REPO_ROOT / path)

    rel_candidates = [relpath_field] if isinstance(relpath_field, str) and relpath_field.strip() else []
    rel_candidates.extend(["localRelpath"])
    for key in rel_candidates:
        raw = row.get(key) if isinstance(key, str) else None
        if isinstance(raw, str) and raw.strip():
            base = dest_root or REPO_ROOT
            return base / raw.strip()

    raise SystemExit(
        "[ERROR] manifest row cannot resolve local paper path; "
        f"expected one of path/paper_path/{relpath_field or 'localRelpath'}"
    )


def _load_tos_manifest_entries(
    manifest_path: Path,
    *,
    dest_root: Optional[Path] = None,
    path_field: Optional[str] = None,
    relpath_field: Optional[str] = None,
) -> list[Dict[str, Any]]:
    rows = _load_structured_records(manifest_path)
    entries: list[Dict[str, Any]] = []
    for row in rows:
        local_path = _resolve_manifest_row_local_path(
            row,
            dest_root=dest_root,
            path_field=path_field,
            relpath_field=relpath_field,
        )
        entries.append({
            "row": row,
            "path": local_path,
        })
    return entries


_TOS_MANIFEST_FETCH_MODULE: Optional[Any] = None


def _load_tos_manifest_fetch_module() -> Any:
    global _TOS_MANIFEST_FETCH_MODULE
    if _TOS_MANIFEST_FETCH_MODULE is not None:
        return _TOS_MANIFEST_FETCH_MODULE

    module_path = REPO_ROOT / "data/input/get/tos_paper_ingest/tos_manifest_fetch.py"
    if not module_path.exists():
        raise SystemExit(f"[ERROR] missing TOS manifest helper: {module_path}")

    spec = importlib.util.spec_from_file_location("agenqa_tos_manifest_fetch", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[ERROR] failed to load TOS manifest helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TOS_MANIFEST_FETCH_MODULE = module
    return module


def _materialize_missing_tos_entries(
    entries: list[Dict[str, Any]],
    *,
    dest_root: Optional[Path],
    creds_source: Optional[str] = None,
    tosutil: Optional[str] = None,
    endpoint: Optional[str] = None,
    region: Optional[str] = None,
    agent_sop: Optional[str] = None,
) -> None:
    missing_rows = [entry["row"] for entry in entries if isinstance(entry.get("path"), Path) and not entry["path"].exists()]
    if not missing_rows:
        return

    if dest_root is None:
        raise SystemExit("[ERROR] TOS manifest materialize requires dest_root to resolve localRelpath")

    logger = logging.getLogger("agenqa.batch")
    logger.info("本地缺失 %s 篇论文，开始通过 TOS manifest 补料到 %s", len(missing_rows), str(dest_root))

    module = _load_tos_manifest_fetch_module()
    rc = module.download_manifest(
        missing_rows,
        tosutil=str(tosutil or module.DEFAULT_TOSUTIL),
        endpoint=str(endpoint or module.DEFAULT_ENDPOINT),
        region=str(region or module.DEFAULT_REGION),
        dest_root=dest_root,
        limit=None,
        force=False,
        dry_run=False,
        creds_source=str(creds_source or "auto"),
        agent_sop=Path(agent_sop) if isinstance(agent_sop, str) and agent_sop.strip() else module.DEFAULT_AGENT_SOP,
    )
    if int(rc) != 0:
        raise SystemExit(f"[ERROR] failed to materialize missing papers from TOS manifest (rc={rc})")

    still_missing = [entry["path"] for entry in entries if isinstance(entry.get("path"), Path) and not entry["path"].exists()]
    if still_missing:
        sample = ", ".join(str(path) for path in still_missing[:3])
        raise SystemExit(f"[ERROR] papers still missing after TOS materialize: {sample}")


def _extract_batch_run_index(run_dir: Path) -> int:
    m = re.match(r"run_paper(\d+)_", run_dir.name)
    if not m:
        raise SystemExit(f"[ERROR] cannot infer batch task index from run dir name: {run_dir.name}")
    return int(m.group(1))


def _is_primary_batch_run_dir(run_dir: Path) -> bool:
    return re.fullmatch(r"run_paper\d+_\d{8}_\d{6}_\d{3}_[0-9a-f]+", run_dir.name) is not None


def _resolve_batch_input_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(config, dict):
        return None
    block = config.get("batch_input")
    if not isinstance(block, dict) or not block:
        return None

    mode = str(block.get("source") or block.get("type") or "").strip().lower()
    if not mode:
        if block.get("paper_list") or block.get("path"):
            mode = "paper_list"
        elif block.get("paper_dir"):
            mode = "paper_dir"
        elif block.get("tos_manifest"):
            mode = "tos_manifest"

    if mode in {"paper_list", "paper-dir", "paper_dir", "dir", "directory"}:
        if mode in {"paper-dir", "paper_dir", "dir", "directory"}:
            mode = "paper_dir"
        else:
            mode = "paper_list"
    elif mode in {"tos", "tos_manifest", "manifest"}:
        mode = "tos_manifest"
    else:
        return None

    resolved: Dict[str, Any] = {"mode": mode}
    if mode == "paper_list":
        resolved["paper_list"] = block.get("paper_list") or block.get("path")
    elif mode == "paper_dir":
        resolved["paper_dir"] = block.get("paper_dir") or block.get("path")
        resolved["glob"] = block.get("glob")
    else:
        tos_block = block.get("tos_manifest")
        if isinstance(tos_block, str):
            tos_block = {"path": tos_block}
        elif not isinstance(tos_block, dict):
            tos_block = {}
        resolved["manifest_path"] = tos_block.get("path") or block.get("manifest_path") or block.get("path")
        resolved["dest_root"] = tos_block.get("dest_root") or block.get("dest_root")
        resolved["path_field"] = tos_block.get("path_field") or block.get("path_field")
        resolved["relpath_field"] = tos_block.get("relpath_field") or block.get("relpath_field")
        resolved["materialize_on_missing"] = bool(
            tos_block.get("materialize_on_missing", block.get("materialize_on_missing", False))
        )
        resolved["creds_source"] = tos_block.get("creds_source") or block.get("creds_source")
        resolved["tosutil"] = tos_block.get("tosutil") or block.get("tosutil")
        resolved["endpoint"] = tos_block.get("endpoint") or block.get("endpoint")
        resolved["region"] = tos_block.get("region") or block.get("region")
        resolved["agent_sop"] = tos_block.get("agent_sop") or block.get("agent_sop")
    max_tasks = block.get("max_tasks")
    if max_tasks is not None:
        resolved["max_tasks"] = max_tasks
    return resolved


def _resolve_batch_papers(args: argparse.Namespace, config: Optional[Dict[str, Any]] = None) -> list[Path]:
    papers: list[Path] = []
    if getattr(args, "paper_list", None):
        list_path = Path(str(args.paper_list))
        if not list_path.is_absolute():
            list_path = REPO_ROOT / list_path
        if not list_path.exists():
            raise SystemExit(f"[ERROR] paper list does not exist: {list_path}")
        papers = [Path(x) for x in _read_batch_paper_list(list_path)]
    elif getattr(args, "paper_dir", None):
        paper_dir = Path(str(args.paper_dir))
        if not paper_dir.is_absolute():
            paper_dir = REPO_ROOT / paper_dir
        if not paper_dir.exists():
            raise SystemExit(f"[ERROR] paper dir does not exist: {paper_dir}")
        pattern = str(getattr(args, "glob", None) or "**/*.pdf")
        papers = sorted(paper_dir.glob(pattern))
    else:
        batch_input = _resolve_batch_input_config(config or {})
        if not batch_input:
            raise SystemExit("[ERROR] agent-batch-run requires --paper-list/--paper-dir, or config.batch_input")
        mode = str(batch_input.get("mode") or "")
        if mode == "paper_list":
            raw_list_path = batch_input.get("paper_list")
            if not isinstance(raw_list_path, str) or not raw_list_path.strip():
                raise SystemExit("[ERROR] config.batch_input.paper_list/path is required")
            list_path = Path(raw_list_path.strip())
            if not list_path.is_absolute():
                list_path = REPO_ROOT / list_path
            if not list_path.exists():
                raise SystemExit(f"[ERROR] config batch paper list does not exist: {list_path}")
            papers = [Path(x) for x in _read_batch_paper_list(list_path)]
        elif mode == "paper_dir":
            raw_paper_dir = batch_input.get("paper_dir")
            if not isinstance(raw_paper_dir, str) or not raw_paper_dir.strip():
                raise SystemExit("[ERROR] config.batch_input.paper_dir/path is required")
            paper_dir = Path(raw_paper_dir.strip())
            if not paper_dir.is_absolute():
                paper_dir = REPO_ROOT / paper_dir
            if not paper_dir.exists():
                raise SystemExit(f"[ERROR] config batch paper dir does not exist: {paper_dir}")
            pattern = str(batch_input.get("glob") or getattr(args, "glob", None) or "**/*.pdf")
            papers = sorted(paper_dir.glob(pattern))
        elif mode == "tos_manifest":
            raw_manifest_path = batch_input.get("manifest_path")
            if not isinstance(raw_manifest_path, str) or not raw_manifest_path.strip():
                raise SystemExit("[ERROR] config.batch_input.tos_manifest.path is required")
            manifest_path = Path(raw_manifest_path.strip())
            if not manifest_path.is_absolute():
                manifest_path = REPO_ROOT / manifest_path
            if not manifest_path.exists():
                raise SystemExit(f"[ERROR] config TOS manifest does not exist: {manifest_path}")
            raw_dest_root = batch_input.get("dest_root")
            dest_root = None
            if isinstance(raw_dest_root, str) and raw_dest_root.strip():
                dest_root = Path(raw_dest_root.strip())
                if not dest_root.is_absolute():
                    dest_root = REPO_ROOT / dest_root
            entries = _load_tos_manifest_entries(
                manifest_path,
                dest_root=dest_root,
                path_field=batch_input.get("path_field"),
                relpath_field=batch_input.get("relpath_field"),
            )
            max_tasks = getattr(args, "max_tasks", None)
            if max_tasks is None:
                max_tasks = batch_input.get("max_tasks")
            if max_tasks is not None:
                entries = entries[: max(0, int(max_tasks))]
            if bool(batch_input.get("materialize_on_missing", False)):
                _materialize_missing_tos_entries(
                    entries,
                    dest_root=dest_root,
                    creds_source=batch_input.get("creds_source"),
                    tosutil=batch_input.get("tosutil"),
                    endpoint=batch_input.get("endpoint"),
                    region=batch_input.get("region"),
                    agent_sop=batch_input.get("agent_sop"),
                )
            papers = [Path(entry["path"]) for entry in entries]
        else:
            raise SystemExit(f"[ERROR] unsupported config.batch_input mode: {mode}")

    resolved: list[Path] = []
    seen: set[str] = set()
    for item in papers:
        path = item if item.is_absolute() else (REPO_ROOT / item)
        norm = str(path.resolve()) if path.exists() else str(path.absolute())
        if norm in seen:
            continue
        seen.add(norm)
        resolved.append(path)

    max_tasks = getattr(args, "max_tasks", None)
    if max_tasks is None:
        batch_input = _resolve_batch_input_config(config or {})
        max_tasks = batch_input.get("max_tasks") if isinstance(batch_input, dict) else None
    if max_tasks is not None:
        resolved = resolved[: max(0, int(max_tasks))]
    if not resolved:
        raise SystemExit("[ERROR] no papers resolved for agent-batch-run")
    return resolved


def _build_batch_summary_md(*, batch_id: str, rows: list[Dict[str, Any]]) -> str:
    total = len(rows)
    success = sum(1 for row in rows if row.get("status") == "success")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    durations = [float(row.get("duration_sec") or 0.0) for row in rows if row.get("duration_sec") is not None]
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0

    lines = [
        f"# Batch Summary: {batch_id}",
        "",
        f"- Total tasks: {total}",
        f"- Success: {success}",
        f"- Failed: {failed}",
        f"- Average duration (sec): {avg_duration:.2f}",
        "",
        "| Task ID | Status | Paper | Run ID | Run Dir | Duration (sec) | Error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {task_id} | {status} | {paper_path} | {run_id} | {run_dir} | {duration_sec} | {error} |".format(
                task_id=row.get("task_id") or "",
                status=row.get("status") or "",
                paper_path=row.get("paper_path") or "",
                run_id=row.get("run_id") or "",
                run_dir=row.get("run_dir") or "",
                duration_sec=f"{float(row.get('duration_sec') or 0.0):.2f}",
                error=str(row.get("error") or "").replace("\n", " "),
            )
        )
    return "\n".join(lines).strip() + "\n"


def _write_batch_results(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_state_brief(run_dir: Path) -> Dict[str, Any]:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return {"step": 0, "history_len": 0, "stop_reason": None}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        history = data.get("history") or []
        return {
            "step": int(data.get("step") or 0),
            "history_len": len(history) if isinstance(history, list) else 0,
            "stop_reason": data.get("stop_reason"),
        }
    except Exception:
        return {"step": 0, "history_len": 0, "stop_reason": None}


def _extract_model_from_line(line: str) -> Optional[str]:
    m = re.search(r"model=([^,\s)]+)", line)
    if m:
        return m.group(1).strip()
    return None


def _classify_error_type(text: str) -> Optional[str]:
    s = text.lower()
    if "status=429" in s:
        return "429"
    if "status=504" in s or "gateway time-out" in s or "gateway timeout" in s:
        return "504"
    if any(token in s for token in [
        "proxyerror",
        "invalidchunklength",
        "chunkedencodingerror",
        "remotedisconnected",
        "connectionerror",
        "readtimeout",
    ]):
        return "network"
    return None


def _role_bucket(role: Optional[str]) -> Optional[str]:
    role_norm = str(role or "").strip().lower()
    if role_norm in {"init", "episode_seed_builder"}:
        return "init"
    if role_norm in {"solver"}:
        return "solver"
    if role_norm in {"expression_judge"}:
        return "expression_judge"
    if role_norm in {"director", "draft_chain", "format", "step_cert_builder", "path_fold", "judge"}:
        return "main_generation"
    return None


def _infer_role_and_stage_from_line(line: str, current_role: Optional[str], current_stage: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    role = current_role
    stage = current_stage
    s = line.lower()

    if "step_0_init" in s or "episode_seed_builder" in s:
        role = "init"
        stage = "init"
    elif "director 决策" in line or "agenqa.nodes.director" in s:
        role = "director"
        stage = stage or "judge"
    elif "draft_chain" in s:
        role = "draft_chain"
        if "/revise/" in s or " revise" in s:
            stage = "revise"
        else:
            stage = "extend"
    elif "formatrunner" in s or "agenqa.skills.formatting" in s or "format.prompt" in s:
        role = "format"
        stage = stage or "extend"
    elif "step_cert_builder" in s:
        role = "step_cert_builder"
        stage = stage or "extend"
    elif "path_fold" in s:
        role = "path_fold"
        stage = stage or "extend"
    elif "expression_judge" in s:
        role = "expression_judge"
        stage = "solve"
    elif "solve[" in s or "agenqa.skills.solving" in s or "/solve/" in s:
        role = "solver"
        stage = "solve"
    elif "judge:" in s or "agenqa.nodes.judge" in s:
        role = "judge"
        stage = "judge"

    if "/extend/" in s:
        stage = "extend"
    elif "/revise/" in s:
        stage = "revise"
    elif "/solve/" in s:
        stage = "solve"

    return role, stage


def _increment_model_counters(model_counters: Dict[str, Dict[str, int]], model: str, *, request: bool = False, error_type: Optional[str] = None) -> None:
    row = model_counters.setdefault(
        model,
        {
            "request_count": 0,
            "429_count": 0,
            "504_count": 0,
            "network_error_count": 0,
        },
    )
    if request:
        row["request_count"] += 1
    if error_type == "429":
        row["429_count"] += 1
    elif error_type == "504":
        row["504_count"] += 1
    elif error_type == "network":
        row["network_error_count"] += 1


def _compute_batch_lineage_meta(batch_dir: Path, manifest: Dict[str, Any]) -> tuple[str, str, int]:
    resumed_from_batch_id = str(manifest.get("resumed_from_batch_id") or "")
    heartbeat_root_batch_dir = manifest.get("heartbeat_root_batch_dir")
    if isinstance(heartbeat_root_batch_dir, str) and heartbeat_root_batch_dir.strip():
        root_dir = Path(heartbeat_root_batch_dir)
        if not root_dir.is_absolute():
            root_dir = REPO_ROOT / root_dir
        try:
            root_manifest = _read_json_file(root_dir / "batch_manifest.json")
            root_batch_id = str(root_manifest.get("batch_id") or root_dir.name)
        except Exception:
            root_batch_id = root_dir.name
        attempt_idx = int(manifest.get("heartbeat_attempt_index") or 1)
        return root_batch_id, resumed_from_batch_id, max(1, attempt_idx)

    parent_dir = manifest.get("resumed_from_batch_dir")
    if not isinstance(parent_dir, str) or not parent_dir.strip():
        return str(manifest.get("batch_id") or batch_dir.name), resumed_from_batch_id, 0

    parent_path = Path(parent_dir)
    if not parent_path.is_absolute():
        parent_path = REPO_ROOT / parent_path
    depth = 1
    seen: set[str] = set()
    root_batch_id = str(manifest.get("batch_id") or batch_dir.name)
    cur = parent_path
    while cur.exists():
        key = str(cur.resolve())
        if key in seen:
            break
        seen.add(key)
        try:
            parent_manifest = _read_json_file(cur / "batch_manifest.json")
        except Exception:
            break
        root_batch_id = str(parent_manifest.get("batch_id") or cur.name)
        next_parent = parent_manifest.get("resumed_from_batch_dir")
        if not isinstance(next_parent, str) or not next_parent.strip():
            break
        cur = Path(next_parent)
        if not cur.is_absolute():
            cur = REPO_ROOT / cur
        depth += 1
    return root_batch_id, resumed_from_batch_id, depth


def _make_live_task_state(task_id: str, paper_path: str, run_dir: Path, status: str, now_iso: str) -> Dict[str, Any]:
    brief = _read_state_brief(run_dir)
    return {
        "task_id": task_id,
        "paper_path": paper_path,
        "status": status,
        "run_dir": str(run_dir),
        "step": int(brief.get("step") or 0),
        "history_len": int(brief.get("history_len") or 0),
        "latest_stage": None,
        "latest_role": None,
        "latest_model": None,
        "last_error": "",
        "last_error_type": None,
        "retry_count": 0,
        "last_update_at": now_iso,
        "stop_reason": brief.get("stop_reason"),
    }


def _hydrate_task_from_run_dir(task: Dict[str, Any], run_dir: Path) -> None:
    log_snapshot = _scan_run_log_snapshot(run_dir)
    if log_snapshot.get("latest_stage") is not None:
        task["latest_stage"] = log_snapshot.get("latest_stage")
    if log_snapshot.get("latest_role") is not None:
        task["latest_role"] = log_snapshot.get("latest_role")
    if log_snapshot.get("latest_model") is not None:
        task["latest_model"] = log_snapshot.get("latest_model")
    if log_snapshot.get("last_error"):
        task["last_error"] = log_snapshot.get("last_error") or ""
        task["last_error_type"] = log_snapshot.get("last_error_type")
        task["last_update_at"] = log_snapshot.get("last_update_at") or task.get("last_update_at")


def _summarize_counts(tasks: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts = {"total": len(tasks), "queued": 0, "running": 0, "success": 0, "failed": 0}
    for task in tasks.values():
        status = str(task.get("status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _snapshot_live_status(
    *,
    batch_id: str,
    batch_dir: Path,
    manifest: Dict[str, Any],
    task_states: Dict[str, Dict[str, Any]],
    model_counters: Dict[str, Dict[str, int]],
    role_counters: Dict[str, int],
    started_at: str,
) -> Dict[str, Any]:
    root_batch_id, resumed_from_batch_id, lineage_depth = _compute_batch_lineage_meta(batch_dir, manifest)
    counts = _summarize_counts(task_states)
    return {
        "batch_id": batch_id,
        "root_batch_id": root_batch_id,
        "resumed_from_batch_id": resumed_from_batch_id,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "concurrency": int(manifest.get("concurrency") or 0),
        "lineage_depth": int(lineage_depth),
        "total": counts["total"],
        "queued": counts["queued"],
        "running": counts["running"],
        "success": counts["success"],
        "failed": counts["failed"],
        "tasks": {k: dict(v) for k, v in sorted(task_states.items())},
        "counters": {
            "models": model_counters,
            "roles": role_counters,
        },
    }


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
        try:
            manifest = _read_json_file(manifest_path)
        except Exception:
            return cur
        hb_root = manifest.get("heartbeat_root_batch_dir")
        if isinstance(hb_root, str) and hb_root.strip():
            root = Path(hb_root)
            if not root.is_absolute():
                root = REPO_ROOT / root
            return root.resolve()
        parent_dir = manifest.get("resumed_from_batch_dir")
        if not isinstance(parent_dir, str) or not parent_dir.strip():
            return cur
        parent = Path(parent_dir)
        if not parent.is_absolute():
            parent = REPO_ROOT / parent
        cur = parent.resolve()


def _load_batch_lineage(root_batch_dir: Path) -> list[Dict[str, Any]]:
    root_batch_dir = _resolve_lineage_root(root_batch_dir)
    root_manifest = _read_json_file(root_batch_dir / "batch_manifest.json")
    candidates: list[Dict[str, Any]] = [
        {
            "path": root_batch_dir,
            "manifest": root_manifest,
        }
    ]
    for child in sorted(root_batch_dir.parent.iterdir()):
        if not child.is_dir() or child == root_batch_dir:
            continue
        manifest_path = child / "batch_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _read_json_file(manifest_path)
        except Exception:
            continue
        candidates.append({"path": child.resolve(), "manifest": manifest})

    selected: Dict[str, Dict[str, Any]] = {str(root_batch_dir): candidates[0]}
    changed = True
    while changed:
        changed = False
        selected_paths = {str(Path(k).resolve()) for k in selected.keys()}
        for item in candidates[1:]:
            path = Path(item["path"]).resolve()
            if str(path) in selected:
                continue
            manifest = item["manifest"]
            hb_root = manifest.get("heartbeat_root_batch_dir")
            if isinstance(hb_root, str) and hb_root.strip():
                hb_root_path = Path(hb_root)
                if not hb_root_path.is_absolute():
                    hb_root_path = REPO_ROOT / hb_root_path
                if str(hb_root_path.resolve()) == str(root_batch_dir.resolve()):
                    selected[str(path)] = item
                    changed = True
                    continue
            resumed_from = manifest.get("resumed_from_batch_dir")
            if isinstance(resumed_from, str) and resumed_from.strip():
                resumed_from_path = Path(resumed_from)
                if not resumed_from_path.is_absolute():
                    resumed_from_path = REPO_ROOT / resumed_from_path
                if str(resumed_from_path.resolve()) in selected_paths:
                    selected[str(path)] = item
                    changed = True

    lineage = list(selected.values())
    lineage.sort(
        key=lambda item: (
            str(item["manifest"].get("created_at") or ""),
            str(item["path"]),
        )
    )
    for idx, item in enumerate(lineage):
        item["order"] = idx
        item["results"] = _read_batch_results_jsonl(Path(item["path"]) / "batch_results.jsonl")
    return lineage


def _collect_latest_lineage_rows(root_batch_dir: Path) -> Dict[str, Dict[str, Any]]:
    latest_rows: Dict[str, Dict[str, Any]] = {}
    for item in _load_batch_lineage(root_batch_dir):
        for row in item.get("results") or []:
            task_id = str(row.get("task_id") or "")
            if not task_id:
                continue
            latest_rows[task_id] = dict(row)
    return latest_rows


def _read_live_status_file(batch_dir: Path) -> Optional[Dict[str, Any]]:
    path = batch_dir / "live_status.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _scan_run_log_snapshot(run_dir: Path) -> Dict[str, Any]:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return {
            "latest_stage": None,
            "latest_role": None,
            "latest_model": None,
            "last_error": "",
            "last_error_type": None,
            "last_update_at": None,
            "model_counters": {},
        }
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    latest_role: Optional[str] = None
    latest_stage: Optional[str] = None
    latest_model: Optional[str] = None
    last_error = ""
    last_error_type: Optional[str] = None
    model_counters: Dict[str, Dict[str, int]] = {}
    for line in text.splitlines():
        latest_role, latest_stage = _infer_role_and_stage_from_line(line, latest_role, latest_stage)
        model = _extract_model_from_line(line)
        if model and "Backend:" in line:
            latest_model = model
            _increment_model_counters(model_counters, model, request=True)
        error_type = _classify_error_type(line)
        if error_type:
            if model:
                latest_model = model
                _increment_model_counters(model_counters, model, error_type=error_type)
            last_error = line.strip()
            last_error_type = error_type
    return {
        "latest_stage": latest_stage,
        "latest_role": latest_role,
        "latest_model": latest_model,
        "last_error": last_error,
        "last_error_type": last_error_type,
        "last_update_at": datetime.fromtimestamp(log_path.stat().st_mtime, timezone.utc).isoformat(),
        "model_counters": model_counters,
    }


def _build_batch_snapshot_from_files(batch_dir: Path) -> Dict[str, Any]:
    manifest = _read_json_file(batch_dir / "batch_manifest.json")
    papers = [str(p) for p in manifest.get("papers") or []]
    results = {
        str(row.get("task_id") or ""): row
        for row in _read_batch_results_jsonl(batch_dir / "batch_results.jsonl")
        if row.get("task_id")
    }
    tasks: Dict[str, Dict[str, Any]] = {}
    model_counters: Dict[str, Dict[str, int]] = {}
    role_counters = {"init": 0, "main_generation": 0, "solver": 0, "expression_judge": 0}
    for idx, paper_path in enumerate(papers, start=1):
        task_id = f"task_{idx:03d}"
        run_dir = _find_primary_run_dir_for_task(batch_dir, task_id)
        state_brief = _read_state_brief(run_dir) if run_dir else {"step": 0, "history_len": 0, "stop_reason": None}
        log_snapshot = _scan_run_log_snapshot(run_dir) if run_dir else {
            "latest_stage": None,
            "latest_role": None,
            "latest_model": None,
            "last_error": "",
            "last_error_type": None,
            "last_update_at": None,
            "model_counters": {},
        }
        row = results.get(task_id)
        if row is not None:
            status = str(row.get("status") or "unknown")
        elif run_dir is not None:
            status = "running"
        else:
            status = "unknown"
        tasks[task_id] = {
            "task_id": task_id,
            "paper_path": paper_path,
            "status": status,
            "run_dir": str(run_dir) if run_dir else "",
            "step": int(state_brief.get("step") or 0),
            "history_len": int(state_brief.get("history_len") or 0),
            "latest_stage": log_snapshot.get("latest_stage"),
            "latest_role": log_snapshot.get("latest_role"),
            "latest_model": log_snapshot.get("latest_model"),
            "last_error": row.get("error") if row is not None and row.get("error") else log_snapshot.get("last_error") or "",
            "last_error_type": log_snapshot.get("last_error_type"),
            "retry_count": 0,
            "last_update_at": log_snapshot.get("last_update_at"),
            "stop_reason": state_brief.get("stop_reason"),
        }
        for model, counts in (log_snapshot.get("model_counters") or {}).items():
            target = model_counters.setdefault(
                model,
                {"request_count": 0, "429_count": 0, "504_count": 0, "network_error_count": 0},
            )
            for key in target.keys():
                target[key] += int(counts.get(key) or 0)
        role_bucket = _role_bucket(tasks[task_id].get("latest_role"))
        if role_bucket:
            role_counters[role_bucket] += 1

    root_batch_id, resumed_from_batch_id, lineage_depth = _compute_batch_lineage_meta(batch_dir, manifest)
    counts = _summarize_counts(tasks)
    return {
        "batch_id": str(manifest.get("batch_id") or batch_dir.name),
        "root_batch_id": root_batch_id,
        "resumed_from_batch_id": resumed_from_batch_id,
        "started_at": str(manifest.get("created_at") or ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "concurrency": int(manifest.get("concurrency") or 0),
        "lineage_depth": int(lineage_depth),
        "total": counts["total"],
        "queued": counts["queued"],
        "running": counts["running"],
        "success": counts["success"],
        "failed": counts["failed"],
        "tasks": tasks,
        "counters": {"models": model_counters, "roles": role_counters},
    }


def _collect_status_snapshot(batch_dir: Path, stale_sec: int) -> Dict[str, Any]:
    root_batch_dir = _resolve_lineage_root(batch_dir)
    lineage = _load_batch_lineage(root_batch_dir)
    aggregate_tasks: Dict[str, Dict[str, Any]] = {}
    aggregate_model_counters: Dict[str, Dict[str, int]] = {}
    aggregate_role_counters = {"init": 0, "main_generation": 0, "solver": 0, "expression_judge": 0}

    for item in lineage:
        path = Path(item["path"])
        live = _read_live_status_file(path) or _build_batch_snapshot_from_files(path)
        item["live"] = live
        for task_id, task in (live.get("tasks") or {}).items():
            aggregate_tasks[task_id] = dict(task)
            aggregate_tasks[task_id]["latest_batch_id"] = live.get("batch_id")
            aggregate_tasks[task_id]["latest_batch_dir"] = str(path)
        for model, counts in (live.get("counters", {}).get("models", {}) or {}).items():
            target = aggregate_model_counters.setdefault(
                model,
                {"request_count": 0, "429_count": 0, "504_count": 0, "network_error_count": 0},
            )
            for key in target.keys():
                target[key] += int(counts.get(key) or 0)
        for key in aggregate_role_counters.keys():
            aggregate_role_counters[key] += int(live.get("counters", {}).get("roles", {}).get(key) or 0)

    now_ts = time.time()
    stale_tasks: list[str] = []
    for task in aggregate_tasks.values():
        last_update_at = task.get("last_update_at")
        if str(task.get("status") or "") != "running":
            continue
        if not isinstance(last_update_at, str) or not last_update_at.strip():
            continue
        try:
            last_ts = datetime.fromisoformat(last_update_at.replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        if now_ts - last_ts >= stale_sec:
            stale_tasks.append(str(task.get("task_id") or ""))
            task["stale"] = True
        else:
            task["stale"] = False

    success_rows: list[tuple[float, int]] = []
    for item in lineage:
        for row in item.get("results") or []:
            if row.get("status") != "success":
                continue
            task = aggregate_tasks.get(str(row.get("task_id") or ""))
            history_len = int((task or {}).get("history_len") or 0)
            duration = float(row.get("duration_sec") or 0.0)
            if duration > 0 and history_len > 0:
                success_rows.append((duration, history_len))
    eta_sec: Optional[float] = None
    if success_rows:
        avg_sec_per_step = sum(duration / max(1, history_len) for duration, history_len in success_rows) / len(success_rows)
        remaining = sum(max(0, 8 - int(task.get("history_len") or 0)) for task in aggregate_tasks.values() if task.get("status") == "running")
        eta_sec = round(avg_sec_per_step * remaining, 1) if remaining > 0 else 0.0

    counts = _summarize_counts(aggregate_tasks)
    snapshot = {
        "root_batch_dir": str(root_batch_dir),
        "root_batch_id": str(lineage[0]["manifest"].get("batch_id") or root_batch_dir.name),
        "lineage": [
            {
                "batch_id": str(item["live"].get("batch_id") or Path(item["path"]).name),
                "batch_dir": str(item["path"]),
                "resumed_from_batch_id": str(item["live"].get("resumed_from_batch_id") or ""),
                "lineage_depth": int(item["live"].get("lineage_depth") or 0),
            }
            for item in lineage
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": counts["total"],
        "queued": counts["queued"],
        "running": counts["running"],
        "success": counts["success"],
        "failed": counts["failed"],
        "stale_tasks": sorted(stale_tasks),
        "eta_sec": eta_sec,
        "tasks": {k: v for k, v in sorted(aggregate_tasks.items())},
        "counters": {
            "models": aggregate_model_counters,
            "roles": aggregate_role_counters,
        },
    }
    return snapshot


def _build_status_markdown(snapshot: Dict[str, Any], *, show_runs: str, top_errors: int) -> str:
    lines = [
        f"# Batch Status: {snapshot.get('root_batch_id')}",
        "",
        "## Overview",
        f"- **Root batch:** `{snapshot.get('root_batch_dir')}`",
        f"- **Updated at:** `{snapshot.get('updated_at')}`",
        f"- **Lineage:** `{', '.join(item.get('batch_id') or '' for item in snapshot.get('lineage') or [])}`",
        f"- **Progress:** Total: **{snapshot.get('total', 0)}** | Queued: {snapshot.get('queued', 0)} | Running: {snapshot.get('running', 0)} | Success: ✅ {snapshot.get('success', 0)} | Failed: ❌ {snapshot.get('failed', 0)} | Stale: ⚠️ {len(snapshot.get('stale_tasks') or [])}",
        f"- **ETA (sec):** {snapshot.get('eta_sec') if snapshot.get('eta_sec') is not None else 'n/a'}",
        "",
    ]

    tasks = snapshot.get("tasks") or {}

    active_roles = {}
    for task in tasks.values():
        if str(task.get("status") or "") in {"queued", "running"}:
            role = task.get("latest_role") or "unknown"
            active_roles[role] = active_roles.get(role, 0) + 1

    if active_roles:
        lines.extend([
            "## Active Status by Role",
            "",
            "| Role | Active Tasks |",
            "| --- | --- |",
        ])
        for role, count in sorted(active_roles.items(), key=lambda x: -x[1]):
            lines.append(f"| `{role}` | {count} |")
        lines.append("")

    lines.extend([
        "## Tasks Details",
        "",
        "| Task ID | Status | State | Step | History | Role | Model | Last Error | Last Update |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    selected: list[Dict[str, Any]] = []
    for task in tasks.values():
        status = str(task.get("status") or "")
        if show_runs == "active" and status != "running":
            continue
        if show_runs == "failed" and status != "failed":
            continue
        selected.append(task)

    for task in sorted(selected, key=lambda item: str(item.get("task_id") or "")):
        status = task.get("status") or ""
        state = "progressing"
        if status == "running":
            if task.get("stale"):
                state = "⚠️ stale"
            elif task.get("last_error"):
                state = "🔄 retrying"
        elif status == "failed":
            state = "❌ failed"
        elif status == "success":
            state = "✅ success"

        lines.append(
            "| `{task_id}` | {status} | {state} | {step} | {history_len} | `{latest_role}` | `{latest_model}` | {last_error} | {last_update_at} |".format(
                task_id=task.get("task_id") or "",
                status=status,
                state=state,
                step=task.get("step") or 0,
                history_len=task.get("history_len") or 0,
                latest_role=task.get("latest_role") or "N/A",
                latest_model=task.get("latest_model") or "N/A",
                last_error=str(task.get("last_error") or "").replace("\n", " ")[:160],
                last_update_at=task.get("last_update_at") or "",
            )
        )

    lines.extend([
        "",
        "## Model Error Rates",
        "",
        "| Model | Requests | 429 | 504 | Network | Fail Rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for model, counts in sorted((snapshot.get("counters", {}).get("models", {}) or {}).items()):
        req = int(counts.get('request_count', 0))
        errs = int(counts.get('429_count', 0)) + int(counts.get('504_count', 0)) + int(counts.get('network_error_count', 0))
        denom = req + errs
        rate = f"{(errs / denom * 100):.1f}%" if denom > 0 else "0.0%"
        lines.append(
            f"| `{model}` | {req} | {counts.get('429_count', 0)} | {counts.get('504_count', 0)} | {counts.get('network_error_count', 0)} | **{rate}** |"
        )

    recent_errors: list[Dict[str, Any]] = []
    for task in tasks.values():
        if task.get("last_error"):
            recent_errors.append(task)
    recent_errors.sort(key=lambda item: str(item.get("last_update_at") or ""), reverse=True)
    lines.extend(["", "## Recent Errors", ""])
    for task in recent_errors[: max(1, int(top_errors))]:
        lines.append(
            f"- `{task.get('task_id')}` | `{task.get('latest_model') or 'N/A'}` | **{task.get('last_error_type') or 'Unknown'}**<br/>{str(task.get('last_error') or '')[:240]}"
        )
    return "\n".join(lines).strip() + "\n"


def _build_inline_batch_status(
    snapshot: Dict[str, Any],
    *,
    active_limit: int,
    top_errors: int,
) -> str:
    lines = [
        "─" * 80,
        "📦 [BATCH] id={batch_id} | T:{total} | Q:{queued} | R:{running} | S:{success} | F:{failed} | 🕒 {updated_at}".format(
            batch_id=snapshot.get("batch_id") or "",
            total=snapshot.get("total", 0),
            queued=snapshot.get("queued", 0),
            running=snapshot.get("running", 0),
            success=snapshot.get("success", 0),
            failed=snapshot.get("failed", 0),
            updated_at=(snapshot.get("updated_at") or "")[11:19],
        ),
    ]

    tasks = list((snapshot.get("tasks") or {}).values())

    stages = {}
    for task in tasks:
        if str(task.get("status") or "") in {"queued", "running"}:
            role = task.get("latest_role") or "unknown"
            stages[role] = stages.get(role, 0) + 1
    if stages:
        stages_str = " | ".join(f"{k}:{v}" for k, v in sorted(stages.items()))
        lines.append(f"📊 [STAGES] {stages_str}")

    active_tasks = [
        task for task in tasks
        if str(task.get("status") or "") in {"queued", "running"}
    ]
    active_tasks.sort(
        key=lambda task: (
            0 if str(task.get("status") or "") == "running" else 1,
            -int(task.get("step") or 0),
            str(task.get("task_id") or ""),
        )
    )
    if active_tasks:
        lines.append("🔄 [ACTIVE]")
        lines.append(f"  {'TID':<24} | {'S':<1} | {'ST':<2} | {'H':<2} | {'ROLE':<16} | {'MODEL':<20} | {'ERROR'}")
        for task in active_tasks[: max(1, int(active_limit))]:
            task_id = str(task.get("task_id") or "")[:24]
            status = "R" if str(task.get("status") or "") == "running" else "Q"
            step = str(task.get("step") or 0)
            hist = str(task.get("history_len") or 0)
            role = str(task.get("latest_role") or "-")[:16]
            model = str(task.get("latest_model") or "-")[:20]
            err = str(task.get("last_error") or "")[:60]
            err_type = str(task.get("last_error_type") or "")
            err_disp = f"[{err_type}] {err}" if err_type else err
            lines.append(f"  {task_id:<24} | {status} | {step:<2} | {hist:<2} | {role:<16} | {model:<20} | {err_disp or '-'}")

    model_rows = []
    for model, counts in (snapshot.get("counters", {}).get("models", {}) or {}).items():
        err_total = int(counts.get("429_count", 0)) + int(counts.get("504_count", 0)) + int(counts.get("network_error_count", 0))
        if err_total <= 0:
            continue
        model_rows.append((err_total, str(model), counts))
    model_rows.sort(reverse=True)
    if model_rows:
        lines.append("⚠️  [MODEL ERRORS]")
        for _, model, counts in model_rows[: max(1, int(top_errors))]:
            req=int(counts.get('request_count', 0))
            err_rate = (err_total / req * 100) if req > 0 else 0
            lines.append(
                "  {model:<24} | REQ:{req:<4} | 429:{s429:<3} | 504:{s504:<3} | NET:{net:<3} | FAIL_RATE:{err_rate:.1f}%".format(
                    model=model[:24],
                    req=req,
                    s429=int(counts.get("429_count", 0)),
                    s504=int(counts.get("504_count", 0)),
                    net=int(counts.get("network_error_count", 0)),
                    err_rate=err_rate,
                )
            )

    recent_errors: list[Dict[str, Any]] = []
    for task in tasks:
        if task.get("last_error"):
            recent_errors.append(task)
    recent_errors.sort(key=lambda task: str(task.get("last_update_at") or ""), reverse=True)
    if recent_errors:
        lines.append("🚨 [RECENT ERRORS]")
        for task in recent_errors[: max(1, int(top_errors))]:
            time_str = (str(task.get("last_update_at") or "")[11:19])
            task_id = str(task.get("task_id") or "")[:15]
            model = str(task.get("latest_model") or "-")[:15]
            etype = str(task.get("last_error_type") or "-")
            msg = str(task.get("last_error") or "")[:80]
            lines.append(f"  [{time_str}] {task_id} | {model} | {etype} | {msg}")

    return "\n".join(lines)


def _fmt_time_short(value: Any) -> str:
    text = str(value or "")
    if len(text) >= 19 and "T" in text:
        return text[11:19]
    return text[:8] if text else "-"


def _truncate_inline(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _task_state_label(task: Dict[str, Any]) -> tuple[str, str]:
    status = str(task.get("status") or "")
    if status == "success":
        return "SUCCESS", "green"
    if status == "failed":
        return "FAILED", "red"
    if status == "queued":
        return "QUEUED", "yellow"
    if task.get("stale"):
        return "STALE", "yellow"
    if task.get("last_error"):
        return "RETRY", "magenta"
    return "RUN", "cyan"


def _build_inline_batch_layout(
    snapshot: Dict[str, Any],
    *,
    active_limit: int,
    top_errors: int,
):
    tasks = list((snapshot.get("tasks") or {}).values())
    active_tasks = [
        task for task in tasks
        if str(task.get("status") or "") in {"queued", "running"}
    ]
    active_tasks.sort(
        key=lambda task: (
            0 if str(task.get("status") or "") == "running" else 1,
            -int(task.get("step") or 0),
            str(task.get("task_id") or ""),
        )
    )

    overview = Table.grid(expand=True)
    overview.add_column(justify="left", ratio=2)
    overview.add_column(justify="right", ratio=3)
    overview.add_row(
        Text(f"Batch {snapshot.get('batch_id') or '-'}", style="bold cyan"),
        Text(
            "Total {total}  Queued {queued}  Running {running}  Success {success}  Failed {failed}".format(
                total=snapshot.get("total", 0),
                queued=snapshot.get("queued", 0),
                running=snapshot.get("running", 0),
                success=snapshot.get("success", 0),
                failed=snapshot.get("failed", 0),
            ),
            style="bold",
        ),
    )
    overview.add_row(
        Text(f"Updated {_fmt_time_short(snapshot.get('updated_at'))}", style="dim"),
        Text(
            f"ETA {snapshot.get('eta_sec') if snapshot.get('eta_sec') is not None else 'n/a'} sec",
            justify="right",
            style="dim",
        ),
    )

    stages: Dict[str, int] = {}
    for task in active_tasks:
        role = str(task.get("latest_role") or "unknown")
        stages[role] = stages.get(role, 0) + 1
    stage_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    stage_table.add_column("Role", style="cyan")
    stage_table.add_column("Active", justify="right", style="bold")
    if stages:
        for role, count in sorted(stages.items(), key=lambda item: (-item[1], item[0])):
            stage_table.add_row(role, str(count))
    else:
        stage_table.add_row("-", "0")

    active_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    active_table.add_column("Task", style="cyan", no_wrap=True)
    active_table.add_column("State", no_wrap=True)
    active_table.add_column("Step", justify="right", no_wrap=True)
    active_table.add_column("Hist", justify="right", no_wrap=True)
    active_table.add_column("Role", style="green", no_wrap=True)
    active_table.add_column("Model", style="magenta")
    active_table.add_column("Error", overflow="fold")
    if active_tasks:
        for task in active_tasks[: max(1, int(active_limit))]:
            state_label, state_style = _task_state_label(task)
            err_type = str(task.get("last_error_type") or "")
            err_msg = _truncate_inline(task.get("last_error") or "-", 96)
            err_display = f"[{err_type}] {err_msg}" if err_type else err_msg
            active_table.add_row(
                str(task.get("task_id") or "-"),
                Text(state_label, style=state_style),
                str(task.get("step") or 0),
                str(task.get("history_len") or 0),
                str(task.get("latest_role") or "-"),
                _truncate_inline(task.get("latest_model") or "-", 28),
                err_display,
            )
    else:
        active_table.add_row("-", "-", "0", "0", "-", "-", "-")

    model_rows = []
    for model, counts in (snapshot.get("counters", {}).get("models", {}) or {}).items():
        req = int(counts.get("request_count", 0))
        c429 = int(counts.get("429_count", 0))
        c504 = int(counts.get("504_count", 0))
        cnet = int(counts.get("network_error_count", 0))
        err_total = c429 + c504 + cnet
        if err_total <= 0:
            continue
        denom = req + err_total
        fail_rate = (err_total / denom * 100.0) if denom > 0 else 0.0
        model_rows.append((err_total, str(model), req, c429, c504, cnet, fail_rate))
    model_rows.sort(key=lambda item: (-item[0], item[1]))

    model_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Req", justify="right")
    model_table.add_column("429", justify="right", style="yellow")
    model_table.add_column("504", justify="right", style="red")
    model_table.add_column("Net", justify="right", style="magenta")
    model_table.add_column("Fail %", justify="right", style="bold")
    if model_rows:
        for _, model, req, c429, c504, cnet, fail_rate in model_rows[: max(1, int(top_errors))]:
            model_table.add_row(
                _truncate_inline(model, 28),
                str(req),
                str(c429),
                str(c504),
                str(cnet),
                f"{fail_rate:.1f}",
            )
    else:
        model_table.add_row("-", "0", "0", "0", "0", "0.0")

    recent_errors = [task for task in tasks if task.get("last_error")]
    recent_errors.sort(key=lambda task: str(task.get("last_update_at") or ""), reverse=True)
    error_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    error_table.add_column("Time", no_wrap=True, style="dim")
    error_table.add_column("Task", no_wrap=True, style="cyan")
    error_table.add_column("Model", style="magenta")
    error_table.add_column("Type", no_wrap=True)
    error_table.add_column("Message", overflow="fold")
    if recent_errors:
        for task in recent_errors[: max(1, int(top_errors))]:
            error_table.add_row(
                _fmt_time_short(task.get("last_update_at")),
                str(task.get("task_id") or "-"),
                _truncate_inline(task.get("latest_model") or "-", 20),
                str(task.get("last_error_type") or "-"),
                _truncate_inline(task.get("last_error") or "-", 120),
            )
    else:
        error_table.add_row("-", "-", "-", "-", "-")

    return Group(
        Panel(overview, title="Overview", border_style="cyan"),
        Panel(stage_table, title="Active Stages", border_style="blue"),
        Panel(active_table, title="Active Tasks", border_style="green"),
        Panel(model_table, title="Model Errors", border_style="yellow"),
        Panel(error_table, title="Recent Errors", border_style="red"),
    )


def _run_batch_jobs(
    *,
    batch_id: str,
    batch_dir: Path,
    jobs: list[Dict[str, Any]],
    seed_rows: list[Dict[str, Any]],
    concurrency: int,
    continue_on_error: bool,
    inline_status: bool = False,
    status_interval_sec: int = 30,
    status_active_limit: int = 8,
    status_top_errors: int = 5,
) -> list[Dict[str, Any]]:
    manifest = _read_json_file(batch_dir / "batch_manifest.json") if (batch_dir / "batch_manifest.json").exists() else {}
    results_by_task: Dict[str, Dict[str, Any]] = {
        str(row.get("task_id") or ""): row for row in seed_rows if row.get("task_id")
    }
    results_path = batch_dir / "batch_results.jsonl"
    summary_path = batch_dir / "batch_summary.md"
    live_status_path = batch_dir / "live_status.json"
    live_events_path = batch_dir / "live_events.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()
    model_counters: Dict[str, Dict[str, int]] = {}
    role_counters = {"init": 0, "main_generation": 0, "solver": 0, "expression_judge": 0}
    task_states: Dict[str, Dict[str, Any]] = {}
    log_offsets: Dict[str, int] = {}
    step_markers: Dict[str, tuple[int, int]] = {}
    started_tasks: set[str] = set()
    last_status_emit_ts = 0.0
    live_dashboard: Optional[Live] = None

    for row in seed_rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        run_dir = Path(str(row.get("run_dir") or ""))
        task_states[task_id] = _make_live_task_state(
            task_id,
            str(row.get("paper_path") or ""),
            run_dir,
            "success",
            started_at,
        )
        task_states[task_id]["last_update_at"] = str(row.get("ended_at") or started_at)
        task_states[task_id]["run_dir"] = str(run_dir)
        task_states[task_id]["stop_reason"] = _read_state_brief(run_dir).get("stop_reason") if run_dir.exists() else None
        if run_dir.exists():
            _hydrate_task_from_run_dir(task_states[task_id], run_dir)

    for job in jobs:
        task_id = str(job.get("task_id") or "")
        run_dir = Path(str(job.get("run_dir") or ""))
        task_states[task_id] = _make_live_task_state(
            task_id,
            str(job.get("paper_path") or ""),
            run_dir,
            "queued",
            started_at,
        )
        if run_dir.exists():
            _hydrate_task_from_run_dir(task_states[task_id], run_dir)
        step_markers[task_id] = (
            int(task_states[task_id].get("step") or 0),
            int(task_states[task_id].get("history_len") or 0),
        )
        log_path = run_dir / "run.log"
        log_offsets[task_id] = log_path.stat().st_size if log_path.exists() else 0
        _append_jsonl(
            live_events_path,
            {
                "timestamp": started_at,
                "event_type": "task_queued",
                "batch_id": batch_id,
                "task_id": task_id,
                "paper_path": str(job.get("paper_path") or ""),
                "run_dir": str(run_dir),
            },
        )
        if manifest.get("resumed_from_batch_dir") or manifest.get("heartbeat_root_batch_dir") or job.get("resume_run_dir") or job.get("resume_prepared"):
            _append_jsonl(
                live_events_path,
                {
                    "timestamp": started_at,
                    "event_type": "resume_spawned",
                    "batch_id": batch_id,
                    "task_id": task_id,
                    "paper_path": str(job.get("paper_path") or ""),
                    "run_dir": str(run_dir),
                    "source_batch_id": str(manifest.get("resumed_from_batch_id") or ""),
                },
            )

    initial_rows = sorted(results_by_task.values(), key=lambda item: str(item.get("task_id") or ""))
    _write_batch_results(results_path, initial_rows)
    summary_path.write_text(
        _build_batch_summary_md(batch_id=batch_id, rows=initial_rows),
        encoding="utf-8",
    )
    _write_json_atomic(
        live_status_path,
        _snapshot_live_status(
            batch_id=batch_id,
            batch_dir=batch_dir,
            manifest=manifest,
            task_states=task_states,
            model_counters=model_counters,
            role_counters=role_counters,
            started_at=started_at,
        ),
    )
    if inline_status:
        initial_snapshot = _snapshot_live_status(
            batch_id=batch_id,
            batch_dir=batch_dir,
            manifest=manifest,
            task_states=task_states,
            model_counters=model_counters,
            role_counters=role_counters,
            started_at=started_at,
        )
        live_dashboard = Live(
            _build_inline_batch_layout(
                initial_snapshot,
                active_limit=status_active_limit,
                top_errors=status_top_errors,
            ),
            refresh_per_second=4,
            auto_refresh=False,
            transient=False,
            console=Console(),
        )
        live_dashboard.start()
        live_dashboard.refresh()
        last_status_emit_ts = time.time()

    if not jobs:
        if live_dashboard is not None:
            live_dashboard.console.print(
                f"[BATCH] nothing to run: batch={batch_id} inherited_successes={len(results_by_task)} summary={summary_path}"
            )
            live_dashboard.stop()
        else:
            print(f"[BATCH] nothing to run: batch={batch_id} inherited_successes={len(results_by_task)} summary={summary_path}")
        return initial_rows

    try:
        with ProcessPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
            future_map = {executor.submit(_batch_agent_run_worker, job): job for job in jobs}
            pending = set(future_map.keys())
            while pending:
                now_iso = datetime.now(timezone.utc).isoformat()
                for future, job in future_map.items():
                    if future.done():
                        continue
                    task_id = str(job.get("task_id") or "")
                    task = task_states.get(task_id)
                    if task is None:
                        continue
                    if future.running() and task_id not in started_tasks:
                        started_tasks.add(task_id)
                        task["status"] = "running"
                        task["last_update_at"] = now_iso
                        _append_jsonl(
                            live_events_path,
                            {
                                "timestamp": now_iso,
                                "event_type": "task_started",
                                "batch_id": batch_id,
                                "task_id": task_id,
                                "run_dir": task.get("run_dir") or "",
                            },
                        )

                    run_dir = Path(str(task.get("run_dir") or ""))
                    brief = _read_state_brief(run_dir)
                    cur_step = int(brief.get("step") or 0)
                    cur_hist = int(brief.get("history_len") or 0)
                    prev_step, prev_hist = step_markers.get(task_id, (cur_step, cur_hist))
                    if (cur_step, cur_hist) != (prev_step, prev_hist):
                        step_markers[task_id] = (cur_step, cur_hist)
                        task["step"] = cur_step
                        task["history_len"] = cur_hist
                        task["stop_reason"] = brief.get("stop_reason")
                        task["last_update_at"] = now_iso
                        _append_jsonl(
                            live_events_path,
                            {
                                "timestamp": now_iso,
                                "event_type": "task_progress",
                                "batch_id": batch_id,
                                "task_id": task_id,
                                "step": cur_step,
                                "history_len": cur_hist,
                                "run_dir": task.get("run_dir") or "",
                            },
                        )

                    log_path = run_dir / "run.log"
                    if log_path.exists():
                        offset = log_offsets.get(task_id, 0)
                        size = log_path.stat().st_size
                        if size > offset:
                            with log_path.open("rb") as f:
                                f.seek(offset)
                                chunk = f.read(size - offset).decode("utf-8", errors="ignore")
                            log_offsets[task_id] = size
                            for line in chunk.splitlines():
                                role, stage = _infer_role_and_stage_from_line(
                                    line,
                                    str(task.get("latest_role") or "") or None,
                                    str(task.get("latest_stage") or "") or None,
                                )
                                if role:
                                    task["latest_role"] = role
                                if stage:
                                    task["latest_stage"] = stage
                                model = _extract_model_from_line(line)
                                if model and "Backend:" in line:
                                    task["latest_model"] = model
                                    _increment_model_counters(model_counters, model, request=True)
                                    bucket = _role_bucket(task.get("latest_role"))
                                    if bucket:
                                        role_counters[bucket] += 1
                                error_type = _classify_error_type(line)
                                if error_type:
                                    if model:
                                        task["latest_model"] = model
                                        _increment_model_counters(model_counters, model, error_type=error_type)
                                    task["last_error"] = line.strip()[:500]
                                    task["last_error_type"] = error_type
                                    task["last_update_at"] = now_iso
                                    _append_jsonl(
                                        live_events_path,
                                        {
                                            "timestamp": now_iso,
                                            "event_type": "request_error",
                                            "batch_id": batch_id,
                                            "task_id": task_id,
                                            "run_dir": task.get("run_dir") or "",
                                            "model": task.get("latest_model"),
                                            "role": task.get("latest_role"),
                                            "stage": task.get("latest_stage"),
                                            "error_type": error_type,
                                            "message": line.strip(),
                                        },
                                    )
                                    if "retrying" in line.lower() and task.get("latest_role") == "expression_judge":
                                        _append_jsonl(
                                            live_events_path,
                                            {
                                                "timestamp": now_iso,
                                                "event_type": "judge_retry",
                                                "batch_id": batch_id,
                                                "task_id": task_id,
                                                "run_dir": task.get("run_dir") or "",
                                                "model": task.get("latest_model"),
                                                "message": line.strip(),
                                            },
                                        )

                live_snapshot = _snapshot_live_status(
                    batch_id=batch_id,
                    batch_dir=batch_dir,
                    manifest=manifest,
                    task_states=task_states,
                    model_counters=model_counters,
                    role_counters=role_counters,
                    started_at=started_at,
                )
                _write_json_atomic(live_status_path, live_snapshot)
                if inline_status and live_dashboard is not None:
                    now_ts = time.time()
                    if now_ts - last_status_emit_ts >= max(1, int(status_interval_sec)):
                        live_dashboard.update(
                            _build_inline_batch_layout(
                                live_snapshot,
                                active_limit=status_active_limit,
                                top_errors=status_top_errors,
                            ),
                            refresh=True,
                        )
                        last_status_emit_ts = now_ts

                done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    pending.discard(future)
                    row = future.result()
                    task_id = str(row.get("task_id") or "")
                    task = task_states.setdefault(
                        task_id,
                        _make_live_task_state(task_id, str(row.get("paper_path") or ""), Path(str(row.get("run_dir") or "")), "running", now_iso),
                    )
                    task["status"] = str(row.get("status") or "unknown")
                    task["run_dir"] = str(row.get("run_dir") or task.get("run_dir") or "")
                    task["last_error"] = str(row.get("error") or "")
                    task["last_error_type"] = _classify_error_type(str(row.get("error") or ""))
                    task["last_update_at"] = str(row.get("ended_at") or now_iso)
                    brief = _read_state_brief(Path(task["run_dir"])) if task["run_dir"] else {"step": 0, "history_len": 0, "stop_reason": None}
                    task["step"] = int(brief.get("step") or task.get("step") or 0)
                    task["history_len"] = int(brief.get("history_len") or task.get("history_len") or 0)
                    task["stop_reason"] = brief.get("stop_reason")
                    results_by_task[task_id] = row
                    merged_rows = sorted(results_by_task.values(), key=lambda item: str(item.get("task_id") or ""))
                    _write_batch_results(results_path, merged_rows)
                    summary_path.write_text(_build_batch_summary_md(batch_id=batch_id, rows=merged_rows), encoding="utf-8")
                    event_type = "task_succeeded" if task["status"] == "success" else "task_failed"
                    _append_jsonl(
                        live_events_path,
                        {
                            "timestamp": task["last_update_at"],
                            "event_type": event_type,
                            "batch_id": batch_id,
                            "task_id": task_id,
                            "run_dir": task.get("run_dir") or "",
                            "status": task["status"],
                            "error": task.get("last_error") or "",
                        },
                    )
                    if live_dashboard is not None:
                        live_dashboard.console.print(f"[BATCH] {row.get('status')}: {row.get('paper_path')} -> {row.get('run_dir')}")
                    else:
                        print(f"[BATCH] {row.get('status')}: {row.get('paper_path')} -> {row.get('run_dir')}")
                    live_snapshot = _snapshot_live_status(
                        batch_id=batch_id,
                        batch_dir=batch_dir,
                        manifest=manifest,
                        task_states=task_states,
                        model_counters=model_counters,
                        role_counters=role_counters,
                        started_at=started_at,
                    )
                    _write_json_atomic(live_status_path, live_snapshot)
                    if inline_status and live_dashboard is not None:
                        live_dashboard.update(
                            _build_inline_batch_layout(
                                live_snapshot,
                                active_limit=status_active_limit,
                                top_errors=status_top_errors,
                            ),
                            refresh=True,
                        )
                        last_status_emit_ts = time.time()
    finally:
        if live_dashboard is not None:
            final_snapshot = _snapshot_live_status(
                batch_id=batch_id,
                batch_dir=batch_dir,
                manifest=manifest,
                task_states=task_states,
                model_counters=model_counters,
                role_counters=role_counters,
                started_at=started_at,
            )
            _write_json_atomic(live_status_path, final_snapshot)
            live_dashboard.update(
                _build_inline_batch_layout(
                    final_snapshot,
                    active_limit=status_active_limit,
                    top_errors=status_top_errors,
                ),
                refresh=True,
            )
            live_dashboard.stop()

    final_rows = sorted(results_by_task.values(), key=lambda item: str(item.get("task_id") or ""))
    failures = [row for row in final_rows if row.get("status") != "success"]
    print(f"[BATCH] complete: total={len(final_rows)} success={len(final_rows) - len(failures)} failed={len(failures)} summary={summary_path}")
    final_snapshot = _snapshot_live_status(
        batch_id=batch_id,
        batch_dir=batch_dir,
        manifest=manifest,
        task_states=task_states,
        model_counters=model_counters,
        role_counters=role_counters,
        started_at=started_at,
    )
    _write_json_atomic(live_status_path, final_snapshot)
    if failures and not continue_on_error:
        raise SystemExit(1)
    return final_rows


def _batch_agent_run_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    run_id = str(job.get("run_id") or "")
    run_dir = str(job.get("run_dir") or "")
    paper_path = str(job.get("paper_path") or "")
    task_id = str(job.get("task_id") or "")
    resume_run_dir = job.get("resume_run_dir")
    args = argparse.Namespace(
        config=str(job.get("config_path") or ""),
        engine="langgraph",
        output=str(job.get("output_root") or ""),
        run_id=run_id,
        no_playback=bool(job.get("no_playback", False)),
        resume_run_dir=str(resume_run_dir) if resume_run_dir else None,
        resume_clone=bool(job.get("resume_clone", False)),
        resume_prepared=bool(job.get("resume_prepared", False)),
        resume_suffix=str(job.get("resume_suffix") or "resume"),
        input_kind="paper",
        paper_path=paper_path,
        episode_seed_contract=None,
        idealab_session_id=None,
        set_env=None,
        unset_env=None,
        max_steps=job.get("max_steps"),
        max_rounds=job.get("max_rounds"),
        max_consecutive_revise=job.get("max_consecutive_revise"),
        revise_retry_limit=job.get("max_consecutive_revise"),
        lang=job.get("lang"),
        symbolic_only=False,
        no_mcq_from_step=None,
        allowed_question_types=None,
        roles_protocol=None,
        format_validation_mode=None,
        source=job.get("source"),
        main_model=None,
        strong_models=None,
        medium_model=None,
        struct_model=None,
        format_model=None,
        client_stream=False,
        consensus_mode=None,
        all_strong=False,
        agent_meta={
            "source": job.get("source"),
            "entrypoint": "agent-batch-run",
            "batch_id": job.get("batch_id"),
            "task_id": task_id,
        },
        func=None,
        command="agent-run",
    )

    def _final_run_dir() -> str:
        resumed = getattr(args, "resume_run_dir", None)
        if resumed:
            return str(resumed)
        return run_dir

    try:
        cmd_agent_run(args)
        ended_at = datetime.now(timezone.utc)
        return {
            "task_id": task_id,
            "paper_path": paper_path,
            "run_id": run_id,
            "run_dir": _final_run_dir(),
            "resume_from_run_dir": str(resume_run_dir or ""),
            "status": "success",
            "error": "",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_sec": round((ended_at - started_at).total_seconds(), 3),
        }
    except SystemExit as exc:
        ended_at = datetime.now(timezone.utc)
        code = getattr(exc, "code", 1)
        return {
            "task_id": task_id,
            "paper_path": paper_path,
            "run_id": run_id,
            "run_dir": _final_run_dir(),
            "resume_from_run_dir": str(resume_run_dir or ""),
            "status": "failed",
            "error": f"SystemExit({code})",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_sec": round((ended_at - started_at).total_seconds(), 3),
        }
    except Exception as exc:  # noqa: BLE001
        ended_at = datetime.now(timezone.utc)
        return {
            "task_id": task_id,
            "paper_path": paper_path,
            "run_id": run_id,
            "run_dir": _final_run_dir(),
            "resume_from_run_dir": str(resume_run_dir or ""),
            "status": "failed",
            "error": str(exc),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_sec": round((ended_at - started_at).total_seconds(), 3),
        }


def _parse_task_id_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return set(parts)
    if isinstance(raw, (list, tuple, set)):
        out: set[str] = set()
        for item in raw:
            if isinstance(item, str):
                out.update(_parse_task_id_set(item))
        return out
    return set()


def _read_json_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"[ERROR] invalid json object: {path}")
    return data


def _read_tail_text(path: Path, max_bytes: int = 65536) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        seek_pos = max(0, size - max_bytes)
        f.seek(seek_pos)
        return f.read().decode("utf-8", errors="ignore")


def _is_retryable_api_failure(*parts: str) -> bool:
    text = "\n".join(part for part in parts if part).lower()
    if not text:
        return False
    needles = [
        "status=429",
        "status=500",
        "status=502",
        "status=503",
        "status=504",
        "proxyerror",
        "invalidchunklength",
        "chunkedencodingerror",
        "remotedisconnected",
        "connectionerror",
        "readtimeout",
        "gateway time-out",
        "gateway timeout",
        "tengine ingress",
        "http chat failed",
    ]
    return any(needle in text for needle in needles)


def _find_primary_run_dir_for_task(batch_dir: Path, task_id: str) -> Optional[Path]:
    m = re.fullmatch(r"task_(\d{3})", task_id)
    if not m:
        return None
    idx = m.group(1)
    runs_root = batch_dir / "runs"
    if not runs_root.exists():
        return None
    for candidate in sorted(runs_root.glob(f"run_paper{idx}_*")):
        if candidate.is_dir() and _is_primary_batch_run_dir(candidate):
            return candidate
    return None


def _read_task_stop_reason(run_dir: Optional[Path]) -> Optional[str]:
    if run_dir is None:
        return None
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        data = _read_json_file(state_path)
        raw = data.get("stop_reason")
        return str(raw) if raw else None
    except Exception:
        return None


def _build_heartbeat_summary_md(*, root_batch_dir: Path, snapshot: Dict[str, Any]) -> str:
    counts = snapshot.get("counts") or {}
    lines = [
        f"# Heartbeat Summary: {root_batch_dir.name}",
        "",
        f"- Root batch: {root_batch_dir}",
        f"- Sweep time: {snapshot.get('timestamp')}",
        f"- Lineage: {', '.join(snapshot.get('lineage_batch_ids') or [])}",
        f"- Success: {counts.get('success', 0)}",
        f"- Failed: {counts.get('failed', 0)}",
        f"- Running: {counts.get('running', 0)}",
        f"- Retry exhausted: {counts.get('retry_exhausted', 0)}",
        f"- Retryable failed: {len(snapshot.get('retryable_failed_task_ids') or [])}",
        "",
        "| Task ID | Status | Retry Count | Retryable | Latest Batch | Resume Source | Error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    tasks = snapshot.get("tasks") or {}
    for task_id in sorted(tasks.keys()):
        row = tasks[task_id]
        lines.append(
            "| {task_id} | {status} | {retry_count} | {retryable} | {latest_batch_id} | {resume_source_batch_id} | {error} |".format(
                task_id=task_id,
                status=row.get("status") or "",
                retry_count=row.get("retry_count") or 0,
                retryable="yes" if row.get("retryable") else "no",
                latest_batch_id=row.get("latest_batch_id") or "",
                resume_source_batch_id=row.get("resume_source_batch_id") or "",
                error=str(row.get("error") or "").replace("\n", " "),
            )
        )
    return "\n".join(lines).strip() + "\n"


def _build_heartbeat_snapshot(
    *,
    root_batch_dir: Path,
    max_auto_resumes_per_task: int,
) -> Dict[str, Any]:
    lineage = _load_batch_lineage(root_batch_dir)
    root_manifest = lineage[0]["manifest"]
    paper_count = int(root_manifest.get("paper_count") or len(root_manifest.get("papers") or []))
    task_ids = [f"task_{idx:03d}" for idx in range(1, paper_count + 1)]

    retry_counts: Dict[str, int] = {task_id: 0 for task_id in task_ids}
    for item in lineage[1:]:
        manifest = item["manifest"]
        for task_id in manifest.get("heartbeat_resumed_task_ids") or []:
            if task_id in retry_counts:
                retry_counts[task_id] += 1

    tasks: Dict[str, Dict[str, Any]] = {}
    counts = {"success": 0, "failed": 0, "running": 0, "retry_exhausted": 0, "unknown": 0}
    retryable_failed_task_ids: list[str] = []

    for task_id in task_ids:
        latest_terminal_row: Optional[Dict[str, Any]] = None
        latest_terminal_order = -1
        latest_batch_path: Optional[Path] = None
        latest_batch_id = ""
        latest_run_dir: Optional[Path] = None
        latest_run_order = -1
        latest_run_batch_id = ""
        latest_error = ""
        latest_live_row: Optional[Dict[str, Any]] = None

        for item in lineage:
            live = _read_live_status_file(Path(item["path"]))
            if isinstance(live, dict):
                task_live = (live.get("tasks") or {}).get(task_id)
                if isinstance(task_live, dict):
                    latest_live_row = task_live
            row = next((r for r in item["results"] if str(r.get("task_id") or "") == task_id), None)
            if row is not None:
                latest_terminal_row = row
                latest_terminal_order = int(item["order"])
                latest_batch_path = Path(item["path"])
                latest_batch_id = str(item["manifest"].get("batch_id") or Path(item["path"]).name)
                latest_error = str(row.get("error") or "")
            run_dir = _find_primary_run_dir_for_task(Path(item["path"]), task_id)
            if run_dir is not None:
                latest_run_dir = run_dir
                latest_run_order = int(item["order"])
                latest_run_batch_id = str(item["manifest"].get("batch_id") or Path(item["path"]).name)

        live_status = str((latest_live_row or {}).get("status") or "")
        if live_status == "failed":
            status = "failed"
        elif latest_run_order > latest_terminal_order or live_status in {"queued", "running"}:
            status = "running"
        elif latest_terminal_row is not None:
            status = str(latest_terminal_row.get("status") or "unknown")
        elif latest_live_row is not None:
            status = str(latest_live_row.get("status") or "unknown")
        elif latest_run_dir is not None:
            status = "running"
        else:
            status = "unknown"

        stop_reason = (latest_live_row or {}).get("stop_reason") or _read_task_stop_reason(latest_run_dir)
        run_log_tail = _read_tail_text(latest_run_dir / "run.log") if latest_run_dir else ""
        retryable = status == "failed" and _is_retryable_api_failure(latest_error, str((latest_live_row or {}).get("last_error") or ""), run_log_tail)
        retry_exhausted = retryable and retry_counts.get(task_id, 0) >= int(max_auto_resumes_per_task)

        if retryable and not retry_exhausted:
            retryable_failed_task_ids.append(task_id)

        counts[status if status in counts else "unknown"] += 1
        if retry_exhausted:
            counts["retry_exhausted"] += 1

        source_batch_id = latest_run_batch_id or latest_batch_id
        tasks[task_id] = {
            "task_id": task_id,
            "paper_path": (
                str((root_manifest.get("papers") or [])[int(task_id[-3:]) - 1])
                if 0 <= int(task_id[-3:]) - 1 < len(root_manifest.get("papers") or [])
                else ""
            ),
            "status": status,
            "retry_count": retry_counts.get(task_id, 0),
            "retryable": retryable,
            "retry_exhausted": retry_exhausted,
            "latest_batch_id": latest_batch_id or latest_run_batch_id,
            "latest_run_dir": str(latest_run_dir) if latest_run_dir else "",
            "resume_source_batch_id": source_batch_id,
            "resume_source_run_dir": str(latest_run_dir) if latest_run_dir else "",
            "stop_reason": stop_reason,
            "error": str((latest_live_row or {}).get("last_error") or latest_error),
        }

    next_attempt_index = max(int(item.get("attempt_index") or 0) for item in lineage) + 1
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "root_batch_dir": str(root_batch_dir),
        "root_batch_id": str(root_manifest.get("batch_id") or root_batch_dir.name),
        "lineage_batch_ids": [str(item["manifest"].get("batch_id") or Path(item["path"]).name) for item in lineage],
        "counts": counts,
        "tasks": tasks,
        "retryable_failed_task_ids": sorted(retryable_failed_task_ids),
        "next_heartbeat_attempt_index": next_attempt_index,
        "root_manifest": root_manifest,
    }


def cmd_agent_batch_status(args: argparse.Namespace) -> None:
    batch_dir = Path(str(args.batch_dir))
    if not batch_dir.is_absolute():
        batch_dir = REPO_ROOT / batch_dir
    if not batch_dir.exists():
        raise SystemExit(f"[ERROR] batch dir does not exist: {batch_dir}")

    interval_sec = max(1, int(getattr(args, "interval_sec", 10)))
    stale_sec = max(1, int(getattr(args, "stale_sec", 300)))
    fmt = str(getattr(args, "format", None) or "table")
    show_runs = str(getattr(args, "show_runs", None) or "active")
    top_errors = max(1, int(getattr(args, "top_errors", 10)))
    watch = bool(getattr(args, "watch", False))

    root_batch_dir = _resolve_lineage_root(batch_dir)
    monitor_snapshot_path = root_batch_dir / "monitor_snapshot.json"
    monitor_summary_path = root_batch_dir / "monitor_summary.md"

    while True:
        snapshot = _collect_status_snapshot(batch_dir=batch_dir, stale_sec=stale_sec)
        _write_json_atomic(monitor_snapshot_path, snapshot)
        md = _build_status_markdown(snapshot, show_runs=show_runs, top_errors=top_errors)
        monitor_summary_path.write_text(md, encoding="utf-8")

        if fmt == "json":
            print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        else:
            print(md, end="")

        if not watch:
            return
        time.sleep(interval_sec)


def _launch_heartbeat_resumed_batch(
    *,
    root_batch_dir: Path,
    snapshot: Dict[str, Any],
    task_ids: list[str],
    config_path: Path,
    concurrency: int,
    no_playback: bool,
    source: Optional[str],
    lang: Optional[str],
    max_steps: Optional[int],
    max_rounds: Optional[int],
    max_consecutive_revise: Optional[int],
    resume_suffix: str,
) -> Dict[str, Any]:
    if not task_ids:
        raise SystemExit("[ERROR] heartbeat resumed batch requires non-empty task_ids")

    attempt_index = int(snapshot.get("next_heartbeat_attempt_index") or 1)
    root_manifest = snapshot["root_manifest"]
    batch_dir = root_batch_dir
    batch_id = str(root_manifest.get("batch_id") or root_batch_dir.name)
    runs_root = batch_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    seed_rows: list[Dict[str, Any]] = []
    jobs: list[Dict[str, Any]] = []

    latest_rows = _collect_latest_lineage_rows(root_batch_dir)
    batch_archive_root = batch_dir / "_resume_archive"
    batch_attempt_dir, batch_attempt_index = _next_archive_attempt_dir(batch_archive_root)
    batch_attempt_dir.mkdir(parents=True, exist_ok=True)
    archived_batch_files: list[str] = []
    for name in ["batch_results.jsonl", "batch_summary.md", "live_status.json", "live_events.jsonl", "monitor_snapshot.json", "monitor_summary.md"]:
        src = batch_dir / name
        if not src.exists():
            continue
        _copy_path(src, batch_attempt_dir / name)
        archived_batch_files.append(name)

    source_map: Dict[str, str] = {}
    resume_manifest: Dict[str, Any] = {
        "attempt_index": batch_attempt_index,
        "heartbeat_attempt_index": attempt_index,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "resume_reason": "heartbeat_auto_resume",
        "source_batch_dir": str(root_batch_dir),
        "root_batch_dir": str(root_batch_dir),
        "root_batch_id": batch_id,
        "archived_batch_files": archived_batch_files,
        "tasks": [],
    }

    for task_id, task in (snapshot.get("tasks") or {}).items():
        latest_row = latest_rows.get(task_id)
        latest_run_dir_raw = str((latest_row or {}).get("run_dir") or task.get("latest_run_dir") or "")
        latest_run_dir = Path(latest_run_dir_raw).resolve() if latest_run_dir_raw else None
        target_run_dir = _find_primary_run_dir_for_task(batch_dir, task_id)
        if latest_run_dir is not None:
            if target_run_dir is None:
                target_run_dir = runs_root / latest_run_dir.name
            materialize_info = _materialize_run_dir(
                source_run_dir=latest_run_dir,
                target_run_dir=target_run_dir,
                archive_dir=batch_attempt_dir,
            )
        else:
            materialize_info = {
                "materialized": False,
                "source_run_dir": "",
                "target_run_dir": str(target_run_dir) if target_run_dir else "",
                "archived_previous_target": "",
            }

        latest_status = str((latest_row or {}).get("status") or task.get("status") or "")
        if latest_status == "success":
            seed_rows.append(
                {
                    "task_id": task_id,
                    "paper_path": (latest_row or {}).get("paper_path") or task.get("paper_path") or "",
                    "run_id": str((latest_row or {}).get("run_id") or ""),
                    "run_dir": str(target_run_dir) if target_run_dir else str((latest_row or {}).get("run_dir") or ""),
                    "status": "success",
                    "error": "",
                    "started_at": str((latest_row or {}).get("started_at") or ""),
                    "ended_at": str((latest_row or {}).get("ended_at") or ""),
                    "duration_sec": float((latest_row or {}).get("duration_sec") or 0.0),
                }
            )
            continue

        if task_id not in task_ids:
            continue
        if target_run_dir is None or not target_run_dir.exists():
            raise SystemExit(f"[ERROR] heartbeat resume source missing for {task_id} under {batch_dir}")
        run_archive_info = _prepare_run_dir_for_in_place_resume(
            target_run_dir,
            resume_reason="heartbeat_auto_resume",
            batch_id=batch_id,
            batch_archive_attempt_dir=batch_attempt_dir,
        )
        source_map[task_id] = str(root_batch_dir)
        jobs.append(
            {
                "task_id": task_id,
                "batch_id": batch_id,
                "config_path": str(config_path),
                "paper_path": task.get("paper_path") or "",
                "run_id": _new_run_id(f"hb{task_id[-3:]}"),
                "run_dir": str(target_run_dir),
                "output_root": str(runs_root),
                "source": source,
                "lang": lang,
                "max_steps": max_steps,
                "max_rounds": max_rounds,
                "max_consecutive_revise": max_consecutive_revise,
                "no_playback": no_playback,
                "resume_run_dir": str(target_run_dir),
                "resume_clone": False,
                "resume_prepared": True,
                "resume_suffix": resume_suffix,
            }
        )
        resume_manifest["tasks"].append(
            {
                "task_id": task_id,
                "paper_path": task.get("paper_path") or "",
                "previous_status": latest_status or "unknown",
                "latest_source_run_dir": str(latest_run_dir) if latest_run_dir else "",
                "materialize": materialize_info,
                "run_archive_dir": str(run_archive_info["archive_dir"]),
            }
        )

    root_batch_manifest = dict(root_manifest)
    root_batch_manifest["last_resumed_at"] = datetime.now(timezone.utc).isoformat()
    root_batch_manifest["resume_attempt_count"] = int(root_manifest.get("resume_attempt_count") or 0) + 1
    root_batch_manifest["last_resume_source_batch_dir"] = str(root_batch_dir)
    root_batch_manifest["last_resume_source_batch_id"] = batch_id
    root_batch_manifest["heartbeat_root_batch_dir"] = str(root_batch_dir.resolve())
    root_batch_manifest["heartbeat_attempt_index"] = attempt_index
    root_batch_manifest["heartbeat_resumed_task_ids"] = list(task_ids)
    root_batch_manifest["heartbeat_resume_sources"] = source_map
    root_batch_manifest["source"] = source
    root_batch_manifest["lang"] = lang
    root_batch_manifest["max_steps"] = max_steps
    root_batch_manifest["max_rounds"] = max_rounds
    root_batch_manifest["max_consecutive_revise"] = max_consecutive_revise
    (batch_dir / "batch_manifest.json").write_text(json.dumps(root_batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_attempt_dir / "resume_manifest.json").write_text(json.dumps(resume_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_resume_history_line(
        batch_dir,
        {
            "timestamp": resume_manifest["archived_at"],
            "event_type": "heartbeat_resume_started",
            "attempt_index": batch_attempt_index,
            "heartbeat_attempt_index": attempt_index,
            "root_batch_dir": str(batch_dir),
            "resumed_task_ids": [item["task_id"] for item in resume_manifest["tasks"]],
            "inherited_successes": len(seed_rows),
        },
    )
    rows = _run_batch_jobs(
        batch_id=batch_id,
        batch_dir=batch_dir,
        jobs=jobs,
        seed_rows=seed_rows,
        concurrency=concurrency,
        continue_on_error=True,
        inline_status=False,
    )
    latest_row_map = {str(row.get("task_id") or ""): row for row in rows if row.get("task_id")}
    finalized_tasks: list[Dict[str, Any]] = []
    for item in resume_manifest.get("tasks") or []:
        task_id = str(item.get("task_id") or "")
        row = latest_row_map.get(task_id, {})
        run_dir = Path(str(row.get("run_dir") or item.get("materialize", {}).get("target_run_dir") or ""))
        brief = _read_state_brief(run_dir) if run_dir.exists() else {"step": 0, "history_len": 0, "stop_reason": None}
        finalized_tasks.append(
            {
                **item,
                "final_status": row.get("status") or item.get("previous_status") or "unknown",
                "final_error": row.get("error") or "",
                "final_run_dir": str(run_dir) if run_dir else "",
                "final_step": brief.get("step"),
                "final_history_len": brief.get("history_len"),
                "final_stop_reason": brief.get("stop_reason"),
            }
        )
    resume_manifest["finalized_at"] = datetime.now(timezone.utc).isoformat()
    resume_manifest["finalized_tasks"] = finalized_tasks
    (batch_attempt_dir / "resume_manifest.json").write_text(json.dumps(resume_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_resume_history_line(
        batch_dir,
        {
            "timestamp": resume_manifest["finalized_at"],
            "event_type": "heartbeat_resume_finalized",
            "attempt_index": batch_attempt_index,
            "heartbeat_attempt_index": attempt_index,
            "root_batch_dir": str(batch_dir),
            "finalized_tasks": [
                {
                    "task_id": item.get("task_id"),
                    "final_status": item.get("final_status"),
                    "final_stop_reason": item.get("final_stop_reason"),
                }
                for item in finalized_tasks
            ],
        },
    )
    return {
        "batch_dir": str(batch_dir),
        "batch_id": batch_id,
        "archive_attempt_dir": str(batch_attempt_dir),
        "task_ids": list(task_ids),
        "rows": rows,
    }


def cmd_agent_batch_heartbeat(args: argparse.Namespace) -> None:
    root_batch_dir = Path(str(args.batch_dir))
    if not root_batch_dir.is_absolute():
        root_batch_dir = REPO_ROOT / root_batch_dir
    if not root_batch_dir.exists():
        raise SystemExit(f"[ERROR] batch dir does not exist: {root_batch_dir}")
    root_manifest = _read_json_file(root_batch_dir / "batch_manifest.json")

    config_path = Path(root_manifest.get("config_path") or args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    interval_sec = max(1, int(getattr(args, "interval_sec", 60)))
    max_auto_resumes = max(0, int(getattr(args, "max_auto_resumes_per_task", 2)))
    source = getattr(args, "source", None) or root_manifest.get("source")
    lang = getattr(args, "lang", None) or root_manifest.get("lang")
    max_steps = getattr(args, "max_steps", None) if getattr(args, "max_steps", None) is not None else root_manifest.get("max_steps")
    max_rounds = getattr(args, "max_rounds", None) if getattr(args, "max_rounds", None) is not None else root_manifest.get("max_rounds")
    max_consecutive_revise = (
        getattr(args, "max_consecutive_revise", None)
        if getattr(args, "max_consecutive_revise", None) is not None
        else root_manifest.get("max_consecutive_revise")
    )
    concurrency = getattr(args, "concurrency", None) if getattr(args, "concurrency", None) is not None else root_manifest.get("concurrency") or 2
    resume_suffix = str(getattr(args, "resume_suffix", None) or "hb")
    continue_until = str(getattr(args, "continue_until", None) or "terminal")
    heartbeat_path = root_batch_dir / "heartbeat.jsonl"
    heartbeat_summary_path = root_batch_dir / "heartbeat_summary.md"
    last_failed: set[str] = set()

    while True:
        snapshot = _build_heartbeat_snapshot(
            root_batch_dir=root_batch_dir,
            max_auto_resumes_per_task=max_auto_resumes,
        )
        current_failed = {
            task_id for task_id, row in (snapshot.get("tasks") or {}).items()
            if row.get("status") == "failed"
        }
        resumable = list(snapshot.get("retryable_failed_task_ids") or [])
        newly_failed = sorted(current_failed - last_failed)
        last_failed = current_failed
        launched: Optional[Dict[str, Any]] = None
        if resumable:
            launched = _launch_heartbeat_resumed_batch(
                root_batch_dir=root_batch_dir,
                snapshot=snapshot,
                task_ids=resumable,
                config_path=config_path,
                concurrency=int(concurrency),
                no_playback=bool(getattr(args, "no_playback", False)),
                source=source,
                lang=lang,
                max_steps=max_steps,
                max_rounds=max_rounds,
                max_consecutive_revise=max_consecutive_revise,
                resume_suffix=resume_suffix,
            )
            snapshot = _build_heartbeat_snapshot(
                root_batch_dir=root_batch_dir,
                max_auto_resumes_per_task=max_auto_resumes,
            )

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "root_batch_dir": str(root_batch_dir),
            "lineage_batch_ids": snapshot.get("lineage_batch_ids") or [],
            "counts": snapshot.get("counts") or {},
            "newly_failed_task_ids": newly_failed,
            "retryable_failed_task_ids": snapshot.get("retryable_failed_task_ids") or [],
            "auto_resumed_task_ids": launched.get("task_ids") if launched else [],
            "auto_resumed_batch_dir": launched.get("batch_dir") if launched else "",
            "auto_resumed_archive_attempt_dir": launched.get("archive_attempt_dir") if launched else "",
            "max_auto_resumes_per_task": max_auto_resumes,
        }
        with heartbeat_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        heartbeat_summary_path.write_text(
            _build_heartbeat_summary_md(root_batch_dir=root_batch_dir, snapshot=snapshot),
            encoding="utf-8",
        )

        tasks = snapshot.get("tasks") or {}
        all_terminal = all(str(row.get("status") or "") in {"success", "failed"} for row in tasks.values())
        retryable_left = bool(snapshot.get("retryable_failed_task_ids"))
        if continue_until == "terminal" and all_terminal and not retryable_left:
            print(f"[HEARTBEAT] terminal: root={root_batch_dir} summary={heartbeat_summary_path}")
            return
        time.sleep(interval_sec)


def cmd_extend_upgrade(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    logger = setup_logger('agenqa.extend', config['logging'], stage='extend_upgrade')
    try:
        from agenqa.skills.extend_upgrade import ExtendUpgradeRunner, ExtendUpgradeConfig  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.error("extend-upgrade 命令在当前分支不可用（缺少 agenqa.skills.extend_upgrade）：%s", str(exc))
        logger.error("建议使用 `python3 src/cli.py agent ...` 或 `python3 src/cli.py agent-run ...` 跑 LangGraph 链路。")
        sys.exit(2)

    # 优先使用 extend_upgrade 配置块，否则回退到 seed
    extend_upgrade_block = config.get('extend_upgrade') or {}
    generator_config: Dict[str, Any] = dict(
        (extend_upgrade_block.get('generator')
         or config.get('seed', {}).get('generator') or {})
    )

    # 优先使用配置文件中的 prompt_path，否则使用命令行参数，最后使用默认值
    if args.prompt:
        prompt_path = Path(args.prompt)
    elif extend_upgrade_block.get('prompt_path'):
        prompt_path = Path(extend_upgrade_block['prompt_path'])
    else:
        prompt_path = REPO_ROOT / 'prompts' / 'py_style' / 'extend_upgrade.prompt'
    try:
        logger.info("Using prompt: %s", str(prompt_path))
    except Exception:
        pass

    # 允许通过 --service-id 从 llm_service 覆盖服务与默认生成参数（优先级：CLI > 配置 > llm_service 默认）
    if getattr(args, 'service_id', None):
        service_config_path = (
            Path(args.service_config)
            if getattr(args, 'service_config', None)
            else DEFAULT_SERVICE_CONFIG
        )
        try:
            service_full_config = load_llm_service_full_config(
                service_config_path,
                args.service_id,
                explicit_model=getattr(args, 'service_model', None),
                fallback_model=generator_config.get('model_name'),
            )
            # 深合并：配置文件作为 base，命令行 service 配置覆盖（命令行优先）
            generator_config = _deep_merge(generator_config, service_full_config)
            # 最后应用 CLI 显式模型覆盖（若提供）
            if getattr(args, 'service_model', None):
                generator_config['model_name'] = args.service_model
            logger.info(
                "已根据 llm_service 覆盖 extend-upgrade 服务: service_id=%s configured_base=%s model=%s",
                generator_config.get('service_id'),
                generator_config.get('api_base') or generator_config.get('base_url'),
                generator_config.get('model_name'),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"加载 llm_service 配置失败: {exc}")
            sys.exit(1)

    # 若未显式传 --service-id，但配置仅含 service_id（缺少 api_base/model），则自动解析（保持配置优先）
    if generator_config.get('service_id') and (not generator_config.get('api_base') or not generator_config.get('model_name')):
        service_config_path = (
            Path(args.service_config)
            if getattr(args, 'service_config', None)
            else DEFAULT_SERVICE_CONFIG
        )
        try:
            service_full_config = load_llm_service_full_config(
                service_config_path,
                generator_config.get('service_id'),
                fallback_model=generator_config.get('model_name'),
            )
            generator_config = _deep_merge(service_full_config, generator_config)
            logger.info(
                "已自动从 llm_service 解析 extend-upgrade 服务: service_id=%s configured_base=%s model=%s",
                generator_config.get('service_id'),
                generator_config.get('api_base') or generator_config.get('base_url'),
                generator_config.get('model_name'),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"加载 llm_service 配置失败: {exc}")
            sys.exit(1)

    # 显式 CLI 覆盖 client.timeout（高于 services.json 与本地配置）
    try:
        if getattr(args, 'client_timeout', None) is not None:
            if 'client' not in generator_config or not isinstance(generator_config.get('client'), dict):
                generator_config['client'] = {}
            generator_config['client']['timeout'] = int(args.client_timeout)
            logger.info("命令行覆盖 client.timeout=%s", generator_config['client']['timeout'])
    except Exception:
        pass

    # 显式 CLI 覆盖 generation 参数
    try:
        gen = generator_config.get('generation')
        if not isinstance(gen, dict):
            gen = {}
            generator_config['generation'] = gen
        changed = []
        if getattr(args, 'max_tokens', None) is not None:
            gen['max_tokens'] = int(args.max_tokens); changed.append('max_tokens')
        if getattr(args, 'temperature', None) is not None:
            gen['temperature'] = float(args.temperature); changed.append('temperature')
        if getattr(args, 'top_p', None) is not None:
            gen['top_p'] = float(args.top_p); changed.append('top_p')
        if getattr(args, 'top_k', None) is not None:
            gen['top_k'] = int(args.top_k); changed.append('top_k')
        if getattr(args, 'min_p', None) is not None:
            gen['min_p'] = float(args.min_p); changed.append('min_p')
        if changed:
            logger.info("命令行覆盖 generation: %s", ",".join(changed))
    except Exception:
        pass

    # 符号-only 开关：CLI 优先，其次配置项 extend_upgrade.symbolic_only / allow_numeric_values
    symbolic_only = False
    try:
        cfg_block = extend_upgrade_block if isinstance(extend_upgrade_block, dict) else {}
        if getattr(args, 'symbolic_only', False):
            symbolic_only = True
        else:
            if isinstance(cfg_block.get('symbolic_only'), bool):
                symbolic_only = bool(cfg_block.get('symbolic_only'))
            elif isinstance(cfg_block.get('allow_numeric_values'), bool):
                symbolic_only = not bool(cfg_block.get('allow_numeric_values'))
    except Exception:
        symbolic_only = False

    try:
        # 构造运行器
        ext_cfg = ExtendUpgradeConfig(
            generator=generator_config,
            prompt_path=Path(prompt_path),
            symbolic_only=symbolic_only,
        )
        runner = ExtendUpgradeRunner(ext_cfg)

        # 路径推导模式：支持脚本传入 --qa-root/--extend-root 与目标 step（使用 --target-step）
        if getattr(args, 'qa_root', None) or getattr(args, 'extend_root', None):
            if getattr(args, 'target_step', None) is not None:
                target_step = int(args.target_step)
            else:
                logger.error("extend-upgrade: 路径模式需要提供 --target-step")
                sys.exit(2)

            if target_step < 1:
                logger.error("extend-upgrade: target-step 必须 >= 1")
                sys.exit(2)

            qa_root = Path(args.qa_root) if getattr(args, 'qa_root', None) else None
            extend_root = Path(args.extend_root) if getattr(args, 'extend_root', None) else None
            if extend_root is None:
                logger.error("extend-upgrade: 缺少 --extend-root")
                sys.exit(2)

            if target_step == 1:
                if qa_root is None:
                    logger.error("extend-upgrade: target-step=1 需要提供 --qa-root（用于定位 init/qa_init 的 step_0 KQA 输出）")
                    sys.exit(2)
                kqa_path = qa_root / 'qa_init_raw_step_0_kqa.jsonl'
                current_step = 0
            else:
                kqa_path = extend_root / f'extend_kqa_step_{target_step - 1}.jsonl'
                current_step = target_step - 1

            if not kqa_path.exists():
                logger.error("extend-upgrade: 输入 KQA 文件不存在: %s", str(kqa_path))
                sys.exit(2)

            output_path = extend_root / f'extend_kqa_step_{target_step}.jsonl'

            # 并发度：优先命令行，其次配置中的 concurrency.max_workers
            try:
                from utils import load_config as _load_cfg_for_conc
                _cfg_for_conc = _load_cfg_for_conc(args.config)
                default_workers = int((_cfg_for_conc.get('concurrency') or {}).get('max_workers', 1))
            except Exception:
                default_workers = 1
            concurrency = int(getattr(args, 'concurrency', None) or default_workers)

            current_step = target_step - 1
            next_step = target_step
            runner.run(kqa_path, output_path, append=False, step_override=current_step, concurrency=concurrency)
            logger.info("Extend-upgrade 完成，step=%s -> %s，输出: %s", current_step, next_step, output_path)
            return

        # 直接模式：要求 --kqa 与 --output，按 i -> i+1 语义
        if not getattr(args, 'kqa', None) or not getattr(args, 'output', None):
            logger.error("extend-upgrade: 需要同时提供 --kqa 与 --output，或使用 --qa-root/--extend-root 组合")
            sys.exit(2)

        # 计算当前 step 与 next_step（默认从 KQA 第一条记录推断）
        current_step = None
        # 优先显式 --current-step；否则从内容推断
        if getattr(args, 'current_step', None) is not None:
            current_step = int(args.current_step)
        else:
            try:
                with open(args.kqa, 'r', encoding='utf-8') as f:
                    first_line = None
                    for line in f:
                        line = line.strip()
                        if line:
                            first_line = line
                            break
                if first_line:
                    first_obj = json.loads(first_line)
                    current_step = int(first_obj.get('step', 0))
            except Exception:
                current_step = 0
        next_step = int(current_step) + 1

        raw_output = Path(args.output)
        # 自动在输出文件名中加入 step 索引；若传入目录则构造默认文件名
        if raw_output.suffix:
            output_path = raw_output.with_name(f"{raw_output.stem}_step_{next_step}{raw_output.suffix}")
            ensure_dir(str(output_path.parent))
        else:
            output_dir = ensure_dir(str(raw_output))
            output_path = output_dir / f"extend_kqa_step_{next_step}.jsonl"

        # 并发度：优先命令行，其次配置中的 concurrency.max_workers
        try:
            from utils import load_config as _load_cfg_for_conc
            _cfg_for_conc = _load_cfg_for_conc(args.config)
            default_workers = int((_cfg_for_conc.get('concurrency') or {}).get('max_workers', 1))
        except Exception:
            default_workers = 1
        concurrency = int(getattr(args, 'concurrency', None) or default_workers)

        # 打印有效运行参数摘要（方便在终端快速确认配置生效情况）
        try:
            gen_summary = _summarize_generator_config(generator_config)
            logging.getLogger('agenqa.extend').info(
                "Effective params | prompt=%s kqa=%s out=%s step=%s->%s concurrency=%d generator=%s",
                str(prompt_path), str(args.kqa), str(output_path), current_step, next_step, concurrency,
                json.dumps(gen_summary, ensure_ascii=False),
            )
        except Exception:
            pass

        runner.run(Path(args.kqa), output_path, append=False, step_override=int(current_step), concurrency=concurrency)
        logger.info("Extend-upgrade 完成，step=%s -> %s，输出: %s", current_step, next_step, output_path)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Extend-upgrade 运行失败: {e}")
        sys.exit(1)

def cmd_solve(args: argparse.Namespace) -> None:
    """对 KQA 记录执行求解（使用 Known + Question，不可见 GT）。"""
    config = load_config(args.config)
    logger = setup_logger('agenqa.solve', config.get('logging', {}), stage='solve')

    # 解析 llm_service 服务（统一用 helper）
    service_id = args.service_id
    service_config_path = (
        Path(args.service_config)
        if getattr(args, 'service_config', None)
        else DEFAULT_SERVICE_CONFIG
    )
    try:
        generator: Dict[str, Any] = load_llm_service_full_config(
            service_config_path,
            service_id,
            explicit_model=getattr(args, 'service_model', None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"加载 llm_service 配置失败: {exc}")
        sys.exit(1)

    prompt_path = Path(args.prompt) if args.prompt else REPO_ROOT / 'prompts' / 'solver.md'

    try:
        runner = SolverRunner(SolverConfig(
            prompt_path=prompt_path,
            generator=generator,
        ))

        output_path = Path(args.output)
        if not output_path.suffix:
            # 给目录时构造默认文件名
            out_dir = ensure_dir(str(output_path))
            output_path = out_dir / 'kqa_solve.jsonl'

        # 并发度：优先命令行，其次配置中的 concurrency.max_workers
        try:
            default_workers = int((config.get('concurrency') or {}).get('max_workers', 1))
        except Exception:
            default_workers = 1
        concurrency = int(getattr(args, 'concurrency', None) or default_workers)

        # 打印有效运行参数摘要（方便在终端快速确认配置生效情况）
        try:
            gen_summary = _summarize_generator_config(generator_config)
            logger.info(
                "Effective params | prompt=%s kqa=%s out=%s step=%s->%s concurrency=%d generator=%s",
                str(prompt_path), str(args.kqa), str(output_path), current_step, next_step, concurrency,
                json.dumps(gen_summary, ensure_ascii=False),
            )
        except Exception:
            pass

        runner.run(Path(args.kqa), output_path, append=False, concurrency=concurrency)
        logger.info("KQA 求解完成，输出: %s", output_path)
    except Exception as e:  # noqa: BLE001
        logger.error(f"KQA 求解失败: {e}")
        sys.exit(1)


def cmd_head_tail(args: argparse.Namespace) -> None:
    """从多步链生成 head–tail KQA（k0,qN,aN）。"""
    logger = setup_logger('agenqa.head_tail', {}, stage='head_tail')

    try:
        composer = HeadTailComposer(HeadTailConfig())

        output_path = Path(args.output)
        # 允许传目录；若为目录自动生成默认文件名
        if not output_path.suffix:
            output_dir = ensure_dir(str(output_path))
            output_path = Path(output_dir) / 'head_tail_kqa.jsonl'

        run_dir = Path(args.dir) if getattr(args, 'dir', None) else None
        head_kqa = Path(args.head_kqa) if getattr(args, 'head_kqa', None) else None
        tail_kqa = Path(args.tail_kqa) if getattr(args, 'tail_kqa', None) else None
        tail_step = args.tail_step if getattr(args, 'tail_step', None) is not None else None
        head_step = args.head_step if getattr(args, 'head_step', None) is not None else 0

        out = composer.run(
            output=output_path,
            run_dir=run_dir,
            head_kqa=head_kqa,
            tail_kqa=tail_kqa,
            head_step=head_step,
            tail_step=tail_step,
            append=False,
        )
        logger.info("Head–Tail 组合完成，输出: %s", out)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Head–Tail 组合失败: {e}")
        sys.exit(1)

def cmd_agent_run(args: argparse.Namespace) -> None:
    """基于 LangGraph 的多智能体编排（实验性）。"""
    # 命令行 env 覆盖需在 load_config 之前生效（兼容 YAML 中的 ${VAR} 占位符）。
    _apply_env_unsets(getattr(args, "unset_env", None), source="--unset-env")
    _apply_env_assignments(getattr(args, "set_env", None), override=True, source="--set-env")
    idealab_session_id = getattr(args, "idealab_session_id", None)
    if isinstance(idealab_session_id, str) and idealab_session_id.strip():
        os.environ["IDEALAB_SESSION_ID"] = idealab_session_id.strip()

    # 可选：续跑前克隆 run 目录到新目录（避免污染原目录，便于对比试验结果）。
    resume_run_dir = getattr(args, "resume_run_dir", None)
    if resume_run_dir and getattr(args, "resume_clone", False):
        orig = Path(resume_run_dir)
        if not orig.is_absolute():
            orig = REPO_ROOT / orig
        new_run_dir = _clone_resume_run_dir(orig, suffix=getattr(args, "resume_suffix", None) or "resume")
        print(f"[RESUME] Cloning run dir {orig} -> {new_run_dir}")
        shutil.copytree(orig, new_run_dir)
        args.resume_run_dir = str(new_run_dir)
        resume_run_dir = args.resume_run_dir

    # 加载配置
    config = load_config(args.config)
    logger = setup_logger('agenqa.agent', config.get('logging', {}), stage='agent')

    # 配置内 runtime_env（默认仅在 env 未设置时生效；可用于固化常用开关）。
    _apply_runtime_env_from_config(config, override=False, logger=logger)

    # CLI 快捷选择输入源类型（paper vs scipedia），并自动调整 loading 行为
    input_kind = str(getattr(args, "input_kind", "") or "").strip().lower()
    if input_kind in {"scipedia", "sci"}:
        init_conf = config.setdefault("init", {})
        src_conf = init_conf.setdefault("source", {})
        src_conf.setdefault("type", "paper")
        src_conf["path"] = os.path.expandvars(os.environ.get(
            "SCICLONE_SCIPEDIA_PATH",
            "experiments/upstream/generation/_template/inputs/scipedia/scipedia_pack_20.s008.principle_of_minimal_sensitivity_pms.jsonl",
        ))
        sp = src_conf.get("scipedia_pack")
        if not isinstance(sp, dict):
            sp = {}
        sp["enable"] = True
        sp.setdefault("include_sections", ["Key Takeaways", "Principles and Mechanisms", "Introduction"])
        sp.setdefault("strip_wiki_tokens", True)
        sp.setdefault("normalize_whitespace", True)
        sp.setdefault("prepend_title", True)
        src_conf["scipedia_pack"] = sp

        logger.info("命令行输入源: scipedia（init.source.path=%s; scipedia_pack.enable=true）", src_conf["path"])

    # CLI 快捷覆盖输入数据源（优先级高于配置与环境变量占位符）
    paper_path = getattr(args, "paper_path", None)
    if isinstance(paper_path, str) and paper_path.strip():
        init_conf = config.setdefault("init", {})
        src_conf = init_conf.setdefault("source", {})
        src_conf.setdefault("type", "paper")
        src_conf["path"] = paper_path.strip()
        logger.info("命令行覆盖: init.source.path=%s", paper_path.strip())

    # CLI 覆盖 EpisodeSeed contract 路径
    contract_path = getattr(args, "episode_seed_contract", None)
    if isinstance(contract_path, str) and contract_path.strip():
        init_conf = config.setdefault("init", {})
        es_conf = init_conf.get("episode_seed")
        if not isinstance(es_conf, dict):
            es_conf = {}
        es_conf["contract_path"] = contract_path.strip()
        init_conf["episode_seed"] = es_conf
        logger.info("命令行覆盖: init.episode_seed.contract_path=%s", es_conf["contract_path"])

    # CLI 覆盖 client.stream：对所有 generator 置 true（高于配置与 services.json）
    if getattr(args, "client_stream", False):
        _force_client_stream(config)
        logger.info("命令行覆盖: client.stream=true（已应用到所有生成器）")

    # Resolve meta/source for standalone `agent-run`.
    meta = getattr(args, "agent_meta", None)
    if not isinstance(meta, dict):
        src = getattr(args, "source", None)
        if isinstance(src, str) and src.strip():
            meta = {"source": src.strip(), "entrypoint": "agent-run"}

    # Agent（Idealab 源）主模型覆盖：允许通过 --main-model 一键切换 Idealab 主模型
    main_model = getattr(args, "main_model", None)
    if isinstance(meta, dict) and meta.get("source") == "idealab" and isinstance(main_model, str) and main_model.strip():
        resolved = _resolve_idealab_main_model_alias(main_model)
        if resolved != main_model:
            logger.info("Idealab 主模型别名解析: %s -> %s", main_model, resolved)
        _override_idealab_main_model(config, resolved.strip(), logger=logger)

    # medium 模型覆盖：允许通过 --medium-model 单独切换 solvers.medium
    medium_model = getattr(args, "medium_model", None)
    if isinstance(medium_model, str) and medium_model.strip():
        resolved = medium_model.strip()
        if isinstance(meta, dict) and meta.get("source") == "idealab":
            resolved_alias = _resolve_idealab_main_model_alias(resolved)
            if resolved_alias != resolved:
                logger.info("Idealab medium 模型别名解析: %s -> %s", resolved, resolved_alias)
            resolved = resolved_alias
        _override_medium_model(config, resolved.strip(), logger=logger)

    # 结构化 roles family 覆盖：允许通过 --struct-model 统一切换 Diagnose/Format/StepCertBuilder 等
    struct_model = getattr(args, "struct_model", None)
    if isinstance(struct_model, str) and struct_model.strip():
        resolved = struct_model.strip()
        if isinstance(meta, dict) and meta.get("source") == "idealab":
            resolved_alias = _resolve_idealab_main_model_alias(resolved)
            if resolved_alias != resolved:
                logger.info("Idealab struct 模型别名解析: %s -> %s", resolved, resolved_alias)
            resolved = resolved_alias
        _override_struct_model(config, resolved.strip(), logger=logger)

    # Format 模型覆盖：允许通过 --format-model 单独切换 Format 角色
    format_model = getattr(args, "format_model", None)
    if isinstance(format_model, str) and format_model.strip():
        resolved = format_model.strip()
        if isinstance(meta, dict) and meta.get("source") == "idealab":
            resolved_alias = _resolve_idealab_main_model_alias(resolved)
            if resolved_alias != resolved:
                logger.info("Idealab format 模型别名解析: %s -> %s", resolved, resolved_alias)
            resolved = resolved_alias
        _override_format_model(config, resolved.strip(), logger=logger)

    # 多 strong 快捷覆盖：--strong-models "7,6,4"（按给定顺序写入 strong 列表）
    strong_models_raw = getattr(args, "strong_models", None)
    if isinstance(strong_models_raw, str) and strong_models_raw.strip():
        tokens = _parse_model_list(strong_models_raw)
        if isinstance(meta, dict) and meta.get("source") == "idealab":
            resolved_models = [_resolve_idealab_main_model_alias(tok) for tok in tokens]
            # 若用户未显式指定 --main-model，则默认将首个 strong 模型同步为主模型（用于 pipeline 其他角色）。
            if not (isinstance(main_model, str) and main_model.strip()) and resolved_models:
                inferred_main = resolved_models[0].strip()
                logger.info("未指定 --main-model，自动使用首个 strong 模型作为主模型: %s", inferred_main)
                _override_idealab_main_model(config, inferred_main, logger=logger)
        else:
            resolved_models = tokens
        _override_strong_models(config, resolved_models, logger=logger)

    # CLI 覆盖 multi-strong 共识触发策略：consensus.mode = none | always
    consensus_mode = getattr(args, "consensus_mode", None)
    if isinstance(consensus_mode, str) and consensus_mode.strip():
        mode_norm = consensus_mode.strip().lower()
        if mode_norm == "always":
            config.setdefault("consensus", {})["mode"] = "always"
            logger.info("命令行覆盖: consensus.mode=always")
        elif mode_norm in {"none", "disabled", "off"}:
            config.setdefault("consensus", {})["mode"] = "none"
            logger.info("命令行覆盖: consensus.mode=none")
    elif getattr(args, "all_strong", False):
        config.setdefault("consensus", {})["mode"] = "always"
        logger.info("命令行覆盖: consensus.mode=always（--all-strong）")

    # 输出目录与 run_id（使用北京时间）；支持 resume
    resume_run_dir = getattr(args, "resume_run_dir", None)
    if not resume_run_dir:
        _preflight_validate_agent_papers_path(config, logger)
    loaded_state = None
    resume_archive_run_dir: Optional[Path] = None
    if resume_run_dir:
        run_dir = Path(resume_run_dir)
        state_path = run_dir / "state.json"
        if not state_path.exists():
            logger.error("resume_run_dir=%s 缺少 state.json，无法续跑", str(run_dir))
            sys.exit(2)
        if not getattr(args, "resume_clone", False) and not getattr(args, "resume_prepared", False):
            agent_meta = getattr(args, "agent_meta", None)
            batch_id = agent_meta.get("batch_id") if isinstance(agent_meta, dict) else ""
            archive_info = _prepare_run_dir_for_in_place_resume(
                run_dir,
                resume_reason="agent_resume",
                batch_id=str(batch_id or ""),
            )
            resume_archive_run_dir = Path(archive_info["run_dir"])
        from agenqa.graph import AgentState
        loaded_state = AgentState.load_from_file(state_path)
        # 续跑场景默认清除 stop_reason，允许在修复后继续往下跑
        loaded_state.stop_reason = None
        # 优先使用已保存的 run_id / artifacts_dir
        run_id = loaded_state.run_id
        output_dir = run_dir
        base_output = run_dir.parent
        logger.info("Resume 模式：从 %s 续跑，run_id=%s", str(run_dir), run_id)
    else:
        base_output = Path(args.output or (config.get('agent', {}) or {}).get('output_dir') or 'data/output/agent_run')
        base_output.mkdir(parents=True, exist_ok=True)
        bj_tz = timezone(timedelta(hours=8))
        run_id = args.run_id or _new_run_id()
        # 每次运行使用带时间戳的子目录，便于多轮产物分离
        output_dir = base_output / f"run_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

    # 在 run 目录中保存一份终端日志
    run_log_path = output_dir / "run.log"
    run_log_handler = logging.FileHandler(str(run_log_path), encoding="utf-8")
    run_log_handler.setLevel(logging.DEBUG)
    run_log_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    # 添加到 root logger 以捕获所有日志
    logging.getLogger().addHandler(run_log_handler)
    logger.info("运行日志将保存到: %s", str(run_log_path))

    # 覆盖最大步数
    if args.max_steps is not None:
        config.setdefault('agent', {})['max_steps'] = int(args.max_steps)
        # 若未显式配置 max_rounds，则默认与 max_steps 保持一致
        config.setdefault('agent', {}).setdefault('max_rounds', int(args.max_steps))

    # 覆盖最大轮数（若显式指定，则覆盖上面的默认值）
    if getattr(args, 'max_rounds', None) is not None:
        config.setdefault('agent', {})['max_rounds'] = int(args.max_rounds)

    # 覆盖连续 Revise 上限（fail-fast stop-policy）
    max_consecutive_revise = getattr(args, "max_consecutive_revise", None)
    if max_consecutive_revise is None:
        max_consecutive_revise = getattr(args, "revise_retry_limit", None)
    if max_consecutive_revise is not None:
        config.setdefault("agent", {})["max_consecutive_revise"] = int(max_consecutive_revise)
        logger.info("命令行覆盖: agent.max_consecutive_revise=%s", str(max_consecutive_revise))

    # 覆盖语言（用于下游 Prompt 语言控制）
    lang = getattr(args, 'lang', None)
    if lang:
        config.setdefault('agent', {})['lang'] = lang

    # 覆盖符号-only 模式（用于出题与判分）
    if getattr(args, 'symbolic_only', False):
        config.setdefault('agent', {})['symbolic_only'] = True

    # 覆盖题型策略：从指定 step 开始禁用 MCQ（默认策略在 director.py 里；这里允许用命令行快速调参）
    if getattr(args, "no_mcq_from_step", None) is not None:
        v = _parse_no_mcq_from_step(getattr(args, "no_mcq_from_step", None))
        agent_block = config.setdefault("agent", {})
        policy = agent_block.get("question_type_policy")
        if not isinstance(policy, dict):
            policy = {}
        policy = dict(policy)
        policy["no_mcq_from_step"] = v
        agent_block["question_type_policy"] = policy
        logger.info("命令行覆盖: agent.question_type_policy.no_mcq_from_step=%s", str(v))

    # 覆盖题型白名单：allowed_question_types（对 base_allowed 做交集过滤；交集为空应 fail-fast）
    if getattr(args, "allowed_question_types", None) is not None:
        _override_allowed_question_types(config, getattr(args, "allowed_question_types", None), logger=logger)

    # 覆盖 roles 输出协议（json / tagged）
    roles_protocol = getattr(args, 'roles_protocol', None)
    # 兼容历史参数名 --draft-protocol
    if roles_protocol is None:
        roles_protocol = getattr(args, 'draft_protocol', None)
    if roles_protocol:
        agent_block = config.setdefault('agent', {})
        agent_block['roles_protocol'] = str(roles_protocol)
        # 兼容旧字段名，避免老 run_config 读取失败
        agent_block.setdefault('draft_protocol', str(roles_protocol))
    format_validation_mode = getattr(args, "format_validation_mode", None)
    if format_validation_mode:
        config.setdefault("agent", {})["format_validation_mode"] = str(format_validation_mode)

    # 引擎目前固定为 langgraph
    engine = getattr(args, 'engine', None) or 'langgraph'
    if engine != 'langgraph':
        logger.error('暂不支持引擎: %s', engine)
        sys.exit(2)

    # 将本次运行的参数与配置快照保存到 run 目录，便于后续回溯
    extra_meta = getattr(args, "agent_meta", None)
    if not isinstance(extra_meta, dict):
        extra_meta = None
    _save_run_config(output_dir, args, config, extra=extra_meta)

    try:
        final_state = run_episode(config, run_id=run_id, output_dir=output_dir, state=loaded_state)
        logger.info('Agent 完成: step=%s stop_reason=%s', str(final_state.step), str(final_state.stop_reason))
        if not getattr(args, "no_playback", False):
            try:
                from infra.playback.run_playback_md import generate_playback_md
                out_path = generate_playback_md(Path(output_dir))
                logger.info("已生成 run playback 回放文档: %s", str(out_path))
            except Exception:
                # playback 是后处理产物，失败不应影响主链路退出码，但必须可见
                logger.exception("run playback 回放文档生成失败（不影响 run 完成）")
        if resume_archive_run_dir is not None:
            _finalize_run_resume_archive(resume_archive_run_dir, success=True, error="")
    except Exception:
        # 确保未捕获异常也写入 run.log，便于排查（同时仍向上抛出保持非 0 退出码）
        logger.exception("Agent 运行过程中发生未捕获异常")
        if resume_archive_run_dir is not None:
            _finalize_run_resume_archive(
                resume_archive_run_dir,
                success=False,
                error="Agent 运行过程中发生未捕获异常",
            )
        raise
    finally:
        # 移除 run 日志 handler 并关闭文件
        logging.getLogger().removeHandler(run_log_handler)
        run_log_handler.close()


def cmd_agent_batch_run(args: argparse.Namespace) -> None:
    """Run multiple papers concurrently by launching isolated agent-run workers."""
    resume_batch_dir = getattr(args, "resume_batch_dir", None)
    if resume_batch_dir and (getattr(args, "paper_list", None) or getattr(args, "paper_dir", None)):
        raise SystemExit("[ERROR] --resume-batch-dir cannot be used with --paper-list/--paper-dir")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(str(config_path))

    jobs: list[Dict[str, Any]] = []
    seed_rows: list[Dict[str, Any]] = []
    batch_resume_attempt_dir: Optional[Path] = None

    if resume_batch_dir:
        source_batch_dir = Path(str(resume_batch_dir))
        if not source_batch_dir.is_absolute():
            source_batch_dir = REPO_ROOT / source_batch_dir
        if not source_batch_dir.exists():
            raise SystemExit(f"[ERROR] resume batch dir does not exist: {source_batch_dir}")
        root_batch_dir = _resolve_lineage_root(source_batch_dir)
        source_manifest_path = root_batch_dir / "batch_manifest.json"
        if not source_manifest_path.exists():
            raise SystemExit(f"[ERROR] missing batch_manifest.json under resume batch dir: {root_batch_dir}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_papers = [str(p) for p in source_manifest.get("papers") or []]
        batch_dir = root_batch_dir
        batch_id = str(source_manifest.get("batch_id") or batch_dir.name)
        runs_root = batch_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        snapshot = _build_heartbeat_snapshot(
            root_batch_dir=root_batch_dir,
            max_auto_resumes_per_task=10_000,
        )
        latest_rows = _collect_latest_lineage_rows(root_batch_dir)

        batch_archive_root = batch_dir / "_resume_archive"
        batch_attempt_dir, batch_attempt_index = _next_archive_attempt_dir(batch_archive_root)
        batch_attempt_dir.mkdir(parents=True, exist_ok=True)
        batch_resume_attempt_dir = batch_attempt_dir
        archived_batch_files: list[str] = []
        for name in ["batch_results.jsonl", "batch_summary.md", "live_status.json", "live_events.jsonl", "monitor_snapshot.json", "monitor_summary.md"]:
            src = batch_dir / name
            if not src.exists():
                continue
            _copy_path(src, batch_attempt_dir / name)
            archived_batch_files.append(name)

        resume_manifest: Dict[str, Any] = {
            "attempt_index": batch_attempt_index,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "resume_reason": "batch_in_place_resume",
            "source_batch_dir": str(source_batch_dir),
            "root_batch_dir": str(root_batch_dir),
            "root_batch_id": batch_id,
            "archived_batch_files": archived_batch_files,
            "tasks": [],
        }

        source_value = getattr(args, "source", None) or source_manifest.get("source")
        lang_value = getattr(args, "lang", None) or source_manifest.get("lang")
        max_steps_value = getattr(args, "max_steps", None) if getattr(args, "max_steps", None) is not None else source_manifest.get("max_steps")
        max_rounds_value = getattr(args, "max_rounds", None) if getattr(args, "max_rounds", None) is not None else source_manifest.get("max_rounds")
        max_consecutive_revise_value = (
            getattr(args, "max_consecutive_revise", None)
            if getattr(args, "max_consecutive_revise", None) is not None
            else source_manifest.get("max_consecutive_revise")
        )

        root_batch_manifest = dict(source_manifest)
        root_batch_manifest["last_resumed_at"] = datetime.now(timezone.utc).isoformat()
        root_batch_manifest["resume_attempt_count"] = int(source_manifest.get("resume_attempt_count") or 0) + 1
        root_batch_manifest["last_resume_source_batch_dir"] = str(source_batch_dir)
        root_batch_manifest["last_resume_source_batch_id"] = str(snapshot.get("lineage_batch_ids")[-1] if snapshot.get("lineage_batch_ids") else batch_id)
        root_batch_manifest["source"] = source_value
        root_batch_manifest["lang"] = lang_value
        root_batch_manifest["max_steps"] = max_steps_value
        root_batch_manifest["max_rounds"] = max_rounds_value
        root_batch_manifest["max_consecutive_revise"] = max_consecutive_revise_value
        (batch_dir / "batch_manifest.json").write_text(json.dumps(root_batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        for idx, paper_path in enumerate(source_papers, start=1):
            task_id = f"task_{idx:03d}"
            task_snapshot = (snapshot.get("tasks") or {}).get(task_id) or {}
            latest_run_dir_raw = str(task_snapshot.get("latest_run_dir") or "")
            latest_run_dir = Path(latest_run_dir_raw).resolve() if latest_run_dir_raw else None

            target_run_dir = _find_primary_run_dir_for_task(batch_dir, task_id)
            if latest_run_dir is not None:
                if target_run_dir is None:
                    target_run_dir = runs_root / latest_run_dir.name
                materialize_info = _materialize_run_dir(
                    source_run_dir=latest_run_dir,
                    target_run_dir=target_run_dir,
                    archive_dir=batch_attempt_dir,
                )
            else:
                materialize_info = {
                    "materialized": False,
                    "source_run_dir": "",
                    "target_run_dir": str(target_run_dir) if target_run_dir else "",
                    "archived_previous_target": "",
                }

            latest_row = latest_rows.get(task_id)
            status = str((latest_row or {}).get("status") or task_snapshot.get("status") or "")
            if latest_row is not None and status == "success":
                seed_row = dict(latest_row)
                if target_run_dir is not None:
                    seed_row["run_dir"] = str(target_run_dir)
                seed_rows.append(seed_row)
            else:
                if target_run_dir is None or not target_run_dir.exists():
                    raise SystemExit(f"[ERROR] no resume source run dir for {task_id} under {batch_dir}")
                run_archive_info = _prepare_run_dir_for_in_place_resume(
                    target_run_dir,
                    resume_reason="batch_resume",
                    batch_id=batch_id,
                    batch_archive_attempt_dir=batch_attempt_dir,
                )
                jobs.append(
                    {
                        "task_id": task_id,
                        "batch_id": batch_id,
                        "config_path": str(config_path),
                        "paper_path": paper_path,
                        "run_id": _new_run_id(getattr(args, "run_prefix", None) or f"paper{idx:03d}"),
                        "run_dir": str(target_run_dir),
                        "output_root": str(runs_root),
                        "source": source_value,
                        "lang": lang_value,
                        "max_steps": max_steps_value,
                        "max_rounds": max_rounds_value,
                        "max_consecutive_revise": max_consecutive_revise_value,
                        "no_playback": getattr(args, "no_playback", False),
                        "resume_run_dir": str(target_run_dir),
                        "resume_clone": False,
                        "resume_prepared": True,
                        "resume_suffix": getattr(args, "resume_suffix", None) or "resume",
                    }
                )
                resume_manifest["tasks"].append(
                    {
                        "task_id": task_id,
                        "paper_path": paper_path,
                        "previous_status": status or "unknown",
                        "latest_source_run_dir": str(latest_run_dir) if latest_run_dir else "",
                        "materialize": materialize_info,
                        "run_archive_dir": str(run_archive_info["archive_dir"]),
                    }
                )

        (batch_attempt_dir / "resume_manifest.json").write_text(json.dumps(resume_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _append_resume_history_line(
            batch_dir,
            {
                "timestamp": resume_manifest["archived_at"],
                "event_type": "batch_resume_started",
                "attempt_index": batch_attempt_index,
                "root_batch_dir": str(batch_dir),
                "source_batch_dir": str(source_batch_dir),
                "resumed_task_ids": [item["task_id"] for item in resume_manifest["tasks"]],
                "inherited_successes": len(seed_rows),
            },
        )

        manifest = root_batch_manifest
    else:
        papers = _resolve_batch_papers(args, config=config)
        batch_id = getattr(args, "batch_id", None) or f"batch_{_new_run_id(getattr(args, 'run_prefix', None) or 'batch')}"
        base_output = Path(args.output or "data/output/agent_batch_run")
        batch_dir = base_output / batch_id
        runs_root = batch_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        manifest = {
            "batch_id": batch_id,
            "config_path": str(config_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "concurrency": int(args.concurrency),
            "paper_count": len(papers),
            "papers": [str(p) for p in papers],
            "source": getattr(args, "source", None),
            "lang": getattr(args, "lang", None),
            "max_steps": getattr(args, "max_steps", None),
            "max_rounds": getattr(args, "max_rounds", None),
            "max_consecutive_revise": getattr(args, "max_consecutive_revise", None),
        }
        for idx, paper_path in enumerate(papers, start=1):
            run_id = _new_run_id(getattr(args, "run_prefix", None) or f"paper{idx:03d}")
            jobs.append(
                {
                    "task_id": f"task_{idx:03d}",
                    "batch_id": batch_id,
                    "config_path": str(config_path),
                    "paper_path": str(paper_path),
                    "run_id": run_id,
                    "run_dir": str(runs_root / f"run_{run_id}"),
                    "output_root": str(runs_root),
                    "source": getattr(args, "source", None),
                    "lang": getattr(args, "lang", None),
                    "max_steps": getattr(args, "max_steps", None),
                    "max_rounds": getattr(args, "max_rounds", None),
                    "max_consecutive_revise": getattr(args, "max_consecutive_revise", None),
                    "no_playback": getattr(args, "no_playback", False),
                }
            )

    (batch_dir / "batch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = _run_batch_jobs(
        batch_id=batch_id,
        batch_dir=batch_dir,
        jobs=jobs,
        seed_rows=seed_rows,
        concurrency=int(args.concurrency),
        continue_on_error=bool(getattr(args, "continue_on_error", False)),
        inline_status=bool(getattr(args, "inline_status", False)),
        status_interval_sec=int(getattr(args, "status_interval_sec", 30) or 30),
        status_active_limit=int(getattr(args, "status_active_limit", 8) or 8),
        status_top_errors=int(getattr(args, "status_top_errors", 5) or 5),
    )
    if batch_resume_attempt_dir is not None:
        resume_manifest_path = batch_resume_attempt_dir / "resume_manifest.json"
        if resume_manifest_path.exists():
            resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        else:
            resume_manifest = {}
        latest_rows = {str(row.get("task_id") or ""): row for row in rows if row.get("task_id")}
        finalized_tasks: list[Dict[str, Any]] = []
        for item in resume_manifest.get("tasks") or []:
            task_id = str(item.get("task_id") or "")
            row = latest_rows.get(task_id, {})
            run_dir = Path(str(row.get("run_dir") or item.get("materialize", {}).get("target_run_dir") or ""))
            brief = _read_state_brief(run_dir) if run_dir.exists() else {"step": 0, "history_len": 0, "stop_reason": None}
            finalized_tasks.append(
                {
                    **item,
                    "final_status": row.get("status") or item.get("previous_status") or "unknown",
                    "final_error": row.get("error") or "",
                    "final_run_dir": str(run_dir) if run_dir else "",
                    "final_step": brief.get("step"),
                    "final_history_len": brief.get("history_len"),
                    "final_stop_reason": brief.get("stop_reason"),
                }
            )
        resume_manifest["finalized_at"] = datetime.now(timezone.utc).isoformat()
        resume_manifest["finalized_tasks"] = finalized_tasks
        resume_manifest_path.write_text(json.dumps(resume_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _append_resume_history_line(
            batch_dir,
            {
                "timestamp": resume_manifest["finalized_at"],
                "event_type": "batch_resume_finalized",
                "attempt_index": resume_manifest.get("attempt_index"),
                "root_batch_dir": str(batch_dir),
                "finalized_tasks": [
                    {
                        "task_id": item.get("task_id"),
                        "final_status": item.get("final_status"),
                        "final_stop_reason": item.get("final_stop_reason"),
                    }
                    for item in finalized_tasks
                ],
            },
        )


def cmd_agent_dry_run(args: argparse.Namespace) -> None:
    """进行一次不依赖 LLM 的干跑，验证产物结构与状态递增语义。"""
    config = load_config(args.config)
    from agenqa.graph import AgentState, KQARecord
    from agenqa.memory.store import save_state
    import json as _json

    output_dir = Path(args.output or 'data/output/agent_run_dryrun')
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"dryrun-{_new_run_id()}"

    state = AgentState(run_id=run_id, artifacts_dir=output_dir, max_steps=int(config.get('agent', {}).get('max_steps', 3)))
    save_state(state)

    # 模拟 Extend QA‑Init（step=0）
    extend_dir = output_dir / 'extend'
    extend_dir.mkdir(parents=True, exist_ok=True)
    qa_init_kqa = {
        'paper_id': 'paper_dry_001',
        'step': 0,
        'known': _json.dumps({'known_0': 'base known for dry-run', 'history': []}, ensure_ascii=False),
        'question': 'Q0: define baseline quantities.',
        'answer': '\\boxed{baseline}',
        'chain': 'k0,q0,a0',
    }
    (extend_dir / 'qa_init_raw_step_0_kqa.jsonl').write_text(_json.dumps(qa_init_kqa, ensure_ascii=False) + '\n', encoding='utf-8')
    state.append_history(KQARecord(
        paper_id=qa_init_kqa['paper_id'],
        step=qa_init_kqa['step'],
        known=str(qa_init_kqa['known']),
        question=qa_init_kqa['question'],
        answer=qa_init_kqa['answer'],
        chain=qa_init_kqa['chain'],
    ))
    save_state(state)

    # 求解器占位产物
    solve_dir = output_dir / "solve"
    solve_dir.mkdir(parents=True, exist_ok=True)
    (solve_dir / 'solve_medium.jsonl').write_text(_json.dumps({'correct': False, 'token_ratio': None}, ensure_ascii=False) + '\n', encoding='utf-8')
    (solve_dir / 'solve_strong_0.jsonl').write_text(_json.dumps({'correct': False, 'token_ratio': None}, ensure_ascii=False) + '\n', encoding='utf-8')

    print(state.to_json())


def cmd_agent(args: argparse.Namespace) -> None:
    """Agent 便捷入口：通过 SJTU Relay / Idealab / AIMux OpenAI 兼容端点运行 agent-run 或 agent-dry-run。"""
    repo_root = REPO_ROOT

    _apply_env_unsets(getattr(args, "unset_env", None), source="--unset-env")
    _apply_env_assignments(getattr(args, "set_env", None), override=True, source="--set-env")
    idealab_session_id = getattr(args, "idealab_session_id", None)
    if isinstance(idealab_session_id, str) and idealab_session_id.strip():
        os.environ["IDEALAB_SESSION_ID"] = idealab_session_id.strip()

    config_path = args.config
    if config_path == "config/default.yaml":
        lang_norm = (getattr(args, "lang", None) or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        if args.source == "idealab":
            config_path = IDEALAB_DEFAULT_CONFIG_EN if use_en else IDEALAB_DEFAULT_CONFIG
        elif args.source == "sjtu":
            config_path = SJTU_DEFAULT_CONFIG_EN if use_en else SJTU_DEFAULT_CONFIG

    resume_run_dir = getattr(args, "resume_run_dir", None)
    if resume_run_dir and getattr(args, "resume_clone", False):
        orig = Path(resume_run_dir)
        if not orig.is_absolute():
            orig = repo_root / orig
        new_run_dir = _clone_resume_run_dir(orig, suffix=getattr(args, "resume_suffix", None) or "r")
        print(f"[RESUME] Cloning run dir {orig} -> {new_run_dir}")
        shutil.copytree(orig, new_run_dir)
        resume_run_dir = str(new_run_dir)

    # 代理环境变量在部分环境下是“必需”的（否则会出现 Idealab 连接失败/SSL Errno 22）。
    # 如需强制直连，可设置：SCICLONE_CLEAR_PROXY=1（参见 infra/llm/service_client.py）。
    local_no_proxy = ["127.0.0.1", "localhost", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        existing = (os.environ.get(key) or "").strip()
        items = [x.strip() for x in existing.split(",") if x.strip()] if existing else []
        for host in local_no_proxy:
            if host not in items:
                items.append(host)
        if items:
            os.environ[key] = ",".join(items)

    if args.source == "sjtu":
        if args.base_url:
            os.environ["AGENT_BASE_URL"] = args.base_url
        else:
            os.environ.setdefault("AGENT_BASE_URL", SJTU_DEFAULT_BASE_URL)
        if args.api_key:
            os.environ["AGENT_API_KEY"] = args.api_key
        if not args.dry_run and not os.environ.get("AGENT_API_KEY"):
            raise SystemExit(
                "[ERROR] Missing AGENT_API_KEY for SJTU Relay. "
                "Use --api-key or export the env var."
            )
    elif args.source == "idealab":
        if args.base_url:
            os.environ["IDEALAB_BASE_URL"] = args.base_url
        else:
            os.environ.setdefault("IDEALAB_BASE_URL", IDEALAB_DEFAULT_BASE_URL)
        if args.api_key:
            os.environ["IDEALAB_API_KEY"] = args.api_key
        # 兼容：部分配置文件使用 AGENT_* 环境变量占位符（如 agent_paper_path_v2.yaml）。
        # 在 Idealab 模式下同步写入 AGENT_*，避免用户需要重复设置两套变量。
        os.environ.setdefault("AGENT_BASE_URL", os.environ.get("IDEALAB_BASE_URL", ""))
        if os.environ.get("IDEALAB_API_KEY"):
            os.environ.setdefault("AGENT_API_KEY", os.environ["IDEALAB_API_KEY"])
        if not args.dry_run and not os.environ.get("IDEALAB_API_KEY"):
            raise SystemExit(
                "[ERROR] Missing IDEALAB_API_KEY for Idealab. "
                "Use --api-key or export the env var."
            )
    elif args.source == "aimux":
        if args.base_url:
            os.environ["AIMUX_BASE_URL"] = args.base_url
        else:
            os.environ.setdefault("AIMUX_BASE_URL", AIMUX_DEFAULT_BASE_URL)
        if args.api_key:
            os.environ["AIMUX_API_KEY"] = args.api_key
        # 兼容：部分配置文件使用 AGENT_* 环境变量占位符。
        os.environ.setdefault("AGENT_BASE_URL", os.environ.get("AIMUX_BASE_URL", ""))
        if os.environ.get("AIMUX_API_KEY"):
            os.environ.setdefault("AGENT_API_KEY", os.environ["AIMUX_API_KEY"])
        if not args.dry_run and not os.environ.get("AIMUX_API_KEY"):
            raise SystemExit(
                "[ERROR] Missing AIMUX_API_KEY for AIMux. "
                "Use --api-key or export the env var."
            )
    elif args.source == "dmxapi":
        if args.base_url:
            os.environ["DMXAPI_BASE_URL"] = args.base_url
        else:
            os.environ.setdefault("DMXAPI_BASE_URL", DMXAPI_DEFAULT_BASE_URL)
        if args.api_key:
            os.environ["DMXAPI_API_KEY"] = args.api_key
        # 兼容：复用通用 agent_paper_path_v2.yaml，统一同步到 AGENT_*。
        os.environ.setdefault("AGENT_BASE_URL", os.environ.get("DMXAPI_BASE_URL", ""))
        if os.environ.get("DMXAPI_API_KEY"):
            os.environ.setdefault("AGENT_API_KEY", os.environ["DMXAPI_API_KEY"])
        if not args.dry_run and not os.environ.get("DMXAPI_API_KEY"):
            raise SystemExit(
                "[ERROR] Missing DMXAPI_API_KEY for DMXAPI. "
                "Use --api-key or export the env var."
            )

    # 处理超时覆盖：通过环境变量传递给下游配置加载
    if getattr(args, "timeout", None) is not None:
        os.environ["AGENT_CLIENT_TIMEOUT"] = str(args.timeout)
        print(f"[INFO] API timeout set to {args.timeout}s")

    # 支持在 config YAML 中固化常用 env 开关（例如 JSON cleaner / Gemini vertex 选项等）。
    # 注意：默认不覆盖已设置的 env；显式覆盖请使用 --set-env 或先 unset。
    try:
        cfg_for_env = load_config(config_path)
        _apply_runtime_env_from_config(cfg_for_env, override=False)
    except Exception:
        # env 固化属于可选能力；加载失败不影响主流程。
        pass

    if not args.dry_run and not args.skip_smoke:
        if args.source == "idealab":
            print("[SMOKE] Checking Idealab connectivity (/models)...")
            cmd = [
                sys.executable,
                str(repo_root / "infra/llm/probes/test_openai_compat_api.py"),
                "--source",
                "idealab",
                "models",
            ]
        elif args.source == "aimux":
            print("[SMOKE] Checking AIMux connectivity (/models)...")
            cmd = [
                sys.executable,
                str(repo_root / "infra/llm/probes/test_openai_compat_api.py"),
                "--source",
                "aimux",
                "models",
            ]
        elif args.source == "dmxapi":
            print("[SMOKE] Checking DMXAPI connectivity (/models)...")
            cmd = [
                sys.executable,
                str(repo_root / "infra/llm/probes/test_openai_compat_api.py"),
                "--source",
                "dmxapi",
                "models",
            ]
        else:
            print("[SMOKE] Checking SJTU Relay connectivity (/models)...")
            cmd = [
                sys.executable,
                str(repo_root / "infra/llm/probes/test_openai_compat_api.py"),
                "--source",
                "sjtu",
                "models",
            ]
        result = subprocess.run(cmd, cwd=repo_root, env=os.environ.copy(), check=False)
        if result.returncode != 0:
            print("[WARN] Smoke test failed; verify API key / VPN / base URL.")

    bj_tz = timezone(timedelta(hours=8))
    run_id = _new_run_id()

    if args.dry_run:
        dry_args = argparse.Namespace(
            config=config_path,
            output=args.output,
            run_id=run_id,
        )
        print(f"[RUN] Dry run → {Path(args.output or 'data/output/agent_run_dryrun')}")
        cmd_agent_dry_run(dry_args)
        run_path = Path(args.output or "data/output/agent_run_dryrun")
    else:
        agent_args = argparse.Namespace(
            config=config_path,
            engine="langgraph",
            output=args.output,
            run_id=run_id,
            resume_run_dir=resume_run_dir,
            input_kind=getattr(args, "input_kind", None),
            paper_path=getattr(args, "paper_path", None),
            episode_seed_contract=getattr(args, "episode_seed_contract", None),
            idealab_session_id=getattr(args, "idealab_session_id", None),
            set_env=getattr(args, "set_env", None),
            unset_env=getattr(args, "unset_env", None),
            max_steps=args.max_steps,
            max_rounds=getattr(args, 'max_rounds', None),
            max_consecutive_revise=getattr(args, 'max_consecutive_revise', None) or getattr(args, 'revise_retry_limit', None),
            lang=args.lang,
            symbolic_only=getattr(args, "symbolic_only", False),
            no_mcq_from_step=getattr(args, "no_mcq_from_step", None),
            allowed_question_types=getattr(args, "allowed_question_types", None),
            roles_protocol=getattr(args, "roles_protocol", None),
            format_validation_mode=getattr(args, "format_validation_mode", None),
            client_stream=getattr(args, "client_stream", False),
            main_model=getattr(args, "main_model", None),
            medium_model=getattr(args, "medium_model", None),
            struct_model=getattr(args, "struct_model", None),
            format_model=getattr(args, "format_model", None),
            strong_models=getattr(args, "strong_models", None),
            consensus_mode=getattr(args, "consensus_mode", None),
            all_strong=getattr(args, "all_strong", False),
            agent_meta={
                "source": args.source,
                "base_url": (
                    os.environ.get("AGENT_BASE_URL")
                    or os.environ.get("IDEALAB_BASE_URL")
                    or os.environ.get("AIMUX_BASE_URL")
                ),
                "use_py_prompts": True,
                "entrypoint": "agent",
            },
        )
        if resume_run_dir:
            run_path = Path(resume_run_dir)
        else:
            base_output = Path(args.output or "data/output/agent_run")
            run_path = base_output / f"run_{run_id}"
        max_rounds_display = getattr(args, 'max_rounds', None) or args.max_steps
        print(
            f"[RUN] Real run via API → {run_path} "
            f"(max_steps={args.max_steps}, max_rounds={max_rounds_display}, base_url={os.environ.get('AGENT_BASE_URL') or os.environ.get('IDEALAB_BASE_URL') or os.environ.get('AIMUX_BASE_URL')})"
        )
        cmd_agent_run(agent_args)

    print("[DONE] Check outputs under:", run_path)
    try:
        from infra.playback.run_playback_md import generate_playback_md
        out_path = generate_playback_md(run_path)
        try:
            summary_dir_main = run_path / "00_Summary"
            summary_dir_main.mkdir(parents=True, exist_ok=True)
            src = out_path if out_path.exists() else run_path / "run_playback.md"
            if not src.exists():
                src = next(iter(run_path.glob("run_playback*.md")), None)  # type: ignore[arg-type]
            if src and src.exists():
                shutil.copy2(src, summary_dir_main / "run_playback.md")
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to generate playback markdown automatically: {exc}")
        print("[HINT] You can generate it manually via:")
        print(f"       python scripts/gen_run_playback_md.py --run-dir {run_path}")
    else:
        print(f"[INFO] Playback markdown generated at: {run_path / 'run_playback.md'}")
        print(f"[INFO] Summary directory: {run_path / '00_Summary'}")

def main():
    """主入口函数"""
    setup_basic_logging()

    parser = argparse.ArgumentParser(
        description='AgenQA - 科学领域数据合成与模型训练框架',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 全局参数
    parser.add_argument(
        '--config', '-c',
        default='config/default.yaml',
        help='配置文件路径 (default: config/default.yaml)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # extend-upgrade 命令 - 基于已有 KQA 记录做一次 i->i+1 迭代
    ext_parser = subparsers.add_parser(
        'extend-upgrade',
        help='基于 KQA JSONL 做一次迭代（从 step i 到 i+1）'
    )
    # 两种用法：
    # 1) 直接模式：--kqa + --output [+ 可选 --current-step 指定输入当前步 i（输出为 i+1）]
    # 2) 目录模式：--qa-root + --extend-root + --target-step（本次产出步 N，输入自动取 N-1）
    ext_parser.add_argument(
        '--kqa',
        help='输入 KQA JSONL 路径（直接模式）'
    )
    ext_parser.add_argument(
        '--output', '-o',
        help='输出 JSONL 文件路径（直接模式）'
    )
    ext_parser.add_argument(
        '--qa-root',
        help='旧 init 输出目录（目录模式，step=1 时用于定位 qa_init_raw_step_0_kqa.jsonl）'
    )
    ext_parser.add_argument(
        '--extend-root',
        help='extend 输出目录（目录模式：输入/输出均在该目录内）'
    )
    ext_parser.add_argument(
        '--prompt',
        help='Prompt 模板完整路径（默认使用 src/agenqa/prompts/extend_upgrade.prompt）'
    )
    # 步骤参数（新命名，取代历史 --step）
    ext_parser.add_argument(
        '--current-step', type=int,
        help='直接模式：输入记录的当前 step（i），未提供则从记录内容推断；输出为 i+1'
    )
    ext_parser.add_argument(
        '--target-step', type=int,
        help='目录模式：产出目标 step（N），输入自动取 N-1（N>=1）'
    )
    ext_parser.add_argument(
        '--service-id',
        help='引用 Service/llm_service/configs/services.json 中的 service_id，用于覆盖推理服务'
    )
    ext_parser.add_argument(
        '--service-config',
        help='llm_service 服务配置文件路径，默认指向 Service/llm_service/configs/services.json'
    )
    ext_parser.add_argument(
        '--service-model',
        help='当 --service-id 缺少模型名称时指定使用的模型名'
    )
    ext_parser.add_argument(
        '--concurrency', type=int,
        help='并发请求数（覆盖配置 concurrency.max_workers）'
    )
    ext_parser.add_argument(
        '--client-timeout', type=int,
        help='HTTP 超时（秒），覆盖 services.json 与配置内的 client.timeout'
    )
    # 生成参数覆盖（运行时动态控制模型输出行为）
    ext_parser.add_argument('--max-tokens', type=int, help='覆盖 generation.max_tokens')
    ext_parser.add_argument('--temperature', type=float, help='覆盖 generation.temperature')
    ext_parser.add_argument('--top-p', type=float, help='覆盖 generation.top_p')
    ext_parser.add_argument('--top-k', type=int, help='覆盖 generation.top_k')
    ext_parser.add_argument('--min-p', type=float, help='覆盖 generation.min_p')
    ext_parser.add_argument(
        '--symbolic-only',
        action='store_true',
        help='启用“符号表达式 ONLY” 模式（Question/Answer 尽量不含具体数值，仅用符号表达式）'
    )
    ext_parser.set_defaults(func=cmd_extend_upgrade)

    # solve 命令 - 对 KQA 记录进行求解（只见 K/Q）
    solve_parser = subparsers.add_parser(
        'solve',
        help='基于 Known/Question 求解，输出 {known,question,answer,solve,correct,token_ratio,metrics} JSONL'
    )
    solve_parser.add_argument(
        '--kqa', required=True,
        help='输入 KQA JSON(L) 路径（来自 init step 输出或 extend-upgrade 输出）'
    )
    solve_parser.add_argument(
        '--output', '-o', required=True,
        help='输出文件路径（.jsonl 或目录）'
    )
    solve_parser.add_argument(
        '--prompt',
        help='Prompt 模板完整路径（默认使用 src/agenqa/prompts/solver.prompt）'
    )
    solve_parser.add_argument(
        '--service-id',
        default='remote:qwen3-30b-a3b-thinking',
        help='引用 Service/llm_service/configs/services.json 中的 service_id（默认 remote:qwen3-30b-a3b-thinking）'
    )
    solve_parser.add_argument(
        '--service-config',
        help='llm_service 服务配置文件路径，默认指向 Service/llm_service/configs/services.json'
    )
    solve_parser.add_argument(
        '--service-model',
        help='当 --service-id 缺少模型名称时指定使用的模型名'
    )
    solve_parser.add_argument(
        '--concurrency', type=int,
        help='并发请求数（覆盖配置 concurrency.max_workers）'
    )
    solve_parser.set_defaults(func=cmd_solve)

    # head-tail 命令 - 从链路中抽头/尾，形成 k0,qN,aN
    ht_parser = subparsers.add_parser(
        'head-tail',
        help='从多步 extend 链中抽掉中间过程，输出 k0,qN,aN 的 KQA JSONL'
    )
    ht_parser.add_argument(
        '--dir',
        help='运行目录（包含 init/extend-upgrade 产物）；未指定 --head-kqa/--tail-kqa 时启用自动探测'
    )
    ht_parser.add_argument(
        '--head-kqa',
        help='显式指定 head(step=head_step) 的 KQA 文件路径（默认自动从 --dir 识别）'
    )
    ht_parser.add_argument(
        '--tail-kqa',
        help='显式指定 tail(step=tail_step) 的 KQA 文件路径（默认自动从 --dir 识别）'
    )
    ht_parser.add_argument(
        '--head-step', type=int, default=0,
        help='head 的 step（默认 0）'
    )
    ht_parser.add_argument(
        '--tail-step', type=int,
        help='tail 的 step（未提供则自动选择 --dir 下的最大 step）'
    )
    ht_parser.add_argument(
        '--output', '-o', required=True,
        help='输出文件路径（.jsonl 或目录）'
    )
    ht_parser.set_defaults(func=cmd_head_tail)

    hb_parser = subparsers.add_parser(
        'agent-batch-heartbeat',
        help='监控一个 batch lineage，并对 API/网络失败任务自动触发 batch 级 resume'
    )
    hb_parser.add_argument(
        '--batch-dir',
        required=True,
        help='根 batch 目录（heartbeat 监控真源）'
    )
    hb_parser.add_argument(
        '--interval-sec',
        type=int,
        default=60,
        help='轮询间隔秒数（默认: 60）'
    )
    hb_parser.add_argument(
        '--max-auto-resumes-per-task',
        type=int,
        default=2,
        help='单 task 自动续跑上限（默认: 2）'
    )
    hb_parser.add_argument(
        '--resume-suffix',
        default='hb',
        help='保留参数：旧版 sibling heartbeat batch 的后缀前缀；当前原地 heartbeat resume 时忽略'
    )
    hb_parser.add_argument(
        '--continue-until',
        choices=['terminal'],
        default='terminal',
        help='持续监控直到何种条件退出（当前仅支持: terminal）'
    )
    hb_parser.add_argument(
        '--concurrency',
        type=int,
        help='自动续跑 batch 的并发 worker 数；默认继承 root batch manifest'
    )
    hb_parser.add_argument(
        '--max-steps',
        type=int,
        help='自动续跑时覆盖最大步数；默认继承 root batch manifest'
    )
    hb_parser.add_argument(
        '--max-rounds',
        type=int,
        help='自动续跑时覆盖最大轮数；默认继承 root batch manifest'
    )
    hb_parser.add_argument(
        '--max-consecutive-revise',
        type=int,
        help='自动续跑时覆盖连续 Revise 上限；默认继承 root batch manifest'
    )
    hb_parser.add_argument(
        '--lang',
        choices=['zh', 'en'],
        help='自动续跑时覆盖语言；默认继承 root batch manifest'
    )
    hb_parser.add_argument(
        '--source',
        choices=['idealab', 'sjtu', 'aimux', 'dmxapi'],
        help='自动续跑时覆盖 source；默认继承 root batch manifest'
    )
    hb_parser.add_argument(
        '--no-playback',
        action='store_true',
        help='自动续跑时禁用每个 run 的 run_playback_*.md 生成'
    )
    hb_parser.set_defaults(func=cmd_agent_batch_heartbeat)

    status_parser = subparsers.add_parser(
        'agent-batch-status',
        help='查看 batch/lineage 的实时状态、错误计数与任务进度'
    )
    status_parser.add_argument(
        '--batch-dir',
        required=True,
        help='任意 batch 目录；命令会自动解析所属 lineage'
    )
    status_parser.add_argument(
        '--watch',
        action='store_true',
        help='持续刷新状态'
    )
    status_parser.add_argument(
        '--interval-sec',
        type=int,
        default=10,
        help='watch 模式刷新间隔（默认: 10）'
    )
    status_parser.add_argument(
        '--format',
        choices=['table', 'json'],
        default='table',
        help='输出格式（默认: table）'
    )
    status_parser.add_argument(
        '--show-runs',
        choices=['all', 'active', 'failed'],
        default='active',
        help='任务表显示范围（默认: active）'
    )
    status_parser.add_argument(
        '--stale-sec',
        type=int,
        default=300,
        help='超过该秒数未更新的 running task 视为 stale（默认: 300）'
    )
    status_parser.add_argument(
        '--top-errors',
        type=int,
        default=10,
        help='展示最近错误条数（默认: 10）'
    )
    status_parser.set_defaults(func=cmd_agent_batch_status)

    # agent-run 命令 - 基于 LangGraph 的多智能体编排
    agent_parser = subparsers.add_parser(
        'agent-run',
        help='基于 LangGraph 的多智能体编排（实验性）'
    )
    agent_parser.add_argument(
        '--engine', default='langgraph',
        help='执行引擎（默认: langgraph）'
    )
    agent_parser.add_argument(
        '--output', '-o',
        help='输出目录（默认: config.agent.output_dir 或 data/output/agent_run）'
    )
    agent_parser.add_argument(
        '--run-id',
        help='运行 ID（默认: 时间戳）'
    )
    agent_parser.add_argument(
        '--no-playback',
        action='store_true',
        help='禁用 run_playback_*.md 回放文档生成（默认会在 run 完成后生成）'
    )
    agent_parser.add_argument(
        '--resume-run-dir',
        help='从已有 run 目录原地续跑（续跑前会把最后失败/未完成尾部归档到 _resume_archive/；需包含 state.json）'
    )
    agent_parser.add_argument(
        '--resume-clone',
        action='store_true',
        help='续跑时先将 run 目录克隆到新目录（便于对比试验结果；新目录名为 *_<resume-suffix>N）'
    )
    agent_parser.add_argument(
        '--resume-suffix',
        default='resume',
        help='配合 --resume-clone：新目录后缀前缀（默认: resume => *_resume1；原地续跑时忽略）'
    )
    agent_parser.add_argument(
        '--input-kind',
        choices=['paper', 'scipedia'],
        help='快捷选择输入类型：paper（默认按配置）或 scipedia（自动启用 init.source.scipedia_pack，并使用 SCICLONE_SCIPEDIA_PATH 或默认路径）'
    )
    agent_parser.add_argument(
        '--paper-path',
        help='快捷覆盖输入数据源路径（写入 init.source.path；支持 .txt/.json/.jsonl/.pdf）'
    )
    agent_parser.add_argument(
        '--episode-seed-contract',
        help='覆盖 init.episode_seed.contract_path（YAML/JSON contract 文件路径）'
    )
    agent_parser.add_argument(
        '--idealab-session-id',
        help='覆盖 IDEALAB_SESSION_ID（用于 x-idealab-session-id 会话亲和性；优先级高于 env）'
    )
    agent_parser.add_argument(
        '--set-env',
        action='append',
        help='设置环境变量 KEY=VALUE（可重复；优先级最高；用于控制 SCICLONE_* 高级开关）'
    )
    agent_parser.add_argument(
        '--unset-env',
        action='append',
        help='删除环境变量 KEY（可重复；用于清理与 config/runtime_env 冲突的外部 env）'
    )
    agent_parser.add_argument(
        '--max-steps', type=int,
        help='覆盖最大步数（题目数量上限）'
    )
    agent_parser.add_argument(
        '--max-rounds', type=int,
        help='覆盖最大轮数（总操作次数上限，包括 Extend/Revise/Compress 等）'
    )
    agent_parser.add_argument(
        '--max-consecutive-revise',
        type=int,
        help='覆盖连续 Revise 上限（写入 agent.max_consecutive_revise；达到上限且 medium/strong 均错则 fail-fast 终止）'
    )
    agent_parser.add_argument(
        '--revise-retry-limit',
        dest='revise_retry_limit',
        type=int,
        help='同义参数：--max-consecutive-revise'
    )
    agent_parser.add_argument(
        '--lang',
        choices=['zh', 'en'],
        help='题目与答案的主要语言（默认: 配置文件，常用 zh）'
    )
    agent_parser.add_argument(
        '--symbolic-only',
        action='store_true',
        help='Agent 级别启用“符号表达式 ONLY” 模式：出题与判分尽量只用符号表达式，禁止具体数值'
    )
    agent_parser.add_argument(
        '--no-mcq-from-step',
        help='题型策略：从指定 step 开始禁用 MCQ（仅允许 Derivation/Numeric）。示例：--no-mcq-from-step 2；禁用策略：--no-mcq-from-step off'
    )
    agent_parser.add_argument(
        '--allowed-question-types',
        nargs='+',
        help='题型白名单：仅允许在该列表中的题型（对 step policy 的 base_allowed 做交集过滤）。示例：--allowed-question-types MCQ Derivation；禁用白名单：--allowed-question-types off'
    )
    agent_parser.add_argument(
        '--roles-protocol',
        choices=['json', 'tagged'],
        help='Roles 链路输出协议（Draft/Format 等）：json（默认，仅 JSON）或 tagged（带字段标记的纯文本，解析时以标记为主）'
    )
    agent_parser.add_argument(
        '--format-validation-mode',
        choices=['soft', 'hard'],
        help='Format 自检 gate：soft（允许进入 solve）或 hard（自检失败则跳过本轮 solve 并不落盘到 history；默认: hard）'
    )
    agent_parser.add_argument(
        '--source',
        choices=['idealab', 'sjtu', 'aimux', 'dmxapi'],
        help='可选：声明推理源（用于启用 provider-specific CLI 兼容行为；当前仅 Idealab 启用数字别名解析）。不填则不启用别名解析。'
    )
    agent_parser.add_argument(
        '--main-model',
        help='仅 Idealab：主模型名称或数字别名（覆盖 init/director/operators.extend/solvers.strong 的 model_name；也会影响 strong-models 未指定 main-model 时的默认推断）。'
    )
    agent_parser.add_argument(
        '--strong-models',
        help='仅 Idealab：快捷指定多个 strong solver 的 model_name（逗号/空格分隔；按给定顺序写入；若未指定 --main-model 则自动同步首个模型）。示例：--strong-models "5,7,9"'
    )
    agent_parser.add_argument(
        '--medium-model',
        help='覆盖 solvers.medium 的 model_name（建议直接写完整模型名；Idealab 数字别名仅在 --source idealab 下可用）'
    )
    agent_parser.add_argument(
        '--struct-model',
        help='覆盖结构化 roles family 的 model_name（Diagnose/Format/StepCertBuilder；写入 operators.*.struct_generator；Idealab 数字别名仅在 --source idealab 下可用）'
    )
    agent_parser.add_argument(
        '--format-model',
        help='覆盖 Format 角色的 model_name（写入 operators.*.format_generator；Idealab 模式支持数字别名）'
    )
    agent_parser.add_argument(
        '--client-stream',
        action='store_true',
        help='启用客户端侧流式聚合（对所有生成器的 client.stream 置 true）'
    )
    agent_parser.add_argument(
        '--consensus-mode',
        choices=['always', 'none', 'disabled', 'off'],
        help='覆盖 multi-strong 共识触发策略（写入 consensus.mode；always=每轮运行所有 strong，none/disabled/off=关闭）'
    )
    agent_parser.add_argument(
        '--all-strong',
        action='store_true',
        help='等价于 --consensus-mode always：强制每轮运行所有 strong'
    )
    agent_parser.set_defaults(func=cmd_agent_run)

    batch_parser = subparsers.add_parser(
        'agent-batch-run',
        help='并发运行多个 paper 的完整 agent-run'
    )
    batch_parser.add_argument(
        '--output', '-o',
        help='批任务输出根目录（默认: data/output/agent_batch_run）'
    )
    batch_input = batch_parser.add_mutually_exclusive_group(required=False)
    batch_input.add_argument(
        '--paper-list',
        help='paper 列表文件（支持 .txt/.json/.jsonl/.csv；CSV 可读取 localRelpath/paper_path/path）'
    )
    batch_input.add_argument(
        '--paper-dir',
        help='paper 目录（配合 --glob 扫描）'
    )
    batch_input.add_argument(
        '--resume-batch-dir',
        help='从已有 batch 目录原地续跑未完成/失败任务（原 batch 目录保持真源；旧尾部状态会归档到 _resume_archive/）'
    )
    batch_parser.add_argument(
        '--glob',
        default='**/*.pdf',
        help='目录模式下的 glob 模式（默认: **/*.pdf；若未传 --paper-list/--paper-dir，也可从 config.batch_input 读取）'
    )
    batch_parser.add_argument(
        '--concurrency',
        type=int,
        default=2,
        help='并发 worker 数（默认: 2）'
    )
    batch_parser.add_argument(
        '--max-tasks',
        type=int,
        help='最多处理多少个 paper（默认: 全部）'
    )
    batch_parser.add_argument(
        '--batch-id',
        help='批任务 ID（默认: 自动生成）'
    )
    batch_parser.add_argument(
        '--run-prefix',
        help='单个 run_id 前缀（默认: paperNNN）'
    )
    batch_parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='即使存在失败任务也以 0 退出码结束'
    )
    batch_parser.add_argument(
        '--inline-status',
        action='store_true',
        help='在 agent-batch-run 同一终端内周期性打印监控摘要（复用 live_status telemetry）'
    )
    batch_parser.add_argument(
        '--status-interval-sec',
        type=int,
        default=30,
        help='配合 --inline-status：监控摘要打印间隔秒数（默认: 30）'
    )
    batch_parser.add_argument(
        '--status-active-limit',
        type=int,
        default=8,
        help='配合 --inline-status：每次摘要最多显示多少条活跃 task（默认: 8）'
    )
    batch_parser.add_argument(
        '--status-top-errors',
        type=int,
        default=5,
        help='配合 --inline-status：每次摘要显示多少条模型/任务错误热点（默认: 5）'
    )
    batch_parser.add_argument(
        '--resume-suffix',
        default='resume',
        help='保留参数：仅在显式 clone 兼容路径下使用；默认原地 batch resume 时忽略'
    )
    batch_parser.add_argument(
        '--no-playback',
        action='store_true',
        help='禁用每个 run 的 run_playback_*.md 回放文档生成'
    )
    batch_parser.add_argument(
        '--max-steps', type=int,
        help='覆盖最大步数（题目数量上限）'
    )
    batch_parser.add_argument(
        '--max-rounds', type=int,
        help='覆盖最大轮数（总操作次数上限）'
    )
    batch_parser.add_argument(
        '--max-consecutive-revise',
        type=int,
        help='覆盖连续 Revise 上限'
    )
    batch_parser.add_argument(
        '--lang',
        choices=['zh', 'en'],
        help='题目与答案的主要语言'
    )
    batch_parser.add_argument(
        '--source',
        choices=['idealab', 'sjtu', 'aimux', 'dmxapi'],
        help='可选：声明推理源'
    )
    batch_parser.set_defaults(func=cmd_agent_batch_run)

    # agent-dry-run 命令 - 不依赖 LLM 的结构验证
    dry_parser = subparsers.add_parser(
        'agent-dry-run',
        help='进行一次不依赖 LLM 的干跑，验证产物结构与状态递增/替换语义'
    )
    dry_parser.add_argument('-o', '--output', help='输出目录（默认: data/output/agent_run_dryrun）')
    dry_parser.add_argument('--run-id', help='运行 ID（默认: dryrun-时间戳）')
    dry_parser.set_defaults(func=cmd_agent_dry_run)

    # agent 命令 - 通过 OpenAI 兼容端点快速运行 Agent
    agent_entry_parser = subparsers.add_parser(
        'agent',
        help='Agent 快速运行入口（SJTU Relay / Idealab / AIMux / DMXAPI OpenAI 兼容端点）'
    )
    agent_entry_parser.add_argument(
        '--source',
        choices=['sjtu', 'idealab', 'aimux', 'dmxapi'],
        default='sjtu',
        help="API 来源：'sjtu'（默认）/'idealab'/'aimux'/'dmxapi'"
    )
    agent_entry_parser.add_argument(
        '--output', '-o',
        default='data/output/agent_run',
        help='输出目录（默认: data/output/agent_run）'
    )
    agent_entry_parser.add_argument(
        '--input-kind',
        choices=['paper', 'scipedia'],
        help='快捷选择输入类型：paper（默认按配置）或 scipedia（自动启用 init.source.scipedia_pack，并使用 SCICLONE_SCIPEDIA_PATH 或默认路径）'
    )
    agent_entry_parser.add_argument(
        '--paper-path',
        help='快捷覆盖输入数据源路径（传递给 agent-run 的 init.source.path；支持 .txt/.json/.jsonl/.pdf）'
    )
    agent_entry_parser.add_argument(
        '--episode-seed-contract',
        help='覆盖 init.episode_seed.contract_path（传递给 agent-run）'
    )
    agent_entry_parser.add_argument(
        '--max-steps',
        type=int,
        default=10,
        help='Episode 最大步数/题目数量上限（默认: 10）'
    )
    agent_entry_parser.add_argument(
        '--max-rounds',
        type=int,
        help='最大轮数/总操作次数上限（默认: 与 max-steps 相同）'
    )
    agent_entry_parser.add_argument(
        '--max-consecutive-revise',
        type=int,
        help='覆盖连续 Revise 上限（写入 agent.max_consecutive_revise；达到上限且 medium/strong 均错则 fail-fast 终止）'
    )
    agent_entry_parser.add_argument(
        '--revise-retry-limit',
        dest='revise_retry_limit',
        type=int,
        help='同义参数：--max-consecutive-revise'
    )
    agent_entry_parser.add_argument(
        '--api-key',
        help='API Key（覆盖环境变量 AGENT_API_KEY / IDEALAB_API_KEY / AIMUX_API_KEY / DMXAPI_API_KEY）'
    )
    agent_entry_parser.add_argument(
        '--base-url',
        help='覆盖 OpenAI-compatible 基础 URL（公开默认: https://api.openai.com/v1）'
    )
    agent_entry_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只执行 agent-dry-run（结构验证，不发起真实推理）'
    )
    agent_entry_parser.add_argument(
        '--skip-smoke',
        action='store_true',
        help='跳过 /models 连通性 smoke test'
    )
    agent_entry_parser.add_argument(
        '--lang',
        choices=['zh', 'en'],
        help='题目与答案的主要语言（传递给 agent-run）'
    )
    agent_entry_parser.add_argument(
        '--resume-run-dir',
        help='从已有 run 目录原地续跑（续跑前会把最后失败/未完成尾部归档到 _resume_archive/）'
    )
    agent_entry_parser.add_argument(
        '--resume-suffix',
        default='r',
        help='保留参数：仅在显式 clone 兼容路径下使用（默认: r => *_r1）；原地续跑时忽略'
    )
    agent_entry_parser.add_argument(
        '--timeout',
        type=int,
        help='API 请求超时（秒），覆盖配置文件中的 client.timeout（默认: 配置文件值或 120）'
    )
    agent_entry_parser.add_argument(
        '--idealab-session-id',
        help='覆盖 IDEALAB_SESSION_ID（用于 x-idealab-session-id 会话亲和性；优先级高于 env）'
    )
    agent_entry_parser.add_argument(
        '--set-env',
        action='append',
        help='设置环境变量 KEY=VALUE（可重复；优先级最高；用于控制 SCICLONE_* 高级开关）'
    )
    agent_entry_parser.add_argument(
        '--unset-env',
        action='append',
        help='删除环境变量 KEY（可重复；用于清理与 config/runtime_env 冲突的外部 env）'
    )
    agent_entry_parser.add_argument(
        '--client-stream',
        action='store_true',
        help='启用客户端侧流式聚合（对所有生成器的 client.stream 置 true）'
    )
    agent_entry_parser.add_argument(
        '--roles-protocol',
        choices=['json', 'tagged'],
        help='Roles 链路输出协议（Draft/Format 等）：json（默认，仅 JSON）或 tagged（带字段标记的纯文本，解析时以标记为主）'
    )
    agent_entry_parser.add_argument(
        '--format-validation-mode',
        choices=['soft', 'hard'],
        help='Format 自检 gate：soft（允许进入 solve）或 hard（自检失败则跳过本轮 solve 并不落盘到 history；默认: hard）'
    )
    agent_entry_parser.add_argument(
        '--symbolic-only',
        action='store_true',
        help='Agent 级别启用“符号表达式 ONLY” 模式：出题与判分尽量只用符号表达式，禁止具体数值（转传给 agent-run 与配置 agent.symbolic_only）'
    )
    agent_entry_parser.add_argument(
        '--no-mcq-from-step',
        help='题型策略：从指定 step 开始禁用 MCQ（仅允许 Derivation/Numeric）。示例：--no-mcq-from-step 2；禁用策略：--no-mcq-from-step off'
    )
    agent_entry_parser.add_argument(
        '--allowed-question-types',
        nargs='+',
        help='题型白名单：仅允许在该列表中的题型（对 step policy 的 base_allowed 做交集过滤）。示例：--allowed-question-types MCQ Derivation；禁用白名单：--allowed-question-types off'
    )
    agent_entry_parser.add_argument(
        '--main-model',
        help='主模型名称（Idealab 模式下覆盖 init/director/operators.extend/solvers.strong 的 model_name，例如 gpt-51-1113-global 或 qwen3-max）'
    )
    agent_entry_parser.add_argument(
        '--medium-model',
        help='仅 Idealab：覆盖 solvers.medium 的 model_name（支持数字别名；例如 8 -> gpt-5-mini-0807-global）'
    )
    agent_entry_parser.add_argument(
        '--struct-model',
        help='仅 Idealab：覆盖结构化 roles family 的 model_name（Diagnose/Format/StepCertBuilder；写入 operators.*.struct_generator；支持数字别名）'
    )
    agent_entry_parser.add_argument(
        '--format-model',
        help='仅 Idealab：覆盖 Format 角色的 model_name（写入 operators.*.format_generator；支持数字别名）'
    )
    agent_entry_parser.add_argument(
        '--strong-models',
        help='仅 Idealab：快捷指定多个 strong solver 的 model_name（逗号/空格分隔；按给定顺序写入；若未指定 --main-model 则自动同步首个模型）。示例：--strong-models "7,6,4"'
    )
    agent_entry_parser.add_argument(
        '--consensus-mode',
        choices=['always', 'none', 'disabled', 'off'],
        help='覆盖 multi-strong 共识触发策略（传递给 agent-run 的 consensus.mode；always=每轮运行所有 strong）'
    )
    agent_entry_parser.add_argument(
        '--all-strong',
        action='store_true',
        help='等价于 --consensus-mode always：强制每轮运行所有 strong'
    )
    agent_entry_parser.set_defaults(func=cmd_agent)

    # 解析命令行参数
    argv = _reorder_global_cli_args(sys.argv[1:], commands=set(subparsers.choices.keys()))
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 执行对应的命令函数
    args.func(args)


if __name__ == '__main__':
    main()
