"""Generate a human-readable playback markdown for an AgenQA run directory.

This is intended to mimic older `run_playback_*.md` style summaries:
step-by-step, with full question/answer, director decisions, operator subruns,
and solver results (including tool usage signals).
"""

from __future__ import annotations

import json
import shlex
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl_first(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                return None
            return obj if isinstance(obj, dict) else None
    return None


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _truncate(text: str, *, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 1] + "…", True


def _extract_round_idx(path: Path) -> Optional[int]:
    # expects .../round_3/...
    for part in path.parts:
        m = re.fullmatch(r"round_(\d+)", part)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def _infer_step_from_dir(dir_path: Path, *, round_idx: Optional[int]) -> Optional[int]:
    # Prefer explicit step dirs (round_1/step_0_init, round_1/step_1_extend, ...)
    for part in dir_path.parts:
        m = re.fullmatch(r"step_(\d+)_.*", part)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None

    # For unified pipeline layout, round_k/{director,extend} belongs to step=k.
    if round_idx is not None and dir_path.name in {"director", "extend", "revise"}:
        return int(round_idx)

    return None


def _label_from_dir(p: Path) -> str:
    name = p.name
    if name.startswith("step_") and "_" in name:
        # step_1_extend -> Extend
        tail = name.split("_", 2)[-1]
        return tail.replace("_", " ").title()
    if name in {"director", "extend", "revise"}:
        return name.title()
    return name.replace("_", " ").title()


@dataclass(frozen=True)
class RoundEntry:
    round_idx: int
    step: int
    label: str
    dir_path: Path
    decision_path: Optional[Path]


def _iter_director_decisions(run_dir: Path) -> List[RoundEntry]:
    entries: List[RoundEntry] = []
    for p in sorted(run_dir.glob("**/director_decision.json")):
        try:
            obj = _read_json(p)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        r_idx = _extract_round_idx(p)
        if r_idx is None:
            continue
        dir_path = p.parent

        inferred_step = _infer_step_from_dir(dir_path, round_idx=r_idx)
        if inferred_step is None:
            step = obj.get("step")
            try:
                inferred_step = int(step)
            except Exception:
                continue

        label = _label_from_dir(dir_path)
        entries.append(
            RoundEntry(
                round_idx=r_idx,
                step=int(inferred_step),
                label=label,
                dir_path=dir_path,
                decision_path=p,
            )
        )

    # Ensure director before extend within same round/step when both exist.
    def _sort_key(e: RoundEntry) -> Tuple[int, int, int, str]:
        label_rank = 0
        if e.label.lower() == "director":
            label_rank = 0
        elif e.label.lower() == "extend":
            label_rank = 1
        elif e.label.lower() == "revise":
            label_rank = 2
        else:
            label_rank = 3
        return (e.step, e.round_idx, label_rank, e.label)

    return sorted(entries, key=_sort_key)


def _get_history_records(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    hist = state.get("history")
    return hist if isinstance(hist, list) else []


def _find_history_by_step(history: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    for rec in history:
        try:
            if int(rec.get("step") or rec.get("qa_idx")) == int(step):
                return rec
        except Exception:
            continue
    return None


def _iter_subrun_jsons(dir_path: Path) -> List[Path]:
    sub = dir_path / "subruns"
    if not sub.exists():
        return []
    return sorted([p for p in sub.glob("*.json") if p.is_file()])


def _iter_solver_views(solve_dir: Path) -> List[Tuple[str, str, Path, Optional[Path]]]:
    # (tier_label, view_label, solve_jsonl, raw_jsonl)
    pairs: List[Tuple[str, str, Path, Optional[Path]]] = []
    for view_prefix in ["solve", "solve_path"]:
        for tier in ["medium", "strong"]:
            p = solve_dir / f"{view_prefix}_{tier}.jsonl"
            raw = solve_dir / f"{view_prefix}_{tier}_raw.jsonl"
            pairs.append((tier, "edge" if view_prefix == "solve" else "path", p, raw if raw.exists() else None))
    return pairs


def _step0_known_snapshot(run_dir: Path, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    seed_path = run_dir / "round_1" / "step_0_init" / "subruns" / "03_seed_init.json"
    episode_seed = None
    if seed_path.exists():
        try:
            seed_obj = _read_json(seed_path)
            if isinstance(seed_obj, dict):
                episode_seed = seed_obj.get("episode_seed")
        except Exception:
            episode_seed = None

    if episode_seed is None:
        mem = state.get("memory")
        if isinstance(mem, dict):
            episode_seed = mem.get("episode_seed")

    if not isinstance(episode_seed, dict) or not episode_seed:
        return None

    schema_version = 2
    mem = state.get("memory")
    if isinstance(mem, dict) and isinstance(mem.get("schema_version"), int):
        schema_version = mem["schema_version"]

    return {
        "schema_version": schema_version,
        "episode_seed": episode_seed,
        "premise_bank": [],
        "fact_bank": [],
        "step_certs": [],
    }


def _shell_quote(val: str) -> str:
    return shlex.quote(val)


def _render_repro_section(run_id: str, cfg: Dict[str, Any]) -> List[str]:
    """Render a best-effort reproducible CLI command from run_config.json.

    The command is reconstructed from cfg["cli_args"]. Secrets (API keys) are
    intentionally not included; users should set them via environment variables.
    """
    if not isinstance(cfg, dict):
        return []
    cli_args = cfg.get("cli_args")
    if not isinstance(cli_args, dict):
        return []

    config_path = cfg.get("config_path")
    if not isinstance(config_path, str) or not config_path.strip():
        return []

    command = cli_args.get("command")
    if not isinstance(command, str) or not command.strip():
        command = "agent-run"

    flags: List[tuple[str, Optional[Any]]] = []

    def add(flag: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        flags.append((flag, value))

    def add_bool(flag: str, enabled: Any) -> None:
        if enabled is True:
            flags.append((flag, None))

    # Core
    add("--source", cli_args.get("source"))
    add("--lang", cli_args.get("lang"))
    add("--input-kind", cli_args.get("input_kind"))
    add("--paper-path", cli_args.get("paper_path"))
    add("--episode-seed-contract", cli_args.get("episode_seed_contract"))

    # Models
    add("--main-model", cli_args.get("main_model"))
    add("--strong-models", cli_args.get("strong_models"))
    add("--medium-model", cli_args.get("medium_model"))
    add("--struct-model", cli_args.get("struct_model"))
    add("--format-model", cli_args.get("format_model"))

    # Policy & limits
    add("--no-mcq-from-step", cli_args.get("no_mcq_from_step"))
    add("--max-steps", cli_args.get("max_steps"))
    add("--max-rounds", cli_args.get("max_rounds"))
    add("--max-consecutive-revise", cli_args.get("max_consecutive_revise"))

    # Multi-strong
    add("--consensus-mode", cli_args.get("consensus_mode"))
    add_bool("--all-strong", cli_args.get("all_strong"))

    # Other toggles
    add_bool("--client-stream", cli_args.get("client_stream"))
    add_bool("--symbolic-only", cli_args.get("symbolic_only"))
    add_bool("--no-playback", cli_args.get("no_playback"))

    # Resume
    add("--resume-run-dir", cli_args.get("resume_run_dir"))

    # Output + run id
    output_dir = cli_args.get("output")
    if isinstance(output_dir, str) and output_dir.strip():
        add("--output", output_dir.strip())
    add("--run-id", run_id)

    cmd_lines: List[str] = []
    src = str(cli_args.get("source") or "").strip()
    if src == "idealab":
        cmd_lines.append('export IDEALAB_API_KEY="sk-..."')
        cmd_lines.append('')
    elif src == "sjtu":
        cmd_lines.append('export AGENT_API_KEY="sk-..."')
        cmd_lines.append('')

    base = f".venv/bin/python src/cli.py -c {_shell_quote(config_path.strip())} {command.strip()}"

    rendered: List[str] = []
    for flag, value in flags:
        if value is None:
            rendered.append(flag)
        else:
            rendered.append(f"{flag} {_shell_quote(str(value))}")

    if not rendered:
        cmd_lines.append(base)
    else:
        cmd_lines.append(base + " \\" )
        for i, item in enumerate(rendered):
            tail = " \\" if i < len(rendered) - 1 else ""
            cmd_lines.append(f"  {item}{tail}")

    md: List[str] = []
    md.append("## Reproduce")
    md.append("")
    md.append("命令来自 `run_config.json` 的 `cli_args`（不包含明文密钥；请自行设置环境变量）。")
    md.append("")
    md.append("```bash")
    md.extend(cmd_lines)
    md.append("```")
    md.append("")
    return md


def generate_playback_md(run_dir: Path, *, output_path: Optional[Path] = None) -> Path:
    run_dir = run_dir.resolve()
    state_path = run_dir / "state.json"
    cfg_path = run_dir / "run_config.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing state.json: {state_path}")
    state = _read_json(state_path)
    if not isinstance(state, dict):
        raise ValueError(f"invalid state.json: {state_path}")
    cfg = _read_json(cfg_path) if cfg_path.exists() else {}

    run_id = str(state.get("run_id") or run_dir.name.replace("run_", "")).strip() or run_dir.name
    max_steps = state.get("max_steps")
    try:
        total_steps = int(max_steps) if isinstance(max_steps, (int, float, str)) else None
    except Exception:
        total_steps = None

    history = _get_history_records(state)
    last_step = state.get("step")
    try:
        last_step_i = int(last_step) if last_step is not None else (
            max(int(r.get("step", 0)) for r in history) if history else 0
        )
    except Exception:
        last_step_i = 0

    decisions = _iter_director_decisions(run_dir)
    decisions_by_step: Dict[int, List[RoundEntry]] = defaultdict(list)
    for e in decisions:
        decisions_by_step[e.step].append(e)

    if output_path is None:
        output_path = run_dir / f"run_playback_{run_id}.md"

    lines: List[str] = []
    lines.append("# Agent Run Playback")
    lines.append("")
    lines.append(f"- **Run ID**: `{run_id}`")
    if total_steps is not None:
        lines.append(f"- **Total Steps**: {total_steps}")
    lines.append(f"- **Artifacts Dir**: `{run_dir}`")
    if isinstance(cfg, dict):
        cp = cfg.get("config_path")
        if isinstance(cp, str) and cp.strip():
            lines.append(f"- **Config Path**: `{cp.strip()}`")
        git = cfg.get("git") if isinstance(cfg.get("git"), dict) else None
        if isinstance(git, dict) and isinstance(git.get("describe"), str):
            lines.append(f"- **Git**: `{git.get('describe')}` ({git.get('subject')})")
    lines.append("")
    # Repro command (best-effort)
    try:
        lines.extend(_render_repro_section(run_id, cfg if isinstance(cfg, dict) else {}))
    except Exception:
        pass

    lines.append("---")
    lines.append("")

    step_min = 0
    step_max = max(last_step_i, max(decisions_by_step.keys()) if decisions_by_step else 0)

    for step in range(step_min, step_max + 1):
        rec = _find_history_by_step(history, step)
        subject = str(state.get("subject") or (rec.get("subject") if rec else None) or "N/A")
        paper_id = (rec.get("paper_id") if rec else state.get("paper_id")) if isinstance(rec, dict) else state.get("paper_id")
        paper_id = paper_id if isinstance(paper_id, str) and paper_id.strip() else "N/A"

        lines.append(f"## Step {step} — {subject if step > 0 else 'N/A'}")
        lines.append("")
        lines.append(f"- **Paper ID**: `{paper_id}`")
        step_entries = decisions_by_step.get(step, [])
        lines.append(f"- **Rounds in this step**: {len(step_entries) if step_entries else 0}")
        lines.append("")

        if step > 0 and rec:
            q = str(rec.get("question") or "")
            a = str(rec.get("answer") or "")
            known = str(rec.get("known") or "")
            lines.append("### Question")
            lines.append("")
            lines.append("```text")
            lines.append(q)
            lines.append("```")
            lines.append("")
            lines.append("### Answer (GT)")
            lines.append("")
            lines.append("```text")
            lines.append(a)
            lines.append("```")
            lines.append("")
            lines.append("### Known")
            lines.append("")
            lines.append("```text")
            known_show, truncated = _truncate(known, max_chars=12000)
            lines.append(known_show)
            if truncated:
                lines.append("")
                lines.append(f"... (truncated; total chars={len(known)})")
            lines.append("```")
            lines.append("")
        else:
            lines.append("### Question")
            lines.append("")
            lines.append("*(empty)*")
            lines.append("")
            lines.append("### Answer (GT)")
            lines.append("")
            lines.append("*(empty)*")
            lines.append("")
            lines.append("### Known")
            lines.append("")
            snap = _step0_known_snapshot(run_dir, state)
            if snap:
                lines.append("```json")
                lines.append(_pretty_json(snap))
                lines.append("```")
            else:
                lines.append("*(empty)*")
            lines.append("")

        for entry in step_entries:
            lines.append(f"### Round {entry.round_idx} — {entry.label}")
            lines.append("")
            lines.append(f"- **Step**: {step}")
            lines.append(f"- **Round**: {entry.round_idx}")
            if entry.decision_path and entry.decision_path.exists():
                try:
                    dec = _read_json(entry.decision_path)
                except Exception:
                    dec = {}
                op = dec.get("operation") if isinstance(dec, dict) else None
                reason = dec.get("reason") if isinstance(dec, dict) else None
                lines.append(f"- **Director Operation**: `{op}`" if isinstance(op, str) else "- **Director Operation**: `N/A`")
                if isinstance(reason, str) and reason.strip():
                    lines.append(f"- **Director Reason**: {reason.strip()}")
            else:
                lines.append("- **Director Operation**: `N/A`")
            lines.append("")

            subrun_paths = _iter_subrun_jsons(entry.dir_path)
            if subrun_paths:
                lines.append("#### Operator Subruns")
                lines.append("")
                for sp in subrun_paths:
                    role_name = re.sub(r"^\\d+_", "", sp.stem)
                    lines.append("<details>")
                    lines.append(f"<summary><strong>{role_name}</strong></summary>")
                    lines.append("")
                    try:
                        obj = _read_json(sp)
                        payload = _pretty_json(obj)
                    except Exception:
                        payload = _read_text(sp)
                    lines.append("```json")
                    lines.append(payload)
                    lines.append("```")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

            noc = entry.dir_path / "subruns_raw" / "numeric_oracle"
            if noc.exists():
                oracle_code = noc / "oracle_code.py"
                exec_res = noc / "executor_result.json"
                if oracle_code.exists() or exec_res.exists():
                    lines.append("#### Numeric Oracle (Internal)")
                    lines.append("")
                    if exec_res.exists():
                        lines.append("<details>")
                        lines.append("<summary><strong>executor_result.json</strong></summary>")
                        lines.append("")
                        lines.append("```json")
                        lines.append(_pretty_json(_read_json(exec_res)))
                        lines.append("```")
                        lines.append("")
                        lines.append("</details>")
                        lines.append("")
                    if oracle_code.exists():
                        lines.append("<details>")
                        lines.append("<summary><strong>oracle_code.py</strong></summary>")
                        lines.append("")
                        lines.append("```python")
                        code, truncated = _truncate(_read_text(oracle_code), max_chars=4000)
                        lines.append(code)
                        if truncated:
                            lines.append("")
                            lines.append("... (truncated)")
                        lines.append("```")
                        lines.append("")
                        lines.append("</details>")
                        lines.append("")

            solve_dir = entry.dir_path / "solve"
            if solve_dir.exists():
                lines.append("#### Solver Results")
                lines.append("")
                for tier, view, jsonl_path, raw_path in _iter_solver_views(solve_dir):
                    row = _read_jsonl_first(jsonl_path) if jsonl_path.exists() else None
                    if not row:
                        continue
                    model = row.get("model")
                    correct = row.get("correct")
                    solve = row.get("solve")
                    err = row.get("error")
                    tool = row.get("tool") if isinstance(row.get("tool"), dict) else {}
                    tool_used = tool.get("used") if isinstance(tool, dict) else None
                    tool_val = tool.get("value") if isinstance(tool, dict) else None
                    tool_exec = tool.get("exec") if isinstance(tool, dict) else None
                    tool_ok = tool_exec.get("success") if isinstance(tool_exec, dict) else None

                    title = f"**{view.upper()} {tier.title()}**"
                    mlabel = f"`{model}`" if isinstance(model, str) and model else "`(unknown model)`"
                    lines.append(f"{title}: {mlabel} — Correct: `{correct}` — ToolUsed: `{tool_used}`")
                    if tool_used and tool_ok is not None:
                        lines.append(f"- Tool exec success: `{tool_ok}`; tool value: `{tool_val}`")
                    if isinstance(solve, str) and solve.strip():
                        lines.append("")
                        lines.append("```text")
                        lines.append(solve.strip())
                        lines.append("```")
                    if isinstance(err, str) and err.strip():
                        lines.append("")
                        lines.append("```text")
                        lines.append(err.strip())
                        lines.append("```")
                    lines.append("")

                    if raw_path and raw_path.exists():
                        raw_row = _read_jsonl_first(raw_path)
                        if raw_row and isinstance(raw_row.get("text"), str) and raw_row.get("text").strip():
                            raw_text, truncated = _truncate(raw_row["text"], max_chars=3000)
                            lines.append("<details>")
                            lines.append(f"<summary><strong>{view.upper()} {tier.title()} Raw Output</strong></summary>")
                            lines.append("")
                            lines.append("```text")
                            lines.append(raw_text)
                            if truncated:
                                lines.append("")
                                lines.append("... (truncated)")
                            lines.append("```")
                            lines.append("")
                            lines.append("</details>")
                            lines.append("")

        lines.append("---")
        lines.append("")

    final_comment = run_dir / "00_Summary" / "final_comment.json"
    if final_comment.exists():
        lines.append("## Final Comment")
        lines.append("")
        lines.append(f"- `{final_comment}`")
        lines.append("")
        lines.append("```json")
        lines.append(_pretty_json(_read_json(final_comment)))
        lines.append("```")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path
