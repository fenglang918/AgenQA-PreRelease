"""Shared helpers for thin AgenQA -> TuningFactory launchers."""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TUNINGFACTORY_ROOT = REPO_ROOT / "external" / "TuningFactory"
DEFAULT_PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_TORCHRUN_BIN = REPO_ROOT / ".venv" / "bin" / "torchrun"


def default_python_bin() -> str:
    return str(DEFAULT_PYTHON_BIN if DEFAULT_PYTHON_BIN.is_file() else Path(sys.executable))


def default_torchrun_bin() -> str:
    return str(DEFAULT_TORCHRUN_BIN if DEFAULT_TORCHRUN_BIN.is_file() else Path("torchrun"))


@dataclass(frozen=True)
class ExportedDatasetPaths:
    dataset_root: str
    summary_path: str
    train_file: str
    eval_file: str | None
    test_file: str | None
    all_file: str


def resolve_exported_dataset(dataset_root: Path) -> ExportedDatasetPaths:
    root = dataset_root.expanduser().resolve()
    tuningfactory_dir = root / "tuningfactory"
    manifests_dir = root / "manifests"
    summary_path = manifests_dir / "summary.json"
    all_file = tuningfactory_dir / "all.json"
    train_file = tuningfactory_dir / "train.json"
    eval_file = tuningfactory_dir / "eval.json"
    test_file = tuningfactory_dir / "test.json"

    if not all_file.is_file():
        raise FileNotFoundError(f"TuningFactory adapter file not found: {all_file}")

    resolved_train = train_file if train_file.is_file() else all_file
    resolved_eval = eval_file if eval_file.is_file() else None
    resolved_test = test_file if test_file.is_file() else None

    return ExportedDatasetPaths(
        dataset_root=str(root),
        summary_path=str(summary_path if summary_path.is_file() else ""),
        train_file=str(resolved_train),
        eval_file=str(resolved_eval) if resolved_eval else None,
        test_file=str(resolved_test) if resolved_test else None,
        all_file=str(all_file),
    )


def append_arg(args: list[str], flag: str, value: str | int | float | None) -> None:
    if value is None:
        return
    args.extend([flag, str(value)])


def append_bool(args: list[str], flag: str, value: bool) -> None:
    args.extend([flag, "True" if value else "False"])


def build_tuningfactory_sft_args(parsed: Any, dataset: ExportedDatasetPaths) -> list[str]:
    args = [
        "--stage",
        "sft",
        "--do_train",
        "--model_name_or_path",
        parsed.model_name_or_path,
        "--file_name",
        dataset.train_file,
        "--prompt",
        "instruction",
        "--query",
        "input",
        "--response",
        "output",
        "--template",
        parsed.template,
        "--finetuning_type",
        "lora",
        "--lora_target",
        parsed.lora_target,
        "--output_dir",
        str(Path(parsed.output_dir).expanduser().resolve()),
        "--lr_scheduler_type",
        parsed.lr_scheduler_type,
        "--learning_rate",
        str(parsed.learning_rate),
        "--cutoff_len",
        str(parsed.cutoff_len),
        "--per_device_train_batch_size",
        str(parsed.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(parsed.gradient_accumulation_steps),
        "--logging_steps",
        str(parsed.logging_steps),
        "--save_steps",
        str(parsed.save_steps),
        "--num_train_epochs",
        str(parsed.num_train_epochs),
        "--preprocessing_num_workers",
        str(parsed.preprocessing_num_workers),
        "--dataloader_num_workers",
        str(parsed.dataloader_num_workers),
        "--report_to",
        parsed.report_to,
    ]

    append_bool(args, "--overwrite_cache", parsed.overwrite_cache)
    append_bool(args, "--overwrite_output_dir", parsed.overwrite_output_dir)
    append_bool(args, "--train_on_prompt", parsed.train_on_prompt)
    append_bool(args, "--enable_thinking", parsed.enable_thinking)

    append_arg(args, "--lora_rank", parsed.lora_rank)
    append_arg(args, "--lora_alpha", parsed.lora_alpha)
    append_arg(args, "--lora_dropout", parsed.lora_dropout)
    append_arg(args, "--max_samples", parsed.max_samples)
    append_arg(args, "--tokenized_path", parsed.tokenized_path)

    if parsed.plot_loss:
        args.append("--plot_loss")

    if getattr(parsed, "eval_steps", None) is not None and dataset.eval_file:
        args.extend(
            [
                "--do_eval",
                "--eval_file_name",
                dataset.eval_file,
                "--per_device_eval_batch_size",
                str(parsed.per_device_eval_batch_size),
                "--evaluation_strategy",
                "steps",
                "--eval_steps",
                str(parsed.eval_steps),
                "--load_best_model_at_end",
            ]
        )

    if parsed.precision == "bf16":
        args.append("--bf16")
    elif parsed.precision == "fp16":
        args.append("--fp16")

    return args


def write_run_spec(
    *,
    output_dir: Path,
    command: Sequence[str],
    dataset: ExportedDatasetPaths,
    config: dict[str, Any],
    launcher: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_spec_path = output_dir / "agenqa_tuningfactory_run_spec.json"
    payload = {
        "launcher": launcher,
        "dataset": asdict(dataset),
        "command": list(command),
        "command_shell": shlex.join(command),
        "config": config,
    }
    run_spec_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_spec_path


def load_config_overrides(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Launcher config must be a JSON object: {path}")
    return payload
