"""Thin local launcher for AgenQA Phase 1 SFT on top of TuningFactory."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .launcher_lib import (
    DEFAULT_TUNINGFACTORY_ROOT,
    ExportedDatasetPaths,
    build_tuningfactory_sft_args,
    default_python_bin,
    default_torchrun_bin,
    load_config_overrides,
    resolve_exported_dataset,
    write_run_spec,
)


def build_sft_command(parsed: argparse.Namespace, dataset: ExportedDatasetPaths) -> list[str]:
    tf_root = Path(parsed.tuningfactory_root).expanduser().resolve()
    train_entry = tf_root / "src" / "train_bash.py"
    if not train_entry.is_file():
        raise FileNotFoundError(f"TuningFactory train entry not found: {train_entry}")

    if parsed.num_gpus > 1:
        command: list[str] = [
            parsed.torchrun_bin,
            "--nproc_per_node",
            str(parsed.num_gpus),
            "--master_port",
            str(parsed.master_port),
            str(train_entry),
        ]
    else:
        command = [parsed.python_bin, str(train_entry)]

    command.extend(build_tuningfactory_sft_args(parsed, dataset))
    return command


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch AgenQA SFT with TuningFactory using exported Phase 1 datasets.")
    parser.add_argument("--config", default=None, help="Optional JSON config file for launcher defaults.")
    parser.add_argument("--dataset-root", required=True, help="Exporter output root containing tuningfactory/*.json and manifests/*.json.")
    parser.add_argument("--model-name-or-path", required=True, help="Base model path or repo id.")
    parser.add_argument("--template", required=True, help="TuningFactory prompt template name, e.g. qwen2_5 or llama3.")
    parser.add_argument("--output-dir", required=True, help="Training output directory.")

    parser.add_argument("--tuningfactory-root", default=str(DEFAULT_TUNINGFACTORY_ROOT), help="Local TuningFactory repo root.")
    parser.add_argument("--python-bin", default=default_python_bin(), help="Python executable used for single-GPU/local launch.")
    parser.add_argument("--torchrun-bin", default=default_torchrun_bin(), help="torchrun executable used for multi-GPU launch.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of local GPUs. Uses torchrun when > 1.")
    parser.add_argument("--master-port", type=int, default=29501, help="torchrun master port for multi-GPU launches.")

    parser.add_argument("--cutoff-len", type=int, default=2048, help="Max sequence length after tokenization.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Training epochs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples for debugging.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging interval.")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint save interval.")
    parser.add_argument("--eval-steps", type=int, default=100, help="Eval interval when eval split exists.")
    parser.add_argument("--lr-scheduler-type", default="cosine", help="HF scheduler type.")
    parser.add_argument("--preprocessing-num-workers", type=int, default=8, help="Dataset preprocessing workers.")
    parser.add_argument("--dataloader-num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--precision", choices=("bf16", "fp16", "none"), default="bf16", help="Mixed precision mode.")
    parser.add_argument("--report-to", default="none", help="Trainer report_to target. Local default is none.")

    parser.add_argument("--lora-target", default="all", help="LoRA target modules.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")

    parser.add_argument("--train-on-prompt", action=argparse.BooleanOptionalAction, default=False, help="Whether prompt tokens contribute to loss.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable reasoning-thought mode in TuningFactory.")
    parser.add_argument("--overwrite-cache", action=argparse.BooleanOptionalAction, default=True, help="Whether to overwrite cached processed data.")
    parser.add_argument("--overwrite-output-dir", action=argparse.BooleanOptionalAction, default=True, help="Whether to overwrite training output dir.")
    parser.add_argument("--plot-loss", action=argparse.BooleanOptionalAction, default=True, help="Whether to save trainer loss curves.")
    parser.add_argument("--tokenized-path", default=None, help="Optional tokenized dataset cache path.")

    parser.add_argument("--run", action="store_true", help="Execute the command. By default only prints the command and writes run spec.")

    if defaults:
        parser.set_defaults(**defaults)

    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    known, _ = pre_parser.parse_known_args()
    defaults = load_config_overrides(known.config)
    parser = build_parser(defaults=defaults)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = resolve_exported_dataset(Path(args.dataset_root))
    command = build_sft_command(args, dataset)
    config = {
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "precision": args.precision,
        "num_gpus": args.num_gpus,
        "cutoff_len": args.cutoff_len,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lora_target": args.lora_target,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "train_on_prompt": args.train_on_prompt,
        "enable_thinking": args.enable_thinking,
        "config_path": args.config,
    }
    run_spec_path = write_run_spec(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        command=command,
        dataset=dataset,
        config=config,
        launcher="local",
    )

    print(" ".join(command))
    print(f"run_spec={run_spec_path}")

    if args.run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
