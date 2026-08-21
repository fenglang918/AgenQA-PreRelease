from __future__ import annotations

import json
from pathlib import Path

from agenqa.downstream.sft.tuningfactory import launcher_lib
from agenqa.downstream.sft.tuningfactory import run_sft_local
from agenqa.downstream.sft.tuningfactory import run_sft_nebula


def test_resolve_exported_dataset_prefers_split_files(tmp_path: Path) -> None:
    tuningfactory_dir = tmp_path / "tuningfactory"
    manifests_dir = tmp_path / "manifests"
    tuningfactory_dir.mkdir()
    manifests_dir.mkdir()
    (tuningfactory_dir / "all.json").write_text("[]\n", encoding="utf-8")
    (tuningfactory_dir / "train.json").write_text("[]\n", encoding="utf-8")
    (tuningfactory_dir / "eval.json").write_text("[]\n", encoding="utf-8")
    (manifests_dir / "summary.json").write_text("{}\n", encoding="utf-8")

    dataset = launcher_lib.resolve_exported_dataset(tmp_path)

    assert dataset.train_file.endswith("train.json")
    assert dataset.eval_file is not None and dataset.eval_file.endswith("eval.json")
    assert dataset.summary_path.endswith("summary.json")


def test_build_sft_command_contains_phase1_defaults(tmp_path: Path) -> None:
    tf_root = tmp_path / "TuningFactory"
    train_entry = tf_root / "src"
    train_entry.mkdir(parents=True)
    (train_entry / "train_bash.py").write_text("print('stub')\n", encoding="utf-8")

    export_root = tmp_path / "export"
    (export_root / "tuningfactory").mkdir(parents=True)
    (export_root / "manifests").mkdir(parents=True)
    (export_root / "tuningfactory" / "all.json").write_text("[]\n", encoding="utf-8")
    (export_root / "tuningfactory" / "train.json").write_text("[]\n", encoding="utf-8")
    (export_root / "tuningfactory" / "eval.json").write_text("[]\n", encoding="utf-8")

    dataset = run_sft_local.resolve_exported_dataset(export_root)
    parser = run_sft_local.build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            str(export_root),
            "--model-name-or-path",
            "Qwen/Qwen2.5-7B-Instruct",
            "--template",
            "qwen2_5",
            "--output-dir",
            str(tmp_path / "out"),
            "--tuningfactory-root",
            str(tf_root),
            "--python-bin",
            "/usr/bin/python3",
        ]
    )

    command = run_sft_local.build_sft_command(args, dataset)

    joined = " ".join(command)
    assert command[0] == "/usr/bin/python3"
    assert "--file_name" in command and dataset.train_file in command
    assert "--eval_file_name" in command and dataset.eval_file in command
    assert "--prompt" in command and "instruction" in command
    assert "--query" in command and "input" in command
    assert "--response" in command and "output" in command
    assert "--enable_thinking" in command and "False" in command
    assert "--train_on_prompt" in command and "False" in command
    assert "--finetuning_type" in command and "lora" in command
    assert "--template" in command and "qwen2_5" in command
    assert "--do_eval" in command
    assert "--bf16" in command
    assert "train_bash.py" in joined


def test_build_sft_command_uses_torchrun_for_multi_gpu(tmp_path: Path) -> None:
    tf_root = tmp_path / "TuningFactory"
    train_entry = tf_root / "src"
    train_entry.mkdir(parents=True)
    (train_entry / "train_bash.py").write_text("print('stub')\n", encoding="utf-8")

    export_root = tmp_path / "export"
    (export_root / "tuningfactory").mkdir(parents=True)
    (export_root / "manifests").mkdir(parents=True)
    (export_root / "tuningfactory" / "all.json").write_text("[]\n", encoding="utf-8")

    dataset = run_sft_local.resolve_exported_dataset(export_root)
    parser = run_sft_local.build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            str(export_root),
            "--model-name-or-path",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "--template",
            "llama3",
            "--output-dir",
            str(tmp_path / "out"),
            "--tuningfactory-root",
            str(tf_root),
            "--num-gpus",
            "2",
            "--torchrun-bin",
            "/usr/bin/torchrun",
        ]
    )

    command = run_sft_local.build_sft_command(args, dataset)

    assert command[0] == "/usr/bin/torchrun"
    assert command[1:5] == ["--nproc_per_node", "2", "--master_port", "29501"]


def test_build_nebula_command_wraps_user_params(tmp_path: Path) -> None:
    tf_root = tmp_path / "TuningFactory"
    tf_src = tf_root / "src"
    tf_scripts = tf_root / "scripts"
    tf_src.mkdir(parents=True)
    tf_scripts.mkdir(parents=True)
    (tf_src / "train_bash.py").write_text("print('stub')\n", encoding="utf-8")
    (tf_scripts / "cluster.json").write_text("{}\n", encoding="utf-8")

    export_root = tmp_path / "export"
    (export_root / "tuningfactory").mkdir(parents=True)
    (export_root / "manifests").mkdir(parents=True)
    (export_root / "tuningfactory" / "all.json").write_text("[]\n", encoding="utf-8")
    (export_root / "tuningfactory" / "train.json").write_text("[]\n", encoding="utf-8")

    dataset = run_sft_nebula.resolve_exported_dataset(export_root)
    parser = run_sft_nebula.build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            str(export_root),
            "--model-name-or-path",
            "Qwen/Qwen2.5-7B-Instruct",
            "--template",
            "qwen2_5",
            "--output-dir",
            str(tmp_path / "out"),
            "--queue",
            "nebula_default",
            "--save-model",
            "project.model/version=v1",
            "--tuningfactory-root",
            str(tf_root),
            "--cluster-file",
            str(tf_scripts / "cluster.json"),
            "--nebulactl-bin",
            "/usr/bin/nebulactl",
        ]
    )

    command = run_sft_nebula.build_nebula_command(args, dataset)
    joined = " ".join(command)

    assert command[0] == "/usr/bin/nebulactl"
    assert "run" in command and "mdl" in command
    assert any(part.startswith("--user_params=") for part in command)
    assert any(part == "--queue=nebula_default" for part in command)
    assert any(part == "--worker_count=4" for part in command)
    assert any(part == "--algo_name=pytorch260" for part in command)
    assert "instruction" in joined and "output" in joined


def test_load_config_overrides_supports_json_templates(tmp_path: Path) -> None:
    config_path = tmp_path / "phase1.json"
    payload = {
        "dataset_root": "/tmp/export",
        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "template": "qwen2_5",
        "output_dir": "/tmp/out",
        "enable_thinking": False,
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    loaded = launcher_lib.load_config_overrides(str(config_path))

    assert loaded["template"] == "qwen2_5"
    assert loaded["enable_thinking"] is False
