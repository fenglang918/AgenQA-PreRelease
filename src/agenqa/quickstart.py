"""One-command entry point for the real AgenQA runtime and bundled demo paper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _repo_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "config" / "agent_openai.yaml").is_file():
        return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "config" / "agent_openai.yaml").is_file():
        return cwd
    raise SystemExit(
        "Cannot locate config/agent_openai.yaml. Run agenqa-demo from the cloned AgenQA repository."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete AgenQA agent pipeline on a paper PDF."
    )
    parser.add_argument("--pdf", help="Paper PDF path; defaults to the bundled synthetic paper.")
    parser.add_argument("--output", default="outputs/quickstart", help="Base output directory.")
    parser.add_argument("--model", default=os.getenv("AGENQA_MODEL", "gpt-5-mini"))
    parser.add_argument("--medium-model", default=os.getenv("AGENQA_MEDIUM_MODEL"))
    parser.add_argument("--strong-model", default=os.getenv("AGENQA_STRONG_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("AGENQA_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument(
        "--api-style",
        choices=["responses", "chat"],
        default=os.getenv("AGENQA_API_STYLE", "responses"),
        help="Use responses for OpenAI; use chat for compatible gateways that expose chat/completions.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="State step limit. The initial state is step 0, so 3 yields up to two QA steps.",
    )
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--no-playback", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "replace-me":
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it in your shell; do not pass secrets as CLI arguments."
        )

    root = _repo_root()
    paper = Path(args.pdf).expanduser() if args.pdf else root / "examples/papers/layered_thermal_transport_demo.pdf"
    if not paper.is_absolute():
        paper = (Path.cwd() / paper).resolve()
    if not paper.is_file():
        raise SystemExit(f"Paper PDF does not exist: {paper}")

    os.environ["AGENQA_API_BASE"] = str(args.base_url).strip()
    os.environ["AGENQA_API_STYLE"] = str(args.api_style).strip()
    os.environ["AGENQA_MODEL"] = str(args.model).strip()
    os.environ["AGENQA_MEDIUM_MODEL"] = str(args.medium_model or args.model).strip()
    os.environ["AGENQA_STRONG_MODEL"] = str(args.strong_model or args.model).strip()
    os.environ["SCICLONE_PAPER_PATH"] = str(paper)

    config = root / "config/agent_openai.yaml"
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    cli_args = [
        "agenqa",
        "--config",
        str(config),
        "agent-run",
        "--paper-path",
        str(paper),
        "--output",
        str(output),
        "--max-steps",
        str(args.max_steps),
        "--max-rounds",
        str(args.max_rounds),
        "--lang",
        str(args.lang),
    ]
    if args.no_playback:
        cli_args.append("--no-playback")

    print("AgenQA full-runtime demo")
    print(f"  paper: {paper}")
    print(f"  endpoint: {args.base_url}")
    print(f"  model: {args.model}")
    print(f"  output: {output}")
    print("  API key: loaded from OPENAI_API_KEY (value hidden)")

    from cli import main as cli_main

    previous_argv = sys.argv
    try:
        sys.argv = cli_args
        cli_main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
