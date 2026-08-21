# Run the Full AgenQA Pipeline

[中文](./quickstart.zh.md)

The public repository ships the complete AgenQA Director--Operator--Evaluator
runtime, a portable API configuration, and a synthetic paper PDF. The demo uses
the same graph, role prompts, KnownTree updates, Path-Fold, solver evaluation,
revision policy, artifact persistence, and playback code as the research
runtime.

## 1. Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

## 2. Configure the key

```bash
export OPENAI_API_KEY="your-key"
```

The key is read from the environment. `agenqa-demo` intentionally has no
command-line key option, so the secret is not exposed in shell history or the
process list.

Optional overrides:

```bash
export AGENQA_MODEL="gpt-5-mini"
export AGENQA_API_BASE="https://api.openai.com/v1"
export AGENQA_API_STYLE="responses"
```

## 3. Run the bundled PDF

```bash
agenqa-demo
```

The bundled input is
[`examples/papers/layered_thermal_transport_demo.pdf`](../examples/papers/layered_thermal_transport_demo.pdf).
It is a two-page synthetic scientific paper authored for this repository.

The default `max_steps=3` starts from state step 0 and produces up to two linked
QA steps. A run makes several model calls per step because it executes the real
init, Director, DraftChain, Format, StepCertBuilder, Path-Fold, medium/strong
solver, judge, and optional revision stages.

## 4. Run another paper

```bash
agenqa-demo --pdf /absolute/path/to/paper.pdf
```

Useful overrides:

```bash
agenqa-demo \
  --pdf /absolute/path/to/paper.pdf \
  --model gpt-5-mini \
  --max-steps 3 \
  --output outputs/my-paper
```

For an OpenAI-compatible gateway that exposes `chat/completions` instead of the
Responses API:

```bash
agenqa-demo \
  --api-style chat \
  --base-url https://gateway.example.com/v1 \
  --model provider-model-name
```

## Outputs

Each run gets a timestamped subdirectory under the selected output directory.
Important artifacts include:

```text
state.json                         final graph state and KnownTree memory
run_config.json                    resolved run configuration
run.log                            execution log
run_playback_*.md                  human-readable trajectory
00_Prompts_Snapshot/               exact role prompts used
round_*/                           Director and operator rounds
  */edge_kqa.json                  local verification view
  */path_kqa.json                  solver-facing folded view
  */answer_contract_report.json    output-contract checks
  */subruns_raw/                   rendered prompts and parsed role outputs
  */solve/                         medium/strong solver and judge artifacts
```

## Direct CLI

The wrapper delegates to the full CLI. The equivalent direct command is:

```bash
agenqa --config config/agent_openai.yaml agent-run \
  --paper-path examples/papers/layered_thermal_transport_demo.pdf \
  --output outputs/manual \
  --max-steps 3 \
  --max-rounds 6 \
  --lang en
```

## PDF notes

The portable configuration uses `pypdf` text-layer extraction. Scanned PDFs
without a usable text layer need an OCR/layout parser configured through
`init.source.pdf_extract`; those heavy optional models are not downloaded by
the default installation.
