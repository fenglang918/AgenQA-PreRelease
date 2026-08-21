# AgenQA Runtime

This directory contains the complete AgenQA runtime snapshot selected from
commit `8e5e4965e13481d417d182cc768ef157502e5d05`:

- `domain/`: KnownTree, Chain-of-KQA state, World Contract, Answer Contract Bank,
  and role/evaluator schemas;
- `prompts/`: the Python prompt source of truth for every semantic and
  executable role;
- `skills/`: init, draft, format, certificate, Path-Fold, solve, and tool-use
  runners;
- `nodes/` and `graph/`: Director--Operator--Evaluator orchestration, revision,
  consensus, judging, stopping, persistence, and playback hooks;
- `memory/`, `evaluation/`, and `downstream/`: artifact views, audits, and
  SFT-export utilities.

The accompanying `src/infra/` package contains the actual HTTP inference
client, PDF loader, prompt tracking, output I/O, playback, and optional code
verifier used by this runtime.

Public-release changes are deliberately narrow: cluster-only default paths were
replaced by environment variables or repository-relative paths; four stale
executable-prompt imports now point to the in-repo prompt contracts; and the
Responses API adapter now handles reasoning-first output and
`max_output_tokens` correctly. The orchestration and role logic are the real
project implementation, not a separate demo pipeline.

See [the quickstart](../../docs/quickstart.md) for an end-to-end PDF run.
