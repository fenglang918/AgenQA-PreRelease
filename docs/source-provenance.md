# Runtime Source Provenance

The public runtime was synchronized from the AgenQA repository at commit:

```text
8e5e4965e13481d417d182cc768ef157502e5d05
refactor(paper): make AAAI-27 the sole manuscript entry
```

The complete `src/agenqa` Python runtime is present: domain models, all prompt
sources, role runners, nodes, graph orchestration, memory, evaluation, and
downstream exporters. The required `infra` Python runtime was moved under the
standard `src/` package layout without changing its module names.

Public portability changes are limited to:

1. replace cluster-only default file paths with environment variables or
   repository-relative paths;
2. provide `config/agent_openai.yaml`, `agenqa-demo`, and standard Python package
   metadata;
3. point four executable prompt modules at the already-canonical in-repo
   `agenqa.prompts.common` definitions instead of the stale external import;
4. make the Responses API client accept legacy `max_tokens`, emit
   `max_output_tokens`, surface HTTP errors, and find visible text after a
   reasoning output item;
5. add the synthetic PDF input and public portability tests.

Internal provider reference documents, model-list snapshots, service
configuration files, credentials, raw runs, private datasets, and manuscript
worktrees are not part of this distribution. Their absence does not replace or
simplify the AgenQA graph runtime.
