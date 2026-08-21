"""Evaluator node package.

Keep this package `__init__` dependency-light so that importing a specific
evaluator (e.g. executable verifier smoke) does not require the full LLM stack.

Import evaluators via their concrete modules:
- `agenqa.nodes.evaluators.solve`
- `agenqa.nodes.evaluators.code_solve`
- `agenqa.nodes.evaluators.final_commenter`
"""

__all__: list[str] = []
