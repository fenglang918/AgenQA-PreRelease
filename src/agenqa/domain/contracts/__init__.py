"""Domain-level contract mechanisms (schemas + governance logic).

This package is the single namespace for AgenQA contract-related artifacts:
- Episode-seed YAML schemas (static resources under schemas/)
- Type1 world contract (world_contract.py)
- Type2 answer contract bank (answer_contract_bank.py)

Visibility model (important):
    Both contract types have a dual presence:

    1) **Structured data** (JSON in KnownTree memory):
       Internal-only governance — NOT injected into solver's KnownTree view
       (``visible_to_path: False``). Used by Director, Revise, and validation
       logic for consistency checks, merge operations, and fail-fast detection.

    2) **Solver-visible projection** (kept separate from the core question body):
       - ``world_contract`` → rendered into a separate ``world_contract_text`` block
         and concatenated with ``question`` only at solver-consumption time.
       - ``answer_contract`` → remains raw/internal in ACB, but its public output-spec
         slice may be projected into ``world_contract_text`` under L4.

    This keeps governance data structured and auditable while still giving solvers
    the necessary visible constraints without duplicating them inside the question body.
"""

from .answer_contract_bank import (
    ACB_ID_PREFIX,
    ANSWER_CONTRACT_MODEL_TEXT_EN,
    ANSWER_CONTRACT_MODEL_TEXT_ZH,
    answer_contract_model_text,
    build_answer_contract_validation_background,
    build_answer_output_spec_prompt_section,
    extract_answer_contract_context,
    extract_answer_output_spec_context,
    make_default_answer_contracts,
    persist_answer_contracts,
    validate_answer_contracts,
)
from .world_contract import (
    WORLD_CONTRACT_LEVELS,
    WORLD_CONTRACT_MODEL_TEXT_EN,
    WORLD_CONTRACT_MODEL_TEXT_ZH,
    WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_EN,
    WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_ZH,
    WORLD_CONTRACT_SCHEMA_VERSION,
    empty_world_contract,
    get_paradigm_id,
    merge_world_contract,
    normalize_world_contract,
    world_contract_model_text,
    world_contract_prompt_guidance_text,
)

__all__ = [
    "ACB_ID_PREFIX",
    "ANSWER_CONTRACT_MODEL_TEXT_ZH",
    "ANSWER_CONTRACT_MODEL_TEXT_EN",
    "answer_contract_model_text",
    "make_default_answer_contracts",
    "validate_answer_contracts",
    "persist_answer_contracts",
    "build_answer_contract_validation_background",
    "build_answer_output_spec_prompt_section",
    "extract_answer_contract_context",
    "extract_answer_output_spec_context",
    "WORLD_CONTRACT_SCHEMA_VERSION",
    "WORLD_CONTRACT_LEVELS",
    "WORLD_CONTRACT_MODEL_TEXT_ZH",
    "WORLD_CONTRACT_MODEL_TEXT_EN",
    "WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_ZH",
    "WORLD_CONTRACT_PROMPT_GUIDANCE_TEXT_EN",
    "empty_world_contract",
    "get_paradigm_id",
    "normalize_world_contract",
    "merge_world_contract",
    "world_contract_model_text",
    "world_contract_prompt_guidance_text",
]
