from agenqa.domain.contracts.answer_contract_bank import make_default_answer_contracts
from agenqa.domain.contracts.world_contract import empty_world_contract
from agenqa.domain.known_tree import KnownTree
from agenqa.prompts.director import DIRECTOR_TEMPLATE
from agenqa.prompts.extend import EXTEND_UPGRADE_V1
from agenqa.prompts.format import FORMAT_V1
from agenqa.prompts.path_fold import PATH_FOLD_V1
from agenqa.prompts.revise import REVISE_V1
from agenqa.prompts.step_cert_builder import STEP_CERT_BUILDER_V1


def test_public_contract_surface_imports() -> None:
    assert KnownTree.normalize_memory({})["world_contract"] == empty_world_contract()
    ids, contracts = make_default_answer_contracts(
        step=1,
        question_type="MCQ",
        question="A. one\nB. two",
        answer="\\boxed{A}",
    )
    assert ids
    assert contracts


def test_public_prompt_surface_is_renderable() -> None:
    prompt_objects = (
        DIRECTOR_TEMPLATE,
        EXTEND_UPGRADE_V1,
        FORMAT_V1,
        PATH_FOLD_V1,
        REVISE_V1,
        STEP_CERT_BUILDER_V1,
    )
    assert all(obj is not None for obj in prompt_objects)
