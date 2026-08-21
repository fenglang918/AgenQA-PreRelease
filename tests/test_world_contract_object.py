from agenqa.domain.known_tree import KnownTree
from agenqa.skills.draft_chain import DraftChainRunner


def test_known_tree_normalize_memory_adds_world_contract_default():
    mem = KnownTree.normalize_memory({})
    wc = mem.get("world_contract")
    assert isinstance(wc, dict)
    assert wc.get("schema_version") == 1
    assert wc.get("status") == "underdetermined"
    sections = wc.get("sections")
    assert isinstance(sections, list)
    assert [s.get("level") for s in sections] == ["L1", "L2", "L3", "L4"]
    for s in sections:
        assert isinstance(s, dict)
        assert isinstance(s.get("points"), list)


def test_draft_chain_parse_world_contract_optional():
    # Avoid constructing DraftChainRunner (it would resolve an LLM session).
    runner = DraftChainRunner.__new__(DraftChainRunner)
    text = """```json
{
  "subtasks": [{"id":"sub_prev","description":"x","result":"y"},{"id":"sub_step","description":"a","result":"b"}],
  "final_subtask_id": "sub_step",
  "dependencies": {"sub_prev": [], "sub_step": ["sub_prev"]},
  "draft_question_explicit": "Q",
  "draft_solution_outline": "S",
  "draft_answer": "\\\\boxed{1}",
  "required_fact_ids": ["F1"],
  "primary_required_fact_id": "F1",
  "reuse_plan": ["use F1"],
  "world_contract": {"sections":[{"level":"L1","points":[{"axis":"paradigm_id","choice":"p1"}]},{"level":"L3","points":[{"axis":"a","choice":true}]}]}
}
```"""
    out = DraftChainRunner._parse_output(runner, text)
    assert out.world_contract == {
        "sections": [
            {"level": "L1", "points": [{"axis": "paradigm_id", "choice": "p1"}]},
            {"level": "L3", "points": [{"axis": "a", "choice": True}]},
        ]
    }

    # Missing field should parse as None (optional).
    text2 = """```json
{
  "subtasks": [],
  "final_subtask_id": "",
  "dependencies": {},
  "draft_question_explicit": "Q",
  "draft_solution_outline": "S",
  "draft_answer": "\\\\boxed{1}",
  "required_fact_ids": [],
  "primary_required_fact_id": "",
  "reuse_plan": []
}
```"""
    out2 = DraftChainRunner._parse_output(runner, text2)
    assert out2.world_contract is None
