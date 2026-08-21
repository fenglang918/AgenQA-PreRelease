import agenqa.nodes.roles_nodes as roles_nodes


def test_strip_trailing_answer_format_paragraph_zh() -> None:
    text = "题目正文。\n\n答案格式：旧要求。"
    assert roles_nodes._strip_trailing_answer_format_paragraph(text) == "题目正文。"


def test_strip_trailing_answer_format_paragraph_en() -> None:
    text = "Question body.\n\nAnswer format: old requirement."
    assert roles_nodes._strip_trailing_answer_format_paragraph(text) == "Question body."


def test_strip_solver_contract_keeps_core_question() -> None:
    text = "题目正文。\n\n答案格式：旧要求。"
    assert roles_nodes._strip_solver_contract_from_question(text) == "题目正文。"
