import unittest

import agenqa.nodes.roles_nodes as roles_nodes


class TestSymbolicConstraintsPerQType(unittest.TestCase):
    def test_derivation_constraints_are_symbolic_only(self) -> None:
        base = "BASE"
        out = roles_nodes._append_symbolic_constraints(base, use_en=False, question_type="Derivation")
        self.assertIn("【符号表达式 ONLY 约束】", out)
        self.assertIn("符号解析式", out)
        self.assertNotIn("【MCQ 禁数值求值口径约束】", out)

    def test_mcq_constraints_do_not_require_symbolic_expression_answer(self) -> None:
        base = "BASE"
        out = roles_nodes._append_symbolic_constraints(base, use_en=False, question_type="MCQ")
        self.assertIn("【MCQ 禁数值求值口径约束】", out)
        self.assertIn("选项字母", out)
        self.assertNotIn("符号解析式", out)
        # MCQ 允许题干出现数值，不应被“禁止具体数值”的约束误伤
        self.assertNotIn("禁止给出具体数值", out)


if __name__ == "__main__":
    unittest.main()
