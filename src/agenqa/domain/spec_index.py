"""Spec blocks index: centralized registry of all specification text blocks.

This module provides a code-based index of all specification blocks used in prompts.
Each block points to its source (code truth) and usage locations (prompt fragments).

Note: This index is for navigation only. The actual text content remains in the
source modules (known_tree.py, solver_schema.py, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# 类型定义
SpecKind = Literal[
    "structure",      # 结构说明（如 Known Tree）
    "schema",         # 输出字段 schema（如 Solver, Draft）
    "metrics",        # Metrics 和 solver 信号说明
    "constraints",    # 角色约束（如 Extend/Revise 约束）
    "format",         # 格式规范（如答案格式、题型定义）
]

@dataclass(frozen=True)
class SpecBlock:
    """A specification block entry in the index.

    Attributes:
        id: Unique identifier (e.g., "known_tree", "solver_output")
        kind: Category of the spec block
        source: Module path to the source (e.g., "agenqa.domain.known_tree.KNOWN_TREE_DESCRIPTION")
        prompt_refs: List of prompt fragments that use this block (e.g., ["agenqa.prompts.common.COMMON_KNOWN_TREE"])
        description: Brief description of what this block contains
    """
    id: str
    kind: SpecKind
    source: str
    prompt_refs: list[str]
    description: str = ""


# 索引列表
SPEC_BLOCKS: list[SpecBlock] = [
    SpecBlock(
        id="known_tree",
        kind="structure",
        source="agenqa.domain.known_tree.KNOWN_TREE_DESCRIPTION",
        prompt_refs=["agenqa.prompts.common.COMMON_KNOWN_TREE"],
        description="KnownTree v2 结构说明：episode_seed（出题侧元信息）、premise_bank（solver 可见前提）、fact_bank/step_certs（edge-only 复用结论/证据）与可见性约定",
    ),
    SpecBlock(
        id="solver_output",
        kind="schema",
        source="agenqa.domain.solver_schema.solver_output_schema_text()",
        prompt_refs=["agenqa.prompts.solver._SOL_JSON"],
        description="Solver 输出字段说明：Answer, QuestionWellPosed, CorrectnessFeedback, DifficultyFeedback 等",
    ),
    SpecBlock(
        id="solver_tool_output",
        kind="schema",
        source="agenqa.domain.solver_tool_schema.solver_tool_output_schema_text()",
        prompt_refs=["agenqa.prompts.solver_tool._JSON_FORMAT"],
        description="SolverTool 输出字段说明：ToolUsed/ToolName/ToolCode/ToolNotes（用于 Numeric 的工具辅助求解/验证）",
    ),
    SpecBlock(
        id="draft_output",
        kind="schema",
        source="agenqa.domain.draft_schema.draft_output_schema_text()",
        prompt_refs=["agenqa.prompts.draft.DRAFT_V1", "agenqa.prompts.draft.DRAFT_REVISE_CORRECTNESS", "agenqa.prompts.draft.DRAFT_REVISE_DIFFICULTY"],
        description="Draft 输出字段说明（legacy）：draft_question, draft_solution, draft_answer, draft_background 等",
    ),
    SpecBlock(
        id="draft_chain_output",
        kind="schema",
        source="agenqa.domain.draft_chain_schema.draft_chain_output_schema_text()",
        prompt_refs=[
            "agenqa.prompts.draft_chain.DRAFT_CHAIN_MCQ",
            "agenqa.prompts.draft_chain.DRAFT_CHAIN_DERIVATION",
            "agenqa.prompts.draft_chain.DRAFT_CHAIN_NUMERIC",
        ],
        description="DraftChain 输出字段说明：draft_question_explicit, required_fact_ids, reuse_plan 等",
    ),
    SpecBlock(
        id="executable_extract_output",
        kind="schema",
        source="agenqa.domain.executable_schema.executable_extract_output_schema_text()",
        prompt_refs=["agenqa.prompts.common.COMMON_EXECUTABLE_EXTRACT_SCHEMA"],
        description="ExecutableExtract 输出字段说明：executable_suitable, notes, task_sketch, initial_sub_steps 等",
    ),
    SpecBlock(
        id="executable_draft_step_output",
        kind="schema",
        source="agenqa.domain.executable_schema.executable_draft_step_output_schema_text()",
        prompt_refs=["agenqa.prompts.common.COMMON_EXECUTABLE_DRAFT_STEP_SCHEMA"],
        description="ExecutableDraftStep 输出字段说明：step_number, sub_step, golden_step_code, dependencies, required_fact_ids, primary_required_fact_id, reuse_plan",
    ),
    SpecBlock(
        id="executable_step_cert_builder_guide",
        kind="constraints",
        source="agenqa.domain.executable_step_cert_schema.EXECUTABLE_CHAIN_CERT_GUIDE_ZH/EN",
        prompt_refs=["agenqa.prompts.executable_step_cert_builder.EXECUTABLE_STEP_CERT_BUILDER_V1"],
        description="ExecutableStepCertBuilder 语义与约束：executable 的 premise/fact/cert 映射与 ID 约定（对齐 semantic KnownTree v2）",
    ),
    SpecBlock(
        id="executable_test_inputs_schema",
        kind="schema",
        source="agenqa.domain.executable_schema.executable_test_inputs_schema_text()",
        prompt_refs=["agenqa.prompts.common.COMMON_EXECUTABLE_TEST_INPUTS_SCHEMA"],
        description="Executable Test Inputs 输出字段说明：test_inputs 数组（args/kwargs）",
    ),
    SpecBlock(
        id="format_output",
        kind="schema",
        source="agenqa.domain.format_schema.format_output_schema_text()",
        prompt_refs=["agenqa.prompts.format.FORMAT_V1", "agenqa.prompts.format.FORMAT_V1_TAGGED"],
        description="Format 输出字段说明：Step, Question, Solution, Answer, validation_* 等",
    ),
    SpecBlock(
        id="step_cert_output",
        kind="schema",
        source="agenqa.domain.step_cert_schema.step_cert_output_schema_text()",
        prompt_refs=["agenqa.prompts.step_cert_builder.STEP_CERT_BUILDER_V1"],
        description="StepCertBuilder 输出字段说明：premise_delta, fact_delta, step_cert, key_fact_id",
    ),
    SpecBlock(
        id="diagnose_output",
        kind="schema",
        source="agenqa.domain.diagnose_schema.diagnose_output_schema_text()",
        prompt_refs=[
            "agenqa.prompts.diagnose.DIAGNOSE_V1",
            "agenqa.prompts.diagnose.DIAGNOSE_REVISE_CORRECTNESS",
            "agenqa.prompts.diagnose.DIAGNOSE_REVISE_DIFFICULTY",
            "agenqa.prompts.executable_diagnose.EXECUTABLE_DIAGNOSE_V1",
        ],
        description="Diagnose 输出字段说明：issues, fix_suggestions, diagnosis",
    ),
    SpecBlock(
        id="extend_output",
        kind="schema",
        source="agenqa.domain.extend_schema.extend_output_schema_text()",
        prompt_refs=["agenqa.prompts.extend.EXTEND_UPGRADE_V1"],
        description="Extend 输出字段说明（legacy）：Step, Question, Solution, Answer, NewBackground, DerivedFacts",
    ),
    SpecBlock(
        id="extract_output",
        kind="schema",
        source="agenqa.domain.extract_schema.extract_output_schema_text()",
        prompt_refs=["agenqa.prompts.extract.EXTRACT_V1"],
        description="Extract 输出字段说明：exam_points, chain_potential",
    ),
    SpecBlock(
        id="qa_init_output",
        kind="schema",
        source="agenqa.domain.qa_init_schema.qa_init_output_schema_text()",
        prompt_refs=["agenqa.prompts.qa_init.QA_INIT_V2"],
        description="QA-Init 输出字段说明：Step, Subject, Known, Question, Solution, Answer",
    ),
    SpecBlock(
        id="final_comment_output",
        kind="schema",
        source="agenqa.domain.final_comment_schema.final_comment_output_schema_text()",
        prompt_refs=["agenqa.prompts.final_commenter.FINAL_COMMENTER_TEMPLATE"],
        description="Final Commenter 输出字段说明：well_posed, question_leaks_previous_conclusions, intermediate_steps_necessary, difficulty, evidence, suggestions",
    ),
    SpecBlock(
        id="answer_schema",
        kind="format",
        source="agenqa.prompts.common.COMMON_ANSWER_SCHEMA",
        prompt_refs=[
            "agenqa.prompts.draft.DRAFT_V1",
            "agenqa.prompts.format.FORMAT_V1",
            "agenqa.prompts.qa_init.QA_INIT_V2",
            "agenqa.prompts.solver.SOLVER_TEMPLATE",
        ],
        description="答案格式规范：\\boxed{} 包裹、MCQ/Derivation/Numeric 题型要求",
    ),
    SpecBlock(
        id="question_types",
        kind="format",
        source="agenqa.prompts.common.COMMON_QUESTION_TYPES",
        prompt_refs=[
            "agenqa.prompts.draft.DRAFT_V1",
            "agenqa.prompts.format.FORMAT_V1",
        ],
        description="题型定义：MCQ / Derivation / Numeric 的规范",
    ),
    SpecBlock(
        id="solution_schema",
        kind="format",
        source="agenqa.prompts.common.COMMON_SOLUTION_SCHEMA",
        prompt_refs=[
            "agenqa.prompts.draft.DRAFT_V1",
            "agenqa.prompts.format.FORMAT_V1",
        ],
        description="Solution 规范：S1/S2/S3 格式、head-tail 兼容性要求",
    ),
    SpecBlock(
        id="extend_constraints",
        kind="constraints",
        source="agenqa.prompts.common.COMMON_EXTEND_CONSTRAINTS",
        prompt_refs=[
            "agenqa.prompts.draft.DRAFT_V1",
            "agenqa.prompts.extend.EXTEND_UPGRADE_V1",
        ],
        description="Extend/Revise 通用约束：NewBackground 规则、History 复用规则",
    ),
    SpecBlock(
        id="reused_conclusions_rules",
        kind="constraints",
        source="agenqa.prompts.common.COMMON_REUSED_CONCLUSIONS_RULES",
        prompt_refs=[
            "agenqa.prompts.draft.DRAFT_V1",
            "agenqa.prompts.draft.DRAFT_REVISE_CORRECTNESS",
            "agenqa.prompts.draft.DRAFT_REVISE_DIFFICULTY",
        ],
        description="复用前序结论的强制规则：必须包含上一步结论、不得在题干中直接赠送、等价改写边界",
    ),
    SpecBlock(
        id="reused_refs_rules",
        kind="constraints",
        source="agenqa.prompts.common.COMMON_REUSED_REFS_RULES",
        prompt_refs=[
            "agenqa.prompts.draft.DRAFT_V1",
            "agenqa.prompts.draft.DRAFT_REVISE_CORRECTNESS",
            "agenqa.prompts.draft.DRAFT_REVISE_DIFFICULTY",
            "agenqa.prompts.format.FORMAT_V1",
            "agenqa.prompts.format.FORMAT_V1_TAGGED",
        ],
        description="结构化复用引用的强制规则：reused_refs 必须包含 source_step=i-1；MCQ 时要求 mcq_choice 与上一步一致",
    ),
    SpecBlock(
        id="new_assumptions_rules",
        kind="constraints",
        source="agenqa.prompts.common.COMMON_NEW_ASSUMPTIONS_RULES",
        prompt_refs=[
            "agenqa.prompts.draft.DRAFT_V1",
            "agenqa.prompts.draft.DRAFT_V1_FIRST_STEP",
            "agenqa.prompts.draft.DRAFT_REVISE_CORRECTNESS",
            "agenqa.prompts.draft.DRAFT_REVISE_DIFFICULTY",
        ],
        description="新增设定职责边界：允许前提条件，禁止泄露答案/重复前序结论，并要求与既有 Background 自洽",
    ),
    SpecBlock(
        id="first_step_reuse_rules",
        kind="constraints",
        source="agenqa.prompts.common.COMMON_FIRST_STEP_REUSE_RULES",
        prompt_refs=[
            "agenqa.prompts.draft.DRAFT_V1_FIRST_STEP",
            "agenqa.prompts.draft.DRAFT_V1_TAGGED_FIRST_STEP",
        ],
        description="首题模式复用字段的特殊规则：reused_refs 为空数组、reused_conclusions 为空或说明无可复用",
    ),
    SpecBlock(
        id="metrics_solvers",
        kind="metrics",
        source="agenqa.domain.metrics_schema.METRICS_AND_SOLVERS_DESCRIPTION_ZH",
        prompt_refs=["agenqa.prompts.common.COMMON_METRICS_DESCRIPTION"],
        description="Metrics 与 Solver 信号说明：correct_medium/strong, token_ratio, edge/path 视角的含义",
    ),
    SpecBlock(
        id="edge_qa_vs_path",
        kind="metrics",
        source="agenqa.domain.metrics_schema.EDGE_QA_VS_PATH_DESCRIPTION_ZH",
        prompt_refs=["agenqa.prompts.common.COMMON_EDGE_QA_VS_PATH"],
        description="Edge QA vs Path QA：同一目标、不同视角（Path 为路径折叠题 Q_fold + 仅 premise_bank），以及“Question 禁止显式陈述前序结论”的关键约束",
    ),
    SpecBlock(
        id="solver_consensus",
        kind="metrics",
        source="agenqa.domain.consensus_schema.SOLVER_CONSENSUS_DESCRIPTION_ZH",
        prompt_refs=["agenqa.prompts.common.COMMON_SOLVER_CONSENSUS_DESCRIPTION"],
        description="Multi-strong solver 共识信号说明：answer_consensus/wellposed_consensus/tie 等字段含义与使用建议",
    ),
]


def get_spec_block(block_id: str) -> SpecBlock | None:
    """Get a spec block by ID."""
    for block in SPEC_BLOCKS:
        if block.id == block_id:
            return block
    return None


def list_spec_blocks_by_kind(kind: SpecKind) -> list[SpecBlock]:
    """List all spec blocks of a given kind."""
    return [block for block in SPEC_BLOCKS if block.kind == kind]
