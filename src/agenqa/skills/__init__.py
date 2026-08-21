"""Skill layer (formerly pipeline.grow).

This package historically re-exported many runners/configs at import-time, but
that created import-time cycles (e.g., JSON policy → optional cleaner → skills
package import). We keep the public re-exports via lazy attribute resolution.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DraftChainConfig",
    "DraftChainInput",
    "DraftChainOutput",
    "DraftChainRunner",
    "FormatConfig",
    "FormatInput",
    "FormatOutput",
    "FormatRunner",
    "SolverConfig",
    "SolverRunner",
    "ExtractConfig",
    "ExtractInput",
    "ExtractOutput",
    "ExtractOutputV2",
    "ExtractRunner",
    "ExtractRunnerV2",
    "ExamPointV2",
    "DiagnoseConfig",
    "DiagnoseInput",
    "DiagnoseRunner",
    "BaseSkillRunner",
    "HeadTailComposer",
    "HeadTailConfig",
    "PaperBriefConfig",
    "PaperBriefRunner",
    "render_brief_text",
    "EpisodeSeedBuilderConfig",
    "EpisodeSeedBuilderRunner",
    "StepCertConfig",
    "StepCertInput",
    "StepCertOutput",
    "StepCertBuilderRunner",
    "KQANode",
    "read_nodes",
    "collect_chain",
    "compose_known",
    "verify_known_materialization",
    "head_tail_view",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name == "BaseSkillRunner":
        from .base import BaseSkillRunner

        return BaseSkillRunner

    if name in {"DraftChainConfig", "DraftChainInput", "DraftChainOutput", "DraftChainRunner"}:
        from .draft_chain import DraftChainConfig, DraftChainInput, DraftChainOutput, DraftChainRunner

        return {
            "DraftChainConfig": DraftChainConfig,
            "DraftChainInput": DraftChainInput,
            "DraftChainOutput": DraftChainOutput,
            "DraftChainRunner": DraftChainRunner,
        }[name]

    if name in {"FormatConfig", "FormatInput", "FormatOutput", "FormatRunner"}:
        from .formatting import FormatConfig, FormatInput, FormatOutput, FormatRunner

        return {
            "FormatConfig": FormatConfig,
            "FormatInput": FormatInput,
            "FormatOutput": FormatOutput,
            "FormatRunner": FormatRunner,
        }[name]

    if name in {"SolverConfig", "SolverRunner"}:
        from .solving import SolverConfig, SolverRunner

        return {"SolverConfig": SolverConfig, "SolverRunner": SolverRunner}[name]

    if name in {
        "ExtractConfig",
        "ExtractInput",
        "ExtractOutput",
        "ExtractOutputV2",
        "ExtractRunner",
        "ExtractRunnerV2",
        "ExamPointV2",
    }:
        from .extracting import (
            ExtractConfig,
            ExtractInput,
            ExtractOutput,
            ExtractOutputV2,
            ExtractRunner,
            ExtractRunnerV2,
            ExamPointV2,
        )

        return {
            "ExtractConfig": ExtractConfig,
            "ExtractInput": ExtractInput,
            "ExtractOutput": ExtractOutput,
            "ExtractOutputV2": ExtractOutputV2,
            "ExtractRunner": ExtractRunner,
            "ExtractRunnerV2": ExtractRunnerV2,
            "ExamPointV2": ExamPointV2,
        }[name]

    if name in {"DiagnoseConfig", "DiagnoseInput", "DiagnoseRunner"}:
        from .diagnosing import DiagnoseConfig, DiagnoseInput, DiagnoseRunner

        return {"DiagnoseConfig": DiagnoseConfig, "DiagnoseInput": DiagnoseInput, "DiagnoseRunner": DiagnoseRunner}[name]

    if name in {"HeadTailComposer", "HeadTailConfig"}:
        from .head_tail import HeadTailComposer, HeadTailConfig

        return {"HeadTailComposer": HeadTailComposer, "HeadTailConfig": HeadTailConfig}[name]

    if name in {"PaperBriefConfig", "PaperBriefRunner", "render_brief_text"}:
        from .paper_brief import PaperBriefConfig, PaperBriefRunner, render_brief_text

        return {"PaperBriefConfig": PaperBriefConfig, "PaperBriefRunner": PaperBriefRunner, "render_brief_text": render_brief_text}[name]

    if name in {"EpisodeSeedBuilderConfig", "EpisodeSeedBuilderRunner"}:
        from .episode_seed_builder import EpisodeSeedBuilderConfig, EpisodeSeedBuilderRunner

        return {"EpisodeSeedBuilderConfig": EpisodeSeedBuilderConfig, "EpisodeSeedBuilderRunner": EpisodeSeedBuilderRunner}[name]

    if name in {"StepCertConfig", "StepCertInput", "StepCertOutput", "StepCertBuilderRunner"}:
        from .step_cert_builder import StepCertConfig, StepCertInput, StepCertOutput, StepCertBuilderRunner

        return {
            "StepCertConfig": StepCertConfig,
            "StepCertInput": StepCertInput,
            "StepCertOutput": StepCertOutput,
            "StepCertBuilderRunner": StepCertBuilderRunner,
        }[name]

    if name in {"KQANode", "read_nodes", "collect_chain", "compose_known", "verify_known_materialization", "head_tail_view"}:
        from .chain_utils import (
            KQANode,
            read_nodes,
            collect_chain,
            compose_known,
            verify_known_materialization,
            head_tail_view,
        )

        return {
            "KQANode": KQANode,
            "read_nodes": read_nodes,
            "collect_chain": collect_chain,
            "compose_known": compose_known,
            "verify_known_materialization": verify_known_materialization,
            "head_tail_view": head_tail_view,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
