"""Extract 角色：从论文摘要中提炼考点（QA‑Init 前置模块）。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used
from infra.llm.service_client import LLMServiceSession
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from agenqa.domain.extract_schema import (
    FIELD_EXAM_POINTS,
    FIELD_CHAIN_POTENTIAL,
    extract_output_to_dict,
)

logger = logging.getLogger(__name__)

def _render_prompt_with_fallback(
    *,
    prompt_template: Template,
    payload: Dict[str, Any],
    prompt_path: Any,
    required_keys: Optional[Iterable[str]] = None,
) -> str:
    """Render prompt with fail-fast checks.

    Supports both `$var` (string.Template) and legacy `{var}` placeholders.
    """
    body = prompt_template.safe_substitute(payload)

    # Backward-compatible: some prompt files accidentally used `{var}` placeholders.
    for key, value in payload.items():
        body = body.replace(f"{{{key}}}", str(value))

    keys_to_check = list(required_keys) if required_keys is not None else list(payload.keys())
    unresolved: List[str] = []
    for key in keys_to_check:
        if f"{{{key}}}" in body or f"${{{key}}}" in body or f"${key}" in body:
            unresolved.append(key)
    if unresolved:
        raise ValueError(
            f"Unresolved prompt placeholders: {unresolved}. "
            f"Check prompt template syntax in {prompt_path} (expected `$var` or `${{var}}`)."
        )
    return body


@dataclass
class ExtractInput:
    paper_brief_json: Dict[str, Any]
    paper_brief_text: str


@dataclass
class ExtractOutput:
    exam_points: List[str]
    chain_potential: str


@dataclass
class ExamPointV2:
    """V2 版本的考点结构（结构化）。"""
    id: str
    layer: str  # Premise/Assumption/Condition/Justification/ReasoningStep/Boundary/Goal
    point: str
    exam_style: str  # explanation/derivation
    transform_hint: str
    assessment: str
    style: str


@dataclass
class ExtractOutputV2:
    """V2 版本的 Extract 输出（结构化 exam_points）。"""
    exam_points: List[ExamPointV2]
    chain_potential: str


@dataclass
class ExtractConfig:
    generator: Dict[str, Any]
    prompt_path: Any  # Path-like
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


class ExtractRunner:
    """给定 PaperBrief，调用 LLM 提取考点规划信息。"""

    def __init__(self, config: ExtractConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        base_text = (
            config.prompt_text
            if getattr(config, "prompt_text", None)
            else config.prompt_path.read_text(encoding="utf-8")
        )
        self.prompt_text: str = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, config.prompt_path)

        try:
            logger.info(
                "ExtractRunner configured: model=%s service_id=%s chat_args=%s",
                self.session.model_name,
                self.session.service_id,
                json.dumps(self._chat_args, ensure_ascii=False),
            )
        except Exception:
            pass

    def _build_prompt(self, ext_in: ExtractInput) -> str:
        lang = (self.config.lang or "zh").lower()
        brief_json_str = json.dumps(ext_in.paper_brief_json, ensure_ascii=False, indent=2)
        payload: Dict[str, Any] = {
            "paper_brief_json": brief_json_str,
            "paper_brief_text": ext_in.paper_brief_text,
        }
        body = _render_prompt_with_fallback(
            prompt_template=self.prompt_template,
            payload=payload,
            prompt_path=self.config.prompt_path,
            required_keys=("paper_brief_json", "paper_brief_text"),
        )
        if lang in {"en", "english"}:
            header = f"Role: Extract | Mode: QA-Init | Language: {lang}"
        else:
            header = f"【Extract 角色 · QA-Init】语言: {lang}"
        return "\n".join(
            [
                header,
                "",
                body,
            ]
        )

    def _parse_output(self, text: str) -> ExtractOutput:
        if not text:
            raise ValueError("Extract output is empty")
        candidate = text.strip()
        think_end = "</think>"
        if think_end in candidate:
            candidate = candidate.split(think_end, 1)[1].strip()

        fenced = extract_preferred_fenced_block(candidate, preferred_langs=("json",))
        if fenced:
            candidate = fenced

        try:
            data = json.loads(candidate)
        except Exception as e:  # noqa: BLE001
            logger.error("Extract JSON parse failed: %s", str(e))
            logger.error("Output preview: %s", (candidate or "")[:300])
            raise ValueError(f"Invalid Extract output format: {e}") from e
        if not isinstance(data, dict):
            logger.error("Extract output is not a JSON object: type=%s", type(data).__name__)
            raise ValueError("Invalid Extract output format: not a JSON object")

        def _as_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (str, int, float, bool)):
                return str(val)
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("Extract: failed to JSON-encode value, using str(): %s", str(e))
                return str(val)

        def _as_list(val: Any) -> List[str]:
            items: List[str] = []
            if val is None:
                return items
            if isinstance(val, list):
                for v in val:
                    s = _as_str(v).strip()
                    if s:
                        items.append(s)
                return items
            s = _as_str(val).strip()
            if s:
                items.append(s)
            return items

        exam_points = _as_list(data.get(FIELD_EXAM_POINTS))
        chain_potential = _as_str(data.get(FIELD_CHAIN_POTENTIAL)).strip()

        if not (exam_points and chain_potential):
            logger.error("Extract JSON missing required fields: exam_points/chain_potential")
            raise ValueError("Invalid Extract output: missing required fields")

        return ExtractOutput(
            exam_points=exam_points,
            chain_potential=chain_potential,
        )

    def run_one(
        self,
        ext_in: ExtractInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
    ) -> Optional[ExtractOutput]:
        prompt_body = self._build_prompt(ext_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "Extract")
        text = self.session.extract_text(response, default="")

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                (snap_path / "prompt.txt").write_text(prompt_body, encoding="utf-8")
                try:
                    (snap_path / "response.json").write_text(
                        json.dumps(response, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    (snap_path / "response.txt").write_text(str(response), encoding="utf-8")
                (snap_path / "response_text.txt").write_text(text or "", encoding="utf-8")
            except Exception:
                snap_path = None
        if unified_prompt_dir is not None:
            try:
                snapshot_prompt_used(
                    self.config.prompt_path,
                    Path(unified_prompt_dir),
                    content=self.prompt_text,
                    name_prefix="prompt_used.extract.",
                    logger=logger,
                )
            except Exception:
                pass

        out = self._parse_output(text)
        if snap_path is not None:
            try:
                parsed_payload = extract_output_to_dict(out)
                (snap_path / "parsed.json").write_text(
                    json.dumps(parsed_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return out


class ExtractRunnerV2:
    """V2 版本的 Extract Runner：支持结构化的 exam_points。"""

    def __init__(self, config: ExtractConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        base_text = (
            config.prompt_text
            if getattr(config, "prompt_text", None)
            else config.prompt_path.read_text(encoding="utf-8")
        )
        self.prompt_text: str = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, config.prompt_path)

        try:
            logger.info(
                "ExtractRunnerV2 configured: model=%s service_id=%s chat_args=%s",
                self.session.model_name,
                self.session.service_id,
                json.dumps(self._chat_args, ensure_ascii=False),
            )
        except Exception:
            pass

    def _build_prompt(self, ext_in: ExtractInput) -> str:
        lang = (self.config.lang or "zh").lower()
        brief_json_str = json.dumps(ext_in.paper_brief_json, ensure_ascii=False, indent=2)
        payload: Dict[str, Any] = {
            "paper_brief_json": brief_json_str,
            "paper_brief_text": ext_in.paper_brief_text,
        }
        body = _render_prompt_with_fallback(
            prompt_template=self.prompt_template,
            payload=payload,
            prompt_path=self.config.prompt_path,
            required_keys=("paper_brief_json", "paper_brief_text"),
        )
        if lang in {"en", "english"}:
            header = f"Role: Extract | Mode: QA-Init | Language: {lang}"
        else:
            header = f"【Extract 角色 · QA-Init】语言: {lang}"
        return "\n".join(
            [
                header,
                "",
                body,
            ]
        )

    def _parse_output(self, text: str) -> ExtractOutputV2:
        if not text:
            raise ValueError("ExtractV2 output is empty")
        candidate = text.strip()
        think_end = "</think>"
        if think_end in candidate:
            candidate = candidate.split(think_end, 1)[1].strip()

        fenced = extract_preferred_fenced_block(candidate, preferred_langs=("json",))
        if fenced:
            candidate = fenced

        try:
            data = json.loads(candidate)
        except Exception as e:  # noqa: BLE001
            logger.error("ExtractV2 JSON parse failed: %s", str(e))
            logger.error("Output preview: %s", (candidate or "")[:300])
            raise ValueError(f"Invalid ExtractV2 output format: {e}") from e
        if not isinstance(data, dict):
            logger.error("ExtractV2 output is not a JSON object: type=%s", type(data).__name__)
            raise ValueError("Invalid ExtractV2 output format: not a JSON object")

        def _as_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (str, int, float, bool)):
                return str(val)
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("ExtractV2: failed to JSON-encode value, using str(): %s", str(e))
                return str(val)

        # 解析结构化的 exam_points
        exam_points_raw = data.get(FIELD_EXAM_POINTS)
        if not isinstance(exam_points_raw, list):
            logger.error("ExtractV2 JSON missing/invalid exam_points list")
            raise ValueError("Invalid ExtractV2 output: missing exam_points")

        exam_points: List[ExamPointV2] = []
        for item in exam_points_raw:
            if not isinstance(item, dict):
                continue
            try:
                exam_point = ExamPointV2(
                    id=str(item.get("id", "")),
                    layer=str(item.get("layer", "")),
                    point=str(item.get("point", "")),
                    exam_style=str(item.get("exam_style", "")),
                    transform_hint=str(item.get("transform_hint", "")),
                    assessment=str(item.get("assessment", "")),
                    style=str(item.get("style", "")),
                )
                exam_points.append(exam_point)
            except Exception as e:
                logger.warning("解析 exam_point 失败: %s", str(e))
                continue

        chain_potential = _as_str(data.get(FIELD_CHAIN_POTENTIAL)).strip()

        if not (exam_points and chain_potential):
            logger.error("ExtractV2 JSON missing required fields: exam_points/chain_potential")
            raise ValueError("Invalid ExtractV2 output: missing required fields")

        return ExtractOutputV2(
            exam_points=exam_points,
            chain_potential=chain_potential,
        )

    def run_one(
        self,
        ext_in: ExtractInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
    ) -> Optional[ExtractOutputV2]:
        prompt_body = self._build_prompt(ext_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "ExtractV2")
        text = self.session.extract_text(response, default="")

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                (snap_path / "prompt.txt").write_text(prompt_body, encoding="utf-8")
                try:
                    (snap_path / "response.json").write_text(
                        json.dumps(response, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    (snap_path / "response.txt").write_text(str(response), encoding="utf-8")
                (snap_path / "response_text.txt").write_text(text or "", encoding="utf-8")
            except Exception:
                snap_path = None
        if unified_prompt_dir is not None:
            try:
                snapshot_prompt_used(
                    self.config.prompt_path,
                    Path(unified_prompt_dir),
                    content=self.prompt_text,
                    name_prefix="prompt_used.extract.",
                    logger=logger,
                )
            except Exception:
                pass

        out = self._parse_output(text)
        if snap_path is not None:
            try:
                parsed_payload = {
                    "exam_points": [
                        {
                            "id": ep.id,
                            "layer": ep.layer,
                            "point": ep.point,
                            "exam_style": ep.exam_style,
                            "transform_hint": ep.transform_hint,
                            "assessment": ep.assessment,
                            "style": ep.style,
                        }
                        for ep in out.exam_points
                    ],
                    "chain_potential": out.chain_potential,
                }
                (snap_path / "parsed.json").write_text(
                    json.dumps(parsed_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return out
