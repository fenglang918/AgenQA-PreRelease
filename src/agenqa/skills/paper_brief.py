"""Paper brief generator: condenses raw paper text into a short, topic-driven sketch.

目标：避免直接把整篇论文的实验叙述塞进 Known_0，而是先提炼学科、关键词与简洁摘要，
供 known-init / extend 后续出题使用。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from infra.llm.service_client import LLMServiceSession
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used, snapshot_rendered_prompt
from infra.data.io import read_text_file
from agenqa.prompts.agent_prompts import PAPER_BRIEF_PROMPT, PAPER_BRIEF_PROMPT_EN
from agenqa.skills.base import BaseSkillRunner
from agenqa.skills.pdf_native_capability import should_attach_pdf_natively
from infra.text.json_policy import clean_json_text


logger = logging.getLogger(__name__)


def _compose_paper_text(paper: Dict[str, Any]) -> str:
    """将论文结构化字段拼接为用于 brief 提取的输入。"""
    parts: list[str] = []
    title = paper.get("title")
    abstract = paper.get("abstract")
    text = paper.get("text")
    meta = paper.get("meta") if isinstance(paper.get("meta"), dict) else {}
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if text:
        parts.append(text)
    elif isinstance(meta, dict) and meta.get("source_kind") == "pdf" and meta.get("pdf_attachment"):
        parts.append("NOTE: The full paper is provided as an attached PDF document. Please read the attachment.")
    return "\n\n".join(parts) if parts else ""


@dataclass
class PaperBriefConfig:
    generator: Dict[str, Any]
    prompt_path: Path
    # 可选：直接提供模板文本，优先于 prompt_path。
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


class PaperBriefRunner:
    """Use LLM to generate a concise subject/keywords/brief JSON from paper text."""

    def __init__(self, config: PaperBriefConfig) -> None:
        self.config = config
        lang_norm = (getattr(config, "lang", None) or "").lower().strip()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            use_code_prompt = os.getenv("SCICLONE_USE_CODE_PROMPTS", "").strip() == "1"
            if use_code_prompt:
                base_text = PAPER_BRIEF_PROMPT_EN if use_en else PAPER_BRIEF_PROMPT
            else:
                try:
                    base_text = read_text_file(config.prompt_path)
                except FileNotFoundError:
                    base_text = PAPER_BRIEF_PROMPT_EN if use_en else PAPER_BRIEF_PROMPT
        self.prompt_template: str = base_text
        log_using_prompt(logger, config.prompt_path)
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)
        try:
            logger.info(
                "PaperBrief chat_args: %s | model=%s service_id=%s",
                json.dumps(self._chat_args, ensure_ascii=False),
                self.session.model_name,
                self.session.service_id,
            )
        except Exception:
            pass

    def _build_prompt(self, paper: Dict[str, Any]) -> str:
        return self.prompt_template.replace("{paper}", _compose_paper_text(paper))

    def _required_keys_for_prompt(self) -> list[str]:
        path_hint = str(getattr(self.config, "prompt_path", "") or "").lower()
        tmpl_hint = (self.prompt_template or "").lower()
        if "v3" in path_hint or "seed" in path_hint or "seed only" in tmpl_hint:
            return ["subject", "keywords"]
        if "v2" in path_hint or "brief‑v2" in tmpl_hint or "brief-v2" in tmpl_hint:
            return [
                "subject",
                "keywords",
                "background",
                "premises",
                "assumptions",
                "evidence",
                "justifications",
                "reasoning_summary",
                "conclusion",
            ]
        return ["subject", "keywords", "brief"]

    def run_one(
        self,
        paper: Dict[str, Any],
        snapshot_dir: Optional[Path] = None,
        *,
        unified_prompt_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        # Prefer PDF-native (multi-modal) input when available; fall back to text extraction if needed.
        meta = paper.get("meta") if isinstance(paper.get("meta"), dict) else {}
        attachment = meta.get("pdf_attachment") if isinstance(meta, dict) else None
        pdf_extract = meta.get("pdf_extract") if isinstance(meta.get("pdf_extract"), dict) else {}
        pdf_skip_text = bool(pdf_extract.get("skip_text", False))

        # If we cannot (or choose not to) attach the PDF, ensure paper["text"] is populated.
        if not str(paper.get("text") or "").strip() and isinstance(meta, dict) and meta.get("source_kind") == "pdf":
            model_name = str(self._chat_args.get("model") or getattr(self.session, "model_name", "") or "")
            can_attach_pdf = should_attach_pdf_natively(
                api_channel=getattr(self.session, "api_channel", ""),
                model_name=model_name,
            ) and isinstance(attachment, dict)
            if pdf_skip_text and not can_attach_pdf:
                raise ValueError(
                    "PDF-native mode requested (data.pdf_extract.skip_text=true) but PDF cannot be attached for this model/path. "
                    f"model={model_name or 'unknown'}; ensure data.pdf_extract.attach_pdf=true and use a multimodal backend (or set SCICLONE_ALLOW_PDF_ATTACHMENT_ANY_MODEL=1)."
                )
            if not can_attach_pdf:
                pdf_path = None
                if isinstance(attachment, dict) and isinstance(attachment.get("path"), str):
                    pdf_path = attachment.get("path")
                if not pdf_path and isinstance(meta.get("source_path"), str):
                    pdf_path = meta.get("source_path")
                if pdf_path:
                    try:
                        from infra.data.pdf_parser import parse_pdf_to_text
                        text = parse_pdf_to_text(
                            str(pdf_path),
                            max_pages=pdf_extract.get("max_pages", None),
                            batch_multiplier=int(pdf_extract.get("batch_multiplier", 2) or 2),
                            text_extractor=pdf_extract.get("text_extractor", None),
                            include_image_placeholders=bool(pdf_extract.get("include_image_placeholders", True)),
                            max_images_in_text=pdf_extract.get("max_images_in_text", 40),
                            include_caption_block=bool(pdf_extract.get("include_caption_block", True)),
                            max_captions=int(pdf_extract.get("max_captions", 30) or 30),
                            include_ocr_block=bool(pdf_extract.get("include_ocr_block", True)),
                            max_ocr_images=int(pdf_extract.get("max_ocr_images", 20) or 20),
                            ocr_engine=pdf_extract.get("ocr_engine", None),
                            ocr_min_confidence=float(pdf_extract.get("ocr_min_confidence", 0.3) or 0.3),
                            ocr_langs=pdf_extract.get("ocr_langs", None),
                        )
                        paper["text"] = text
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("PaperBrief: failed to extract PDF text fallback: %s", str(exc))

        prompt = self._build_prompt(paper)
        if snapshot_dir:
            try:
                snapshot_prompt_used(self.config.prompt_path, snapshot_dir, content=self.prompt_template, logger=logger)
                snapshot_rendered_prompt(prompt, snapshot_dir, logger=logger)
            except Exception:
                pass
        if unified_prompt_dir:
            try:
                snapshot_prompt_used(
                    self.config.prompt_path,
                    unified_prompt_dir,
                    content=self.prompt_template,
                    name_prefix="prompt_used.paper_brief.",
                    logger=logger,
                )
            except Exception:
                pass
        lang = (self.config.lang or "").lower() if hasattr(self.config, "lang") else ""
        messages = build_messages_with_background(prompt, lang=lang or None)

        # Optional: attach the source PDF for multi-modal backends that can read application/pdf directly.
        # This is primarily supported via Gemini/Vertex (inline_data with mime_type=application/pdf).
        if isinstance(attachment, dict) and isinstance(attachment.get("path"), str) and attachment["path"].strip():
            model_name = str(self._chat_args.get("model") or getattr(self.session, "model_name", "") or "")
            if should_attach_pdf_natively(
                api_channel=getattr(self.session, "api_channel", ""),
                model_name=model_name,
            ):
                try:
                    max_bytes = int(os.getenv("SCICLONE_PDF_ATTACHMENT_MAX_BYTES", "15728640") or "15728640")
                except Exception:
                    max_bytes = 15 * 1024 * 1024
                try:
                    pdf_path = Path(attachment["path"])
                    pdf_bytes = pdf_path.read_bytes()
                    if max_bytes > 0 and len(pdf_bytes) > max_bytes:
                        logger.warning(
                            "PaperBrief: skip PDF attachment due to size limit (size=%s max=%s): %s",
                            len(pdf_bytes),
                            max_bytes,
                            str(pdf_path),
                        )
                    else:
                        sha256 = hashlib.sha256(pdf_bytes).hexdigest()
                        b64 = base64.b64encode(pdf_bytes).decode("ascii")
                        mime_type = str(attachment.get("mime_type") or "application/pdf").strip() or "application/pdf"
                        messages[-1]["content"] = [
                            {"type": "text", "text": str(messages[-1].get("content") or "")},
                            {"type": "input_file", "mime_type": mime_type, "data": b64},
                        ]
                        logger.info(
                            "PaperBrief: attached PDF to request (mime=%s bytes=%s sha256=%s model=%s)",
                            mime_type,
                            len(pdf_bytes),
                            sha256,
                            model_name or "unknown",
                        )
                        if snapshot_dir:
                            try:
                                (snapshot_dir / "pdf_attachment_meta.json").write_text(
                                    json.dumps(
                                        {
                                            "path": str(pdf_path),
                                            "mime_type": mime_type,
                                            "bytes": len(pdf_bytes),
                                            "sha256": sha256,
                                            "model": model_name or "",
                                        },
                                        ensure_ascii=False,
                                        indent=2,
                                    ),
                                    encoding="utf-8",
                                )
                            except Exception:
                                pass
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PaperBrief: failed to attach PDF; fallback to text-only: %s", str(exc))
            else:
                logger.info(
                    "PaperBrief: pdf_attachment present but model does not look multimodal; using text-only (model=%s)",
                    model_name or "unknown",
                )

        # Debug/verification: snapshot a compact request view (without dumping huge base64 blobs).
        if snapshot_dir:
            try:
                def _compact(obj: Any) -> Any:
                    if isinstance(obj, list):
                        return [_compact(x) for x in obj]
                    if isinstance(obj, dict):
                        out: Dict[str, Any] = {}
                        for k, v in obj.items():
                            if k == "data" and isinstance(v, str) and len(v) > 200:
                                out[k] = f"<base64 len={len(v)}>"
                                continue
                            out[k] = _compact(v)
                        return out
                    return obj

                (snapshot_dir / "request_messages_compact.json").write_text(
                    json.dumps(_compact(messages), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                # If this is a Gemini model, also snapshot the converted contents payload.
                model_name = str(self._chat_args.get("model") or getattr(self.session, "model_name", "") or "")
                if "gemini" in model_name.lower():
                    try:
                        from infra.llm.service_client import _openai_messages_to_gemini_contents  # type: ignore

                        gemini_contents = _openai_messages_to_gemini_contents(messages)
                        (snapshot_dir / "request_gemini_contents_compact.json").write_text(
                            json.dumps(_compact(gemini_contents), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            resp = self.session.chat(messages, **self._chat_args)
        except Exception as e:  # noqa: BLE001
            logger.error("PaperBrief LLM call failed: %s", str(e))
            raise
        BaseSkillRunner._check_finish_reason(resp, "PaperBrief")
        text = self.session.extract_text(resp, default="") if self.session else ""

        # Always snapshot the raw response for debugging (even when extracted text is empty).
        if snapshot_dir:
            try:
                (snapshot_dir / "paper_brief_response.json").write_text(
                    json.dumps(resp, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                try:
                    (snapshot_dir / "paper_brief_response.txt").write_text(str(resp), encoding="utf-8")
                except Exception:
                    pass
            try:
                (snapshot_dir / "paper_brief_extracted_text.txt").write_text(text or "", encoding="utf-8")
            except Exception:
                pass

        candidate = (text or "").strip()
        if not candidate:
            raise ValueError("PaperBrief output is empty")
        # 将原始 LLM 文本输出快照到 snapshot_dir，便于后续排查解析问题
        if snapshot_dir:
            try:
                raw_path = snapshot_dir / "paper_brief_raw.txt"
                raw_path.write_text(candidate, encoding="utf-8")
            except Exception:
                pass

        # 优先解析 JSON；若失败则返回 None（让上层回退原文）
        def _to_dict(txt: str) -> Optional[Dict[str, Any]]:
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
            return None

        def _looks_valid_schema(obj: Dict[str, Any]) -> bool:
            required = self._required_keys_for_prompt()
            for k in required:
                if k not in obj:
                    return False
            # Minimal type checks.
            if not isinstance(obj.get("subject"), str) or not obj.get("subject", "").strip():
                return False
            if "keywords" in required and not isinstance(obj.get("keywords"), list):
                return False
            return True

        data = _to_dict(candidate)
        if not data:
            # 容错：如果文本内含代码块，尝试截取
            if "```" in candidate:
                try:
                    start = candidate.index("```") + 3
                    end = candidate.index("```", start)
                    block = candidate[start:end].strip()
                    # 去掉代码块语言标记（如 ```json）
                    lines = block.splitlines()
                    if lines and not lines[0].lstrip().startswith(("{", "[")):
                        block = "\n".join(lines[1:]).strip()
                    data = _to_dict(block)
                except Exception:
                    data = None
        if not data:
            # 最后再尝试从首个 '{' 开始截取，增强对包裹文本的兼容性
            try:
                brace_idx = candidate.index("{")
                data = _to_dict(candidate[brace_idx:].strip())
            except Exception:
                data = None
        if data and not _looks_valid_schema(data):
            data = None
        if not data:
            cleaned_text = clean_json_text(
                candidate,
                generator=self.config.generator,
                task_name="paper_brief",
                lang=lang or "zh",
                required_keys=self._required_keys_for_prompt(),
                prompt_body=prompt,
                snapshot_dir=snapshot_dir,
            )
            if cleaned_text:
                data = _to_dict(cleaned_text)
                if data and not _looks_valid_schema(data):
                    data = None
            if not data:
                logger.error("PaperBrief failed to extract valid JSON")
                logger.error("Raw output preview: %s", candidate[:300])
                raise ValueError("Invalid PaperBrief output format: no valid JSON extracted")
        return data


def render_brief_text(data: Dict[str, Any]) -> str:
    """Compose a compact text for init from brief JSON.

    支持 V1 和 V2 两种格式：
    - V1: subject, keywords, brief
    - V2: subject, background, premises, assumptions, evidence, justifications, reasoning_summary, conclusion
    """
    subject = str(data.get("subject") or "").strip()
    keywords = data.get("keywords") if isinstance(data.get("keywords"), (list, tuple)) else []
    brief = str(data.get("brief") or "").strip()
    kw_txt = ", ".join(str(k) for k in keywords if isinstance(k, str) and k.strip())
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if kw_txt:
        parts.append(f"Keywords: {kw_txt}")

    # V2 字段处理
    background = str(data.get("background") or "").strip()
    premises = data.get("premises", [])
    assumptions = data.get("assumptions", [])
    evidence = data.get("evidence", [])
    justifications = data.get("justifications", [])
    reasoning_summary = str(data.get("reasoning_summary") or "").strip()
    conclusion = str(data.get("conclusion") or "").strip()

    # 判断是否为 V2 格式（有 V2 特有字段）
    is_v2 = bool(background or premises or assumptions or evidence or justifications or reasoning_summary or conclusion)

    if is_v2:
        # V2 格式：优先使用结构化字段
        if background:
            parts.append(f"Background: {background}")
        if premises:
            premises_txt = ", ".join(str(p) for p in premises if str(p).strip())
            if premises_txt:
                parts.append(f"Premises: {premises_txt}")
        if assumptions:
            assumptions_parts = []
            for a in assumptions:
                if isinstance(a, dict):
                    a_txt = str(a.get("assumption", "")).strip()
                    if a_txt:
                        assumptions_parts.append(a_txt)
                elif isinstance(a, str) and a.strip():
                    assumptions_parts.append(a.strip())
            if assumptions_parts:
                parts.append(f"Assumptions: {'; '.join(assumptions_parts)}")
        if evidence:
            evidence_txt = ", ".join(str(e) for e in evidence if str(e).strip())
            if evidence_txt:
                parts.append(f"Evidence: {evidence_txt}")
        if justifications:
            just_parts = []
            for j in justifications:
                if isinstance(j, dict):
                    j_txt = str(j.get("because", "")).strip()
                    if j_txt:
                        just_parts.append(j_txt)
                elif isinstance(j, str) and j.strip():
                    just_parts.append(j.strip())
            if just_parts:
                parts.append(f"Justifications: {'; '.join(just_parts)}")
        if reasoning_summary:
            parts.append(f"Reasoning: {reasoning_summary}")
        if conclusion:
            parts.append(f"Conclusion: {conclusion}")
    else:
        # V1 格式：使用 brief 字段
        if brief:
            parts.append(f"Brief: {brief}")

    return "\n".join(parts).strip()
