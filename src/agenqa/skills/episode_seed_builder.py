"""EpisodeSeedBuilder: derive contract-defined episode_seed from paper text.

Input: paper text (title/abstract/body concatenated by caller) + contract (instruction + output_schema).
Output: a dict that must conform to contract.output_schema (validated by jsonschema).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema

from infra.llm.inference import resolve_inference
from infra.data.io import read_text_file
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used, snapshot_rendered_prompt
from infra.llm.service_client import LLMServiceSession
from infra.text.json_policy import clean_json_text
from agenqa.skills.base import BaseSkillRunner
from agenqa.skills.pdf_native_capability import should_attach_pdf_natively

from agenqa.prompts.episode_seed_builder import EPISODE_SEED_BUILDER_PROMPT, EPISODE_SEED_BUILDER_PROMPT_EN

logger = logging.getLogger(__name__)


@dataclass
class EpisodeSeedBuilderConfig:
    generator: Dict[str, Any]
    prompt_path: Path
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


def _to_dict(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_seed_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data or {})
    keywords = out.get("keywords")
    if isinstance(keywords, list):
        normalized: list[str] = []
        seen: set[str] = set()
        for item in keywords:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(s)
        if len(normalized) > 10:
            normalized = normalized[:10]
        out["keywords"] = normalized
    return out


def _compose_episode_seed_input_text(paper_text: str, paper: Dict[str, Any] | None) -> str:
    if isinstance(paper_text, str) and paper_text.strip():
        return paper_text
    if not isinstance(paper, dict):
        return ""
    parts: list[str] = []
    title = paper.get("title")
    abstract = paper.get("abstract")
    text = paper.get("text") or paper.get("content")
    meta = paper.get("meta") if isinstance(paper.get("meta"), dict) else {}
    if isinstance(title, str) and title.strip():
        parts.append(f"Title: {title.strip()}")
    if isinstance(abstract, str) and abstract.strip():
        parts.append(f"Abstract: {abstract.strip()}")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    elif isinstance(meta, dict) and meta.get("source_kind") == "pdf" and meta.get("pdf_attachment"):
        parts.append("NOTE: The full paper is provided as an attached PDF document. Please read the attachment.")
    return "\n\n".join(parts).strip()


class EpisodeSeedBuilderRunner:
    def __init__(self, config: EpisodeSeedBuilderConfig) -> None:
        self.config = config
        lang_norm = (getattr(config, "lang", None) or "").lower().strip()
        use_en = lang_norm in {"en", "english"}

        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            use_code_prompt = os.getenv("SCICLONE_USE_CODE_PROMPTS", "").strip() == "1"
            if use_code_prompt:
                base_text = EPISODE_SEED_BUILDER_PROMPT_EN if use_en else EPISODE_SEED_BUILDER_PROMPT
            else:
                try:
                    base_text = read_text_file(config.prompt_path)
                except FileNotFoundError:
                    base_text = EPISODE_SEED_BUILDER_PROMPT_EN if use_en else EPISODE_SEED_BUILDER_PROMPT

        self.prompt_template: str = base_text
        log_using_prompt(logger, config.prompt_path)
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

    def _build_prompt(self, paper_text: str, *, contract: Dict[str, Any]) -> str:
        instruction = str(contract.get("instruction") or "").strip()
        output_schema = contract.get("output_schema") if isinstance(contract.get("output_schema"), dict) else {}
        schema_json = json.dumps(output_schema, ensure_ascii=False, indent=2)
        return (
            self.prompt_template.replace("{paper_text}", paper_text)
            .replace("{contract_instruction}", instruction)
            .replace("{output_schema_json}", schema_json)
        )

    def run_one(
        self,
        paper_text: str,
        *,
        paper: Optional[Dict[str, Any]] = None,
        contract: Dict[str, Any],
        strict_validation: bool = True,
        snapshot_dir: Optional[Path] = None,
        unified_prompt_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        meta = paper.get("meta") if isinstance(paper, dict) and isinstance(paper.get("meta"), dict) else {}
        attachment = meta.get("pdf_attachment") if isinstance(meta, dict) else None
        pdf_extract = meta.get("pdf_extract") if isinstance(meta.get("pdf_extract"), dict) else {}
        pdf_skip_text = bool(pdf_extract.get("skip_text", False))

        input_text = _compose_episode_seed_input_text(paper_text, paper)

        if not input_text and isinstance(meta, dict) and meta.get("source_kind") == "pdf":
            model_name = str(self._chat_args.get("model") or getattr(self.session, "model_name", "") or "")
            can_attach_pdf = should_attach_pdf_natively(
                api_channel=getattr(self.session, "api_channel", ""),
                model_name=model_name,
            ) and isinstance(attachment, dict)
            if pdf_skip_text and not can_attach_pdf:
                raise ValueError(
                    "PDF-native mode requested (init.source.pdf_extract.skip_text=true) but EpisodeSeedBuilder "
                    f"cannot attach PDF for model={model_name or 'unknown'}. "
                    "Ensure attach_pdf=true and use a Gemini/Vertex-capable model (or set "
                    "SCICLONE_ALLOW_PDF_ATTACHMENT_ANY_MODEL=1)."
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
                        input_text = _compose_episode_seed_input_text(text, paper)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("EpisodeSeedBuilder: failed to extract PDF text fallback: %s", str(exc))

        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("EpisodeSeedBuilder requires a non-empty paper text input or a readable PDF attachment")

        prompt = self._build_prompt(input_text, contract=contract)
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
                    name_prefix="prompt_used.episode_seed_builder.",
                    logger=logger,
                )
            except Exception:
                pass

        lang = (self.config.lang or "").lower() if hasattr(self.config, "lang") else ""
        messages = build_messages_with_background(prompt, lang=lang or None)

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
                            "EpisodeSeedBuilder: skip PDF attachment due to size limit (size=%s max=%s): %s",
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
                            "EpisodeSeedBuilder: attached PDF to request (mime=%s bytes=%s sha256=%s model=%s)",
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
                    logger.warning("EpisodeSeedBuilder: failed to attach PDF; fallback to text-only: %s", str(exc))
            else:
                logger.info(
                    "EpisodeSeedBuilder: pdf_attachment present but model does not look multimodal; using text-only (model=%s)",
                    model_name or "unknown",
                )

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
        except Exception as exc:  # noqa: BLE001
            logger.error("EpisodeSeedBuilder LLM call failed: %s", str(exc))
            raise

        BaseSkillRunner._check_finish_reason(resp, "EpisodeSeedBuilder")
        text = self.session.extract_text(resp, default="") if self.session else ""

        if snapshot_dir:
            try:
                (snapshot_dir / "episode_seed_builder_response.json").write_text(
                    json.dumps(resp, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                try:
                    (snapshot_dir / "episode_seed_builder_response.txt").write_text(str(resp), encoding="utf-8")
                except Exception:
                    pass
            try:
                (snapshot_dir / "episode_seed_builder_extracted_text.txt").write_text(text or "", encoding="utf-8")
            except Exception:
                pass

        candidate = (text or "").strip()
        if snapshot_dir:
            try:
                (snapshot_dir / "episode_seed_builder_raw.txt").write_text(candidate or "", encoding="utf-8")
            except Exception:
                pass

        output_schema = contract.get("output_schema") if isinstance(contract, dict) else None
        output_schema = output_schema if isinstance(output_schema, dict) else {}
        required_keys: list[str] = []
        if isinstance(output_schema.get("required"), list):
            required_keys = [str(x).strip() for x in output_schema.get("required") if str(x).strip()]

        cleaned_text = clean_json_text(
            candidate,
            generator=self.config.generator,
            task_name="episode_seed_builder",
            lang=lang or "zh",
            required_keys=tuple(required_keys),
            prompt_body=prompt,
            snapshot_dir=snapshot_dir,
        )
        if not cleaned_text:
            logger.error("EpisodeSeedBuilder failed to extract valid JSON")
            logger.error("Raw output preview: %s", (candidate or "")[:300])
            raise ValueError("Invalid EpisodeSeedBuilder output format: no valid JSON extracted")

        data = _normalize_seed_payload(_to_dict(cleaned_text) or {})
        if not isinstance(data, dict) or not data:
            raise ValueError("EpisodeSeedBuilder output must be a non-empty JSON object")

        if output_schema:
            try:
                jsonschema.validate(data, output_schema)
            except jsonschema.ValidationError as exc:
                if strict_validation:
                    raise ValueError(f"EpisodeSeedBuilder output schema mismatch: {exc.message}") from exc
                logger.warning("EpisodeSeedBuilder output schema mismatch (strict=false): %s", exc.message)
        return data


__all__ = ["EpisodeSeedBuilderConfig", "EpisodeSeedBuilderRunner"]
