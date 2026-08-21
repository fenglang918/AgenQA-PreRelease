"""PDF parsing helpers.

Important: do NOT load heavy OCR models at import time.
This module uses lazy-loading so importing AgenQA does not immediately download/load weights.
"""

from __future__ import annotations

import logging
import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL_LST: Optional[list[Any]] = None
_MARKER_ARTIFACT_DICT: Optional[dict[str, Any]] = None
_PATCHED_TRANSFORMERS: bool = False
_SURYA_OCR_BUNDLE: Optional["SuryaOCRBundle"] = None


def _normalize_simple_pdf_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_usable_pdf_text(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    non_ws = re.sub(r"\s+", "", s)
    if len(non_ws) >= 200:
        return True
    alpha_count = sum(1 for ch in s if ch.isalpha())
    return alpha_count >= 80


def _extract_pdf_text_with_pypdf(
    pdf_path: str,
    *,
    max_pages: int | None = None,
) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing lightweight PDF text dependency: pypdf. "
            "Please install pypdf or choose text_extractor=marker."
        ) from exc

    reader = PdfReader(pdf_path)
    chunks: List[str] = []
    total_pages = len(reader.pages)
    limit = min(total_pages, max_pages) if isinstance(max_pages, int) and max_pages > 0 else total_pages
    for idx in range(limit):
        page = reader.pages[idx]
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        page_text = _normalize_simple_pdf_text(page_text)
        if page_text:
            chunks.append(page_text)
    return "\n\n".join(chunks).strip()


@dataclass
class SuryaOCRBundle:
    det_model: Any
    det_processor: Any
    rec_model: Any
    rec_processor: Any


def _patch_transformers_once() -> None:
    global _PATCHED_TRANSFORMERS
    if _PATCHED_TRANSFORMERS:
        return

    try:
        from transformers import PretrainedConfig, configuration_utils
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing PDF parsing dependencies (transformers). "
            "Please install Marker + Transformers in your environment."
        ) from exc

    # Surya/Marker have changed package layouts several times. These flags are
    # best-effort compatibility shims; absence of a specific class should not
    # block the whole PDF pipeline.
    config_candidates: list[Any] = []
    for mod_name, attr_name in (
        ("surya.model.ordering.config", "SuryaOrderConfig"),
        ("surya.model.recognition.config", "SuryaOCRConfig"),
        ("surya.model.table_rec.config", "SuryaTableRecConfig"),
        ("surya.table_rec.model.config", "SuryaTableRecConfig"),
        ("surya.common.surya.config", "SuryaModelConfig"),
        ("surya.common.surya.encoder.config", "SuryaEncoderConfig"),
        ("surya.common.surya.decoder.config", "SuryaDecoderConfig"),
    ):
        try:
            module = __import__(mod_name, fromlist=[attr_name])
            cfg_cls = getattr(module, attr_name, None)
            if cfg_cls is not None:
                config_candidates.append(cfg_cls)
        except Exception:
            continue

    for cfg_cls in config_candidates:
        try:
            cfg_cls.has_no_defaults_at_init = True
        except Exception:
            continue

    orig_get_text_config = PretrainedConfig.get_text_config

    def _get_text_config_fallback(self, decoder=None, encoder=None):
        try:
            return orig_get_text_config(self, decoder=decoder, encoder=encoder)
        except ValueError:
            if decoder is None and encoder is None:
                if hasattr(self, "decoder") and getattr(self, "decoder") is not None:
                    return getattr(self, "decoder")
                if hasattr(self, "text_encoder") and getattr(self, "text_encoder") is not None:
                    return getattr(self, "text_encoder")
            raise

    PretrainedConfig.get_text_config = _get_text_config_fallback

    orig_recursive_diff_dict = configuration_utils.recursive_diff_dict

    def _recursive_diff_dict_fallback(dict_a, dict_b, config_obj=None):
        if isinstance(config_obj, dict):
            diff = {}
            default = config_obj
            for key, value in dict_a.items():
                obj_value = config_obj.get(str(key), None)
                if (
                    isinstance(obj_value, PretrainedConfig)
                    and key in dict_b
                    and isinstance(dict_b[key], dict)
                ):
                    diff_value = _recursive_diff_dict_fallback(value, dict_b[key], config_obj=obj_value)
                    diff[key] = diff_value
                elif key not in dict_b or (key not in default) or (value != default[key]):
                    diff[key] = value
            return diff
        return orig_recursive_diff_dict(dict_a, dict_b, config_obj=config_obj)

    configuration_utils.recursive_diff_dict = _recursive_diff_dict_fallback
    _PATCHED_TRANSFORMERS = True


def _get_marker_models() -> list[Any]:
    global _MODEL_LST
    if _MODEL_LST is not None:
        return _MODEL_LST

    _patch_transformers_once()
    try:
        from marker.models import load_all_models
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing PDF parsing dependency: marker. "
            "Please install Marker in your environment."
        ) from exc

    logger.info("Loading Marker models (first run may download a few GB)...")
    _MODEL_LST = load_all_models()
    logger.info("Marker models loaded.")
    return _MODEL_LST


def _get_marker_artifacts() -> dict[str, Any]:
    global _MARKER_ARTIFACT_DICT
    if _MARKER_ARTIFACT_DICT is not None:
        return _MARKER_ARTIFACT_DICT

    _patch_transformers_once()
    try:
        from marker.models import create_model_dict
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing PDF parsing dependency: marker. "
            "Please install Marker in your environment."
        ) from exc

    logger.info("Loading Marker artifacts (first run may download a few GB)...")
    _MARKER_ARTIFACT_DICT = create_model_dict()
    logger.info("Marker artifacts loaded.")
    return _MARKER_ARTIFACT_DICT


def _get_surya_ocr_bundle() -> SuryaOCRBundle:
    global _SURYA_OCR_BUNDLE
    if _SURYA_OCR_BUNDLE is not None:
        return _SURYA_OCR_BUNDLE

    _patch_transformers_once()
    try:
        from surya.model.detection import model as det_model_mod
        from surya.model.recognition import model as rec_model_mod
        from surya.model.recognition import processor as rec_proc_mod
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Surya OCR dependencies not available") from exc

    det_model = det_model_mod.load_model()
    det_processor = det_model_mod.load_processor()
    rec_model = rec_model_mod.load_model()
    rec_processor = rec_proc_mod.load_processor()

    _SURYA_OCR_BUNDLE = SuryaOCRBundle(
        det_model=det_model,
        det_processor=det_processor,
        rec_model=rec_model,
        rec_processor=rec_processor,
    )
    return _SURYA_OCR_BUNDLE


def ocr_pil_images_with_surya(
    images: List[Any],
    *,
    langs: Optional[List[str]] = None,
    min_confidence: float = 0.3,
    batch_size: Optional[int] = None,
) -> List[str]:
    """Run Surya OCR on a list of PIL images and return extracted text per image.

    This is intended for extracting plot/figure labels that are not present in the PDF text layer.
    """
    if not images:
        return []
    try:
        from surya.ocr import run_ocr
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import surya.ocr.run_ocr") from exc

    bundle = _get_surya_ocr_bundle()
    langs_per_image: List[Optional[List[str]]] = []
    for _ in images:
        langs_per_image.append(list(langs) if langs else None)

    results = run_ocr(
        images=images,
        langs=langs_per_image,
        det_model=bundle.det_model,
        det_processor=bundle.det_processor,
        rec_model=bundle.rec_model,
        rec_processor=bundle.rec_processor,
        batch_size=batch_size,
    )

    out: List[str] = []
    for res in results or []:
        lines = getattr(res, "text_lines", None) or []
        texts: List[str] = []
        for line in lines:
            text = getattr(line, "text", "") if line is not None else ""
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            if not text:
                continue
            conf = getattr(line, "confidence", None)
            try:
                if conf is not None and float(conf) < float(min_confidence):
                    continue
            except Exception:
                pass
            texts.append(text)
        out.append("\n".join(texts).strip())
    return out


def parse_pdf(
    pdf_path: str,
    *,
    max_pages: int | None = None,
    batch_multiplier: int = 2,
) -> Tuple[str, Any, Any]:
    """Convert one PDF into (markdown_text, images, meta) via Marker."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        from marker.convert import convert_single_pdf
    except Exception:
        convert_single_pdf = None

    if callable(convert_single_pdf):
        model_lst = _get_marker_models()
        full_text, images, out_meta = convert_single_pdf(
            pdf_path,
            model_lst,
            max_pages=max_pages,
            batch_multiplier=batch_multiplier,
        )
        return str(full_text or ""), images, out_meta

    # Marker >= 1.10 moved away from `marker.convert.convert_single_pdf`.
    try:
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing PDF parsing dependency: marker. "
            "Please install Marker in your environment."
        ) from exc

    config: Dict[str, Any] = {}
    if isinstance(max_pages, int) and max_pages > 0:
        config["page_range"] = list(range(max_pages))
    # New Marker no longer exposes `batch_multiplier`; keep it as best-effort metadata only.
    if isinstance(batch_multiplier, int) and batch_multiplier > 0:
        config["batch_multiplier"] = batch_multiplier

    artifacts = _get_marker_artifacts()
    converter = PdfConverter(artifact_dict=artifacts, config=config or None)
    rendered = converter(pdf_path)
    full_text, _ext, images = text_from_rendered(rendered)
    out_meta = getattr(rendered, "metadata", {}) or {}
    return str(full_text or ""), images, out_meta


def marker_markdown_to_text(
    markdown_text: str,
    *,
    include_image_placeholders: bool = True,
    max_images_in_text: int | None = 40,
) -> Tuple[str, List[Dict[str, str]]]:
    """Convert Marker markdown into plain-ish text, optionally preserving image placeholders.

    Returns:
      (clean_text, image_refs)
    where image_refs is a list of {"alt": "...", "src": "..."} extracted from markdown.
    """
    text = str(markdown_text or "")
    image_refs: List[Dict[str, str]] = []
    img_idx = 0

    def _img_repl(match: re.Match[str]) -> str:
        nonlocal img_idx
        alt = (match.group(1) or "").strip()
        src = (match.group(2) or "").strip()
        if src:
            image_refs.append({"alt": alt, "src": src})
        img_idx += 1

        if not include_image_placeholders:
            return ""
        if isinstance(max_images_in_text, int) and img_idx > max_images_in_text:
            # Keep the text short: drop further placeholders, but the summary can be appended by caller.
            return ""
        if alt:
            return f"\n[IMAGE {img_idx}: alt={alt} src={src}]\n"
        return f"\n[IMAGE {img_idx}: src={src}]\n"

    # 1) Replace image markdown first: ![alt](src)
    text = re.sub(r"!\[(.*?)\]\((.*?)\)", _img_repl, text)

    # 2) Replace links: [text](url) -> text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

    # 3) Normalize whitespace a bit
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip(), image_refs


def _extract_caption_candidates(
    text: str,
    *,
    max_captions: int = 30,
    max_caption_chars: int = 500,
) -> List[str]:
    """Extract figure/table caption-like lines from text (best-effort)."""
    if not text:
        return []
    lines = [ln.strip() for ln in str(text).splitlines()]
    pattern = re.compile(
        r"^(?:(figure|fig\.?|table|tbl\.?)\s*([0-9]+|[ivx]+)\s*[:.\-]?\s*)(.+)$",
        re.IGNORECASE,
    )
    zh_pattern = re.compile(r"^(图|表)\s*([0-9]+)\s*[:：.\-]?\s*(.+)$")

    out: List[str] = []
    i = 0
    while i < len(lines) and len(out) < max_captions:
        ln = lines[i]
        if not ln:
            i += 1
            continue
        m = pattern.match(ln)
        mzh = zh_pattern.match(ln) if not m else None
        if not (m or mzh):
            i += 1
            continue

        if m:
            head = f"{m.group(1)} {m.group(2)}"
            rest = (m.group(3) or "").strip()
        else:
            head = f"{mzh.group(1)}{mzh.group(2)}"
            rest = (mzh.group(3) or "").strip()

        # Include a small continuation window if it looks like a wrapped caption.
        parts = [rest] if rest else []
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt:
                break
            # Stop at a new section heading / new caption.
            if pattern.match(nxt) or zh_pattern.match(nxt):
                break
            if nxt.startswith("#") or nxt.startswith("[") and nxt.endswith("]"):
                break
            if sum(len(p) for p in parts) >= max_caption_chars:
                break
            # Likely caption continuation: short-ish line.
            if len(nxt) > 120:
                break
            parts.append(nxt)
            j += 1

        caption = " ".join([p for p in parts if p]).strip()
        if caption:
            out.append(f"{head}: {caption}"[:max_caption_chars])
        else:
            out.append(f"{head}"[:max_caption_chars])
        i = j
    return out


def _extract_ocr_notes_from_marker_images(
    marker_images: Any,
    image_refs: List[Dict[str, str]],
    *,
    max_images: int = 20,
    max_chars_per_image: int = 800,
) -> List[str]:
    """Try to extract OCR/caption fields from Marker `images` output (shape varies by version)."""
    if not marker_images:
        return []

    if isinstance(marker_images, dict):
        items = marker_images.get("images") or marker_images.get("items") or []
    else:
        items = marker_images
    if not isinstance(items, list):
        return []

    notes: List[str] = []
    for idx, item in enumerate(items[:max_images], start=1):
        if not isinstance(item, dict):
            continue
        src = ""
        alt = ""
        if idx - 1 < len(image_refs):
            alt = str(image_refs[idx - 1].get("alt") or "")
            src = str(image_refs[idx - 1].get("src") or "")

        # Common-ish candidate keys (best effort).
        candidates: List[Tuple[str, Any]] = []
        for k in (
            "caption",
            "captions",
            "ocr",
            "ocr_text",
            "text",
            "extracted_text",
            "markdown",
            "md",
            "alt",
            "title",
            "label",
        ):
            if k in item and item.get(k) not in (None, "", [], {}):
                candidates.append((k, item.get(k)))
        # Search nested metadata dicts.
        for nk in ("meta", "metadata", "info"):
            nested = item.get(nk)
            if isinstance(nested, dict):
                for k, v in nested.items():
                    kl = str(k).lower()
                    if "ocr" in kl or "caption" in kl:
                        if v not in (None, "", [], {}):
                            candidates.append((f"{nk}.{k}", v))

        chunks: List[str] = []
        for k, v in candidates:
            if isinstance(v, str):
                s = v.strip()
            else:
                s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                s = s.strip()
            if not s:
                continue
            if len(s) > max_chars_per_image:
                s = s[: max_chars_per_image - 3] + "..."
            chunks.append(f"{k}: {s}")

        if not chunks:
            continue
        head = f"IMAGE {idx}"
        if alt:
            head += f" alt={alt}"
        if src:
            head += f" src={src}"
        notes.append(head + "\n" + "\n".join(chunks))

    return notes


def parse_pdf_to_text(
    pdf_path: str,
    *,
    max_pages: int | None = None,
    batch_multiplier: int = 2,
    text_extractor: str | None = None,  # "auto" | "simple" | "marker" | None
    include_image_placeholders: bool = True,
    max_images_in_text: int | None = 40,
    include_caption_block: bool = True,
    max_captions: int = 30,
    include_ocr_block: bool = True,
    max_ocr_images: int = 20,
    ocr_engine: str | None = None,  # "surya" | None
    ocr_min_confidence: float = 0.3,
    ocr_langs: Optional[List[str]] = None,
) -> str:
    """Parse PDF into cleaned plain text.

    Extraction strategy:
    - `simple`: lightweight text-layer extraction via pypdf
    - `marker`: heavy Marker-based parsing with layout/images/OCR hooks
    - `auto`/None: try `simple` first, then fall back to `marker`
    """
    extractor = (text_extractor or "auto").strip().lower()
    if extractor not in {"auto", "simple", "marker"}:
        extractor = "auto"

    if extractor in {"auto", "simple"}:
        try:
            simple_text = _extract_pdf_text_with_pypdf(pdf_path, max_pages=max_pages)
            if _looks_like_usable_pdf_text(simple_text):
                logger.info(
                    "PDF text extracted via lightweight pypdf path (extractor=%s, chars=%s): %s",
                    extractor,
                    len(simple_text),
                    pdf_path,
                )
                return simple_text
            logger.info(
                "Lightweight pypdf extraction produced insufficient text (chars=%s), falling back: %s",
                len(simple_text or ""),
                pdf_path,
            )
            if extractor == "simple":
                return simple_text.strip()
        except Exception as exc:  # noqa: BLE001
            if extractor == "simple":
                raise
            logger.info("Lightweight pypdf extraction unavailable/failed, falling back to Marker: %s", str(exc))

    full_text, images, _meta = parse_pdf(
        pdf_path,
        max_pages=max_pages,
        batch_multiplier=batch_multiplier,
    )

    clean_text, image_refs = marker_markdown_to_text(
        full_text,
        include_image_placeholders=include_image_placeholders,
        max_images_in_text=max_images_in_text,
    )
    if include_image_placeholders and image_refs and isinstance(max_images_in_text, int) and len(image_refs) > max_images_in_text:
        clean_text = (
            f"{clean_text}\n\n"
            f"[IMAGE SUMMARY: extracted {len(image_refs)} images; only first {max_images_in_text} placeholders included above]"
        )

    blocks: List[str] = []
    if include_caption_block:
        try:
            caps = _extract_caption_candidates(clean_text, max_captions=max_captions)
        except Exception:
            caps = []
        if caps:
            lines = ["[FIGURE_CAPTIONS]"]
            for i, cap in enumerate(caps, start=1):
                lines.append(f"({i}) {cap}")
            lines.append("[/FIGURE_CAPTIONS]")
            blocks.append("\n".join(lines))

    if include_ocr_block:
        try:
            ocr_notes = _extract_ocr_notes_from_marker_images(images, image_refs, max_images=max_ocr_images)
        except Exception:
            ocr_notes = []
        if ocr_notes:
            lines = ["[IMAGE_OCR_NOTES]"]
            for note in ocr_notes:
                lines.append(note)
                lines.append("")  # separator
            lines.append("[/IMAGE_OCR_NOTES]")
            blocks.append("\n".join(lines).strip())

    # Optional: run OCR on extracted images (PIL) and append text.
    engine = (ocr_engine or "").strip().lower()
    if engine in {"surya"}:
        try:
            if isinstance(images, dict):
                keys = sorted([k for k in images.keys() if isinstance(k, str)])
                selected = keys[:max_ocr_images] if isinstance(max_ocr_images, int) else keys
                pil_images = [images[k] for k in selected if k in images]
                ocr_texts = ocr_pil_images_with_surya(
                    pil_images,
                    langs=ocr_langs,
                    min_confidence=ocr_min_confidence,
                )
                payload_lines: List[str] = ["[IMAGE_OCR_TEXT]"]
                for k, t in zip(selected, ocr_texts):
                    t = (t or "").strip()
                    if not t:
                        continue
                    payload_lines.append(f"IMAGE_KEY: {k}")
                    payload_lines.append(t)
                    payload_lines.append("")
                payload_lines.append("[/IMAGE_OCR_TEXT]")
                if len(payload_lines) > 2:
                    blocks.append("\n".join(payload_lines).strip())
        except Exception:
            # OCR is best-effort and optional; never break the pipeline for it.
            pass

    if blocks:
        clean_text = f"{clean_text}\n\n" + "\n\n".join(blocks)
    return clean_text.strip()


__all__ = [
    "parse_pdf",
    "parse_pdf_to_text",
    "marker_markdown_to_text",
    "ocr_pil_images_with_surya",
]
