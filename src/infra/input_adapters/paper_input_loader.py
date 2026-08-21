from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from infra.data.ids import generate_paper_id
from infra.input_adapters.scipedia_pack import build_scipedia_pack


def _read_first_jsonl_record_with_lineno(path: Path) -> Tuple[int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"expected JSON object at line {lineno}, got {type(obj).__name__}")
            return lineno, obj
    raise RuntimeError(f"no JSON object found in {path}")


def _as_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    return default


def load_one_paper_like_record(agent_conf: Dict[str, Any]) -> Dict[str, Any]:
    """Load exactly one paper-like record for agent-run KnownInit.

    Input forms:
    - .txt (or no suffix): use file content as `text`
    - .pdf: parse PDF into `text` (Marker-based, may be slow on first run)
    - .json: use the JSON object
    - .jsonl: use the first JSON object

    Optional: if `init.source.scipedia_pack.enable=true`, the record's text will be replaced by a
    multi-section packed text, while keeping paper_id stable.
    """
    init_conf = agent_conf.get("init") if isinstance(agent_conf, dict) else None
    if not isinstance(init_conf, dict):
        raise ValueError("Missing top-level `init` config block (required for paper-like input).")
    init_source = init_conf.get("source")
    if not isinstance(init_source, dict):
        raise ValueError("Missing init.source (required for paper-like input).")

    path_raw = init_source.get("path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        raise ValueError("Missing init.source.path (required for paper-like input).")
    papers_path = Path(path_raw.strip())

    suffix = papers_path.suffix.lower()
    if suffix == ".txt" or suffix == "":
        try:
            content = papers_path.read_text(encoding="utf-8")
        except Exception:
            content = papers_path.read_text(errors="ignore")
        paper_id = generate_paper_id({"text": content})
        return {"paper_id": paper_id, "text": content}
    if suffix == ".pdf":
        pdf_conf = init_source.get("pdf_extract") if isinstance(init_source.get("pdf_extract"), dict) else {}
        attach_pdf = bool(pdf_conf.get("attach_pdf", False))
        skip_text = bool(pdf_conf.get("skip_text", False))
        try:
            max_pages = pdf_conf.get("max_pages", None)
            max_pages = int(max_pages) if max_pages is not None else None
        except Exception:
            max_pages = None
        try:
            batch_multiplier = int(pdf_conf.get("batch_multiplier", 2) or 2)
        except Exception:
            batch_multiplier = 2
        text_extractor = pdf_conf.get("text_extractor", None)
        if isinstance(text_extractor, str):
            text_extractor = text_extractor.strip().lower() or None
        else:
            text_extractor = None
        include_image_placeholders = bool(pdf_conf.get("include_image_placeholders", True))
        try:
            max_images_in_text_raw = pdf_conf.get("max_images_in_text", 40)
            max_images_in_text = int(max_images_in_text_raw) if max_images_in_text_raw is not None else None
        except Exception:
            max_images_in_text = 40
        include_caption_block = bool(pdf_conf.get("include_caption_block", True))
        try:
            max_captions = int(pdf_conf.get("max_captions", 30) or 30)
        except Exception:
            max_captions = 30
        include_ocr_block = bool(pdf_conf.get("include_ocr_block", True))
        try:
            max_ocr_images = int(pdf_conf.get("max_ocr_images", 20) or 20)
        except Exception:
            max_ocr_images = 20
        ocr_engine = pdf_conf.get("ocr_engine", None)
        if isinstance(ocr_engine, str):
            ocr_engine = ocr_engine.strip().lower() or None
        try:
            ocr_min_confidence = float(pdf_conf.get("ocr_min_confidence", 0.3) or 0.3)
        except Exception:
            ocr_min_confidence = 0.3
        ocr_langs = pdf_conf.get("ocr_langs", None)
        if isinstance(ocr_langs, str) and ocr_langs.strip():
            ocr_langs = [x.strip() for x in ocr_langs.split(",") if x.strip()]
        if not isinstance(ocr_langs, list):
            ocr_langs = None
        else:
            ocr_langs = [str(x).strip() for x in ocr_langs if str(x).strip()] or None

        text = ""
        if not skip_text:
            # Lazy import: avoid loading heavy OCR models unless we actually parse PDF -> text.
            from infra.data.pdf_parser import parse_pdf_to_text

            text = parse_pdf_to_text(
                str(papers_path),
                max_pages=max_pages,
                batch_multiplier=batch_multiplier,
                text_extractor=text_extractor,
                include_image_placeholders=include_image_placeholders,
                max_images_in_text=max_images_in_text,
                include_caption_block=include_caption_block,
                max_captions=max_captions,
                include_ocr_block=include_ocr_block,
                max_ocr_images=max_ocr_images,
                ocr_engine=ocr_engine,
                ocr_min_confidence=ocr_min_confidence,
                ocr_langs=ocr_langs,
            )
            if not text.strip():
                raise RuntimeError(f"PDF parse returned empty text: {papers_path}")

        paper_id_seed = {"text": text} if text.strip() else {"source_path": str(papers_path)}
        paper_id = generate_paper_id(paper_id_seed)
        return {
            "paper_id": paper_id,
            "text": text,
            "meta": {
                "source_path": str(papers_path),
                "source_kind": "pdf",
                "pdf_extract": {
                    "max_pages": max_pages,
                    "batch_multiplier": batch_multiplier,
                    "text_extractor": text_extractor,
                    "attach_pdf": attach_pdf,
                    "skip_text": skip_text,
                    "include_image_placeholders": include_image_placeholders,
                    "max_images_in_text": max_images_in_text,
                    "include_caption_block": include_caption_block,
                    "max_captions": max_captions,
                    "include_ocr_block": include_ocr_block,
                    "max_ocr_images": max_ocr_images,
                    "ocr_engine": ocr_engine,
                    "ocr_min_confidence": ocr_min_confidence,
                    "ocr_langs": ocr_langs,
                },
                "pdf_attachment": {"path": str(papers_path), "mime_type": "application/pdf"} if attach_pdf else None,
            },
        }

    lineno: int
    record: Dict[str, Any]
    if suffix == ".json":
        raw = papers_path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise TypeError(f"expected JSON object in {papers_path}, got {type(obj).__name__}")
        lineno, record = 1, obj
    else:
        lineno, record = _read_first_jsonl_record_with_lineno(papers_path)

    if "text" not in record and isinstance(record.get("content"), str):
        record["text"] = record.get("content")

    if "text" not in record:
        pages = record.get("pages")
        if isinstance(pages, list):
            page_texts: list[str] = []
            for page in pages:
                if not isinstance(page, dict):
                    continue
                text = page.get("text")
                if isinstance(text, str):
                    text = text.strip()
                    if text:
                        page_texts.append(text)
            if page_texts:
                record["text"] = "\n\n".join(page_texts)

    original_text = str(record.get("text") or "")
    if not original_text:
        raise RuntimeError(f"input record missing text field: {papers_path} line {lineno}")

    if not record.get("paper_id"):
        record["paper_id"] = generate_paper_id({"text": original_text})

    sp = init_source.get("scipedia_pack") if isinstance(init_source.get("scipedia_pack"), dict) else {}
    if _as_bool(sp.get("enable"), default=False):
        include_sections = sp.get("include_sections") or ["Key Takeaways", "Principles and Mechanisms", "Introduction"]
        if not isinstance(include_sections, list):
            include_sections = [str(include_sections)]
        include_sections = [str(s).strip() for s in include_sections if str(s).strip()]

        strip_wiki_tokens = _as_bool(sp.get("strip_wiki_tokens"), default=True)
        normalize_whitespace = _as_bool(sp.get("normalize_whitespace"), default=True)
        prepend_title = _as_bool(sp.get("prepend_title"), default=True)

        title = str(record.get("title") or "").strip()
        pack_text, meta = build_scipedia_pack(
            title=title or "SciPedia Entry",
            text=original_text,
            include_sections=include_sections,
            strip_wiki_tokens=strip_wiki_tokens,
            normalize_whitespace=normalize_whitespace,
            prepend_title=prepend_title,
        )
        record["text"] = pack_text

        meta_obj = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        meta_obj["scipedia_pack"] = {
            **meta,
            "source_path": str(papers_path),
            "source_lineno": lineno,
        }
        record["meta"] = meta_obj

    return record
